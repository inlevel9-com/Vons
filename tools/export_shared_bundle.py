"""Split the existing pilot graphs into one encoder and two unchanged heads.

This is a graph partition, not retraining. Historical bundles are read-only.
The selected graph cuts are specific to the checked Vons pilot exports.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

MODEL_LIMIT_BYTES = 64 * 1024 * 1024
ENCODER_INPUTS = ["input_ids", "attention_mask", "token_type_ids", "option_mask"]
EMBEDDINGS = "view_19"
POOLED = "div"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _onnx() -> Any:
    import onnx

    return onnx


def _rename(model: Any, names: dict[str, str]) -> None:
    for node in model.graph.node:
        for fields in (node.input, node.output):
            for index, name in enumerate(fields):
                fields[index] = names.get(name, name)
    for values in (model.graph.input, model.graph.output, model.graph.value_info,
                   model.graph.initializer):
        for value in values:
            value.name = names.get(value.name, value.name)


def _validate_external_paths(path: Path) -> None:
    """Check external tensor locations before loading any external bytes."""
    onnx = _onnx()
    model = onnx.load(path, load_external_data=False)
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {item.key: item.value for item in tensor.external_data}
        location = metadata.get("location", "")
        relative = Path(location)
        if not location or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe external tensor location: {location}")
        target = path.parent / relative
        if target.is_symlink() or not target.resolve(strict=True).is_relative_to(path.parent.resolve()):
            raise ValueError(f"external tensor escapes graph directory: {location}")


def _tensor_hash(tensor: Any) -> str:
    onnx = _onnx()
    array = onnx.numpy_helper.to_array(tensor)
    header = json.dumps([int(tensor.data_type), list(tensor.dims)]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def encoder_parameters(model: Any) -> Counter[str]:
    """Include named encoder tensors and exporter-folded parameter matrices.

    Exporter-folded matmul weights lose their original names. The pilot's
    large floating point initializers are parameter matrices; positional/index
    constants use integral dtypes. Small arithmetic constants are not weights.
    """
    onnx = _onnx()
    return Counter(
        _tensor_hash(tensor)
        for tensor in model.graph.initializer
        if tensor.data_type == onnx.TensorProto.FLOAT
        and (tensor.name.startswith("encoder.") or math.prod(tensor.dims) >= 4096)
    )


def partition_graphs(direct: Any, diffusion: Any) -> dict[str, Any]:
    onnx = _onnx()
    from onnx.utils import Extractor

    encoder = Extractor(direct).extract_model(ENCODER_INPUTS, [EMBEDDINGS, POOLED])
    other_encoder = Extractor(diffusion).extract_model(ENCODER_INPUTS, [EMBEDDINGS, POOLED])
    if not encoder_parameters(encoder) or encoder_parameters(encoder) != encoder_parameters(other_encoder):
        raise ValueError("encoder parameter tensors differ; sharing would change the model")
    direct_head = Extractor(direct).extract_model(
        [EMBEDDINGS, POOLED, "option_mask"], ["logits", "answerability"],
    )
    diffusion_head = Extractor(diffusion).extract_model(
        [POOLED, "option_mask", "initial_noise"], ["scores", "answerability"],
    )
    # The original fixed graph obtains its batch size from token_type_ids.
    # The pooled representation has precisely the same leading batch axis.
    for node in diffusion_head.graph.node:
        if "token_type_ids" in node.input:
            attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
            if node.op_type != "Shape" or attributes.get("start", 0) != 0 or attributes.get("end") != 1:
                raise ValueError("unsupported token dependency in diffusion head")
            for index, name in enumerate(node.input):
                if name == "token_type_ids":
                    node.input[index] = POOLED
    result = {"encoder": encoder, "direct": direct_head, "diffusion": diffusion_head}
    for graph in result.values():
        _rename(graph, {EMBEDDINGS: "candidate_embeddings", POOLED: "pooled"})
        onnx.checker.check_model(graph)
    return result


def export_bundle(direct_path: Path, diffusion_path: Path, tokenizer: Path, output: Path,
                  calibration: Path | None = None) -> dict[str, Any]:
    onnx = _onnx()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; historical bundles are never overwritten")
    for path in (direct_path, diffusion_path):
        _validate_external_paths(path)
    direct = onnx.load(direct_path)
    diffusion = onnx.load(diffusion_path)
    parts = partition_graphs(direct, diffusion)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, graph in parts.items():
        path = output / f"{name}.onnx"
        onnx.save_model(copy.deepcopy(graph), path, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=f"{name}.onnx.data",
                        size_threshold=1024, convert_attribute=False)
        onnx.checker.check_model(str(path))
        records.append({"path": path.name, "role": f"{name}_graph"})
        external = path.with_suffix(".onnx.data")
        if external.exists():
            records.append({"path": external.name, "role": f"{name}_weights"})
    token_dir = output / "tokenizer"
    token_dir.mkdir()
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        source = tokenizer / filename
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"required tokenizer asset missing or symlinked: {filename}")
        shutil.copyfile(source, token_dir / filename)
        records.append({"path": f"tokenizer/{filename}", "role": "tokenizer"})
    calibration_data: dict[str, Any] = {"direct": None, "diffusion": None}
    calibration_source = None
    if calibration is not None:
        raw = json.loads(calibration.read_text())
        fit = raw.get("calibration", raw.get("config", {}).get("calibration"))
        if not isinstance(fit, dict):
            raise ValueError("direct calibration input has no fitted calibration parameters")
        calibration_data["direct"] = fit
        calibration_source = {"sha256": digest(calibration), "scope": "direct only"}
    (output / "calibration.json").write_text(json.dumps(calibration_data, indent=2) + "\n")
    records.append({"path": "calibration.json", "role": "calibration"})
    settings = {
        "max_tokens": 512, "max_options": 32, "max_questions": 8,
        "diffusion_inference_steps": 4, "diffusion_training_steps": 32,
        "diffusion_candidate_slots": 32, "initial_noise": "host-supplied standard normal float32",
        "default_abstain_threshold": 0.55, "default_answerability_threshold": 0.5,
        "calibration": "calibration.json", "calibration_source": calibration_source,
    }
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    records.append({"path": "config.json", "role": "config"})
    for record in records:
        path = output / record["path"]
        record.update(bytes=path.stat().st_size, sha256=digest(path))
    total = sum(record["bytes"] for record in records)
    source_assets = []
    for label, source in (("direct", direct_path), ("diffusion", diffusion_path)):
        source_names = {source.name}
        source_model = onnx.load(source, load_external_data=False)
        for tensor in source_model.graph.initializer:
            source_names.update(item.value for item in tensor.external_data if item.key == "location")
        source_assets.extend({"backend": label, "path": filename,
                              "bytes": (source.parent / filename).stat().st_size,
                              "sha256": digest(source.parent / filename)}
                             for filename in sorted(source_names))
    manifest = {
        "schema": "vons.shared-bundle/v1", "model_id": "vons-pilot-shared-v1",
        "transformation": "partition existing graphs; no training or parameter changes",
        "encoder": {"graph": "encoder.onnx", "outputs": ["candidate_embeddings", "pooled"]},
        "heads": {
            "direct": {"graph": "direct.onnx", "inputs": ["candidate_embeddings", "pooled", "option_mask"]},
            "diffusion": {"graph": "diffusion.onnx", "inputs": ["pooled", "option_mask", "initial_noise"]},
        },
        "files": records, "model_asset_bytes": total, "limit_bytes": MODEL_LIMIT_BYTES,
        "size_pass": total <= MODEL_LIMIT_BYTES,
        "source_graphs": {"direct_sha256": digest(direct_path), "diffusion_sha256": digest(diffusion_path)},
        "source_assets": source_assets,
        "source_tool_sha256": digest(Path(__file__)),
        "release_status": "research_candidate; parity and notices required before release",
        "runtime_bytes": None, "runtime_bytes_reason": "runtime is not included in this model-only bundle",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "manifest.sha256").write_text(f"{digest(manifest_path)}  manifest.json\n")
    verify_manifest(manifest_path)
    return manifest


def verify_manifest(path: Path, *, expected_digest: str | None = None) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "vons.shared-bundle/v1":
        raise ValueError("unsupported shared bundle manifest")
    expected = expected_digest or path.with_suffix(".sha256").read_text().split()[0]
    if digest(path) != expected:
        raise ValueError("release manifest digest mismatch")
    root = path.parent.resolve()
    names = set()
    total = 0
    for item in manifest["files"]:
        relative = Path(item["path"])
        target = root / relative
        if relative.is_absolute() or ".." in relative.parts or target.is_symlink():
            raise ValueError("unsafe manifest asset path")
        if not target.resolve(strict=True).is_relative_to(root):
            raise ValueError("manifest asset escapes bundle")
        if relative.as_posix() in names:
            raise ValueError("duplicate manifest asset")
        names.add(relative.as_posix())
        if target.stat().st_size != item["bytes"] or digest(target) != item["sha256"]:
            raise ValueError(f"asset digest or byte count mismatch: {relative}")
        total += target.stat().st_size
    if total != manifest["model_asset_bytes"] or manifest["size_pass"] != (total <= MODEL_LIMIT_BYTES):
        raise ValueError("bundle byte total mismatch")
    required = {"encoder.onnx", "direct.onnx", "diffusion.onnx", "config.json",
                "calibration.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if not required.issubset(names):
        raise ValueError("manifest omits required model assets")
    onnx = _onnx()
    for filename in ("encoder.onnx", "direct.onnx", "diffusion.onnx"):
        _validate_external_paths(root / filename)
        graph = onnx.load(root / filename, load_external_data=False)
        for tensor in graph.graph.initializer:
            for item in tensor.external_data:
                if item.key == "location" and item.value not in names:
                    raise ValueError("manifest omits referenced external tensor data")
    return {"pass": True, "model_asset_bytes": total, "size_pass": total <= MODEL_LIMIT_BYTES,
            "manifest_sha256": digest(path)}


def verify_parity(bundle: Path, direct_path: Path, diffusion_path: Path,
                  *, cases: int = 20) -> dict[str, Any]:
    """Compare graph partitions with historical graphs using identical tensors.

    This verifies graph transformation only. It is not task-quality evaluation
    or a latency benchmark, and does not validate browser tokenization.
    """
    import numpy as np
    import onnxruntime as ort

    verified = verify_manifest(bundle / "manifest.json")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    def session(path: Path) -> Any:
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    encoder = session(bundle / "encoder.onnx")
    heads = {name: session(bundle / f"{name}.onnx") for name in ("direct", "diffusion")}
    full = {"direct": session(direct_path), "diffusion": session(diffusion_path)}
    samples = []
    for case in range(cases):
        count = (2, 4, 8, 16, 32)[case % 5]
        batch = 2 if case % 4 == 3 else 1
        length = (8, 31, 64, 127)[case % 4]
        # Token IDs are valid BERT vocabulary entries. Fixed local arrays avoid
        # downloads and isolate graph partitioning from tokenizer differences.
        random = np.random.default_rng(700 + case)
        tokens = random.integers(100, 2000, (batch, count, length), dtype=np.int64)
        mask = np.zeros((batch, 32), dtype=np.bool_)
        mask[:, :count] = True
        ids = np.zeros((batch, 32, 512), np.int64)
        attention = np.zeros_like(ids)
        ids[:, :count, :length] = tokens
        attention[:, :count, :length] = 1
        feed = {"input_ids": ids, "attention_mask": attention,
                "token_type_ids": np.zeros_like(ids), "option_mask": mask}
        representation = encoder.run(None, feed)
        for backend in ("direct", "diffusion"):
            for seed in ((7, 17, 27) if backend == "diffusion" else (7,)):
                noise = np.random.default_rng(seed + case).standard_normal((batch, 32)).astype(np.float32)
                shared_feed = {"pooled": representation[1], "option_mask": mask}
                original_feed = dict(feed)
                if backend == "direct":
                    shared_feed["candidate_embeddings"] = representation[0]
                else:
                    shared_feed["initial_noise"] = noise
                    original_feed["initial_noise"] = noise
                expected = full[backend].run(None, original_feed)
                actual = heads[backend].run(None, shared_feed)
                finite = np.broadcast_to(mask, expected[0].shape)
                scores_finite = bool(np.isfinite(expected[0][finite]).all()
                                     and np.isfinite(actual[0][finite]).all())
                masked = bool(np.isneginf(expected[0][~finite]).all()
                              and np.isneginf(actual[0][~finite]).all())
                answers_finite = bool(np.isfinite(expected[1]).all() and np.isfinite(actual[1]).all())
                passed = bool(scores_finite and masked and answers_finite
                              and np.allclose(expected[0][finite], actual[0][finite], atol=1e-5, rtol=1e-5)
                              and np.allclose(expected[1], actual[1], atol=1e-5, rtol=1e-5))
                samples.append({
                    "case": case, "backend": backend, "seed": seed, "candidates": count,
                    "batch": batch, "shape": list(ids.shape), "active_tokens": length,
                    "input_sha256": hashlib.sha256(ids.tobytes() + attention.tobytes() + mask.tobytes()).hexdigest(),
                    "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest() if backend == "diffusion" else None,
                    "max_abs_scores": float(np.max(np.abs(expected[0][finite] - actual[0][finite]))),
                    "max_abs_answerability": float(np.max(np.abs(expected[1] - actual[1]))),
                    "pass": passed,
                })
    return {"schema": "vons.shared-bundle-parity/v1", "manifest_sha256": verified["manifest_sha256"],
            "provider": "CPUExecutionProvider", "ort_version": ort.__version__,
            "intra_op_threads": 2, "atol": 1e-5, "rtol": 1e-5, "cases": cases,
            "scope": "graph partition only; no quality, browser, calibration or latency claim",
            "samples": samples, "pass": all(sample["pass"] for sample in samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-calibration", type=Path)
    parser.add_argument("--parity-report", type=Path)
    args = parser.parse_args()
    report = export_bundle(args.direct, args.diffusion, args.tokenizer, args.output,
                           args.direct_calibration)
    print(json.dumps({key: report[key] for key in ("model_asset_bytes", "size_pass", "release_status")}))
    if args.parity_report is not None:
        parity = verify_parity(args.output, args.direct, args.diffusion)
        args.parity_report.parent.mkdir(parents=True, exist_ok=True)
        args.parity_report.write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"parity_pass": parity["pass"], "samples": len(parity["samples"])}))
        if not parity["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
