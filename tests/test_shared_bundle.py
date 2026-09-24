"""Offline graph-partition and release-integrity tests using small toy graphs."""

import importlib.util
import json
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
spec = importlib.util.spec_from_file_location(
    "vons_shared_export", Path(__file__).parents[1] / "tools/export_shared_bundle.py",
)
assert spec is not None and spec.loader is not None
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
digest = exporter.digest
export_bundle = exporter.export_bundle
partition_graphs = exporter.partition_graphs
verify_manifest = exporter.verify_manifest


def toy_model(backend: str, *, weight: float = 1.0):
    helper = onnx.helper
    tensor = onnx.TensorProto
    inputs = [helper.make_tensor_value_info(name, tensor.INT64, ["batch", "options", 2])
              for name in ("input_ids", "attention_mask", "token_type_ids")]
    inputs.append(helper.make_tensor_value_info("option_mask", tensor.BOOL, ["batch", "options"]))
    values = [helper.make_tensor_value_info("view_19", tensor.FLOAT, ["batch", "options", 2]),
              helper.make_tensor_value_info("div", tensor.FLOAT, ["batch", 2])]
    weights = [onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32) * weight,
                                           name="encoder.weight"),
               onnx.numpy_helper.from_array(np.array([1], np.int64), name="axis1"),
               onnx.numpy_helper.from_array(np.array([2], np.int64), name="axis2")]
    nodes = [helper.make_node("Cast", ["input_ids"], ["cast"], to=tensor.FLOAT),
             helper.make_node("MatMul", ["cast", "encoder.weight"], ["view_19"]),
             helper.make_node("ReduceMean", ["view_19", "axis1"], ["div"], keepdims=0),
             helper.make_node("ReduceSum", ["div", "axis1"], ["answerability"], keepdims=0)]
    output = "logits" if backend == "direct" else "scores"
    if backend == "direct":
        nodes.append(helper.make_node("ReduceSum", ["view_19", "axis2"], [output], keepdims=0))
    else:
        inputs.append(helper.make_tensor_value_info("initial_noise", tensor.FLOAT, ["batch", "options"]))
        nodes.extend([
            helper.make_node("Shape", ["token_type_ids"], ["batch_shape"], start=0, end=1),
            helper.make_node("Expand", ["answerability", "batch_shape"], ["expanded"]),
            helper.make_node("Unsqueeze", ["expanded", "axis1"], ["scalar"]),
            helper.make_node("Add", ["initial_noise", "scalar"], [output]),
        ])
    outputs = [helper.make_tensor_value_info(output, tensor.FLOAT, ["batch", "options"]),
               helper.make_tensor_value_info("answerability", tensor.FLOAT, ["batch"])]
    model = helper.make_model(helper.make_graph(nodes, "toy", inputs, outputs, weights,
                                               value_info=values),
                              opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def test_partition_keeps_outputs_and_removes_token_dependency():
    parts = partition_graphs(toy_model("direct"), toy_model("diffusion"))
    for model in parts.values():
        onnx.checker.check_model(model)
    assert [x.name for x in parts["encoder"].graph.output] == ["candidate_embeddings", "pooled"]
    assert all("token_type_ids" not in n.input for n in parts["diffusion"].graph.node)


def test_different_encoders_cannot_be_silently_shared():
    with pytest.raises(ValueError, match="parameter tensors differ"):
        partition_graphs(toy_model("direct"), toy_model("diffusion", weight=2))


def test_partition_matches_full_graph_with_nonzero_noise():
    ort = pytest.importorskip("onnxruntime")
    original = {name: toy_model(name) for name in ("direct", "diffusion")}
    parts = partition_graphs(original["direct"], original["diffusion"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {key: ort.InferenceSession(value.SerializeToString(), options,
                                         providers=["CPUExecutionProvider"])
                for key, value in parts.items()}
    feed = {name: np.array([[[1, 2], [3, 4], [5, 6]]], np.int64)
            for name in ("input_ids", "attention_mask", "token_type_ids")}
    feed["option_mask"] = np.array([[True, True, True]])
    encoded = sessions["encoder"].run(None, feed)
    for backend, model in original.items():
        shared = {"candidate_embeddings": encoded[0], "pooled": encoded[1],
                  "option_mask": feed["option_mask"]}
        full_feed = dict(feed)
        if backend == "diffusion":
            shared.pop("candidate_embeddings")
            shared["initial_noise"] = np.array([[0.2, -0.7, 1.1]], np.float32)
            full_feed["initial_noise"] = shared["initial_noise"]
        full = ort.InferenceSession(model.SerializeToString(), options,
                                    providers=["CPUExecutionProvider"]).run(None, full_feed)
        split = sessions[backend].run(None, shared)
        for expected, actual in zip(full, split, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-6)


def make_bundle(tmp_path: Path):
    for backend in ("direct", "diffusion"):
        onnx.save(toy_model(backend), tmp_path / f"{backend}.onnx")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    target = tmp_path / "bundle"
    manifest = export_bundle(tmp_path / "direct.onnx", tmp_path / "diffusion.onnx", tokenizer, target)
    return target, manifest


def test_asset_count_includes_config_calibration_and_both_heads(tmp_path):
    target, manifest = make_bundle(tmp_path)
    assert {r["role"] for r in manifest["files"]} >= {
        "encoder_graph", "direct_graph", "diffusion_graph", "calibration", "config", "tokenizer",
    }
    assert manifest["model_asset_bytes"] == sum((target / r["path"]).stat().st_size
                                               for r in manifest["files"])
    assert verify_manifest(target / "manifest.json")["pass"]
    with pytest.raises(ValueError, match="new or empty"):
        export_bundle(tmp_path / "direct.onnx", tmp_path / "diffusion.onnx",
                      tmp_path / "tokenizer", target)


def test_tamper_and_external_expected_digest_are_rejected(tmp_path):
    target, _ = make_bundle(tmp_path)
    path = target / "manifest.json"
    with pytest.raises(ValueError, match="release manifest digest"):
        verify_manifest(path, expected_digest="0" * 64)
    (target / "config.json").write_text("tampered")
    with pytest.raises(ValueError, match="asset digest"):
        verify_manifest(path)


def test_referenced_assets_cannot_be_omitted(tmp_path):
    target, manifest = make_bundle(tmp_path)
    manifest["files"] = [r for r in manifest["files"] if r["path"] != "calibration.json"]
    manifest["model_asset_bytes"] = sum(r["bytes"] for r in manifest["files"])
    path = target / "manifest.json"
    path.write_text(json.dumps(manifest))
    path.with_suffix(".sha256").write_text(digest(path))
    with pytest.raises(ValueError, match="omits required"):
        verify_manifest(path)
