"""Independently check browser traces against the pinned Python CPU ONNX graph.

This consumes historical full-graph bundles, not the shared deployment format.
No output is overwritten. Score-head rejection is an unsupported capability,
never a successful numerical comparison. The current browser path does not
apply calibration; matching it must not be described as calibrated parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from vons.bundle import verify_bundle_manifest

HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
SCHEMA = "vons.browser-parity/v1"


class ParityError(ValueError):
    """A mismatch or missing evidence prevents a successful comparison."""


class InputOverflow(ParityError):
    """The independently tokenized question exceeds the declared budget."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    """Match the harness's JSON.stringify/localeCompare for its fixture domain.

    Python's Unicode sorting is not JS localeCompare. Restrict object keys to
    lowercase ASCII letters (the complete current fixture domain), and numeric
    values to safe integers. Unsupported collation/number formatting fails
    closed instead of silently attesting to a differently canonicalized hash.
    Unicode *values* are preserved, including the current multilingual states.
    """
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not re.fullmatch(r"[a-z]+", key) for key in value):
            raise ParityError("canonical request keys require lowercase ASCII letters")
        return "{" + ",".join(
            json.dumps(key) + ":" + stable_json(value[key]) for key in sorted(value)
        ) + "}"
    if isinstance(value, list):
        return "[" + ",".join(stable_json(item) for item in value) + "]"
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or int(value) != value or abs(value) > 2**53 - 1:
            raise ParityError("canonical request numbers must be finite safe integers")
        return str(int(value))
    raise ParityError("request is not JSON-compatible")


def request_hash(request: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(request).encode("utf-8")).hexdigest()


def _required_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ParityError(f"{label} requires a lowercase SHA-256 digest")
    return value


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ParityError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs,
                      parse_constant=reject_constant)


def _relative_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ParityError("manifest file path must be a relative POSIX path")
    value = PurePosixPath(relative)
    if value.is_absolute() or PureWindowsPath(relative).drive or ".." in value.parts:
        raise ParityError(f"manifest path escapes the bundle: {relative}")
    current = root
    for part in value.parts:
        current = current / part
        if current.is_symlink():
            raise ParityError(f"manifest path contains a symlink: {relative}")
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve()):
        raise ParityError(f"manifest file is missing or outside bundle: {relative}")
    return current


def validate_artifacts(report: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ParityError("browser report must be an object")
    if report.get("schema") != SCHEMA:
        raise ParityError("unsupported browser report schema")
    if report.get("requested_provider") not in {"wasm", "webgpu"}:
        raise ParityError("requested_provider must identify wasm or webgpu")
    expected = _required_hash(report.get("manifest_hash"), "manifest_hash")
    if file_hash(manifest_path) != expected:
        raise ParityError("browser manifest_hash does not match the supplied manifest bytes")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "vons.bundle.manifest/v1":
        raise ParityError("only historical full-graph bundle manifests are supported")
    backend = report.get("backend")
    if backend not in {"direct", "diffusion"} or manifest.get("metadata", {}).get("backend") != backend:
        raise ParityError("report backend does not match manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ParityError("manifest file inventory is empty")
    for item in files:
        if not isinstance(item, dict):
            raise ParityError("manifest file entries must be objects")
        path = _relative_file(manifest_path.parent, item.get("path"))
        if type(item.get("bytes")) is not int or item["bytes"] < 0:
            raise ParityError("manifest bytes must be a nonnegative integer")
        if path.stat().st_size != item["bytes"] or file_hash(path) != _required_hash(
            item.get("sha256"), f"file {item['path']}"
        ):
            raise ParityError(f"manifest file hash/bytes mismatch: {item['path']}")
    # Also inspect actual ONNX external-data references and bundle byte totals.
    verification = verify_bundle_manifest(manifest_path)
    tokenizers = [item for item in files if item.get("role") == "tokenizer"
                  and item["path"].endswith("/tokenizer.json")]
    tokenizers += [item for item in files if item.get("role") == "tokenizer"
                   and item["path"] == "tokenizer.json"]
    graphs = [item for item in files if item.get("role") == "model_graph"]
    if len(tokenizers) != 1 or len(graphs) != 1:
        raise ParityError("manifest must select exactly one tokenizer.json and model graph")
    if tokenizers[0]["sha256"] != _required_hash(report.get("tokenizer_hash"), "tokenizer_hash"):
        raise ParityError("browser tokenizer_hash does not match the pinned tokenizer")
    graph = _relative_file(manifest_path.parent, graphs[0]["path"])
    if graph.parent != manifest_path.parent:
        raise ParityError("historical graph must be beside its manifest for external-data resolution")
    return {"manifest": manifest, "verification": verification, "graph": graph,
            "tokenizer": _relative_file(manifest_path.parent, tokenizers[0]["path"])}


def validate_request(request: Any) -> list[dict[str, Any]]:
    if not isinstance(request, dict):
        raise ParityError("request must be an object")
    state = request.get("state")
    if not isinstance(state, (str, dict)) or isinstance(state, str) and not state.strip():
        raise ParityError("state must be a nonempty string or an object")
    questions = request.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 8:
        raise ParityError("questions must contain 1..8 items")
    ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ParityError("questions must contain objects")
        key = question.get("id")
        if not isinstance(key, str) or not key or key in ids:
            raise ParityError("question IDs must be unique nonempty strings")
        ids.add(key)
        kind, options = question.get("type"), question.get("options")
        if kind not in {"choice", "boolean", "score"}:
            raise ParityError("unsupported question type")
        if not isinstance(question.get("prompt"), str) or not question["prompt"].strip():
            raise ParityError("question prompt must be a nonempty string")
        if not isinstance(options, list) or any(not isinstance(x, str) or not x.strip()
                                                for x in options):
            raise ParityError("options must contain nonempty strings")
        if len(set(options)) != len(options):
            raise ParityError("options must be unique")
        if kind == "boolean":
            if options not in ([], ["true", "false"]):
                raise ParityError("boolean options must be true/false")
        elif not 2 <= len(options) <= 32:
            raise ParityError("candidate count must be 2..32")
        rubric = question.get("rubric", [])
        if not isinstance(rubric, list):
            raise ParityError("rubric must be a list")
        if kind == "score":
            if not 2 <= len(rubric) <= 10 or any(not isinstance(x, str) for x in rubric):
                raise ParityError("score requires 2..10 string rubric levels")
        elif rubric:
            raise ParityError("only score may provide rubric levels")
    return questions


def candidate_options(question: dict[str, Any]) -> list[str]:
    return ["true", "false"] if question["type"] == "boolean" and not question["options"] \
        else question["options"]


def build_inputs(tokenizer: Any, state: Any, question: dict[str, Any],
                 metadata: dict[str, Any], backend: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    options = candidate_options(question)
    sequence = metadata.get("sequence_length", 512)
    limit = metadata.get("option_count", 32)
    if type(sequence) is not int or not 1 <= sequence <= 512:
        raise ParityError("manifest sequence_length must be an integer in 1..512")
    if type(limit) is not int or not len(options) <= limit <= 32:
        raise ParityError("manifest option_count cannot accommodate the question")
    slots = limit if backend == "diffusion" else len(options)
    state_text = state if isinstance(state, str) else stable_json(state)
    prefix = f"{state_text}\nQuestion: {question['prompt']}\n"
    aggregate = tokenizer.encode(prefix + "Candidates:\n" + "\n".join(options))
    if len(aggregate.ids) > sequence:
        raise InputOverflow(f"aggregate tokens {len(aggregate.ids)} exceed {sequence}")
    encoded = [tokenizer.encode(prefix + "Candidate: " + option) for option in options]
    if any(len(item.ids) > sequence for item in encoded):
        raise InputOverflow(f"candidate tokens exceed {sequence}")
    padding = tokenizer.token_to_id("[PAD]")
    feed = {"input_ids": np.full((1, slots, sequence), padding or 0, dtype=np.int64),
            "attention_mask": np.zeros((1, slots, sequence), dtype=np.int64),
            "token_type_ids": np.zeros((1, slots, sequence), dtype=np.int64),
            "option_mask": np.zeros((1, slots), dtype=np.bool_)}
    for index, item in enumerate(encoded):
        size = len(item.ids)
        if len(item.attention_mask) != size or len(item.type_ids) != size:
            raise ParityError("Python tokenizer returned inconsistent array lengths")
        feed["input_ids"][0, index, :size] = item.ids
        feed["attention_mask"][0, index, :size] = item.attention_mask
        feed["token_type_ids"][0, index, :size] = item.type_ids
        feed["option_mask"][0, index] = True
    shape = {"live_candidates": len(options), "allocated_candidates": slots,
             "live_tokens": sum(len(item.ids) for item in encoded), "sequence_length": sequence}
    return feed, shape


def finite_array(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ParityError(f"{label} is not numeric") from exc
    if not array.size or not np.isfinite(array).all():
        raise ParityError(f"{label} is empty or non-finite")
    return array


def assert_close(actual: Any, expected: Any, label: str, atol: float, rtol: float) -> float:
    left, right = finite_array(actual, label), finite_array(expected, label)
    if left.shape != right.shape or not np.allclose(left, right, atol=atol, rtol=rtol,
                                                   equal_nan=False):
        raise ParityError(f"{label} differs from Python reference (atol={atol}, rtol={rtol})")
    return float(np.max(np.abs(left - right)))


def postprocess(scores: Any, answerability_logit: float, options: list[str],
                question_id: str) -> dict[str, Any]:
    values = finite_array(scores, "reference scores")
    if values.shape != (len(options),):
        raise ParityError("reference score count differs from candidates")
    logit = float(finite_array([answerability_logit], "reference answerability")[0])
    exp = np.exp(values - np.max(values))
    probabilities = exp / exp.sum()
    answerability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index] * answerability)
    reason = "answerability_below_threshold" if answerability < 0.5 else (
        "confidence_below_threshold" if confidence < 0.55 else None
    )
    return {"question_id": question_id, "choice": None if reason else options[index],
            "probabilities": [{"option": option, "probability": float(probabilities[i])}
                              for i, option in enumerate(options)],
            "confidence": confidence, "status": "abstain" if reason else "ok",
            "abstain_reason": reason}


def compare_answer(actual: Any, expected: dict[str, Any], atol: float, rtol: float) -> None:
    if not isinstance(actual, dict):
        raise ParityError("answer is missing")
    for field in ("question_id", "choice", "status", "abstain_reason"):
        if field not in actual or actual[field] != expected[field]:
            raise ParityError(f"answer {field} differs from Python reference")
    assert_close([actual.get("confidence")], [expected["confidence"]], "confidence", atol, rtol)
    probabilities = actual.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(expected["probabilities"]):
        raise ParityError("answer probability count differs")
    for left, right in zip(probabilities, expected["probabilities"], strict=True):
        if not isinstance(left, dict) or left.get("option") != right["option"]:
            raise ParityError("answer probability options differ")
    assert_close([item.get("probability") for item in probabilities],
                 [item["probability"] for item in expected["probabilities"]],
                 "answer probabilities", atol, rtol)


class CpuReference:
    def __init__(self, graph: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
        self.version = ort.__version__

    def run(self, feed: dict[str, np.ndarray], backend: str) -> tuple[np.ndarray, np.ndarray]:
        scores, answerability = self.session.run(
            ["scores" if backend == "diffusion" else "logits", "answerability"], feed
        )
        return scores, answerability


def verify_trace(trace: dict[str, Any], timing: dict[str, Any], response_answer: Any,
                 request: dict[str, Any], question: dict[str, Any], tokenizer: Any,
                 metadata: dict[str, Any], backend: str, noise_values: Any,
                 reference: Any, atol: float, rtol: float) -> dict[str, Any]:
    if trace.get("questionId") != question["id"] or timing.get("questionId") != question["id"]:
        raise ParityError("trace/timing question order or ID differs")
    feed, shape = build_inputs(tokenizer, request["state"], question, metadata, backend)
    for field, expected in shape.items():
        if type(timing.get(field)) is not int or timing[field] != expected:
            raise ParityError(f"timing shape {field} differs from independent tokenization")
    for browser_key, python_key in (("inputIds", "input_ids"), ("attentionMask", "attention_mask"),
                                    ("tokenTypeIds", "token_type_ids"), ("optionMask", "option_mask")):
        actual = trace.get(browser_key)
        if not isinstance(actual, list) or any(type(item) is not int for item in actual):
            raise ParityError(f"{browser_key} must be an integer array")
        if actual != feed[python_key].astype(np.int64).ravel().tolist():
            raise ParityError(f"{browser_key} differs from independent tokenizer/padding")
    noise_hash = None
    if backend == "diffusion":
        expected_shape = (shape["allocated_candidates"],)
        noise = finite_array(trace.get("initialNoise"), "initialNoise")
        frozen = finite_array(noise_values, "report noise_values")
        if noise.shape != expected_shape or frozen.shape != expected_shape:
            raise ParityError("initial noise dimensions differ from candidate slots")
        # Both sources must represent the same exact stored Float32 values.
        if not np.array_equal(noise, noise.astype(np.float32).astype(np.float64)):
            raise ParityError("trace initialNoise is not an exact Float32 array")
        if not np.array_equal(noise, frozen):
            raise ParityError("trace noise differs from report's frozen noise_values")
        noise32 = noise.astype("<f4")
        noise_hash = hashlib.sha256(noise32.tobytes()).hexdigest()
        feed["initial_noise"] = noise32.reshape(1, -1)
    elif trace.get("initialNoise") is not None or noise_values is not None:
        raise ParityError("Direct must not carry diffusion noise")
    raw_scores, raw_answerability = reference.run(feed, backend)
    all_scores = finite_array(raw_scores, "CPU raw scores")
    raw_answerability = finite_array(raw_answerability, "CPU answerability")
    if all_scores.shape != (1, shape["allocated_candidates"]) or raw_answerability.size != 1:
        raise ParityError("CPU output tensor shape differs from bundle contract")
    scores = all_scores[0, :shape["live_candidates"]]
    logit = float(raw_answerability.ravel()[0])
    score_delta = assert_close(trace.get("rawScores"), scores, "raw scores", atol, rtol)
    logit_delta = assert_close([trace.get("answerabilityLogit")], [logit], "answerability", atol, rtol)
    answer = postprocess(scores, logit, candidate_options(question), question["id"])
    probabilities = [item["probability"] for item in answer["probabilities"]]
    probability_delta = assert_close(trace.get("probabilities"), probabilities,
                                     "trace probabilities", atol, rtol)
    compare_answer(trace.get("answer"), answer, atol, rtol)
    compare_answer(response_answer, answer, atol, rtol)
    return {"question_id": question["id"], "status": "passed", "shape": shape,
            "noise_float32_le_sha256": noise_hash,
            "raw_scores": scores.tolist(), "answerability_logit": logit,
            "answer": answer, "max_absolute_difference": {
                "raw_scores": score_delta, "answerability_logit": logit_delta,
                "probabilities": probability_delta}, "calibration": None}


def verify_records(report: dict[str, Any], manifest: dict[str, Any], tokenizer: Any,
                   reference: Any, atol: float = 1e-5, rtol: float = 1e-5) -> dict[str, Any]:
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ParityError("tolerances must be finite and nonnegative")
    records = report.get("records")
    if not isinstance(records, list) or not records or report.get("fixture_count") != len(records):
        raise ParityError("report requires nonempty records and matching fixture_count")
    backend, metadata = report["backend"], manifest["metadata"]
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ParityError("fixture records must be objects")
        result: dict[str, Any] = {"fixture_id": row.get("fixture_id"), "questions": []}
        results.append(result)
        try:
            key = row.get("fixture_id")
            if not isinstance(key, str) or not key or key in seen:
                raise ParityError("fixture IDs must be unique nonempty strings")
            seen.add(key)
            request = row.get("request")
            fingerprint = _required_hash(row.get("input_hash"), "input_hash")
            if request_hash(request) != fingerprint:
                raise ParityError("input_hash differs from canonical request")
            result["input_hash"] = fingerprint
            questions = validate_request(request)
            status = row.get("status")
            traces, timings = row.get("traces"), row.get("timings")
            if not isinstance(traces, list) or not isinstance(timings, list):
                raise ParityError("trace and timing arrays are required, including for errors")
            if status == "expected_error":
                if traces or timings or not isinstance(row.get("error"), str) or not row["error"]:
                    raise ParityError("expected errors require an error and empty trace/timing arrays")
                if any(question["type"] == "score" for question in questions):
                    if "score questions are not supported" not in row["error"]:
                        raise ParityError("score rejection does not match the recorded SDK limitation")
                    result.update(status="unverified", reason="score_head_unsupported",
                                  numerical_comparison=False, contract_error_verified=False)
                    continue
                overflow = None
                for question in questions:
                    try:
                        build_inputs(tokenizer, request["state"], question, metadata, backend)
                    except InputOverflow as exc:
                        overflow = str(exc)
                        break
                if overflow is None or not any(text in row["error"].lower()
                                                for text in ("token", "budget", "overflow")):
                    raise ParityError("expected overflow could not be independently confirmed")
                result.update(status="contract_error_verified", reason=overflow,
                              numerical_comparison=False, contract_error_verified=True)
                continue
            if status != "ok":
                raise ParityError(f"browser request did not succeed: {row.get('error', status)}")
            if any(question["type"] == "score" for question in questions):
                raise ParityError("current full-graph reference does not implement score questions")
            response = row.get("response")
            if not isinstance(response, dict) or response.get("backend") != backend:
                raise ParityError("response backend is missing or mismatched")
            expected_model_id = metadata.get("model_id", f"vons-{backend}-onnx-web")
            if response.get("model_id") != expected_model_id:
                raise ParityError("response model_id differs from the supplied manifest")
            answers = response.get("answers")
            if not isinstance(answers, list) or not len(traces) == len(timings) == len(answers) == len(questions):
                raise ParityError("every question requires exactly one trace, timing and answer")
            for question, trace, timing, answer in zip(questions, traces, timings, answers, strict=True):
                item = {"question_id": question["id"], "status": "failed"}
                result["questions"].append(item)
                try:
                    item.update(verify_trace(trace, timing, answer, request, question, tokenizer,
                                             metadata, backend, report.get("noise_values"), reference,
                                             atol, rtol))
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    item["reason"] = str(exc)
            failed = any(item["status"] != "passed" for item in result["questions"])
            result.update(status="failed" if failed else "passed", numerical_comparison=True,
                          contract_error_verified=False)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result.update(status="failed", reason=str(exc), numerical_comparison=False,
                          contract_error_verified=False)
    counts = Counter(item["status"] for item in results)
    numeric = Counter(item["status"] for row in results for item in row["questions"])
    return {"records": results,
            "counts": {key: counts[key] for key in
                       ("passed", "failed", "unverified", "contract_error_verified")},
            "numerical_questions": {key: numeric[key] for key in ("passed", "failed")},
            "all_numerical_comparisons_passed": numeric["passed"] > 0 and not numeric["failed"]
            and not counts["failed"],
            "complete_contract_verified": not counts["failed"] and not counts["unverified"],
            "calibration": None,
            "postprocess": {"temperature": 1.0, "abstain_threshold": 0.55,
                            "answerability_threshold": 0.5},
            "limitations": [
                "The browser path uses uncalibrated probabilities and default thresholds.",
                "Unsupported score questions do not count as numerical or contract passes.",
                "Provider execution partition, lifecycle, latency and offline operation are not verified.",
                "Canonical request hashes support lowercase ASCII object keys and safe integer numbers; Unicode string values are supported.",
                "This reference checks the supplied full graph, not shared-bundle SDK execution or task accuracy.",
            ]}


def verify_report(report_path: Path, manifest_path: Path, output_path: Path,
                  atol: float = 1e-5, rtol: float = 1e-5) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    from tokenizers import Tokenizer

    report = read_json(report_path)
    artifacts = validate_artifacts(report, manifest_path)
    tokenizer = Tokenizer.from_file(str(artifacts["tokenizer"]))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    reference = CpuReference(artifacts["graph"])
    results = verify_records(report, artifacts["manifest"], tokenizer, reference, atol, rtol)
    output = {"schema": "vons.browser-parity-reference/v1", "backend": report["backend"],
              "requested_provider": report.get("requested_provider"),
              "provider_execution_verified": False,
              "manifest_sha256": file_hash(manifest_path),
              "report_sha256": file_hash(report_path), "tokenizer_sha256": report["tokenizer_hash"],
              "verifier_sha256": file_hash(Path(__file__)), "manifest_verification": artifacts["verification"],
              "reference_runtime": {"provider": "CPUExecutionProvider", "ort_version": reference.version,
                                    "intra_op_threads": 1, "inter_op_threads": 1},
              "tolerances": {"atol": atol, "rtol": rtol, "equal_nan": False}, **results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        result = verify_report(args.report, args.manifest, args.output, args.atol, args.rtol)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"parity verification rejected: {exc}\n")
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "numerical_questions": result["numerical_questions"]}, allow_nan=False))
    return 0 if result["all_numerical_comparisons_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
