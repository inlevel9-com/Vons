"""Verifier regressions with tiny tokenizers and mocked CPU outputs only."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "browser_parity_reference", Path(__file__).parents[1] / "tools/verify_browser_parity.py"
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class TinyTokenizer:
    def encode(self, text):
        ids = [101, *text.encode("utf-8"), 102]
        return SimpleNamespace(ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids))

    def token_to_id(self, token):
        assert token == "[PAD]"
        return 0


class Reference:
    def __init__(self):
        self.feeds = []
        self.answerability = 2.0
        self.invalid = False

    def run(self, feed, backend):
        self.feeds.append(copy.deepcopy(feed))
        scores = np.zeros(feed["option_mask"].shape, np.float32)
        scores[0, 0] = float("nan") if self.invalid else 3.0
        return scores, np.array([self.answerability], np.float32)


def question(key="q", kind="choice"):
    return {"id": key, "type": kind, "prompt": "Choose", "options": ["first", "second"]}


def expected_answer(q, logit=2.0):
    options = q["options"] or ["true", "false"]
    probability = 1 / (1 + math.exp(-3))
    answerability = 1 / (1 + math.exp(-logit))
    confidence = probability * answerability
    reason = "answerability_below_threshold" if answerability < 0.5 else (
        "confidence_below_threshold" if confidence < 0.55 else None
    )
    return {"question_id": q["id"], "choice": None if reason else options[0],
            "probabilities": [{"option": options[0], "probability": probability},
                              {"option": options[1], "probability": 1 - probability}],
            "confidence": confidence, "status": "abstain" if reason else "ok",
            "abstain_reason": reason}


def fixture(backend="direct", multi=False):
    questions = [question()]
    if multi:
        questions.append({"id": "q2", "type": "boolean", "prompt": "Continue?", "options": []})
    request = {"state": "ready", "questions": questions, "seed": 7}
    metadata = {"backend": backend, "sequence_length": 128, "option_count": 4}
    manifest = {"metadata": metadata}
    noise = [0.25, -0.5, 1.0, 2.0] if backend == "diffusion" else None
    traces, timings, answers = [], [], []
    for q in questions:
        options = q["options"] or ["true", "false"]
        slots = 4 if backend == "diffusion" else 2
        ids, masks, types, live = [], [], [], 0
        for index in range(slots):
            raw = [101, *f"ready\nQuestion: {q['prompt']}\nCandidate: {options[index]}".encode(), 102] \
                if index < 2 else []
            live += len(raw)
            ids.extend(raw + [0] * (128 - len(raw)))
            masks.extend([1] * len(raw) + [0] * (128 - len(raw)))
            types.extend([0] * 128)
        answer = expected_answer(q)
        traces.append({"questionId": q["id"], "inputIds": ids, "attentionMask": masks,
                       "tokenTypeIds": types, "optionMask": [1, 1] + [0] * (slots - 2),
                       "initialNoise": noise, "rawScores": [3.0, 0.0], "answerabilityLogit": 2.0,
                       "probabilities": [p["probability"] for p in answer["probabilities"]],
                       "answer": copy.deepcopy(answer)})
        timings.append({"questionId": q["id"], "live_candidates": 2, "allocated_candidates": slots,
                        "live_tokens": live, "sequence_length": 128})
        answers.append(answer)
    row = {"fixture_id": "fixture", "input_hash": verifier.request_hash(request),
           "request": request, "status": "ok", "traces": traces, "timings": timings,
           "response": {"backend": backend, "model_id": f"vons-{backend}-onnx-web",
                        "answers": answers}}
    report = {"schema": verifier.SCHEMA, "backend": backend, "fixture_count": 1,
              "noise_values": noise, "records": [row]}
    return report, manifest


def run(report, manifest, reference=None):
    return verifier.verify_records(report, manifest, TinyTokenizer(), reference or Reference())


def test_canonical_hash_matches_browser_fixture_domain_and_unicode_values():
    value = {"state": {"z": "🌐 검토", "a": 2}, "seed": 7, "questions": []}
    canonical = '{"questions":[],"seed":7,"state":{"a":2,"z":"🌐 검토"}}'
    assert verifier.stable_json(value) == canonical
    assert verifier.request_hash(value) == hashlib.sha256(canonical.encode()).hexdigest()
    assert verifier.stable_json({"seed": -0.0}) == '{"seed":0}'


@pytest.mark.parametrize("value", [{"Mixed": 1}, {"a_b": 1}, {"é": 1}, {"a": 1.25},
                                   {"a": 2**53}, {"a": float("nan")}])
def test_unsupported_json_canonicalization_fails_closed(value):
    with pytest.raises(verifier.ParityError):
        verifier.stable_json(value)


@pytest.mark.parametrize("backend", ["direct", "diffusion"])
def test_every_multiquestion_trace_and_response_compares(backend):
    report, manifest = fixture(backend, multi=True)
    reference = Reference()
    result = run(report, manifest, reference)
    assert result["counts"] == {"passed": 1, "failed": 0, "unverified": 0,
                               "contract_error_verified": 0}
    assert result["numerical_questions"] == {"passed": 2, "failed": 0}
    assert result["all_numerical_comparisons_passed"]
    assert len(reference.feeds) == 2
    assert result["calibration"] is None
    if backend == "diffusion":
        expected = hashlib.sha256(np.array(report["noise_values"], dtype="<f4").tobytes()).hexdigest()
        assert all(q["noise_float32_le_sha256"] == expected for q in result["records"][0]["questions"])
        np.testing.assert_array_equal(reference.feeds[0]["initial_noise"], [[0.25, -0.5, 1, 2]])


@pytest.mark.parametrize("field", ["inputIds", "attentionMask", "tokenTypeIds", "optionMask"])
def test_exact_token_arrays_are_required_before_inference(field):
    report, manifest = fixture()
    report["records"][0]["traces"][0][field][0] += 1
    reference = Reference()
    result = run(report, manifest, reference)
    assert result["counts"]["failed"] == 1
    assert not reference.feeds
    assert field in result["records"][0]["questions"][0]["reason"]


@pytest.mark.parametrize("mutation", ["missing_trace", "missing_hash", "bad_hash", "missing_question",
                                     "wrong_shape", "response_choice", "response_probability"])
def test_missing_or_tampered_evidence_cannot_pass(mutation):
    report, manifest = fixture(multi=True)
    row = report["records"][0]
    if mutation == "missing_trace":
        row["traces"] = []
    elif mutation == "missing_hash":
        del row["input_hash"]
    elif mutation == "bad_hash":
        row["input_hash"] = "0" * 64
    elif mutation == "missing_question":
        row["response"]["answers"].pop()
    elif mutation == "wrong_shape":
        row["timings"][0]["sequence_length"] = 512
    elif mutation == "response_choice":
        row["response"]["answers"][1]["choice"] = "false"
    else:
        row["response"]["answers"][1]["probabilities"][0]["probability"] = 0.1
    result = run(report, manifest)
    assert result["counts"]["failed"] == 1
    assert not result["all_numerical_comparisons_passed"]


@pytest.mark.parametrize("field", ["rawScores", "probabilities", "answerabilityLogit"])
def test_nonfinite_browser_values_never_pass(field):
    report, manifest = fixture()
    trace = report["records"][0]["traces"][0]
    trace[field] = float("nan") if field == "answerabilityLogit" else [float("nan"), 0.0]
    result = run(report, manifest)
    assert result["counts"]["failed"] == 1


def test_nonfinite_cpu_values_never_pass_even_with_large_tolerance():
    report, manifest = fixture()
    reference = Reference()
    reference.invalid = True
    result = verifier.verify_records(report, manifest, TinyTokenizer(), reference, atol=100, rtol=100)
    assert result["counts"]["failed"] == 1
    assert "non-finite" in result["records"][0]["questions"][0]["reason"]


def test_noise_uses_array_identity_not_seed_identity():
    report, manifest = fixture("diffusion")
    report["records"][0]["traces"][0]["initialNoise"] = [0.5, -0.5, 1.0, 2.0]
    reference = Reference()
    result = run(report, manifest, reference)
    assert result["counts"]["failed"] == 1
    assert not reference.feeds


def test_non_float32_noise_fails_before_inference():
    report, manifest = fixture("diffusion")
    report["noise_values"][0] = 0.1
    result = run(report, manifest)
    assert "exact Float32" in result["records"][0]["questions"][0]["reason"]


def test_postprocess_both_abstention_gates_and_tie_order():
    answer = verifier.postprocess([4.0, 0.0], -1.0, ["first", "second"], "q")
    assert answer["abstain_reason"] == "answerability_below_threshold" and answer["choice"] is None
    answer = verifier.postprocess([0.0, 0.0], 20.0, ["first", "second"], "q")
    assert answer["abstain_reason"] == "confidence_below_threshold" and answer["choice"] is None
    answer = verifier.postprocess([3.0, 0.0], 2.0, ["first", "second"], "q")
    assert answer["choice"] == "first" and answer["status"] == "ok"
    with pytest.raises(verifier.ParityError):
        verifier.postprocess([-math.inf, -math.inf], 2, ["first", "second"], "q")


def test_score_unsupported_is_unverified_not_contract_or_numerical_pass():
    report, manifest = fixture()
    row = report["records"][0]
    row["request"]["questions"][0].update(type="score", rubric=["poor", "good"])
    row["input_hash"] = verifier.request_hash(row["request"])
    row.update(status="expected_error", traces=[], timings=[],
               error="Error: score questions are not supported by this candidate-selection ONNX head")
    result = run(report, manifest)
    assert result["counts"]["unverified"] == 1
    assert result["numerical_questions"] == {"passed": 0, "failed": 0}
    assert not result["all_numerical_comparisons_passed"]
    assert not result["complete_contract_verified"]


def test_expected_overflow_requires_independent_exact_tokenization():
    report, manifest = fixture()
    row = report["records"][0]
    row.update(status="expected_error", traces=[], timings=[], error="RangeError: token overflow")
    assert run(report, manifest)["counts"]["failed"] == 1
    row["request"]["state"] = "x" * 200
    row["input_hash"] = verifier.request_hash(row["request"])
    result = run(report, manifest)
    assert result["counts"]["contract_error_verified"] == 1
    assert result["numerical_questions"]["passed"] == 0
    assert not result["all_numerical_comparisons_passed"]


def test_aggregate_budget_can_overflow_when_individual_candidates_fit():
    q = question()
    q["options"] = ["x" * 60, "y" * 60]
    with pytest.raises(verifier.InputOverflow, match="aggregate"):
        verifier.build_inputs(TinyTokenizer(), "ready", q,
                              {"sequence_length": 128, "option_count": 32}, "direct")


def test_empty_duplicate_and_inconsistent_fixture_sets_fail():
    report, manifest = fixture()
    duplicate = copy.deepcopy(report["records"][0])
    report["records"].append(duplicate)
    report["fixture_count"] = 2
    assert run(report, manifest)["counts"]["failed"] == 1
    report["fixture_count"] = 3
    with pytest.raises(verifier.ParityError, match="fixture_count"):
        run(report, manifest)
    report.update(records=[], fixture_count=0)
    with pytest.raises(verifier.ParityError, match="nonempty"):
        run(report, manifest)


def make_artifacts(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    graph, tokenizer = root / "model.onnx", root / "tokenizer.json"
    graph.write_bytes(b"mock graph; no ORT load")
    tokenizer.write_text("{}")
    records = [{"path": path.name, "bytes": path.stat().st_size,
                "sha256": verifier.file_hash(path), "role": role}
               for path, role in [(graph, "model_graph"), (tokenizer, "tokenizer")]]
    manifest_path = root / "bundle-manifest-v1.json"
    manifest_path.write_text(json.dumps({"schema_version": "vons.bundle.manifest/v1",
                                         "metadata": {"backend": "direct"}, "files": records}))
    report = {"schema": verifier.SCHEMA, "backend": "direct",
              "requested_provider": "wasm",
              "manifest_hash": verifier.file_hash(manifest_path),
              "tokenizer_hash": verifier.file_hash(tokenizer)}
    return root, report, manifest_path


def test_artifact_identity_pinned_before_reference_loading(tmp_path, monkeypatch):
    root, report, manifest_path = make_artifacts(tmp_path)
    calls = []
    monkeypatch.setattr(verifier, "verify_bundle_manifest", lambda path: calls.append(path) or {"pass": True})
    artifacts = verifier.validate_artifacts(report, manifest_path)
    assert calls == [manifest_path]
    assert artifacts["graph"] == root / "model.onnx"
    report["manifest_hash"] = "0" * 64
    with pytest.raises(verifier.ParityError, match="manifest_hash"):
        verifier.validate_artifacts(report, manifest_path)
    assert calls == [manifest_path]


@pytest.mark.parametrize("mutation", ["missing_hash", "tokenizer_hash", "mutated_file", "missing_file"])
def test_missing_and_changed_artifacts_fail(tmp_path, monkeypatch, mutation):
    root, report, path = make_artifacts(tmp_path)
    monkeypatch.setattr(verifier, "verify_bundle_manifest", lambda path: {"pass": True})
    if mutation == "missing_hash":
        del report["manifest_hash"]
    elif mutation == "tokenizer_hash":
        report["tokenizer_hash"] = "0" * 64
    elif mutation == "mutated_file":
        (root / "model.onnx").write_bytes(b"tampered")
    else:
        (root / "model.onnx").unlink()
    with pytest.raises(verifier.ParityError):
        verifier.validate_artifacts(report, path)


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "C:/drive", "https:remote", "x\\y"])
def test_manifest_path_escape_rejected(tmp_path, relative):
    with pytest.raises(verifier.ParityError):
        verifier._relative_file(tmp_path, relative)


def test_symlink_parent_even_inside_bundle_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "data").write_bytes(b"data")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(verifier.ParityError, match="symlink"):
        verifier._relative_file(tmp_path, "linked/data")


def test_json_duplicate_keys_and_nan_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    for payload in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(payload)
        with pytest.raises(verifier.ParityError):
            verifier.read_json(path)


def test_existing_evidence_is_not_overwritten_or_loaded(tmp_path):
    output = tmp_path / "evidence.json"
    output.write_text("preserved")
    with pytest.raises(FileExistsError, match="overwrite"):
        verifier.verify_report(tmp_path / "missing-report", tmp_path / "missing-manifest", output)
    assert output.read_text() == "preserved"


@pytest.mark.parametrize("atol,rtol", [(math.nan, 1e-5), (1e-5, math.inf), (-1, 0)])
def test_invalid_tolerances_rejected(atol, rtol):
    report, manifest = fixture()
    with pytest.raises(verifier.ParityError, match="tolerances"):
        verifier.verify_records(report, manifest, TinyTokenizer(), Reference(), atol, rtol)
