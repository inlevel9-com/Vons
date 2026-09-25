import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("predict_mind2web", Path(__file__).parents[1] / "tools/predict_mind2web.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(identifier="a"):
    candidates = [{"id": str(i), "text": f"Element {i}"} for i in range(32)]
    return {"id": identifier, "task_id": "task", "split": "test_task",
            "state": {"goal": "select"}, "candidates": candidates,
            "generated_candidates": [item["id"] for item in candidates],
            "input_sha256": "a" * 64}


def test_exact_k_and_label_free_contract():
    assert len(module.make_request(row(), 5)["candidates"]) == 5
    assert len(module.make_request(row(), 32)["candidates"]) == 32
    with pytest.raises(ValueError, match="label-free"):
        module.make_request({**row(), "positive_ids": ["0"]}, 5)
    bad = row()
    bad["generated_candidates"].reverse()
    with pytest.raises(ValueError, match="order"):
        module.make_request(bad, 5)
    missing = row()
    del missing["input_sha256"]
    with pytest.raises(ValueError, match="input_sha256 before inference"):
        module.make_request(missing, 5)


def test_decode_abstention_finite_and_ids():
    result = module.decode([0, 10], 10, ["first", "second"])
    assert result["selection"] == "second"
    assert result["abstention_threshold"] == pytest.approx(0.55 / 2)
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert module.decode([0, 10], -10, ["first", "second"])["selection"] is None
    with pytest.raises(ValueError, match="invalid scores"):
        module.decode([float("nan"), 1], 10, ["first", "second"])
    with pytest.raises(module.ModelOutputError, match="invalid scores"):
        module.decode([-float("inf"), -float("inf")], 10, ["first", "second"])


def test_runner_invokes_each_k_and_preserves_errors(tmp_path):
    source = tmp_path / "retrieval.jsonl"
    source.write_text("\n".join(json.dumps(row(identifier)) for identifier in ["a", "b"]))
    calls = []

    def select(request):
        calls.append(len(request["candidates"]))
        if len(calls) % 2 == 0:
            raise ValueError("input_overflow")
        return {"selection": request["candidates"][0]["id"], "status": "ok"}

    for k in (5, 10):
        output = tmp_path / f"k{k}.jsonl"
        report = module.predict_file(source, output, split="test_task", k=k,
                                     select=select, provenance={"backend": "test"})
        predictions = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(predictions) == 2
        assert predictions[1]["selection"] is None
        assert report["counts"] == {"ok": 1, "error": 1}
        assert not report["selection_reused_across_k"]
        assert all(len(item["generated_candidates"]) == k for item in predictions)
        with pytest.raises(ValueError, match="overwrite"):
            module.predict_file(source, output, split="test_task", k=k,
                                select=select, provenance={})
    assert calls == [5, 5, 10, 10]


def test_nonfinite_model_output_stops_comparison_and_keeps_journal(tmp_path):
    source = tmp_path / "retrieval.jsonl"
    source.write_text(json.dumps(row()))
    output = tmp_path / "predictions.jsonl"

    def invalid(request):
        raise module.ModelOutputError("non-finite output")

    with pytest.raises(module.ModelOutputError):
        module.predict_file(source, output, split="test_task", k=5,
                            select=invalid, provenance={})
    assert not output.exists()
    assert output.with_suffix(".jsonl.partial").exists()
    assert not output.with_suffix(".manifest.json").exists()
