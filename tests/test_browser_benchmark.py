import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "verify_browser_benchmark.py"
    spec = importlib.util.spec_from_file_location("vons_browser_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load browser benchmark verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark_tool = _load_tool()


def _sample(repeat: int, status: str = "ok") -> dict[str, object]:
    values = {
        "tokenizeMs": 1.0 + repeat,
        "prepareFeedMs": 2.0 + repeat,
        "inferenceAndReadbackMs": 3.0 + repeat,
        "postprocessMs": 4.0 + repeat,
        "totalRequestMs": 10.0 + repeat,
    }
    if status == "error":
        values = {key: None for key in values}
    return {
        **values,
        "session_start": 1,
        "repeat": repeat,
        "run_id": "run-1",
        "case_id": f"case-{repeat + 1}",
        "backend": "direct",
        "requested_provider": "wasm",
        "observed_provider": None,
        "observed_provider_status": "session_created; execution_partition_not_exposed",
        "live_candidates": 4 if status == "ok" else None,
        "allocated_candidates": 4 if status == "ok" else None,
        "live_tokens": 12 if status == "ok" else None,
        "sequence_length": 512 if status == "ok" else None,
        "bundle_hash": "a" * 64,
        "tokenizer_hash": "b" * 64,
        "input_hash": "c" * 64,
        "noise_hash": None,
        "status": status,
        "js_heap_used_bytes_before": None,
        "js_heap_used_bytes_after": None,
    }


class BrowserBenchmarkTests(unittest.TestCase):
    def test_recomputes_raw_samples_and_allows_unavailable_memory(self) -> None:
        report = {
            "schema": "vons.browser-benchmark/v1",
            "sessions": 1,
            "warmup_excluded_per_session": 1,
            "repeats_per_session": 3,
            "run_id": "run-1",
            "backend": "direct",
            "cases": 1,
            "requested_provider": "wasm",
            "environment": {"user_agent": "test", "cross_origin_isolated": False},
            "runtime_info": None,
            "execution_provider_evidence": "test evidence",
            "session_records": [{"session_start": 1, "load_ms": 5.0, "status": "ok"}],
            "samples": [_sample(0), _sample(1), _sample(2, "error")],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = benchmark_tool.verify_report(path)

        self.assertEqual(result["sample_rows"], 3)
        self.assertEqual(result["successful_samples"], 2)
        self.assertEqual(result["failed_samples"], 1)
        self.assertEqual(result["recomputed_summary"]["total_request_ms_p50"], 10.5)
        self.assertIsNone(result["recomputed_summary"]["js_heap_before_max"])

    def test_rejects_failed_sample_with_fabricated_timing(self) -> None:
        report = {
            "schema": "vons.browser-benchmark/v1",
            "sessions": 1,
            "warmup_excluded_per_session": 1,
            "repeats_per_session": 1,
            "run_id": "run-1",
            "backend": "direct",
            "cases": 1,
            "requested_provider": "webgpu",
            "environment": {"user_agent": "test", "cross_origin_isolated": False},
            "runtime_info": None,
            "execution_provider_evidence": "test evidence",
            "session_records": [{"session_start": 1, "load_ms": 5.0, "status": "ok"}],
            "samples": [_sample(0, "error")],
        }
        report["samples"][0]["requested_provider"] = "webgpu"
        report["samples"][0]["totalRequestMs"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark_tool.verify_report(path)

    def test_rejects_sample_without_provenance_hashes(self) -> None:
        report = {
            "schema": "vons.browser-benchmark/v1",
            "sessions": 1,
            "warmup_excluded_per_session": 1,
            "repeats_per_session": 1,
            "run_id": "run-1",
            "backend": "direct",
            "cases": 1,
            "requested_provider": "wasm",
            "environment": {"user_agent": "test", "cross_origin_isolated": False},
            "runtime_info": None,
            "execution_provider_evidence": "test evidence",
            "session_records": [{"session_start": 1, "load_ms": 5.0, "status": "ok"}],
            "samples": [_sample(0)],
        }
        report["samples"][0]["input_hash"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark_tool.verify_report(path)

    def test_rejects_partial_session_matrix_by_default(self) -> None:
        report = {
            "schema": "vons.browser-benchmark/v1",
            "sessions": 2,
            "warmup_excluded_per_session": 1,
            "repeats_per_session": 1,
            "run_id": "run-1",
            "backend": "direct",
            "cases": 1,
            "requested_provider": "wasm",
            "environment": {"user_agent": "test", "cross_origin_isolated": False},
            "runtime_info": None,
            "execution_provider_evidence": "test evidence",
            "session_records": [
                {"session_start": 1, "load_ms": 5.0, "status": "ok"},
                {"session_start": 2, "load_ms": 1.0, "status": "error", "error_kind": "fetch"},
            ],
            "samples": [_sample(0)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark_tool.verify_report(path)
            result = benchmark_tool.verify_report(path, allow_incomplete=True)

        self.assertFalse(result["complete"])
        self.assertTrue(result["incomplete_reasons"])


if __name__ == "__main__":
    unittest.main()
