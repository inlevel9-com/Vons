import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_tool(name: str, file_name: str):
    path = Path(__file__).parents[1] / "tools" / file_name
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark_tool = _load_tool("verify_browser_benchmark", "verify_browser_benchmark.py")
merge_tool = _load_tool("vons_merge_browser_benchmarks", "merge_browser_benchmarks.py")


def _sample(repeat: int) -> dict[str, object]:
    return {
        "questionId": f"case-{repeat + 1}",
        "live_candidates": 4,
        "allocated_candidates": 4,
        "live_tokens": 12,
        "sequence_length": 512,
        "tokenizeMs": 1.0,
        "prepareFeedMs": 2.0,
        "inferenceAndReadbackMs": 3.0 + repeat,
        "postprocessMs": 4.0,
        "totalRequestMs": 10.0 + repeat,
        "run_id": f"run-{repeat}",
        "session_start": 1,
        "repeat": repeat,
        "case_id": f"case-{repeat + 1}",
        "backend": "direct",
        "requested_provider": "wasm",
        "observed_provider": None,
        "observed_provider_status": "session_created; execution_partition_not_exposed",
        "bundle_hash": "c" * 64,
        "tokenizer_hash": "a" * 64,
        "input_hash": "b" * 64,
        "noise_hash": None,
        "js_heap_used_bytes_before": None,
        "js_heap_used_bytes_after": None,
        "status": "ok",
    }


def _report(run_id: str, offset: int) -> dict[str, object]:
    return {
        "schema": "vons.browser-benchmark/v1",
        "run_id": run_id,
        "backend": "direct",
        "requested_provider": "wasm",
        "sessions": 1,
        "warmup_excluded_per_session": 1,
        "repeats_per_session": 2,
        "cases": 2,
        "session_start_semantics": "new ORT session on one page; module and browser caches may persist",
        "execution_provider_evidence": "requested provider and session creation only; graph partition status is not exposed by this harness",
        "environment": {"user_agent": "test", "cross_origin_isolated": False},
        "runtime_info": None,
        "session_records": [{"session_start": 1, "load_ms": 5.0, "status": "ok"}],
        "samples": [_sample(offset), _sample(offset + 1)],
    }


class MergeBrowserBenchmarkTests(unittest.TestCase):
    def test_merges_complete_one_session_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps(_report("first", 0)), encoding="utf-8")
            second.write_text(json.dumps(_report("second", 0)), encoding="utf-8")
            merged = merge_tool.merge_reports([first, second])

        self.assertEqual(merged["sessions"], 2)
        self.assertEqual(len(merged["samples"]), 4)
        self.assertEqual({sample["session_start"] for sample in merged["samples"]}, {1, 2})
        self.assertEqual(merged["merge"]["source_count"], 2)
        self.assertEqual(merged["summary"]["successful_samples"], 4)

    def test_rejects_mismatched_provider(self) -> None:
        first_report = _report("first", 0)
        second_report = _report("second", 0)
        second_report["requested_provider"] = "webgpu"
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps(first_report), encoding="utf-8")
            second.write_text(json.dumps(second_report), encoding="utf-8")
            with self.assertRaises(ValueError):
                merge_tool.merge_reports([first, second])


if __name__ == "__main__":
    unittest.main()
