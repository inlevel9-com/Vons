"""Merge independently saved one-session browser benchmark reports.

Running one fresh ORT session per page/report avoids conflating session
teardown and reinitialization failures with the timed request population. The
merged report is still passed through the strict raw-sample verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from verify_browser_benchmark import file_sha256, percentile, verify_report


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample["status"] == "ok"]
    failed = len(samples) - len(successful)

    def values(field: str) -> list[float]:
        return [float(sample[field]) for sample in successful]

    def memory_values(field: str) -> list[float]:
        return [float(sample[field]) for sample in successful if sample.get(field) is not None]

    before = memory_values("js_heap_used_bytes_before")
    after = memory_values("js_heap_used_bytes_after")
    return {
        "successful_samples": len(successful),
        "failed_samples": failed,
        "total_request_ms_p50": percentile(values("totalRequestMs"), 0.5),
        "total_request_ms_p95": percentile(values("totalRequestMs"), 0.95),
        "inference_and_readback_ms_p50": percentile(values("inferenceAndReadbackMs"), 0.5),
        "inference_and_readback_ms_p95": percentile(values("inferenceAndReadbackMs"), 0.95),
        "js_heap_before_max": max(before) if before else None,
        "js_heap_after_max": max(after) if after else None,
        "memory_note": "performance.memory before/after observations only; not a peak or GPU-memory measurement",
    }


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two one-session reports are required")
    raw_reports = [_read_report(path) for path in paths]
    for path in paths:
        verify_report(path)

    baseline = raw_reports[0]
    comparable_keys = (
        "backend",
        "requested_provider",
        "warmup_excluded_per_session",
        "repeats_per_session",
        "cases",
        "environment",
        "runtime_info",
        "execution_provider_evidence",
    )
    for index, report in enumerate(raw_reports, start=1):
        if report.get("sessions") != 1:
            raise ValueError(f"source report {index} must have sessions=1")
        for key in comparable_keys:
            if report.get(key) != baseline.get(key):
                raise ValueError(f"source report {index} does not match baseline field {key}")

    source_hashes = [file_sha256(path) for path in paths]
    merged_run_id = "merged-" + hashlib.sha256("\n".join(source_hashes).encode()).hexdigest()[:16]
    samples: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    for session_start, report in enumerate(raw_reports, start=1):
        record = dict(report["session_records"][0])
        record["session_start"] = session_start
        session_records.append(record)
        for raw_sample in report["samples"]:
            sample = dict(raw_sample)
            sample["source_run_id"] = sample.get("run_id")
            sample["run_id"] = merged_run_id
            sample["session_start"] = session_start
            samples.append(sample)

    return {
        "schema": "vons.browser-benchmark/v1",
        "run_id": merged_run_id,
        "backend": baseline["backend"],
        "requested_provider": baseline["requested_provider"],
        "sessions": len(raw_reports),
        "warmup_excluded_per_session": baseline["warmup_excluded_per_session"],
        "repeats_per_session": baseline["repeats_per_session"],
        "cases": baseline["cases"],
        "session_start_semantics": "one fresh page/session per source report; browser caches may persist",
        "execution_provider_evidence": baseline["execution_provider_evidence"],
        "environment": baseline["environment"],
        "runtime_info": baseline.get("runtime_info"),
        "merge": {
            "source_count": len(paths),
            "source_reports": [
                {"path": str(path), "sha256": digest} for path, digest in zip(paths, source_hashes, strict=True)
            ],
        },
        "session_records": session_records,
        "samples": samples,
        "summary": _summary(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_reports(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(merged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
