"""Validate and recompute a saved Vons browser benchmark report.

The browser page is responsible for collecting raw samples. This tool is the
second, offline boundary: it rejects malformed samples and recomputes summary
statistics instead of trusting the values emitted by the page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

TIMING_FIELDS = (
    "tokenizeMs",
    "prepareFeedMs",
    "inferenceAndReadbackMs",
    "postprocessMs",
    "totalRequestMs",
)
PROVIDERS = {"wasm", "webgpu"}
BACKENDS = {"direct", "diffusion"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number or null")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return numeric


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex string" )
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _assert_close(actual: Any, expected: float | None, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label} must be null, got {actual!r}")
        return
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"{label} does not match recomputed value {expected!r}: {actual!r}")


def verify_report(report_path: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("browser benchmark report must be an object")
    if report.get("schema") != "vons.browser-benchmark/v1":
        raise ValueError("unexpected browser benchmark schema")

    sessions = _positive_integer(report.get("sessions"), "sessions")
    warmup = _positive_integer(report.get("warmup_excluded_per_session"), "warmup_excluded_per_session")
    repeats = _positive_integer(report.get("repeats_per_session"), "repeats_per_session")
    _positive_integer(report.get("cases"), "cases")
    if not isinstance(report.get("run_id"), str) or not report["run_id"]:
        raise TypeError("run_id is required")
    backend = report.get("backend")
    if backend not in BACKENDS:
        raise ValueError("backend must be direct or diffusion")
    provider = report.get("requested_provider")
    if provider not in PROVIDERS:
        raise ValueError("requested_provider must be wasm or webgpu")
    environment = report.get("environment")
    if (
        not isinstance(environment, dict)
        or not isinstance(environment.get("user_agent"), str)
        or not environment["user_agent"]
    ):
        raise TypeError("environment.user_agent is required")
    if not isinstance(environment.get("cross_origin_isolated"), bool):
        raise TypeError("environment.cross_origin_isolated must be boolean")
    runtime_info = report.get("runtime_info")
    if runtime_info is not None and not isinstance(runtime_info, dict):
        raise TypeError("runtime_info must be an object or null")
    if not isinstance(report.get("execution_provider_evidence"), str) or not report["execution_provider_evidence"]:
        raise TypeError("execution_provider_evidence is required")
    samples = report.get("samples")
    session_records = report.get("session_records")
    if not isinstance(samples, list) or not isinstance(session_records, list):
        raise TypeError("samples and session_records must be lists")

    seen_slots: set[tuple[int, int]] = set()
    successful: list[dict[str, Any]] = []
    failures = 0
    observed_providers: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise TypeError(f"sample {index} must be an object")
        session_start = sample.get("session_start")
        repeat = sample.get("repeat")
        if (
            isinstance(session_start, bool)
            or not isinstance(session_start, int)
            or not 1 <= session_start <= sessions
        ):
            raise ValueError(f"sample {index} has an invalid session_start")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not 0 <= repeat < repeats:
            raise ValueError(f"sample {index} has an invalid repeat")
        slot = (session_start, repeat)
        if slot in seen_slots:
            raise ValueError(f"duplicate sample slot {slot}")
        seen_slots.add(slot)
        if sample.get("requested_provider") != provider:
            raise ValueError(f"sample {index} requested_provider disagrees with report")
        if sample.get("backend") != backend:
            raise ValueError(f"sample {index} backend disagrees with report")
        if not isinstance(sample.get("run_id"), str) or not sample["run_id"]:
            raise TypeError(f"sample {index}.run_id is required")
        if not isinstance(sample.get("case_id"), str) or not sample["case_id"]:
            raise TypeError(f"sample {index}.case_id is required")
        _sha256(sample.get("bundle_hash"), f"sample {index}.bundle_hash")
        _sha256(sample.get("tokenizer_hash"), f"sample {index}.tokenizer_hash")
        _sha256(sample.get("input_hash"), f"sample {index}.input_hash")
        noise_hash = _sha256(sample.get("noise_hash"), f"sample {index}.noise_hash", allow_none=True)
        if backend == "diffusion" and noise_hash is None:
            raise ValueError(f"sample {index}.noise_hash is required for diffusion")
        if backend == "direct" and noise_hash is not None:
            raise ValueError(f"sample {index}.noise_hash must be null for direct")
        observed = sample.get("observed_provider")
        if observed is not None:
            if observed not in PROVIDERS:
                raise ValueError(f"sample {index} has an invalid observed_provider")
            observed_providers.add(observed)
        status = sample.get("status")
        if status not in {"ok", "error"}:
            raise ValueError(f"sample {index} status must be ok or error")
        if not isinstance(sample.get("observed_provider_status"), str) or not sample["observed_provider_status"]:
            raise ValueError(f"sample {index} requires observed_provider_status")
        for field in TIMING_FIELDS:
            timing = _finite_number(sample.get(field), f"sample {index}.{field}", allow_none=True)
            if status == "ok" and timing is None:
                raise ValueError(f"successful sample {index}.{field} must be numeric")
            if status == "error" and timing is not None:
                raise ValueError(f"failed sample {index}.{field} must be null")
        shape_fields = ("live_candidates", "allocated_candidates", "live_tokens", "sequence_length")
        for field in shape_fields:
            value = sample.get(field)
            if status == "ok":
                _nonnegative_integer(value, f"sample {index}.{field}")
            elif value is not None:
                raise ValueError(f"failed sample {index}.{field} must be null")
        if status == "ok" and sample["allocated_candidates"] < sample["live_candidates"]:
            raise ValueError(f"sample {index} allocated_candidates is below live_candidates")
        for field in ("js_heap_used_bytes_before", "js_heap_used_bytes_after"):
            _finite_number(sample.get(field), f"sample {index}.{field}", allow_none=True)
        if status == "ok":
            successful.append(sample)
        else:
            failures += 1

    session_record_ids: set[int] = set()
    session_failures: list[int] = []
    for index, record in enumerate(session_records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"session record {index} must be an object")
        session_start = record.get("session_start")
        if (
            isinstance(session_start, bool)
            or not isinstance(session_start, int)
            or not 1 <= session_start <= sessions
        ):
            raise ValueError(f"session record {index} has an invalid session_start")
        if session_start in session_record_ids:
            raise ValueError(f"duplicate session record {session_start}")
        session_record_ids.add(session_start)
        _finite_number(record.get("load_ms"), f"session record {index}.load_ms")
        if record.get("status") not in {"ok", "error"}:
            raise ValueError(f"session record {index} status must be ok or error")
        if record["status"] == "error":
            session_failures.append(session_start)

    incomplete_reasons: list[str] = []
    if len(session_records) != sessions:
        incomplete_reasons.append(f"expected {sessions} session records, got {len(session_records)}")
    if session_failures:
        incomplete_reasons.append(f"session creation failed for starts {session_failures}")
    expected_samples = sessions * repeats
    if len(samples) != expected_samples:
        incomplete_reasons.append(f"expected {expected_samples} timed samples, got {len(samples)}")
    if incomplete_reasons and not allow_incomplete:
        raise ValueError("incomplete browser benchmark: " + "; ".join(incomplete_reasons))

    total_values = [float(sample["totalRequestMs"]) for sample in successful]
    inference_values = [float(sample["inferenceAndReadbackMs"]) for sample in successful]
    before_values = [
        float(sample["js_heap_used_bytes_before"])
        for sample in successful
        if sample.get("js_heap_used_bytes_before") is not None
    ]
    after_values = [
        float(sample["js_heap_used_bytes_after"])
        for sample in successful
        if sample.get("js_heap_used_bytes_after") is not None
    ]
    recomputed_summary = {
        "successful_samples": len(successful),
        "failed_samples": failures,
        "total_request_ms_p50": percentile(total_values, 0.5),
        "total_request_ms_p95": percentile(total_values, 0.95),
        "inference_and_readback_ms_p50": percentile(inference_values, 0.5),
        "inference_and_readback_ms_p95": percentile(inference_values, 0.95),
        "js_heap_before_max": max(before_values) if before_values else None,
        "js_heap_after_max": max(after_values) if after_values else None,
    }
    supplied_summary = report.get("summary")
    if supplied_summary is not None:
        if not isinstance(supplied_summary, dict):
            raise TypeError("summary must be an object")
        for key, expected in recomputed_summary.items():
            _assert_close(supplied_summary.get(key), expected, f"summary.{key}")

    return {
        "schema": "vons.browser-benchmark-verification/v1",
        "measurement_label": "verified_browser_benchmark_raw_samples",
        "complete": not incomplete_reasons,
        "incomplete_reasons": incomplete_reasons,
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "requested_provider": provider,
        "observed_providers": sorted(observed_providers),
        "sessions": sessions,
        "warmup_excluded_per_session": warmup,
        "repeats_per_session": repeats,
        "sample_rows": len(samples),
        "successful_samples": len(successful),
        "failed_samples": failures,
        "recomputed_summary": recomputed_summary,
        "memory_note": "JS heap values are before/after observations only; null means unavailable; no GPU peak is inferred.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="validate and summarize a partial diagnostic run without treating it as complete",
    )
    args = parser.parse_args()
    result = verify_report(args.report, allow_incomplete=args.allow_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
