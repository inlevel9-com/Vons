"""Summarize a JSONL stream of Vons browser benchmark samples.

Inputs are one raw browser sample per JSONL line. Each line must carry the
full per-sample schema plus a provenance record that describes the session
configuration. The tool performs strict validation before producing any
summary: identity duplicates, malformed types, non-finite timings, mixed
experimental conditions, and diffusion rows without noise hashes all cause
the tool to fail before the output path is touched.

Successful rows contribute to linearly interpolated p50/p95 percentiles per
timing stage. Failed rows are counted explicitly and retain nullable timing
slots. Heap maxima are derived only from actually-measured observations.
Measurement completeness (>= 5 independent session starts, >= 10 warmups
excluded per session, >= 100 timed rows total) is reported as a separate
flag without elevating partial runs to "complete". Because the page harness
performs in-page reloads only, "process cold" is never claimed, and the
requested provider is never represented as proven.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

TIMING_FIELDS: tuple[str, ...] = (
    "tokenizeMs",
    "prepareFeedMs",
    "inferenceAndReadbackMs",
    "postprocessMs",
    "totalRequestMs",
)

HEAP_FIELDS: tuple[str, ...] = (
    "js_heap_used_bytes_before",
    "js_heap_used_bytes_after",
)

PROVIDERS = {"wasm", "webgpu"}
BACKENDS = {"direct", "diffusion"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_IDENTITY_FIELDS: tuple[str, ...] = ("run_id", "session_start", "repeat")
REQUIRED_TOP_FIELDS: tuple[str, ...] = (
    "run_id",
    "session_start",
    "repeat",
    "case_id",
    "backend",
    "requested_provider",
    "observed_provider",
    "bundle_hash",
    "tokenizer_hash",
    "input_hash",
    "candidate_count",
    "padded_option_slots",
    "sequence_length",
    "seed",
    "ort_version",
    "status",
)

REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "environment",
    "execution_provider_evidence",
    "warmup_excluded_per_session",
    "session_start_semantics",
)

CONDITION_KEYS: tuple[str, ...] = (
    "run_id",
    "backend",
    "requested_provider",
    "observed_provider",
    "bundle_hash",
    "tokenizer_hash",
    "candidate_count",
    "padded_option_slots",
    "sequence_length",
    "ort_version",
)

MAX_PADDED_OPTION_SLOTS = 32
MAX_SEQUENCE_LENGTH = 512


def _finite_number(
    value: Any, label: str, *, allow_none: bool = False
) -> float | None:
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
        raise ValueError(f"{label} must be a lowercase SHA-256 hex string")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be within [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    )


def _provenance_condition_signature(provenance: dict[str, Any]) -> dict[str, Any]:
    environment = provenance["environment"]
    evidence = provenance["execution_provider_evidence"]
    warmup = provenance["warmup_excluded_per_session"]
    semantics = provenance["session_start_semantics"]
    return {
        "environment": environment,
        "execution_provider_evidence": evidence,
        "warmup_excluded_per_session": warmup,
        "session_start_semantics": semantics,
    }


def _validate_row(row: dict[str, Any], line_no: int) -> None:
    if not isinstance(row, dict):
        raise TypeError(f"line {line_no}: sample must be a JSON object")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError(f"line {line_no}: provenance object is required")
    for field in REQUIRED_TOP_FIELDS:
        if field not in row:
            raise ValueError(f"line {line_no}: missing required field {field}")
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in provenance:
            raise ValueError(
                f"line {line_no}: provenance missing required field {field}"
            )

    _nonempty_string(row["run_id"], f"line {line_no}.run_id")
    _positive_integer(row["session_start"], f"line {line_no}.session_start")
    if isinstance(row["repeat"], bool) or not isinstance(row["repeat"], int) or row["repeat"] < 0:
        raise ValueError(f"line {line_no}.repeat must be a non-negative integer")

    _nonempty_string(row["case_id"], f"line {line_no}.case_id")
    if row["backend"] not in BACKENDS:
        raise ValueError(f"line {line_no}.backend must be direct or diffusion")
    if row["requested_provider"] not in PROVIDERS:
        raise ValueError(
            f"line {line_no}.requested_provider must be wasm or webgpu"
        )
    observed = row["observed_provider"]
    if observed is not None and observed not in PROVIDERS:
        raise ValueError(
            f"line {line_no}.observed_provider must be wasm, webgpu, or null"
        )

    _sha256(row["bundle_hash"], f"line {line_no}.bundle_hash")
    _sha256(row["tokenizer_hash"], f"line {line_no}.tokenizer_hash")
    _sha256(row["input_hash"], f"line {line_no}.input_hash")
    noise_hash = _sha256(
        row.get("noise_hash"), f"line {line_no}.noise_hash", allow_none=True
    )
    if row["backend"] == "diffusion" and noise_hash is None:
        raise ValueError(
            f"line {line_no}.noise_hash is required for diffusion backend"
        )

    _positive_integer(row["candidate_count"], f"line {line_no}.candidate_count")
    _positive_integer(
        row["padded_option_slots"], f"line {line_no}.padded_option_slots"
    )
    _positive_integer(row["sequence_length"], f"line {line_no}.sequence_length")
    if "seed" not in row:
        raise ValueError(f"line {line_no}: missing required field seed")
    seed_value = row["seed"]
    if seed_value is None:
        if row["backend"] != "direct" or noise_hash is not None:
            raise ValueError(
                f"line {line_no}.seed may only be null for direct backend with "
                "noise_hash=null (deterministic-direct; seed is not_applicable); "
                "diffusion or direct-with-noise require a non-negative integer seed"
            )
    else:
        _nonnegative_integer(seed_value, f"line {line_no}.seed")
    if row["candidate_count"] > row["padded_option_slots"]:
        raise ValueError(
            f"line {line_no}: candidate_count must be <= padded_option_slots"
        )
    if row["padded_option_slots"] > MAX_PADDED_OPTION_SLOTS:
        raise ValueError(
            f"line {line_no}: padded_option_slots must be <= {MAX_PADDED_OPTION_SLOTS}"
        )
    if row["sequence_length"] > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"line {line_no}: sequence_length must be <= {MAX_SEQUENCE_LENGTH}"
        )
    _nonempty_string(row["ort_version"], f"line {line_no}.ort_version")

    status = row["status"]
    if status not in {"ok", "error", "timeout"}:
        raise ValueError(
            f"line {line_no}.status must be ok, error, or timeout"
        )
    error = row.get("error")
    if status in {"error", "timeout"}:
        if error is None:
            pass
        elif not isinstance(error, str) or not error:
            raise ValueError(
                f"line {line_no}.error must be a non-empty string for status {status}"
            )
    elif status == "ok" and error is not None:
        raise ValueError(
            f"line {line_no}.error must be null for successful samples"
        )

    for field in TIMING_FIELDS:
        timing = _finite_number(
            row.get(field), f"line {line_no}.{field}", allow_none=True
        )
        if status == "ok" and timing is None:
            raise ValueError(
                f"line {line_no}: successful sample requires {field}"
            )

    for field in HEAP_FIELDS:
        _finite_number(row.get(field), f"line {line_no}.{field}", allow_none=True)

    environment = provenance["environment"]
    if not isinstance(environment, dict):
        raise TypeError(f"line {line_no}.provenance.environment must be an object")
    _nonempty_string(
        provenance["execution_provider_evidence"],
        f"line {line_no}.provenance.execution_provider_evidence",
    )
    _positive_integer(
        provenance["warmup_excluded_per_session"],
        f"line {line_no}.provenance.warmup_excluded_per_session",
    )
    _nonempty_string(
        provenance["session_start_semantics"],
        f"line {line_no}.provenance.session_start_semantics",
    )


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        line_no = 0
        for raw in handle:
            line_no += 1
            stripped = raw.strip()
            if not stripped:
                continue
            yield line_no, json.loads(stripped)


def _condition_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    provenance_sig = _provenance_condition_signature(row["provenance"])
    return (
        tuple(row[key] for key in CONDITION_KEYS),
        provenance_sig["warmup_excluded_per_session"],
        provenance_sig["session_start_semantics"],
        json.dumps(provenance_sig["environment"], sort_keys=True),
        provenance_sig["execution_provider_evidence"],
    )


def validate_and_aggregate(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    rows: list[dict[str, Any]] = []
    seen_identities: set[tuple[str, int, int]] = set()
    first_condition: tuple[Any, ...] | None = None
    provenance_reference: dict[str, Any] | None = None
    session_samples: dict[int, int] = {}
    success_count = 0
    failure_count = 0
    total_rows = 0

    stage_timings: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}
    heap_values: dict[str, list[float]] = {field: [] for field in HEAP_FIELDS}

    for line_no, row in _iter_jsonl(input_path):
        _validate_row(row, line_no)
        total_rows += 1
        identity = (row["run_id"], row["session_start"], row["repeat"])
        if identity in seen_identities:
            raise ValueError(
                f"duplicate (run_id, session_start, repeat) identity at line {line_no}: {identity}"
            )
        seen_identities.add(identity)

        condition = _condition_tuple(row)
        if first_condition is None:
            first_condition = condition
            provenance_reference = _provenance_condition_signature(
                row["provenance"]
            )
        elif condition != first_condition:
            raise ValueError(
                f"line {line_no}: mixed experimental conditions detected; "
                "bundle, tokenizer, shape, provider, runtime, environment, "
                "and config provenance fields must be homogeneous across all rows"
            )

        session_start = row["session_start"]
        session_samples[session_start] = session_samples.get(session_start, 0) + 1

        status = row["status"]
        if status == "ok":
            success_count += 1
            for field in TIMING_FIELDS:
                stage_timings[field].append(float(row[field]))
            for field in HEAP_FIELDS:
                value = row.get(field)
                if value is not None:
                    heap_values[field].append(float(value))
        else:
            failure_count += 1

        rows.append(row)

    if total_rows == 0:
        raise ValueError("input JSONL contains no samples")

    return _build_summary(
        rows=rows,
        input_path=input_path,
        provenance_reference=provenance_reference,
        first_condition=first_condition,
        session_samples=session_samples,
        success_count=success_count,
        failure_count=failure_count,
        total_rows=total_rows,
        stage_timings=stage_timings,
        heap_values=heap_values,
    )


def _build_summary(
    *,
    rows: list[dict[str, Any]],
    input_path: Path,
    provenance_reference: dict[str, Any] | None,
    first_condition: tuple[Any, ...] | None,
    session_samples: dict[int, int],
    success_count: int,
    failure_count: int,
    total_rows: int,
    stage_timings: dict[str, list[float]],
    heap_values: dict[str, list[float]],
) -> dict[str, Any]:
    assert provenance_reference is not None
    assert first_condition is not None
    assert rows

    first_row = rows[0]
    unique_sessions = len(session_samples)
    warmup_per_session = provenance_reference["warmup_excluded_per_session"]
    timed_requests_total = success_count + failure_count
    complete = (
        unique_sessions >= 5
        and warmup_per_session >= 10
        and timed_requests_total >= 100
    )
    incomplete_reasons: list[str] = []
    if unique_sessions < 5:
        incomplete_reasons.append(
            f"only {unique_sessions} independent session starts observed (>= 5 required)"
        )
    if warmup_per_session < 10:
        incomplete_reasons.append(
            f"warmup_excluded_per_session={warmup_per_session} (>= 10 required)"
        )
    if timed_requests_total < 100:
        incomplete_reasons.append(
            f"only {timed_requests_total} timed requests total (>= 100 required)"
        )

    process_cold_claimable = False

    requested_provider = first_row["requested_provider"]
    observed_provider = first_row["observed_provider"]
    requested_is_proven = False

    summary_timings: dict[str, dict[str, float | None]] = {}
    for field in TIMING_FIELDS:
        values = stage_timings[field]
        summary_timings[field] = {
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }

    heap_maxima: dict[str, float | None] = {}
    for field in HEAP_FIELDS:
        values = heap_values[field]
        heap_maxima[field] = max(values) if values else None

    per_session_samples: dict[str, int] = {
        str(session_start): count
        for session_start, count in sorted(session_samples.items())
    }

    summary: dict[str, Any] = {
        "schema": "vons.browser-benchmark-summary/v1",
        "generated_from": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "experimental_condition": {
            "run_id": first_row["run_id"],
            "backend": first_row["backend"],
            "requested_provider": requested_provider,
            "observed_provider": observed_provider,
            "bundle_hash": first_row["bundle_hash"],
            "tokenizer_hash": first_row["tokenizer_hash"],
            "candidate_count": first_row["candidate_count"],
            "padded_option_slots": first_row["padded_option_slots"],
            "sequence_length": first_row["sequence_length"],
            "seed": first_row["seed"],
            "ort_version": first_row["ort_version"],
            "provenance": provenance_reference,
        },
        "provider_claim_note": (
            "requested_provider is not represented as proven in this v1 summary. "
            "observed_provider preserves the browser-reported value and "
            "execution_provider_evidence preserves the raw evidence string, "
            "but no structured verified execution artifact exists to elevate "
            "a match to a proven claim; requested_is_proven is always False in v1."
        ),
        "process_cold_note": (
            "process-level cold start identity is not verifiable from page-only "
            "browser observations in this v1 schema; session_start_semantics is "
            "preserved as a raw label but process_cold_claimable is always False "
            "regardless of semantics string content."
        ),
        "sample_counts": {
            "total_rows": total_rows,
            "successful": success_count,
            "failed": failure_count,
            "unique_session_starts": unique_sessions,
            "per_session_samples": per_session_samples,
        },
        "stage_percentiles_ms": summary_timings,
        "heap_maxima_bytes": heap_maxima,
        "measurement_completeness": {
            "complete": complete,
            "required_min_session_starts": 5,
            "required_min_warmup_excluded_per_session": 10,
            "required_min_timed_requests_total": 100,
            "actual_session_starts": unique_sessions,
            "actual_warmup_excluded_per_session": warmup_per_session,
            "actual_timed_requests_total": timed_requests_total,
            "incomplete_reasons": incomplete_reasons,
        },
        "process_cold_claimable": bool(process_cold_claimable),
        "requested_is_proven": bool(requested_is_proven),
    }
    return summary


def _iter_report_samples(
    report: dict[str, Any], report_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (top_provenance, normalized_rows) from a v1 browser-benchmark report.

    Normalizes current report field names to the JSONL schema used by the
    validator: live_candidates -> candidate_count, allocated_candidates ->
    padded_option_slots, sequence_length preferred with sequence_tokens as
    fallback alias. Never invents missing values. Raises ValueError on any
    required field that cannot be resolved.
    """
    if not isinstance(report, dict):
        raise TypeError("browser benchmark report must be a JSON object")
    if report.get("schema") != "vons.browser-benchmark/v1":
        raise ValueError("unexpected browser benchmark schema")

    _nonempty_string(report.get("run_id"), f"{report_path.name}.run_id")
    backend = report.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"{report_path.name}.backend must be direct or diffusion")
    requested_provider = report.get("requested_provider")
    if requested_provider not in PROVIDERS:
        raise ValueError(
            f"{report_path.name}.requested_provider must be wasm or webgpu"
        )
    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise TypeError(
            f"{report_path.name}.environment is required as provenance input"
        )
    execution_provider_evidence = _nonempty_string(
        report.get("execution_provider_evidence"),
        f"{report_path.name}.execution_provider_evidence",
    )
    warmup_excluded_per_session = _positive_integer(
        report.get("warmup_excluded_per_session"),
        f"{report_path.name}.warmup_excluded_per_session",
    )
    session_start_semantics_raw = report.get("session_start_semantics")
    if session_start_semantics_raw is None:
        raise ValueError(
            f"{report_path.name}.session_start_semantics is required to build "
            "per-row provenance; do not invent missing values."
        )
    session_start_semantics = _nonempty_string(
        session_start_semantics_raw,
        f"{report_path.name}.session_start_semantics",
    )
    top_provenance = {
        "environment": environment,
        "execution_provider_evidence": execution_provider_evidence,
        "warmup_excluded_per_session": warmup_excluded_per_session,
        "session_start_semantics": session_start_semantics,
    }

    samples = report.get("samples")
    if not isinstance(samples, list):
        raise TypeError(f"{report_path.name}.samples must be a list")
    if not samples:
        raise ValueError(f"{report_path.name}.samples contains no rows")

    ort_version = report.get("ort_version")
    ort_version_is_str = isinstance(ort_version, str) and bool(ort_version)
    if not ort_version_is_str:
        runtime_info = report.get("runtime_info")
        if isinstance(runtime_info, dict):
            rt_ort = runtime_info.get("ortVersion")
            if isinstance(rt_ort, str) and rt_ort:
                ort_version = rt_ort
                ort_version_is_str = True

    normalized: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise TypeError(f"{report_path.name} sample {idx} must be an object")

        live_candidates = sample.get("live_candidates")
        if live_candidates is None:
            raise ValueError(
                f"{report_path.name} sample {idx} is missing live_candidates "
                "(source for candidate_count normalization)"
            )
        allocated_candidates = sample.get("allocated_candidates")
        if allocated_candidates is None:
            raise ValueError(
                f"{report_path.name} sample {idx} is missing allocated_candidates "
                "(source for padded_option_slots normalization)"
            )
        sequence_length = sample.get("sequence_length")
        if sequence_length is None:
            sequence_length = sample.get("sequence_tokens")
            if sequence_length is None:
                raise ValueError(
                    f"{report_path.name} sample {idx} is missing sequence_length "
                    "(sequence_tokens not present as alias fallback)"
                )
        sample_ort_version = sample.get("ort_version")
        if not (isinstance(sample_ort_version, str) and sample_ort_version):
            if ort_version_is_str:
                sample_ort_version = ort_version
            else:
                raise ValueError(
                    f"{report_path.name} sample {idx} is missing ort_version and "
                    "report does not provide a top-level ort_version fallback "
                    "(checked both ort_version and runtime_info.ortVersion)"
                )
        sample_backend = sample.get("backend")
        noise_hash = sample.get("noise_hash")
        sample_seed = sample.get("seed")
        if sample_seed is None and sample_backend == "direct" and noise_hash is None:
            resolved_seed = None
        else:
            if "seed" not in sample:
                raise ValueError(
                    f"{report_path.name} sample {idx} is missing seed; "
                    "null seed is only permitted for direct backend with "
                    "noise_hash=null (deterministic-direct, not_applicable)"
                )
            resolved_seed = sample_seed
        error_value = sample.get("error")
        observed_provider = sample.get("observed_provider")

        mapped: dict[str, Any] = {
            "run_id": sample["run_id"],
            "session_start": sample["session_start"],
            "repeat": sample["repeat"],
            "case_id": sample["case_id"],
            "backend": sample_backend,
            "requested_provider": sample["requested_provider"],
            "observed_provider": observed_provider,
            "bundle_hash": sample["bundle_hash"],
            "tokenizer_hash": sample["tokenizer_hash"],
            "input_hash": sample["input_hash"],
            "noise_hash": noise_hash,
            "candidate_count": live_candidates,
            "padded_option_slots": allocated_candidates,
            "sequence_length": sequence_length,
            "seed": resolved_seed,
            "ort_version": sample_ort_version,
            "status": sample["status"],
            "error": error_value,
            "provenance": top_provenance,
        }
        for field in TIMING_FIELDS:
            mapped[field] = sample.get(field)
        for field in HEAP_FIELDS:
            if field in sample:
                mapped[field] = sample.get(field)
        normalized.append(mapped)
    return top_provenance, normalized


def validate_and_aggregate_report(report_path: Path) -> dict[str, Any]:
    """Entry point for --report-input: loads an existing v1 browser-benchmark
    JSON report, normalizes its samples to the JSONL schema, then runs the
    same validation and aggregation used by validate_and_aggregate.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance_reference, normalized_rows = _iter_report_samples(report, report_path)

    total_rows = 0
    seen_identities: set[tuple[str, int, int]] = set()
    first_condition: tuple[Any, ...] | None = None
    session_samples: dict[int, int] = {}
    success_count = 0
    failure_count = 0
    stage_timings: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}
    heap_values: dict[str, list[float]] = {field: [] for field in HEAP_FIELDS}
    accepted_rows: list[dict[str, Any]] = []

    for line_no, row in enumerate(normalized_rows, start=1):
        _validate_row(row, line_no)
        total_rows += 1
        identity = (row["run_id"], row["session_start"], row["repeat"])
        if identity in seen_identities:
            raise ValueError(
                f"duplicate (run_id, session_start, repeat) identity in sample {line_no}: {identity}"
            )
        seen_identities.add(identity)
        condition = _condition_tuple(row)
        if first_condition is None:
            first_condition = condition
        elif condition != first_condition:
            raise ValueError(
                f"sample {line_no}: mixed experimental conditions detected in report"
            )
        session_start = row["session_start"]
        session_samples[session_start] = session_samples.get(session_start, 0) + 1
        if row["status"] == "ok":
            success_count += 1
            for field in TIMING_FIELDS:
                stage_timings[field].append(float(row[field]))
            for field in HEAP_FIELDS:
                value = row.get(field)
                if value is not None:
                    heap_values[field].append(float(value))
        else:
            failure_count += 1
        accepted_rows.append(row)

    if total_rows == 0:
        raise ValueError("report samples yielded no rows after normalization")

    return _build_summary(
        rows=accepted_rows,
        input_path=report_path,
        provenance_reference=provenance_reference,
        first_condition=first_condition,
        session_samples=session_samples,
        success_count=success_count,
        failure_count=failure_count,
        total_rows=total_rows,
        stage_timings=stage_timings,
        heap_values=heap_values,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path,
                       help="JSONL file of raw browser benchmark samples")
    group.add_argument("--report-input", type=Path,
                       help="Existing vons.browser-benchmark/v1 JSON report; "
                            "normalizes live_candidates/allocated_candidates/"
                            "sequence_tokens to JSONL schema and carries top-level provenance")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination path for the summary JSON")
    args = parser.parse_args()

    if args.input is not None:
        summary = validate_and_aggregate(args.input)
    else:
        summary = validate_and_aggregate_report(args.report_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
