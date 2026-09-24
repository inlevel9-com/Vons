"""Tests for tools/summarize_browser_benchmark.py.

Covers:
- Exact expected linearly interpolated percentiles (p50, p95) for known datasets
- Null/failed/timeout row preservation with nullable timing slots
- Malformed rows, non-finite timings, wrong types, bad SHA256
- Homogeneous condition enforcement and mixed-condition rejection
- Duplicate (run_id, session_start, repeat) identity rejection
- Insufficient-observation completeness flag without error
- CLI behavior via subprocess on a small valid fixture
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
SUMMARY_TOOL = TOOLS_DIR / "summarize_browser_benchmark.py"

sys.path.insert(0, str(TOOLS_DIR))
from summarize_browser_benchmark import (  # noqa: E402, RUF100
    TIMING_FIELDS,
    _validate_row,
    percentile,
    validate_and_aggregate,
    validate_and_aggregate_report,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_NOISE = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

BASE_ENV = {
    "user_agent": "TestBrowser/1.0",
    "cross_origin_isolated": True,
    "platform": "Test OS",
}


def _provenance(*, warmup: int = 10, semantics: str = "page_reload") -> dict[str, Any]:
    return {
        "environment": dict(BASE_ENV),
        "execution_provider_evidence": "window.ort.env.backend='wasm'",
        "warmup_excluded_per_session": warmup,
        "session_start_semantics": semantics,
    }


def _row(
    *,
    session_start: int = 1,
    repeat: int = 0,
    case_id: str = "case-0",
    backend: str = "direct",
    status: str = "ok",
    error: str | None = None,
    timings: dict[str, float] | None = None,
    candidate_count: int = 4,
    padded_option_slots: int = 8,
    sequence_length: int = 64,
    seed: int = 42,
    noise_hash: str | None = None,
    heap_before: float | None = None,
    heap_after: float | None = None,
    observed_provider: str = "wasm",
) -> dict[str, Any]:
    if backend == "diffusion" and noise_hash is None:
        noise_hash = SHA_NOISE
    ok_timings: dict[str, float] = (
        timings
        if timings is not None
        else {
            "tokenizeMs": 1.0,
            "prepareFeedMs": 0.5,
            "inferenceAndReadbackMs": 10.0,
            "postprocessMs": 0.25,
            "totalRequestMs": 11.75,
        }
    )
    row: dict[str, Any] = {
        "run_id": "run-alpha",
        "session_start": session_start,
        "repeat": repeat,
        "case_id": case_id,
        "backend": backend,
        "requested_provider": "wasm",
        "observed_provider": observed_provider,
        "bundle_hash": SHA_A,
        "tokenizer_hash": SHA_B,
        "input_hash": SHA_C,
        "noise_hash": noise_hash,
        "candidate_count": candidate_count,
        "padded_option_slots": padded_option_slots,
        "sequence_length": sequence_length,
        "seed": seed,
        "ort_version": "1.20.1",
        "status": status,
        "error": error,
        "provenance": _provenance(),
    }
    if status == "ok":
        row.update(ok_timings)
        if heap_before is not None:
            row["js_heap_used_bytes_before"] = heap_before
        if heap_after is not None:
            row["js_heap_used_bytes_after"] = heap_after
    else:
        for field in TIMING_FIELDS:
            row[field] = None
        if timings is not None:
            for field, value in timings.items():
                row[field] = value
        if error is None:
            row["error"] = "simulated failure"
    return row


def _write_jsonl(tmp_path: Path, rows: list[dict[str, Any]], *, name: str = "input.jsonl") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# percentile linear interpolation correctness
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_is_null(self) -> None:
        assert percentile([], 0.5) is None

    def test_single_value(self) -> None:
        assert percentile([7.0], 0.5) == 7.0

    def test_two_values_p50_is_midpoint(self) -> None:
        # index = (2-1)*0.5 = 0.5 -> lower=0, upper=1 -> midpoint
        assert percentile([10.0, 20.0], 0.5) == pytest.approx(15.0)

    def test_two_values_p95_near_upper(self) -> None:
        # index = 1 * 0.95 = 0.95 -> 10 + 10*0.95 = 19.5
        assert percentile([10.0, 20.0], 0.95) == pytest.approx(19.5)

    def test_four_values_p50_exact_interpolation(self) -> None:
        # sorted: [1,2,3,4]; index = (4-1)*0.5 = 1.5
        # lower=1 -> 2, upper=2 -> 3, fraction=0.5 -> 2 + 1*0.5 = 2.5
        assert percentile([3.0, 1.0, 4.0, 2.0], 0.5) == pytest.approx(2.5)

    def test_eight_values_p95(self) -> None:
        values = [float(x) for x in range(1, 9)]
        # sorted: [1..8]; index = (8-1)*0.95 = 7*0.95 = 6.65
        # lower=6 -> 7.0, upper=7 -> 8.0, delta=0.65 -> 7.0 + 1.0*0.65 = 7.65
        assert percentile(values, 0.95) == pytest.approx(7.65)

    def test_fraction_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], 1.5)
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], -0.1)


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------


class TestRowValidation:
    def test_missing_status_rejected(self) -> None:
        row = _row()
        del row["status"]
        with pytest.raises(ValueError, match="missing required field status"):
            _validate_row(row, 1)

    def test_invalid_status_string_rejected(self) -> None:
        row = _row(status="pending")
        with pytest.raises(ValueError, match="status must be ok, error, or timeout"):
            _validate_row(row, 1)

    def test_ok_row_with_error_value_rejected(self) -> None:
        row = _row(status="ok", error="should be empty")
        with pytest.raises(ValueError, match="error must be null for successful"):
            _validate_row(row, 1)

    def test_failed_row_with_partial_timings_allowed(self) -> None:
        row = _row(status="error", timings={"totalRequestMs": 500.0, "tokenizeMs": 10.0})
        assert row["status"] == "error"
        assert row["totalRequestMs"] == 500.0
        assert row["tokenizeMs"] == 10.0
        assert row["prepareFeedMs"] is None
        _validate_row(row, 1)

    def test_successful_row_missing_timing_rejected(self) -> None:
        row = _row()
        del row["inferenceAndReadbackMs"]
        with pytest.raises(ValueError, match="successful sample requires inferenceAndReadbackMs"):
            _validate_row(row, 1)

    def test_nonfinite_timing_rejected(self) -> None:
        row = _row(timings={"tokenizeMs": math.nan, "prepareFeedMs": 0.5,
                            "inferenceAndReadbackMs": 10.0, "postprocessMs": 0.25,
                            "totalRequestMs": 11.75})
        with pytest.raises((ValueError, TypeError), match="finite"):
            _validate_row(row, 1)

    def test_negative_timing_rejected(self) -> None:
        row = _row(timings={"tokenizeMs": -1.0, "prepareFeedMs": 0.5,
                            "inferenceAndReadbackMs": 10.0, "postprocessMs": 0.25,
                            "totalRequestMs": 11.75})
        with pytest.raises(ValueError, match="finite non-negative"):
            _validate_row(row, 1)

    def test_invalid_sha256_rejected(self) -> None:
        row = _row()
        row["bundle_hash"] = "short-and-wrong"
        with pytest.raises(ValueError, match="SHA-256 hex string"):
            _validate_row(row, 1)

    def test_uppercase_sha256_rejected(self) -> None:
        row = _row()
        row["bundle_hash"] = "A" * 64
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            _validate_row(row, 1)

    def test_non_positive_shape_rejected(self) -> None:
        row = _row(candidate_count=0)
        with pytest.raises(ValueError, match="positive integer"):
            _validate_row(row, 1)
        row2 = _row(padded_option_slots=-5)
        with pytest.raises(ValueError, match="positive integer"):
            _validate_row(row2, 2)
        row3 = _row(sequence_length=0)
        with pytest.raises(ValueError, match="positive integer"):
            _validate_row(row3, 3)

    def test_diffusion_noise_required(self) -> None:
        row = _row(backend="diffusion", noise_hash=None)
        row["noise_hash"] = None
        with pytest.raises(ValueError, match="noise_hash is required for diffusion"):
            _validate_row(row, 1)

    def test_repeat_negative_rejected(self) -> None:
        row = _row(repeat=-1)
        with pytest.raises(ValueError, match="repeat must be a non-negative integer"):
            _validate_row(row, 1)

    def test_requested_provider_must_be_wasm_or_webgpu(self) -> None:
        row = _row()
        row["requested_provider"] = "cpu"
        with pytest.raises(ValueError, match="requested_provider must be wasm or webgpu"):
            _validate_row(row, 1)


# ---------------------------------------------------------------------------
# End-to-end aggregation with actual expected percentiles
# ---------------------------------------------------------------------------


def _build_120row_fixture(
    *,
    backend: str = "direct",
    extra_failures: int = 0,
    warmup: int = 10,
) -> list[dict[str, Any]]:
    """10 sessions x 12 repeats = 120 rows with deterministic increasing timings.

    totalRequestMs values for successful rows = 1.0, 2.0, ..., N_success.0.
    p50 (120 successes, all ok):
      index = (120-1)*0.5 = 59.5 -> values[59]=60.0, values[60]=61.0, frac=0.5 -> 60.5
    p95:
      index = 119*0.95 = 113.05 -> values[113]=114.0, values[114]=115.0, frac=0.05
      -> 114.0 + 1.0*0.05 = 114.05
    """
    rows: list[dict[str, Any]] = []
    counter = 0
    for session in range(1, 11):
        for repeat in range(12):
            counter += 1
            if extra_failures > 0 and counter % 15 == 0:
                rows.append(
                    _row(
                        session_start=session,
                        repeat=repeat,
                        case_id=f"case-{counter}",
                        backend=backend,
                        status="error",
                        error="simulated",
                    )
                )
                extra_failures -= 1
            else:
                total = float(counter)
                tokenize = 0.1 * counter
                prepare = 0.05 * counter
                inference = 0.8 * counter
                post = 0.05 * counter
                rows.append(
                    _row(
                        session_start=session,
                        repeat=repeat,
                        case_id=f"case-{counter}",
                        backend=backend,
                        status="ok",
                        timings={
                            "tokenizeMs": tokenize,
                            "prepareFeedMs": prepare,
                            "inferenceAndReadbackMs": inference,
                            "postprocessMs": post,
                            "totalRequestMs": total,
                        },
                        heap_before=1000.0 + counter,
                        heap_after=2000.0 + counter,
                    )
                )
            rows[-1]["provenance"] = _provenance(warmup=warmup)
    return rows


class TestAggregationPercentiles:
    def test_120rows_percentiles_match_exact_expected(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)

        assert summary["sample_counts"]["successful"] == 120
        assert summary["sample_counts"]["failed"] == 0
        assert summary["sample_counts"]["unique_session_starts"] == 10
        assert summary["measurement_completeness"]["complete"] is True

        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        p95 = summary["stage_percentiles_ms"]["totalRequestMs"]["p95"]
        assert p50 == pytest.approx(60.5)
        assert p95 == pytest.approx(114.05)

        # tokenizeMs values = 0.1 * counter for counter=1..120
        # sorted values are 0.1, 0.2, ..., 12.0
        # p50: index=59.5 -> 6.0 + (6.1-6.0)*0.5 = 6.05
        assert summary["stage_percentiles_ms"]["tokenizeMs"]["p50"] == pytest.approx(6.05)
        # p95: index=113.05 -> values[113]=11.4, values[114]=11.5, frac=0.05 -> 11.405
        assert summary["stage_percentiles_ms"]["tokenizeMs"]["p95"] == pytest.approx(11.405)

    def test_failed_rows_counted_timings_nullable(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture(extra_failures=8)
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)

        assert summary["sample_counts"]["successful"] == 112
        assert summary["sample_counts"]["failed"] == 8
        assert summary["measurement_completeness"]["complete"] is True

        # 112 successes: totalRequestMs are the 112 counter values that
        # are not divisible by 15. Counters divisible by 15 (15,30,...,105,120)
        # are failures. Total counters divisible by 15 up to 120: 120//15=8.
        # totalRequestMs values are 1..14,16..29,...,106..119 (8 missing
        # values: 15,30,45,60,75,90,105,120). All increasing order is still
        # preserved because values increase monotonically with counter and
        # only the multiples of 15 are removed.
        #
        # For percentile of 112 values:
        # p50 index = (112-1)*0.5 = 55.5 -> values[55] and values[56]
        # Before each multiple-of-15 threshold, 1 value has been skipped per
        # prior group. Which values are at index 55/56?
        # For each group of 15 counters, 14 survive. Group 1 (1-14) -> indices 0..13 (14).
        # Group 2 (16-29) -> indices 14..27 (14, cumulative 28).
        # Group 3 (31-44) -> indices 28..41 (14, cumulative 42).
        # Group 4 (46-59) -> indices 42..55 (14, cumulative 56).
        # So index 55 = last of group 4 -> counter 59 (value 59.0).
        # Index 56 = first of group 5 -> counter 61 (value 61.0).
        # Interpolation at 55.5 -> 0.5 fraction: 59.0 + 2.0 * 0.5 = 60.0
        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        assert p50 == pytest.approx(60.0)

    def test_heap_maxima_from_observations_only(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        # counter 1..120 -> heap_before = 1000 + counter -> max at 1120
        # counter 1..120 -> heap_after = 2000 + counter -> max at 2120
        assert summary["heap_maxima_bytes"]["js_heap_used_bytes_before"] == pytest.approx(1120.0)
        assert summary["heap_maxima_bytes"]["js_heap_used_bytes_after"] == pytest.approx(2120.0)

    def test_all_null_heap_yields_null_maxima(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        for row in rows:
            if row["status"] == "ok":
                row.pop("js_heap_used_bytes_before", None)
                row.pop("js_heap_used_bytes_after", None)
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["heap_maxima_bytes"]["js_heap_used_bytes_before"] is None
        assert summary["heap_maxima_bytes"]["js_heap_used_bytes_after"] is None


# ---------------------------------------------------------------------------
# Empty input, duplicates, mixed conditions
# ---------------------------------------------------------------------------


class TestValidationFailuresAggregate:
    def test_empty_input_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no samples"):
            validate_and_aggregate(path)

    def test_duplicate_identity_raises(self, tmp_path: Path) -> None:
        rows = [
            _row(session_start=1, repeat=0, case_id="a"),
            _row(session_start=1, repeat=0, case_id="b"),
        ]
        input_path = _write_jsonl(tmp_path, rows)
        with pytest.raises(ValueError, match="duplicate.*identity"):
            validate_and_aggregate(input_path)

    def test_mixed_bundle_hash_raises(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0), _row(session_start=1, repeat=1)]
        rows[1]["bundle_hash"] = SHA_D
        input_path = _write_jsonl(tmp_path, rows)
        with pytest.raises(ValueError, match="mixed experimental conditions"):
            validate_and_aggregate(input_path)

    def test_mixed_backend_raises(self, tmp_path: Path) -> None:
        rows = [
            _row(session_start=1, repeat=0, backend="direct"),
            _row(session_start=1, repeat=1, backend="diffusion", noise_hash=SHA_NOISE),
        ]
        input_path = _write_jsonl(tmp_path, rows)
        with pytest.raises(ValueError, match="mixed experimental conditions"):
            validate_and_aggregate(input_path)

    def test_mixed_environment_raises(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0), _row(session_start=1, repeat=1)]
        rows[1]["provenance"] = _provenance()
        rows[1]["provenance"]["environment"]["platform"] = "Different OS"
        input_path = _write_jsonl(tmp_path, rows)
        with pytest.raises(ValueError, match="mixed experimental conditions"):
            validate_and_aggregate(input_path)

    def test_mixed_warmup_raises(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0), _row(session_start=1, repeat=1)]
        rows[1]["provenance"] = _provenance(warmup=20)
        input_path = _write_jsonl(tmp_path, rows)
        with pytest.raises(ValueError, match="mixed experimental conditions"):
            validate_and_aggregate(input_path)

    def test_diffusion_backend_homogeneous(self, tmp_path: Path) -> None:
        """Diffusion samples must carry noise_hash; varying noise_hash by row is allowed."""
        rows = []
        for session in range(1, 7):
            for repeat in range(20):
                idx = session * 100 + repeat
                rows.append(
                    _row(
                        session_start=session,
                        repeat=repeat,
                        case_id=f"case-{idx}",
                        backend="diffusion",
                        noise_hash=f"{idx:x}".zfill(64),
                    )
                )
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["experimental_condition"]["backend"] == "diffusion"
        assert summary["sample_counts"]["successful"] == 120
        assert summary["sample_counts"]["failed"] == 0


# ---------------------------------------------------------------------------
# Completeness flag and provenance semantics
# ---------------------------------------------------------------------------


class TestCompletenessAndProvenance:
    def test_insufficient_sessions_reported_incomplete(self, tmp_path: Path) -> None:
        rows = []
        for session in range(1, 4):  # only 3 sessions
            for repeat in range(50):
                rows.append(
                    _row(
                        session_start=session,
                        repeat=repeat,
                        case_id=f"case-{session}-{repeat}",
                    )
                )
        # warmup per session = 10, rows = 150 -> two thresholds satisfied,
        # but session count < 5
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        comp = summary["measurement_completeness"]
        assert comp["complete"] is False
        assert comp["actual_session_starts"] == 3
        reasons = " ".join(comp["incomplete_reasons"])
        assert "independent session starts" in reasons

    def test_insufficient_warmup_reported_incomplete(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture(warmup=5)
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        comp = summary["measurement_completeness"]
        assert comp["complete"] is False
        assert comp["actual_warmup_excluded_per_session"] == 5

    def test_120rows_10sessions_is_complete(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["measurement_completeness"]["complete"] is True
        assert summary["measurement_completeness"]["incomplete_reasons"] == []

    def test_page_semantics_not_process_cold_claimable(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0)]
        rows[0]["provenance"]["session_start_semantics"] = "in_page_repeat"
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["process_cold_claimable"] is False

    def test_process_restart_semantics_still_false_v1(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0)]
        rows[0]["provenance"]["session_start_semantics"] = "process_restart"
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["process_cold_claimable"] is False
        assert "always False regardless of semantics string" in summary["process_cold_note"]

    def test_requested_provider_match_still_false_v1(self, tmp_path: Path) -> None:
        rows = [
            _row(session_start=1, repeat=0, observed_provider="wasm"),
            _row(session_start=1, repeat=1, observed_provider="wasm"),
        ]
        rows[0]["provenance"]["execution_provider_evidence"] = "window.ort.env.backend='wasm'"
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["requested_is_proven"] is False
        assert summary["experimental_condition"]["observed_provider"] == "wasm"
        assert summary["experimental_condition"]["provenance"]["execution_provider_evidence"] == "window.ort.env.backend='wasm'"
        assert "requested_is_proven is always False in v1" in summary["provider_claim_note"]

    def test_observed_provider_mismatch_not_proven(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0, observed_provider=None)]
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["requested_is_proven"] is False


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_writes_summary(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        input_path = _write_jsonl(tmp_path, rows)
        output_path = tmp_path / "summary.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SUMMARY_TOOL),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["schema"] == "vons.browser-benchmark-summary/v1"
        assert loaded["sample_counts"]["successful"] == 120
        assert loaded["stage_percentiles_ms"]["totalRequestMs"]["p50"] == pytest.approx(60.5)

    def test_cli_validation_fails_before_writing(self, tmp_path: Path) -> None:
        rows = [
            _row(session_start=1, repeat=0),
            _row(session_start=1, repeat=0, case_id="dup"),
        ]
        input_path = _write_jsonl(tmp_path, rows)
        output_path = tmp_path / "summary.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SUMMARY_TOOL),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode != 0
        assert not output_path.exists() or output_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# Error rows preserve nullable timing slots at summary level (not erased)
# ---------------------------------------------------------------------------


class TestErrorRowPreservation:
    def test_timeout_rows_counted_as_failed_with_nullable_slots(self, tmp_path: Path) -> None:
        rows = [
            _row(session_start=1, repeat=0, status="timeout", error="gpu hang"),
            _row(session_start=1, repeat=1, status="ok"),
        ]
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["sample_counts"]["failed"] == 1
        assert summary["sample_counts"]["successful"] == 1
        # successful timings are only row[1]; all percentiles equal that row's values
        assert summary["stage_percentiles_ms"]["totalRequestMs"]["p50"] == pytest.approx(11.75)


# ---------------------------------------------------------------------------
# Integrator regressions: v1 hard-pessimism, seed 0, partial error timings,
# shape bounds
# ---------------------------------------------------------------------------


class TestIntegratorRegressions:
    def test_unknown_semantics_still_false_process_cold(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0)]
        rows[0]["provenance"]["session_start_semantics"] = "totally_unknown_semantic_XYZ"
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["process_cold_claimable"] is False
        assert summary["experimental_condition"]["provenance"]["session_start_semantics"] == "totally_unknown_semantic_XYZ"
        assert "always False regardless of semantics string" in summary["process_cold_note"]

    def test_matching_provider_unverified_evidence_not_proven(self, tmp_path: Path) -> None:
        rows = [_row(session_start=1, repeat=0, observed_provider="wasm")]
        rows[0]["requested_provider"] = "wasm"
        rows[0]["observed_provider"] = "wasm"
        rows[0]["provenance"]["execution_provider_evidence"] = "raw browser string: wasm observed"
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["requested_is_proven"] is False
        assert summary["experimental_condition"]["requested_provider"] == "wasm"
        assert summary["experimental_condition"]["observed_provider"] == "wasm"
        assert summary["experimental_condition"]["provenance"]["execution_provider_evidence"] == "raw browser string: wasm observed"
        assert "no structured verified execution artifact" in summary["provider_claim_note"]

    def test_seed_zero_is_valid(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        for row in rows:
            row["seed"] = 0
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["sample_counts"]["successful"] == 120
        assert summary["experimental_condition"]["candidate_count"] == 4
        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        assert p50 == pytest.approx(60.5)

    def test_partial_measured_error_row_validated_and_excluded_from_percentiles(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        error_row = _row(
            session_start=11,
            repeat=0,
            case_id="case-extra-error",
            status="error",
            error="inference timeout after prepare",
            timings={
                "tokenizeMs": 12.0,
                "prepareFeedMs": 5.5,
                "totalRequestMs": 500.0,
            },
        )
        rows.append(error_row)
        input_path = _write_jsonl(tmp_path, rows)
        summary = validate_and_aggregate(input_path)
        assert summary["sample_counts"]["successful"] == 120
        assert summary["sample_counts"]["failed"] == 1
        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        p95 = summary["stage_percentiles_ms"]["totalRequestMs"]["p95"]
        assert p50 == pytest.approx(60.5)
        assert p95 == pytest.approx(114.05)

    def test_shape_candidate_exceeds_padded_rejected(self) -> None:
        row = _row(candidate_count=10, padded_option_slots=8)
        with pytest.raises(ValueError, match="candidate_count must be <= padded_option_slots"):
            _validate_row(row, 1)

    def test_shape_padded_exceeds_32_rejected(self) -> None:
        row = _row(candidate_count=16, padded_option_slots=33)
        with pytest.raises(ValueError, match="padded_option_slots must be <= 32"):
            _validate_row(row, 1)

    def test_shape_sequence_exceeds_512_rejected(self) -> None:
        row = _row(sequence_length=513)
        with pytest.raises(ValueError, match="sequence_length must be <= 512"):
            _validate_row(row, 1)

    def test_shape_boundary_values_accepted(self) -> None:
        row = _row(candidate_count=32, padded_option_slots=32, sequence_length=512)
        _validate_row(row, 1)


# ---------------------------------------------------------------------------
# --report-input mode: normalize live_candidates/allocated_candidates/
# sequence_tokens from existing v1 browser-benchmark JSON report, preserve
# top-level provenance, and never invent missing values
# ---------------------------------------------------------------------------


def _build_v1_report_fixture(
    *,
    backend: str = "direct",
    live_field_name: str = "live_candidates",
    allocated_field_name: str = "allocated_candidates",
    seq_field_name: str = "sequence_length",
    top_level_ort: str | None = "1.20.1",
    include_session_start_semantics: bool = True,
    include_environment: bool = True,
    extra_failures: int = 0,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    counter = 0
    for session in range(1, 11):
        for repeat in range(12):
            counter += 1
            if extra_failures > 0 and counter % 15 == 0:
                sample: dict[str, Any] = {
                    "run_id": "run-alpha",
                    "session_start": session,
                    "repeat": repeat,
                    "case_id": f"case-{counter}",
                    "backend": backend,
                    "requested_provider": "wasm",
                    "observed_provider": "wasm",
                    "bundle_hash": SHA_A,
                    "tokenizer_hash": SHA_B,
                    "input_hash": SHA_C,
                    "noise_hash": None,
                    live_field_name: 4,
                    allocated_field_name: 8,
                    seq_field_name: 64,
                    "seed": 42,
                    "ort_version": top_level_ort,
                    "status": "error",
                    "error": "simulated",
                    "tokenizeMs": None,
                    "prepareFeedMs": None,
                    "inferenceAndReadbackMs": None,
                    "postprocessMs": None,
                    "totalRequestMs": None,
                }
                if backend == "diffusion":
                    sample["noise_hash"] = SHA_NOISE
                samples.append(sample)
                extra_failures -= 1
            else:
                total = float(counter)
                tokenize = 0.1 * counter
                prepare = 0.05 * counter
                inference = 0.8 * counter
                post = 0.05 * counter
                sample = {
                    "run_id": "run-alpha",
                    "session_start": session,
                    "repeat": repeat,
                    "case_id": f"case-{counter}",
                    "backend": backend,
                    "requested_provider": "wasm",
                    "observed_provider": "wasm",
                    "bundle_hash": SHA_A,
                    "tokenizer_hash": SHA_B,
                    "input_hash": SHA_C,
                    "noise_hash": None,
                    live_field_name: 4,
                    allocated_field_name: 8,
                    seq_field_name: 64,
                    "seed": 42,
                    "ort_version": top_level_ort,
                    "status": "ok",
                    "error": None,
                    "tokenizeMs": tokenize,
                    "prepareFeedMs": prepare,
                    "inferenceAndReadbackMs": inference,
                    "postprocessMs": post,
                    "totalRequestMs": total,
                    "js_heap_used_bytes_before": 1000.0 + counter,
                    "js_heap_used_bytes_after": 2000.0 + counter,
                }
                if backend == "diffusion":
                    sample["noise_hash"] = SHA_NOISE
                samples.append(sample)
    report: dict[str, Any] = {
        "schema": "vons.browser-benchmark/v1",
        "run_id": "run-alpha",
        "backend": backend,
        "requested_provider": "wasm",
        "warmup_excluded_per_session": 10,
        "samples": samples,
    }
    if include_environment:
        report["environment"] = dict(BASE_ENV)
    report["execution_provider_evidence"] = "window.ort.env.backend='wasm'"
    if include_session_start_semantics:
        report["session_start_semantics"] = "page_reload"
    if top_level_ort is not None:
        report["ort_version"] = top_level_ort
    return report


def _write_report(tmp_path: Path, report: dict[str, Any], *, name: str = "report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


class TestReportInputNormalization:
    def test_report_happy_path_matches_jsonl_percentiles(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture()
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["sample_counts"]["successful"] == 120
        assert summary["sample_counts"]["failed"] == 0
        assert summary["measurement_completeness"]["complete"] is True
        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        p95 = summary["stage_percentiles_ms"]["totalRequestMs"]["p95"]
        assert p50 == pytest.approx(60.5)
        assert p95 == pytest.approx(114.05)
        assert summary["experimental_condition"]["provenance"]["session_start_semantics"] == "page_reload"
        assert summary["process_cold_claimable"] is False
        assert summary["requested_is_proven"] is False

    def test_report_sequence_tokens_fallback_alias(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture(seq_field_name="sequence_tokens")
        for sample in report["samples"]:
            assert "sequence_length" not in sample
            assert sample["sequence_tokens"] == 64
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["experimental_condition"]["sequence_length"] == 64
        assert summary["stage_percentiles_ms"]["totalRequestMs"]["p50"] == pytest.approx(60.5)

    def test_report_missing_live_candidates_rejected(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture()
        for sample in report["samples"]:
            del sample["live_candidates"]
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="missing live_candidates"):
            validate_and_aggregate_report(report_path)

    def test_report_missing_allocated_candidates_rejected(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture()
        for sample in report["samples"]:
            del sample["allocated_candidates"]
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="missing allocated_candidates"):
            validate_and_aggregate_report(report_path)

    def test_report_missing_sequence_both_names_rejected(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture()
        for sample in report["samples"]:
            del sample["sequence_length"]
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="missing sequence_length"):
            validate_and_aggregate_report(report_path)

    def test_report_missing_session_start_semantics_rejected_no_invent(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture(include_session_start_semantics=False)
        assert "session_start_semantics" not in report
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="do not invent missing values"):
            validate_and_aggregate_report(report_path)

    def test_report_missing_ort_version_sample_and_top_rejected(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture(top_level_ort=None)
        for sample in report["samples"]:
            del sample["ort_version"]
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="missing ort_version"):
            validate_and_aggregate_report(report_path)

    def test_report_top_level_ort_used_as_fallback(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture(top_level_ort="1.21.0-custom")
        for sample in report["samples"]:
            del sample["ort_version"]
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["experimental_condition"]["ort_version"] == "1.21.0-custom"
        assert summary["sample_counts"]["successful"] == 120

    def test_report_wrong_schema_rejected(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture()
        report["schema"] = "other.schema/v2"
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="unexpected browser benchmark schema"):
            validate_and_aggregate_report(report_path)


# ---------------------------------------------------------------------------
# CLI --report-input subprocess paths
# ---------------------------------------------------------------------------


class TestCLIReportInput:
    def test_cli_report_input_writes_summary(self, tmp_path: Path) -> None:
        report = _build_v1_report_fixture()
        report_path = _write_report(tmp_path, report)
        output_path = tmp_path / "summary.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SUMMARY_TOOL),
                "--report-input",
                str(report_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded["schema"] == "vons.browser-benchmark-summary/v1"
        assert loaded["sample_counts"]["successful"] == 120
        assert loaded["stage_percentiles_ms"]["totalRequestMs"]["p50"] == pytest.approx(60.5)
        assert loaded["process_cold_claimable"] is False
        assert loaded["requested_is_proven"] is False

    def test_cli_mutually_exclusive_inputs(self, tmp_path: Path) -> None:
        rows = _build_120row_fixture()
        input_path = _write_jsonl(tmp_path, rows)
        report = _build_v1_report_fixture()
        report_path = _write_report(tmp_path, report)
        output_path = tmp_path / "summary.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SUMMARY_TOOL),
                "--input",
                str(input_path),
                "--report-input",
                str(report_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode != 0
        assert "not allowed with" in result.stderr or "argument --report-input: not allowed" in result.stderr


# ---------------------------------------------------------------------------
# Pass 3: Real report compatibility regressions — matches actual report
# shape in reports/browser-direct-{wasm,webgpu-controlled}-merged.json:
# - ort_version via top-level runtime_info.ortVersion (camelCase, nested)
# - seed absent in samples => null allowed for direct backend when
#   noise_hash=null (deterministic-direct not_applicable)
# - observed_provider=null (harness doesn't expose execution partition)
# - heap keys js_heap_used_bytes_before/after canonical
# ---------------------------------------------------------------------------


def _build_real_direct_report(
    *,
    runtime_ort: str = "1.30.0",
    top_ort_version_key_present: bool = False,
    top_ort_value: str | None = None,
    seed_present_in_samples: bool = False,
    noise_hash_value: None | str = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    counter = 0
    for session in range(1, 7):
        for repeat in range(20):
            counter += 1
            total = float(counter)
            sample: dict[str, Any] = {
                "run_id": "run-real-shape",
                "session_start": session,
                "repeat": repeat,
                "case_id": f"case-{counter}",
                "backend": "direct",
                "requested_provider": "wasm",
                "observed_provider": None,
                "observed_provider_status": "session_created; execution_partition_not_exposed",
                "bundle_hash": SHA_A,
                "tokenizer_hash": SHA_B,
                "input_hash": SHA_C,
                "noise_hash": noise_hash_value,
                "live_candidates": 4,
                "allocated_candidates": 8,
                "sequence_length": 64,
                "status": "ok",
                "error": None,
                "tokenizeMs": 0.1 * counter,
                "prepareFeedMs": 0.05 * counter,
                "inferenceAndReadbackMs": 0.8 * counter,
                "postprocessMs": 0.05 * counter,
                "totalRequestMs": total,
                "js_heap_used_bytes_before": 1000.0 + counter,
                "js_heap_used_bytes_after": 2000.0 + counter,
            }
            if seed_present_in_samples:
                sample["seed"] = 42
            samples.append(sample)
    report: dict[str, Any] = {
        "schema": "vons.browser-benchmark/v1",
        "run_id": "run-real-shape",
        "backend": "direct",
        "requested_provider": "wasm",
        "environment": dict(BASE_ENV),
        "execution_provider_evidence": "requested provider and session creation only",
        "session_start_semantics": "one fresh page/session per source report; browser caches may persist",
        "warmup_excluded_per_session": 10,
        "runtime_info": {
            "ortVersion": runtime_ort,
            "provider": "wasm",
            "wasmNumThreads": 1,
            "webgpuAdapterRequested": False,
            "webgpuDeviceRequested": False,
        },
        "samples": samples,
    }
    if top_ort_version_key_present:
        report["ort_version"] = top_ort_value
    return report


class TestRealReportCompatibility:
    def test_runtime_info_ortversion_used_as_top_fallback(self, tmp_path: Path) -> None:
        report = _build_real_direct_report(runtime_ort="1.30.0-fromRuntimeInfo")
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["experimental_condition"]["ort_version"] == "1.30.0-fromRuntimeInfo"
        assert summary["sample_counts"]["successful"] == 120
        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        assert p50 == pytest.approx(60.5)

    def test_top_ort_version_preferred_over_runtime_info(self, tmp_path: Path) -> None:
        report = _build_real_direct_report(
            runtime_ort="1.30.0-runtime-should-not-be-used",
            top_ort_version_key_present=True,
            top_ort_value="1.31.0-topPreferred",
        )
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["experimental_condition"]["ort_version"] == "1.31.0-topPreferred"

    def test_missing_both_ort_raises_mentions_runtimeinfo(self, tmp_path: Path) -> None:
        report = _build_real_direct_report(runtime_ort="1.30.0")
        del report["runtime_info"]
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="runtime_info.ortVersion"):
            validate_and_aggregate_report(report_path)

    def test_direct_samples_no_seed_noisehash_null_resolves_seed_null(self, tmp_path: Path) -> None:
        report = _build_real_direct_report(seed_present_in_samples=False, noise_hash_value=None)
        for sample in report["samples"]:
            assert "seed" not in sample
            assert sample["noise_hash"] is None
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["experimental_condition"]["seed"] is None
        assert summary["sample_counts"]["successful"] == 120
        p50 = summary["stage_percentiles_ms"]["totalRequestMs"]["p50"]
        assert p50 == pytest.approx(60.5)

    def test_direct_with_noise_hash_still_requires_seed_not_null(self, tmp_path: Path) -> None:
        report = _build_real_direct_report(seed_present_in_samples=False, noise_hash_value=SHA_NOISE)
        for sample in report["samples"]:
            assert "seed" not in sample
            assert sample["noise_hash"] == SHA_NOISE
        report_path = _write_report(tmp_path, report)
        with pytest.raises(ValueError, match="missing seed"):
            validate_and_aggregate_report(report_path)

    def test_diffusion_samples_require_seed_never_null(self) -> None:
        row = _row(backend="diffusion", noise_hash=SHA_NOISE)
        row["seed"] = None
        with pytest.raises(ValueError, match="seed may only be null for direct backend"):
            _validate_row(row, 1)

    def test_direct_with_seed_integer_still_accepted(self) -> None:
        row = _row(seed=17)
        row["noise_hash"] = None
        row["seed"] = 17
        _validate_row(row, 1)

    def test_direct_seed_null_noise_hash_null_accepted_at_row_level(self) -> None:
        row = _row(seed=42)
        row["seed"] = None
        row["noise_hash"] = None
        _validate_row(row, 1)

    def test_observed_provider_null_preserved_in_summary(self, tmp_path: Path) -> None:
        report = _build_real_direct_report()
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["experimental_condition"]["observed_provider"] is None
        assert summary["requested_is_proven"] is False

    def test_heap_keys_canonical_and_values_match(self, tmp_path: Path) -> None:
        report = _build_real_direct_report()
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        keys = sorted(summary["heap_maxima_bytes"].keys())
        assert keys == ["js_heap_used_bytes_after", "js_heap_used_bytes_before"]
        # counter = 1..120 => heap_before = 1000 + counter => max=1120
        assert summary["heap_maxima_bytes"]["js_heap_used_bytes_before"] == pytest.approx(1120.0)
        assert summary["heap_maxima_bytes"]["js_heap_used_bytes_after"] == pytest.approx(2120.0)

    def test_completeness_passes_with_real_reports_120(self, tmp_path: Path) -> None:
        report = _build_real_direct_report()
        report_path = _write_report(tmp_path, report)
        summary = validate_and_aggregate_report(report_path)
        assert summary["measurement_completeness"]["complete"] is True
        assert summary["measurement_completeness"]["incomplete_reasons"] == []
        assert summary["sample_counts"]["unique_session_starts"] == 6
