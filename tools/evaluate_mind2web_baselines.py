"""Retrospective candidate-selection baselines for frozen Mind2Web v2 reports.

Compares the frozen selector against two purely post-hoc baselines:
(a) exact uniform-random expected selection accuracy over each returned
    candidate set of size c, with p embedded positives: expected = p / c;
(b) first-listed candidate selection: 1 if the first returned candidate is
    a positive on a recalled row, else 0.

Bootstrap intervals are deterministic, paired over task clusters, stratified
by official split and k. Every attempt remains accounted for across
positive/no-positive rows, accepted/error/abstain/missing execution status,
recall coverage, the selector's actual choice, overflow counts, invalid
candidate references, and absent selections on recalled rows.

Outputs contain aggregate counts, derived metrics, and SHA-256 digests of
all source inputs only. Row IDs, task IDs, goals, candidate text, DOM
content, and any raw data payloads are deliberately excluded from all
reports, logs, and intermediate structures beyond the per-row numeric
accumulators required to derive the aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SPLITS = ("test_task", "test_website", "test_domain")
KS = (5, 10, 20, 32)
CELLS = tuple((split, k) for split in SPLITS for k in KS)
SOURCE_FILES = ("tools/evaluate_mind2web_baselines.py",)
SCOPE_LABEL = "post_hoc_retrospective"
SCOPE_DESCRIPTION = (
    "Candidate-selection analysis only. This retrospective compares a frozen "
    "selector against two analytical baselines computed from the same returned "
    "candidate sets. It is not a confirmatory holdout, a measure of "
    "generalization, or a browser task success evaluation."
)
REPORT_SCHEMA = "vons.mind2web-baselines/v1"
SUMMARY_SCHEMA = "vons.mind2web-baselines-summary/v1"
FROZEN_RUN_SUMMARY_SCHEMA = "vons.mind2web-run-summary/v1"
RUN_INDEX_SCHEMA = "vons.mind2web-prediction-index/v1"
NORMALIZED_STATUS_KEYS = frozenset({"ok", "error", "abstain"})
FROZEN_RUN_REQUIRED_CELL_KEYS = frozenset(
    {
        "split",
        "k",
        "prediction_rows",
        "predictions_sha256",
        "prediction_manifest_sha256",
        "config_sha256",
        "counts",
    }
)
FROZEN_SUMMARY_DENOMINATOR_KEYS = frozenset(
    {
        "prediction_rows",
        "rows_total",
        "positive_rows",
        "recalled_positive_rows",
    }
)


def _nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _load(path: Path) -> dict[str, Any]:
    return _object(
        json.loads(path.read_text(encoding="utf-8"), parse_constant=_nonfinite),
        "JSON document",
    )


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _object(json.loads(line, parse_constant=_nonfinite), "JSONL row")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _normalize_status_counts(
    counts_raw: object,
    label: str,
    *,
    total_rows: int,
    zero_fill_absent: bool = False,
) -> dict[str, int]:
    if not isinstance(counts_raw, dict):
        raise TypeError(f"{label} must be a JSON object of status counts")
    counts: dict[str, int] = {}
    for status, count in counts_raw.items():
        if not isinstance(status, str):
            raise TypeError(f"{label} status keys must be strings")
        counts[str(status)] = _nonneg_int(count, f"{label} count {status!r}")
    for extra in counts:
        if extra not in NORMALIZED_STATUS_KEYS:
            raise ValueError(f"{label} contains unknown status class {extra!r}")
    for required in NORMALIZED_STATUS_KEYS:
        if required not in counts:
            if zero_fill_absent:
                counts[required] = 0
            else:
                raise ValueError(f"{label} omits required status class {required!r}")
    declared_total = sum(counts.values())
    if declared_total != total_rows:
        raise ValueError(f"{label} status counts sum to {declared_total}, expected {total_rows}")
    return counts


def _collect_frozen_denominator_sources(
    run_obj: dict[str, Any],
) -> dict[str, list[tuple[str, Any]]]:
    by_canonical: dict[str, list[tuple[str, Any]]] = {}
    for top_key, top_val in run_obj.items():
        if top_key in FROZEN_SUMMARY_DENOMINATOR_KEYS:
            by_canonical.setdefault(top_key, []).append((f"top-level {top_key!r}", top_val))
    nested = run_obj.get("metrics")
    if isinstance(nested, dict):
        for nested_key, nested_val in nested.items():
            if nested_key == "rows":
                canonical_key = "prediction_rows"
                source_label = "nested metrics.rows (alias)"
            elif nested_key in FROZEN_SUMMARY_DENOMINATOR_KEYS:
                canonical_key = nested_key
                source_label = f"nested metrics.{nested_key!r}"
            else:
                continue
            by_canonical.setdefault(canonical_key, []).append((source_label, nested_val))
    return by_canonical


def _frozen_metrics(run_obj: dict[str, Any]) -> dict[str, Any]:
    sources = _collect_frozen_denominator_sources(run_obj)
    merged: dict[str, Any] = {}
    for canonical_key, entries in sources.items():
        values = [entry[1] for entry in entries]
        first_value = values[0]
        for idx, entry in enumerate(entries[1:], start=1):
            if entry[1] != first_value:
                labels = " vs ".join(label for label, _ in entries)
                values_text = " vs ".join(repr(val) for val in values)
                raise ValueError(
                    f"frozen summary {canonical_key!r} has conflicting declared "
                    f"values across sources: {labels} have values {values_text}"
                )
        merged[canonical_key] = first_value
    return merged


def _coerce_denominator_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a nonnegative integer")
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, float) and value.is_integer():
        candidate = int(value)
    else:
        raise TypeError(f"{label} must be a nonnegative integer")
    if candidate < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return candidate


def _nonneg_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {name: root / name for name in SOURCE_FILES}


class CellAccumulator:
    """Numeric-only accumulator for one split/k cell.

    Stores only integer counts and float aggregators keyed by opaque integer
    row indices. No identifiers or raw content leaves instances of this
    class; the row-level arrays are discarded after reduce().

    Primary selection denominator convention (matches vons/mind2web.py):
      - selection_accuracy_given_recall uses recalled_rows as the denominator.
      - Missing or invalid selections on recalled rows are errors in the
        primary denominator (they are not silently excluded, nor imputed,
        and they remain excluded from the secondary complete-case metric).
      - Random and first-position baselines share the same recalled_rows
        denominator so selector, random, and first metrics are directly
        comparable on an identical recalled-row population.
    """

    def __init__(self, split: str, k: int) -> None:
        self.split = split
        self.k = k
        self.rows_total = 0
        self.positive_rows = 0
        self.no_positive_rows = 0
        self.prediction_records = 0
        self.status_ok = 0
        self.status_error = 0
        self.status_abstain = 0
        self.status_missing = 0
        self.recalled_rows = 0
        self.selector_correct = 0
        self.selector_evaluated_recalled = 0
        self.selector_missing_selection = 0
        self.selector_invalid_selection = 0
        self.overflow_rows = 0
        self.overflow_reason_counts: dict[str, int] = {
            "input_overflow": 0,
            "candidate_input_overflow": 0,
            "returned_count_exceeds_k": 0,
        }
        self.random_expected_correct = 0.0
        self.first_correct = 0
        self._task_groups: dict[str, dict[str, Any]] = {}

    def feed_row(
        self,
        *,
        row_index: int,
        task_key: str,
        no_positive: bool,
        positive_ids_in_candidates: int,
        candidate_count: int,
        has_prediction_record: bool,
        execution_status: str,
        recalled: bool,
        selector_present: bool,
        selector_in_candidates: bool,
        selector_correct: bool,
        first_candidate_correct: bool,
        overflow: bool,
        overflow_reason: str | None = None,
    ) -> None:
        del row_index
        self.rows_total += 1
        if no_positive:
            self.no_positive_rows += 1
        else:
            self.positive_rows += 1
        if has_prediction_record:
            self.prediction_records += 1
        if execution_status == "ok":
            self.status_ok += 1
        elif execution_status == "error":
            self.status_error += 1
        elif execution_status == "abstain":
            self.status_abstain += 1
        elif execution_status == "missing":
            self.status_missing += 1
        if overflow:
            self.overflow_rows += 1
            if overflow_reason in self.overflow_reason_counts:
                self.overflow_reason_counts[overflow_reason] += 1
            else:
                self.overflow_reason_counts["returned_count_exceeds_k"] += 1
        if recalled:
            self.recalled_rows += 1
            if not selector_present:
                self.selector_missing_selection += 1
            elif not selector_in_candidates:
                self.selector_invalid_selection += 1
            else:
                self.selector_evaluated_recalled += 1
                if selector_correct:
                    self.selector_correct += 1
            if candidate_count > 0 and positive_ids_in_candidates > 0:
                self.random_expected_correct += positive_ids_in_candidates / candidate_count
            if first_candidate_correct:
                self.first_correct += 1
        group = self._task_groups.setdefault(
            task_key,
            {
                "positive_rows": 0,
                "recalled_rows": 0,
                "selector_correct": 0,
                "selector_evaluated_recalled": 0,
                "random_expected_correct": 0.0,
                "first_correct": 0,
            },
        )
        if not no_positive:
            group["positive_rows"] += 1
        if recalled:
            group["recalled_rows"] += 1
            if selector_present and selector_in_candidates and selector_correct:
                group["selector_correct"] += 1
            if selector_present and selector_in_candidates:
                group["selector_evaluated_recalled"] += 1
            if candidate_count > 0 and positive_ids_in_candidates > 0:
                group["random_expected_correct"] += positive_ids_in_candidates / candidate_count
            if first_candidate_correct:
                group["first_correct"] += 1

    def task_values(self) -> dict[str, list[float]]:
        recall: list[float] = []
        selector: list[float] = []
        selector_complete: list[float] = []
        random_baseline: list[float] = []
        first_baseline: list[float] = []
        delta_random: list[float] = []
        delta_first: list[float] = []
        for group in self._task_groups.values():
            if group["positive_rows"] > 0:
                recall.append(group["recalled_rows"] / group["positive_rows"])
            if group["recalled_rows"] > 0:
                selector_val = group["selector_correct"] / group["recalled_rows"]
                random_val = group["random_expected_correct"] / group["recalled_rows"]
                first_val = group["first_correct"] / group["recalled_rows"]
                selector.append(selector_val)
                random_baseline.append(random_val)
                first_baseline.append(first_val)
                delta_random.append(selector_val - random_val)
                delta_first.append(selector_val - first_val)
                if group["selector_evaluated_recalled"] > 0:
                    selector_complete.append(
                        group["selector_correct"] / group["selector_evaluated_recalled"]
                    )
        return {
            "recall": recall,
            "selector": selector,
            "selector_complete_case": selector_complete,
            "random": random_baseline,
            "first": first_baseline,
            "delta_random": delta_random,
            "delta_first": delta_first,
        }

    def reduce(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "rows_total": self.rows_total,
            "positive_rows": self.positive_rows,
            "no_positive_rows": self.no_positive_rows,
            "prediction_records": self.prediction_records,
            "execution_status_counts": {
                "accepted": self.status_ok,
                "error": self.status_error,
                "abstain": self.status_abstain,
                "missing_record": self.status_missing,
            },
            "recalled_rows": self.recalled_rows,
            "overflow_rows": self.overflow_rows,
            "overflow_reason_counts": dict(self.overflow_reason_counts),
            "selector": {
                "correct": self.selector_correct,
                "evaluated_recalled": self.selector_evaluated_recalled,
                "missing_selection_on_recalled": self.selector_missing_selection,
                "invalid_selection_on_recalled": self.selector_invalid_selection,
            },
        }
        if self.positive_rows > 0:
            metrics["candidate_recall_micro"] = self.recalled_rows / self.positive_rows
        else:
            metrics["candidate_recall_micro"] = None
        if self.recalled_rows > 0:
            metrics["selector_accuracy_given_recall_micro"] = (
                self.selector_correct / self.recalled_rows
            )
            metrics["random_expected_accuracy_micro"] = (
                self.random_expected_correct / self.recalled_rows
            )
            metrics["first_position_accuracy_micro"] = self.first_correct / self.recalled_rows
            metrics["selector_minus_random_micro"] = (
                metrics["selector_accuracy_given_recall_micro"]
                - metrics["random_expected_accuracy_micro"]
            )
            metrics["selector_minus_first_micro"] = (
                metrics["selector_accuracy_given_recall_micro"]
                - metrics["first_position_accuracy_micro"]
            )
        else:
            metrics["selector_accuracy_given_recall_micro"] = None
            metrics["random_expected_accuracy_micro"] = None
            metrics["first_position_accuracy_micro"] = None
            metrics["selector_minus_random_micro"] = None
            metrics["selector_minus_first_micro"] = None
        if self.selector_evaluated_recalled > 0:
            metrics["selector_complete_case_accuracy_micro"] = (
                self.selector_correct / self.selector_evaluated_recalled
            )
        else:
            metrics["selector_complete_case_accuracy_micro"] = None
        values = self.task_values()

        def mean(values_list: list[float]) -> float | None:
            return sum(values_list) / len(values_list) if values_list else None

        metrics["candidate_recall_task_macro"] = mean(values["recall"])
        metrics["selector_accuracy_given_recall_task_macro"] = mean(values["selector"])
        metrics["selector_complete_case_accuracy_task_macro"] = mean(
            values["selector_complete_case"]
        )
        metrics["random_expected_accuracy_task_macro"] = mean(values["random"])
        metrics["first_position_accuracy_task_macro"] = mean(values["first"])
        metrics["selector_minus_random_task_macro"] = mean(values["delta_random"])
        metrics["selector_minus_first_task_macro"] = mean(values["delta_first"])
        metrics["task_group_count"] = len(self._task_groups)
        metrics["primary_denominator_convention"] = (
            "selection_accuracy_given_recall uses recalled_rows; "
            "missing/invalid selections on recalled rows are errors "
            "(zero-contributing numerator, retained denominator). "
            "Random and first baselines share the recalled_rows denominator."
        )
        return metrics


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _paired_bootstrap_interval(
    values_x: Sequence[float],
    values_y: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> list[float] | None:
    if len(values_x) != len(values_y) or not values_x:
        return None
    n = len(values_x)
    rng = random.Random(seed)
    deltas = [values_x[i] - values_y[i] for i in range(n)]
    samples: list[float] = []
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(deltas[i] for i in indices) / n)
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _marginal_bootstrap_interval(
    values: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    n = len(values)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(values[i] for i in indices) / n)
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _validate_and_index_gold(
    rows_path: Path,
) -> tuple[dict[str, Any], dict[str, tuple[bool, tuple[str, ...], tuple[str, ...], str]]]:
    """Return (metadata, private structural index).

    The private index is keyed by opaque example_id and stores only
    (no_positive flag, positive_ids tuple, candidate_ids universe tuple,
    task_id string). Identifiers are retained here only because the
    prediction file keys against them; they never appear in reports.

    Gold universe normalization contract:
      * ``candidate_ids`` (``candidates`` fallback) may be empty (an
        implicit "no universe declared" form).  When non-empty, entries
        must be unique non-empty strings.
      * ``positive_ids`` (``target_ids`` / ``targets`` / legacy
        ``target_id`` fallback) must be unique non-empty strings when
        present.
      * ``positive_ids ⊆ candidate_ids`` is enforced **only** when both
        collections are non-empty.  An empty universe never triggers
        the subset check, and rows flagged ``no_positive`` are
        required to carry no positive_ids.
      * Rows that represent positive actions (``no_positive`` False)
        must declare at least one ``positive_id``; this preserves the
        positivity semantics the evaluator needs for recall.
    """
    index: dict[str, tuple[bool, tuple[str, ...], tuple[str, ...], str]] = {}
    counts_by_split: dict[str, int] = {}
    for raw in _jsonl(rows_path):
        example_id = raw.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("gold row requires a string id")
        if example_id in index:
            raise ValueError("duplicate gold row id")
        split = raw.get("split")
        if not isinstance(split, str):
            raise TypeError("gold row requires split")
        if split not in SPLITS:
            raise ValueError(f"gold row split {split!r} is not an official split")
        no_positive = raw.get("no_positive", False)
        if not isinstance(no_positive, bool):
            raise TypeError("gold no_positive must be boolean")
        candidates_raw = raw.get("candidate_ids", raw.get("candidates"))
        if candidates_raw is None:
            candidates: list[str] = []
        else:
            if not isinstance(candidates_raw, list):
                raise TypeError("gold candidate_ids must be a list, when present")
            if any(not isinstance(item, str) or not item for item in candidates_raw):
                raise TypeError("gold candidate_ids entries must be nonempty strings, when present")
            candidates = [str(item) for item in candidates_raw]
        candidate_ids = tuple(candidates)
        if candidate_ids and len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("gold candidate_ids must be unique")
        positive_raw = raw.get("positive_ids", raw.get("target_ids", raw.get("targets")))
        legacy_target = raw.get("target_id")
        if positive_raw is None and legacy_target not in (None, ""):
            positive_ids = (str(legacy_target),)
        elif positive_raw is None:
            positive_ids = ()
        else:
            if not isinstance(positive_raw, list) or any(
                not isinstance(item, str) or not item for item in positive_raw
            ):
                raise TypeError("gold positive_ids must be a list of nonempty strings")
            positive_ids = tuple(str(item) for item in positive_raw)
        if positive_ids and len(positive_ids) != len(set(positive_ids)):
            raise ValueError("gold positive_ids must be unique")
        if candidate_ids and positive_ids and not set(positive_ids).issubset(set(candidate_ids)):
            raise ValueError(
                "gold positive_ids must lie within the gold candidate_ids universe "
                "when both collections are non-empty"
            )
        if no_positive and positive_ids:
            raise ValueError("no_positive gold rows cannot have positive_ids")
        if not no_positive and not positive_ids:
            raise ValueError("positive gold rows require positive_ids or a target")
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            task_id = example_id
        counts_by_split[split] = counts_by_split.get(split, 0) + 1
        index[example_id] = (
            no_positive,
            positive_ids,
            candidate_ids,
            str(task_id),
        )
    metadata = {"counts_by_split": counts_by_split, "rows_total": len(index)}
    return metadata, index


def _iterate_prediction_cells(
    predictions_dir: Path,
    cells: Sequence[tuple[str, int]],
) -> dict[tuple[str, int], tuple[Path, Path]]:
    found: dict[tuple[str, int], tuple[Path, Path]] = {}
    for split, k in cells:
        prediction_path = predictions_dir / f"{split}-k{k}.jsonl"
        manifest_path = prediction_path.with_suffix(".manifest.json")
        if not prediction_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"missing prediction or manifest for {split}/k{k}")
        found[(split, k)] = (prediction_path, manifest_path)
    return found


def _validate_prediction_row(
    row: dict[str, Any],
    *,
    split: str,
    k: int,
    gold_index: Mapping[str, tuple[bool, tuple[str, ...], tuple[str, ...], str]],
    config_hash: str,
) -> dict[str, Any]:
    """Return a structural-only digest of the prediction row.

    The returned dict contains counts, booleans, and opaque id-comparison
    results only; identifiers are dropped as soon as the comparisons have
    been performed.

    Overflow is sourced from the prediction's explicit status+reason
    (``error`` with ``reason in {input_overflow, candidate_input_overflow}``)
    with a belt-and-suspenders fallback when ``len(generated_candidates)``
    exceeds k so that historical overflows remain counted.
    """
    OVERFLOW_REASONS = frozenset({"input_overflow", "candidate_input_overflow"})
    example_id = row.get("id")
    if not isinstance(example_id, str) or example_id not in gold_index:
        raise ValueError("prediction row id unknown or duplicated")
    if row.get("split") != split:
        raise ValueError("prediction split disagrees with cell")
    row_k = _nonneg_int(row.get("k"), "prediction k")
    if row_k != k:
        raise ValueError("prediction k disagrees with cell")
    task_id_pred = row.get("task_id")
    no_positive, positive_ids, gold_candidate_universe, task_id_gold = gold_index[example_id]
    if isinstance(task_id_pred, str) and task_id_pred != task_id_gold:
        raise ValueError("prediction task_id disagrees with gold task_id")
    if row.get("config_sha256") != config_hash:
        raise ValueError("prediction row config hash disagrees with manifest")
    for key in ("request_sha256", "retrieval_input_sha256"):
        _hex64(row.get(key), key)
    candidates_raw = row.get("generated_candidates")
    if (
        not isinstance(candidates_raw, list)
        or any(not isinstance(item, str) or not item for item in candidates_raw)
        or len(candidates_raw) != len(set(candidates_raw))
    ):
        raise ValueError("prediction generated_candidates must be unique nonempty strings")
    full_candidates = tuple(str(item) for item in candidates_raw)
    universe = set(gold_candidate_universe)
    if universe:
        for candidate in full_candidates:
            if candidate not in universe:
                raise ValueError(
                    "generated_candidates contains ids outside gold candidate universe"
                )
    length_overflow = len(full_candidates) > k
    candidate_ids = full_candidates[:k]
    if length_overflow:
        del full_candidates
    status = row.get("status")
    if status not in ("ok", "error", "abstain"):
        raise ValueError("prediction status must be ok/error/abstain")
    reason = row.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise TypeError("prediction reason must be a string or null")
    reason_overflow = status == "error" and reason in OVERFLOW_REASONS
    overflow = reason_overflow or length_overflow
    if reason_overflow:
        overflow_reason = str(reason)
    elif length_overflow:
        overflow_reason = "returned_count_exceeds_k"
    else:
        overflow_reason = None
    if status == "ok" and (not isinstance(row.get("selection"), str) or not row["selection"]):
        raise ValueError("ok status requires a nonempty string selection")
    if status != "ok" and row.get("selection") is not None:
        raise ValueError("non-ok status must not carry a selection")
    selection = row.get("selection")
    positive_in_candidates = 0
    first_correct = False
    for idx, candidate in enumerate(candidate_ids):
        if not no_positive and candidate in positive_ids:
            positive_in_candidates += 1
            if idx == 0:
                first_correct = True
    recalled = positive_in_candidates > 0
    selector_present = isinstance(selection, str) and bool(selection)
    selector_in_candidates = selector_present and selection in candidate_ids
    selector_correct = selector_in_candidates and not no_positive and selection in positive_ids
    return {
        "example_id": example_id,
        "task_key": task_id_gold,
        "no_positive": no_positive,
        "positive_in_candidates": positive_in_candidates,
        "candidate_count": len(candidate_ids),
        "has_record": True,
        "execution_status": status,
        "recalled": recalled,
        "selector_present": selector_present,
        "selector_in_candidates": selector_in_candidates,
        "selector_correct": selector_correct,
        "first_correct": first_correct,
        "overflow": overflow,
        "overflow_reason": overflow_reason,
    }


def evaluate_cell(
    *,
    split: str,
    k: int,
    gold_index: Mapping[str, tuple[bool, tuple[str, ...], tuple[str, ...], str]],
    split_gold_ids: set[str],
    prediction_path: Path,
    manifest_path: Path,
    rows_digest: str,
    run_index_digest: str,
    frozen_summary_digest: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], set[str]]:
    manifest = _load(manifest_path)
    if manifest.get("schema") != "vons.mind2web-prediction/v1":
        raise ValueError("unsupported prediction manifest schema")
    config = _object(manifest.get("config"), "manifest config")
    if config.get("split") != split or _nonneg_int(config.get("k"), "manifest k") != k:
        raise ValueError("manifest config split/k mismatch")
    config_hash = _object_hash(config)
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("manifest config hash mismatch")
    manifest_prediction_hash = _hex64(
        manifest.get("predictions_sha256"), "manifest predictions_sha256"
    )
    actual_prediction_hash = _file_hash(prediction_path)
    if manifest_prediction_hash != actual_prediction_hash:
        raise ValueError("manifest prediction file hash mismatch")
    _nonneg_int(manifest.get("rows"), "manifest rows")
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, dict):
        raise TypeError("manifest counts must be a dict of status counts")
    accumulator = CellAccumulator(split, k)
    seen_ids: set[str] = set()
    row_counter = 0
    actual_counts: dict[str, int] = {}
    for raw in _jsonl(prediction_path):
        structural = _validate_prediction_row(
            raw, split=split, k=k, gold_index=gold_index, config_hash=config_hash
        )
        execution_status = structural["execution_status"]
        actual_counts[execution_status] = actual_counts.get(execution_status, 0) + 1
        example_id = structural["example_id"]
        if example_id in seen_ids:
            raise ValueError("duplicate prediction row id")
        if example_id not in split_gold_ids:
            raise ValueError("prediction row id does not belong to this split")
        seen_ids.add(example_id)
        accumulator.feed_row(
            row_index=row_counter,
            task_key=structural["task_key"],
            no_positive=structural["no_positive"],
            positive_ids_in_candidates=structural["positive_in_candidates"],
            candidate_count=structural["candidate_count"],
            has_prediction_record=structural["has_record"],
            execution_status=execution_status,
            recalled=structural["recalled"],
            selector_present=structural["selector_present"],
            selector_in_candidates=structural["selector_in_candidates"],
            selector_correct=structural["selector_correct"],
            first_candidate_correct=structural["first_correct"],
            overflow=structural["overflow"],
            overflow_reason=structural.get("overflow_reason"),
        )
        row_counter += 1
    missing_ids = split_gold_ids - seen_ids
    for row_index, example_id in enumerate(sorted(missing_ids), start=row_counter):
        no_positive, _positive_ids, _cand_universe, task_key = gold_index[example_id]
        accumulator.feed_row(
            row_index=row_index,
            task_key=task_key,
            no_positive=no_positive,
            positive_ids_in_candidates=0,
            candidate_count=0,
            has_prediction_record=False,
            execution_status="missing",
            recalled=False,
            selector_present=False,
            selector_in_candidates=False,
            selector_correct=False,
            first_candidate_correct=False,
            overflow=False,
            overflow_reason=None,
        )
    if manifest["rows"] != len(seen_ids):
        raise ValueError("manifest rows disagrees with actual prediction row count")
    for status_name, count in manifest_counts.items():
        if actual_counts.get(status_name, 0) != _nonneg_int(count, f"manifest count {status_name}"):
            raise ValueError(f"manifest status count mismatch for {status_name!r}")
    reduced = accumulator.reduce()
    values = accumulator.task_values()
    del accumulator
    bootstrap: dict[str, Any] = {
        "draws": bootstrap_draws,
        "seed": bootstrap_seed,
        "unit": "task_id_or_example_id",
        "task_group_count": reduced["task_group_count"],
        "convention": (
            "selector_accuracy_given_recall over recalled_rows primary population; "
            "paired deltas: selector_accuracy_given_recall minus each baseline over "
            "identical task-macro arrays from the same task groups."
        ),
    }
    bootstrap["candidate_recall_task_macro_ci95"] = _marginal_bootstrap_interval(
        values["recall"], draws=bootstrap_draws, seed=bootstrap_seed
    )
    bootstrap["selector_accuracy_given_recall_task_macro_ci95"] = _marginal_bootstrap_interval(
        values["selector"], draws=bootstrap_draws, seed=bootstrap_seed
    )
    bootstrap["selector_complete_case_accuracy_task_macro_ci95"] = _marginal_bootstrap_interval(
        values["selector_complete_case"], draws=bootstrap_draws, seed=bootstrap_seed
    )
    bootstrap["random_expected_task_macro_ci95"] = _marginal_bootstrap_interval(
        values["random"], draws=bootstrap_draws, seed=bootstrap_seed
    )
    bootstrap["first_position_task_macro_ci95"] = _marginal_bootstrap_interval(
        values["first"], draws=bootstrap_draws, seed=bootstrap_seed
    )
    if values["selector"]:
        bootstrap["selector_minus_random_paired_ci95"] = _paired_bootstrap_interval(
            values["selector"],
            values["random"],
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        bootstrap["selector_minus_first_paired_ci95"] = _paired_bootstrap_interval(
            values["selector"],
            values["first"],
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
    else:
        bootstrap["selector_minus_random_paired_ci95"] = None
        bootstrap["selector_minus_first_paired_ci95"] = None
    cell_report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "scope_label": SCOPE_LABEL,
        "scope": SCOPE_DESCRIPTION,
        "split": split,
        "k": k,
        "input_sha256": {
            "rows": rows_digest,
            "run_index": run_index_digest,
            "frozen_summary": frozen_summary_digest,
            "predictions": actual_prediction_hash,
            "prediction_manifest": _file_hash(manifest_path),
            "manifest_config": config_hash,
        },
        "metrics": reduced,
        "bootstrap": bootstrap,
    }
    return cell_report, seen_ids


def evaluate_baselines(
    rows_path: Path,
    run_index_path: Path,
    predictions_dir: Path,
    output_dir: Path,
    frozen_summary_path: Path,
    *,
    bootstrap_draws: int = 1000,
    bootstrap_seed: int = 7,
    allow_subset: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("refusing to overwrite baseline evidence")
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    sources = _source_paths()
    source_hashes = {name: _file_hash(path) for name, path in sources.items()}
    code_snapshot = {path: source_hashes[name] for name, path in sources.items()}
    rows_digest = _file_hash(rows_path)
    run_index_digest = _file_hash(run_index_path)
    frozen_summary_digest = _file_hash(frozen_summary_path)
    snapshot: dict[Path, str] = {
        **code_snapshot,
        rows_path: rows_digest,
        run_index_path: run_index_digest,
        frozen_summary_path: frozen_summary_digest,
    }
    run_index = _load(run_index_path)
    run_index_schema = run_index.get("schema")
    if run_index_schema != RUN_INDEX_SCHEMA:
        raise ValueError(
            f"run index schema must be {RUN_INDEX_SCHEMA!r} (got {run_index_schema!r})"
        )
    _hex64(run_index.get("source_sha256"), "run index source_sha256")
    _hex64(run_index.get("runner_sha256"), "run index runner_sha256")
    frozen_summary = _load(frozen_summary_path)
    if frozen_summary.get("schema") != FROZEN_RUN_SUMMARY_SCHEMA:
        raise ValueError(f"frozen summary schema must be {FROZEN_RUN_SUMMARY_SCHEMA!r}")
    frozen_input_sha = _object(frozen_summary.get("input_sha256"), "frozen summary input_sha256")
    _hex64(frozen_input_sha.get("rows"), "frozen summary input_sha256.rows")
    _hex64(frozen_input_sha.get("run_index"), "frozen summary input_sha256.run_index")
    _hex64(
        frozen_input_sha.get("retrieval_source_recorded"),
        "frozen summary input_sha256.retrieval_source_recorded",
    )
    if frozen_input_sha["rows"] != rows_digest:
        raise ValueError("frozen summary rows input_sha256 disagrees with current rows file")
    if frozen_input_sha["run_index"] != run_index_digest:
        raise ValueError(
            "frozen summary run_index input_sha256 disagrees with current run index file"
        )
    if frozen_input_sha["retrieval_source_recorded"] != run_index["source_sha256"]:
        raise ValueError(
            "frozen summary retrieval_source_recorded disagrees with run index source_sha256"
        )
    frozen_runner = frozen_summary.get("prediction_runner_sha256")
    _hex64(frozen_runner, "frozen summary prediction_runner_sha256")
    if frozen_runner != run_index["runner_sha256"]:
        raise ValueError(
            "frozen summary prediction_runner_sha256 disagrees with run index runner_sha256"
        )
    frozen_matrix = _object(frozen_summary.get("matrix"), "frozen summary matrix")
    frozen_expected_raw = frozen_matrix.get("expected_cells")
    if not isinstance(frozen_expected_raw, list):
        raise TypeError("frozen summary matrix.expected_cells must be a list")
    frozen_expected_cells: list[tuple[str, int]] = []
    seen_expected: set[tuple[str, int]] = set()
    for raw_cell in frozen_expected_raw:
        cell_obj = _object(raw_cell, "frozen summary expected cell")
        key = (str(cell_obj["split"]), _nonneg_int(cell_obj["k"], "expected cell k"))
        if key in seen_expected:
            raise ValueError(
                f"frozen summary matrix.expected_cells contains duplicate cell {key!r}"
            )
        seen_expected.add(key)
        frozen_expected_cells.append(key)
    if set(frozen_expected_cells) != set(CELLS):
        raise ValueError(
            "frozen summary matrix.expected_cells does not match the official 12 split/k cells"
        )
    frozen_expected_set = set(frozen_expected_cells)
    frozen_included_raw = frozen_matrix.get("included_cells")
    if not isinstance(frozen_included_raw, list) or not frozen_included_raw:
        raise ValueError("frozen summary matrix.included_cells must be a nonempty list")
    frozen_included: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_cell in frozen_included_raw:
        cell = _object(raw_cell, "frozen summary included cell")
        split = cell.get("split")
        k = _nonneg_int(cell.get("k"), "frozen summary cell k")
        if split not in SPLITS or k not in KS:
            raise ValueError("frozen summary included cell references unknown split/k")
        key = (str(split), k)
        if key in frozen_included:
            raise ValueError("frozen summary matrix contains duplicate included cells")
        if key not in frozen_expected_set:
            raise ValueError("frozen summary included cell is not listed in matrix.expected_cells")
        frozen_included[key] = cell
    frozen_complete = frozen_matrix.get("complete")
    if not isinstance(frozen_complete, bool):
        raise TypeError("frozen summary matrix.complete must be a boolean")
    frozen_subset_allowed = frozen_matrix.get("allow_subset")
    if not isinstance(frozen_subset_allowed, bool):
        raise TypeError("frozen summary matrix.allow_subset must be a boolean")
    included_equals_expected = set(frozen_included) == frozen_expected_set
    if frozen_complete != included_equals_expected:
        raise ValueError(
            "frozen summary matrix.complete is inconsistent with "
            "matrix.included_cells vs matrix.expected_cells"
        )
    frozen_runs_raw = frozen_summary.get("runs")
    if not isinstance(frozen_runs_raw, list) or not frozen_runs_raw:
        raise ValueError("frozen summary runs must be a nonempty list")
    frozen_runs: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_run in frozen_runs_raw:
        run_obj = _object(raw_run, "frozen summary run")
        has_prediction_rows = "prediction_rows" in run_obj
        nested_metrics = run_obj.get("metrics")
        if isinstance(nested_metrics, dict) and "rows" in nested_metrics:
            has_prediction_rows = True
        missing_keys = (FROZEN_RUN_REQUIRED_CELL_KEYS - {"prediction_rows"}) - run_obj.keys()
        if not has_prediction_rows:
            missing_keys = missing_keys | {"prediction_rows"}
        if missing_keys:
            raise ValueError(f"frozen summary run missing required keys {sorted(missing_keys)!r}")
        split = run_obj.get("split")
        k = _nonneg_int(run_obj.get("k"), "frozen summary run k")
        if split not in SPLITS or k not in KS:
            raise ValueError("frozen summary run references unknown split/k")
        key = (str(split), k)
        if key in frozen_runs:
            raise ValueError("frozen summary contains duplicate split/k runs")
        run_denominators = _frozen_metrics(run_obj)
        _nonneg_int(
            run_denominators.get("prediction_rows", run_obj.get("prediction_rows")),
            "frozen summary prediction_rows",
        )
        _hex64(run_obj["predictions_sha256"], "frozen summary predictions_sha256")
        _hex64(
            run_obj["prediction_manifest_sha256"],
            "frozen summary prediction_manifest_sha256",
        )
        _hex64(run_obj["config_sha256"], "frozen summary config_sha256")
        if not isinstance(run_obj["counts"], dict):
            raise TypeError("frozen summary run counts must be a dict of status counts")
        frozen_runs[key] = run_obj
    if set(frozen_runs) != set(frozen_included):
        raise ValueError("frozen summary runs and matrix.included_cells disagree on cell identity")
    runs_raw = run_index.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("run index must contain a nonempty runs array")
    requested_cells: list[tuple[str, int]] = []
    run_index_records: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in runs_raw:
        run = _object(raw, "run index entry")
        split = run.get("split")
        k = _nonneg_int(run.get("k"), "run k")
        if split not in SPLITS or k not in KS:
            raise ValueError("run index entry contains an unknown split or k")
        cell_key = (str(split), k)
        if cell_key in run_index_records:
            raise ValueError("run index contains duplicate split/k cells")
        for key in ("rows", "counts", "predictions_sha256"):
            if key not in run:
                raise ValueError(f"run index entry missing required key {key!r}")
        _nonneg_int(run["rows"], "run index rows")
        if not isinstance(run["counts"], dict):
            raise TypeError("run index counts must be a dict of status counts")
        _hex64(run["predictions_sha256"], "run index predictions_sha256")
        if "config_sha256" in run:
            _hex64(run["config_sha256"], "run index config_sha256")
        requested_cells.append(cell_key)
        run_index_records[cell_key] = run
    if len(requested_cells) != len(set(requested_cells)):
        raise ValueError("run index contains duplicate split/k cells")
    if not allow_subset and set(requested_cells) != set(CELLS):
        raise ValueError("all 12 official split/k cells are required unless --allow-subset")
    if not frozen_complete and not (frozen_subset_allowed and allow_subset):
        raise ValueError(
            "frozen summary reports incomplete matrix but both --allow-subset "
            "and matrix.allow_subset=true are required"
        )
    if frozen_complete and set(requested_cells) != frozen_expected_set:
        raise ValueError(
            "frozen summary reports complete matrix but requested cells do not cover "
            "all matrix.expected_cells"
        )
    if not set(requested_cells).issubset(set(frozen_runs)):
        missing_cells = sorted(set(requested_cells) - set(frozen_runs))
        raise ValueError(f"frozen summary is missing runs for requested cells: {missing_cells}")
    gold_metadata, gold_index = _validate_and_index_gold(rows_path)
    split_to_gold: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in _jsonl(rows_path):
        example_id = str(row.get("id"))
        row_split = str(row.get("split"))
        if row_split in SPLITS:
            split_to_gold[row_split].add(example_id)
    for split in SPLITS:
        if gold_metadata["counts_by_split"].get(split, 0) != len(split_to_gold[split]):
            raise ValueError("gold split counts disagree between metadata and rows pass")
    split_row_counts = gold_metadata["counts_by_split"]
    prediction_files = _iterate_prediction_cells(predictions_dir, [c for c in requested_cells])
    for pred_path, manifest_path in prediction_files.values():
        snapshot[pred_path] = _file_hash(pred_path)
        snapshot[manifest_path] = _file_hash(manifest_path)
    for cell_key in requested_cells:
        run = run_index_records[cell_key]
        pred_path, manifest_path = prediction_files[cell_key]
        manifest = _load(manifest_path)
        frozen_run = frozen_runs[cell_key]
        actual_pred_digest = _file_hash(pred_path)
        if run["predictions_sha256"] != actual_pred_digest:
            raise ValueError(
                f"run index predictions_sha256 disagrees with actual file for cell {cell_key}"
            )
        if run["predictions_sha256"] != manifest.get("predictions_sha256"):
            raise ValueError(
                f"run index predictions_sha256 disagrees with manifest for cell {cell_key}"
            )
        if "config_sha256" in run and run["config_sha256"] != manifest.get("config_sha256"):
            raise ValueError(f"run index config_sha256 disagrees with manifest for cell {cell_key}")
        actual_pred_rows = sum(1 for _ in _jsonl(pred_path))
        if run["rows"] != actual_pred_rows:
            raise ValueError(
                f"run index rows count disagrees with actual prediction file for cell {cell_key}"
            )
        if manifest.get("rows") != actual_pred_rows:
            raise ValueError(
                f"manifest rows count disagrees with actual prediction file for cell {cell_key}"
            )
        if frozen_run["predictions_sha256"] != actual_pred_digest:
            raise ValueError(
                f"frozen summary predictions_sha256 disagrees with actual file for cell {cell_key}"
            )
        if frozen_run["prediction_manifest_sha256"] != _file_hash(manifest_path):
            raise ValueError(
                f"frozen summary prediction_manifest_sha256 disagrees with actual manifest "
                f"for cell {cell_key}"
            )
        if frozen_run["config_sha256"] != manifest.get("config_sha256"):
            raise ValueError(
                f"frozen summary config_sha256 disagrees with manifest for cell {cell_key}"
            )
        frozen_denom_early = _frozen_metrics(frozen_run)
        frozen_prediction_rows_raw = frozen_denom_early.get(
            "prediction_rows", frozen_run.get("prediction_rows")
        )
        if frozen_prediction_rows_raw != actual_pred_rows:
            raise ValueError(
                f"frozen summary prediction_rows disagrees with actual prediction row count "
                f"for cell {cell_key}"
            )
        raw_actual_counts: dict[str, int] = {}
        for pred_row in _jsonl(pred_path):
            status = pred_row.get("status")
            if isinstance(status, str):
                raw_actual_counts[status] = raw_actual_counts.get(status, 0) + 1
        actual_counts = _normalize_status_counts(
            raw_actual_counts,
            f"actual prediction rows status for cell {cell_key}",
            total_rows=actual_pred_rows,
            zero_fill_absent=True,
        )
        run_index_counts = _normalize_status_counts(
            run["counts"],
            f"run index counts for cell {cell_key}",
            total_rows=actual_pred_rows,
            zero_fill_absent=True,
        )
        manifest_counts = _normalize_status_counts(
            manifest.get("counts"),
            f"manifest counts for cell {cell_key}",
            total_rows=actual_pred_rows,
            zero_fill_absent=True,
        )
        frozen_counts = _normalize_status_counts(
            frozen_run["counts"],
            f"frozen summary counts for cell {cell_key}",
            total_rows=actual_pred_rows,
            zero_fill_absent=True,
        )
        for status in NORMALIZED_STATUS_KEYS:
            if run_index_counts[status] != actual_counts[status]:
                raise ValueError(
                    f"run index {status!r} count mismatch in cell {cell_key}: "
                    f"declared {run_index_counts[status]} vs actual {actual_counts[status]}"
                )
            if manifest_counts[status] != actual_counts[status]:
                raise ValueError(
                    f"manifest {status!r} count mismatch in cell {cell_key}: "
                    f"declared {manifest_counts[status]} vs actual {actual_counts[status]}"
                )
            if frozen_counts[status] != actual_counts[status]:
                raise ValueError(
                    f"frozen summary {status!r} count mismatch in cell {cell_key}: "
                    f"declared {frozen_counts[status]} vs actual {actual_counts[status]}"
                )
        all_sources = _collect_frozen_denominator_sources(frozen_run)
        for denom_key in FROZEN_SUMMARY_DENOMINATOR_KEYS:
            entries = all_sources.get(denom_key, [])
            if not entries:
                continue
            for source_label, frozen_val in entries:
                label = f"frozen summary {denom_key!r} for cell {cell_key} (source: {source_label})"
                try:
                    coerced = _coerce_denominator_int(frozen_val, label)
                except TypeError as exc:
                    raise ValueError(str(exc)) from exc
                if denom_key == "prediction_rows":
                    if coerced != actual_pred_rows:
                        raise ValueError(
                            f"frozen summary {denom_key!r} from {source_label} "
                            f"disagrees with expected aggregate for cell {cell_key}"
                        )
                elif denom_key == "rows_total":
                    expected_rows = split_row_counts.get(cell_key[0], 0)
                    if coerced != expected_rows:
                        raise ValueError(
                            f"frozen summary {denom_key!r} from {source_label} "
                            f"disagrees with expected aggregate for cell {cell_key}"
                        )
                elif denom_key == "positive_rows":
                    split_positive = 0
                    for gold_id in split_to_gold[cell_key[0]]:
                        no_positive = gold_index[gold_id][0]
                        if not no_positive:
                            split_positive += 1
                    if coerced != split_positive:
                        raise ValueError(
                            f"frozen summary {denom_key!r} from {source_label} "
                            f"disagrees with expected aggregate for cell {cell_key}"
                        )
                elif denom_key == "recalled_positive_rows":
                    split_recalled = 0
                    for pred_row in _jsonl(pred_path):
                        example_id = str(pred_row.get("id"))
                        if example_id not in gold_index:
                            continue
                        no_positive, positive_ids, _cand_univ, _task = gold_index[example_id]
                        if no_positive:
                            continue
                        returned_raw = pred_row.get("generated_candidates")
                        if not isinstance(returned_raw, list):
                            continue
                        returned_k = tuple(str(item) for item in returned_raw)[: cell_key[1]]
                        if not set(positive_ids).isdisjoint(set(returned_k)):
                            split_recalled += 1
                    if coerced != split_recalled:
                        raise ValueError(
                            f"frozen summary {denom_key!r} from {source_label} "
                            f"disagrees with expected aggregate for cell {cell_key}"
                        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".mind2web-baselines-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "reports"
        staging.mkdir()
        cell_reports: dict[tuple[str, int], dict[str, Any]] = {}
        all_seen_ids: set[str] = set()
        for split, k in requested_cells:
            pred_path, manifest_path = prediction_files[(split, k)]
            report, seen = evaluate_cell(
                split=split,
                k=k,
                gold_index=gold_index,
                split_gold_ids=split_to_gold[split],
                prediction_path=pred_path,
                manifest_path=manifest_path,
                rows_digest=rows_digest,
                run_index_digest=run_index_digest,
                frozen_summary_digest=frozen_summary_digest,
                bootstrap_draws=bootstrap_draws,
                bootstrap_seed=bootstrap_seed,
            )
            for path_key, digest in code_snapshot.items():
                if _file_hash(path_key) != digest:
                    raise ValueError("source mutation during baseline evaluation")
            report["source_sha256"] = source_hashes
            report_name = f"{split}-k{k}-baselines.json"
            report_path = staging / report_name
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            cell_reports[(split, k)] = report
            all_seen_ids |= seen
        summary_runs: list[dict[str, Any]] = []
        for split, k in requested_cells:
            report = cell_reports[(split, k)]
            metrics = report["metrics"]
            bootstrap = report["bootstrap"]
            summary_runs.append(
                {
                    "split": split,
                    "k": k,
                    "report": str(output_dir / f"{split}-k{k}-baselines.json"),
                    "sha256": _file_hash(staging / f"{split}-k{k}-baselines.json"),
                    "rows_total": metrics["rows_total"],
                    "positive_rows": metrics["positive_rows"],
                    "recalled_rows": metrics["recalled_rows"],
                    "overflow_rows": metrics["overflow_rows"],
                    "overflow_reason_counts": dict(metrics["overflow_reason_counts"]),
                    "primary_denominator_convention": metrics["primary_denominator_convention"],
                    "candidate_recall_micro": metrics["candidate_recall_micro"],
                    "selector_accuracy_given_recall_micro": metrics[
                        "selector_accuracy_given_recall_micro"
                    ],
                    "selector_complete_case_accuracy_micro": metrics[
                        "selector_complete_case_accuracy_micro"
                    ],
                    "random_expected_accuracy_micro": metrics["random_expected_accuracy_micro"],
                    "first_position_accuracy_micro": metrics["first_position_accuracy_micro"],
                    "selector_minus_random_micro": metrics["selector_minus_random_micro"],
                    "selector_minus_first_micro": metrics["selector_minus_first_micro"],
                    "candidate_recall_task_macro": metrics["candidate_recall_task_macro"],
                    "selector_accuracy_given_recall_task_macro": metrics[
                        "selector_accuracy_given_recall_task_macro"
                    ],
                    "selector_complete_case_accuracy_task_macro": metrics[
                        "selector_complete_case_accuracy_task_macro"
                    ],
                    "random_expected_task_macro": metrics["random_expected_accuracy_task_macro"],
                    "first_position_task_macro": metrics["first_position_accuracy_task_macro"],
                    "selector_minus_random_task_macro": metrics["selector_minus_random_task_macro"],
                    "selector_minus_first_task_macro": metrics["selector_minus_first_task_macro"],
                    "task_group_count": metrics["task_group_count"],
                    "bootstrap_draws": bootstrap["draws"],
                    "bootstrap_seed": bootstrap["seed"],
                    "selector_minus_random_paired_ci95": bootstrap[
                        "selector_minus_random_paired_ci95"
                    ],
                    "selector_minus_first_paired_ci95": bootstrap[
                        "selector_minus_first_paired_ci95"
                    ],
                    "execution_status_counts": metrics["execution_status_counts"],
                    "selector_counts": metrics["selector"],
                    "rows_input_sha256": report["input_sha256"]["rows"],
                    "run_index_input_sha256": report["input_sha256"]["run_index"],
                    "frozen_summary_input_sha256": report["input_sha256"]["frozen_summary"],
                    "prediction_sha256": report["input_sha256"]["predictions"],
                    "manifest_sha256": report["input_sha256"]["prediction_manifest"],
                }
            )
        summary: dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "scope_label": SCOPE_LABEL,
            "scope": SCOPE_DESCRIPTION,
            "source_sha256": source_hashes,
            "input_sha256": {
                "rows": snapshot[rows_path],
                "run_index": snapshot[run_index_path],
                "frozen_summary": snapshot[frozen_summary_path],
                "retrieval_source_recorded": run_index.get("source_sha256"),
            },
            "prediction_runner_sha256": run_index.get("runner_sha256"),
            "matrix": {
                "complete": set(requested_cells) == set(CELLS),
                "allow_subset": allow_subset,
                "expected_cells": [{"split": split, "k": k} for split, k in CELLS],
                "included_cells": [{"split": split, "k": k} for split, k in requested_cells],
            },
            "bootstrap_parameters": {
                "draws": bootstrap_draws,
                "seed": bootstrap_seed,
                "paired": "selector vs each baseline, per-task deltas",
                "interval": "95% percentile",
            },
            "denominator_reconciliation": {
                "gold_rows_total": gold_metadata["rows_total"],
                "gold_rows_by_split": gold_metadata["counts_by_split"],
                "cells_requested": len(requested_cells),
                "gold_rows_in_included_splits": sum(
                    gold_metadata["counts_by_split"].get(split, 0) for split, _ in requested_cells
                ),
            },
            "runs": summary_runs,
            "limitations": [
                (
                    "Random and first-position baselines are computed analytically from the "
                    "same returned candidate sets used by the frozen selector; they do not "
                    "constitute an independent holdout evaluation."
                ),
                (
                    "Missing prediction records remain in the gold denominator; bootstrap "
                    "intervals are over task groups with a positive row count only."
                ),
            ],
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for path_key, digest in snapshot.items():
            if _file_hash(path_key) != digest:
                raise ValueError("source mutation during baseline evaluation")
        if output_dir.exists():
            raise ValueError("refusing to overwrite baseline evidence")
        staging.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="normalized Mind2Web rows JSONL")
    parser.add_argument(
        "--run-index", type=Path, required=True, help="frozen prediction run index JSON"
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="directory containing split-k{k}.jsonl prediction files and .manifest.json sidecars",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="output directory (must not exist)"
    )
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        required=True,
        help="read-only frozen vons.mind2web-run-summary/v1 aggregate summary JSON",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    try:
        summary = evaluate_baselines(
            args.rows,
            args.run_index,
            args.predictions_dir,
            args.output_dir,
            args.frozen_summary,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
            allow_subset=args.allow_subset,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(
            2,
            f"Mind2Web retrospective baselines failed ({type(exc).__name__}); "
            f"no output committed.\n",
        )
    print(
        json.dumps(
            {
                "status": "evaluated",
                "cells": len(summary["runs"]),
                "complete_matrix": summary["matrix"]["complete"],
                "scope_label": summary["scope_label"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
