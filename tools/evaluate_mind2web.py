"""Run the restricted Mind2Web candidate/selection evaluation.

The input rows must be normalized JSONL rows accepted by
``Mind2WebExample.from_mapping``. Predictions are intentionally a separate
JSONL file so retrieval misses and missing selections remain observable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from vons.mind2web import Mind2WebExample, evaluate_mind2web, task_bootstrap_intervals


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(dict(value))
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a JSON list")
    values: list[str] = []
    for item in value:
        candidate = item.get("id") if isinstance(item, Mapping) else item
        if not isinstance(candidate, str) or not candidate:
            raise TypeError(f"{label} ids must be non-empty strings")
        values.append(candidate)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} ids must be unique")
    return tuple(values)


class PredictionBundle(NamedTuple):
    generated_candidates: dict[str, tuple[str, ...]]
    selections: dict[str, str | None]
    declared_k: int | None


def load_predictions(path: Path) -> PredictionBundle:
    generated_candidates: dict[str, tuple[str, ...]] = {}
    selections: dict[str, str | None] = {}
    declared_ks: set[int] = set()
    missing_k = False
    for row in _read_jsonl(path):
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"{path}: prediction requires id")
        key = example_id
        if key in generated_candidates:
            raise ValueError(f"{path}: duplicate prediction id {key}")
        if "generated_candidates" not in row:
            raise ValueError(f"{path}: prediction requires generated_candidates")
        generated_candidates[key] = _string_sequence(row["generated_candidates"], "generated_candidates")
        selection = row.get("selection", row.get("selected_id"))
        selections[key] = None if selection in (None, "") else str(selection)
        row_k = row.get("k")
        if row_k is None:
            missing_k = True
        elif isinstance(row_k, bool) or not isinstance(row_k, int) or row_k <= 0:
            raise ValueError(f"{path}: prediction k must be a positive integer")
        else:
            declared_ks.add(row_k)
    if declared_ks and missing_k:
        raise ValueError(f"{path}: every prediction row must declare k when any row does")
    if len(declared_ks) > 1:
        raise ValueError(f"{path}: prediction rows must use one k, found {sorted(declared_ks)}")
    return PredictionBundle(
        generated_candidates=generated_candidates,
        selections=selections,
        declared_k=next(iter(declared_ks), None),
    )


def evaluate_files(
    rows_path: Path,
    predictions_path: Path,
    ks: Sequence[int],
    split: str | None = None,
) -> dict[str, Any]:
    raw_rows = _read_jsonl(rows_path)
    if split is not None:
        raw_rows = [row for row in raw_rows if row.get("split") == split]
    examples = [Mind2WebExample.from_mapping(row) for row in raw_rows]
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError(f"{rows_path}: duplicate gold row ids")
    predictions = load_predictions(predictions_path)
    generated_candidates = predictions.generated_candidates
    example_ids = {row.example_id for row in examples}
    unknown_prediction_ids = sorted(set(generated_candidates) - example_ids)
    if unknown_prediction_ids:
        raise ValueError(f"predictions contain unknown example ids: {unknown_prediction_ids[:5]}")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must not contain duplicates")

    if predictions.declared_k is not None and tuple(ks) != (predictions.declared_k,):
        raise ValueError(
            f"predictions are k={predictions.declared_k}; evaluate exactly that single k, not {list(ks)}"
        )
    selection_is_k_specific = predictions.declared_k is not None or len(ks) == 1
    selections = predictions.selections if selection_is_k_specific else {}
    metric_mappings: list[dict[str, Any]] = []
    bootstrap_by_k: dict[str, dict[str, Any]] = {}
    for k in ks:
        metric = evaluate_mind2web(examples, generated_candidates, selections, k=k)
        mapping = metric.to_mapping()
        if not selection_is_k_specific:
            mapping["selection_accuracy_given_recall"] = None
            mapping["complete_case_selection_accuracy_given_recall"] = None
            mapping["selection_metrics_status"] = "not_evaluated_recall_only"
        metric_mappings.append(mapping)
        bootstrap = task_bootstrap_intervals(examples, generated_candidates, selections, k=k)
        if not selection_is_k_specific:
            bootstrap["selection_accuracy_given_recall_task_macro_ci95"] = None
        bootstrap_by_k[str(k)] = bootstrap
    return {
        "schema": "vons.mind2web-evaluation/v1",
        "rows_path": str(rows_path),
        "predictions_path": str(predictions_path),
        "rows": len(examples),
        "prediction_rows": len(generated_candidates),
        "split": split,
        "input_sha256": {
            "rows": _file_sha256(rows_path),
            "predictions": _file_sha256(predictions_path),
        },
        "prediction_k": predictions.declared_k,
        "metrics_by_k": metric_mappings,
        "task_bootstrap_by_k": bootstrap_by_k,
        "selection_evaluation": {
            "ranking_and_selection_predictions_supplied_once": True,
            "selection_reused_across_k": False,
            "selection_metrics_valid": selection_is_k_specific,
            "interpretation": (
                "Selection is evaluated only at the declared single prediction k. "
                "When multiple k values are requested without a declared k, metrics are recall-only; "
                "separate k-specific prediction files are required for selection."
            ),
        },
        "scope": "candidate recall and candidate-in-set selection only; no browser task success",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="normalized Mind2Web rows JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="retriever/selector predictions JSONL")
    parser.add_argument("--output", type=Path, required=True, help="evaluation report JSON")
    parser.add_argument("--split", help="evaluate only rows whose normalized split matches this value")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 32])
    args = parser.parse_args()

    report = evaluate_files(args.rows, args.predictions, args.k, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
