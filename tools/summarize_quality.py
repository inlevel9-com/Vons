"""Recompute quality metrics from saved Vons prediction reports.

This tool never fabricates unavailable external-task results. Its current
reports are explicitly labeled synthetic-pilot measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from vons.data import read_jsonl
from vons.evaluation import Prediction, summarize


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("predictions"), list):
        raise TypeError(f"{path} must contain a predictions list")
    return value


def to_predictions(raw: list[dict[str, Any]], path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    seen: set[str] = set()
    for index, value in enumerate(raw, start=1):
        if not isinstance(value, dict):
            raise TypeError(f"{path}: prediction {index} must be an object")
        example_id = str(value.get("example_id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"{path}: duplicate or missing example_id at row {index}")
        seen.add(example_id)
        probabilities = value.get("probabilities")
        if not isinstance(probabilities, dict):
            raise TypeError(f"{path}: probabilities must be an object at row {index}")
        numeric_probabilities = {str(key): float(item) for key, item in probabilities.items()}
        confidence = float(value.get("confidence"))
        if not math.isfinite(confidence) or any(not math.isfinite(item) for item in numeric_probabilities.values()):
            raise ValueError(f"{path}: non-finite probability or confidence at row {index}")
        predictions.append(
            Prediction(
                example_id=example_id,
                choice=None if value.get("choice") is None else str(value["choice"]),
                probabilities=numeric_probabilities,
                confidence=confidence,
                abstained=bool(value.get("abstained", False)),
                latency_ms=None if value.get("latency_ms") is None else float(value["latency_ms"]),
                error=None if value.get("error") is None else str(value["error"]),
            )
        )
    return predictions


def summarize_report(data_path: Path, report_path: Path) -> dict[str, Any]:
    examples = read_jsonl(data_path)
    report = load_report(report_path)
    predictions = to_predictions(report["predictions"], report_path)
    expected_ids = {row.id for row in examples}
    prediction_ids = {item.example_id for item in predictions}
    summary = summarize(examples, predictions)
    config = report.get("config", {})
    if not isinstance(config, dict):
        raise TypeError(f"{report_path}: config must be an object")
    return {
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "backend": config.get("backend"),
        "seed": config.get("seed"),
        "model_id": config.get("model_id"),
        "dataset": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "dataset_rows": len(examples),
        "prediction_rows": len(predictions),
        "missing_prediction_rows": len(expected_ids - prediction_ids),
        "extra_prediction_rows": len(prediction_ids - expected_ids),
        "answerable_rows": sum(row.answerable for row in examples),
        "answerable_prediction_rows": sum(row.answerable and row.id in prediction_ids for row in examples),
        "abstained_rows": sum(item.abstained for item in predictions),
        "error_rows": sum(item.error is not None for item in predictions),
        "metrics": summary,
        "measurement_label": "measured_synthetic_pilot",
        "quality_scope": "synthetic test split only; not Mind2Web or browser task success",
    }


def descriptive_seed_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["backend"])].append(row)
    fields = ("accuracy_answerable", "macro_f1_answerable", "coverage", "selective_risk", "nll_answerable", "brier_answerable", "ece")
    result: dict[str, Any] = {}
    for backend, backend_rows in sorted(grouped.items()):
        result[backend] = {}
        for field in fields:
            values = [row["metrics"].get(field) for row in backend_rows]
            defined = [float(value) for value in values if value is not None]
            result[backend][field] = {
                "mean": statistics.mean(defined) if defined else None,
                "min": min(defined) if defined else None,
                "max": max(defined) if defined else None,
                "seed_count": len(backend_rows),
                "defined_seed_count": len(defined),
                "null_seed_count": len(values) - len(defined),
                "interpretation": "descriptive across saved seeds; not an independent test population",
            }
    return result


def build_quality_summary(data_path: Path, report_paths: list[Path], *, version: str = "v1") -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one report is required")
    rows = [summarize_report(data_path, path) for path in report_paths]
    if version not in {"v1", "v2"}:
        raise ValueError("version must be v1 or v2")
    result = {
        "schema": f"vons.quality-summary/{version}",
        "measurement_label": "measured_synthetic_pilot",
        "dataset": {"path": str(data_path), "sha256": file_sha256(data_path), "rows": len(read_jsonl(data_path))},
        "reports": rows,
        "descriptive_seed_stats": descriptive_seed_stats(rows),
        "external_evaluation": {
            "mind2web": {
                "status": "not_evaluated",
                "candidate_recall": None,
                "conditional_selection_accuracy": None,
                "reason": "No Mind2Web evaluation report was supplied to this synthetic summary.",
            },
            "jev": {"status": "not_run", "agreement": None},
        },
        "disclosure": "These numbers are recomputed from saved synthetic-pilot predictions. They do not establish external-task quality, browser task success, or generalization.",
    }
    if version == "v2":
        result["metric_version"] = "vons.metrics/v2"
        result["input_sha256"] = {
            "dataset": file_sha256(data_path),
            "reports": {str(path): file_sha256(path) for path in report_paths},
        }
        result["disclosure"] += " Version 2 also records every input digest and uses the vons.metrics/v2 definitions present in each saved report."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    result = build_quality_summary(args.data, args.report, version=args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
