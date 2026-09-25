"""Replay frozen Mind2Web predictions without running a model or changing inputs.

Requires <split>-k<k>.jsonl and matching .manifest.json sidecars. By default the
run index must cover all three official held-out splits and k=5/10/20/32.
--allow-subset permits an explicitly partial run index, never unknown cells.
Missing prediction records remain in the gold denominator and are reported
separately from recorded errors and abstentions. Existing output is preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vons.mind2web import Mind2WebExample

SPLITS = ("test_task", "test_website", "test_domain")
KS = (5, 10, 20, 32)
CELLS = tuple((split, k) for split in SPLITS for k in KS)
SOURCE_FILES = (
    "vons/mind2web.py",
    "vons/mind2web_adapter.py",
    "tools/evaluate_mind2web.py",
    "tools/predict_mind2web.py",
    "tools/replay_mind2web_evaluation.py",
)
SCOPE = (
    "Replay of saved exact-k action-target predictions over the released candidate pool; "
    "no model execution, test-set adaptation, or browser task success."
)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _load(path: Path) -> dict[str, object]:
    return _object(
        json.loads(path.read_text(encoding="utf-8"), parse_constant=_nonfinite), "JSON document"
    )


def _jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield _object(json.loads(line, parse_constant=_nonfinite), "JSONL row")


def _hash(path: Path) -> str:
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


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _counts(value: object) -> dict[str, int]:
    counts = _object(value, "status counts")
    return {
        key: count for key, raw in counts.items() if (count := _integer(raw, "status count")) > 0
    }


def _source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {name: root / name for name in SOURCE_FILES}


def _unchanged(snapshot: Mapping[Path, str]) -> None:
    for path, digest in snapshot.items():
        if not path.is_file() or _hash(path) != digest:
            raise ValueError("source mutation during replay")


def _load_evaluator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("vons_frozen_mind2web_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen Mind2Web evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_index(
    index: Mapping[str, object],
    allow_subset: bool,
) -> dict[tuple[str, int], dict[str, object]]:
    runs = index.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("run index must contain a nonempty runs array")
    cells: dict[tuple[str, int], dict[str, object]] = {}
    for raw in runs:
        run = _object(raw, "run")
        split = run.get("split")
        k = _integer(run.get("k"), "run k")
        if split not in SPLITS or k not in KS:
            raise ValueError("run index contains an unknown official split/k cell")
        cell = (str(split), k)
        if cell in cells:
            raise ValueError("run index contains a duplicate split/k cell")
        _integer(run.get("rows"), "run rows")
        _digest(run.get("predictions_sha256"), "run prediction hash")
        _counts(run.get("counts"))
        cells[cell] = run
    if not allow_subset and set(cells) != set(CELLS):
        raise ValueError(
            "all 12 official split/k cells are required; use --allow-subset explicitly"
        )
    _digest(index.get("source_sha256"), "retrieval source hash")
    _digest(index.get("runner_sha256"), "prediction runner hash")
    return cells


def _read_gold(rows_path: Path, evaluator: ModuleType) -> dict[str, dict[str, Mind2WebExample]]:
    by_split: dict[str, dict[str, Mind2WebExample]] = {split: {} for split in SPLITS}
    seen: set[str] = set()
    tasks: dict[str, str] = {}
    for raw in _jsonl(rows_path):
        row = evaluator.Mind2WebExample.from_mapping(raw)
        if row.split not in SPLITS or row.task_id is None:
            raise ValueError("gold rows require an official split and task_id")
        if row.example_id in seen:
            raise ValueError("duplicate gold row id")
        seen.add(row.example_id)
        if row.task_id in tasks and tasks[row.task_id] != row.split:
            raise ValueError("gold task identity appears in multiple splits")
        tasks[row.task_id] = row.split
        by_split[row.split][row.example_id] = row
    return by_split


def _inspect_predictions(
    path: Path,
    *,
    split: str,
    k: int,
    config_hash: str,
    gold: Mapping[str, Mind2WebExample],
) -> dict[str, object]:
    seen: set[str] = set()
    statuses: Counter[str] = Counter()
    reasons: Counter[str | None] = Counter()
    status_reasons: Counter[tuple[str, str | None]] = Counter()
    failure_reasons: Counter[str] = Counter()
    missing_reason_fields = 0
    for row in _jsonl(path):
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in gold or row_id in seen:
            raise ValueError("prediction id is unknown or duplicated")
        seen.add(row_id)
        if (
            row.get("split") != split
            or _integer(row.get("k"), "prediction k") != k
            or row.get("task_id") != gold[row_id].task_id
        ):
            raise ValueError("prediction split/k/task identity disagrees with its cell")
        if row.get("config_sha256") != config_hash:
            raise ValueError("prediction config hash disagrees with its manifest")
        for key in ("request_sha256", "retrieval_input_sha256"):
            _digest(row.get(key), key)
        candidates = row.get("generated_candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) > k
            or any(not isinstance(item, str) or not item for item in candidates)
            or len(candidates) != len(set(candidates))
            or not set(candidates).issubset(gold[row_id].candidate_ids)
        ):
            raise ValueError("prediction candidates violate the gold universe or exact-k limit")
        status = row.get("status")
        selection = row.get("selection")
        if status not in ("ok", "error", "abstain"):
            raise ValueError("unknown prediction status")
        if (status == "ok" and (not isinstance(selection, str) or selection not in candidates)) or (
            status != "ok" and selection is not None
        ):
            raise ValueError("prediction status disagrees with its selection")
        reason = row.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError("prediction reason must be null or a nonempty string")
        missing_reason_fields += int("reason" not in row)
        statuses[status] += 1
        reasons[reason] += 1
        status_reasons[(status, reason)] += 1
        if status != "ok" and reason is not None:
            failure_reasons[reason] += 1
    denominator = len(gold)
    outcomes = {
        "accepted": statuses["ok"],
        "error": statuses["error"],
        "abstain": statuses["abstain"],
        "missing": denominator - len(seen),
    }

    def reason_order(value: str | None) -> tuple[bool, str]:
        return value is not None, value or ""

    return {
        "prediction_rows": len(seen),
        "execution_status_counts": dict(sorted(statuses.items())),
        "execution_reason_counts": [
            {"reason": reason, "count": reasons[reason]}
            for reason in sorted(reasons, key=reason_order)
        ],
        "execution_status_reason_counts": [
            {"status": status, "reason": reason, "count": count}
            for (status, reason), count in sorted(
                status_reasons.items(), key=lambda item: (item[0][0], reason_order(item[0][1]))
            )
        ],
        "execution_failure_counts": dict(sorted(failure_reasons.items())),
        "missing_reason_fields": missing_reason_fields,
        "execution_outcomes": outcomes,
        "acceptance_coverage": statuses["ok"] / denominator if denominator else None,
        "prediction_coverage": len(seen) / denominator if denominator else None,
        "coverage_denominator_rows": denominator,
        "missing_prediction_record_count": outcomes["missing"],
        "missing_prediction_interpretation": (
            "Missing records have no raw status; error and abstain are observed records. "
            "The frozen metric's missing_predictions also includes absent selections on recalled rows."
        ),
    }


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_table(path: Path, runs: list[dict[str, object]]) -> None:
    fields = (
        "split",
        "k",
        "rows",
        "positive_rows",
        "recalled_rows",
        "recall_micro",
        "recall_task_macro",
        "taskmacroCIlo",
        "taskmacroCIhi",
        "conditional_accuracy_micro",
        "conditional_accuracy_task_macro",
        "coverage",
        "error_count",
        "abstain_count",
        "missing_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            metrics = _object(run["metrics"], "metrics")
            bootstrap = _object(run["bootstrap"], "bootstrap")
            outcomes = _object(run["execution_outcomes"], "execution outcomes")
            interval = bootstrap["candidate_recall_task_macro_ci95"]
            if interval is not None and (not isinstance(interval, list) or len(interval) != 2):
                raise ValueError("frozen evaluator returned an invalid bootstrap interval")
            writer.writerow(
                {
                    "split": run["split"],
                    "k": run["k"],
                    "rows": metrics["rows"],
                    "positive_rows": metrics["positive_rows"],
                    "recalled_rows": metrics["recalled_positive_rows"],
                    "recall_micro": metrics["candidate_recall_step_micro"],
                    "recall_task_macro": metrics["candidate_recall_task_macro"],
                    "taskmacroCIlo": interval[0] if interval is not None else None,
                    "taskmacroCIhi": interval[1] if interval is not None else None,
                    "conditional_accuracy_micro": metrics[
                        "selection_accuracy_given_recall_step_micro"
                    ],
                    "conditional_accuracy_task_macro": metrics[
                        "selection_accuracy_given_recall_task_macro"
                    ],
                    "coverage": run["acceptance_coverage"],
                    "error_count": outcomes["error"],
                    "abstain_count": outcomes["abstain"],
                    "missing_count": outcomes["missing"],
                }
            )


def replay(
    rows_path: Path,
    run_index_path: Path,
    predictions_dir: Path,
    output_dir: Path,
    *,
    allow_subset: bool = False,
) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("refusing to overwrite replay evidence")
    sources = _source_paths()
    source_hashes = {name: _hash(path) for name, path in sources.items()}
    code_snapshot = {path: source_hashes[name] for name, path in sources.items()}
    snapshot = {**code_snapshot, rows_path: _hash(rows_path), run_index_path: _hash(run_index_path)}
    index = _load(run_index_path)
    cells = _read_index(index, allow_subset)
    if index["runner_sha256"] != source_hashes["tools/predict_mind2web.py"]:
        raise ValueError("prediction runner source differs from the frozen run index")
    evaluator = _load_evaluator(sources["tools/evaluate_mind2web.py"])
    _unchanged(code_snapshot)
    gold = _read_gold(rows_path, evaluator)
    common_config_hash: str | None = None
    prepared: dict[tuple[str, int], tuple[Path, Path, dict[str, object], dict[str, object]]] = {}
    for split, k in CELLS:
        if (split, k) not in cells:
            continue
        run = cells[(split, k)]
        if not gold[split]:
            raise ValueError("requested split has no gold rows")
        path = predictions_dir / f"{split}-k{k}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        snapshot[path] = _hash(path)
        snapshot[manifest_path] = _hash(manifest_path)
        manifest = _load(manifest_path)
        if manifest.get("schema") != "vons.mind2web-prediction/v1":
            raise ValueError("unsupported prediction manifest schema")
        for record in (run, manifest):
            if record.get("predictions_sha256") != snapshot[path]:
                raise ValueError("prediction file hash disagrees with run index or manifest")
        if (
            manifest.get("source_sha256") != index["source_sha256"]
            or manifest.get("runner_sha256") != index["runner_sha256"]
            or manifest.get("selection_reused_across_k") is not False
        ):
            raise ValueError("prediction provenance or exact-k contract mismatch")
        config = _object(manifest.get("config"), "prediction config")
        if config.get("split") != split or _integer(config.get("k"), "config k") != k:
            raise ValueError("prediction manifest config split/k mismatch")
        config_hash = _object_hash(config)
        if manifest.get("config_sha256") != config_hash:
            raise ValueError("prediction manifest config hash mismatch")
        if "config_sha256" in run and run["config_sha256"] != config_hash:
            raise ValueError("run index config hash mismatch")
        shared = _object_hash(
            {key: value for key, value in config.items() if key not in {"split", "k"}}
        )
        if common_config_hash is not None and shared != common_config_hash:
            raise ValueError("prediction cells mix incompatible frozen configurations")
        common_config_hash = shared
        inspection = _inspect_predictions(
            path, split=split, k=k, config_hash=config_hash, gold=gold[split]
        )
        for record in (run, manifest):
            if _integer(record.get("rows"), "prediction rows") != inspection["prediction_rows"]:
                raise ValueError("prediction row count disagrees with run index or manifest")
            if _counts(record.get("counts")) != inspection["execution_status_counts"]:
                raise ValueError("raw status counts disagree with run index or manifest")
        prepared[(split, k)] = (path, manifest_path, manifest, inspection)
    _unchanged(snapshot)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".mind2web-replay-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "reports"
        staging.mkdir()
        runs: list[dict[str, object]] = []
        for (split, k), (path, manifest_path, manifest, inspection) in prepared.items():
            _unchanged(code_snapshot)
            report = evaluator.evaluate_files(rows_path, path, [k], split=split)
            _unchanged({**code_snapshot, rows_path: snapshot[rows_path], path: snapshot[path]})
            if (
                report["rows"] != len(gold[split])
                or report["prediction_rows"] != inspection["prediction_rows"]
                or report["input_sha256"]
                != {"rows": snapshot[rows_path], "predictions": snapshot[path]}
            ):
                raise ValueError("frozen evaluator input/count provenance mismatch")
            report.update(inspection)
            report["source_sha256"] = source_hashes
            report["replay_provenance"] = {
                "run_index_sha256": snapshot[run_index_path],
                "prediction_manifest_sha256": snapshot[manifest_path],
                "retrieval_source_sha256": index["source_sha256"],
                "config_sha256": manifest["config_sha256"],
            }
            name = f"{split}-k{k}-metrics.json"
            _write(staging / name, report)
            runs.append(
                {
                    "split": split,
                    "k": k,
                    "report": str(output_dir / name),
                    "sha256": _hash(staging / name),
                    "predictions_sha256": snapshot[path],
                    "prediction_manifest_sha256": snapshot[manifest_path],
                    "config_sha256": manifest["config_sha256"],
                    "counts": inspection["execution_status_counts"],
                    **inspection,
                    "metrics": report["metrics_by_k"][0],
                    "bootstrap": report["task_bootstrap_by_k"][str(k)],
                }
            )
        table_path = staging / "summary.csv"
        _write_table(table_path, runs)
        summary: dict[str, object] = {
            "schema": "vons.mind2web-run-summary/v1",
            "scope": SCOPE,
            "source_sha256": source_hashes,
            "input_sha256": {
                "rows": snapshot[rows_path],
                "run_index": snapshot[run_index_path],
                "retrieval_source_recorded": index["source_sha256"],
            },
            "prediction_runner_sha256": index["runner_sha256"],
            "common_config_sha256": common_config_hash,
            "matrix": {
                "complete": set(cells) == set(CELLS),
                "allow_subset": allow_subset,
                "expected_cells": [{"split": split, "k": k} for split, k in CELLS],
                "included_cells": [{"split": run["split"], "k": run["k"]} for run in runs],
            },
            "runs": runs,
            "table": {
                "path": str(output_dir / table_path.name),
                "sha256": _hash(table_path),
                "bytes": table_path.stat().st_size,
                "null_encoding": "empty CSV cell; null in JSON",
                "coverage": "accepted raw status=ok records divided by all gold rows in the split",
                "missing_count": "gold rows with no prediction record, excluding error/abstain records",
                "taskmacroCI": "95% task-bootstrap interval for candidate recall task macro",
            },
            "limitations": [
                "Retrieval-source and per-request hashes are preserved from predictions; this CLI does not reload retrieval text or model assets.",
                "Missing prediction records remain in the gold denominator; bootstrap settings come unchanged from the frozen evaluator.",
            ],
        }
        _write(staging / "summary.json", summary)
        _unchanged(snapshot)
        if output_dir.exists():
            raise ValueError("refusing to overwrite replay evidence")
        staging.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    try:
        summary = replay(
            args.rows,
            args.run_index,
            args.predictions_dir,
            args.output_dir,
            allow_subset=args.allow_subset,
        )
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(
            2, f"Mind2Web replay failed ({type(exc).__name__}); no replay output committed.\n"
        )
    print(
        json.dumps(
            {
                "status": "replayed",
                "cells": len(summary["runs"]),
                "complete_matrix": summary["matrix"]["complete"],
            }
        )
    )


if __name__ == "__main__":
    main()
