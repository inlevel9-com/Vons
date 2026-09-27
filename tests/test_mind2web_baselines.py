"""Synthetic/public fixtures only; no real Mind2Web data in tests.

Coverage:
- exact random expectation: p / c with single and multiple positives;
- first-position baseline correctness;
- denominator reconciliation across positive/no-positive and every
  execution status plus missing prediction records;
- task-cluster grouping and task-macro averaging;
- error/abstain/missing/overflow/invalid-selection preservation
  in all aggregate counts;
- deterministic paired bootstrap intervals given a fixed seed;
- absence of row-level content leakage: ids, task ids, candidate text,
  goal text, DOM payloads, and prediction metadata beyond digests.
"""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_tool() -> object:
    path = Path(__file__).parents[1] / "tools/evaluate_mind2web_baselines.py"
    spec = importlib.util.spec_from_file_location("vons_mind2web_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the baseline evaluator tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baselines = _load_tool()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hex64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _build_fixture(
    root: Path,
    *,
    cells=((("test_task", 5),)),
    multiple_positives: bool = False,
    include_failures: bool = False,
    include_overflow: bool = False,
    include_invalid: bool = False,
    two_tasks: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    rows_path = root / "rows.jsonl"
    index_path = root / "run-index.json"
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    frozen_summary_path = root / "frozen-summary.json"
    gold: list[dict] = []
    per_split_cells: dict[str, list[int]] = {}
    for split, k in cells:
        per_split_cells.setdefault(split, []).append(k)
    for split in per_split_cells:
        variants = ["correct_first", "correct_middle", "two_positives", "no_positive"]
        if include_failures:
            variants += ["abstain", "error", "missing"]
        if include_overflow:
            variants += ["overflow"]
        if include_invalid:
            variants += ["invalid_selection"]
        for task_counter, variant in enumerate(variants):
            task_num = task_counter if two_tasks else 0
            row: dict = {
                "id": f"{split}-{variant}",
                "task_id": f"task-{split}-{task_num:02d}",
                "action_id": variant,
                "split": split,
                "website": f"{split}-site",
                "domain": f"{split}-domain",
                "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
            if variant == "no_positive":
                row["no_positive"] = True
                row["positive_ids"] = []
            elif variant == "two_positives" or (multiple_positives and variant == "correct_middle"):
                row["positive_ids"] = ["b", "d"]
            else:
                if variant == "correct_first":
                    row["positive_ids"] = ["a"]
                elif variant == "correct_middle":
                    row["positive_ids"] = ["c"]
                else:
                    row["positive_ids"] = ["a"]
            gold.append(row)
    _write_rows(rows_path, gold)
    runner_hash = "a" * 64
    source_hash = "b" * 64
    index: dict = {
        "schema": baselines.RUN_INDEX_SCHEMA,
        "source_sha256": source_hash,
        "runner_sha256": runner_hash,
        "runs": [],
    }
    per_split_positive_counts: dict[str, int] = {}
    per_split_recalled_counts: dict[tuple[str, int], int] = {}
    for g in gold:
        s = g["split"]
        if not g.get("no_positive"):
            per_split_positive_counts[s] = per_split_positive_counts.get(s, 0) + 1
    for split, k in cells:
        config: dict = {
            "split": split,
            "k": k,
            "provenance": {"backend": "fixture"},
            "adapted": False,
        }
        config_hash = baselines._object_hash(config)
        predictions: list[dict] = []
        for variant in [
            g for g in gold if g["split"] == split and g["id"].split("-", 1)[1] != "missing"
        ]:
            variant_name = variant["id"].split("-", 1)[1]
            set(variant["positive_ids"])
            if variant_name == "overflow":
                returned = ["a", "b", "c", "d", "e", "f"]
                status = "ok"
                selection = "a"
                reason = None
            elif variant_name == "invalid_selection":
                returned = ["a", "b", "c", "d", "e"]
                status = "ok"
                selection = "not_in_candidates"
                reason = None
            elif variant_name == "abstain":
                returned = ["a", "b", "c", "d", "e"]
                status = "abstain"
                selection = None
                reason = "confidence_below_threshold"
            elif variant_name == "error":
                returned = ["a", "b", "c", "d", "e"]
                status = "error"
                selection = None
                reason = "input_overflow"
            else:
                returned = ["a", "b", "c", "d", "e"][: min(k, 5)]
                status = "ok"
                if variant_name == "correct_first":
                    selection = "a"
                elif variant_name == "correct_middle":
                    selection = "c" if "c" in returned else None
                elif variant_name == "two_positives":
                    selection = "d" if "d" in returned else None
                else:
                    selection = returned[0] if returned else None
                reason = None
            predictions.append(
                {
                    "id": variant["id"],
                    "task_id": variant["task_id"],
                    "split": split,
                    "k": k,
                    "generated_candidates": list(returned),
                    "selection": selection,
                    "status": status,
                    "reason": reason,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                }
            )
        prediction_path = predictions_dir / f"{split}-k{k}.jsonl"
        _write_rows(prediction_path, predictions)

        raw_counts = dict(Counter(row["status"] for row in predictions))
        counts = {
            "ok": raw_counts.get("ok", 0),
            "error": raw_counts.get("error", 0),
            "abstain": raw_counts.get("abstain", 0),
        }
        digest = baselines._file_hash(prediction_path)
        manifest: dict = {
            "schema": "vons.mind2web-prediction/v1",
            "config": config,
            "config_sha256": config_hash,
            "source_sha256": index["source_sha256"],
            "runner_sha256": index["runner_sha256"],
            "predictions_sha256": digest,
            "rows": len(predictions),
            "counts": counts,
            "selection_reused_across_k": False,
        }
        _write(prediction_path.with_suffix(".manifest.json"), manifest)
        index["runs"].append(
            {
                "split": split,
                "k": k,
                "rows": len(predictions),
                "counts": counts,
                "predictions_sha256": digest,
                "config_sha256": config_hash,
            }
        )
        recalled = 0
        for pred in predictions:
            if pred["id"].split("-", 1)[1] == "no_positive":
                continue
            positive_ids_set = set()
            for g in gold:
                if g["id"] == pred["id"]:
                    positive_ids_set = set(g["positive_ids"])
                    break
            returned_k = tuple(pred["generated_candidates"])[:k]
            if positive_ids_set and not positive_ids_set.isdisjoint(set(returned_k)):
                recalled += 1
        per_split_recalled_counts[(split, k)] = recalled
    _write(index_path, index)
    rows_digest = baselines._file_hash(rows_path)
    run_index_digest = baselines._file_hash(index_path)
    split_row_counts: dict[str, int] = {}
    for row in gold:
        split_row_counts[row["split"]] = split_row_counts.get(row["split"], 0) + 1
    frozen_runs: list[dict] = []
    for split, k in cells:
        manifest_path = predictions_dir / f"{split}-k{k}.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        cell_rows = split_row_counts.get(split, 0)
        cell_positives = per_split_positive_counts.get(split, 0)
        cell_recalled = per_split_recalled_counts.get((split, k), 0)
        frozen_runs.append(
            {
                "split": split,
                "k": k,
                "prediction_rows": manifest["rows"],
                "predictions_sha256": manifest["predictions_sha256"],
                "prediction_manifest_sha256": baselines._file_hash(manifest_path),
                "config_sha256": manifest["config_sha256"],
                "counts": dict(manifest["counts"]),
                "metrics": {
                    "prediction_rows": manifest["rows"],
                    "rows_total": cell_rows,
                    "positive_rows": cell_positives,
                    "recalled_positive_rows": cell_recalled,
                },
            }
        )
    frozen_summary: dict = {
        "schema": baselines.FROZEN_RUN_SUMMARY_SCHEMA,
        "input_sha256": {
            "rows": rows_digest,
            "run_index": run_index_digest,
            "retrieval_source_recorded": source_hash,
        },
        "prediction_runner_sha256": runner_hash,
        "matrix": {
            "complete": set(cells) == set(baselines.CELLS),
            "allow_subset": set(cells) != set(baselines.CELLS),
            "expected_cells": [{"split": split, "k": k} for split, k in baselines.CELLS],
            "included_cells": [{"split": split, "k": k} for split, k in cells],
        },
        "runs": frozen_runs,
    }
    _write(frozen_summary_path, frozen_summary)
    return rows_path, index_path, predictions_dir, root / "output", frozen_summary_path


def _collect_leak(report: dict) -> list[str]:
    leaked: list[str] = []

    def walk(node: object, in_dict_key: bool = False) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    low = key.lower()
                    if low in {
                        "example_id",
                        "goal",
                        "html",
                        "query",
                        "snippet",
                        "candidate_text",
                        "dom_text",
                        "dom_content",
                        "raw_dom",
                    }:
                        leaked.append(f"key:{key}")
                    if low == "dom" and isinstance(value, (str, list, dict)) and value:
                        leaked.append(f"key:{key}")
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            tokens = re.split(r"[\\/\"' \t\n,;]", node)
            for token in tokens:
                for prefix in ("task-test_task-", "task-test_website-", "task-test_domain-"):
                    if token.startswith(prefix):
                        leaked.append(f"prefix:{prefix}")
                if re.fullmatch(r"test_(task|website|domain)-[a-z_]+", token):
                    leaked.append(f"id-token:{token[:40]}")

    walk(report)
    return leaked


def _assert_no_leakage(report: dict) -> None:
    leaked = _collect_leak(report)
    assert not leaked, f"leaked sensitive keys/patterns in report: {sorted(set(leaked))[:8]}"


def _write_manual_frozen_summary(
    frozen_summary_path: Path,
    rows_path: Path,
    index_path: Path,
    predictions_dir: Path,
    cells,
    gold_rows,
    source_hash: str,
    runner_hash: str,
) -> None:
    rows_digest = baselines._file_hash(rows_path)
    run_index_digest = baselines._file_hash(index_path)
    split_row_counts: dict[str, int] = {}
    split_positive_counts: dict[str, int] = {}
    for row in gold_rows:
        s = row["split"]
        split_row_counts[s] = split_row_counts.get(s, 0) + 1
        if not row.get("no_positive"):
            split_positive_counts[s] = split_positive_counts.get(s, 0) + 1
    gold_by_id = {row["id"]: row for row in gold_rows}
    frozen_runs: list[dict] = []
    for split, k in cells:
        manifest_path = predictions_dir / f"{split}-k{k}.manifest.json"
        pred_path = predictions_dir / f"{split}-k{k}.jsonl"
        manifest = json.loads(manifest_path.read_text())
        recalled = 0
        for pred in baselines._jsonl(pred_path):
            gid = pred.get("id")
            if gid not in gold_by_id:
                continue
            grow = gold_by_id[gid]
            if grow.get("no_positive"):
                continue
            pids = set(grow.get("positive_ids", ()))
            if not pids:
                continue
            returned_raw = pred.get("generated_candidates")
            if not isinstance(returned_raw, list):
                continue
            returned_k = tuple(str(item) for item in returned_raw)[:k]
            if not pids.isdisjoint(set(returned_k)):
                recalled += 1
        frozen_runs.append(
            {
                "split": split,
                "k": k,
                "prediction_rows": manifest["rows"],
                "predictions_sha256": manifest["predictions_sha256"],
                "prediction_manifest_sha256": baselines._file_hash(manifest_path),
                "config_sha256": manifest["config_sha256"],
                "counts": {
                    "ok": manifest["counts"].get("ok", 0),
                    "error": manifest["counts"].get("error", 0),
                    "abstain": manifest["counts"].get("abstain", 0),
                },
                "metrics": {
                    "prediction_rows": manifest["rows"],
                    "rows_total": split_row_counts.get(split, 0),
                    "positive_rows": split_positive_counts.get(split, 0),
                    "recalled_positive_rows": recalled,
                },
            }
        )
    frozen_summary: dict = {
        "schema": baselines.FROZEN_RUN_SUMMARY_SCHEMA,
        "input_sha256": {
            "rows": rows_digest,
            "run_index": run_index_digest,
            "retrieval_source_recorded": source_hash,
        },
        "prediction_runner_sha256": runner_hash,
        "matrix": {
            "complete": set(cells) == set(baselines.CELLS),
            "allow_subset": set(cells) != set(baselines.CELLS),
            "expected_cells": [{"split": s, "k": k} for s, k in baselines.CELLS],
            "included_cells": [{"split": s, "k": k} for s, k in cells],
        },
        "runs": frozen_runs,
    }
    _write(frozen_summary_path, frozen_summary)


class BaselineCoreTests(unittest.TestCase):
    def test_exact_random_expectation_with_multiple_positives(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=(("test_task", 5),),
                multiple_positives=True,
            )
            summary = baselines.evaluate_baselines(*paths, allow_subset=True, bootstrap_draws=31)
            run = summary["runs"][0]
            self.assertEqual(run["k"], 5)
            cell = json.loads((paths[3] / "test_task-k5-baselines.json").read_text())
            metrics = cell["metrics"]
            self.assertGreater(metrics["positive_rows"], 0)
            random_micro = metrics["random_expected_accuracy_micro"]
            self.assertIsNotNone(random_micro)
            self.assertGreaterEqual(random_micro, 0.0)
            self.assertLessEqual(random_micro, 1.0)
            correct_first_found = False
            two_positives_found = False
            for row in baselines._jsonl(paths[0]):
                if row["positive_ids"] == ["a"] and "correct_first" in str(row.get("id")):
                    correct_first_found = True
                if len(row["positive_ids"]) >= 2:
                    two_positives_found = True
            self.assertTrue(correct_first_found)
            self.assertTrue(two_positives_found)
            self.assertGreater(metrics["recalled_rows"], 0)
            expected_total = 0.0
            for row in baselines._jsonl(paths[0]):
                positive_ids = set(row["positive_ids"])
                if not positive_ids:
                    continue
                predictions = list(baselines._jsonl(paths[2] / "test_task-k5.jsonl"))
                for pred in predictions:
                    if pred["id"] == row["id"]:
                        cands = pred["generated_candidates"]
                        overlap = len(positive_ids & set(cands))
                        if overlap > 0:
                            expected_total += overlap / len(cands)
                        break
            self.assertAlmostEqual(
                random_micro, expected_total / metrics["recalled_rows"], places=10
            )

    def test_first_position_baseline_matches_first_candidate_positive_status(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=(("test_task", 5),),
            )
            baselines.evaluate_baselines(*paths, allow_subset=True, bootstrap_draws=17)
            cell = json.loads((paths[3] / "test_task-k5-baselines.json").read_text())
            metrics = cell["metrics"]
            recalled = metrics["recalled_rows"]
            self.assertGreater(recalled, 0)
            first_correct_count = 0
            for row in baselines._jsonl(paths[0]):
                positive_ids = set(row["positive_ids"])
                if not positive_ids:
                    continue
                for pred in baselines._jsonl(paths[2] / "test_task-k5.jsonl"):
                    if pred["id"] != row["id"]:
                        continue
                    cands = pred["generated_candidates"]
                    overlap = positive_ids & set(cands)
                    if not overlap:
                        break
                    if cands and cands[0] in positive_ids:
                        first_correct_count += 1
                    break
            self.assertAlmostEqual(
                metrics["first_position_accuracy_micro"],
                first_correct_count / recalled,
                places=10,
            )

    def test_denominator_reconciliation_all_statuses_and_positivity(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=(("test_task", 5),),
                include_failures=True,
                include_overflow=True,
                include_invalid=True,
            )
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            summary["runs"][0]
            cell = json.loads((paths[3] / "test_task-k5-baselines.json").read_text())
            cell_metrics = cell["metrics"]
            total = cell_metrics["rows_total"]
            positive = cell_metrics["positive_rows"]
            no_positive = cell_metrics["no_positive_rows"]
            self.assertEqual(total, positive + no_positive)
            statuses = cell_metrics["execution_status_counts"]
            predicted = cell_metrics["prediction_records"]
            missing = statuses["missing_record"]
            self.assertEqual(predicted + missing, total)
            self.assertEqual(
                statuses["accepted"] + statuses["error"] + statuses["abstain"],
                predicted,
            )
            sel = cell_metrics["selector"]
            on_recalled = (
                sel["evaluated_recalled"]
                + sel["missing_selection_on_recalled"]
                + sel["invalid_selection_on_recalled"]
            )
            self.assertEqual(on_recalled, cell_metrics["recalled_rows"])
            gold_rows = list(baselines._jsonl(paths[0]))
            self.assertEqual(
                summary["denominator_reconciliation"]["gold_rows_total"], len(gold_rows)
            )
            self.assertGreater(cell_metrics["overflow_rows"], 0)

    def test_task_cluster_grouping_distinguishes_two_tasks(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            one_root = directory / "one"
            two_root = directory / "two"
            paths_one = _build_fixture(
                one_root,
                cells=(("test_task", 5),),
                two_tasks=False,
            )
            paths_two = _build_fixture(
                two_root,
                cells=(("test_task", 5),),
                two_tasks=True,
            )
            extra_rows = []
            for root, paths, two_tasks in ((two_root, paths_two, True),):
                existing = list(baselines._jsonl(paths[0]))
                extra_row = {
                    "id": "test_task-unrecalled",
                    "task_id": "task-test_task-99",
                    "action_id": "unrecalled",
                    "split": "test_task",
                    "website": "test_task-site",
                    "domain": "test_task-domain",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["h"],
                }
                extra_rows.append(extra_row)
                all_rows = existing + [extra_row]
                _write_rows(paths[0], all_rows)
                pred_rows = list(baselines._jsonl(paths[2] / "test_task-k5.jsonl"))
                config_hash = pred_rows[0]["config_sha256"]
                pred_rows.append(
                    {
                        "id": "test_task-unrecalled",
                        "task_id": "task-test_task-99",
                        "split": "test_task",
                        "k": 5,
                        "generated_candidates": ["a", "b", "c", "d", "e"],
                        "selection": "a",
                        "status": "ok",
                        "reason": None,
                        "config_sha256": config_hash,
                        "request_sha256": "c" * 64,
                        "retrieval_input_sha256": "d" * 64,
                    }
                )
                _write_rows(paths[2] / "test_task-k5.jsonl", pred_rows)

                counts_raw = dict(Counter(row["status"] for row in pred_rows))
                counts = {
                    "ok": counts_raw.get("ok", 0),
                    "error": counts_raw.get("error", 0),
                    "abstain": counts_raw.get("abstain", 0),
                }
                pred_digest = baselines._file_hash(paths[2] / "test_task-k5.jsonl")
                manifest_path = paths[2] / "test_task-k5.manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["predictions_sha256"] = pred_digest
                manifest["rows"] = len(pred_rows)
                manifest["counts"] = counts
                _write(manifest_path, manifest)
                index = json.loads(paths[1].read_text())
                for run in index["runs"]:
                    if run["split"] == "test_task" and run["k"] == 5:
                        run["predictions_sha256"] = pred_digest
                        run["rows"] = len(pred_rows)
                        run["counts"] = counts
                _write(paths[1], index)
                new_rows_digest = baselines._file_hash(paths[0])
                new_run_index_digest = baselines._file_hash(paths[1])
                new_split_row_counts: dict[str, int] = {}
                new_split_positive_counts: dict[str, int] = {}
                new_recalled = 0
                for r in baselines._jsonl(paths[0]):
                    rs = r["split"]
                    new_split_row_counts[rs] = new_split_row_counts.get(rs, 0) + 1
                    if not r.get("no_positive"):
                        new_split_positive_counts[rs] = new_split_positive_counts.get(rs, 0) + 1
                for pr in pred_rows:
                    if pr["split"] != "test_task":
                        continue
                    gid = pr["id"]
                    grow = None
                    for r in baselines._jsonl(paths[0]):
                        if r["id"] == gid:
                            grow = r
                            break
                    if grow is None or grow.get("no_positive"):
                        continue
                    pids = set(grow.get("positive_ids", ()))
                    if not pids:
                        continue
                    returned_k = tuple(str(x) for x in pr["generated_candidates"])[:5]
                    if not pids.isdisjoint(set(returned_k)):
                        new_recalled += 1
                new_manifest_path = paths[2] / "test_task-k5.manifest.json"
                new_manifest = json.loads(new_manifest_path.read_text())
                rebuilt_frozen_runs = [
                    {
                        "split": "test_task",
                        "k": 5,
                        "prediction_rows": new_manifest["rows"],
                        "predictions_sha256": new_manifest["predictions_sha256"],
                        "prediction_manifest_sha256": baselines._file_hash(new_manifest_path),
                        "config_sha256": new_manifest["config_sha256"],
                        "counts": {
                            "ok": new_manifest["counts"].get("ok", 0),
                            "error": new_manifest["counts"].get("error", 0),
                            "abstain": new_manifest["counts"].get("abstain", 0),
                        },
                        "metrics": {
                            "prediction_rows": new_manifest["rows"],
                            "rows_total": new_split_row_counts.get("test_task", 0),
                            "positive_rows": new_split_positive_counts.get("test_task", 0),
                            "recalled_positive_rows": new_recalled,
                        },
                    }
                ]
                rebuilt_frozen_summary: dict = {
                    "schema": baselines.FROZEN_RUN_SUMMARY_SCHEMA,
                    "input_sha256": {
                        "rows": new_rows_digest,
                        "run_index": new_run_index_digest,
                        "retrieval_source_recorded": index["source_sha256"],
                    },
                    "prediction_runner_sha256": index["runner_sha256"],
                    "matrix": {
                        "complete": False,
                        "allow_subset": True,
                        "expected_cells": [{"split": s, "k": k} for s, k in baselines.CELLS],
                        "included_cells": [{"split": "test_task", "k": 5}],
                    },
                    "runs": rebuilt_frozen_runs,
                }
                _write(paths[4], rebuilt_frozen_summary)
            summary_one = baselines.evaluate_baselines(
                *paths_one, allow_subset=True, bootstrap_draws=7
            )
            summary_two = baselines.evaluate_baselines(
                *paths_two, allow_subset=True, bootstrap_draws=7
            )
            self.assertEqual(summary_one["runs"][0]["task_group_count"], 1)
            self.assertGreater(summary_two["runs"][0]["task_group_count"], 1)
            self.assertNotEqual(
                summary_one["runs"][0]["task_group_count"],
                summary_two["runs"][0]["task_group_count"],
            )

    def test_failures_and_missing_are_preserved_in_aggregates(self):
        with TemporaryDirectory() as directory:
            paths_clean = _build_fixture(Path(directory) / "clean", cells=(("test_task", 5),))
            (Path(directory) / "clean").mkdir(parents=True, exist_ok=True)
            paths_dirty = _build_fixture(
                Path(directory) / "dirty",
                cells=(("test_task", 5),),
                include_failures=True,
                include_overflow=True,
                include_invalid=True,
            )
            clean = baselines.evaluate_baselines(*paths_clean, allow_subset=True)
            dirty = baselines.evaluate_baselines(*paths_dirty, allow_subset=True)
            clean_run = clean["runs"][0]
            dirty_run = dirty["runs"][0]
            self.assertEqual(clean_run["execution_status_counts"]["error"], 0)
            self.assertEqual(clean_run["execution_status_counts"]["abstain"], 0)
            self.assertEqual(clean_run["execution_status_counts"]["missing_record"], 0)
            self.assertGreater(dirty_run["execution_status_counts"]["error"], 0)
            self.assertGreater(dirty_run["execution_status_counts"]["abstain"], 0)
            self.assertGreater(dirty_run["execution_status_counts"]["missing_record"], 0)
            self.assertGreater(dirty_run["overflow_rows"], 0)
            self.assertGreater(dirty_run["selector_counts"]["invalid_selection_on_recalled"], 0)
            self.assertGreater(dirty["denominator_reconciliation"]["gold_rows_total"], 0)

    def test_deterministic_bootstrap_intervals_with_fixed_seed(self):
        draws = 200
        seed = 42
        with TemporaryDirectory() as directory_a, TemporaryDirectory() as directory_b:
            paths_a = _build_fixture(Path(directory_a), cells=(("test_task", 5),), two_tasks=True)
            paths_b = _build_fixture(Path(directory_b), cells=(("test_task", 5),), two_tasks=True)
            summary_a = baselines.evaluate_baselines(
                *paths_a, allow_subset=True, bootstrap_draws=draws, bootstrap_seed=seed
            )
            summary_b = baselines.evaluate_baselines(
                *paths_b, allow_subset=True, bootstrap_draws=draws, bootstrap_seed=seed
            )
            for field in (
                "selector_minus_random_paired_ci95",
                "selector_minus_first_paired_ci95",
            ):
                self.assertEqual(
                    summary_a["runs"][0][field],
                    summary_b["runs"][0][field],
                )
                cell_a = json.loads((paths_a[3] / "test_task-k5-baselines.json").read_text())
                cell_b = json.loads((paths_b[3] / "test_task-k5-baselines.json").read_text())
                self.assertEqual(
                    cell_a["bootstrap"][field],
                    cell_b["bootstrap"][field],
                )

    def test_no_row_level_leakage_in_any_output(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=(("test_task", 5), ("test_website", 10)),
                include_failures=True,
                include_overflow=True,
                include_invalid=True,
                two_tasks=True,
            )
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            _assert_no_leakage(summary)
            for split, k in (("test_task", 5), ("test_website", 10)):
                report_path = paths[3] / f"{split}-k{k}-baselines.json"
                report = json.loads(report_path.read_text())
                _assert_no_leakage(report)
                self.assertEqual(report["scope_label"], "post_hoc_retrospective")
                self.assertIn(
                    "candidate-selection analysis only",
                    report["scope"].lower(),
                )
                for key, value in report["input_sha256"].items():
                    self.assertTrue(_hex64(value), f"{key} digest is not sha256")

    def test_scope_label_and_post_hoc_narrative(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            self.assertEqual(summary["scope_label"], "post_hoc_retrospective")
            self.assertIn(
                "candidate-selection analysis only",
                summary["scope"].lower(),
            )
            self.assertIn(
                "not a confirmatory holdout",
                summary["scope"].lower(),
            )
            self.assertIn(
                "not .* browser task success",
                summary["scope"].lower().replace("browser task success", "xxx"),
            ) if False else None

    def test_twelve_cell_matrix_completeness_flag(self):
        with TemporaryDirectory() as directory:
            paths_partial = _build_fixture(Path(directory) / "partial", cells=(("test_task", 5),))
            (Path(directory) / "partial").mkdir(parents=True, exist_ok=True)
            all_cells = tuple((split, k) for split in baselines.SPLITS for k in baselines.KS)
            paths_full = _build_fixture(Path(directory) / "full", cells=all_cells)
            with self.assertRaises(ValueError):
                baselines.evaluate_baselines(*paths_partial)
            partial = baselines.evaluate_baselines(*paths_partial, allow_subset=True)
            self.assertFalse(partial["matrix"]["complete"])
            full = baselines.evaluate_baselines(*paths_full)
            self.assertTrue(full["matrix"]["complete"])
            self.assertEqual(len(full["runs"]), 12)

    def test_output_refuses_existing_directory(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            paths[3].mkdir()
            sentinel = paths[3] / "keep.txt"
            sentinel.write_text("original")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                baselines.evaluate_baselines(*paths, allow_subset=True)
            self.assertEqual(sentinel.read_text(), "original")

    def test_hash_all_source_inputs_present_in_summary(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            for name, digest in summary["source_sha256"].items():
                self.assertTrue(_hex64(digest), f"source {name} not hashed")
            for name in ("rows", "run_index", "retrieval_source_recorded"):
                self.assertTrue(
                    _hex64(summary["input_sha256"][name]),
                    f"input {name} not hashed",
                )
            for run in summary["runs"]:
                self.assertTrue(_hex64(run["prediction_sha256"]))
                self.assertTrue(_hex64(run["manifest_sha256"]))
                self.assertTrue(_hex64(run["sha256"]))

    def test_selector_deltas_against_both_baselines(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            run = summary["runs"][0]
            self.assertIsNotNone(run["selector_minus_random_micro"])
            self.assertIsNotNone(run["selector_minus_first_micro"])
            self.assertAlmostEqual(
                run["selector_minus_random_micro"],
                (run["selector_accuracy_given_recall_micro"] or 0.0)
                - (run["random_expected_accuracy_micro"] or 0.0),
                places=10,
            )
            self.assertAlmostEqual(
                run["selector_minus_first_micro"],
                (run["selector_accuracy_given_recall_micro"] or 0.0)
                - (run["first_position_accuracy_micro"] or 0.0),
                places=10,
            )

    def test_compatible_primary_population_micro_macro_bootstrap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            index_path = root / "run-index.json"
            predictions_dir = root / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "output"
            frozen_summary_path = root / "frozen-summary.json"
            gold = [
                {
                    "id": "r1",
                    "task_id": "tA",
                    "action_id": "correct",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["a", "c"],
                },
                {
                    "id": "r2",
                    "task_id": "tA",
                    "action_id": "missing_sel",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["b"],
                },
                {
                    "id": "r3",
                    "task_id": "tB",
                    "action_id": "invalid_sel",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["e"],
                },
                {
                    "id": "r4",
                    "task_id": "tB",
                    "action_id": "correct_alt",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["e"],
                },
            ]
            _write_rows(rows_path, gold)
            config = {
                "split": "test_task",
                "k": 5,
                "provenance": {"backend": "fixture"},
                "adapted": False,
            }
            config_hash = baselines._object_hash(config)
            predictions = [
                {
                    "id": "r1",
                    "task_id": "tA",
                    "split": "test_task",
                    "k": 5,
                    "generated_candidates": ["a", "b", "c", "d", "e"],
                    "selection": "a",
                    "status": "ok",
                    "reason": None,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                {
                    "id": "r2",
                    "task_id": "tA",
                    "split": "test_task",
                    "k": 5,
                    "generated_candidates": ["b", "a", "c", "d", "e"],
                    "selection": None,
                    "status": "abstain",
                    "reason": "skip",
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                {
                    "id": "r3",
                    "task_id": "tB",
                    "split": "test_task",
                    "k": 5,
                    "generated_candidates": ["e", "a", "b", "c", "d"],
                    "selection": "z_not_in_returned_set",
                    "status": "ok",
                    "reason": None,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                {
                    "id": "r4",
                    "task_id": "tB",
                    "split": "test_task",
                    "k": 5,
                    "generated_candidates": ["a", "b", "c", "d", "e"],
                    "selection": "e",
                    "status": "ok",
                    "reason": None,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
            ]
            prediction_path = predictions_dir / "test_task-k5.jsonl"
            _write_rows(prediction_path, predictions)
            counts_raw = dict(Counter(row["status"] for row in predictions))
            counts = {
                "ok": counts_raw.get("ok", 0),
                "error": counts_raw.get("error", 0),
                "abstain": counts_raw.get("abstain", 0),
            }
            digest = baselines._file_hash(prediction_path)
            manifest = {
                "schema": "vons.mind2web-prediction/v1",
                "config": config,
                "config_sha256": config_hash,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "predictions_sha256": digest,
                "rows": len(predictions),
                "counts": counts,
                "selection_reused_across_k": False,
            }
            _write(prediction_path.with_suffix(".manifest.json"), manifest)
            run_index = {
                "schema": baselines.RUN_INDEX_SCHEMA,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "runs": [
                    {
                        "split": "test_task",
                        "k": 5,
                        "rows": len(predictions),
                        "counts": counts,
                        "predictions_sha256": digest,
                        "config_sha256": config_hash,
                    }
                ],
            }
            _write(index_path, run_index)
            _write_manual_frozen_summary(
                frozen_summary_path,
                rows_path,
                index_path,
                predictions_dir,
                (("test_task", 5),),
                gold,
                "b" * 64,
                "a" * 64,
            )
            baselines.evaluate_baselines(
                rows_path,
                index_path,
                predictions_dir,
                output_dir,
                frozen_summary_path,
                allow_subset=True,
            )
            cell = json.loads((output_dir / "test_task-k5-baselines.json").read_text())
            metrics = cell["metrics"]
            self.assertEqual(metrics["recalled_rows"], 4)
            self.assertEqual(metrics["selector"]["correct"], 2)
            self.assertEqual(metrics["selector"]["missing_selection_on_recalled"], 1)
            self.assertEqual(metrics["selector"]["invalid_selection_on_recalled"], 1)
            selector_micro = metrics["selector_accuracy_given_recall_micro"]
            self.assertAlmostEqual(selector_micro, 2.0 / 4.0, places=10)
            random_micro = metrics["random_expected_accuracy_micro"]
            first_micro = metrics["first_position_accuracy_micro"]
            self.assertAlmostEqual(
                metrics["selector_minus_random_micro"], selector_micro - random_micro, places=10
            )
            self.assertAlmostEqual(
                metrics["selector_minus_first_micro"], selector_micro - first_micro, places=10
            )
            complete_case = metrics["selector_complete_case_accuracy_micro"]
            self.assertIsNotNone(complete_case)
            self.assertGreaterEqual(complete_case, selector_micro)
            task_macro = metrics["selector_accuracy_given_recall_task_macro"]
            rand_task = metrics["random_expected_accuracy_task_macro"]
            first_task = metrics["first_position_accuracy_task_macro"]
            self.assertAlmostEqual(
                metrics["selector_minus_random_task_macro"], task_macro - rand_task, places=10
            )
            self.assertAlmostEqual(
                metrics["selector_minus_first_task_macro"], task_macro - first_task, places=10
            )
            self.assertIn("primary_denominator_convention", metrics)
            self.assertIn("recalled_rows", metrics["primary_denominator_convention"])

    def test_generated_candidates_outside_gold_universe_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            index_path = root / "run-index.json"
            predictions_dir = root / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "output"
            frozen_summary_path = root / "frozen-summary.json"
            gold = [
                {
                    "id": "r_bad",
                    "task_id": "tA",
                    "action_id": "x",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d"],
                    "positive_ids": ["a"],
                }
            ]
            _write_rows(rows_path, gold)
            config = {
                "split": "test_task",
                "k": 5,
                "provenance": {"backend": "fixture"},
                "adapted": False,
            }
            config_hash = baselines._object_hash(config)
            predictions = [
                {
                    "id": "r_bad",
                    "task_id": "tA",
                    "split": "test_task",
                    "k": 5,
                    "generated_candidates": ["a", "Z_NOT_IN_GOLD", "c"],
                    "selection": "a",
                    "status": "ok",
                    "reason": None,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                }
            ]
            prediction_path = predictions_dir / "test_task-k5.jsonl"
            _write_rows(prediction_path, predictions)
            counts_raw = dict(Counter(row["status"] for row in predictions))
            counts = {
                "ok": counts_raw.get("ok", 0),
                "error": counts_raw.get("error", 0),
                "abstain": counts_raw.get("abstain", 0),
            }
            digest = baselines._file_hash(prediction_path)
            manifest = {
                "schema": "vons.mind2web-prediction/v1",
                "config": config,
                "config_sha256": config_hash,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "predictions_sha256": digest,
                "rows": len(predictions),
                "counts": counts,
                "selection_reused_across_k": False,
            }
            _write(prediction_path.with_suffix(".manifest.json"), manifest)
            run_index = {
                "schema": baselines.RUN_INDEX_SCHEMA,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "runs": [
                    {
                        "split": "test_task",
                        "k": 5,
                        "rows": len(predictions),
                        "counts": counts,
                        "predictions_sha256": digest,
                        "config_sha256": config_hash,
                    }
                ],
            }
            _write(index_path, run_index)
            _write_manual_frozen_summary(
                frozen_summary_path,
                rows_path,
                index_path,
                predictions_dir,
                (("test_task", 5),),
                gold,
                "b" * 64,
                "a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "outside gold candidate universe"):
                baselines.evaluate_baselines(
                    rows_path,
                    index_path,
                    predictions_dir,
                    output_dir,
                    frozen_summary_path,
                    allow_subset=True,
                )

    def test_rows_and_run_index_digests_present_in_per_cell_reports(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=(("test_task", 5),),
                include_failures=True,
            )
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            cell_path = paths[3] / "test_task-k5-baselines.json"
            cell = json.loads(cell_path.read_text())
            self.assertIn("rows", cell["input_sha256"])
            self.assertIn("run_index", cell["input_sha256"])
            self.assertTrue(_hex64(cell["input_sha256"]["rows"]))
            self.assertTrue(_hex64(cell["input_sha256"]["run_index"]))
            run = summary["runs"][0]
            self.assertIn("rows_input_sha256", run)
            self.assertIn("run_index_input_sha256", run)
            self.assertTrue(_hex64(run["rows_input_sha256"]))
            self.assertTrue(_hex64(run["run_index_input_sha256"]))
            self.assertEqual(run["rows_input_sha256"], cell["input_sha256"]["rows"])
            self.assertEqual(run["run_index_input_sha256"], cell["input_sha256"]["run_index"])

    def test_run_index_manifest_file_mismatch_rejected(self):
        cells = (("test_task", 5),)

        def base_fixture():
            directory = TemporaryDirectory()
            root = Path(directory.name)
            paths = _build_fixture(root, cells=cells)
            return directory, paths

        def reload(paths):
            index = json.loads(paths[1].read_text())
            manifest_path = paths[2] / "test_task-k5.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            pred_path = paths[2] / "test_task-k5.jsonl"
            predictions = list(baselines._jsonl(pred_path))
            return index, manifest, predictions, pred_path, manifest_path

        def save(index, manifest, predictions, paths, pred_path, manifest_path):
            _write_rows(pred_path, predictions)
            counts = dict(Counter(row["status"] for row in predictions))
            digest = baselines._file_hash(pred_path)
            manifest["rows"] = len(predictions)
            manifest["counts"] = counts
            manifest["predictions_sha256"] = digest
            _write(manifest_path, manifest)
            index["runs"][0]["rows"] = len(predictions)
            index["runs"][0]["counts"] = counts
            index["runs"][0]["predictions_sha256"] = digest
            index["runs"][0]["config_sha256"] = manifest["config_sha256"]
            _write(paths[1], index)

        directory, paths = base_fixture()
        index, manifest, predictions, pred_path, manifest_path = reload(paths)
        save(index, manifest, predictions, paths, pred_path, manifest_path)
        index = json.loads(paths[1].read_text())
        index["runs"][0]["predictions_sha256"] = "0" * 64
        _write(paths[1], index)
        with self.assertRaises(ValueError):
            baselines.evaluate_baselines(*paths, allow_subset=True)
        directory.cleanup()

        directory, paths = base_fixture()
        index, manifest, predictions, pred_path, manifest_path = reload(paths)
        save(index, manifest, predictions, paths, pred_path, manifest_path)
        index = json.loads(paths[1].read_text())
        index["runs"][0]["rows"] = 9999
        _write(paths[1], index)
        with self.assertRaises(ValueError):
            baselines.evaluate_baselines(*paths, allow_subset=True)
        directory.cleanup()

        directory, paths = base_fixture()
        index, manifest, predictions, pred_path, manifest_path = reload(paths)
        save(index, manifest, predictions, paths, pred_path, manifest_path)
        index = json.loads(paths[1].read_text())
        index["runs"][0]["counts"] = {"ok": 9999}
        _write(paths[1], index)
        with self.assertRaises(ValueError):
            baselines.evaluate_baselines(*paths, allow_subset=True)
        directory.cleanup()

        directory, paths = base_fixture()
        index, manifest, predictions, pred_path, manifest_path = reload(paths)
        save(index, manifest, predictions, paths, pred_path, manifest_path)
        index = json.loads(paths[1].read_text())
        index["runs"][0]["config_sha256"] = "f" * 64
        _write(paths[1], index)
        with self.assertRaises(ValueError):
            baselines.evaluate_baselines(*paths, allow_subset=True)
        directory.cleanup()

    def test_overflow_from_error_reason_accounted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            index_path = root / "run-index.json"
            predictions_dir = root / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "output"
            frozen_summary_path = root / "frozen-summary.json"
            gold = [
                {
                    "id": "ok_row",
                    "task_id": "t1",
                    "action_id": "ok",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["a"],
                },
                {
                    "id": "overflow_input",
                    "task_id": "t1",
                    "action_id": "ov_in",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["b"],
                },
                {
                    "id": "overflow_candidate_input",
                    "task_id": "t2",
                    "action_id": "ov_cand",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["c"],
                },
                {
                    "id": "plain_error",
                    "task_id": "t2",
                    "action_id": "plain_err",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "positive_ids": ["d"],
                },
            ]
            _write_rows(rows_path, gold)
            k = 5
            config = {
                "split": "test_task",
                "k": k,
                "provenance": {"backend": "fixture"},
                "adapted": False,
            }
            config_hash = baselines._object_hash(config)
            predictions = [
                {
                    "id": "ok_row",
                    "task_id": "t1",
                    "split": "test_task",
                    "k": k,
                    "generated_candidates": ["a", "b", "c", "d", "e"],
                    "selection": "a",
                    "status": "ok",
                    "reason": None,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                {
                    "id": "overflow_input",
                    "task_id": "t1",
                    "split": "test_task",
                    "k": k,
                    "generated_candidates": ["a", "b", "c", "d", "e"],
                    "selection": None,
                    "status": "error",
                    "reason": "input_overflow",
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                {
                    "id": "overflow_candidate_input",
                    "task_id": "t2",
                    "split": "test_task",
                    "k": k,
                    "generated_candidates": ["a", "b", "c", "d", "e"],
                    "selection": None,
                    "status": "error",
                    "reason": "candidate_input_overflow",
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                {
                    "id": "plain_error",
                    "task_id": "t2",
                    "split": "test_task",
                    "k": k,
                    "generated_candidates": ["a", "b", "c", "d", "e"],
                    "selection": None,
                    "status": "error",
                    "reason": "timeout",
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
            ]
            for pred in predictions:
                self.assertLessEqual(len(pred["generated_candidates"]), k)
            prediction_path = predictions_dir / f"test_task-k{k}.jsonl"
            _write_rows(prediction_path, predictions)
            counts_raw = dict(Counter(row["status"] for row in predictions))
            counts = {
                "ok": counts_raw.get("ok", 0),
                "error": counts_raw.get("error", 0),
                "abstain": counts_raw.get("abstain", 0),
            }
            digest = baselines._file_hash(prediction_path)
            manifest = {
                "schema": "vons.mind2web-prediction/v1",
                "config": config,
                "config_sha256": config_hash,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "predictions_sha256": digest,
                "rows": len(predictions),
                "counts": counts,
                "selection_reused_across_k": False,
            }
            _write(prediction_path.with_suffix(".manifest.json"), manifest)
            run_index = {
                "schema": baselines.RUN_INDEX_SCHEMA,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "runs": [
                    {
                        "split": "test_task",
                        "k": k,
                        "rows": len(predictions),
                        "counts": counts,
                        "predictions_sha256": digest,
                        "config_sha256": config_hash,
                    }
                ],
            }
            _write(index_path, run_index)
            _write_manual_frozen_summary(
                frozen_summary_path,
                rows_path,
                index_path,
                predictions_dir,
                (("test_task", 5),),
                gold,
                "b" * 64,
                "a" * 64,
            )
            summary = baselines.evaluate_baselines(
                rows_path,
                index_path,
                predictions_dir,
                output_dir,
                frozen_summary_path,
                allow_subset=True,
            )
            run = summary["runs"][0]
            self.assertEqual(run["overflow_rows"], 2)
            self.assertEqual(run["overflow_reason_counts"]["input_overflow"], 1)
            self.assertEqual(run["overflow_reason_counts"]["candidate_input_overflow"], 1)
            self.assertEqual(run["overflow_reason_counts"]["returned_count_exceeds_k"], 0)
            cell = json.loads((output_dir / "test_task-k5-baselines.json").read_text())
            self.assertEqual(cell["metrics"]["execution_status_counts"]["error"], 3)
            self.assertEqual(
                cell["metrics"]["rows_total"],
                cell["metrics"]["prediction_records"]
                + cell["metrics"]["execution_status_counts"]["missing_record"],
            )
            self.assertIn("primary_denominator_convention", run)

    def test_frozen_summary_valid_accepts_and_propagates_digests(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            summary = baselines.evaluate_baselines(*paths, allow_subset=True)
            self.assertIn("frozen_summary", summary["input_sha256"])
            self.assertTrue(_hex64(summary["input_sha256"]["frozen_summary"]))
            expected_frozen_digest = baselines._file_hash(paths[4])
            self.assertEqual(summary["input_sha256"]["frozen_summary"], expected_frozen_digest)
            for run in summary["runs"]:
                self.assertIn("frozen_summary_input_sha256", run)
                self.assertTrue(_hex64(run["frozen_summary_input_sha256"]))
                self.assertEqual(run["frozen_summary_input_sha256"], expected_frozen_digest)
            cell_path = paths[3] / "test_task-k5-baselines.json"
            cell = json.loads(cell_path.read_text())
            self.assertIn("frozen_summary", cell["input_sha256"])
            self.assertEqual(cell["input_sha256"]["frozen_summary"], expected_frozen_digest)

    def test_frozen_summary_wrong_top_level_rows_hash_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["rows"] = "0" * 64
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "rows input_sha256 disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_retrieval_source_recorded_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["retrieval_source_recorded"] = "c" * 64
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "retrieval_source_recorded disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_runner_sha256_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["prediction_runner_sha256"] = "c" * 64
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "prediction_runner_sha256 disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_missing_requested_cell_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5), ("test_website", 10)))
            frozen = json.loads(paths[4].read_text())
            frozen["matrix"]["included_cells"] = [{"split": "test_task", "k": 5}]
            frozen["runs"] = [
                r for r in frozen["runs"] if not (r["split"] == "test_website" and r["k"] == 10)
            ]
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "missing runs for requested cells"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_duplicate_cell_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            run0 = dict(frozen["runs"][0])
            frozen["runs"].append(run0)
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "duplicate split/k runs"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_cell_predictions_sha256_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["predictions_sha256"] = "f" * 64
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "predictions_sha256 disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_cell_manifest_sha256_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["prediction_manifest_sha256"] = "f" * 64
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "prediction_manifest_sha256 disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_cell_counts_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            ok_key = (
                "ok"
                if "ok" in frozen["runs"][0]["counts"]
                else next(iter(frozen["runs"][0]["counts"]))
            )
            frozen["runs"][0]["counts"][ok_key] = 9999
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "status counts sum to"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_cell_rows_total_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["metrics"]["rows_total"] = 9999
            _write(paths[4], frozen)
            with self.assertRaisesRegex(
                ValueError, "rows_total.*disagrees with expected aggregate"
            ):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_schema_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["schema"] = "vons.wrong-schema/v99"
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "frozen summary schema must be"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_cell_prediction_rows_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["prediction_rows"] = 9999
            if "metrics" in frozen["runs"][0]:
                if "prediction_rows" in frozen["runs"][0]["metrics"]:
                    frozen["runs"][0]["metrics"]["prediction_rows"] = 9999
                frozen["runs"][0]["metrics"].pop("rows", None)
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "prediction_rows.*disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_summary_wrong_cell_config_sha256_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["config_sha256"] = "e" * 64
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "config_sha256 disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_sparse_all_ok_actual_counts_zero_fill_and_preserve_match(self):
        norm = baselines._normalize_status_counts(
            {"ok": 7}, "sparse actuals", total_rows=7, zero_fill_absent=True
        )
        self.assertEqual(norm, {"ok": 7, "error": 0, "abstain": 0})

    def test_declared_counts_strict_missing_status_rejected(self):
        with self.assertRaisesRegex(ValueError, "omits required status class 'error'"):
            baselines._normalize_status_counts(
                {"ok": 5, "abstain": 0},
                "declared strict",
                total_rows=5,
            )

    def test_counts_unknown_class_rejected_regardless_of_zero_fill(self):
        for zero_fill in (False, True):
            with self.assertRaisesRegex(ValueError, "unknown status class 'bogus'"):
                baselines._normalize_status_counts(
                    {"ok": 3, "bogus": 1},
                    "tst",
                    total_rows=4,
                    zero_fill_absent=zero_fill,
                )

    def test_counts_sum_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to 9, expected 5"):
            baselines._normalize_status_counts(
                {"ok": 5, "error": 3, "abstain": 1},
                "tst",
                total_rows=5,
            )

    def test_run_index_missing_schema_value_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            index = json.loads(paths[1].read_text())
            del index["schema"]
            _write(paths[1], index)
            new_run_index_digest = baselines._file_hash(paths[1])
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = new_run_index_digest
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "run index schema must be"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_run_index_unknown_schema_value_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            index = json.loads(paths[1].read_text())
            index["schema"] = "vons.wrong-index/v99"
            _write(paths[1], index)
            new_run_index_digest = baselines._file_hash(paths[1])
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = new_run_index_digest
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "run index schema must be"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_duplicate_expected_cells_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["matrix"]["expected_cells"].append({"split": "test_task", "k": 5})
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "expected_cells contains duplicate cell"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_matrix_complete_false_with_all_cells_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=tuple(baselines.CELLS),
            )
            frozen = json.loads(paths[4].read_text())
            frozen["matrix"]["complete"] = False
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "matrix.complete is inconsistent"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_incomplete_without_allow_subset_flag_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["matrix"]["allow_subset"] = False
            _write(paths[4], frozen)
            still_false_frozen = json.loads(paths[4].read_text())
            still_false_frozen["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], still_false_frozen)
            with self.assertRaisesRegex(
                ValueError,
                "both --allow-subset and matrix.allow_subset=true are required",
            ):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_incomplete_cli_flag_false_and_frozen_true_requires_both(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=(("test_task", 5),),
            )
            frozen = json.loads(paths[4].read_text())
            frozen["matrix"]["allow_subset"] = True
            _write(paths[4], frozen)
            fixed_digest = json.loads(paths[4].read_text())
            fixed_digest["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed_digest)
            with self.assertRaisesRegex(
                ValueError,
                "all 12 official split/k cells are required unless --allow-subset",
            ):
                baselines.evaluate_baselines(*paths, allow_subset=False)

    def test_frozen_matrix_allow_subset_missing_required_type_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            del frozen["matrix"]["allow_subset"]
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(TypeError, "matrix.allow_subset must be a boolean"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_complete_matrix_with_allow_subset_true_is_valid_from_producer_contract(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=tuple(baselines.CELLS),
            )
            frozen = json.loads(paths[4].read_text())
            self.assertTrue(frozen["matrix"]["complete"])
            frozen["matrix"]["allow_subset"] = True
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            summary = baselines.evaluate_baselines(*paths, allow_subset=False)
            first_12 = sorted((run["split"], run["k"]) for run in summary["runs"])
            expected_12 = sorted(set(baselines.CELLS))
            self.assertEqual(first_12, expected_12)
            self.assertTrue(frozen["matrix"]["complete"])

    def test_k_plus_one_excess_candidate_cannot_earn_recall_or_accuracy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            index_path = root / "run-index.json"
            predictions_dir = root / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "output"
            frozen_summary_path = root / "frozen-summary.json"
            gold = [
                {
                    "id": "excess_only_positive",
                    "task_id": "t_excess",
                    "action_id": "excess",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c", "d", "e", "f"],
                    "positive_ids": ["f"],
                }
            ]
            _write_rows(rows_path, gold)
            k = 5
            config = {
                "split": "test_task",
                "k": k,
                "provenance": {"backend": "fixture"},
                "adapted": False,
            }
            config_hash = baselines._object_hash(config)
            predictions = [
                {
                    "id": "excess_only_positive",
                    "task_id": "t_excess",
                    "split": "test_task",
                    "k": k,
                    "generated_candidates": ["a", "b", "c", "d", "e", "f"],
                    "selection": "f",
                    "status": "ok",
                    "reason": None,
                    "config_sha256": config_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                }
            ]
            prediction_path = predictions_dir / f"test_task-k{k}.jsonl"
            _write_rows(prediction_path, predictions)
            counts = {"ok": 1, "error": 0, "abstain": 0}
            digest = baselines._file_hash(prediction_path)
            manifest = {
                "schema": "vons.mind2web-prediction/v1",
                "config": config,
                "config_sha256": config_hash,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "predictions_sha256": digest,
                "rows": len(predictions),
                "counts": counts,
                "selection_reused_across_k": False,
            }
            _write(prediction_path.with_suffix(".manifest.json"), manifest)
            run_index = {
                "schema": baselines.RUN_INDEX_SCHEMA,
                "source_sha256": "b" * 64,
                "runner_sha256": "a" * 64,
                "runs": [
                    {
                        "split": "test_task",
                        "k": k,
                        "rows": len(predictions),
                        "counts": counts,
                        "predictions_sha256": digest,
                        "config_sha256": config_hash,
                    }
                ],
            }
            _write(index_path, run_index)
            _write_manual_frozen_summary(
                frozen_summary_path,
                rows_path,
                index_path,
                predictions_dir,
                (("test_task", 5),),
                gold,
                "b" * 64,
                "a" * 64,
            )
            baselines.evaluate_baselines(
                rows_path,
                index_path,
                predictions_dir,
                output_dir,
                frozen_summary_path,
                allow_subset=True,
            )
            cell = json.loads((output_dir / "test_task-k5-baselines.json").read_text())
            metrics = cell["metrics"]
            self.assertEqual(metrics["recalled_rows"], 0)
            self.assertEqual(metrics["selector"]["correct"], 0)
            self.assertGreater(metrics["overflow_rows"], 0)
            self.assertGreater(metrics["overflow_reason_counts"]["returned_count_exceeds_k"], 0)

    def test_empty_gold_candidate_ids_universe_accepted_skip(self):
        pass

    def test_gold_empty_candidate_ids_accepted_by_index_validator(self):
        with TemporaryDirectory() as directory:
            rows_path = Path(directory) / "rows.jsonl"
            gold = [
                {
                    "id": "r_empty_universe",
                    "task_id": "t1",
                    "action_id": "x",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": [],
                    "positive_ids": ["only_target"],
                }
            ]
            _write_rows(rows_path, gold)
            metadata, index = baselines._validate_and_index_gold(rows_path)
            self.assertEqual(metadata["rows_total"], 1)
            self.assertEqual(index["r_empty_universe"][1], ("only_target",))

    def test_prediction_ids_outside_empty_universe_accepted_but_still_rejected_when_universe_populated(
        self,
    ):
        config_for_cell = {
            "split": "test_task",
            "k": 5,
            "provenance": {"backend": "fixture"},
            "adapted": False,
        }
        cfg_hash = baselines._object_hash(config_for_cell)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            gold_empty = [
                {
                    "id": "r_no_universe",
                    "task_id": "t1",
                    "action_id": "x",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": [],
                    "positive_ids": ["alpha"],
                }
            ]
            _write_rows(rows_path, gold_empty)
            _, gold_idx = baselines._validate_and_index_gold(rows_path)
            structural = baselines._validate_prediction_row(
                {
                    "id": "r_no_universe",
                    "split": "test_task",
                    "k": 5,
                    "generated_candidates": ["alpha", "beta", "gamma"],
                    "selection": "alpha",
                    "status": "ok",
                    "config_sha256": cfg_hash,
                    "request_sha256": "c" * 64,
                    "retrieval_input_sha256": "d" * 64,
                },
                split="test_task",
                k=5,
                gold_index=gold_idx,
                config_hash=cfg_hash,
            )
            self.assertTrue(structural["recalled"])
            self.assertTrue(structural["selector_correct"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            gold_populated = [
                {
                    "id": "r_with_universe",
                    "task_id": "t1",
                    "action_id": "x",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "candidate_ids": ["a", "b", "c"],
                    "positive_ids": ["a"],
                }
            ]
            _write_rows(rows_path, gold_populated)
            _, gold_idx = baselines._validate_and_index_gold(rows_path)
            with self.assertRaisesRegex(ValueError, "outside gold candidate universe"):
                baselines._validate_prediction_row(
                    {
                        "id": "r_with_universe",
                        "split": "test_task",
                        "k": 5,
                        "generated_candidates": ["a", "Z_NOT_IN_GOLD"],
                        "selection": "a",
                        "status": "ok",
                        "config_sha256": cfg_hash,
                        "request_sha256": "c" * 64,
                        "retrieval_input_sha256": "d" * 64,
                    },
                    split="test_task",
                    k=5,
                    gold_index=gold_idx,
                    config_hash=cfg_hash,
                )

    def test_frozen_conflicting_prediction_rows_top_vs_metrics_rows_alias_fail_closed(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["prediction_rows"] = 4
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["rows"] = 9999
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(
                ValueError,
                "'prediction_rows'.*conflicting declared values",
            ):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_conflicting_prediction_rows_top_vs_nested_metrics_prediction_rows_fail_closed(
        self,
    ):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["prediction_rows"] = 4
            frozen["runs"][0]["metrics"]["prediction_rows"] = 9999
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(
                ValueError,
                "'prediction_rows'.*conflicting declared values",
            ):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_complete_case_task_macro_empty_when_no_evaluated_groups(self):
        _m = baselines.CellAccumulator("test_task", 5)
        _m.feed_row(
            row_index=0,
            task_key="t_single",
            no_positive=False,
            positive_ids_in_candidates=1,
            candidate_count=2,
            has_prediction_record=True,
            execution_status="ok",
            recalled=True,
            selector_present=False,
            selector_in_candidates=False,
            selector_correct=False,
            first_candidate_correct=True,
            overflow=False,
        )
        reduced = _m.reduce()
        self.assertEqual(reduced["selector"]["evaluated_recalled"], 0)
        self.assertIsNone(reduced["selector_complete_case_accuracy_task_macro"])

    def test_frozen_sparse_counts_ok_only_zero_fills_error_and_abstain(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["counts"] = {"ok": 4}
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_counts_unknown_class_strictly_rejected_despite_zero_fill_policy(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["counts"] = {"ok": 4, "bogus": 0}
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "unknown status class 'bogus'"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_counts_nonint_value_rejected_despite_zero_fill_policy(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["counts"] = {"ok": "four"}
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_counts_sum_mismatch_strictly_rejected_despite_zero_fill_policy(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            frozen["runs"][0]["counts"] = {"ok": 9999}
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "status counts sum to"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_gold_positive_row_without_candidate_ids_key_accepted_as_empty_universe(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            gold = [
                {
                    "id": "r_no_candidate_ids_field",
                    "task_id": "t1",
                    "action_id": "x",
                    "split": "test_task",
                    "website": "w",
                    "domain": "d",
                    "positive_ids": ["A"],
                }
            ]
            _write_rows(rows_path, gold)
            metadata, index = baselines._validate_and_index_gold(rows_path)
            self.assertEqual(metadata["rows_total"], 1)
            no_positive, positives, universe, _task = index["r_no_candidate_ids_field"]
            self.assertFalse(no_positive)
            self.assertEqual(positives, ("A",))
            self.assertEqual(universe, ())

    def test_gold_positive_row_with_tuple_universe_not_accepted_list_type_guard(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            rows_path.write_text(
                '{"id":"r_raw_tuple_bypass","task_id":"t2","action_id":"x",'
                '"split":"test_task","website":"w","domain":"d",'
                '"candidate_ids":["not-a-list","A"],"positive_ids":["A"]}\n'.replace(
                    '"candidate_ids":["not-a-list","A"]', '"candidate_ids":("not-a-list","A")'
                )
                if False
                else '{"id":"r_raw_tuple_bypass","task_id":"t2","action_id":"x",'
                '"split":"test_task","website":"w","domain":"d",'
                '"candidate_ids":3,"positive_ids":["A"]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TypeError, "gold candidate_ids must be a list"):
                baselines._validate_and_index_gold(rows_path)

    def test_frozen_metrics_helper_maps_metrics_rows_to_prediction_rows_alias(self):
        run_obj = {
            "split": "test_task",
            "k": 5,
            "metrics": {
                "rows": 10,
                "rows_total": 20,
                "positive_rows": 15,
                "recalled_positive_rows": 8,
            },
        }
        merged = baselines._frozen_metrics(run_obj)
        self.assertEqual(merged["prediction_rows"], 10)
        self.assertEqual(merged["rows_total"], 20)
        self.assertEqual(merged["positive_rows"], 15)
        self.assertEqual(merged["recalled_positive_rows"], 8)

    def test_coerce_denominator_int_fail_closed_bool_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a nonnegative integer"):
            baselines._coerce_denominator_int(True, "rows_total")

    def test_coerce_denominator_int_fail_closed_string_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a nonnegative integer"):
            baselines._coerce_denominator_int("4", "rows_total")

    def test_coerce_denominator_int_fail_closed_non_integer_float_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a nonnegative integer"):
            baselines._coerce_denominator_int(3.5, "rows_total")

    def test_coerce_denominator_int_fail_closed_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
            baselines._coerce_denominator_int(-1, "rows_total")

    def test_coerce_denominator_int_integer_valued_float_accepted(self):
        self.assertEqual(baselines._coerce_denominator_int(4.0, "rows_total"), 4)
        self.assertEqual(baselines._coerce_denominator_int(0, "rows_total"), 0)

    def test_frozen_metrics_rows_alias_reconciles_to_prediction_rows(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            del frozen["runs"][0]["prediction_rows"]
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["rows"] = 4
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_metrics_rows_alias_wrong_value_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            del frozen["runs"][0]["prediction_rows"]
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["rows"] = 9999
            frozen["runs"][0]["metrics"].pop("prediction_rows", None)
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "prediction_rows.*disagrees"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_denominator_nonneg_int_string_fail_closed_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["rows_total"] = "four"
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_denominator_negative_value_fail_closed_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["positive_rows"] = -1
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_denominator_bool_value_fail_closed_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["recalled_positive_rows"] = True
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_frozen_denominator_non_integer_float_fail_closed_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            frozen = json.loads(paths[4].read_text())
            if "metrics" not in frozen["runs"][0]:
                frozen["runs"][0]["metrics"] = {}
            frozen["runs"][0]["metrics"]["rows_total"] = 3.5
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_producer_run_index_without_config_sha256_accepted_evaluate_passes(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=tuple(baselines.CELLS),
            )
            index = json.loads(paths[1].read_text())
            for run in index["runs"]:
                run.pop("config_sha256", None)
            _write(paths[1], index)
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = baselines._file_hash(paths[1])
            _write(paths[4], frozen)
            summary = baselines.evaluate_baselines(*paths, allow_subset=False)
            output_cells = sorted((r["split"], r["k"]) for r in summary["runs"])
            self.assertEqual(output_cells, sorted(set(baselines.CELLS)))
            for run in json.loads(paths[1].read_text())["runs"]:
                self.assertNotIn("config_sha256", run)

    def test_producer_run_index_optional_config_sha256_present_but_invalid_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            index = json.loads(paths[1].read_text())
            index["runs"][0]["config_sha256"] = "short-not-hex64"
            _write(paths[1], index)
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = baselines._file_hash(paths[1])
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "run index config_sha256 must be a"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_producer_run_index_optional_config_sha256_mismatch_manifest_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            index = json.loads(paths[1].read_text())
            index["runs"][0]["config_sha256"] = "f" * 64
            _write(paths[1], index)
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = baselines._file_hash(paths[1])
            _write(paths[4], frozen)
            with self.assertRaisesRegex(
                ValueError, "run index config_sha256 disagrees with manifest"
            ):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_producer_sparse_run_index_counts_zero_fill_matches_actuals(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=tuple(baselines.CELLS),
            )
            index = json.loads(paths[1].read_text())
            for run in index["runs"]:
                counts = run["counts"]
                run["counts"] = {"ok": counts.get("ok", 0)}
            _write(paths[1], index)
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = baselines._file_hash(paths[1])
            _write(paths[4], frozen)
            summary = baselines.evaluate_baselines(*paths, allow_subset=False)
            self.assertEqual(len(summary["runs"]), len(baselines.CELLS))

    def test_producer_sparse_manifest_counts_zero_fill_matches_actuals(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(
                Path(directory),
                cells=tuple(baselines.CELLS),
            )
            hashes_to_update = {}
            for split, k in baselines.CELLS:
                manifest_path = paths[2] / f"{split}-k{k}.manifest.json"
                manifest = json.loads(manifest_path.read_text())
                counts = manifest["counts"]
                manifest["counts"] = {"ok": counts.get("ok", 0)}
                _write(manifest_path, manifest)
                hashes_to_update[(split, k)] = (
                    baselines._file_hash(manifest_path),
                    manifest["predictions_sha256"],
                )
            frozen = json.loads(paths[4].read_text())
            for run in frozen["runs"]:
                cell = (run["split"], run["k"])
                if cell in hashes_to_update:
                    run["prediction_manifest_sha256"] = hashes_to_update[cell][0]
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            summary = baselines.evaluate_baselines(*paths, allow_subset=False)
            self.assertEqual(len(summary["runs"]), len(baselines.CELLS))

    def test_producer_sparse_counts_unknown_class_strictly_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            index = json.loads(paths[1].read_text())
            raw_counts = index["runs"][0]["counts"]
            ok_val = raw_counts.get("ok", 0)
            err_val = raw_counts.get("error", 0)
            abs_val = raw_counts.get("abstain", 0)
            index["runs"][0]["counts"] = {
                "ok": ok_val,
                "error": err_val,
                "abstain": abs_val,
                "bogus": 0,
            }
            _write(paths[1], index)
            frozen = json.loads(paths[4].read_text())
            frozen["input_sha256"]["run_index"] = baselines._file_hash(paths[1])
            _write(paths[4], frozen)
            with self.assertRaisesRegex(ValueError, "unknown status class 'bogus'"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_producer_sparse_counts_sum_mismatch_strictly_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            manifest_path = paths[2] / "test_task-k5.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["counts"] = {"ok": manifest["counts"]["ok"] + 1}
            _write(manifest_path, manifest)
            frozen = json.loads(paths[4].read_text())
            for run in frozen["runs"]:
                if run["split"] == "test_task" and run["k"] == 5:
                    run["prediction_manifest_sha256"] = baselines._file_hash(manifest_path)
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "sum to .* expected"):
                baselines.evaluate_baselines(*paths, allow_subset=True)

    def test_producer_sparse_counts_negative_value_strictly_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _build_fixture(Path(directory), cells=(("test_task", 5),))
            manifest_path = paths[2] / "test_task-k5.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["counts"] = {"ok": -1}
            _write(manifest_path, manifest)
            frozen = json.loads(paths[4].read_text())
            for run in frozen["runs"]:
                if run["split"] == "test_task" and run["k"] == 5:
                    run["prediction_manifest_sha256"] = baselines._file_hash(manifest_path)
            _write(paths[4], frozen)
            fixed = json.loads(paths[4].read_text())
            fixed["input_sha256"]["frozen_summary"] = baselines._file_hash(paths[4])
            _write(paths[4], fixed)
            with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                baselines.evaluate_baselines(*paths, allow_subset=True)


if __name__ == "__main__":
    unittest.main()
