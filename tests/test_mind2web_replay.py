"""Small fabricated inputs only; no model or official-corpus replay in tests."""

import csv
import hashlib
import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _load_replay():
    path = Path(__file__).parents[1] / "tools/replay_mind2web_evaluation.py"
    spec = importlib.util.spec_from_file_location("vons_mind2web_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load replay tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = _load_replay()


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _fixture(root, cells=(("test_task", 5),)):
    rows_path = root / "rows.jsonl"
    index_path = root / "run-index.json"
    predictions_dir = root / "predictions"
    predictions_dir.mkdir()
    gold = []
    for split in dict.fromkeys(split for split, _ in cells):
        for suffix in ("accepted", "abstained", "errored", "missing"):
            gold.append(
                {
                    "id": f"{split}-{suffix}",
                    "task_id": f"{split}-task-{suffix}",
                    "action_id": suffix,
                    "split": split,
                    "candidate_ids": ["x", "y"],
                    "positive_ids": ["x"],
                }
            )
    _write_rows(rows_path, gold)
    runner_hash = replay._hash(replay._source_paths()["tools/predict_mind2web.py"])
    index = {"source_sha256": "e" * 64, "runner_sha256": runner_hash, "runs": []}
    for split, k in cells:
        config = {
            "split": split,
            "k": k,
            "provenance": {"backend": "fixture"},
            "adapted": False,
            "question": "Fixture question",
        }
        config_hash = replay._object_hash(config)
        predictions = []
        for suffix, status, reason in (
            ("accepted", "ok", None),
            ("abstained", "abstain", "confidence_below_threshold"),
            ("errored", "error", "input_overflow"),
        ):
            predictions.append(
                {
                    "id": f"{split}-{suffix}",
                    "task_id": f"{split}-task-{suffix}",
                    "split": split,
                    "k": k,
                    "generated_candidates": ["x", "y"],
                    "selection": "x" if status == "ok" else None,
                    "status": status,
                    "reason": reason,
                    "config_sha256": config_hash,
                    "request_sha256": "a" * 64,
                    "retrieval_input_sha256": "b" * 64,
                }
            )
        prediction_path = predictions_dir / f"{split}-k{k}.jsonl"
        _write_rows(prediction_path, predictions)
        counts = dict(Counter(row["status"] for row in predictions))
        digest = replay._hash(prediction_path)
        manifest = {
            "schema": "vons.mind2web-prediction/v1",
            "config": config,
            "config_sha256": config_hash,
            "source_sha256": index["source_sha256"],
            "runner_sha256": runner_hash,
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
            }
        )
    _write(index_path, index)
    return rows_path, index_path, predictions_dir, root / "output"


def _refresh_prediction_hashes(index_path, prediction_path):
    digest = replay._hash(prediction_path)
    manifest_path = prediction_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["predictions_sha256"] = digest
    _write(manifest_path, manifest)
    index = json.loads(index_path.read_text())
    for run in index["runs"]:
        if prediction_path.name == f"{run['split']}-k{run['k']}.jsonl":
            run["predictions_sha256"] = digest
    _write(index_path, index)


class Mind2WebReplayTests(unittest.TestCase):
    def test_observed_outcomes_missing_denominator_and_real_bootstrap(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            summary = replay.replay(*paths, allow_subset=True)
            run = summary["runs"][0]
            self.assertEqual(run["counts"], {"ok": 1, "abstain": 1, "error": 1})
            self.assertEqual(
                run["execution_outcomes"], {"accepted": 1, "error": 1, "abstain": 1, "missing": 1}
            )
            self.assertEqual(run["acceptance_coverage"], 0.25)
            self.assertEqual(run["prediction_coverage"], 0.75)
            self.assertEqual(run["coverage_denominator_rows"], 4)
            self.assertEqual(run["missing_prediction_record_count"], 1)
            self.assertEqual(
                run["execution_failure_counts"],
                {"confidence_below_threshold": 1, "input_overflow": 1},
            )
            self.assertIn({"reason": None, "count": 1}, run["execution_reason_counts"])
            self.assertEqual(run["metrics"]["correct_selections"], 1)
            self.assertEqual(run["metrics"]["missing_predictions"], 2)
            self.assertEqual(run["bootstrap"]["draws"], 1000)
            self.assertEqual(run["bootstrap"]["seed"], 7)
            self.assertFalse(summary["matrix"]["complete"])
            output = paths[-1]
            report_path = output / "test_task-k5-metrics.json"
            self.assertEqual(run["sha256"], hashlib.sha256(report_path.read_bytes()).hexdigest())
            report = json.loads(report_path.read_text())
            self.assertEqual(report["input_sha256"]["rows"], replay._hash(paths[0]))
            self.assertEqual(report["source_sha256"], summary["source_sha256"])
            self.assertEqual(summary["input_sha256"]["run_index"], replay._hash(paths[1]))
            table_path = output / "summary.csv"
            with table_path.open(newline="") as handle:
                table_row = next(csv.DictReader(handle))
            self.assertEqual(table_row["coverage"], "0.25")
            self.assertEqual(table_row["missing_count"], "1")
            self.assertEqual(table_row["error_count"], "1")
            self.assertEqual(summary["table"]["sha256"], replay._hash(table_path))

    def test_undefined_metric_cells_remain_blank_in_csv_and_null_in_json(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            rows = _read_rows(paths[0])
            for row in rows:
                row["positive_ids"] = []
                row["no_positive"] = True
            _write_rows(paths[0], rows)
            summary = replay.replay(*paths, allow_subset=True)
            self.assertIsNone(summary["runs"][0]["metrics"]["candidate_recall_step_micro"])
            with (paths[-1] / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            for field in (
                "recall_micro",
                "recall_task_macro",
                "taskmacroCIlo",
                "taskmacroCIhi",
                "conditional_accuracy_micro",
                "conditional_accuracy_task_macro",
            ):
                self.assertEqual(row[field], "")
            self.assertEqual(row["positive_rows"], "0")

    def test_default_requires_complete_matrix_and_tiny_twelve_cells_succeed(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "all 12"):
                replay.replay(*paths)
            self.assertFalse(paths[-1].exists())
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory), replay.CELLS)
            summary = replay.replay(*paths)
            self.assertTrue(summary["matrix"]["complete"])
            self.assertEqual(len(summary["runs"]), 12)
            self.assertEqual(len(list(paths[-1].glob("*-metrics.json"))), 12)

    def test_duplicate_unknown_and_bool_cells_fail_closed(self):
        for mutation in ("duplicate", "unknown_split", "unknown_k", "bool_k"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                paths = _fixture(Path(directory))
                index = json.loads(paths[1].read_text())
                if mutation == "duplicate":
                    index["runs"].append(index["runs"][0])
                elif mutation == "unknown_split":
                    index["runs"][0]["split"] = "train"
                else:
                    index["runs"][0]["k"] = True if mutation == "bool_k" else 8
                _write(paths[1], index)
                with self.assertRaises(ValueError):
                    replay.replay(*paths, allow_subset=True)
                self.assertFalse(paths[-1].exists())

    def test_tampered_predictions_fail_index_hash_check(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            prediction_path = paths[2] / "test_task-k5.jsonl"
            with prediction_path.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "prediction file hash"):
                replay.replay(*paths, allow_subset=True)

    def test_row_count_and_raw_status_claims_are_checked(self):
        for field in ("rows", "counts"):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                paths = _fixture(Path(directory))
                index = json.loads(paths[1].read_text())
                index["runs"][0][field] = 4 if field == "rows" else {"ok": 3}
                _write(paths[1], index)
                with self.assertRaisesRegex(ValueError, "row count|raw status counts"):
                    replay.replay(*paths, allow_subset=True)

    def test_manifest_config_and_source_identity_hashes_are_checked(self):
        for field in ("config_split", "config_hash", "source_hash", "runner_hash", "reuse"):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                paths = _fixture(Path(directory))
                manifest_path = paths[2] / "test_task-k5.manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if field == "config_split":
                    manifest["config"]["split"] = "test_domain"
                elif field == "config_hash":
                    manifest["config_sha256"] = "0" * 64
                elif field == "source_hash":
                    manifest["source_sha256"] = "0" * 64
                elif field == "runner_hash":
                    manifest["runner_sha256"] = "0" * 64
                else:
                    manifest["selection_reused_across_k"] = True
                _write(manifest_path, manifest)
                with self.assertRaises(ValueError):
                    replay.replay(*paths, allow_subset=True)

    def test_prediction_row_identity_config_and_status_consistency(self):
        mutations = {
            "split": "test_domain",
            "k": 10,
            "task_id": "another-task",
            "config_sha256": "0" * 64,
            "request_sha256": "bad",
            "status": "error",
            "selection": "outside",
            "generated_candidates": ["outside"],
        }
        for key, value in mutations.items():
            with self.subTest(key=key), TemporaryDirectory() as directory:
                paths = _fixture(Path(directory))
                prediction_path = paths[2] / "test_task-k5.jsonl"
                rows = _read_rows(prediction_path)
                rows[0][key] = value
                _write_rows(prediction_path, rows)
                _refresh_prediction_hashes(paths[1], prediction_path)
                with self.assertRaises(ValueError):
                    replay.replay(*paths, allow_subset=True)

    def test_unknown_or_duplicate_prediction_ids_fail_closed(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), TemporaryDirectory() as directory:
                paths = _fixture(Path(directory))
                prediction_path = paths[2] / "test_task-k5.jsonl"
                rows = _read_rows(prediction_path)
                rows[1]["id"] = rows[0]["id"] if duplicate else "unknown"
                _write_rows(prediction_path, rows)
                _refresh_prediction_hashes(paths[1], prediction_path)
                with self.assertRaisesRegex(ValueError, "unknown or duplicated"):
                    replay.replay(*paths, allow_subset=True)

    def test_missing_reason_is_counted_without_inventing_a_failure_reason(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            prediction_path = paths[2] / "test_task-k5.jsonl"
            rows = _read_rows(prediction_path)
            del rows[-1]["reason"]
            _write_rows(prediction_path, rows)
            _refresh_prediction_hashes(paths[1], prediction_path)
            run = replay.replay(*paths, allow_subset=True)["runs"][0]
            self.assertEqual(run["missing_reason_fields"], 1)
            self.assertNotIn("input_overflow", run["execution_failure_counts"])
            self.assertIn({"reason": None, "count": 2}, run["execution_reason_counts"])

    def test_replay_refuses_existing_evidence(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            paths[-1].mkdir()
            sentinel = paths[-1] / "keep.txt"
            sentinel.write_text("original")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                replay.replay(*paths, allow_subset=True)
            self.assertEqual(sentinel.read_text(), "original")

    def test_input_mutation_during_evaluation_discards_all_staged_reports(self):
        for which in ("gold", "prediction", "manifest", "index"):
            with self.subTest(which=which), TemporaryDirectory() as directory:
                paths = _fixture(Path(directory))
                evaluator = replay._load_evaluator(
                    replay._source_paths()["tools/evaluate_mind2web.py"]
                )
                evaluate = evaluator.evaluate_files
                targets = {
                    "gold": paths[0],
                    "index": paths[1],
                    "prediction": paths[2] / "test_task-k5.jsonl",
                    "manifest": paths[2] / "test_task-k5.manifest.json",
                }

                def mutate(*args, evaluate=evaluate, target=targets[which], **kwargs):
                    report = evaluate(*args, **kwargs)
                    with target.open("a") as handle:
                        handle.write("\n")
                    return report

                evaluator.evaluate_files = mutate
                with (
                    patch.object(replay, "_load_evaluator", return_value=evaluator),
                    self.assertRaisesRegex(ValueError, "source mutation"),
                ):
                    replay.replay(*paths, allow_subset=True)
                self.assertFalse(paths[-1].exists())

    def test_code_mutation_during_replay_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _fixture(root)
            sources = replay._source_paths()
            stand_in = root / "source-snapshot.py"
            stand_in.write_text("# Initial source snapshot\n")
            sources["tools/replay_mind2web_evaluation.py"] = stand_in
            evaluator = replay._load_evaluator(sources["tools/evaluate_mind2web.py"])
            evaluate = evaluator.evaluate_files

            def mutate(*args, **kwargs):
                report = evaluate(*args, **kwargs)
                stand_in.write_text("# Changed during replay\n")
                return report

            evaluator.evaluate_files = mutate
            with (
                patch.object(replay, "_source_paths", return_value=sources),
                patch.object(replay, "_load_evaluator", return_value=evaluator),
                self.assertRaisesRegex(ValueError, "source mutation"),
            ):
                replay.replay(*paths, allow_subset=True)
            self.assertFalse(paths[-1].exists())

    def test_incompatible_cell_configurations_are_rejected(self):
        with TemporaryDirectory() as directory:
            paths = _fixture(Path(directory), (("test_task", 5), ("test_task", 10)))
            prediction_path = paths[2] / "test_task-k10.jsonl"
            manifest_path = prediction_path.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["config"]["provenance"]["backend"] = "different-model"
            manifest["config_sha256"] = replay._object_hash(manifest["config"])
            _write(manifest_path, manifest)
            rows = _read_rows(prediction_path)
            for row in rows:
                row["config_sha256"] = manifest["config_sha256"]
            _write_rows(prediction_path, rows)
            _refresh_prediction_hashes(paths[1], prediction_path)
            with self.assertRaisesRegex(ValueError, "incompatible frozen"):
                replay.replay(*paths, allow_subset=True)


if __name__ == "__main__":
    unittest.main()
