import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_runner():
    path = Path(__file__).parents[1] / "tools" / "evaluate_mind2web.py"
    spec = importlib.util.spec_from_file_location("vons_mind2web_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Mind2Web runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class Mind2WebRunnerTests(unittest.TestCase):
    def test_runner_preserves_missing_prediction_rows(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            predictions_path = root / "predictions.jsonl"
            rows_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a", "candidate_ids": [], "positive_ids": ["target"]}),
                        json.dumps({"id": "b", "candidate_ids": [], "positive_ids": ["target"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps(
                    {"id": "a", "generated_candidates": ["target"], "selection": "target"}
                )
                + "\n",
                encoding="utf-8",
            )

            report = runner.evaluate_files(rows_path, predictions_path, [1, 5])

            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["prediction_rows"], 1)
            self.assertEqual(report["metrics_by_k"][0]["candidate_recall"], 0.5)
            self.assertIsNone(report["metrics_by_k"][0]["selection_accuracy_given_recall"])
            self.assertFalse(report["selection_evaluation"]["selection_metrics_valid"])
            self.assertEqual(len(report["input_sha256"]["rows"]), 64)
            self.assertEqual(len(report["input_sha256"]["predictions"]), 64)
            self.assertIn("task_bootstrap_by_k", report)

            single_k_report = runner.evaluate_files(rows_path, predictions_path, [1])
            self.assertEqual(single_k_report["metrics_by_k"][0]["selection_accuracy_given_recall"], 1.0)

    def test_runner_requires_prediction_k_to_match_single_requested_k(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            predictions_path = root / "predictions.jsonl"
            rows_path.write_text(
                json.dumps({"id": "a", "candidate_ids": [], "positive_ids": ["target"]}) + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps({"id": "a", "k": 5, "generated_candidates": ["target"], "selection": "target"})
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exactly that single k"):
                runner.evaluate_files(rows_path, predictions_path, [10])

    def test_runner_filters_normalized_rows_by_split_before_matching(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            predictions_path = root / "predictions.jsonl"
            rows_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "keep", "split": "test_task", "candidate_ids": [], "positive_ids": ["target"]}),
                        json.dumps({"id": "drop", "split": "test_domain", "candidate_ids": [], "positive_ids": ["target"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps({"id": "keep", "generated_candidates": ["target"], "selection": "target"})
                + "\n",
                encoding="utf-8",
            )

            report = runner.evaluate_files(rows_path, predictions_path, [1], split="test_task")

            self.assertEqual(report["split"], "test_task")
            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["metrics_by_k"][0]["candidate_recall"], 1.0)

    def test_runner_rejects_unknown_prediction_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            predictions_path = root / "predictions.jsonl"
            rows_path.write_text(
                json.dumps({"id": "a", "candidate_ids": [], "positive_ids": ["target"]}) + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps({"id": "unknown", "generated_candidates": []}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown example ids"):
                runner.evaluate_files(rows_path, predictions_path, [5])

    def test_runner_rejects_duplicate_gold_ids_and_missing_generated_candidates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.jsonl"
            predictions_path = root / "predictions.jsonl"
            rows_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "a", "candidate_ids": [], "positive_ids": ["target"]}),
                        json.dumps({"id": "a", "candidate_ids": [], "positive_ids": ["target"]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(json.dumps({"id": "a"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate gold"):
                runner.evaluate_files(rows_path, predictions_path, [5])

            rows_path.write_text(
                json.dumps({"id": "a", "candidate_ids": [], "positive_ids": ["target"]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "generated_candidates"):
                runner.evaluate_files(rows_path, predictions_path, [5])


if __name__ == "__main__":
    unittest.main()
