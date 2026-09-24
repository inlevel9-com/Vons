import unittest

from vons.mind2web import Mind2WebExample, evaluate_mind2web, recall_at_k, task_bootstrap_intervals


class Mind2WebEvaluationTests(unittest.TestCase):
    def test_recall_and_selection_are_separate(self) -> None:
        rows = [
            Mind2WebExample("a", ("x", "target"), "target"),
            Mind2WebExample("b", ("target", "y"), "target"),
        ]
        metrics = evaluate_mind2web(
            rows,
            {"a": ("x", "target"), "b": ("y",)},
            {"a": "x", "b": "target"},
        )
        self.assertEqual(metrics.candidate_recall, 0.5)
        self.assertEqual(metrics.selection_accuracy_given_recall, 0.0)
        self.assertEqual(metrics.complete_case_selection_accuracy_given_recall, 0.0)
        self.assertEqual(metrics.evaluated_recalled_rows, 1)
        self.assertNotIn("browser_task_success", metrics.to_mapping())

    def test_mapping_keeps_positive_ids_out_of_retrieved_candidates(self) -> None:
        row = Mind2WebExample.from_mapping(
            {"id": "a", "candidate_ids": ["x"], "target_id": "target"}
        )
        self.assertEqual(row.candidate_ids, ("x",))
        self.assertEqual(row.positive_ids, ("target",))

        retrieved = Mind2WebExample.from_mapping(
            {"id": "a", "candidate_ids": ["target"], "target_id": "target"}
        )
        self.assertEqual(retrieved.candidate_ids, ("target",))

    def test_mapping_supports_multiple_positives_and_metadata(self) -> None:
        row = Mind2WebExample.from_mapping(
            {
                "id": "a",
                "candidate_ids": ["x", "y"],
                "positive_ids": ["target", "target-alias"],
                "task_id": "task-1",
                "action_id": "action-2",
                "split": "cross-website",
                "website": "example.test",
                "domain": "retail",
            }
        )
        self.assertEqual(row.positive_ids, ("target", "target-alias"))
        self.assertEqual(row.target_id, "target")
        self.assertEqual(row.split, "cross-website")

    def test_mapping_supports_explicit_no_positive_and_empty_candidates(self) -> None:
        row = Mind2WebExample.from_mapping({"id": "none", "candidates": [], "no_positive": True})
        self.assertTrue(row.no_positive)
        self.assertEqual(row.positive_ids, ())

    def test_mapping_rejects_string_false_and_null_candidate_ids(self) -> None:
        with self.assertRaisesRegex(TypeError, "no_positive"):
            Mind2WebExample.from_mapping({"id": "none", "candidates": [], "no_positive": "false"})
        with self.assertRaisesRegex(TypeError, "ids"):
            Mind2WebExample.from_mapping({"id": "bad", "candidates": [{"id": None}], "positive_ids": ["target"]})

    def test_missing_prediction_is_excluded_from_conditional_accuracy(self) -> None:
        rows = [
            Mind2WebExample.from_mapping(
                {"id": "a", "candidate_ids": ["x"], "positive_ids": ["target"]}
            ),
            Mind2WebExample.from_mapping(
                {"id": "b", "candidate_ids": ["x"], "positive_ids": ["target"]}
            ),
        ]
        metrics = evaluate_mind2web(
            rows,
            {"a": ("target",), "b": ("target",)},
            {"a": None, "b": "target"},
        )
        self.assertEqual(metrics.candidate_recall, 1.0)
        self.assertEqual(metrics.selection_accuracy_given_recall, 0.5)
        self.assertEqual(metrics.complete_case_selection_accuracy_given_recall, 1.0)
        self.assertEqual(metrics.missing_predictions, 1)
        self.assertEqual(metrics.evaluated_recalled_rows, 1)

    def test_invalid_selection_is_evaluated_but_not_correct(self) -> None:
        row = Mind2WebExample.from_mapping(
            {"id": "a", "candidate_ids": ["x"], "positive_ids": ["target"]}
        )
        metrics = evaluate_mind2web({row}, {"a": ("target", "x")}, {"a": "not-retrieved"})
        self.assertEqual(metrics.selection_accuracy_given_recall, 0.0)
        self.assertEqual(metrics.invalid_selections, 1)

    def test_undefined_metrics_are_null_and_no_positive_is_separate(self) -> None:
        row = Mind2WebExample.from_mapping({"id": "none", "no_positive": True})
        metrics = evaluate_mind2web([row], {"none": ()}, {"none": None})
        self.assertIsNone(metrics.candidate_recall)
        self.assertIsNone(metrics.selection_accuracy_given_recall)
        self.assertEqual(metrics.no_positive_rows, 1)
        self.assertIsNone(metrics.to_mapping()["candidate_recall"])

    def test_recall_at_k_uses_retriever_order(self) -> None:
        row = Mind2WebExample.from_mapping(
            {"id": "a", "candidate_ids": [], "positive_ids": ["target"]}
        )
        self.assertEqual(recall_at_k([row], {"a": ("x", "target")}, 1), 0.0)
        self.assertEqual(recall_at_k([row], {"a": ("x", "target")}, 2), 1.0)

    def test_task_macro_and_step_micro_metrics_are_separate(self) -> None:
        rows = [
            Mind2WebExample("a1", ("target",), "target", task_id="task-a"),
            Mind2WebExample("a2", ("x",), "target", task_id="task-a"),
            Mind2WebExample("b1", ("target",), "target", task_id="task-b"),
        ]
        metrics = evaluate_mind2web(
            rows,
            {"a1": ("target",), "a2": ("x",), "b1": ("target",)},
            {"a1": "target", "b1": "target"},
        )
        self.assertEqual(metrics.candidate_recall, 2 / 3)
        self.assertEqual(metrics.candidate_recall_task_macro, 0.75)
        self.assertEqual(metrics.selection_accuracy_given_recall_task_macro, 1.0)
        self.assertEqual(metrics.to_mapping()["candidate_recall_step_micro"], 2 / 3)
        self.assertEqual(metrics.task_group_count, 2)

    def test_task_bootstrap_is_deterministic_and_task_weighted(self) -> None:
        rows = [
            Mind2WebExample("a1", ("target",), "target", task_id="task-a"),
            Mind2WebExample("a2", ("x",), "target", task_id="task-a"),
            Mind2WebExample("b1", ("target",), "target", task_id="task-b"),
        ]
        args = (
            rows,
            {"a1": ("target",), "a2": ("x",), "b1": ("target",)},
            {"a1": "target", "b1": "target"},
        )
        first = task_bootstrap_intervals(*args, draws=20, seed=11)
        second = task_bootstrap_intervals(*args, draws=20, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(first["task_group_count"], 2)
        self.assertEqual(len(first["candidate_recall_task_macro_ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
