import json
import math
import tempfile
import unittest
from pathlib import Path

from vons.data import Example, smoke_examples
from vons.evaluation import (
    METRIC_VERSION,
    Prediction,
    accuracy,
    brier_score,
    coverage,
    ece,
    macro_f1,
    nll,
    selective_risk,
    summarize,
    validate_distribution,
    write_report,
)


def _row(identifier: str, label: str | None, *, answerable: bool = True, options: tuple[str, ...] = ("a", "b")) -> Example:
    return Example(
        id=identifier,
        task_group="evaluation",
        state="state",
        question="choose",
        options=options,
        label=label,
        answerable=answerable,
        split="test",
        provenance={},
        metadata={},
    )


def _reject_constant(value: str) -> float:
    raise ValueError(f"non-strict JSON constant {value}")


class EvaluationTests(unittest.TestCase):
    def test_metrics_include_abstention_and_probability_checks(self) -> None:
        rows = smoke_examples()[:3]
        predictions = [
            Prediction(row.id, row.label, {option: 1.0 if option == row.label else 0.0 for option in row.options}, 1.0, False)
            for row in rows
        ]
        self.assertEqual(macro_f1(rows, predictions), 1.0)
        self.assertEqual(nll(rows, predictions), 0.0)
        self.assertEqual(brier_score(rows, predictions), 0.0)
        self.assertEqual(selective_risk(rows, predictions), 0.0)
        summary = summarize(rows, predictions)
        self.assertIn("macro_f1_answerable", summary)
        self.assertEqual(summary["metric_version"], METRIC_VERSION)
        self.assertIn("brier_answerable", summary["metric_definitions"])
        self.assertEqual(summary["probability_population"], "complete")

    def test_missing_predictions_stay_in_accuracy_and_coverage_denominators(self) -> None:
        rows = [_row(f"r{index}", "a") for index in range(10)]
        predictions = [Prediction("r0", "a", {"a": 0.99, "b": 0.01}, 0.99, False)]

        summary = summarize(rows, predictions)

        self.assertEqual(summary["accuracy_answerable"], 0.1)
        self.assertEqual(summary["coverage"], 0.1)
        self.assertEqual(summary["selective_risk"], 0.0)
        self.assertEqual(summary["counts"]["missing_prediction_rows"], 9)
        self.assertEqual(summary["counts"]["probability_rows"], 1)
        self.assertEqual(summary["counts"]["probability_unavailable_answerable_rows"], 9)
        self.assertEqual(summary["probability_population"], "incomplete")
        self.assertAlmostEqual(summary["nll_answerable"], -math.log(0.99))

    def test_abstained_and_error_predictions_are_never_correct_or_covered(self) -> None:
        rows = [_row("abstained", "a"), _row("failed", "a")]
        predictions = [
            Prediction("abstained", "a", {"a": 0.9, "b": 0.1}, 0.9, True),
            Prediction("failed", "a", {}, 0.0, False, error="timeout"),
        ]

        summary = summarize(rows, predictions)

        self.assertEqual(accuracy(rows, predictions), 0.0)
        self.assertEqual(macro_f1(rows, predictions), 0.0)
        self.assertEqual(coverage(rows, predictions), 0.0)
        self.assertIsNone(selective_risk(rows, predictions))
        self.assertEqual(summary["counts"]["abstained_prediction_rows"], 1)
        self.assertEqual(summary["counts"]["error_prediction_rows"], 1)
        self.assertEqual(summary["counts"]["covered_rows"], 0)
        self.assertEqual(summary["counts"]["probability_rows"], 1)

    def test_covered_unanswerable_prediction_is_a_selective_risk_error(self) -> None:
        rows = [_row("answerable", "a"), _row("unanswerable", None, answerable=False)]
        predictions = [
            Prediction("answerable", "a", {"a": 0.9, "b": 0.1}, 0.9, False),
            Prediction("unanswerable", "a", {"a": 0.9, "b": 0.1}, 0.9, False),
        ]

        self.assertEqual(selective_risk(rows, predictions), 0.5)
        self.assertEqual(coverage(rows, predictions), 1.0)
        self.assertAlmostEqual(ece(rows, predictions), 0.4)
        self.assertEqual(summarize(rows, predictions)["counts"]["covered_unanswerable_rows"], 1)

    def test_undefined_metrics_are_none(self) -> None:
        rows = [_row("a", "a"), _row("b", "b")]
        abstained = [
            Prediction("a", None, {"a": 0.2, "b": 0.8}, 0.2, True),
            Prediction("b", None, {"a": 0.9, "b": 0.1}, 0.1, True),
        ]
        self.assertIsNone(selective_risk(rows, abstained))
        self.assertIsNone(ece(rows, abstained))
        self.assertEqual(coverage(rows, abstained), 0.0)

        unanswerable = [_row("u", None, answerable=False)]
        answered = [Prediction("u", "a", {"a": 0.5, "b": 0.5}, 0.5, False)]
        for metric in (accuracy, macro_f1, nll, brier_score):
            self.assertIsNone(metric(unanswerable, answered))
        self.assertIsNone(coverage([], []))

    def test_macro_f1_ignores_choices_outside_answerable_rows(self) -> None:
        rows = [_row("a", "a", options=("a", "b", "zzz")), _row("b", "b", options=("a", "b", "zzz")), _row("u", None, answerable=False, options=("a", "b", "zzz"))]
        predictions = [
            Prediction("a", "a", {}, 0.9, False),
            Prediction("b", "b", {}, 0.9, False),
            Prediction("u", "zzz", {}, 0.9, False),
        ]
        self.assertEqual(macro_f1(rows, predictions), 1.0)

    def test_unknown_and_duplicate_ids_fail_closed(self) -> None:
        rows = [_row("a", "a")]
        with self.assertRaisesRegex(ValueError, "unknown example id"):
            summarize(rows, [Prediction("ghost", "a", {}, 0.9, False)])
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            summarize(rows, [Prediction("a", "a", {}, 0.9, False), Prediction("a", "b", {}, 0.9, False)])
        with self.assertRaisesRegex(ValueError, "duplicate example id"):
            summarize([_row("a", "a"), _row("a", "a")], [])

    def test_invalid_confidence_and_probability_vectors_fail_closed(self) -> None:
        rows = [_row("a", "a")]
        for confidence in (math.nan, -0.1, 1.8):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(ValueError, "confidence"):
                ece(rows, [Prediction("a", "a", {"a": 1.0}, confidence, False)])
        for probabilities in ({"a": math.nan, "b": 0.5}, {"a": -0.1, "b": 1.1}, {"a": 0.5, "b": 0.2}, {"a": 0.5, "c": 0.5}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                nll(rows, [Prediction("a", "a", probabilities, 0.5, False)])
        with self.assertRaisesRegex(ValueError, "no choice"):
            coverage(rows, [Prediction("a", None, {"a": 1.0}, 1.0, False)])

    def test_answerable_row_without_label_fails_closed(self) -> None:
        rows = [_row("a", None)]
        with self.assertRaisesRegex(ValueError, "needs a label"):
            accuracy(rows, [Prediction("a", None, {}, 0.1, True)])

    def test_brier_score_is_the_standard_multiclass_sum(self) -> None:
        for options, expected in ((("a", "b"), 0.5), (("a", "b", "c", "d"), 0.75)):
            rows = [_row("r", "a", options=options)]
            uniform = {option: 1.0 / len(options) for option in options}
            self.assertAlmostEqual(brier_score(rows, [Prediction("r", "a", uniform, 0.5, False)]), expected)

    def test_probability_metrics_are_conditional_on_valid_distributions(self) -> None:
        rows = [_row("valid", "a"), _row("failed", "a"), _row("absent", "a"), _row("zero", "a")]
        predictions = [
            Prediction("valid", "a", {"a": 0.8, "b": 0.2}, 0.8, False),
            Prediction("failed", None, {}, 0.0, False, error="crash"),
            Prediction("absent", "a", {}, 0.7, False),
            Prediction("zero", "b", {"b": 1.0}, 1.0, False),
        ]

        summary = summarize(rows, predictions)

        self.assertEqual(summary["counts"]["probability_rows"], 2)
        self.assertEqual(summary["counts"]["probability_unavailable_answerable_rows"], 2)
        self.assertEqual(summary["counts"]["nll_floor_clipped_rows"], 1)
        self.assertEqual(summary["probability_population"], "incomplete")
        self.assertAlmostEqual(summary["nll_answerable"], (-math.log(0.8) - math.log(1e-12)) / 2)
        self.assertEqual(validate_distribution({"b": 1.0}, ("a", "b")), {"b": 1.0})

    def test_ece_uses_last_bin_for_confidence_one(self) -> None:
        rows = [_row("a", "a"), _row("b", "b"), _row("c", "a"), _row("u", None, answerable=False)]
        predictions = [
            Prediction("a", "a", {"a": 0.9, "b": 0.1}, 0.9, False),
            Prediction("b", "a", {"a": 0.9, "b": 0.1}, 0.9, False),
            Prediction("c", "a", {"a": 1.0}, 1.0, False),
            Prediction("u", "b", {"a": 0.3, "b": 0.7}, 0.3, False),
        ]
        self.assertAlmostEqual(ece(rows, predictions), 0.275)
        with self.assertRaisesRegex(ValueError, "bins"):
            ece(rows, predictions, bins=0)

    def test_write_report_is_strict_json(self) -> None:
        rows = [_row("a", "a")]
        predictions = [Prediction("a", "a", {"a": 1.0}, 1.0, False, latency_ms=1.5)]
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                write_report(invalid, config={}, summary={"nll_answerable": math.nan}, predictions=predictions)
            self.assertFalse(invalid.exists())

            valid = Path(directory) / "valid.json"
            write_report(valid, config={"model_id": "m"}, summary=summarize(rows, predictions), predictions=predictions)
            payload = json.loads(valid.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            self.assertEqual(payload["summary"]["metric_version"], METRIC_VERSION)
            self.assertEqual(payload["summary"]["latency_p50_ms"], 1.5)


if __name__ == "__main__":
    unittest.main()
