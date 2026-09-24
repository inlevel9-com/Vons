import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from vons.calibration import (
    ABSTAIN_ALL_THRESHOLD,
    CALIBRATION_VERSION,
    apply_calibration,
    calibrate_report,
    fit_abstention_threshold,
    fit_abstention_threshold_details,
    fit_temperature,
    fit_temperature_details,
)
from vons.data import Example, write_jsonl
from vons.evaluation import METRIC_VERSION, Prediction, nll, summarize, write_report


def _example(identifier: str, label: str | None, answerable: bool, split: str = "calibration") -> Example:
    return Example(
        id=identifier,
        task_group="calibration",
        state="state",
        question="choose",
        options=("a", "b"),
        label=label,
        answerable=answerable,
        split=split,
        provenance={},
        metadata={"scenario_id": identifier},
    )


def _reject_constant(value: str) -> float:
    raise ValueError(f"non-strict JSON constant {value}")


class CalibrationTests(unittest.TestCase):
    def test_temperature_fit_finds_interior_optimum_and_does_not_increase_nll(self) -> None:
        examples = [_example(f"r{index}", "a" if index < 3 else "b", True) for index in range(4)]
        predictions = [Prediction(f"r{index}", "a", {"a": 0.6, "b": 0.4}, 0.6, False) for index in range(4)]

        details = fit_temperature_details(examples, predictions)
        calibrated = apply_calibration(examples, predictions, temperature=details.temperature, abstention_threshold=0.0)

        self.assertAlmostEqual(details.temperature, math.log(1.5) / math.log(3.0), places=6)
        self.assertEqual(details.rows, 4)
        self.assertFalse(details.at_lower_bound)
        self.assertFalse(details.at_upper_bound)
        self.assertEqual(fit_temperature(examples, predictions), details.temperature)
        self.assertLess(nll(examples, calibrated), nll(examples, predictions))

    def test_temperature_bound_hit_is_flagged(self) -> None:
        examples = [_example(f"s{index}", "a", True) for index in range(4)]
        predictions = [Prediction(f"s{index}", "a", {"a": 0.9, "b": 0.1}, 0.9, False) for index in range(4)]

        details = fit_temperature_details(examples, predictions)

        self.assertTrue(details.at_lower_bound)
        self.assertAlmostEqual(details.temperature, 0.05)

    def test_temperature_scaling_preserves_true_zero_support(self) -> None:
        examples = [_example("zero", "a", True)]
        predictions = [Prediction("zero", "a", {"a": 1.0, "b": 0.0}, 1.0, False)]

        calibrated = apply_calibration(examples, predictions, temperature=20.0, abstention_threshold=0.0)

        self.assertEqual(calibrated[0].probabilities, {"a": 1.0, "b": 0.0})

    def test_invalid_probability_vectors_fail_closed(self) -> None:
        examples = [_example("x", "a", True)]
        for probabilities in ({"a": 0.5, "b": 0.2}, {"a": math.nan, "b": 1.0}, {"a": -0.5, "b": 1.5}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                apply_calibration(
                    examples,
                    [Prediction("x", "a", probabilities, 0.5, False)],
                    temperature=1.0,
                    abstention_threshold=0.0,
                )

    def test_calibration_recomputes_argmax_and_never_covers_without_choice(self) -> None:
        examples = [_example("was_abstained", "a", True), _example("stale_choice", "a", True)]
        predictions = [
            Prediction("was_abstained", None, {"a": 0.6, "b": 0.4}, 0.42, True),
            Prediction("stale_choice", "b", {"a": 0.6, "b": 0.4}, 0.6, False),
        ]

        calibrated = apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=0.2)

        self.assertEqual([(item.choice, item.abstained) for item in calibrated], [("a", False), ("a", False)])

    def test_answerability_gate_is_preserved_and_labels_do_not_gate(self) -> None:
        examples = [_example("low_answerability", "a", True), _example("unanswerable", None, False)]
        predictions = [
            Prediction("low_answerability", "a", {"a": 0.6, "b": 0.4}, 0.2, False),
            Prediction("unanswerable", "a", {"a": 0.6, "b": 0.4}, 0.6, False),
        ]

        default = apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=0.0)
        relaxed = apply_calibration(
            examples,
            predictions,
            temperature=1.0,
            abstention_threshold=0.0,
            answerability_threshold=0.3,
        )

        self.assertEqual((default[0].choice, default[0].abstained), (None, True))
        self.assertEqual((default[1].choice, default[1].abstained), ("a", False))
        self.assertEqual((relaxed[0].choice, relaxed[0].abstained), ("a", False))

    def test_failed_and_absent_distributions_remain_abstained(self) -> None:
        examples = [_example("failed", "a", True), _example("absent", "a", True)]
        predictions = [
            Prediction("failed", None, {}, 0.0, False, error="timeout"),
            Prediction("absent", "a", {}, 0.9, False),
        ]

        calibrated = apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=0.0)

        self.assertEqual((calibrated[0].choice, calibrated[0].abstained, calibrated[0].error), (None, True, "timeout"))
        self.assertEqual((calibrated[1].choice, calibrated[1].abstained, calibrated[1].probabilities), (None, True, {}))

    def test_confidence_above_max_probability_fails_closed(self) -> None:
        examples = [_example("x", "a", True)]
        with self.assertRaisesRegex(ValueError, "max_probability"):
            apply_calibration(
                examples,
                [Prediction("x", "a", {"a": 0.5, "b": 0.5}, 0.9, False)],
                temperature=1.0,
                abstention_threshold=0.0,
            )

    def test_threshold_fit_uses_argmax_and_counts_unanswerable_as_errors(self) -> None:
        examples = [_example("correct", "a", True), _example("stale_choice", "a", True), _example("unknown", None, False)]
        predictions = [
            Prediction("correct", "a", {"a": 0.9, "b": 0.1}, 0.9, False),
            Prediction("stale_choice", "b", {"a": 0.6, "b": 0.4}, 0.58, False),
            Prediction("unknown", "a", {"a": 0.55, "b": 0.45}, 0.5, False),
        ]

        details = fit_abstention_threshold_details(examples, predictions, target_risk=0.0)
        calibrated = apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=details.threshold)

        self.assertAlmostEqual(details.threshold, 0.58)
        self.assertEqual((details.rows, details.covered_rows, details.risk, details.abstain_all), (3, 2, 0.0, False))
        self.assertEqual(fit_abstention_threshold(examples, predictions, target_risk=0.0), details.threshold)
        self.assertEqual([item.abstained for item in calibrated], [False, False, True])

    def test_all_abstain_sentinel_covers_nothing_even_at_confidence_one(self) -> None:
        examples = [_example("wrong", "a", True)]
        predictions = [Prediction("wrong", "b", {"a": 0.0, "b": 1.0}, 1.0, False)]

        threshold = fit_abstention_threshold(examples, predictions, target_risk=0.0)
        at_one = apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=1.0)[0]
        sentinel = apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=threshold)[0]

        self.assertEqual(threshold, ABSTAIN_ALL_THRESHOLD)
        self.assertFalse(at_one.abstained)
        self.assertEqual((sentinel.choice, sentinel.abstained), (None, True))
        for invalid in (1.5, -0.1, math.nan):
            with self.subTest(threshold=invalid), self.assertRaisesRegex(ValueError, "abstention_threshold"):
                apply_calibration(examples, predictions, temperature=1.0, abstention_threshold=invalid)

    def test_unknown_or_duplicate_prediction_ids_fail_closed(self) -> None:
        examples = [_example("a", "a", True)]
        with self.assertRaisesRegex(ValueError, "unknown example id"):
            apply_calibration(
                examples,
                [Prediction("ghost", "a", {"a": 1.0}, 1.0, False)],
                temperature=1.0,
                abstention_threshold=0.0,
            )
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            fit_abstention_threshold(examples, [Prediction("a", "a", {"a": 1.0}, 1.0, False)] * 2)

    def test_calibrate_report_records_provenance_and_rejects_identity_mismatch(self) -> None:
        config = {
            "model_id": "vons-test",
            "backend": "direct",
            "seed": 7,
            "encoder": {"name": "encoder", "revision": "r1"},
            "answerability_threshold": 0.8,
        }
        calibration_rows = [_example("c1", "a", True), _example("c2", "b", True), _example("c3", None, False)]
        calibration_predictions = [
            Prediction("c1", "a", {"a": 0.9, "b": 0.1}, 0.9, False),
            Prediction("c2", "b", {"a": 0.3, "b": 0.7}, 0.63, False),
            Prediction("c3", None, {"a": 0.6, "b": 0.4}, 0.3, True),
        ]
        test_rows = [_example("t1", "a", True, "test"), _example("t2", "b", True, "test"), _example("t3", None, False, "test")]
        test_predictions = [
            Prediction("t1", "a", {"a": 0.8, "b": 0.2}, 0.8, False),
            Prediction("t2", "b", {"a": 0.4, "b": 0.6}, 0.42, False),
            Prediction("t3", None, {}, 0.0, False, error="timeout"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_input, calibration_report = root / "calibration.jsonl", root / "calibration.json"
            test_input, test_report, output = root / "test.jsonl", root / "test.json", root / "calibrated.json"
            write_jsonl(calibration_input, calibration_rows)
            write_jsonl(test_input, test_rows)
            write_report(calibration_report, config=config, summary=summarize(calibration_rows, calibration_predictions), predictions=calibration_predictions)
            write_report(test_report, config=config, summary=summarize(test_rows, test_predictions), predictions=test_predictions)
            arguments = {
                "calibration_input": calibration_input,
                "calibration_report": calibration_report,
                "test_input": test_input,
                "test_report": test_report,
            }

            calibrate_report(**arguments, output=output, target_risk=0.0)
            payload = json.loads(output.read_text(encoding="utf-8"), parse_constant=_reject_constant)

            fit = payload["calibration"]
            self.assertEqual(fit["version"], CALIBRATION_VERSION)
            self.assertEqual(fit["answerability_threshold"], 0.8)
            self.assertEqual((fit["temperature_fit_rows"], fit["threshold_fit_rows"], fit["rows"]), (2, 3, 3))
            self.assertTrue(fit["temperature_at_lower_bound"])
            self.assertLessEqual(fit["nll_after"], fit["nll_before"])
            self.assertEqual(payload["inputs"]["test_report"]["sha256"], hashlib.sha256(test_report.read_bytes()).hexdigest())
            self.assertEqual(payload["model_identity"]["model_id"], "vons-test")
            self.assertTrue(any("checkpoint_sha256 unavailable" in item for item in payload["limitations"]))
            self.assertEqual(payload["summary"]["metric_version"], METRIC_VERSION)
            decisions = {item["example_id"]: (item["choice"], item["abstained"]) for item in payload["predictions"]}
            self.assertEqual(decisions, {"t1": ("a", False), "t2": (None, True), "t3": (None, True)})

            mismatched = root / "mismatched.json"
            write_report(mismatched, config={**config, "model_id": "other"}, summary={}, predictions=test_predictions)
            with self.assertRaisesRegex(ValueError, "model_id"):
                calibrate_report(**{**arguments, "test_report": mismatched}, output=root / "rejected.json")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                calibrate_report(**arguments, output=test_report)
            self.assertFalse((root / "rejected.json").exists())


if __name__ == "__main__":
    unittest.main()
