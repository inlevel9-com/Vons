"""Post-hoc probability and abstention calibration for saved Vons reports.

Calibration contract ``vons.calibration/v3``:

- Probability vectors must be finite, non-negative, and normalized. Vector
  scaling fits one bounded temperature and bias per class; options with true
  zero probability stay at zero.
- Saved predictions follow the producer contract
  ``confidence = max_probability * answerability``. The answerability factor
  is reconstructed only when ``0 <= confidence <= max_probability`` (within
  ``CONFIDENCE_FACTOR_TOLERANCE``); any other confidence fails closed.
- A prediction is eligible for coverage only when it has no error, carries a
  probability vector, and its answerability meets the configured threshold
  (report config ``answerability_threshold``, default 0.5). Ground-truth labels
  never gate eligibility; they are used only to measure fitting risk.
- The default report pipeline derives its abstention threshold from calibrated
  multiclass Brier losses. An eligible prediction is covered when its
  calibrated confidence meets that fitted threshold; every other prediction
  abstains with ``choice=None``, and a failed prediction keeps its error.
- ``ABSTAIN_ALL_THRESHOLD`` (2.0) is the documented sentinel for "abstain on
  every prediction". Confidences are validated in [0, 1], so the sentinel
  covers nothing, including wrong predictions with confidence 1.0.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .data import Example, read_jsonl
from .evaluation import (
    NLL_PROBABILITY_FLOOR,
    Prediction,
    summarize,
    validate_distribution,
    validate_predictions,
)

CALIBRATION_VERSION = "vons.calibration/v3"
ABSTAIN_ALL_THRESHOLD = 2.0
DEFAULT_ANSWERABILITY_THRESHOLD = 0.5
CONFIDENCE_FACTOR_TOLERANCE = 1e-6
DECISION_EPSILON = 1e-12
BOUND_TOLERANCE = 1e-6
IDENTITY_FIELDS = ("backend", "model_id", "encoder", "seed", "checkpoint", "checkpoint_sha256")


@dataclass(frozen=True)
class CalibrationFit:
    """Parameters fitted on a calibration split and then reused unchanged.

    ``rows`` counts every calibration example. ``temperature_fit_rows`` counts
    the answerable rows with a usable probability vector, and
    ``threshold_fit_rows`` counts the rows with a prediction.
    """

    temperature: float
    abstention_threshold: float
    target_risk: float | None
    rows: int
    nll_before: float | None
    nll_after: float | None
    temperature_fit_rows: int = 0
    threshold_fit_rows: int = 0
    threshold_covered_rows: int = 0
    threshold_risk: float | None = None
    temperature_lower_bound: float = 0.1
    temperature_upper_bound: float = 10.0
    temperature_at_lower_bound: bool = False
    temperature_at_upper_bound: bool = False
    abstain_all: bool = False
    answerability_threshold: float = DEFAULT_ANSWERABILITY_THRESHOLD
    calibration_method: str = "vector_scaling"
    temperatures: Mapping[str, float] | None = None
    biases: Mapping[str, float] | None = None
    log_temperature_l2: float = 0.01
    brier_before: float | None = None
    brier_after: float | None = None
    target_brier: float | None = None
    threshold_brier: float | None = None
    threshold_objective: str = "calibrated_multiclass_brier"
    version: str = CALIBRATION_VERSION


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    rows: int
    lower: float
    upper: float
    at_lower_bound: bool
    at_upper_bound: bool
    log_temperature_l2: float = 0.01


@dataclass(frozen=True)
class VectorScalingFit:
    temperatures: Mapping[str, float]
    biases: Mapping[str, float]
    rows: int
    lower: float
    upper: float
    log_temperature_l2: float
    nll_before: float | None
    nll_after: float | None
    at_lower_bound: bool
    at_upper_bound: bool


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    rows: int
    covered_rows: int
    risk: float | None
    abstain_all: bool


@dataclass(frozen=True)
class BrierThresholdFit:
    threshold: float
    rows: int
    covered_rows: int
    target_brier: float | None
    covered_brier: float | None
    abstain_all: bool


@dataclass(frozen=True)
class _Decision:
    probabilities: Mapping[str, float]
    confidence: float
    choice: str | None
    eligible: bool


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a finite positive number")
    return value


def _validate_unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _validate_abstention_threshold(value: float) -> float:
    number = float(value)
    if number == ABSTAIN_ALL_THRESHOLD:
        return number
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("abstention_threshold must be in [0, 1] or equal ABSTAIN_ALL_THRESHOLD")
    return number


def _scaled_probabilities(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scale a validated vector; true zeros stay zero."""
    _validate_temperature(temperature)
    values = validate_distribution(probabilities)
    if not values:
        return {}
    logs = {option: math.log(value) / temperature for option, value in values.items() if value > 0.0}
    pivot = max(logs.values())
    weights = {option: math.exp(value - pivot) for option, value in logs.items()}
    total = math.fsum(weights.values())
    return {option: weights[option] / total if option in weights else 0.0 for option in values}


def _validate_vector_parameters(
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    if set(temperatures) != set(biases):
        raise ValueError("vector scaling temperatures and biases must use identical classes")
    validated_temperatures = {
        str(option): _validate_temperature(value) for option, value in temperatures.items()
    }
    validated_biases: dict[str, float] = {}
    for option, value in biases.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("vector scaling biases must be finite")
        validated_biases[str(option)] = number
    return validated_temperatures, validated_biases


def _vector_scaled_probabilities(
    probabilities: Mapping[str, float],
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
) -> dict[str, float]:
    """Apply diagonal vector scaling to a probability vector.

    Input probabilities are treated as logits in log space. Structural zeros
    remain zero so calibration cannot invent support for a class the producer
    declared impossible.
    """
    values = validate_distribution(probabilities)
    if not values:
        return {}
    validated_temperatures, validated_biases = _validate_vector_parameters(
        temperatures, biases
    )
    missing = set(values) - set(validated_temperatures)
    if missing:
        raise ValueError(f"vector scaling parameters are missing classes: {sorted(missing)}")
    logits = {
        option: math.log(value) / validated_temperatures[option] + validated_biases[option]
        for option, value in values.items()
        if value > 0.0
    }
    pivot = max(logits.values())
    weights = {option: math.exp(value - pivot) for option, value in logits.items()}
    total = math.fsum(weights.values())
    return {option: weights.get(option, 0.0) / total for option in values}


def _argmax(probabilities: Mapping[str, float], options: Sequence[str]) -> str:
    best = max(probabilities.get(option, 0.0) for option in options)
    return next(option for option in options if probabilities.get(option, 0.0) == best)


def _answerability_factor(prediction: Prediction, distribution: Mapping[str, float]) -> float:
    maximum = max(distribution.values())
    confidence = float(prediction.confidence)
    if not -CONFIDENCE_FACTOR_TOLERANCE <= confidence <= maximum + CONFIDENCE_FACTOR_TOLERANCE:
        raise ValueError(
            f"prediction {prediction.example_id!r} violates confidence = max_probability * answerability "
            f"(confidence={confidence!r}, max_probability={maximum!r})"
        )
    return min(1.0, max(0.0, confidence) / maximum)


def _decide(prediction: Prediction, row: Example, *, temperature: float, answerability_threshold: float) -> _Decision:
    if prediction.error is not None:
        return _Decision(prediction.probabilities, float(prediction.confidence), None, False)
    distribution = validate_distribution(prediction.probabilities, row.options)
    if not distribution:
        return _Decision({}, float(prediction.confidence), None, False)
    factor = _answerability_factor(prediction, distribution)
    calibrated = _scaled_probabilities(distribution, temperature)
    confidence = min(1.0, max(0.0, factor * max(calibrated.values())))
    eligible = factor + DECISION_EPSILON >= answerability_threshold
    return _Decision(calibrated, confidence, _argmax(calibrated, row.options), eligible)


def _decide_vector(
    prediction: Prediction,
    row: Example,
    *,
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
    answerability_threshold: float,
) -> _Decision:
    if prediction.error is not None:
        return _Decision(prediction.probabilities, float(prediction.confidence), None, False)
    distribution = validate_distribution(prediction.probabilities, row.options)
    if not distribution:
        return _Decision({}, float(prediction.confidence), None, False)
    factor = _answerability_factor(prediction, distribution)
    calibrated = _vector_scaled_probabilities(distribution, temperatures, biases)
    confidence = min(1.0, max(0.0, factor * max(calibrated.values())))
    eligible = factor + DECISION_EPSILON >= answerability_threshold
    return _Decision(calibrated, confidence, _argmax(calibrated, row.options), eligible)


def _is_covered(decision: _Decision, threshold: float) -> bool:
    return decision.eligible and decision.confidence + DECISION_EPSILON >= threshold


def _temperature_rows(
    rows: Mapping[str, Example],
    lookup: Mapping[str, Prediction],
) -> list[tuple[Example, dict[str, float]]]:
    usable: list[tuple[Example, dict[str, float]]] = []
    for row in rows.values():
        item = lookup.get(row.id)
        if not row.answerable or row.label is None or item is None or item.error is not None:
            continue
        distribution = validate_distribution(item.probabilities, row.options)
        if distribution:
            usable.append((row, distribution))
    return usable


def _mean_nll(usable: Sequence[tuple[Example, Mapping[str, float]]], temperature: float) -> float:
    total = math.fsum(
        -math.log(max(_scaled_probabilities(distribution, temperature).get(str(row.label), 0.0), NLL_PROBABILITY_FLOOR))
        for row, distribution in usable
    )
    return total / len(usable)


def _nll_at(usable: Sequence[tuple[Example, Mapping[str, float]]], temperature: float) -> float | None:
    return _mean_nll(usable, temperature) if usable else None


def _mean_vector_nll(
    usable: Sequence[tuple[Example, Mapping[str, float]]],
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
) -> float:
    return math.fsum(
        -math.log(
            max(
                _vector_scaled_probabilities(distribution, temperatures, biases).get(
                    str(row.label), 0.0
                ),
                NLL_PROBABILITY_FLOOR,
            )
        )
        for row, distribution in usable
    ) / len(usable)


def _mean_vector_brier(
    usable: Sequence[tuple[Example, Mapping[str, float]]],
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
) -> float:
    return math.fsum(
        math.fsum(
            (
                _vector_scaled_probabilities(distribution, temperatures, biases).get(option, 0.0)
                - float(option == row.label)
            )
            ** 2
            for option in row.options
        )
        for row, distribution in usable
    ) / len(usable)


def _bounded_minimum(
    objective: Any,
    lower: float,
    upper: float,
    iterations: int,
) -> float:
    left, right = lower, upper
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - golden * (right - left)
    second = left + golden * (right - left)
    first_value, second_value = objective(first), objective(second)
    for _ in range(max(1, iterations)):
        if first_value <= second_value:
            right, second, second_value = second, first, first_value
            first = right - golden * (right - left)
            first_value = objective(first)
        else:
            left, first, first_value = first, second, second_value
            second = left + golden * (right - left)
            second_value = objective(second)
    return (left + right) / 2.0


def fit_vector_scaling_details(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    lower: float = 0.1,
    upper: float = 10.0,
    log_temperature_l2: float = 0.01,
    passes: int = 12,
    iterations: int = 32,
) -> VectorScalingFit:
    """Fit per-class temperature and bias parameters by coordinate descent."""
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower <= 0 or upper <= lower:
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")
    if not math.isfinite(log_temperature_l2) or log_temperature_l2 < 0:
        raise ValueError("log_temperature_l2 must be finite and non-negative")
    if passes < 1 or iterations < 1:
        raise ValueError("passes and iterations must be positive")
    rows, lookup = validate_predictions(examples, predictions)
    usable = _temperature_rows(rows, lookup)
    classes = sorted({option for row, _ in usable for option in row.options})
    if not usable or not classes:
        return VectorScalingFit(
            temperatures={},
            biases={},
            rows=0,
            lower=lower,
            upper=upper,
            log_temperature_l2=log_temperature_l2,
            nll_before=None,
            nll_after=None,
            at_lower_bound=False,
            at_upper_bound=False,
        )

    log_temperatures = {option: 0.0 for option in classes}
    biases = {option: 0.0 for option in classes}
    log_lower, log_upper = math.log(lower), math.log(upper)

    def parameters() -> dict[str, float]:
        return {option: math.exp(log_temperatures[option]) for option in classes}

    def objective() -> float:
        nll_value = _mean_vector_nll(usable, parameters(), biases)
        temperature_penalty = math.fsum(value * value for value in log_temperatures.values())
        bias_penalty = math.fsum(value * value for value in biases.values())
        return nll_value + log_temperature_l2 * (
            temperature_penalty / len(classes) + 0.1 * bias_penalty / len(classes)
        )

    previous = objective()
    for _ in range(passes):
        for option in classes:
            original = log_temperatures[option]

            def temperature_objective(
                value: float, _option: str = option, _original: float = original
            ) -> float:
                log_temperatures[_option] = value
                result = objective()
                log_temperatures[_option] = _original
                return result

            candidate = _bounded_minimum(
                temperature_objective, log_lower, log_upper, iterations
            )
            log_temperatures[option] = candidate
        for option in classes:
            original = biases[option]

            def bias_objective(
                value: float,
                _option: str = option,
                _original: float = original,
                _biases: dict[str, float] = biases,
            ) -> float:
                _biases[_option] = value
                result = objective()
                _biases[_option] = _original
                return result

            biases[option] = _bounded_minimum(bias_objective, -4.0, 4.0, iterations)
        # A common bias offset is unidentifiable under softmax. Centering keeps
        # the saved parameters stable without changing calibrated outputs.
        mean_bias = math.fsum(biases.values()) / len(biases)
        biases = {option: value - mean_bias for option, value in biases.items()}
        current = objective()
        if previous - current <= 1e-10:
            break
        previous = current

    fitted_temperatures = parameters()
    return VectorScalingFit(
        temperatures=fitted_temperatures,
        biases=biases,
        rows=len(usable),
        lower=lower,
        upper=upper,
        log_temperature_l2=log_temperature_l2,
        nll_before=_mean_vector_nll(
            usable,
            {option: 1.0 for option in classes},
            {option: 0.0 for option in classes},
        ),
        nll_after=_mean_vector_nll(usable, fitted_temperatures, biases),
        at_lower_bound=any(value - lower <= BOUND_TOLERANCE for value in fitted_temperatures.values()),
        at_upper_bound=any(upper - value <= BOUND_TOLERANCE for value in fitted_temperatures.values()),
    )


def fit_temperature_details(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    lower: float = 0.1,
    upper: float = 10.0,
    log_temperature_l2: float = 0.01,
    iterations: int = 80,
) -> TemperatureFit:
    """Fit a regularized scalar temperature for backwards-compatible callers."""
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower <= 0 or upper <= lower:
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")
    if not math.isfinite(log_temperature_l2) or log_temperature_l2 < 0:
        raise ValueError("log_temperature_l2 must be finite and non-negative")
    rows, lookup = validate_predictions(examples, predictions)
    usable = _temperature_rows(rows, lookup)
    if not usable:
        return TemperatureFit(1.0, 0, lower, upper, False, False, log_temperature_l2)

    def objective(log_temperature: float) -> float:
        return _mean_nll(usable, math.exp(log_temperature)) + log_temperature_l2 * (
            log_temperature**2
        )

    left, right = math.log(lower), math.log(upper)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - golden * (right - left)
    second = left + golden * (right - left)
    first_value, second_value = objective(first), objective(second)
    for _ in range(max(1, iterations)):
        if first_value <= second_value:
            right, second, second_value = second, first, first_value
            first = right - golden * (right - left)
            first_value = objective(first)
        else:
            left, first, first_value = first, second, second_value
            second = left + golden * (right - left)
            second_value = objective(second)
    log_temperature = (left + right) / 2.0
    return TemperatureFit(
        temperature=math.exp(log_temperature),
        rows=len(usable),
        lower=lower,
        upper=upper,
        at_lower_bound=log_temperature - math.log(lower) <= BOUND_TOLERANCE,
        at_upper_bound=math.log(upper) - log_temperature <= BOUND_TOLERANCE,
        log_temperature_l2=log_temperature_l2,
    )


def fit_temperature(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    lower: float = 0.1,
    upper: float = 10.0,
    log_temperature_l2: float = 0.01,
    iterations: int = 80,
) -> float:
    """Return the fitted temperature; see ``fit_temperature_details`` for bound flags."""
    return fit_temperature_details(
        examples,
        predictions,
        lower=lower,
        upper=upper,
        log_temperature_l2=log_temperature_l2,
        iterations=iterations,
    ).temperature


def fit_abstention_threshold_details(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    target_risk: float = 0.05,
    temperature: float = 1.0,
    answerability_threshold: float = DEFAULT_ANSWERABILITY_THRESHOLD,
) -> ThresholdFit:
    """Choose the lowest confidence cutoff with maximum coverage whose risk meets the target.

    Coverage, argmax choices, and eligibility match ``apply_calibration``. A
    covered prediction on an unanswerable row counts as an error. When no
    cutoff meets the target, the fit returns ``ABSTAIN_ALL_THRESHOLD``.
    """
    target = float(target_risk)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_risk must be between 0 and 1")
    _validate_temperature(temperature)
    _validate_unit_interval(answerability_threshold, "answerability_threshold")
    rows, lookup = validate_predictions(examples, predictions)
    decisions = [
        (rows[example_id], _decide(item, rows[example_id], temperature=temperature, answerability_threshold=answerability_threshold))
        for example_id, item in lookup.items()
    ]
    eligible = [(row, decision) for row, decision in decisions if decision.eligible]
    best: tuple[int, float, float] | None = None
    for threshold in sorted({0.0, *(decision.confidence for _, decision in eligible)}):
        covered = [(row, decision) for row, decision in eligible if _is_covered(decision, threshold)]
        if not covered:
            continue
        errors = sum(not row.answerable or decision.choice != row.label for row, decision in covered)
        risk = errors / len(covered)
        if risk <= target and (best is None or len(covered) > best[0]):
            best = (len(covered), threshold, risk)
    if best is None:
        return ThresholdFit(ABSTAIN_ALL_THRESHOLD, len(decisions), 0, None, True)
    return ThresholdFit(best[1], len(decisions), best[0], best[2], False)


def fit_abstention_threshold(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    target_risk: float = 0.05,
    temperature: float = 1.0,
    answerability_threshold: float = DEFAULT_ANSWERABILITY_THRESHOLD,
) -> float:
    """Return the fitted threshold, or ``ABSTAIN_ALL_THRESHOLD`` when nothing can be covered."""
    return fit_abstention_threshold_details(
        examples,
        predictions,
        target_risk=target_risk,
        temperature=temperature,
        answerability_threshold=answerability_threshold,
    ).threshold


def apply_calibration(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    temperature: float,
    abstention_threshold: float,
    answerability_threshold: float = DEFAULT_ANSWERABILITY_THRESHOLD,
) -> list[Prediction]:
    """Apply fitted parameters without changing the original prediction objects.

    Covered outputs always carry the argmax choice of the calibrated vector;
    every other output abstains with ``choice=None``.
    """
    _validate_temperature(temperature)
    threshold = _validate_abstention_threshold(abstention_threshold)
    _validate_unit_interval(answerability_threshold, "answerability_threshold")
    rows, _lookup = validate_predictions(examples, predictions)
    calibrated: list[Prediction] = []
    for prediction in predictions:
        if prediction.error is not None:
            calibrated.append(replace(prediction, choice=None, abstained=True))
            continue
        decision = _decide(
            prediction,
            rows[prediction.example_id],
            temperature=temperature,
            answerability_threshold=answerability_threshold,
        )
        covered = _is_covered(decision, threshold)
        calibrated.append(
            replace(
                prediction,
                choice=decision.choice if covered else None,
                probabilities=dict(decision.probabilities),
                confidence=decision.confidence,
                abstained=not covered,
            )
        )
    return calibrated


def fit_brier_abstention_threshold_details(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
    target_brier: float | None = None,
    answerability_threshold: float = DEFAULT_ANSWERABILITY_THRESHOLD,
) -> BrierThresholdFit:
    """Fit the widest-coverage threshold meeting a calibrated Brier target.

    When no target is supplied, the mean calibrated answerable-row Brier
    score becomes the data-derived target. This selects the widest confidence
    region whose aggregate calibration loss is no worse than the fitted
    answerable population instead of discarding the higher-loss half by
    construction. Unanswerable rows receive the
    maximum unit penalty when covered so the threshold does not learn to accept
    unsupported decisions merely because they have a sharp distribution.
    """
    validated_temperatures, validated_biases = _validate_vector_parameters(
        temperatures, biases
    )
    _validate_unit_interval(answerability_threshold, "answerability_threshold")
    rows, lookup = validate_predictions(examples, predictions)
    decisions = [
        (
            rows[example_id],
            _decide_vector(
                item,
                rows[example_id],
                temperatures=validated_temperatures,
                biases=validated_biases,
                answerability_threshold=answerability_threshold,
            ),
        )
        for example_id, item in lookup.items()
    ]

    def row_brier(row: Example, decision: _Decision) -> float:
        if not row.answerable or row.label is None:
            return 1.0
        return math.fsum(
            (decision.probabilities.get(option, 0.0) - float(option == row.label)) ** 2
            for option in row.options
        )

    answerable_scores = [
        row_brier(row, decision)
        for row, decision in decisions
        if decision.eligible and row.answerable and row.label is not None
    ]
    if target_brier is None:
        resolved_target = (
            math.fsum(answerable_scores) / len(answerable_scores)
            if answerable_scores
            else None
        )
    else:
        resolved_target = float(target_brier)
        if not math.isfinite(resolved_target) or not 0.0 <= resolved_target <= 2.0:
            raise ValueError("target_brier must be finite and in [0, 2]")
    if resolved_target is None:
        return BrierThresholdFit(
            ABSTAIN_ALL_THRESHOLD, len(decisions), 0, None, None, True
        )

    eligible = [(row, decision) for row, decision in decisions if decision.eligible]
    best: tuple[int, float, float] | None = None
    for threshold in sorted({0.0, *(decision.confidence for _, decision in eligible)}):
        covered = [
            (row, decision)
            for row, decision in eligible
            if _is_covered(decision, threshold)
        ]
        if not covered:
            continue
        mean_brier = math.fsum(row_brier(row, decision) for row, decision in covered) / len(
            covered
        )
        if mean_brier <= resolved_target + DECISION_EPSILON and (
            best is None or len(covered) > best[0]
        ):
            best = (len(covered), threshold, mean_brier)
    if best is None:
        return BrierThresholdFit(
            ABSTAIN_ALL_THRESHOLD,
            len(decisions),
            0,
            resolved_target,
            None,
            True,
        )
    return BrierThresholdFit(
        best[1], len(decisions), best[0], resolved_target, best[2], False
    )


def apply_vector_calibration(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    *,
    temperatures: Mapping[str, float],
    biases: Mapping[str, float],
    abstention_threshold: float,
    answerability_threshold: float = DEFAULT_ANSWERABILITY_THRESHOLD,
) -> list[Prediction]:
    """Apply vector scaling and the fitted dynamic abstention threshold."""
    validated_temperatures, validated_biases = _validate_vector_parameters(
        temperatures, biases
    )
    threshold = _validate_abstention_threshold(abstention_threshold)
    _validate_unit_interval(answerability_threshold, "answerability_threshold")
    rows, _lookup = validate_predictions(examples, predictions)
    calibrated: list[Prediction] = []
    for prediction in predictions:
        if prediction.error is not None:
            calibrated.append(replace(prediction, choice=None, abstained=True))
            continue
        decision = _decide_vector(
            prediction,
            rows[prediction.example_id],
            temperatures=validated_temperatures,
            biases=validated_biases,
            answerability_threshold=answerability_threshold,
        )
        covered = _is_covered(decision, threshold)
        calibrated.append(
            replace(
                prediction,
                choice=decision.choice if covered else None,
                probabilities=dict(decision.probabilities),
                confidence=decision.confidence,
                abstained=not covered,
            )
        )
    return calibrated


def _read_report(path: str | Path) -> tuple[dict[str, Any], list[Prediction]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: report must be a JSON object")
    config = payload.get("config", {})
    raw = payload.get("predictions", [])
    if not isinstance(config, Mapping) or not isinstance(raw, list):
        raise TypeError(f"{path}: report needs a config object and a predictions list")
    return dict(config), [Prediction(**item) for item in raw]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(calibration_config: Mapping[str, Any], test_config: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for field in IDENTITY_FIELDS:
        left, right = calibration_config.get(field), test_config.get(field)
        if left != right:
            raise ValueError(f"calibration and test reports disagree on {field}: {left!r} != {right!r}")
        identity[field] = left
    return identity


def _answerability_threshold(calibration_config: Mapping[str, Any], test_config: Mapping[str, Any]) -> float:
    values: set[float] = set()
    for config in (calibration_config, test_config):
        value = config.get("answerability_threshold")
        values.add(_validate_unit_interval(DEFAULT_ANSWERABILITY_THRESHOLD if value is None else value, "answerability_threshold"))
    if len(values) != 1:
        raise ValueError("calibration and test reports use different answerability thresholds")
    return values.pop()


def _limitations(identity: Mapping[str, Any]) -> list[str]:
    limitations: list[str] = []
    if identity.get("checkpoint_sha256") is None:
        limitations.append(
            "checkpoint_sha256 unavailable: report configs carry no immutable checkpoint hash, so model identity "
            "is checked only on backend, model_id, encoder, seed, and checkpoint fields when present"
        )
    if identity.get("backend") == "diffusion":
        limitations.append(
            "N1 (numerics review, unchanged in vons.calibration/v2): Diffusion probabilities are a softmax over "
            "x0 estimates, so a perfect one-hot estimate over four options has max probability 0.4754"
        )
        limitations.append(
            "N2 (numerics review, unchanged in vons.calibration/v2): the 32-step schedule reuses the 1000-step "
            "beta range, so terminal alpha_bar is 0.7234 while sampling starts from N(0, I)"
        )
    return limitations


def calibrate_report(
    *,
    calibration_input: str | Path,
    calibration_report: str | Path,
    test_input: str | Path,
    test_report: str | Path,
    output: str | Path,
    target_risk: float | None = None,
    target_brier: float | None = None,
) -> Path:
    """Fit on calibration rows and write a calibrated test report as strict JSON."""
    inputs = {
        "calibration_input": Path(calibration_input),
        "calibration_report": Path(calibration_report),
        "test_input": Path(test_input),
        "test_report": Path(test_report),
    }
    target = Path(output)
    if target.resolve() in {path.resolve() for path in inputs.values()}:
        raise ValueError("output must not overwrite a calibration input")
    calibration_examples = read_jsonl(calibration_input)
    calibration_config, calibration_predictions = _read_report(calibration_report)
    test_examples = read_jsonl(test_input)
    test_config, test_predictions = _read_report(test_report)
    identity = _model_identity(calibration_config, test_config)
    answerability_threshold = _answerability_threshold(calibration_config, test_config)
    if target_risk is not None and target_brier is not None:
        raise ValueError("specify target_brier or legacy target_risk, not both")
    resolved_target_brier = target_brier if target_brier is not None else target_risk

    vector_fit = fit_vector_scaling_details(calibration_examples, calibration_predictions)
    if vector_fit.temperatures:
        temperatures = dict(vector_fit.temperatures)
        biases = dict(vector_fit.biases)
    else:
        classes = sorted({option for row in calibration_examples for option in row.options})
        temperatures = {option: 1.0 for option in classes}
        biases = {option: 0.0 for option in classes}
    threshold_fit = fit_brier_abstention_threshold_details(
        calibration_examples,
        calibration_predictions,
        temperatures=temperatures,
        biases=biases,
        target_brier=resolved_target_brier,
        answerability_threshold=answerability_threshold,
    )
    calibrated_test = apply_vector_calibration(
        test_examples,
        test_predictions,
        temperatures=temperatures,
        biases=biases,
        abstention_threshold=threshold_fit.threshold,
        answerability_threshold=answerability_threshold,
    )
    usable = _temperature_rows(*validate_predictions(calibration_examples, calibration_predictions))
    identity_temperatures = {option: 1.0 for option in temperatures}
    identity_biases = {option: 0.0 for option in temperatures}
    geometric_temperature = math.exp(
        math.fsum(math.log(value) for value in temperatures.values()) / len(temperatures)
    )
    fit = CalibrationFit(
        temperature=geometric_temperature,
        abstention_threshold=threshold_fit.threshold,
        target_risk=None,
        rows=len(calibration_examples),
        nll_before=vector_fit.nll_before,
        nll_after=vector_fit.nll_after,
        temperature_fit_rows=vector_fit.rows,
        threshold_fit_rows=threshold_fit.rows,
        threshold_covered_rows=threshold_fit.covered_rows,
        threshold_risk=None,
        temperature_lower_bound=vector_fit.lower,
        temperature_upper_bound=vector_fit.upper,
        temperature_at_lower_bound=vector_fit.at_lower_bound,
        temperature_at_upper_bound=vector_fit.at_upper_bound,
        abstain_all=threshold_fit.abstain_all,
        answerability_threshold=answerability_threshold,
        temperatures=temperatures,
        biases=biases,
        log_temperature_l2=vector_fit.log_temperature_l2,
        brier_before=(
            _mean_vector_brier(usable, identity_temperatures, identity_biases)
            if usable
            else None
        ),
        brier_after=(
            _mean_vector_brier(usable, temperatures, biases) if usable else None
        ),
        target_brier=threshold_fit.target_brier,
        threshold_brier=threshold_fit.covered_brier,
    )
    payload = json.dumps(
        {
            "config": {**test_config, "calibration": asdict(fit)},
            "calibration": asdict(fit),
            "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()},
            "model_identity": identity,
            "limitations": _limitations(identity),
            "summary": summarize(test_examples, calibrated_test),
            "predictions": [asdict(item) for item in calibrated_test],
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target
