"""Metrics and evidence serialization for Vons experiments.

Metric definitions are versioned by ``METRIC_VERSION``; ``summarize`` embeds
the version, a definition per public metric key, and the row counts behind
every denominator. Inputs fail closed: duplicate or unknown ids, invalid
confidences, invalid probability vectors, and covered predictions without a
choice raise ``ValueError`` instead of being dropped, clipped, or guessed.
A metric whose denominator is empty is ``None``, never a perfect-looking 0.0.
"""

from __future__ import annotations

import json
import math
import numbers
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Example

METRIC_VERSION = "vons.metrics/v2"
PROBABILITY_TOLERANCE = 1e-5
CONFIDENCE_TOLERANCE = 1e-9
NLL_PROBABILITY_FLOOR = 1e-12

METRIC_DEFINITIONS: dict[str, str] = {
    "accuracy_answerable": (
        "correct covered predictions / all answerable rows; missing, error, and abstained "
        "predictions count as misses"
    ),
    "macro_f1_answerable": (
        "unweighted mean F1 over the labels of answerable rows and the covered choices on "
        "answerable rows; missing, error, and abstained predictions count as misses"
    ),
    "coverage": "covered predictions (no error, not abstained) / all rows, including rows without a prediction",
    "selective_risk": "errors / covered predictions; a covered prediction on an unanswerable row is an error",
    "nll_answerable": (
        "mean -log(max(p(label), 1e-12)) over answerable rows whose prediction has no error and "
        "a valid probability vector; conditional on that population (see counts)"
    ),
    "brier_answerable": (
        "mean standard multiclass Brier score, summed over the row's options (range 0..2), over "
        "answerable rows whose prediction has no error and a valid probability vector; "
        "vons.metrics/v1 divided each row by its option count"
    ),
    "ece": (
        "10 equal-width confidence bins (1.0 in the last bin) over covered predictions; correct "
        "means answerable and choice == label, so covered unanswerable rows are incorrect"
    ),
    "latency_p50_ms": "median latency over predictions that report latency",
    "latency_p95_ms": "linearly interpolated 95th percentile latency over predictions that report latency",
}

METRIC_POLICY: dict[str, str] = {
    "undefined": "a metric is null when its denominator is empty",
    "coverage_partition": "prediction_rows = covered_rows + abstained_prediction_rows + error_prediction_rows",
    "probability_vectors": (
        "finite, non-negative, keyed by candidate options, and summing to 1 within 1e-5; options "
        "absent from the vector have true zero probability; error predictions never contribute a vector"
    ),
    "invalid_inputs": (
        "duplicate or unknown ids, confidence outside [0, 1], invalid probability vectors, and covered "
        "predictions without a choice raise ValueError"
    ),
    "json": "reports are strict JSON; NaN and Infinity are rejected",
}


@dataclass(frozen=True)
class Prediction:
    """One model decision for one example.

    A prediction is *covered* only when ``error`` is ``None`` and ``abstained``
    is false; a covered prediction must carry a ``choice``. ``probabilities``
    may be empty when no distribution is available.
    """

    example_id: str
    choice: str | None
    probabilities: Mapping[str, float]
    confidence: float
    abstained: bool
    latency_ms: float | None = None
    error: str | None = None


def is_covered(prediction: Prediction) -> bool:
    """Return whether the prediction commits to a decision."""
    return prediction.error is None and not prediction.abstained


def validate_distribution(
    probabilities: Mapping[str, float],
    options: Sequence[str] | None = None,
) -> dict[str, float]:
    """Return a validated probability vector, or ``{}`` when none is available.

    A present vector must hold finite, non-negative values that sum to one
    within ``PROBABILITY_TOLERANCE``. When ``options`` is given, every key must
    be one of them; options missing from the vector have true zero probability.
    """
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping")
    if not probabilities:
        return {}
    allowed = None if options is None else set(options)
    values: dict[str, float] = {}
    for key, value in probabilities.items():
        option = str(key)
        if allowed is not None and option not in allowed:
            raise ValueError(f"probability key {option!r} is not a candidate option")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"probability for {option!r} must be a real number")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"probability for {option!r} must be finite and non-negative")
        values[option] = number
    total = math.fsum(values.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"probabilities must sum to 1 within {PROBABILITY_TOLERANCE} (sum={total!r})")
    return values


def _confidence(prediction: Prediction) -> float:
    value = prediction.confidence
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"confidence for {prediction.example_id!r} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not -CONFIDENCE_TOLERANCE <= number <= 1.0 + CONFIDENCE_TOLERANCE:
        raise ValueError(f"confidence for {prediction.example_id!r} must be finite and in [0, 1] (got {number!r})")
    return min(1.0, max(0.0, number))


def _index_examples(examples: Sequence[Example]) -> dict[str, Example]:
    rows: dict[str, Example] = {}
    for row in examples:
        if row.id in rows:
            raise ValueError(f"duplicate example id {row.id!r}")
        if row.answerable and (row.label is None or row.label not in row.options):
            raise ValueError(f"answerable example {row.id!r} needs a label among its options")
        if not row.answerable and row.label is not None:
            raise ValueError(f"unanswerable example {row.id!r} cannot have a label")
        rows[row.id] = row
    return rows


def validate_predictions(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
) -> tuple[dict[str, Example], dict[str, Prediction]]:
    """Validate rows and predictions together and index both by example id."""
    rows = _index_examples(examples)
    lookup: dict[str, Prediction] = {}
    for item in predictions:
        if item.example_id in lookup:
            raise ValueError(f"duplicate prediction for example id {item.example_id!r}")
        row = rows.get(item.example_id)
        if row is None:
            raise ValueError(f"prediction for unknown example id {item.example_id!r}")
        _confidence(item)
        if is_covered(item) and item.choice is None:
            raise ValueError(f"covered prediction {item.example_id!r} has no choice; set abstained or error")
        if item.latency_ms is not None:
            latency = float(item.latency_ms)
            if not math.isfinite(latency) or latency < 0.0:
                raise ValueError(f"latency for {item.example_id!r} must be finite and non-negative")
        validate_distribution(item.probabilities, row.options)
        lookup[item.example_id] = item
    return rows, lookup


def _is_correct(row: Example, item: Prediction | None) -> bool:
    return item is not None and is_covered(item) and row.answerable and item.choice == row.label


def _answerable(rows: Mapping[str, Example]) -> list[Example]:
    return [row for row in rows.values() if row.answerable]


def _covered(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> list[tuple[Example, Prediction]]:
    return [(row, lookup[row.id]) for row in rows.values() if row.id in lookup and is_covered(lookup[row.id])]


def _probability_rows(
    rows: Mapping[str, Example],
    lookup: Mapping[str, Prediction],
) -> list[tuple[Example, dict[str, float]]]:
    usable: list[tuple[Example, dict[str, float]]] = []
    for row in _answerable(rows):
        item = lookup.get(row.id)
        if item is None or item.error is not None:
            continue
        distribution = validate_distribution(item.probabilities, row.options)
        if distribution:
            usable.append((row, distribution))
    return usable


def _accuracy(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> float | None:
    answerable = _answerable(rows)
    if not answerable:
        return None
    return sum(_is_correct(row, lookup.get(row.id)) for row in answerable) / len(answerable)


def _macro_f1(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> float | None:
    answerable = _answerable(rows)
    if not answerable:
        return None
    decided: dict[str, str | None] = {}
    for row in answerable:
        item = lookup.get(row.id)
        decided[row.id] = item.choice if item is not None and is_covered(item) else None
    labels = sorted({str(row.label) for row in answerable} | {choice for choice in decided.values() if choice is not None})
    scores: list[float] = []
    for label in labels:
        true_positive = sum(row.label == label and decided[row.id] == label for row in answerable)
        false_positive = sum(row.label != label and decided[row.id] == label for row in answerable)
        false_negative = sum(row.label == label and decided[row.id] != label for row in answerable)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.mean(scores)


def _coverage(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> float | None:
    if not rows:
        return None
    return len(_covered(rows, lookup)) / len(rows)


def _selective_risk(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> float | None:
    covered = _covered(rows, lookup)
    if not covered:
        return None
    return sum(not _is_correct(row, item) for row, item in covered) / len(covered)


def _nll(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> float | None:
    usable = _probability_rows(rows, lookup)
    if not usable:
        return None
    return statistics.mean(-math.log(max(distribution.get(str(row.label), 0.0), NLL_PROBABILITY_FLOOR)) for row, distribution in usable)


def _brier(rows: Mapping[str, Example], lookup: Mapping[str, Prediction]) -> float | None:
    usable = _probability_rows(rows, lookup)
    if not usable:
        return None
    return statistics.mean(
        math.fsum((distribution.get(option, 0.0) - float(option == row.label)) ** 2 for option in row.options)
        for row, distribution in usable
    )


def _bin_index(confidence: float, bins: int) -> int:
    if confidence >= 1.0:
        return bins - 1
    index = min(bins - 1, int(confidence * bins))
    while index > 0 and confidence < index / bins:
        index -= 1
    while index < bins - 1 and confidence >= (index + 1) / bins:
        index += 1
    return index


def _ece(rows: Mapping[str, Example], lookup: Mapping[str, Prediction], bins: int) -> float | None:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    covered = _covered(rows, lookup)
    if not covered:
        return None
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for row, item in covered:
        confidence = _confidence(item)
        buckets[_bin_index(confidence, bins)].append((confidence, float(_is_correct(row, item))))
    total = 0.0
    for bucket in buckets:
        if bucket:
            mean_confidence = statistics.mean(value for value, _ in bucket)
            mean_correct = statistics.mean(value for _, value in bucket)
            total += len(bucket) / len(covered) * abs(mean_confidence - mean_correct)
    return total


def accuracy(examples: Sequence[Example], predictions: Sequence[Prediction]) -> float | None:
    return _accuracy(*validate_predictions(examples, predictions))


def macro_f1(examples: Sequence[Example], predictions: Sequence[Prediction]) -> float | None:
    """Multiclass macro-F1 over all answerable rows; missing, error, and abstention count as misses."""
    return _macro_f1(*validate_predictions(examples, predictions))


def nll(examples: Sequence[Example], predictions: Sequence[Prediction]) -> float | None:
    return _nll(*validate_predictions(examples, predictions))


def brier_score(examples: Sequence[Example], predictions: Sequence[Prediction]) -> float | None:
    """Standard multiclass Brier score summed over options (vons.metrics/v2)."""
    return _brier(*validate_predictions(examples, predictions))


def coverage(examples: Sequence[Example], predictions: Sequence[Prediction]) -> float | None:
    return _coverage(*validate_predictions(examples, predictions))


def selective_risk(examples: Sequence[Example], predictions: Sequence[Prediction]) -> float | None:
    return _selective_risk(*validate_predictions(examples, predictions))


def ece(examples: Sequence[Example], predictions: Sequence[Prediction], bins: int = 10) -> float | None:
    rows, lookup = validate_predictions(examples, predictions)
    return _ece(rows, lookup, bins)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(examples: Sequence[Example], predictions: Sequence[Prediction]) -> dict[str, object]:
    """Compute every metric with its version, definitions, and denominator counts."""
    rows, lookup = validate_predictions(examples, predictions)
    answerable = _answerable(rows)
    covered = _covered(rows, lookup)
    probability_rows = _probability_rows(rows, lookup)
    latencies = [float(item.latency_ms) for item in lookup.values() if item.latency_ms is not None]
    counts = {
        "rows": len(rows),
        "answerable_rows": len(answerable),
        "unanswerable_rows": len(rows) - len(answerable),
        "prediction_rows": len(lookup),
        "missing_prediction_rows": len(rows) - len(lookup),
        "error_prediction_rows": sum(item.error is not None for item in lookup.values()),
        "abstained_prediction_rows": sum(item.error is None and item.abstained for item in lookup.values()),
        "covered_rows": len(covered),
        "covered_unanswerable_rows": sum(not row.answerable for row, _ in covered),
        "correct_answerable_rows": sum(_is_correct(row, lookup.get(row.id)) for row in answerable),
        "probability_rows": len(probability_rows),
        "probability_unavailable_answerable_rows": len(answerable) - len(probability_rows),
        "nll_floor_clipped_rows": sum(
            distribution.get(str(row.label), 0.0) < NLL_PROBABILITY_FLOOR for row, distribution in probability_rows
        ),
        "ece_rows": len(covered),
        "latency_rows": len(latencies),
    }
    return {
        "metric_version": METRIC_VERSION,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "metric_policy": dict(METRIC_POLICY),
        "counts": counts,
        "probability_population": "complete" if counts["probability_unavailable_answerable_rows"] == 0 else "incomplete",
        "rows": len(examples),
        "accuracy_answerable": _accuracy(rows, lookup),
        "macro_f1_answerable": _macro_f1(rows, lookup),
        "coverage": _coverage(rows, lookup),
        "selective_risk": _selective_risk(rows, lookup),
        "nll_answerable": _nll(rows, lookup),
        "brier_answerable": _brier(rows, lookup),
        "ece": _ece(rows, lookup, 10),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 0.95) if latencies else None,
    }


def write_report(path: str | Path, *, config: Mapping[str, object], summary: Mapping[str, object], predictions: Iterable[Prediction]) -> None:
    """Write a strict-JSON report; NaN or Infinity anywhere raises ``ValueError`` before writing."""
    payload = json.dumps(
        {"config": dict(config), "summary": dict(summary), "predictions": [asdict(item) for item in predictions]},
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
