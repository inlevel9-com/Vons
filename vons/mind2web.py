"""Restricted Mind2Web-style candidate recall and selection evaluation.

This module intentionally evaluates only the two observable stages available
from a released candidate/target file. It never reports browser task success,
which would require an executor and environment state outside Vons.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Mind2WebExample:
    example_id: str
    candidate_ids: tuple[str, ...]
    target_id: str | None = None
    positive_ids: tuple[str, ...] = ()
    no_positive: bool = False
    task_id: str | None = None
    action_id: str | None = None
    split: str | None = None
    website: str | None = None
    domain: str | None = None

    def __post_init__(self) -> None:
        example_id = str(self.example_id)
        candidate_ids = tuple(self.candidate_ids)
        target_id = None if self.target_id in (None, "") else self.target_id
        positive_ids = tuple(self.positive_ids)

        if not example_id:
            raise ValueError("Mind2Web row requires an id")
        if any(not isinstance(item, str) or not item for item in candidate_ids):
            raise TypeError("Mind2Web candidate ids must be non-empty strings")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Mind2Web candidates must be unique")
        if target_id is not None and (not isinstance(target_id, str) or not target_id):
            raise TypeError("Mind2Web target_id must be a non-empty string")
        if any(not isinstance(item, str) or not item for item in positive_ids):
            raise TypeError("Mind2Web positive ids must be non-empty strings")
        if len(positive_ids) != len(set(positive_ids)):
            raise ValueError("Mind2Web positive ids must be unique")
        if target_id is not None:
            if positive_ids and target_id not in positive_ids:
                raise ValueError("target_id must be one of positive_ids")
            if not positive_ids:
                positive_ids = (target_id,)
        if self.no_positive and positive_ids:
            raise ValueError("no_positive rows cannot have positive ids")
        if not self.no_positive and not positive_ids:
            raise ValueError("Mind2Web row requires positive_ids or no_positive=true")

        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "target_id", target_id or (positive_ids[0] if positive_ids else None))
        object.__setattr__(self, "positive_ids", positive_ids)
        for field_name in ("task_id", "action_id", "split", "website", "domain"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, None if value is None else str(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Mind2WebExample:
        if not isinstance(value, Mapping):
            raise TypeError("Mind2Web row must be an object")
        candidates = _string_sequence(value.get("candidate_ids", value.get("candidates", ())), "candidates")
        positive_value = value.get("positive_ids", value.get("target_ids", value.get("targets")))
        legacy_target = value.get("target_id")
        no_positive = value.get("no_positive", False)
        if not isinstance(no_positive, bool):
            raise TypeError("Mind2Web no_positive must be a boolean")

        if positive_value is None and legacy_target not in (None, ""):
            positive_ids = (str(legacy_target),)
        elif positive_value is None:
            positive_ids = ()
        else:
            positive_ids = _string_sequence(positive_value, "positive_ids")
            if legacy_target not in (None, "") and str(legacy_target) not in positive_ids:
                raise ValueError("target_id must be one of positive_ids")

        if not value.get("id"):
            raise ValueError("Mind2Web row requires an id")
        if len(candidates) != len(set(candidates)):
            raise ValueError("Mind2Web candidates must be unique")
        if not positive_ids and not no_positive:
            raise ValueError("Mind2Web row requires positive_ids or no_positive=true")

        return cls(
            example_id=str(value["id"]),
            candidate_ids=candidates,
            target_id=positive_ids[0] if positive_ids else None,
            positive_ids=positive_ids,
            no_positive=no_positive,
            task_id=_optional_string(value.get("task_id")),
            action_id=_optional_string(value.get("action_id")),
            split=_optional_string(value.get("split")),
            website=_optional_string(value.get("website")),
            domain=_optional_string(value.get("domain")),
        )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"Mind2Web {label} must be a list")
    values: list[str] = []
    for item in value:
        candidate = item.get("id") if isinstance(item, Mapping) else item
        if not isinstance(candidate, str) or not candidate:
            raise TypeError(f"Mind2Web {label} ids must be non-empty strings")
        values.append(candidate)
    return tuple(values)


@dataclass(frozen=True)
class Mind2WebMetrics:
    rows: int
    candidate_recall: float | None
    selection_accuracy_given_recall: float | None
    complete_case_selection_accuracy_given_recall: float | None
    evaluated_recalled_rows: int
    positive_rows: int
    no_positive_rows: int
    recalled_positive_rows: int
    correct_selections: int
    missing_predictions: int
    invalid_selections: int
    k: int | None = None
    candidate_recall_task_macro: float | None = None
    selection_accuracy_given_recall_task_macro: float | None = None
    task_group_count: int = 0

    def to_mapping(self) -> dict[str, int | float | str | None]:
        return {
            "rows": self.rows,
            "candidate_recall": self.candidate_recall,
            "selection_accuracy_given_recall": self.selection_accuracy_given_recall,
            "complete_case_selection_accuracy_given_recall": self.complete_case_selection_accuracy_given_recall,
            "evaluated_recalled_rows": self.evaluated_recalled_rows,
            "positive_rows": self.positive_rows,
            "no_positive_rows": self.no_positive_rows,
            "recalled_positive_rows": self.recalled_positive_rows,
            "correct_selections": self.correct_selections,
            "missing_predictions": self.missing_predictions,
            "invalid_selections": self.invalid_selections,
            "k": self.k,
            "candidate_recall_step_micro": self.candidate_recall,
            "selection_accuracy_given_recall_step_micro": self.selection_accuracy_given_recall,
            "candidate_recall_task_macro": self.candidate_recall_task_macro,
            "selection_accuracy_given_recall_task_macro": self.selection_accuracy_given_recall_task_macro,
            "task_group_count": self.task_group_count,
            "scope": "candidate recall and candidate-in-set selection only; no browser task success",
        }


def evaluate_mind2web(
    examples: Sequence[Mind2WebExample],
    generated_candidates: Mapping[str, Sequence[str]],
    selections: Mapping[str, str | None],
    *,
    k: int | None = None,
) -> Mind2WebMetrics:
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    positive_rows = sum(not row.no_positive for row in examples)
    no_positive_rows = sum(row.no_positive for row in examples)
    recalled = 0
    selected_correct = 0
    evaluated_recalled_rows = 0
    missing_predictions = 0
    invalid_selections = 0
    task_groups: dict[str, dict[str, int]] = {}
    for row in examples:
        group_key = row.task_id or row.example_id
        group = task_groups.setdefault(group_key, {"positive": 0, "recalled": 0, "correct": 0})
        if not row.no_positive:
            group["positive"] += 1
        candidates = tuple(str(item) for item in generated_candidates.get(row.example_id, ()))
        if k is not None:
            candidates = candidates[:k]
        if row.no_positive or not set(row.positive_ids).intersection(candidates):
            continue
        recalled += 1
        group["recalled"] += 1
        selection = selections.get(row.example_id)
        if selection is None:
            missing_predictions += 1
            continue
        evaluated_recalled_rows += 1
        if str(selection) not in candidates:
            invalid_selections += 1
            continue
        if str(selection) in row.positive_ids:
            selected_correct += 1
            group["correct"] += 1
    task_recall_values = [
        group["recalled"] / group["positive"]
        for group in task_groups.values()
        if group["positive"]
    ]
    task_selection_values = [
        group["correct"] / group["recalled"]
        for group in task_groups.values()
        if group["recalled"]
    ]
    return Mind2WebMetrics(
        rows=len(examples),
        candidate_recall=recalled / positive_rows if positive_rows else None,
        selection_accuracy_given_recall=selected_correct / recalled if recalled else None,
        complete_case_selection_accuracy_given_recall=(
            selected_correct / evaluated_recalled_rows if evaluated_recalled_rows else None
        ),
        evaluated_recalled_rows=evaluated_recalled_rows,
        positive_rows=positive_rows,
        no_positive_rows=no_positive_rows,
        recalled_positive_rows=recalled,
        correct_selections=selected_correct,
        missing_predictions=missing_predictions,
        invalid_selections=invalid_selections,
        k=k,
        candidate_recall_task_macro=(
            sum(task_recall_values) / len(task_recall_values) if task_recall_values else None
        ),
        selection_accuracy_given_recall_task_macro=(
            sum(task_selection_values) / len(task_selection_values) if task_selection_values else None
        ),
        task_group_count=len(task_groups),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def task_bootstrap_intervals(
    examples: Sequence[Mind2WebExample],
    generated_candidates: Mapping[str, Sequence[str]],
    selections: Mapping[str, str | None],
    *,
    k: int | None = None,
    draws: int = 1000,
    seed: int = 7,
) -> dict[str, Any]:
    """Bootstrap task-macro recall and conditional selection at a fixed k.

    Sampling is over task groups, not individual action rows. Missing and
    invalid selections remain errors in the primary recalled-row denominator.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    groups: dict[str, dict[str, int]] = {}
    for row in examples:
        group = groups.setdefault(row.task_id or row.example_id, {"positive": 0, "recalled": 0, "correct": 0})
        if row.no_positive:
            continue
        group["positive"] += 1
        candidates = tuple(str(item) for item in generated_candidates.get(row.example_id, ()))
        if k is not None:
            candidates = candidates[:k]
        if not set(row.positive_ids).intersection(candidates):
            continue
        group["recalled"] += 1
        selection = selections.get(row.example_id)
        if selection is not None and str(selection) in candidates and str(selection) in row.positive_ids:
            group["correct"] += 1

    recall_values = [
        group["recalled"] / group["positive"] for group in groups.values() if group["positive"]
    ]
    selection_values = [
        group["correct"] / group["recalled"] for group in groups.values() if group["recalled"]
    ]
    rng = random.Random(seed)

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        samples = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws)]
        return [_percentile(samples, 0.025), _percentile(samples, 0.975)]

    return {
        "unit": "task_id_or_example_id",
        "task_group_count": len(groups),
        "draws": draws,
        "seed": seed,
        "candidate_recall_task_macro_ci95": interval(recall_values),
        "selection_accuracy_given_recall_task_macro_ci95": interval(selection_values),
    }


def recall_at_k(
    examples: Sequence[Mind2WebExample],
    generated_candidates: Mapping[str, Sequence[str]],
    k: int,
) -> float | None:
    """Return positive-row candidate recall at ``k``; undefined is ``None``."""

    return evaluate_mind2web(examples, generated_candidates, {}, k=k).candidate_recall
