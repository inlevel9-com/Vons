"""Label-free context compression for offline Mind2Web selection.

The adapter is deliberately independent of the evaluator and gold labels. It
keeps only the last two interaction-history entries, removes presentation-only
DOM attributes, limits each candidate with the exact runtime tokenizer, and
then reduces candidate/state text until the serialized inference prompt is at
most 512 tokens.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

CONTEXT_ADAPTER_VERSION = "vons.mind2web-context/v1"
MAX_HISTORY_ACTIONS = 2
MAX_CANDIDATE_TOKENS = 15
MAX_PROMPT_TOKENS = 512

_SVG_BLOCK = re.compile(r"<svg\b[^>]*>.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)
_ATTRIBUTE = re.compile(
    r"(?<!\S)([A-Za-z_:][-A-Za-z0-9_:.]*)=(\"[^\"]*\"|'[^']*'|\S+)"
)
_WHITESPACE = re.compile(r"\s+")
_KEPT_ATTRIBUTES = frozenset({"type", "aria-label", "id"})


class _Encoding(Protocol):
    ids: Sequence[int]


class ExactTokenizer(Protocol):
    def encode(self, text: str) -> _Encoding: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class CompressionStats:
    version: str
    original_prompt_tokens: int
    prompt_tokens: int
    candidate_token_budget: int
    max_candidate_tokens: int
    history_actions_before: int
    history_actions_after: int
    candidates: int


def strip_dom_presentation(text: str) -> str:
    """Keep visible text, tag names, and the small semantic attribute allowlist."""
    value = _SVG_BLOCK.sub(" ", text)
    value = _STYLE_BLOCK.sub(" ", value)

    def replace_attribute(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        return match.group(0) if name in _KEPT_ATTRIBUTES else ""

    value = _ATTRIBUTE.sub(replace_attribute, value)
    return _WHITESPACE.sub(" ", value).strip()


def _token_ids(tokenizer: ExactTokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text).ids)


def _decode_prefix(tokenizer: ExactTokenizer, ids: Sequence[int], limit: int) -> str:
    if limit <= 0 or not ids:
        return ""
    prefix = list(ids[:limit])
    value = tokenizer.decode(prefix, skip_special_tokens=True).strip()
    while value and len(_token_ids(tokenizer, value)) > limit:
        prefix.pop()
        value = tokenizer.decode(prefix, skip_special_tokens=True).strip()
    return value


def _state_mapping(value: object) -> dict[str, object]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise TypeError("Mind2Web state must be a JSON object")
    goal = parsed.get("goal", "")
    history = parsed.get("previous_actions", [])
    if not isinstance(goal, str):
        raise TypeError("Mind2Web goal must be a string")
    if not isinstance(history, list) or any(not isinstance(item, str) for item in history):
        raise TypeError("Mind2Web previous_actions must be a string array")
    return {"goal": goal, "previous_actions": history[-MAX_HISTORY_ACTIONS:]}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _prompt_text(state: Mapping[str, object], question: str, candidates: Sequence[str]) -> str:
    prefix = f"{_canonical(state)}\nQuestion: {question}"
    return prefix + "\nCandidates:\n" + "\n".join(candidates)


def _truncate_goal_to_fit(
    tokenizer: ExactTokenizer,
    state: dict[str, object],
    question: str,
    candidates: Sequence[str],
    maximum: int,
) -> dict[str, object]:
    candidate_state = {"goal": str(state["goal"]), "previous_actions": []}
    if len(_token_ids(tokenizer, _prompt_text(candidate_state, question, candidates))) <= maximum:
        return candidate_state
    goal_ids = _token_ids(tokenizer, str(state["goal"]))
    left, right = 0, len(goal_ids)
    best = {"goal": "", "previous_actions": []}
    while left <= right:
        middle = (left + right) // 2
        trial = {
            "goal": _decode_prefix(tokenizer, goal_ids, middle),
            "previous_actions": [],
        }
        if len(_token_ids(tokenizer, _prompt_text(trial, question, candidates))) <= maximum:
            best = trial
            left = middle + 1
        else:
            right = middle - 1
    return best


def compress_request(
    request: Mapping[str, Any],
    tokenizer: ExactTokenizer,
    *,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    max_candidate_tokens: int = MAX_CANDIDATE_TOKENS,
) -> tuple[dict[str, object], CompressionStats]:
    """Return an inference request that is guaranteed to fit the hard budget."""
    if max_prompt_tokens <= 0 or max_candidate_tokens < 0:
        raise ValueError("token budgets must be positive")
    question = request.get("question")
    candidates = request.get("candidates")
    if not isinstance(question, str) or not isinstance(candidates, list):
        raise TypeError("Mind2Web request requires question and candidates")
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("id"), str)
        or not isinstance(item.get("text"), str)
        for item in candidates
    ):
        raise TypeError("Mind2Web candidates require string id and text")

    raw_state = _state_mapping(request.get("state"))
    raw_history = request.get("state")
    if isinstance(raw_history, str):
        raw_history = json.loads(raw_history)
    history_before = (
        len(raw_history.get("previous_actions", []))
        if isinstance(raw_history, Mapping)
        and isinstance(raw_history.get("previous_actions", []), list)
        else 0
    )
    raw_texts = [str(item["text"]) for item in candidates]
    original_prompt_tokens = len(
        _token_ids(tokenizer, _prompt_text(raw_state, question, raw_texts))
    )
    cleaned = [strip_dom_presentation(text) for text in raw_texts]
    candidate_ids = [_token_ids(tokenizer, text) for text in cleaned]

    def texts_for_budget(budget: int) -> list[str]:
        return [_decode_prefix(tokenizer, ids, budget) for ids in candidate_ids]

    left, right = 0, max_candidate_tokens
    selected_budget = 0
    selected_texts = texts_for_budget(0)
    while left <= right:
        middle = (left + right) // 2
        trial_texts = texts_for_budget(middle)
        token_count = len(
            _token_ids(tokenizer, _prompt_text(raw_state, question, trial_texts))
        )
        if token_count <= max_prompt_tokens:
            selected_budget = middle
            selected_texts = trial_texts
            left = middle + 1
        else:
            right = middle - 1

    state = raw_state
    if len(_token_ids(tokenizer, _prompt_text(state, question, selected_texts))) > max_prompt_tokens:
        selected_budget = 0
        selected_texts = texts_for_budget(0)
        state = _truncate_goal_to_fit(
            tokenizer,
            raw_state,
            question,
            selected_texts,
            max_prompt_tokens,
        )

    prompt_tokens = len(_token_ids(tokenizer, _prompt_text(state, question, selected_texts)))
    if prompt_tokens > max_prompt_tokens:
        raise ValueError("fixed Mind2Web prompt scaffolding exceeds the hard token budget")
    compressed_candidates = [
        {"id": str(item["id"]), "text": text}
        for item, text in zip(candidates, selected_texts, strict=True)
    ]
    max_observed_candidate_tokens = max(
        (len(_token_ids(tokenizer, text)) for text in selected_texts),
        default=0,
    )
    compressed = {
        "state": _canonical(state),
        "question": question,
        "candidates": compressed_candidates,
    }
    stats = CompressionStats(
        version=CONTEXT_ADAPTER_VERSION,
        original_prompt_tokens=original_prompt_tokens,
        prompt_tokens=prompt_tokens,
        candidate_token_budget=selected_budget,
        max_candidate_tokens=max_observed_candidate_tokens,
        history_actions_before=history_before,
        history_actions_after=len(state["previous_actions"]),
        candidates=len(compressed_candidates),
    )
    return compressed, stats
