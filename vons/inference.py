"""Runtime inference helpers for Direct and Diffusion decision backends.

This module owns the per-request tensor-preparation contract: tokenize each
candidate, measure the longest live tokenized sequence, and pad tensors to
that length (bounded at the manifest budget, usually 512). The exported ONNX
graphs declare dynamic ``options`` and ``sequence_length`` axes so the same
weight file can run with either fixed or variable-length feeds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import Backend, DecisionRequest, Question

MAX_SEQUENCE_BUDGET = 512


@dataclass(frozen=True)
class TokenizedCandidate:
    """Single candidate after tokenization, before padding."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class PaddedCandidates:
    """Per-request tensors padded to the longest live candidate length."""

    input_ids: Any
    attention_mask: Any
    token_type_ids: Any
    option_mask: Any
    live_candidates: int
    allocated_candidates: int
    live_sequence_length: int
    max_sequence_budget: int


def serialize_state(state: str | Mapping[str, Any]) -> str:
    """Return a canonical string form for the request state."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def candidate_text(state: str | Mapping[str, Any], question: Question, option: str) -> str:
    """Render a single candidate using the same template as the web runtime."""
    return f"{serialize_state(state)}\nQuestion: {question.prompt}\nCandidate: {option}"


def candidate_list_text(state: str | Mapping[str, Any], question: Question, options: Sequence[str]) -> str:
    """Aggregate template used to sanity-check the token budget."""
    joined = "\n".join(str(option) for option in options)
    return f"{serialize_state(state)}\nQuestion: {question.prompt}\nCandidates:\n{joined}"


def _require_transformers_tokenizer(tokenizer: Any) -> None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose encode()")
    if not hasattr(tokenizer, "pad_token_id") or not hasattr(tokenizer, "cls_token_id"):
        raise TypeError("tokenizer must expose pad_token_id and cls_token_id")


def _encode_one(tokenizer: Any, text: str, max_sequence_budget: int) -> tuple[list[int], list[int], list[int]]:
    """Encode a single candidate string with the tokenizer's standard output."""
    result = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_sequence_budget,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    if hasattr(result, "ids"):
        ids = list(result.ids)
        attention_mask = list(result.attention_mask)
        token_type_ids = list(result.token_type_ids)
    elif isinstance(result, Mapping):
        ids = [int(value) for value in result["input_ids"]]
        attention_mask = [int(value) for value in result["attention_mask"]]
        token_type_ids = [int(value) for value in result.get("token_type_ids", [0] * len(ids))]
    else:
        ids = [int(value) for value in result]
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
    if len(ids) != len(attention_mask) or len(ids) != len(token_type_ids):
        raise ValueError("tokenizer returned inconsistent input/attention/type lengths")
    if any(not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError("tokenizer returned an invalid token id")
    return ids, attention_mask, token_type_ids


def tokenize_candidates(
    tokenizer: Any,
    state: str | Mapping[str, Any],
    question: Question,
    options: Sequence[str],
    *,
    max_sequence_budget: int = MAX_SEQUENCE_BUDGET,
) -> list[TokenizedCandidate]:
    """Tokenize each candidate independently.

    The individual candidates are each truncated to ``max_sequence_budget`` so
    downstream code can treat an over-length request as a validation error
    rather than silently dropping tokens from the middle of a prompt.
    """
    if not 1 <= max_sequence_budget <= MAX_SEQUENCE_BUDGET:
        raise ValueError(f"max_sequence_budget must be in 1..{MAX_SEQUENCE_BUDGET}")
    _require_transformers_tokenizer(tokenizer)
    aggregate = tokenizer.encode(candidate_list_text(state, question, options), add_special_tokens=True)
    aggregate_tokens = len(aggregate.ids) if hasattr(aggregate, "ids") else len(aggregate)
    if aggregate_tokens > max_sequence_budget:
        raise ValueError(
            f"question {question.id!r} exceeds the {max_sequence_budget}-token aggregate input budget"
            f" (tokens={aggregate_tokens})"
        )
    out: list[TokenizedCandidate] = []
    for option in options:
        ids, attention_mask, token_type_ids = _encode_one(
            tokenizer,
            candidate_text(state, question, option),
            max_sequence_budget,
        )
        if len(ids) > max_sequence_budget:
            raise ValueError(
                f"question {question.id!r} candidate has {len(ids)} tokens;"
                f" the bundle limit is {max_sequence_budget}"
            )
        out.append(TokenizedCandidate(tuple(ids), tuple(attention_mask), tuple(token_type_ids)))
    return out


def compute_live_sequence_length(
    tokenized: Sequence[TokenizedCandidate],
    *,
    max_sequence_budget: int = MAX_SEQUENCE_BUDGET,
) -> int:
    """Return the longest live candidate length, clamped to the budget.

    An empty candidate set is not valid; callers validate option counts
    against the contract before reaching this helper.
    """
    if not tokenized:
        raise ValueError("at least one tokenized candidate is required")
    if not 1 <= max_sequence_budget <= MAX_SEQUENCE_BUDGET:
        raise ValueError(f"max_sequence_budget must be in 1..{MAX_SEQUENCE_BUDGET}")
    longest = max(candidate.token_count for candidate in tokenized)
    if longest < 1:
        raise ValueError("each candidate must contain at least one token")
    return min(longest, max_sequence_budget)


def pad_candidates(
    tokenized: Sequence[TokenizedCandidate],
    *,
    slots: int,
    live_sequence_length: int,
    padding_id: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[int]]:
    """Pad candidates to ``slots`` x ``live_sequence_length`` using ints.

    The caller owns conversion to backend-specific tensors (torch, numpy,
    ort, etc.). Option mask slots beyond ``len(tokenized)`` remain zero so
    diffusion and direct heads can treat them as masked-out padding.
    """
    if not isinstance(slots, int) or slots < len(tokenized) or slots < 1:
        raise ValueError("slots must be at least len(tokenized) and positive")
    if not isinstance(live_sequence_length, int) or live_sequence_length < 1:
        raise ValueError("live_sequence_length must be positive")
    if not isinstance(padding_id, int) or padding_id < 0:
        raise ValueError("padding_id must be a non-negative integer")
    input_ids: list[list[int]] = [[padding_id] * live_sequence_length for _ in range(slots)]
    attention_mask: list[list[int]] = [[0] * live_sequence_length for _ in range(slots)]
    token_type_ids: list[list[int]] = [[0] * live_sequence_length for _ in range(slots)]
    option_mask: list[int] = [0] * slots
    for option_index, candidate in enumerate(tokenized):
        option_mask[option_index] = 1
        if candidate.token_count > live_sequence_length:
            raise ValueError(
                f"candidate {option_index} has {candidate.token_count} tokens;"
                f" cannot fit in live_sequence_length={live_sequence_length}"
            )
        for token_index in range(candidate.token_count):
            input_ids[option_index][token_index] = candidate.input_ids[token_index]
            attention_mask[option_index][token_index] = candidate.attention_mask[token_index]
            token_type_ids[option_index][token_index] = candidate.token_type_ids[token_index]
    return input_ids, attention_mask, token_type_ids, option_mask


def resolve_allocated_slots(
    backend: Backend,
    live_candidates: int,
    *,
    option_count: int,
    direct_option_slots: int | None = None,
) -> int:
    """Return the number of candidate slots to allocate for one question.

    * ``diffusion`` always uses ``option_count`` (its noise vector and the
      exported head are sized to the full slot budget).
    * ``direct`` defaults to the live candidate count when no override is
      supplied, otherwise the caller's ``direct_option_slots`` experiment
      value, clamped to ``[live_candidates, option_count]``.
    """
    if not 2 <= option_count <= 32:
        raise ValueError("option_count must be in 2..32 (vons manifest bound)")
    if not 1 <= live_candidates <= option_count:
        raise ValueError("live_candidates must be in 1..option_count")
    if backend is Backend.DIFFUSION:
        return option_count
    if direct_option_slots is None:
        return live_candidates
    if not isinstance(direct_option_slots, int):
        raise TypeError("direct_option_slots must be an integer or None")
    if not live_candidates <= direct_option_slots <= option_count:
        raise ValueError(
            f"direct_option_slots must be in [{live_candidates}, {option_count}]"
        )
    return direct_option_slots


def prepare_question_tensors(
    tokenizer: Any,
    request: DecisionRequest,
    question: Question,
    *,
    option_count: int,
    max_sequence_budget: int = MAX_SEQUENCE_BUDGET,
    direct_option_slots: int | None = None,
) -> tuple[PaddedCandidates, list[TokenizedCandidate]]:
    """Tokenize, measure, and pad a single question.

    Returns the padded container plus the raw tokenized candidates for
    callers that need per-option token counts (benchmarks, traces).
    """
    if max_sequence_budget > MAX_SEQUENCE_BUDGET:
        raise ValueError(f"max_sequence_budget may not exceed {MAX_SEQUENCE_BUDGET}")
    options = question.options if question.options else ("true", "false")
    if len(options) == 0 or len(options) > option_count:
        raise ValueError(f"question {question.id!r} has unsupported candidate count")
    tokenized = tokenize_candidates(
        tokenizer,
        request.state,
        question,
        options,
        max_sequence_budget=max_sequence_budget,
    )
    live_candidates = len(tokenized)
    live_sequence_length = compute_live_sequence_length(
        tokenized,
        max_sequence_budget=max_sequence_budget,
    )
    slots = resolve_allocated_slots(
        request.backend,
        live_candidates,
        option_count=option_count,
        direct_option_slots=direct_option_slots,
    )
    padding_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids, attention_mask, token_type_ids, option_mask = pad_candidates(
        tokenized,
        slots=slots,
        live_sequence_length=live_sequence_length,
        padding_id=padding_id,
    )
    padded = PaddedCandidates(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        option_mask=option_mask,
        live_candidates=live_candidates,
        allocated_candidates=slots,
        live_sequence_length=live_sequence_length,
        max_sequence_budget=max_sequence_budget,
    )
    return padded, tokenized
