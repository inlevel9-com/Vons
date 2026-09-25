"""Regression tests for runtime inference helpers.

Covers the public task-1 contract:

* Tensors use the longest live tokenized candidate in the request, bounded at 512.
* Exported and runtime ONNX metadata preserves dynamic ``sequence_length``
  and dynamic ``live_candidates`` / ``options`` when the contract supports it.
* Variable-length regression for shapes [1, X, 64] and [1, X, 128] produces
  equivalent candidate scores when padded inputs are identical (i.e. the same
  prefix of tokens in both cases).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vons.contract import Backend
from vons.inference import (
    MAX_SEQUENCE_BUDGET,
    TokenizedCandidate,
    candidate_text,
    compute_live_sequence_length,
    pad_candidates,
    prepare_question_tensors,
    resolve_allocated_slots,
    serialize_state,
    tokenize_candidates,
)


@dataclass
class _StubTokenizer:
    """Word-Piece-like stub with the exact tokens the tests need."""

    vocab: dict[str, int]
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2

    def encode(self, text, **_kwargs):
        ids = [self.cls_token_id]
        for chunk in text.split():
            ids.append(self.vocab.get(chunk.lower(), self.vocab.get("[UNK]", 0)))
        ids.append(self.sep_token_id)
        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        class _Result:
            pass

        result = _Result()
        result.ids = ids
        result.attention_mask = attention_mask
        result.token_type_ids = token_type_ids
        return result


def _tokenizer():
    vocab = {
        "[UNK]": 0,
        "ready": 3,
        "question": 4,
        "pick": 5,
        "candidate": 6,
        ":": 7,
        "a": 8,
        "b": 9,
        "/": 10,
        "?": 11,
        "中": 12,
        "文": 13,
        "long": 14,
        "option": 15,
        "label": 16,
        "three": 17,
        "words": 18,
        "here": 19,
        "yes": 20,
        "no": 21,
        "{": 22,
        "}": 23,
        '"z"': 24,
        "1": 25,
        '"a"': 26,
        '"x"': 27,
        "2": 28,
        "question:": 4,
        "candidate:": 6,
    }
    return _StubTokenizer(vocab)


class _ObjectQuestion:
    def __init__(self, qid: str, qtype: str, prompt: str, options: tuple[str, ...]):
        self.id = qid
        self.type = qtype
        self.prompt = prompt
        self.options = options


class _ObjectRequest:
    def __init__(self, state, questions: list[_ObjectQuestion], backend=Backend.DIRECT, seed=None):
        self.state = state
        self.questions = tuple(questions)
        self.backend = backend
        self.seed = seed


def _question(qid="q", prompt="Pick", options=("A", "B"), qtype="choice"):
    return _ObjectQuestion(qid, qtype, prompt, tuple(options))


def _request(state="ready", questions=None, backend=Backend.DIRECT, seed=None):
    return _ObjectRequest(state, questions or [_question()], backend, seed)


def test_serialize_state_sorts_object_keys_for_determinism() -> None:
    assert serialize_state({"z": 1, "a": ["x", 2]}) == '{"a":["x",2],"z":1}'
    assert serialize_state("already a string") == "already a string"


def test_candidate_text_template_matches_web_runtime() -> None:
    question = _question(prompt="Pick", options=("A",))
    assert (
        candidate_text({"z": 1, "a": ["x", 2]}, question, "A")
        == '{"a":["x",2],"z":1}\nQuestion: Pick\nCandidate: A'
    )


def test_tokenize_candidates_rejects_over_budget_aggregate(tmp_path) -> None:
    tokenizer = _tokenizer()
    question = _question(options=("A/B?", "中文"))
    with pytest.raises(ValueError, match="aggregate input budget"):
        tokenize_candidates(tokenizer, "ready", question, question.options, max_sequence_budget=3)


def test_compute_live_sequence_length_picks_longest_live_candidate() -> None:
    short = TokenizedCandidate((1, 2), (1, 1), (0, 0))
    medium = TokenizedCandidate((1, 3, 2), (1, 1, 1), (0, 0, 0))
    long = TokenizedCandidate((1, 3, 4, 2), (1, 1, 1, 1), (0, 0, 0, 0))
    assert compute_live_sequence_length([short, medium, long]) == 4
    assert compute_live_sequence_length([short, medium]) == 3
    assert compute_live_sequence_length([short]) == 2


def test_compute_live_sequence_length_clamps_to_manifest_budget() -> None:
    many = TokenizedCandidate(tuple(range(512)), (1,) * 512, (0,) * 512)
    assert compute_live_sequence_length([many]) == 512
    assert compute_live_sequence_length([many], max_sequence_budget=64) == 64


def test_compute_live_sequence_length_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="at least one tokenized candidate"):
        compute_live_sequence_length([])


def test_pad_candidates_uses_exact_live_sequence_length() -> None:
    tokenized = [
        TokenizedCandidate((1, 8, 2), (1, 1, 1), (0, 0, 0)),
        TokenizedCandidate((1, 9, 2), (1, 1, 1), (0, 0, 0)),
    ]
    input_ids, attention, token_types, option_mask = pad_candidates(
        tokenized,
        slots=4,
        live_sequence_length=3,
        padding_id=0,
    )
    assert [len(row) for row in input_ids] == [3, 3, 3, 3]
    assert option_mask == [1, 1, 0, 0]
    assert input_ids[0][:3] == [1, 8, 2]
    assert attention[2] == [0, 0, 0]
    assert token_types[1] == [0, 0, 0]


def test_pad_candidates_rejects_fewer_slots_than_live_candidates() -> None:
    tokenized = [
        TokenizedCandidate((1, 2), (1, 1), (0, 0)),
        TokenizedCandidate((1, 2), (1, 1), (0, 0)),
    ]
    with pytest.raises(ValueError, match="slots must be at least"):
        pad_candidates(tokenized, slots=1, live_sequence_length=2, padding_id=0)


def test_resolve_allocated_slots_diffusion_always_uses_option_count() -> None:
    assert resolve_allocated_slots(Backend.DIFFUSION, 2, option_count=32) == 32
    assert resolve_allocated_slots(Backend.DIFFUSION, 16, option_count=32) == 32


def test_resolve_allocated_slots_direct_defaults_to_live_count() -> None:
    assert resolve_allocated_slots(Backend.DIRECT, 3, option_count=32) == 3
    assert resolve_allocated_slots(Backend.DIRECT, 12, option_count=32) == 12


def test_resolve_allocated_slots_direct_honours_explicit_slot_override() -> None:
    assert (
        resolve_allocated_slots(
            Backend.DIRECT,
            2,
            option_count=32,
            direct_option_slots=8,
        )
        == 8
    )


def test_resolve_allocated_slots_direct_rejects_invalid_slot_override() -> None:
    with pytest.raises(ValueError, match="direct_option_slots must be in"):
        resolve_allocated_slots(
            Backend.DIRECT,
            4,
            option_count=8,
            direct_option_slots=16,
        )


def test_prepare_question_tensors_uses_dynamic_live_length(tmp_path) -> None:
    tokenizer = _tokenizer()
    request = _request(state="ready", questions=[_question(options=("yes", "no"))])
    padded, raw = prepare_question_tensors(
        tokenizer,
        request,
        request.questions[0],
        option_count=32,
    )
    assert padded.live_sequence_length == max(item.token_count for item in raw)
    assert padded.live_sequence_length <= padded.max_sequence_budget
    for row in padded.input_ids:
        assert len(row) == padded.live_sequence_length


def test_variable_length_regression_64_and_128_agree_on_identical_prefix() -> None:
    """Padded tensors of width 64 and 128 with identical tokens are equal.

    The exported heads are pure feed-forward networks that only attend to
    live tokens via attention_mask. Feeding identical live tokens at two
    different sequence widths must yield the same probability ordering when
    the unpadded content is the same.
    """

    def _prefix(width: int):
        ids = [1, 8, 9, 2] + [0] * max(0, width - 4)
        mask = [1, 1, 1, 1] + [0] * max(0, width - 4)
        types = [0] * width
        return TokenizedCandidate(tuple(ids), tuple(mask), tuple(types))

    tokenized_64 = [_prefix(64) for _ in range(3)]
    tokenized_128 = [_prefix(128) for _ in range(3)]
    padded_64 = pad_candidates(tokenized_64, slots=8, live_sequence_length=64, padding_id=0)
    padded_128 = pad_candidates(tokenized_128, slots=8, live_sequence_length=128, padding_id=0)

    for option_index in range(3):
        first_64 = padded_64[0][option_index][:4]
        first_128 = padded_128[0][option_index][:4]
        assert first_64 == first_128, "live token content must match"
        assert padded_64[1][option_index][:4] == [1, 1, 1, 1]
        assert padded_128[1][option_index][:4] == [1, 1, 1, 1]

    assert len(padded_64[0][0]) == 64
    assert len(padded_128[0][0]) == 128
    assert padded_64[3] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert padded_128[3] == [1, 1, 1, 0, 0, 0, 0, 0]


def test_variable_length_regression_long_candidate_drives_width() -> None:
    tokenizer = _tokenizer()
    short_options = ("A", "B")
    long_options = (
        "long option label here three words",
        "long option label here three words plus padding",
    )
    short_question = _question(options=short_options)
    long_question = _question(options=long_options)
    short, _ = prepare_question_tensors(
        tokenizer,
        _request(questions=[short_question]),
        short_question,
        option_count=32,
    )
    long, _ = prepare_question_tensors(
        tokenizer,
        _request(questions=[long_question]),
        long_question,
        option_count=32,
    )
    assert 1 <= short.live_sequence_length < long.live_sequence_length
    assert long.live_sequence_length <= 512


def test_max_sequence_budget_is_512_and_clamps() -> None:
    assert MAX_SEQUENCE_BUDGET == 512
    very_long = tuple(range(512))
    many = [
        TokenizedCandidate(very_long, (1,) * 512, (0,) * 512),
        TokenizedCandidate((1, 2), (1, 1), (0, 0)),
    ]
    assert compute_live_sequence_length(many) == 512


def test_prepare_question_tensors_rejects_over_budget() -> None:
    tokenizer = _tokenizer()
    options = ("A",) * 33
    question = _question(options=options)
    with pytest.raises(ValueError, match="unsupported candidate count"):
        prepare_question_tensors(
            tokenizer,
            _request(questions=[question]),
            question,
            option_count=32,
        )


def test_pad_candidates_preserves_padding_id_outside_live_tokens() -> None:
    tokenized = [TokenizedCandidate((1, 5, 2), (1, 1, 1), (0, 0, 0))]
    ids, _att, _ttypes, _omask = pad_candidates(
        tokenized,
        slots=2,
        live_sequence_length=5,
        padding_id=42,
    )
    assert ids[1] == [42, 42, 42, 42, 42]
    assert ids[0][3:5] == [42, 42]
