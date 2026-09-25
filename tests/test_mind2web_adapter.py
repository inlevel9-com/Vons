import json
from dataclasses import dataclass

from vons.mind2web_adapter import compress_request, strip_dom_presentation


@dataclass
class Encoding:
    ids: list[int]


class WordTokenizer:
    def __init__(self) -> None:
        self.tokens: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text: str) -> Encoding:
        ids = []
        for token in text.split():
            if token not in self.tokens:
                index = len(self.tokens) + 1
                self.tokens[token] = index
                self.reverse[index] = token
            ids.append(self.tokens[token])
        return Encoding(ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self.reverse[index] for index in ids)


def test_dom_compressor_keeps_semantics_and_removes_presentation() -> None:
    raw = (
        'button class="large primary" style="color:red" id=save '
        'type=submit aria-label="Save" data-tracking=secret Save changes '
        "<svg class=icon><path id=shape /></svg>"
    )
    compressed = strip_dom_presentation(raw)
    assert compressed.startswith("button")
    assert "id=save" in compressed
    assert "type=submit" in compressed
    assert 'aria-label="Save"' in compressed
    assert "Save changes" in compressed
    assert "class=" not in compressed
    assert "style=" not in compressed
    assert "data-tracking=" not in compressed
    assert "svg" not in compressed.lower()


def test_compress_request_enforces_history_candidate_and_total_budgets() -> None:
    tokenizer = WordTokenizer()
    request = {
        "state": json.dumps(
            {
                "goal": " ".join(f"goal-{index}" for index in range(80)),
                "previous_actions": [f"action-{index}" for index in range(5)],
            }
        ),
        "question": "choose the next element",
        "candidates": [
            {
                "id": str(index),
                "text": "button class=decorative id=target "
                + " ".join(f"visible-{part}" for part in range(20)),
            }
            for index in range(8)
        ],
    }
    compressed, stats = compress_request(
        request,
        tokenizer,
        max_prompt_tokens=40,
        max_candidate_tokens=3,
    )
    state = json.loads(compressed["state"])
    assert len(state["previous_actions"]) <= 2
    assert stats.history_actions_before == 5
    assert stats.history_actions_after <= 2
    assert stats.prompt_tokens <= 40
    assert stats.max_candidate_tokens <= 3
    assert all(
        len(tokenizer.encode(candidate["text"]).ids) <= 3
        for candidate in compressed["candidates"]
    )


def test_compress_request_preserves_last_two_actions_when_they_fit() -> None:
    tokenizer = WordTokenizer()
    request = {
        "state": {
            "goal": "save the form",
            "previous_actions": ["first", "second", "third", "fourth"],
        },
        "question": "choose",
        "candidates": [{"id": "save", "text": "button id=save Save"}],
    }
    compressed, stats = compress_request(request, tokenizer)
    state = json.loads(compressed["state"])
    assert state["previous_actions"] == ["third", "fourth"]
    assert stats.history_actions_after == 2
    assert stats.prompt_tokens <= 512
