"""Stable request, response, and KASI compatibility contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass
import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from math import isfinite
from typing import Any


class QuestionType(StrEnum):
    CHOICE = "choice"
    BOOLEAN = "boolean"
    SCORE = "score"


class Backend(StrEnum):
    DIRECT = "direct"
    DIFFUSION = "diffusion"


class ResponseStatus(StrEnum):
    OK = "ok"
    ABSTAIN = "abstain"


class KASIAction(StrEnum):
    CALL = "call"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    REFUSE = "refuse"
    RESPOND = "respond"


_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_RISK_RANKS = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _coerce_enum(enum_type: type[Any], value: Any, field_name: str) -> Any:
    raw = value.value if isinstance(value, enum_type) else value
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _validate_tool_name(name: Any) -> str:
    if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError("tool name must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
    return name


def _normalize_risk(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _RISK_RANKS else None


def _call_fingerprint(call: Mapping[str, Any]) -> str:
    name = _validate_tool_name(call.get("name"))
    arguments = call.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise TypeError("KASI call arguments must be an object")
    try:
        payload = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("KASI call arguments must be JSON-compatible") from exc
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    prompt: str
    options: tuple[str, ...]
    rubric: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("question id must be a non-empty string")
        if not isinstance(self.type, QuestionType):
            raise TypeError("question type must be a QuestionType")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("question prompt must be a non-empty string")
        if any(not isinstance(item, str) or not item.strip() for item in self.options):
            raise ValueError("question options must be non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.rubric):
            raise ValueError("question rubric must be non-empty strings")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Question:
        if not isinstance(value, Mapping):
            raise TypeError("question must be an object")
        options = value.get("options", ())
        rubric = value.get("rubric", ())
        if isinstance(options, (str, bytes, bytearray)) or not isinstance(options, Sequence):
            raise TypeError("question options must be a list")
        if isinstance(rubric, (str, bytes, bytearray)) or not isinstance(rubric, Sequence):
            raise TypeError("question rubric must be a list")
        try:
            question_id = value["id"]
            prompt = value["prompt"]
            if not isinstance(question_id, str) or not isinstance(prompt, str):
                raise TypeError("question id and prompt must be strings")
            if any(not isinstance(item, str) for item in options) or any(not isinstance(item, str) for item in rubric):
                raise TypeError("question options and rubric must contain strings")
            return cls(
                id=question_id,
                type=_coerce_enum(QuestionType, value["type"], "question type"),
                prompt=prompt,
                options=tuple(options),
                rubric=tuple(rubric),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid question: {value!r}") from exc

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "prompt": self.prompt,
            "options": list(self.options),
        }
        if self.rubric:
            result["rubric"] = list(self.rubric)
        return result


@dataclass(frozen=True)
class DecisionRequest:
    state: str | Mapping[str, Any]
    questions: tuple[Question, ...]
    backend: Backend = Backend.DIRECT
    seed: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DecisionRequest:
        if not isinstance(value, Mapping):
            raise TypeError("decision request must be an object")
        questions = value.get("questions", ())
        if isinstance(questions, (str, bytes, bytearray)) or not isinstance(questions, Sequence):
            raise TypeError("questions must be a list")
        return cls(
            state=value.get("state", ""),
            questions=tuple(Question.from_mapping(item) for item in questions),
            backend=_coerce_enum(Backend, value.get("backend", Backend.DIRECT.value), "backend"),
            seed=None if value.get("seed") is None else int(value["seed"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "questions": [question.to_mapping() for question in self.questions],
            "backend": self.backend.value,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class OptionProbability:
    option: str
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.option, str) or not self.option.strip():
            raise ValueError("probability option must be a non-empty string")
        if not isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")


@dataclass(frozen=True)
class QuestionAnswer:
    question_id: str
    choice: str | None
    probabilities: tuple[OptionProbability, ...]
    confidence: float
    status: ResponseStatus
    abstain_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResponseStatus):
            raise TypeError("status must be a ResponseStatus")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        options = [item.option for item in self.probabilities]
        if len(options) != len(set(options)):
            raise ValueError("probabilities must not contain duplicate options")
        if self.probabilities and abs(sum(item.probability for item in self.probabilities) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1")
        if self.status is ResponseStatus.ABSTAIN and not self.abstain_reason:
            raise ValueError("abstain responses require abstain_reason")
        if self.status is ResponseStatus.ABSTAIN and self.choice is not None:
            raise ValueError("abstain responses must not contain choice")
        if self.status is ResponseStatus.OK and self.choice is None:
            raise ValueError("ok responses require choice")
        if self.status is ResponseStatus.OK and self.probabilities and self.choice not in options:
            raise ValueError("ok choice must be present in probabilities")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QuestionAnswer:
        if not isinstance(value, Mapping):
            raise TypeError("answer must be an object")
        probabilities = value.get("probabilities", ())
        if isinstance(probabilities, (str, bytes, bytearray)) or not isinstance(probabilities, Sequence):
            raise TypeError("probabilities must be a list")
        return cls(
            question_id=str(value["question_id"]),
            choice=None if value.get("choice") is None else str(value["choice"]),
            probabilities=tuple(
                OptionProbability(option=str(item["option"]), probability=float(item["probability"]))
                for item in probabilities
            ),
            confidence=float(value["confidence"]),
            status=_coerce_enum(ResponseStatus, value["status"], "response status"),
            abstain_reason=None if value.get("abstain_reason") is None else str(value["abstain_reason"]),
        )


@dataclass(frozen=True)
class DecisionResponse:
    answers: tuple[QuestionAnswer, ...]
    backend: Backend
    model_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, Backend):
            raise TypeError("backend must be a Backend")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        ids = [answer.question_id for answer in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("answers must have unique question ids")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be an object")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DecisionResponse:
        if not isinstance(value, Mapping):
            raise TypeError("decision response must be an object")
        answers = value.get("answers", ())
        if isinstance(answers, (str, bytes, bytearray)) or not isinstance(answers, Sequence):
            raise TypeError("answers must be a list")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be an object")
        return cls(
            answers=tuple(QuestionAnswer.from_mapping(item) for item in answers),
            backend=_coerce_enum(Backend, value["backend"], "backend"),
            model_id=str(value["model_id"]),
            metadata=dict(metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "answers": [
                {
                    "question_id": answer.question_id,
                    "choice": answer.choice,
                    "probabilities": [
                        {"option": item.option, "probability": item.probability}
                        for item in answer.probabilities
                    ],
                    "confidence": answer.confidence,
                    "status": answer.status.value,
                    "abstain_reason": answer.abstain_reason,
                }
                for answer in self.answers
            ],
            "backend": self.backend.value,
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
        }


def _approximate_tokens(value: Any) -> int:
    if isinstance(value, Mapping):
        return 2 + sum(_approximate_tokens(key) + _approximate_tokens(item) for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return 2 + sum(_approximate_tokens(item) for item in value)
    # A conservative lower bound for an English WordPiece tokenizer. The
    # budget is a safety boundary, so under-counting is worse than rejection.
    return max(1, (len(str(value)) + 1) // 2)


def validate_request(request: DecisionRequest, *, max_questions: int = 8, max_tokens: int = 512) -> None:
    """Validate the public contract before tokenization or model execution."""
    if not isinstance(request.state, (str, Mapping)):
        raise TypeError("state must be a string or object")
    if isinstance(request.state, str) and not request.state.strip():
        raise ValueError("state must not be empty")
    if not 1 <= len(request.questions) <= max_questions:
        raise ValueError(f"questions must contain 1..{max_questions} items")

    seen_ids: set[str] = set()
    for question in request.questions:
        if not question.id or question.id in seen_ids:
            raise ValueError("question ids must be non-empty and unique")
        seen_ids.add(question.id)
        if not question.prompt.strip():
            raise ValueError(f"question {question.id!r} has an empty prompt")
        if question.type is QuestionType.BOOLEAN:
            if question.options and question.options != ("true", "false"):
                raise ValueError(f"boolean question {question.id!r} must use true/false options")
        elif not 2 <= len(question.options) <= 32:
            raise ValueError(f"question {question.id!r} must contain 2..32 options")
        elif len(set(question.options)) != len(question.options):
            raise ValueError(f"question {question.id!r} has duplicate options")
        if question.type is QuestionType.SCORE and not 2 <= len(question.rubric) <= 10:
            raise ValueError(f"score question {question.id!r} must contain a 2..10 item rubric")
        if question.type is not QuestionType.SCORE and question.rubric:
            raise ValueError(f"only score question {question.id!r} may provide rubric")

    estimated_tokens = _approximate_tokens(request.to_mapping())
    if estimated_tokens > max_tokens:
        raise ValueError(f"request exceeds the {max_tokens}-token input budget (estimate={estimated_tokens})")


def derive_kasi_action(*, proposed_calls: Sequence[Mapping[str, Any]], confidence: float,
                       risk: str | None = None, confirmed_tools: Collection[str] | None = None,
                       confirmed_call_fingerprints: Collection[str] | None = None,
                       tool_policy: Mapping[str, str] | None = None,
                       threshold: float = 0.55) -> KASIAction:
    """Derive KASI's public action in deterministic host code.

    The model can propose ordinary calls. It cannot emit or authorize confirm.
    """
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        return KASIAction.CLARIFY
    if not isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite and in (0, 1]")
    if isinstance(confirmed_tools, (str, bytes, bytearray, Mapping)):
        raise TypeError("confirmed_tools must be a collection of tool names")
    if isinstance(confirmed_call_fingerprints, (str, bytes, bytearray, Mapping)):
        raise TypeError("confirmed_call_fingerprints must be a collection")
    if tool_policy is not None and not isinstance(tool_policy, Mapping):
        raise TypeError("tool_policy must be an object")
    if confidence < threshold:
        return KASIAction.CLARIFY
    if not proposed_calls:
        return KASIAction.REFUSE

    confirmed = set(confirmed_call_fingerprints or ())
    advisory_risk = _normalize_risk(risk)
    if risk is not None and advisory_risk is None:
        return KASIAction.CONFIRM
    for call in proposed_calls:
        if not isinstance(call, Mapping):
            raise TypeError("each proposed call must be an object")
        name = _validate_tool_name(call.get("name"))
        if tool_policy is None or name not in tool_policy:
            return KASIAction.REFUSE
        host_risk = _normalize_risk(tool_policy[name])
        if host_risk is None:
            return KASIAction.CONFIRM
        requires_consent = advisory_risk is None or max(_RISK_RANKS[host_risk], _RISK_RANKS[advisory_risk]) >= _RISK_RANKS["high"]
        if requires_consent and _call_fingerprint(call) not in confirmed:
            return KASIAction.CONFIRM
    return KASIAction.CALL
