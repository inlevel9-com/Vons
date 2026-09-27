"""ONNX-backed decision runtime with hash-verified bundles.

Host-owned runtime layer. The caller supplies a bundle directory with a valid
``vons.bundle.manifest/v1`` manifest; this module verifies every recorded byte
before executing the graph and strictly validates request/response against the
public ``DecisionRequest`` / ``DecisionResponse`` contract.

Synthetic backends (``SyntheticAbstainBackend``, ``SyntheticChoiceBackend``)
exist for protocol testing and must never be used as substitutes for a verified
ONNX decision run.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import verify_bundle_manifest
from .contract import (
    Backend,
    DecisionRequest,
    DecisionResponse,
    OptionProbability,
    QuestionAnswer,
    ResponseStatus,
    validate_request,
)


class OnnxRuntimeError(RuntimeError):
    """Raised when the ONNX runtime or bundle verification fails.

    The message is intentionally generic. Callers must not echo exception data
    or request payloads into shared logs.
    """


class OnnxRuntimeUnavailableError(OnnxRuntimeError):
    """Raised when an optional local inference dependency is unavailable."""


@dataclass(frozen=True)
class VerifiedBundle:
    """A manifest-verified bundle directory, ready for inference."""

    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    model_id: str

    @classmethod
    def load(cls, bundle_dir: str | Path) -> VerifiedBundle:
        try:
            root = Path(bundle_dir).resolve(strict=True)
        except (OSError, FileNotFoundError):
            raise OnnxRuntimeError("bundle path does not exist") from None
        if not root.is_dir():
            raise OnnxRuntimeError("bundle path is not a directory")
        manifest_path = None
        for candidate in (
            root / "bundle-manifest.json",
            root / "bundle-manifest-v1.json",
        ):
            if not candidate.is_file():
                continue
            try:
                candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(candidate_payload, Mapping)
                and candidate_payload.get("schema_version") == "vons.bundle.manifest/v1"
            ):
                manifest_path = candidate
                break
        if manifest_path is None:
            raise OnnxRuntimeError("bundle manifest is missing")
        try:
            verification = verify_bundle_manifest(manifest_path)
        except Exception:  # noqa: BLE001 - normalize to sanitized form
            raise OnnxRuntimeError("bundle manifest verification failed") from None
        _ = verification  # passed; record consumed, payload never echoed
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise TypeError("bundle manifest must be an object")
            model_id = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except (OSError, TypeError, ValueError):
            raise OnnxRuntimeError("bundle manifest could not be read") from None
        return cls(root=root, manifest_path=manifest_path, manifest=manifest, model_id=model_id)


class DecisionBackend(ABC):
    """Strictly-typed decision backend.

    Implementations validate request input against the public contract before
    execution and validate every response field before returning to the host.
    No backend may emit a fallback or coerced choice when validation fails.
    """

    backend_type: Backend

    @abstractmethod
    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Execute a validated request and return a validated response."""

    def _validate_request(self, request: DecisionRequest) -> None:
        if not isinstance(request, DecisionRequest):
            raise OnnxRuntimeError("invalid decision request type")
        try:
            validate_request(request)
        except (TypeError, ValueError) as exc:
            raise OnnxRuntimeError("decision request failed validation") from exc


class SyntheticAbstainBackend(DecisionBackend):
    """Test-only backend that always abstains with a deterministic reason.

    The reason byte is preserved exactly so protocol round-trips can be
    asserted without running model weights.
    """

    backend_type = Backend.DIRECT

    def __init__(self, abstain_reason: str = "synthetic backend requested abstention") -> None:
        self.abstain_reason = abstain_reason

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self._validate_request(request)
        answers = tuple(
            QuestionAnswer(
                question_id=question.id,
                choice=None,
                probabilities=(),
                confidence=0.0,
                status=ResponseStatus.ABSTAIN,
                abstain_reason=self.abstain_reason,
            )
            for question in request.questions
        )
        return DecisionResponse(
            answers=answers,
            backend=self.backend_type,
            model_id="synthetic-abstain-v1",
            metadata={"synthetic": True},
        )


class SyntheticChoiceBackend(DecisionBackend):
    """Test-only backend that picks a fixed option by position.

    Used to validate exact question-id reconciliation and response shape
    without loading model weights. Callers set ``choice_index`` to a valid
    position for every question in the request; ``ValueError`` is raised
    when the index is out of range.
    """

    backend_type = Backend.DIRECT

    def __init__(self, choice_index: int = 0, confidence: float = 0.9) -> None:
        if not isinstance(choice_index, int) or choice_index < 0:
            raise ValueError("choice_index must be a non-negative integer")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        self.choice_index = choice_index
        self.confidence = float(confidence)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self._validate_request(request)
        answers: list[QuestionAnswer] = []
        for question in request.questions:
            options = question.options or ("true", "false")
            if self.choice_index >= len(options):
                raise OnnxRuntimeError("synthetic backend choice index out of range")
            choice = options[self.choice_index]
            probs: list[OptionProbability] = []
            remaining = 1.0 - self.confidence
            for i, option in enumerate(options):
                if i == self.choice_index:
                    p = self.confidence
                else:
                    p = remaining / max(1, len(options) - 1)
                probs.append(OptionProbability(option=option, probability=float(p)))
            answers.append(
                QuestionAnswer(
                    question_id=question.id,
                    choice=choice,
                    probabilities=tuple(probs),
                    confidence=self.confidence,
                    status=ResponseStatus.OK,
                )
            )
        return DecisionResponse(
            answers=tuple(answers),
            backend=self.backend_type,
            model_id="synthetic-choice-v1",
            metadata={"synthetic": True, "choice_index": self.choice_index},
        )


class OnnxRuntimeBackend(DecisionBackend):
    """Verified ONNX Runtime backend.

    Requires the optional local-inference dependencies. The bundle is verified
    byte-for-byte before the session is created. Tokenizer files are resolved
    from the verified bundle only; remote tokenizer discovery is never used.
    """

    def __init__(
        self,
        bundle: VerifiedBundle,
        *,
        tokenizer: Any | None = None,
        session: Any | None = None,
    ) -> None:
        if not isinstance(bundle, VerifiedBundle):
            raise OnnxRuntimeError("ONNX backend requires a verified bundle")
        self.bundle = bundle
        metadata = bundle.manifest.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise OnnxRuntimeError("bundle runtime metadata is invalid")
        try:
            self.backend_type = Backend(metadata.get("backend", Backend.DIRECT.value))
        except (TypeError, ValueError):
            raise OnnxRuntimeError("bundle backend is unsupported") from None
        if self.backend_type not in {Backend.DIRECT, Backend.DIFFUSION}:
            raise OnnxRuntimeError("bundle backend is unsupported")
        option_count = metadata.get("option_count", 32)
        sequence_length = metadata.get("sequence_length", 512)
        if (
            isinstance(option_count, bool)
            or not isinstance(option_count, int)
            or not 2 <= option_count <= 32
            or isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or not 1 <= sequence_length <= 512
        ):
            raise OnnxRuntimeError("bundle inference dimensions are invalid")
        self.option_count = option_count
        self.sequence_length = sequence_length
        self._metadata = metadata
        self._tokenizer = tokenizer
        self._session = session
        self._input_names = self._string_tuple(metadata.get("input_names"), "input_names")
        self._output_names = self._string_tuple(metadata.get("output_names"), "output_names")
        self._abstain_threshold = self._threshold(metadata.get("abstain_threshold", 0.55))
        self._answerability_threshold = self._threshold(
            metadata.get("answerability_threshold", 0.5)
        )
        conditioning = metadata.get("diffusion_conditioning", "pooled")
        if self.backend_type is Backend.DIFFUSION and conditioning not in {
            "pooled",
            "token_cross_attention",
        }:
            raise OnnxRuntimeError("bundle diffusion conditioning is unsupported")
        self._diffusion_conditioning = conditioning

    @staticmethod
    def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise OnnxRuntimeError(f"bundle {name} is invalid")
        if len(value) != len(set(value)):
            raise OnnxRuntimeError(f"bundle {name} contains duplicates")
        return tuple(value)

    @staticmethod
    def _threshold(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OnnxRuntimeError("bundle decision threshold is invalid")
        result = float(value)
        if not 0.0 <= result <= 1.0:
            raise OnnxRuntimeError("bundle decision threshold is invalid")
        return result

    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise OnnxRuntimeUnavailableError("numpy is not installed") from exc
        return np

    def _tokenizer_path(self, *, suffix: str) -> Path:
        files = self.bundle.manifest.get("files", ())
        if not isinstance(files, (list, tuple)):
            raise OnnxRuntimeError("bundle tokenizer inventory is invalid")
        candidates = [
            item
            for item in files
            if isinstance(item, Mapping)
            and item.get("role") == "tokenizer"
            and isinstance(item.get("path"), str)
            and str(item["path"]).endswith(suffix)
        ]
        if len(candidates) != 1:
            raise OnnxRuntimeError("bundle tokenizer file is missing or ambiguous")
        try:
            path = (self.bundle.root / str(candidates[0]["path"])).resolve(strict=True)
            path.relative_to(self.bundle.root.resolve(strict=True))
        except (OSError, ValueError):
            raise OnnxRuntimeError("bundle tokenizer path escapes the bundle") from None
        if not path.is_file():
            raise OnnxRuntimeError("bundle tokenizer file is missing")
        return path

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise OnnxRuntimeUnavailableError("tokenizers is not installed") from exc
        tokenizer_path = self._tokenizer_path(suffix="tokenizer.json")
        config_path = self._tokenizer_path(suffix="tokenizer_config.json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
            pad_token = self._special_token_content(config.get("pad_token"))
            cls_token = self._special_token_content(config.get("cls_token"))
            pad_token_id = raw_tokenizer.token_to_id(pad_token) if pad_token else None
            cls_token_id = raw_tokenizer.token_to_id(cls_token) if cls_token else None
        except Exception:  # noqa: BLE001 - sanitize third-party errors
            raise OnnxRuntimeError("bundle tokenizer could not be loaded") from None
        if pad_token_id is None or cls_token_id is None:
            raise OnnxRuntimeError("bundle tokenizer is missing required special tokens")
        self._tokenizer = _LocalTokenizer(raw_tokenizer, pad_token_id, cls_token_id)
        return self._tokenizer

    @staticmethod
    def _special_token_content(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and isinstance(value.get("content"), str):
            return value["content"]
        return None

    def _validate_session_contract(self, session: Any) -> None:
        try:
            input_names = {item.name for item in session.get_inputs()}
            output_names = {item.name for item in session.get_outputs()}
        except Exception:  # noqa: BLE001 - sanitize third-party errors
            raise OnnxRuntimeError("ONNX session metadata is unavailable") from None
        if not self._input_names or input_names != set(self._input_names):
            raise OnnxRuntimeError("ONNX graph inputs disagree with the verified manifest")
        if not self._output_names or output_names != set(self._output_names):
            raise OnnxRuntimeError("ONNX graph outputs disagree with the verified manifest")
        expected_inputs = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "option_mask",
        }
        if self.backend_type is Backend.DIFFUSION:
            expected_inputs.add("initial_noise")
        if input_names != expected_inputs:
            raise OnnxRuntimeError("ONNX graph input contract is unsupported")
        score_name = "logits" if self.backend_type is Backend.DIRECT else "scores"
        if output_names != {score_name, "answerability"}:
            raise OnnxRuntimeError("ONNX graph output contract is unsupported")

    def prepare(self) -> None:
        """Load all local runtime dependencies before a client request is sent."""
        self._numpy()
        self._require_tokenizer()
        self._validate_session_contract(self._require_session())

    def _require_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise OnnxRuntimeUnavailableError("onnxruntime is not installed") from exc
        graph_item = None
        for item in self.bundle.manifest.get("files", ()):
            if isinstance(item, Mapping) and item.get("role") == "model_graph":
                graph_item = item
                break
        if graph_item is None:
            raise OnnxRuntimeError("bundle has no model_graph entry")
        try:
            graph_path = (self.bundle.root / str(graph_item["path"])).resolve(strict=True)
            graph_path.relative_to(self.bundle.root.resolve(strict=True))
        except (OSError, ValueError):
            raise OnnxRuntimeError("bundle graph path is invalid") from None
        if not graph_path.is_file():
            raise OnnxRuntimeError("bundle graph file is missing")
        try:
            self._session = ort.InferenceSession(
                str(graph_path), providers=["CPUExecutionProvider"]
            )
            self._validate_session_contract(self._session)
        except OnnxRuntimeError:
            raise
        except Exception:  # noqa: BLE001 - sanitize third-party errors
            raise OnnxRuntimeError("ONNX session could not be created") from None
        return self._session

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self._validate_request(request)
        if request.backend is not self.backend_type:
            raise OnnxRuntimeError("request backend does not match the verified bundle")
        try:
            from .inference import prepare_question_tensors

            np = self._numpy()
            tokenizer = self._require_tokenizer()
            session = self._require_session()
            self._validate_session_contract(session)
            score_name = "logits" if self.backend_type is Backend.DIRECT else "scores"
            answers: list[QuestionAnswer] = []
            for question_index, question in enumerate(request.questions):
                padded, _ = prepare_question_tensors(
                    tokenizer,
                    request,
                    question,
                    option_count=self.option_count,
                    max_sequence_budget=self.sequence_length,
                )
                feed: dict[str, Any] = {
                    "input_ids": np.asarray(padded.input_ids, dtype=np.int64)[None, :, :],
                    "attention_mask": np.asarray(padded.attention_mask, dtype=np.int64)[None, :, :],
                    "token_type_ids": np.asarray(padded.token_type_ids, dtype=np.int64)[None, :, :],
                    "option_mask": np.asarray(padded.option_mask, dtype=np.bool_)[None, :],
                }
                if self.backend_type is Backend.DIFFUSION:
                    seed = None if request.seed is None else request.seed + question_index
                    rng = np.random.default_rng(seed)
                    feed["initial_noise"] = rng.standard_normal(
                        (1, padded.allocated_candidates), dtype=np.float32
                    )
                outputs = session.run(list(self._output_names), feed)
                if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
                    self._output_names
                ):
                    raise OnnxRuntimeError("ONNX graph returned an invalid output list")
                output_map = dict(zip(self._output_names, outputs, strict=True))
                scores = np.asarray(output_map[score_name], dtype=np.float64)
                raw_answerability = np.asarray(output_map["answerability"], dtype=np.float64)
                live_options = question.options or ("true", "false")
                if scores.shape != (1, padded.allocated_candidates):
                    raise OnnxRuntimeError("ONNX graph returned invalid score dimensions")
                if raw_answerability.size != 1:
                    raise OnnxRuntimeError("ONNX graph returned invalid answerability")
                live_scores = scores[0, : len(live_options)]
                if not np.isfinite(live_scores).all() or not np.isfinite(raw_answerability).all():
                    raise OnnxRuntimeError("ONNX graph returned non-finite scores")
                shifted = live_scores - np.max(live_scores)
                exp_scores = np.exp(shifted)
                probabilities = exp_scores / exp_scores.sum()
                answerability_logit = float(raw_answerability.reshape(-1)[0])
                if answerability_logit >= 0:
                    answerability = 1.0 / (1.0 + np.exp(-answerability_logit))
                else:
                    exp_logit = np.exp(answerability_logit)
                    answerability = exp_logit / (1.0 + exp_logit)
                choice_index = int(np.argmax(probabilities))
                max_probability = float(probabilities[choice_index])
                confidence = max_probability * float(answerability)
                abstain = (
                    confidence < self._abstain_threshold
                    or answerability < self._answerability_threshold
                )
                answers.append(
                    QuestionAnswer(
                        question_id=question.id,
                        choice=None if abstain else live_options[choice_index],
                        probabilities=tuple(
                            OptionProbability(
                                option=option, probability=float(probabilities[index])
                            )
                            for index, option in enumerate(live_options)
                        ),
                        confidence=confidence,
                        status=ResponseStatus.ABSTAIN if abstain else ResponseStatus.OK,
                        abstain_reason=(
                            "below configured local decision threshold" if abstain else None
                        ),
                    )
                )
            return DecisionResponse(
                answers=tuple(answers),
                backend=self.backend_type,
                model_id=self.bundle.model_id,
                metadata={"verified_bundle_sha256": self.bundle.model_id},
            )
        except OnnxRuntimeError:
            raise
        except Exception:  # noqa: BLE001 - sanitize tokenizer/ORT failures
            raise OnnxRuntimeError("local ONNX inference failed") from None


@dataclass(frozen=True)
class _LocalEncoding:
    ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]


class _LocalTokenizer:
    """Small adapter from the bundled Rust tokenizer to Vons tensor preparation."""

    def __init__(self, tokenizer: Any, pad_token_id: int, cls_token_id: int) -> None:
        self._tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int = 512,
        **_: Any,
    ) -> _LocalEncoding:
        if not isinstance(text, str):
            raise TypeError("tokenizer input must be text")
        if truncation:
            self._tokenizer.enable_truncation(max_length=max_length)
        try:
            encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        finally:
            if truncation:
                self._tokenizer.no_truncation()
        return _LocalEncoding(
            ids=list(encoded.ids),
            attention_mask=list(encoded.attention_mask),
            token_type_ids=list(encoded.type_ids),
        )
