"""Auditable Ollama adapter for local data generation and cross-review."""

from __future__ import annotations

import ipaddress
import json
import math
import platform
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import Any

from .contract import (
    Backend,
    DecisionRequest,
    DecisionResponse,
    OptionProbability,
    QuestionAnswer,
    ResponseStatus,
    validate_request,
)
from .data import Example
from .onnx_runtime import DecisionBackend


@dataclass(frozen=True)
class OllamaModelInfo:
    tag: str
    digest: str | None
    details: Mapping[str, Any]
    capabilities: tuple[str, ...]
    license_present: bool


@dataclass(frozen=True)
class OllamaRecord:
    model: str
    prompt_kind: str
    prompt: str
    response: str | None
    created_at: str
    elapsed_ms: float
    metadata: Mapping[str, Any]
    error: str | None = None


class OllamaClient:
    """Ollama metadata and JSON client restricted to the local daemon."""

    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> None:
        try:
            _validate_loopback_host(host)
        except OllamaToolCallError:
            raise ValueError("OllamaClient host must be a loopback URL") from None
        parsed_host = urllib.parse.urlsplit(host)
        if parsed_host.path not in {"", "/"}:
            raise ValueError("OllamaClient host must not include a path")
        self.host, self.timeout = host.rstrip("/"), timeout
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _LoopbackRedirectHandler,
        )

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.host + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def version(self) -> str:
        value = self._request("/api/version")
        if not isinstance(value, Mapping) or not isinstance(value.get("version"), str):
            raise TypeError("malformed Ollama version response")
        return value["version"]

    def tags(self) -> list[Mapping[str, Any]]:
        value = self._request("/api/tags")
        models = value.get("models", []) if isinstance(value, Mapping) else []
        return [item for item in models if isinstance(item, Mapping)]

    def show(self, model: str) -> OllamaModelInfo:
        value = self._request("/api/show", {"model": model})
        if not isinstance(value, Mapping):
            raise TypeError("malformed Ollama model response")
        digest = value.get("digest")
        if not digest:
            digest = next(
                (
                    item.get("digest")
                    for item in self.tags()
                    if item.get("name") == model or item.get("model") == model
                ),
                None,
            )
        return OllamaModelInfo(
            model,
            digest,
            dict(value.get("details", {})),
            tuple(value.get("capabilities", ())),
            bool(value.get("license")),
        )

    def chat_json(
        self, model: str, *, system: str, prompt: str, schema: Mapping[str, Any], seed: int = 7
    ) -> OllamaRecord:
        started = time.perf_counter()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        body: Mapping[str, Any] = {}
        try:
            body = self._request(
                "/api/chat",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": schema,
                    "options": {"temperature": 0, "seed": seed},
                    "think": False,
                },
            )
            if not isinstance(body, Mapping):
                raise TypeError("malformed Ollama chat response")
            message = body.get("message")
            if not isinstance(message, Mapping) or not isinstance(message.get("content", ""), str):
                raise TypeError("malformed Ollama chat message")
            response = message.get("content", "")
            info = self.show(model)
            error = None
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError) as exc:
            response, info, error = None, None, f"{type(exc).__name__}: {exc}"
        ollama_version = None
        ollama_version_error = None
        if error is None:
            try:
                ollama_version = self.version()
            except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError) as exc:
                ollama_version_error = type(exc).__name__
        return OllamaRecord(
            model,
            "chat_json",
            prompt,
            response,
            created_at,
            (time.perf_counter() - started) * 1000,
            {
                "host": self.host,
                "platform": platform.platform(),
                "ollama_version": ollama_version,
                "ollama_version_error": ollama_version_error,
                "digest": info.digest if info else None,
                "details": dict(info.details) if info else {},
                "capabilities": list(info.capabilities) if info else [],
                "license_present": info.license_present if info else None,
                "raw_fields": sorted(body),
                "raw_response": body,
            },
            error,
        )


def record_to_mapping(record: OllamaRecord) -> dict[str, Any]:
    return asdict(record)


def benchmark_ollama(
    client: OllamaClient,
    model: str,
    examples: Sequence[Example],
    *,
    role: str = "judge",
    limit: int = 200,
    seed: int = 7,
) -> dict[str, Any]:
    """Measure JSON compliance and judgment on a bounded, known-label set.

    The model's self-reported confidence is retained only as raw output. It is
    never used as a probability or as a substitute for the dataset label.
    """
    schema = {
        "type": "object",
        "properties": {"choice": {"type": ["string", "null"]}, "abstain": {"type": "boolean"}},
        "required": ["choice", "abstain"],
        "additionalProperties": False,
    }
    system = "Return only JSON with a choice from the candidates or null and an abstain boolean. Do not call tools."
    records: list[dict[str, Any]] = []
    for row in list(examples)[:limit]:
        prompt = json.dumps(
            {"state": row.state, "question": row.question, "candidates": list(row.options)},
            ensure_ascii=False,
        )
        record = client.chat_json(model, system=system, prompt=prompt, schema=schema, seed=seed)
        item = record_to_mapping(record)
        item.update(
            {
                "example_id": row.id,
                "role": role,
                "expected_label": row.label,
                "expected_answerable": row.answerable,
            }
        )
        parsed: Mapping[str, Any] | None = None
        valid_json = False
        if record.response:
            try:
                candidate = json.loads(record.response)
                if isinstance(candidate, Mapping) and isinstance(candidate.get("abstain"), bool):
                    choice = candidate.get("choice")
                    valid_json = choice is None or isinstance(choice, str)
                    parsed = candidate if valid_json else None
            except json.JSONDecodeError:
                parsed = None
        item["parsed"] = dict(parsed) if parsed is not None else None
        item["valid_json"] = valid_json
        item["correct"] = bool(
            valid_json
            and (
                (
                    row.answerable
                    and parsed
                    and parsed.get("choice") == row.label
                    and not parsed.get("abstain")
                )
                or (not row.answerable and parsed and parsed.get("abstain"))
            )
        )
        records.append(item)
    valid = [item for item in records if item["valid_json"]]
    correct = [item for item in records if item["correct"]]
    latencies = sorted(float(item["elapsed_ms"]) for item in records)
    return {
        "model": model,
        "role": role,
        "seed": seed,
        "rows": len(records),
        "json_valid_rate": len(valid) / len(records) if records else 0.0,
        "judgment_accuracy": len(correct) / len(records) if records else 0.0,
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "records": records,
    }


def write_benchmark(path: str | Path, result: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


_LOOPBACK_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "::1", "0000:0000:0000:0000:0000:0000:0000:0001"}
)
_VONS_TOOL_NAME = "vons_decide"
_ERROR_ID_BYTES = 6
_MAX_MESSAGES = 32
_MAX_MESSAGE_CONTENT_CHARS = 16_384
_MAX_TOOL_ARGUMENT_BYTES = 65_536
_MAX_REQUEST_BYTES = 262_144
_MAX_RESPONSE_BYTES = 524_288
_MAX_MODEL_NAME_CHARS = 256
_MAX_SEED = 2**31 - 1


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


class OllamaToolCallError(RuntimeError):
    """Sanitized host error for Ollama tool-call workflows.

    The message is a fixed short literal. A 12-character hex `error_id` is
    attached for correlation, but no cause text, request payload, or response text
    is ever echoed in the string form.
    """

    def __init__(self, message: str = "ollama tool-call host aborted") -> None:
        super().__init__(message)
        self.error_id = secrets.token_hex(_ERROR_ID_BYTES)[:12]

    def sanitized(self) -> dict[str, str]:
        return {"error": "ollama_tool_call_failed", "error_id": self.error_id}


def _validate_loopback_host(url: str) -> None:
    """Reject a URL whose host portion is not a documented loopback address.

    Runs BEFORE any socket open. Rejects URI schemes, hostnames, raw IPs, or bracket
    IPv6 forms that do not resolve to ``127.0.0.1``, ``::1``, or ``localhost``.
    """
    if not isinstance(url, str) or not url:
        raise OllamaToolCallError()
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise OllamaToolCallError() from None
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise OllamaToolCallError()
    host_lower = hostname.lower()
    if host_lower in _LOOPBACK_HOSTS:
        return
    try:
        addr = ipaddress.ip_address(host_lower)
    except ValueError:
        raise OllamaToolCallError() from None
    if not addr.is_loopback:
        raise OllamaToolCallError()


class _LoopbackRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib redirect handler that refuses to leave loopback addresses.

    Applied to every outgoing tool-call requests so a compromised local Ollama proxy cannot
    bounce a tool dispatch to a remote service.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override,no-untyped-def]
        _validate_loopback_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_301(self, req, fp, code, msg, headers):  # type: ignore[override,no-untyped-def]
        return self._handle_redirect(req, fp, code, msg, headers)

    def http_error_302(self, req, fp, code, msg, headers):  # type: ignore[override,no-untyped-def]
        return self._handle_redirect(req, fp, code, msg, headers)

    def http_error_303(self, req, fp, code, msg, headers):  # type: ignore[override,no-untyped-def]
        return self._handle_redirect(req, fp, code, msg, headers)

    def http_error_307(self, req, fp, code, msg, headers):  # type: ignore[override,no-untyped-def]
        return self._handle_redirect(req, fp, code, msg, headers)

    def http_error_308(self, req, fp, code, msg, headers):  # type: ignore[override,no-untyped-def]
        return self._handle_redirect(req, fp, code, msg, headers)

    def _handle_redirect(self, req, fp, code, msg, headers):  # type: ignore[no-untyped-def]
        newurl = headers.get("Location") or headers.get("location")
        if isinstance(newurl, bytes):
            newurl = newurl.decode("latin-1")
        if newurl:
            _validate_loopback_host(str(newurl))
        method = getattr(urllib.request.HTTPRedirectHandler, f"http_error_{code}", None)
        if method is None:
            raise OllamaToolCallError()
        return method(self, req, fp, code, msg, headers)


_VONS_DECIDE_TOOL_SCHEMA: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": _VONS_TOOL_NAME,
        "description": "Run a verified local Vons ONNX decision on a structured candidate set.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "oneOf": [{"type": "string"}, {"type": "object"}],
                    "description": "Serialized conversation/task state or structured object.",
                },
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "enum": ["choice", "boolean", "score"]},
                            "prompt": {"type": "string", "minLength": 1},
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 32,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "rubric": {
                                "type": "array",
                                "maxItems": 10,
                                "items": {"type": "string", "minLength": 1},
                            },
                        },
                        "required": ["id", "type", "prompt", "options"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["state", "questions"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class OllamaToolCallResult:
    """Host-owned result of a single Ollama chat-with-tools turn.

    The decoded Ollama JSON response mapping is preserved before decision
    validation; it is not a byte-for-byte copy of the wire response. The
    validated ``backend_result`` is populated only when (1) the model requested exactly one ``vons_decide``
    call, (2) its arguments validated against ``DecisionRequest``, (3) the
    local backend ran, and (4) the backend output reconciled with the request.
    Every other case yields a sanitized error and leaves the backend empty.
    """

    model: str
    error_id: str | None
    raw_ollama_response: Mapping[str, Any] | None
    backend_result: DecisionResponse | None
    tool_invoked: bool
    error: OllamaToolCallError | None


class OllamaToolCallHost:
    """Host-owned Ollama chat adapter for the Vons native tool-call workflow.

    Security boundaries enforced here, in order:

    1. Host address must resolve to loopback.
    2. Redirects that would leave loopback are blocked.
    3. Only the ``vons_decide`` tool is advertised; any other tool call
       the model requests is rejected *before* the backend is reached.
    4. Tool-call arguments must parse to a valid ``DecisionRequest``.
    5. The local backend is the ONLY decision authority. The model's own
       returned text or confidence is never used as a choice or authorization.
    6. Backend response is reconciled against the exact request question IDs.
    7. Raw Ollama output is preserved separately; no fields from it are
       trusted to be a decision.
    8. All errors are sanitized literals with short id correlations; no request
       text, arguments, or responses are echoed to diagnostic streams.
    """

    def __init__(self, host: str, backend: DecisionBackend, *, timeout: float = 120.0) -> None:
        _validate_loopback_host(host)
        parsed_host = urllib.parse.urlsplit(host)
        if parsed_host.path not in {"", "/"}:
            raise OllamaToolCallError()
        if not isinstance(backend, DecisionBackend) or not isinstance(
            backend.backend_type, Backend
        ):
            raise OllamaToolCallError()
        self.host = host.rstrip("/")
        self.backend = backend
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise OllamaToolCallError()
        try:
            finite_timeout = math.isfinite(timeout)
        except OverflowError:
            finite_timeout = False
        if not finite_timeout or timeout <= 0:
            raise OllamaToolCallError()
        self.timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _LoopbackRedirectHandler,
        )

    def _post(self, path: str, payload: Mapping[str, Any]) -> Any:
        _validate_loopback_host(self.host + path)
        try:
            body_bytes = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (RecursionError, TypeError, ValueError) as exc:
            raise OllamaToolCallError() from exc
        if len(body_bytes) > _MAX_REQUEST_BYTES:
            raise OllamaToolCallError()
        request = urllib.request.Request(
            self.host + path,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if not isinstance(response, HTTPResponse) or response.status != 200:
                    raise OllamaToolCallError()
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except OllamaToolCallError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise OllamaToolCallError() from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise OllamaToolCallError()
        try:
            return json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (RecursionError, UnicodeDecodeError, ValueError) as exc:
            raise OllamaToolCallError() from exc

    def _parse_tool_call(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise OllamaToolCallError()
        call = tool_calls[0]
        if not isinstance(call, Mapping):
            raise OllamaToolCallError()
        if set(call) - {"id", "type", "function"}:
            raise OllamaToolCallError()
        if call.get("type", "function") != "function":
            raise OllamaToolCallError()
        call_id = call.get("id")
        if call_id is not None and (
            not isinstance(call_id, str)
            or not call_id
            or len(call_id) > _MAX_MODEL_NAME_CHARS
            or any(ord(char) < 32 for char in call_id)
        ):
            raise OllamaToolCallError()
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise OllamaToolCallError()
        if set(function) - {"index", "name", "arguments"} or not {
            "name",
            "arguments",
        } <= set(function):
            raise OllamaToolCallError()
        if "index" in function and (
            isinstance(function["index"], bool)
            or not isinstance(function["index"], int)
            or function["index"] < 0
        ):
            raise OllamaToolCallError()
        name = function.get("name")
        if name != _VONS_TOOL_NAME:
            raise OllamaToolCallError()
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            if len(arguments.encode("utf-8")) > _MAX_TOOL_ARGUMENT_BYTES:
                raise OllamaToolCallError()
            try:
                arguments = json.loads(
                    arguments,
                    object_pairs_hook=_strict_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (RecursionError, ValueError) as exc:
                raise OllamaToolCallError() from exc
        if not isinstance(arguments, Mapping) or any(not isinstance(key, str) for key in arguments):
            raise OllamaToolCallError()
        try:
            encoded_arguments = json.dumps(
                arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError) as exc:
            raise OllamaToolCallError() from exc
        if len(encoded_arguments) > _MAX_TOOL_ARGUMENT_BYTES:
            raise OllamaToolCallError()
        return arguments

    @staticmethod
    def _validate_tool_arguments(arguments: Mapping[str, Any]) -> None:
        allowed_root = {"state", "questions"}
        if set(arguments) - allowed_root or not {"state", "questions"} <= set(arguments):
            raise OllamaToolCallError()
        questions = arguments.get("questions")
        if not isinstance(questions, list):
            raise OllamaToolCallError()
        for question in questions:
            if not isinstance(question, Mapping):
                raise OllamaToolCallError()
            allowed_question = {"id", "type", "prompt", "options", "rubric"}
            if set(question) - allowed_question or not {
                "id",
                "type",
                "prompt",
                "options",
            } <= set(question):
                raise OllamaToolCallError()

    def _reconcile(self, request: DecisionRequest, response: DecisionResponse) -> None:
        if (
            not isinstance(response, DecisionResponse)
            or response.backend is not request.backend
            or not isinstance(response.model_id, str)
            or not response.model_id.strip()
            or not isinstance(response.metadata, Mapping)
        ):
            raise OllamaToolCallError()
        try:
            metadata_bytes = json.dumps(
                response.metadata,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError):
            raise OllamaToolCallError() from None
        if len(metadata_bytes) > _MAX_TOOL_ARGUMENT_BYTES:
            raise OllamaToolCallError()
        if not isinstance(response.answers, tuple) or len(request.questions) != len(
            response.answers
        ):
            raise OllamaToolCallError()
        for question, answer in zip(request.questions, response.answers, strict=True):
            if (
                not isinstance(answer, QuestionAnswer)
                or not isinstance(answer.question_id, str)
                or question.id != answer.question_id
                or not isinstance(answer.status, ResponseStatus)
            ):
                raise OllamaToolCallError()
            if (
                not isinstance(answer.confidence, (int, float))
                or isinstance(answer.confidence, bool)
                or not math.isfinite(answer.confidence)
                or not 0 <= answer.confidence <= 1
            ):
                raise OllamaToolCallError()
            options = question.options or ("true", "false")
            if answer.status is ResponseStatus.OK:
                if (
                    not isinstance(answer.choice, str)
                    or answer.choice not in options
                    or answer.abstain_reason is not None
                ):
                    raise OllamaToolCallError()
            elif (
                answer.choice is not None
                or not isinstance(answer.abstain_reason, str)
                or not answer.abstain_reason.strip()
            ):
                raise OllamaToolCallError()
            probabilities = answer.probabilities
            if not isinstance(probabilities, tuple) or any(
                not isinstance(item, OptionProbability) for item in probabilities
            ):
                raise OllamaToolCallError()
            if probabilities and tuple(item.option for item in probabilities) != options:
                raise OllamaToolCallError()
            if answer.status is ResponseStatus.OK and not probabilities:
                raise OllamaToolCallError()
            if any(
                isinstance(item.probability, bool)
                or not isinstance(item.probability, (int, float))
                or not math.isfinite(item.probability)
                or not 0 <= item.probability <= 1
                for item in probabilities
            ):
                raise OllamaToolCallError()
            if probabilities and abs(sum(item.probability for item in probabilities) - 1.0) > 1e-6:
                raise OllamaToolCallError()

    @staticmethod
    def _validate_messages(messages: Any) -> list[dict[str, str]]:
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes, bytearray, Mapping))
            or not 1 <= len(messages) <= _MAX_MESSAGES
        ):
            raise OllamaToolCallError()
        validated: list[dict[str, str]] = []
        total_chars = 0
        for message in messages:
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise OllamaToolCallError()
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
                raise OllamaToolCallError()
            if not isinstance(content, str) or not content.strip():
                raise OllamaToolCallError()
            if len(content) > _MAX_MESSAGE_CONTENT_CHARS:
                raise OllamaToolCallError()
            total_chars += len(content)
            validated.append({"role": role, "content": content})
        if total_chars > _MAX_REQUEST_BYTES:
            raise OllamaToolCallError()
        return validated

    def chat_with_tools(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        seed: int = 7,
        think: bool | None = None,
    ) -> OllamaToolCallResult:
        """Run one /api/chat turn advertising only the ``vons_decide`` tool.

        The caller supplies the user/system messages. The host appends exactly one
        tool definition (``vons_decide``). If the model replies with exactly
        one matching tool call, its arguments are validated, the backend
        runs, the response reconciles, and both halves are returned. In
        every other case a sanitized error is returned; no backend runs. The
        optional ``think`` override is sent only when explicitly supplied, so
        ``None`` preserves Ollama's per-model default.
        """
        if (
            not isinstance(model, str)
            or not model.strip()
            or len(model) > _MAX_MODEL_NAME_CHARS
            or any(ord(char) < 32 for char in model)
        ):
            err = OllamaToolCallError()
            return OllamaToolCallResult(
                model="",
                error_id=err.error_id,
                raw_ollama_response=None,
                backend_result=None,
                tool_invoked=False,
                error=err,
            )
        try:
            validated_messages = self._validate_messages(messages)
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= _MAX_SEED
                or (think is not None and not isinstance(think, bool))
            ):
                raise OllamaToolCallError()
        except OllamaToolCallError as err:
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=None,
                backend_result=None,
                tool_invoked=False,
                error=err,
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": validated_messages,
            "tools": [dict(_VONS_DECIDE_TOOL_SCHEMA)],
            "stream": False,
            "options": {"temperature": 0, "seed": seed},
        }
        if think is not None:
            payload["think"] = think
        raw: Mapping[str, Any] | None = None
        try:
            raw = self._post("/api/chat", payload)
        except OllamaToolCallError as err:
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=False,
                error=err,
            )
        if not isinstance(raw, Mapping):
            err = OllamaToolCallError()
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=False,
                error=err,
            )
        message = raw.get("message")
        if not isinstance(message, Mapping):
            err = OllamaToolCallError()
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=False,
                error=err,
            )
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            return OllamaToolCallResult(
                model=model,
                error_id=None,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=False,
                error=None,
            )
        try:
            arguments = self._parse_tool_call(message)
            self._validate_tool_arguments(arguments)
            request = DecisionRequest.from_mapping(
                {
                    "state": arguments["state"],
                    "questions": arguments["questions"],
                    "backend": self.backend.backend_type.value,
                    "seed": seed,
                }
            )
            validate_request(request)
        except OllamaToolCallError as err:
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=True,
                error=err,
            )
        except (TypeError, ValueError):
            err = OllamaToolCallError()
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=True,
                error=err,
            )
        try:
            backend_response = self.backend.decide(request)
        except Exception:  # noqa: BLE001 - backend may raise arbitrary
            err = OllamaToolCallError()
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=True,
                error=err,
            )
        try:
            self._reconcile(request, backend_response)
        except Exception:  # noqa: BLE001 - malformed backend output fails closed
            err = OllamaToolCallError()
            return OllamaToolCallResult(
                model=model,
                error_id=err.error_id,
                raw_ollama_response=raw,
                backend_result=None,
                tool_invoked=True,
                error=err,
            )
        return OllamaToolCallResult(
            model=model,
            error_id=None,
            raw_ollama_response=raw,
            backend_result=backend_response,
            tool_invoked=True,
            error=None,
        )


def tool_call_result_to_mapping(result: OllamaToolCallResult) -> dict[str, Any]:
    backend = None
    if result.backend_result is not None:
        backend = result.backend_result.to_mapping()
    return {
        "model": result.model,
        "error_id": result.error_id,
        "tool_invoked": result.tool_invoked,
        "raw_ollama_response": (
            dict(result.raw_ollama_response) if result.raw_ollama_response is not None else None
        ),
        "backend_result": backend,
        "error": result.error.sanitized() if result.error is not None else None,
    }
