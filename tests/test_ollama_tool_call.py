"""Synthetic tests for the Ollama native tool-call vertical slice.

All tests use injected synthetic backends and canned HTTP responses via
``unittest.mock``. No real network calls, no real model weights, no restricted
Mind2Web rows, no real Ollama daemon is contacted.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from collections.abc import Mapping
from http.client import HTTPResponse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.request import ProxyHandler, Request

from vons.contract import (
    Backend,
    DecisionRequest,
    DecisionResponse,
    OptionProbability,
    QuestionAnswer,
    ResponseStatus,
)
from vons.ollama import (
    OllamaClient,
    OllamaToolCallError,
    OllamaToolCallHost,
    _LoopbackRedirectHandler,
    _validate_loopback_host,
    tool_call_result_to_mapping,
)
from vons.onnx_runtime import (
    DecisionBackend,
    OnnxRuntimeBackend,
    OnnxRuntimeError,
    OnnxRuntimeUnavailableError,
    SyntheticAbstainBackend,
    SyntheticChoiceBackend,
    VerifiedBundle,
    _LocalTokenizer,
)


def _canned_chat_payload(message_body: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"message": message_body, "model": "synthetic", "done": True}


def _vons_decide_tool_call(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "vons_decide",
                    "arguments": json.dumps(arguments)
                    if isinstance(arguments, dict)
                    else arguments,
                },
            }
        ],
    }


def _patch_post(host: OllamaToolCallHost, message_body: Mapping[str, Any]) -> None:
    host._post = MagicMock(return_value=_canned_chat_payload(message_body))  # type: ignore[method-assign]


_VALID_REQUEST = {
    "state": "A weather tool is available. Should we call it?",
    "questions": [
        {
            "id": "q-next",
            "type": "choice",
            "prompt": "Next action?",
            "options": ["call", "clarify", "confirm", "refuse"],
        }
    ],
}


class _MemorySocket:
    def __init__(self, wire_response: bytes) -> None:
        self._buffer = io.BytesIO(wire_response)

    def makefile(self, mode: str) -> io.BytesIO:
        return self._buffer

    def close(self) -> None:
        self._buffer.close()


class _CountingHTTPResponse(HTTPResponse):
    def __init__(self, wire_response: bytes) -> None:
        super().__init__(_MemorySocket(wire_response), method="POST")
        self.read_called = False
        self.begin()

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        self.read_called = True
        return super().read(*args, **kwargs)


def _http_response(status: int, body: bytes) -> _CountingHTTPResponse:
    reason = {200: "OK", 201: "Created", 302: "Found"}.get(status, "Error")
    wire = (
        f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    )
    return _CountingHTTPResponse(wire)


class _SpyBackend(DecisionBackend):
    """Test backend that records whether it was called and with what."""

    backend_type = Backend.DIRECT

    def __init__(
        self, response: DecisionResponse | None = None, *, raise_on_call: Exception | None = None
    ) -> None:
        self.calls: list[DecisionRequest] = []
        self._response = response
        self._raise = raise_on_call

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self._validate_request(request)
        self.calls.append(request)
        if self._raise is not None:
            raise self._raise
        if self._response is not None:
            return self._response
        return SyntheticChoiceBackend(choice_index=0).decide(request)


class LoopbackValidationTests(unittest.TestCase):
    def test_accepts_localhost_variants(self) -> None:
        for url in (
            "http://127.0.0.1:11434",
            "http://localhost:11434/api",
            "http://[::1]:11434/chat",
        ):
            _validate_loopback_host(url)  # must not raise

    def test_rejects_non_loopback(self) -> None:
        for url in (
            "http://example.com:11434",
            "https://192.168.1.1/api/chat",
            "http://0.0.0.0:11434",
            "http://[2001:db8::1]:11434",
            "ftp://127.0.0.1:11434",
        ):
            with self.assertRaises(OllamaToolCallError):
                _validate_loopback_host(url)

    def test_empty_or_invalid_urls(self) -> None:
        for url in ("", "not-a-url", "127.0.0.1"):
            with self.assertRaises(OllamaToolCallError):
                _validate_loopback_host(url)  # type: ignore[arg-type]


class OllamaClientBoundaryTests(unittest.TestCase):
    def test_client_rejects_remote_host_and_non_root_path(self) -> None:
        for host in ("http://192.168.1.20:11434", "http://ollama.internal:11434"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                OllamaClient(host)
        with self.assertRaisesRegex(ValueError, "must not include a path"):
            OllamaClient("http://localhost:11434/api")

    def test_client_disables_ambient_proxy_and_installs_redirect_guard(self) -> None:
        with patch("urllib.request.getproxies", return_value={"http": "http://proxy.invalid"}):
            client = OllamaClient()
            host = OllamaToolCallHost("http://127.0.0.1:11434", SyntheticAbstainBackend())

        self.assertFalse(any(isinstance(item, ProxyHandler) for item in client._opener.handlers))
        self.assertFalse(any(isinstance(item, ProxyHandler) for item in host._opener.handlers))
        self.assertTrue(
            any(isinstance(item, _LoopbackRedirectHandler) for item in client._opener.handlers)
        )
        self.assertTrue(
            any(isinstance(item, _LoopbackRedirectHandler) for item in host._opener.handlers)
        )

    def test_chat_result_survives_optional_version_metadata_failure(self) -> None:
        client = OllamaClient()

        def request(path: str, _payload: Mapping[str, Any] | None = None) -> Any:
            if path == "/api/chat":
                return {"message": {"content": '{"choice":"call"}'}}
            if path == "/api/show":
                return {"digest": "a" * 64, "details": {}, "capabilities": []}
            if path == "/api/version":
                return {"version": 12}
            raise AssertionError(f"unexpected endpoint {path}")

        client._request = request  # type: ignore[method-assign]
        record = client.chat_json("synthetic", system="system", prompt="prompt", schema={})

        self.assertIsNone(record.error)
        self.assertEqual(record.response, '{"choice":"call"}')
        self.assertIsNone(record.metadata["ollama_version"])
        self.assertEqual(record.metadata["ollama_version_error"], "TypeError")


class HostInitTests(unittest.TestCase):
    def test_requires_loopback_host(self) -> None:
        backend = SyntheticAbstainBackend()
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://example.com:11434", backend)

    def test_requires_decision_backend_instance(self) -> None:
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", "not a backend")  # type: ignore[arg-type]

    def test_valid_init(self) -> None:
        backend = SyntheticAbstainBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434/", backend)
        self.assertEqual(host.host, "http://127.0.0.1:11434")
        self.assertIs(host.backend, backend)

    def test_finite_positive_timeout_accepted(self) -> None:
        backend = SyntheticAbstainBackend()
        for value in (1.0, 30, 120.5, 2**31 - 1):
            host = OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=value)
            self.assertEqual(host.timeout, value)

    def test_boolean_timeout_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        for value in (True, False):
            with self.assertRaises(OllamaToolCallError):
                OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=value)  # type: ignore[arg-type]

    def test_non_numeric_timeout_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        for value in ("120", None, [], {}, b"120"):
            with self.assertRaises(OllamaToolCallError):
                OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=value)  # type: ignore[arg-type]

    def test_non_positive_timeout_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        for value in (0, -1, -0.5, -1e100):
            with self.assertRaises(OllamaToolCallError):
                OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=value)

    def test_integer_timeout_too_large_for_finite_float_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=10**1000)

    def test_nan_timeout_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=float("nan"))
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=math.nan)

    def test_positive_infinity_timeout_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=float("inf"))
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=math.inf)

    def test_negative_infinity_timeout_rejected(self) -> None:
        backend = SyntheticAbstainBackend()
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=float("-inf"))
        with self.assertRaises(OllamaToolCallError):
            OllamaToolCallHost("http://127.0.0.1:11434", backend, timeout=-math.inf)


class ToolCallRoundTripTests(unittest.TestCase):
    def test_default_thinking_control_is_omitted_from_request(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "pick one"}])

        self.assertIsNone(result.error)
        request_payload = host._post.call_args.args[1]
        self.assertNotIn("think", request_payload)

    def test_explicit_thinking_control_is_forwarded(self) -> None:
        for think in (False, True):
            with self.subTest(think=think):
                backend = _SpyBackend()
                host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
                _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

                result = host.chat_with_tools(
                    "synthetic-model", [{"role": "user", "content": "pick one"}], think=think
                )

                self.assertIsNone(result.error)
                request_payload = host._post.call_args.args[1]
                self.assertIs(request_payload["think"], think)

    def test_non_boolean_thinking_control_is_rejected_before_post(self) -> None:
        for think in (0, 1, "false", [], {}):
            with self.subTest(think=think):
                backend = _SpyBackend()
                host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
                host._post = MagicMock()  # type: ignore[method-assign]

                result = host.chat_with_tools(
                    "synthetic-model",
                    [{"role": "user", "content": "pick one"}],
                    think=think,  # type: ignore[arg-type]
                )

                self.assertIsNotNone(result.error)
                host._post.assert_not_called()
                self.assertEqual(backend.calls, [])

    def test_successful_tool_call_invokes_backend(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))
        messages = [{"role": "user", "content": "pick one"}]

        result = host.chat_with_tools("synthetic-model", messages)

        self.assertIsNone(result.error)
        self.assertIsNone(result.error_id)
        self.assertTrue(result.tool_invoked)
        self.assertIsNotNone(result.backend_result)
        self.assertEqual(len(backend.calls), 1)
        request = backend.calls[0]
        self.assertEqual(len(request.questions), 1)
        self.assertEqual(request.questions[0].id, "q-next")
        self.assertEqual(request.backend, Backend.DIRECT)
        self.assertEqual(request.seed, 7)
        # Reconciliation: backend answer IDs match request IDs exactly, positionally
        self.assertEqual(result.backend_result.answers[0].question_id, "q-next")

    def test_backend_and_seed_are_host_owned(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools(
            "synthetic-model", [{"role": "user", "content": "pick one"}], seed=42
        )

        self.assertIsNone(result.error)
        request_payload = host._post.call_args.args[1]
        properties = request_payload["tools"][0]["function"]["parameters"]["properties"]
        self.assertNotIn("backend", properties)
        self.assertNotIn("seed", properties)
        self.assertEqual(request_payload["options"]["seed"], 42)
        self.assertEqual(backend.calls[0].backend, Backend.DIRECT)
        self.assertEqual(backend.calls[0].seed, 42)

    def test_model_cannot_supply_backend_or_seed(self) -> None:
        for injected in (
            {**_VALID_REQUEST, "backend": "diffusion"},
            {**_VALID_REQUEST, "seed": 999},
        ):
            with self.subTest(injected_key=set(injected) - set(_VALID_REQUEST)):
                backend = _SpyBackend()
                host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
                _patch_post(host, _vons_decide_tool_call(injected))

                result = host.chat_with_tools(
                    "synthetic-model", [{"role": "user", "content": "pick one"}]
                )

                self.assertIsNotNone(result.error)
                self.assertEqual(result.error_id, result.error.error_id)
                self.assertEqual(backend.calls, [])

    def test_accepts_observed_ollama_tool_call_shape(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        observed_message = {
            "role": "assistant",
            "content": "",
            "thinking": "synthetic reasoning metadata",
            "tool_calls": [
                {
                    "id": "call_synthetic",
                    "function": {
                        "index": 0,
                        "name": "vons_decide",
                        "arguments": _VALID_REQUEST,
                    },
                }
            ],
        }
        _patch_post(host, observed_message)

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNone(result.error)
        self.assertTrue(result.tool_invoked)
        self.assertEqual(len(backend.calls), 1)

    def test_invalid_ollama_function_index_blocks_backend_call(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        observed_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_synthetic",
                    "function": {
                        "index": -1,
                        "name": "vons_decide",
                        "arguments": _VALID_REQUEST,
                    },
                }
            ],
        }
        _patch_post(host, observed_message)

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNotNone(result.error)
        self.assertEqual(len(backend.calls), 0)

    def test_raw_and_validated_separate(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        # Raw Ollama response is preserved as-is (contains the exact tool_call shape)
        self.assertIsNotNone(result.raw_ollama_response)
        raw_message = (
            result.raw_ollama_response.get("message", {})
            if isinstance(result.raw_ollama_response, dict)
            else {}
        )
        self.assertEqual(len(raw_message.get("tool_calls", [])), 1)
        # Validated backend result is a separate, typed DecisionResponse
        self.assertIsNotNone(result.backend_result)
        self.assertIsInstance(result.backend_result, DecisionResponse)
        self.assertIsNot(result.raw_ollama_response, result.backend_result)

    def test_model_text_reply_without_tool_call_is_not_an_error(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, {"role": "assistant", "content": "I think you should call."})

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertFalse(result.tool_invoked)
        self.assertIsNone(result.backend_result)
        self.assertIsNone(result.error)
        self.assertEqual(len(backend.calls), 0)

    def test_wrong_tool_name_blocks_backend_call(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        wrong_tool = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "delete_everything",
                        "arguments": json.dumps({"scope": "all"}),
                    },
                }
            ],
        }
        _patch_post(host, wrong_tool)

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertTrue(result.tool_invoked)  # model tried to invoke a tool
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)
        self.assertEqual(len(backend.calls), 0)  # backend never called

    def test_malformed_arguments_blocks_backend_call(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        missing_required = {"state": "", "questions": []}  # empty + no questions
        _patch_post(host, _vons_decide_tool_call(missing_required))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertTrue(result.tool_invoked)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)
        self.assertEqual(len(backend.calls), 0)

    def test_unknown_root_argument_blocks_backend_call(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        arguments = {**_VALID_REQUEST, "_authorization": "approve"}
        _patch_post(host, _vons_decide_tool_call(arguments))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNotNone(result.error)
        self.assertEqual(result.error_id, result.error.error_id)
        self.assertEqual(len(backend.calls), 0)

    def test_unknown_question_argument_blocks_backend_call(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        question = {**_VALID_REQUEST["questions"][0], "execute": True}
        arguments = {**_VALID_REQUEST, "questions": [question]}
        _patch_post(host, _vons_decide_tool_call(arguments))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNotNone(result.error)
        self.assertEqual(len(backend.calls), 0)

    def test_oversized_tool_arguments_block_backend_call(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        arguments = {**_VALID_REQUEST, "state": "x" * 70_000}
        _patch_post(host, _vons_decide_tool_call(arguments))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNotNone(result.error)
        self.assertEqual(len(backend.calls), 0)

    def test_invalid_message_shape_is_rejected_before_post(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        host._post = MagicMock(side_effect=AssertionError("network must not be reached"))  # type: ignore[method-assign]

        result = host.chat_with_tools("synthetic-model", [{"role": ["user"], "content": "x"}])

        self.assertIsNotNone(result.error)
        host._post.assert_not_called()

    def test_duplicate_question_ids_block_backend(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        dup_request = {
            "state": "state",
            "questions": [
                {"id": "same", "type": "choice", "prompt": "first", "options": ["a", "b"]},
                {"id": "same", "type": "choice", "prompt": "second", "options": ["c", "d"]},
            ],
        }
        _patch_post(host, _vons_decide_tool_call(dup_request))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertTrue(result.tool_invoked)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)
        self.assertEqual(len(backend.calls), 0)

    def test_backend_returning_mismatched_ids_is_rejected(self) -> None:
        """The backend must return answers whose IDs match the request order exactly.

        This guards against a compromised or buggy backend reordering or dropping
        answers so a client cannot silently misattribute choices to questions.
        """
        # Build a response with the WRONG question_id
        bad_answer = QuestionAnswer(
            question_id="q-SURPRISE",
            choice="call",
            probabilities=(OptionProbability(option="call", probability=1.0),),
            confidence=1.0,
            status=ResponseStatus.OK,
        )
        bad_response = DecisionResponse(
            answers=(bad_answer,),
            backend=Backend.DIRECT,
            model_id="buggy-backend",
        )
        backend = _SpyBackend(response=bad_response)
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertTrue(result.tool_invoked)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)
        self.assertEqual(len(backend.calls), 1)  # backend DID run, but its output rejected

    def test_backend_raising_is_sanitized(self) -> None:
        backend = _SpyBackend(raise_on_call=RuntimeError("weights corrupted, DB password leaked"))
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertTrue(result.tool_invoked)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)
        # Sanitized error must NOT contain the leaked exception text
        error_str = str(result.error) + " " + json.dumps(result.error.sanitized())
        self.assertNotIn("DB password", error_str)
        self.assertNotIn("weights corrupted", error_str)

    def test_empty_messages_list_is_error(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        host._post = MagicMock(side_effect=AssertionError("network must not be reached"))  # type: ignore[method-assign]

        result = host.chat_with_tools("synthetic-model", [])

        self.assertIsNotNone(result.error)
        self.assertFalse(result.tool_invoked)
        host._post.assert_not_called()

    def test_invalid_model_arg(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        result = host.chat_with_tools("", [{"role": "user", "content": "x"}])
        self.assertIsNotNone(result.error)
        self.assertEqual(result.model, "")

    def test_result_mapping_preserves_raw_separate_from_validated(self) -> None:
        backend = SyntheticChoiceBackend(choice_index=0)
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])
        mapped = tool_call_result_to_mapping(result)

        self.assertEqual(mapped["model"], "synthetic-model")
        self.assertTrue(mapped["tool_invoked"])
        self.assertIsNone(mapped["error"])
        backend_result = mapped["backend_result"]
        self.assertIsInstance(backend_result, dict)
        self.assertEqual(backend_result["answers"][0]["question_id"], "q-next")
        self.assertEqual(mapped["raw_ollama_response"], result.raw_ollama_response)
        self.assertIsNot(mapped["raw_ollama_response"], mapped["backend_result"])

    def test_response_with_empty_ok_probabilities_is_rejected(self) -> None:
        answer = QuestionAnswer(
            question_id="q-next",
            choice="call",
            probabilities=(),
            confidence=0.8,
            status=ResponseStatus.OK,
        )
        backend = _SpyBackend(
            response=DecisionResponse(
                answers=(answer,), backend=Backend.DIRECT, model_id="synthetic"
            )
        )
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)

    def test_response_with_non_json_metadata_is_rejected(self) -> None:
        response = SyntheticChoiceBackend().decide(DecisionRequest.from_mapping(_VALID_REQUEST))
        object.__setattr__(response, "metadata", {"unsafe": object()})
        backend = _SpyBackend(response=response)
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)

    def test_abstain_backend_preserves_reason_through_host(self) -> None:
        EXACT_REASON = "synthetic backend requested abstention"
        backend = SyntheticAbstainBackend(abstain_reason=EXACT_REASON)
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        _patch_post(host, _vons_decide_tool_call(_VALID_REQUEST))

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.backend_result)
        answer = result.backend_result.answers[0]
        self.assertEqual(answer.status, ResponseStatus.ABSTAIN)
        self.assertIsNone(answer.choice)
        self.assertEqual(answer.abstain_reason, EXACT_REASON)

    def test_multiple_tool_calls_rejected(self) -> None:
        backend = _SpyBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        double = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "vons_decide", "arguments": json.dumps(_VALID_REQUEST)},
                },
                {
                    "type": "function",
                    "function": {"name": "vons_decide", "arguments": json.dumps(_VALID_REQUEST)},
                },
            ],
        }
        _patch_post(host, double)

        result = host.chat_with_tools("synthetic-model", [{"role": "user", "content": "x"}])

        self.assertTrue(result.tool_invoked)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.backend_result)
        self.assertEqual(len(backend.calls), 0)


class RedirectEscapeTests(unittest.TestCase):
    def test_host_validates_url_before_each_http_call(self) -> None:
        backend = SyntheticAbstainBackend()
        host = OllamaToolCallHost("http://127.0.0.1:11434", backend)
        # Simulate opener.open raising an error - we already know the host
        # validates host on the way in via constructor. Now confirm the
        # per-call _post path also re-validates by replacing host.host with
        # a remote host and calling _post directly.
        original_host = host.host
        try:
            host.host = "http://evil.example.com:11434"  # type: ignore[misc]
            with self.assertRaises(OllamaToolCallError):
                host._post("/api/chat", {"model": "x"})
        finally:
            host.host = original_host

    def test_non_loopback_redirect_is_blocked_by_handler(self) -> None:
        """`_LoopbackRedirectHandler` rejects redirects leaving loopback.

        We do not actually execute the redirect handler through an HTTP server
        in synthetic fixtures; we confirm the handler class exists and the
        helper `_validate_loopback_host` rejects a redirect-style URL that
        would point away from loopback.
        """
        from vons.ollama import _LoopbackRedirectHandler

        self.assertTrue(callable(_LoopbackRedirectHandler.redirect_request))
        with self.assertRaises(OllamaToolCallError):
            _validate_loopback_host("http://10.0.0.5:11434/redirected")
        with self.assertRaises(OllamaToolCallError):
            _validate_loopback_host("https://remote.internal:8443/api/chat")

    def test_redirect_handler_rejects_remote_location_before_following(self) -> None:
        handler = _LoopbackRedirectHandler()
        request = Request("http://127.0.0.1:11434/api/chat")

        with self.assertRaises(OllamaToolCallError):
            handler.http_error_302(
                request,
                io.BytesIO(),
                302,
                "Found",
                {"Location": "http://192.168.1.7:11434/api/chat"},
            )


class HttpPostBoundaryTests(unittest.TestCase):
    def test_rejects_non_200_status_before_reading_body(self) -> None:
        host = OllamaToolCallHost("http://127.0.0.1:11434", SyntheticAbstainBackend())
        response = _http_response(201, b"not json")

        with (
            patch.object(host._opener, "open", return_value=response),
            self.assertRaises(OllamaToolCallError),
        ):
            host._post("/api/chat", {"model": "synthetic"})

        self.assertFalse(response.read_called)

    def test_accepts_synthetic_200_response(self) -> None:
        host = OllamaToolCallHost("http://127.0.0.1:11434", SyntheticAbstainBackend())
        response = _http_response(200, b'{"message":{"content":"ok"}}')

        with patch.object(host._opener, "open", return_value=response):
            parsed = host._post("/api/chat", {"model": "synthetic"})

        self.assertEqual(parsed, {"message": {"content": "ok"}})
        self.assertTrue(response.read_called)

    def test_rejects_oversized_response_body(self) -> None:
        host = OllamaToolCallHost("http://127.0.0.1:11434", SyntheticAbstainBackend())
        response = _http_response(200, b" " * (524_288 + 1))

        with (
            patch.object(host._opener, "open", return_value=response),
            self.assertRaises(OllamaToolCallError),
        ):
            host._post("/api/chat", {"model": "synthetic"})

    def test_rejects_duplicate_json_keys(self) -> None:
        host = OllamaToolCallHost("http://127.0.0.1:11434", SyntheticAbstainBackend())
        response = _http_response(200, b'{"message":{},"message":{}}')

        with (
            patch.object(host._opener, "open", return_value=response),
            self.assertRaises(OllamaToolCallError),
        ):
            host._post("/api/chat", {"model": "synthetic"})


class SyntheticBackendContractTests(unittest.TestCase):
    def test_synthetic_abstain_requires_reason(self) -> None:
        backend = SyntheticAbstainBackend("my reason")
        request = DecisionRequest.from_mapping(_VALID_REQUEST)
        response = backend.decide(request)
        for answer in response.answers:
            self.assertEqual(answer.abstain_reason, "my reason")
            self.assertIsNone(answer.choice)

    def test_synthetic_choice_preserves_position(self) -> None:
        backend = SyntheticChoiceBackend(choice_index=1)
        request = DecisionRequest.from_mapping(_VALID_REQUEST)
        response = backend.decide(request)
        self.assertEqual(response.answers[0].choice, "clarify")  # index 1
        probs = {p.option: p.probability for p in response.answers[0].probabilities}
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_synthetic_choice_out_of_range_is_error(self) -> None:
        backend = SyntheticChoiceBackend(choice_index=99)
        request = DecisionRequest.from_mapping(_VALID_REQUEST)
        with self.assertRaises(OnnxRuntimeError):
            backend.decide(request)


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.token_type_ids = [0] * len(ids)


class _FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 1

    def encode(self, text: str, **_kwargs: Any) -> _FakeEncoding:
        return _FakeEncoding([1, *range(3, 3 + len(text.split())), 2])


class _FakeOnnxSession:
    def __init__(self, backend: Backend, *, bad_shape: bool = False) -> None:
        self.backend = backend
        self.bad_shape = bad_shape
        self.feeds: list[dict[str, Any]] = []
        self.input_names = ["input_ids", "attention_mask", "token_type_ids", "option_mask"]
        if backend is Backend.DIFFUSION:
            self.input_names.append("initial_noise")
        self.output_names = [
            "logits" if backend is Backend.DIRECT else "scores",
            "answerability",
        ]

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self.input_names]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self.output_names]

    def run(self, output_names: list[str], feed: dict[str, Any]) -> list[Any]:
        import numpy as np

        self.feeds.append({key: np.array(value, copy=True) for key, value in feed.items()})
        slots = feed["option_mask"].shape[1]
        width = slots + 1 if self.bad_shape else slots
        scores = np.linspace(2.0, -2.0, width, dtype=np.float32)[None, :]
        values = {
            self.output_names[0]: scores,
            "answerability": np.array([8.0], dtype=np.float32),
        }
        return [values[name] for name in output_names]


def _runtime_backend(
    backend_type: Backend, *, bad_shape: bool = False
) -> tuple[OnnxRuntimeBackend, _FakeOnnxSession]:
    inputs = ["input_ids", "attention_mask", "token_type_ids", "option_mask"]
    if backend_type is Backend.DIFFUSION:
        inputs.append("initial_noise")
    outputs = ["logits" if backend_type is Backend.DIRECT else "scores", "answerability"]
    manifest = {
        "metadata": {
            "backend": backend_type.value,
            "option_count": 4,
            "sequence_length": 64,
            "input_names": inputs,
            "output_names": outputs,
            "diffusion_conditioning": "token_cross_attention",
        },
        "files": [],
    }
    bundle = VerifiedBundle(
        root=Path("."),
        manifest_path=Path("bundle-manifest-v1.json"),
        manifest=manifest,
        model_id="a" * 64,
    )
    session = _FakeOnnxSession(backend_type, bad_shape=bad_shape)
    return (
        OnnxRuntimeBackend(bundle, tokenizer=_FakeTokenizer(), session=session),
        session,
    )


class OnnxRuntimeExecutionTests(unittest.TestCase):
    def test_local_tokenizer_adapter_exposes_token_ids_for_budget_check(self) -> None:
        class RawTokenizer:
            max_length: int | None = None

            def enable_truncation(self, *, max_length: int) -> None:
                self.max_length = max_length

            def no_truncation(self) -> None:
                self.max_length = None

            def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
                ids = [1, *range(2, 2 + len(text.split())), 3]
                return SimpleNamespace(
                    ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids)
                )

        tokenizer = _LocalTokenizer(RawTokenizer(), pad_token_id=0, cls_token_id=1)

        encoded = tokenizer.encode("one two three", max_length=32)

        self.assertGreater(len(encoded.ids), 3)
        self.assertEqual(len(encoded.ids), len(encoded.attention_mask))
        self.assertIsNone(tokenizer._tokenizer.max_length)

    def test_executes_direct_session_and_returns_validated_choice(self) -> None:
        backend, session = _runtime_backend(Backend.DIRECT)
        request = DecisionRequest.from_mapping(_VALID_REQUEST)

        result = backend.decide(request)

        self.assertEqual(result.backend, Backend.DIRECT)
        self.assertEqual(result.answers[0].question_id, "q-next")
        self.assertEqual(result.answers[0].choice, "call")
        self.assertEqual(session.feeds[0]["input_ids"].shape[:2], (1, 4))
        self.assertNotIn("initial_noise", session.feeds[0])

    def test_executes_diffusion_session_with_seeded_noise(self) -> None:
        backend, session = _runtime_backend(Backend.DIFFUSION)
        request = DecisionRequest.from_mapping(
            {**_VALID_REQUEST, "backend": "diffusion", "seed": 7}
        )

        first = backend.decide(request)
        second = backend.decide(request)

        import numpy as np

        self.assertEqual(first.answers[0].choice, "call")
        self.assertEqual(second.answers[0].choice, "call")
        self.assertEqual(session.feeds[0]["initial_noise"].shape, (1, 4))
        np.testing.assert_array_equal(
            session.feeds[0]["initial_noise"], session.feeds[1]["initial_noise"]
        )

    def test_injected_session_must_match_bundle_contract(self) -> None:
        backend, session = _runtime_backend(Backend.DIRECT)
        session.input_names.remove("option_mask")

        with self.assertRaises(OnnxRuntimeError):
            backend.decide(DecisionRequest.from_mapping(_VALID_REQUEST))

    def test_invalid_onnx_score_shape_fails_closed(self) -> None:
        backend, _session = _runtime_backend(Backend.DIRECT, bad_shape=True)

        with self.assertRaisesRegex(OnnxRuntimeError, "invalid score dimensions"):
            backend.decide(DecisionRequest.from_mapping(_VALID_REQUEST))


class BundleLoadTests(unittest.TestCase):
    def test_missing_directory_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope"
            with self.assertRaises(OnnxRuntimeError):
                VerifiedBundle.load(missing)

    def test_directory_without_manifest_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as td, self.assertRaises(OnnxRuntimeError):
            VerifiedBundle.load(td)

    def test_directory_with_manifest_missing_graph_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # A manifest with a bogus file inventory will fail verify_bundle_manifest
            bad_manifest = {
                "schema_version": "vons.bundle.manifest/v1",
                "manifest": {"path": "bundle-manifest.json"},
                "files": [
                    {"path": "nope.onnx", "role": "model_graph", "bytes": 1, "sha256": "0" * 64}
                ],
                "summary": {
                    "model_asset_bytes": 1,
                    "manifest_bytes": 0,
                    "complete_download_bytes": 0,
                    "within_model_asset_limit": True,
                    "runtime_loaded_file_count": 0,
                },
            }
            (root / "bundle-manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
            with self.assertRaises(OnnxRuntimeError):
                VerifiedBundle.load(root)


class SanitizedErrorTests(unittest.TestCase):
    def test_error_id_is_12_hex_chars(self) -> None:
        err = OllamaToolCallError()
        self.assertRegex(err.error_id, r"^[0-9a-f]{12}$")
        self.assertEqual(len(err.error_id), 12)

    def test_two_errors_have_different_ids(self) -> None:
        a = OllamaToolCallError()
        b = OllamaToolCallError()
        self.assertNotEqual(a.error_id, b.error_id)

    def test_sanitized_payload_omits_cause_text(self) -> None:
        err = OllamaToolCallError()
        san = err.sanitized()
        self.assertEqual(san["error"], "ollama_tool_call_failed")
        self.assertIn("error_id", san)
        self.assertEqual(len(san), 2)


class CLISmokeTests(unittest.TestCase):
    def test_cli_parser_accepts_ollama_tool_call_command(self) -> None:
        from vons.cli import _parser

        parser = _parser()
        args = parser.parse_args(
            [
                "ollama-tool-call",
                "--model",
                "qwen3.8:27b-mlx",
                "--synthetic",
                "abstain",
                "--input",
                "ignored.json",
            ]
        )
        self.assertEqual(args.command, "ollama-tool-call")
        self.assertEqual(args.model, "qwen3.8:27b-mlx")
        self.assertEqual(args.synthetic, "abstain")
        self.assertIsNone(args.think)

    def test_cli_parser_exposes_both_thinking_modes(self) -> None:
        from vons.cli import _parser

        parser = _parser()
        common = [
            "ollama-tool-call",
            "--model",
            "qwen3.8:27b-mlx",
            "--synthetic",
            "abstain",
            "--input",
            "ignored.json",
        ]

        self.assertIs(parser.parse_args([*common, "--think"]).think, True)
        self.assertIs(parser.parse_args([*common, "--no-think"]).think, False)

    def test_cli_forwards_thinking_override_to_host(self) -> None:
        from vons.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "input.json"
            input_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "synthetic"}]}),
                encoding="utf-8",
            )
            for option, expected in (("--think", True), ("--no-think", False)):
                with self.subTest(option=option):
                    host = MagicMock()
                    host.chat_with_tools.return_value = SimpleNamespace(error=None)
                    with (
                        patch("vons.cli.OllamaToolCallHost", return_value=host),
                        patch("vons.cli.tool_call_result_to_mapping", return_value={}),
                        patch("builtins.print"),
                    ):
                        rc = cli_main(
                            [
                                "ollama-tool-call",
                                "--model",
                                "synthetic-model",
                                "--synthetic",
                                "abstain",
                                "--input",
                                str(input_path),
                                option,
                            ]
                        )

                    self.assertEqual(rc, 0)
                    host.chat_with_tools.assert_called_once()
                    self.assertIs(host.chat_with_tools.call_args.kwargs["think"], expected)

    def test_cli_runtime_tool_call_error_has_distinct_exit_code(self) -> None:
        from vons.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "input.json"
            input_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "synthetic"}]}),
                encoding="utf-8",
            )
            host = MagicMock()
            host.chat_with_tools.return_value = SimpleNamespace(error=OllamaToolCallError())
            with (
                patch("vons.cli.OllamaToolCallHost", return_value=host),
                patch("vons.cli.tool_call_result_to_mapping", return_value={}),
                patch("builtins.print"),
            ):
                rc = cli_main(
                    [
                        "ollama-tool-call",
                        "--model",
                        "synthetic-model",
                        "--synthetic",
                        "abstain",
                        "--input",
                        str(input_path),
                    ]
                )

        self.assertEqual(rc, 18)
        self.assertNotEqual(rc, 17)

    def test_cli_rejects_non_loopback_ollama_check_host(self) -> None:
        from vons.cli import main as cli_main

        with (
            patch("vons.cli.OllamaClient", side_effect=ValueError("non-loopback")) as factory,
            patch("sys.stderr"),
        ):
            rc = cli_main(
                ["ollama-check", "--models", "synthetic", "--host", "http://192.0.2.4:11434"]
            )

        self.assertEqual(rc, 5)
        factory.assert_called_once_with("http://192.0.2.4:11434")

    def test_cli_requires_backend_flag(self) -> None:
        from vons.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "in.json"
            input_path.write_text(json.dumps({"messages": []}), encoding="utf-8")
            rc = cli_main(["ollama-tool-call", "--model", "qwen", "--input", str(input_path)])
        # Exit code 2 means neither --bundle nor --synthetic supplied
        self.assertEqual(rc, 2)

    def test_cli_missing_messages_is_error(self) -> None:
        from vons.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "in.json"
            input_path.write_text("{}", encoding="utf-8")
            rc = cli_main(
                [
                    "ollama-tool-call",
                    "--model",
                    "qwen",
                    "--synthetic",
                    "abstain",
                    "--input",
                    str(input_path),
                ]
            )
        self.assertEqual(rc, 3)

    def test_tool_call_input_reader_enforces_byte_limit(self) -> None:
        from vons.cli import _MAX_TOOL_CALL_INPUT_BYTES, _read_tool_call_input

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "oversized.json"
            path.write_bytes(b" " * (_MAX_TOOL_CALL_INPUT_BYTES + 1))

            with self.assertRaisesRegex(ValueError, "size limit"):
                _read_tool_call_input(path)

    def test_tool_call_input_reader_rejects_unknown_root_keys(self) -> None:
        from vons.cli import _read_tool_call_input

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.json"
            path.write_text('{"messages": [], "extra": true}', encoding="utf-8")

            with self.assertRaises(TypeError):
                _read_tool_call_input(path)

    def test_cli_backend_preflight_codes_are_distinct_and_happen_before_host(self) -> None:
        from vons.cli import main as cli_main

        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "input.json"
            input_path.write_text(
                json.dumps({"messages": [{"role": "user", "content": "synthetic"}]}),
                encoding="utf-8",
            )
            bundle = VerifiedBundle(
                root=Path(td),
                manifest_path=Path(td) / "bundle-manifest-v1.json",
                manifest={
                    "metadata": {
                        "backend": "direct",
                        "option_count": 4,
                        "sequence_length": 64,
                        "input_names": [
                            "input_ids",
                            "attention_mask",
                            "token_type_ids",
                            "option_mask",
                        ],
                        "output_names": ["logits", "answerability"],
                    },
                    "files": [],
                },
                model_id="a" * 64,
            )
            for error, expected_code in (
                (OnnxRuntimeUnavailableError("missing synthetic dependency"), 16),
                (OnnxRuntimeError("invalid synthetic backend"), 17),
            ):
                with self.subTest(expected_code=expected_code):
                    with (
                        patch("vons.cli.VerifiedBundle.load", return_value=bundle),
                        patch.object(OnnxRuntimeBackend, "prepare", side_effect=error),
                        patch("vons.cli.OllamaToolCallHost") as host_factory,
                    ):
                        rc = cli_main(
                            [
                                "ollama-tool-call",
                                "--model",
                                "synthetic",
                                "--bundle",
                                td,
                                "--input",
                                str(input_path),
                            ]
                        )

                    self.assertEqual(rc, expected_code)
                    host_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
