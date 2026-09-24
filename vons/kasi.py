"""KASI proposal adapter for Vons decisions.

KASI owns the host-facing action contract. Vons can rank candidates and
abstain, while this adapter keeps consent and tool execution outside the
model. The adapter is intentionally serializable so a future KASI package can
consume it without importing the research model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from .contract import (
    KASIAction,
    _call_fingerprint,
    _normalize_risk,
    _validate_tool_name,
    derive_kasi_action,
)


@dataclass(frozen=True)
class ToolPolicy:
    """Host-owned policy for one executable tool.

    The proposal's risk field is advisory. This policy is the authority that
    decides whether an unconfirmed call may pass through the adapter.
    """

    risk: str

    def __post_init__(self) -> None:
        if _normalize_risk(self.risk) is None:
            raise ValueError("tool policy risk must be low, medium, high, or critical")


@dataclass(frozen=True)
class KASICall:
    """A proposed tool call; it is never executed by Vons."""

    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if not isinstance(self.arguments, Mapping):
            raise TypeError("KASI call arguments must be an object")
        # Break aliases to caller-owned nested objects before the call can be
        # displayed, fingerprinted, or handed to a host executor.
        copied = deepcopy(dict(self.arguments))
        _call_fingerprint({"name": self.name, "arguments": copied})
        object.__setattr__(self, "arguments", MappingProxyType(copied))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> KASICall:
        if not isinstance(value, Mapping):
            raise TypeError("KASI call must be an object")
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise TypeError("KASI call arguments must be an object")
        return cls(name=name, arguments=arguments)

    def fingerprint(self) -> str:
        return _call_fingerprint(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": deepcopy(dict(self.arguments))}


@dataclass(frozen=True)
class KASIProposal:
    """Structured output from a frontier model or Vons choice head."""

    calls: tuple[KASICall, ...] = ()
    confidence: float = 0.0
    risk: str | None = None
    response: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.calls, tuple) or any(not isinstance(call, KASICall) for call in self.calls):
            raise TypeError("calls must be a tuple of KASICall values")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("KASI confidence must be finite and in [0, 1]")
        if self.risk is not None and not isinstance(self.risk, str):
            raise TypeError("KASI risk must be a string or None")
        if self.response is not None and not isinstance(self.response, str):
            raise TypeError("KASI response must be a string or None")
        if self.response is not None and not self.response.strip():
            raise ValueError("KASI response must not be empty")
        if self.calls and self.response is not None:
            raise ValueError("KASI proposal cannot contain both calls and response")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> KASIProposal:
        if not isinstance(value, Mapping):
            raise TypeError("KASI proposal must be an object")
        confidence = float(value.get("confidence", 0.0))
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("KASI confidence must be finite and in [0, 1]")
        calls_value = value.get("calls", ())
        if not isinstance(calls_value, Sequence) or isinstance(calls_value, (str, bytes, bytearray)):
            raise TypeError("KASI calls must be a list")
        if any(not isinstance(item, Mapping) for item in calls_value):
            raise TypeError("each KASI call must be an object")
        proposal = cls(
            calls=tuple(KASICall.from_mapping(item) for item in calls_value),
            confidence=confidence,
            risk=None if value.get("risk") is None else str(value["risk"]),
            response=None if value.get("response") is None else str(value["response"]),
        )
        return proposal

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "calls": [call.to_mapping() for call in self.calls],
            "confidence": self.confidence,
            "risk": self.risk,
        }
        if self.response is not None:
            result["response"] = self.response
        return result


@dataclass(frozen=True)
class KASIActionDecision:
    action: KASIAction
    calls: tuple[KASICall, ...] = ()
    prompt: str | None = None
    reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "calls": [call.to_mapping() for call in self.calls],
            "prompt": self.prompt,
            "reason": self.reason,
        }


class KASIAdapter:
    """Turn a ranked Vons proposal into a safe, host-executable KASI action."""

    def __init__(self, *, tool_policy: Mapping[str, ToolPolicy | str] | None = None,
                 threshold: float = 0.55) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if tool_policy is not None and not isinstance(tool_policy, Mapping):
            raise TypeError("tool_policy must be an object")
        normalized: dict[str, str] = {}
        for name, policy in (tool_policy or {}).items():
            _validate_tool_name(name)
            normalized[name] = policy.risk if isinstance(policy, ToolPolicy) else ToolPolicy(policy).risk
        self._tool_policy = MappingProxyType(normalized)
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def tool_policy(self) -> Mapping[str, str]:
        return self._tool_policy

    def decide(self, proposal: KASIProposal, *, confirmed_tools: Collection[str] | None = None,
               confirmed_calls: Collection[str] | None = None) -> KASIActionDecision:
        if not isinstance(proposal, KASIProposal):
            raise TypeError("proposal must be a KASIProposal")
        # confirmed_tools is retained as a compatibility input but deliberately
        # cannot authorize a high-risk call without its exact argument digest.
        if isinstance(confirmed_tools, (str, bytes, bytearray, Mapping)):
            raise TypeError("confirmed_tools must be a collection of tool names")
        if isinstance(confirmed_calls, (str, bytes, bytearray, Mapping)):
            raise TypeError("confirmed_calls must be a collection of fingerprints")
        if confirmed_calls is not None and any(not isinstance(item, str) for item in confirmed_calls):
            raise TypeError("confirmed_calls must contain strings")
        confirmed = set(confirmed_calls or ())
        if proposal.response and not proposal.calls and proposal.confidence >= self.threshold:
            return KASIActionDecision(KASIAction.RESPOND, prompt=proposal.response, reason="direct_response")
        action = derive_kasi_action(
            proposed_calls=tuple(call.to_mapping() for call in proposal.calls),
            confidence=proposal.confidence,
            risk=proposal.risk,
            confirmed_tools=confirmed_tools,
            confirmed_call_fingerprints=confirmed,
            tool_policy=self._tool_policy,
            threshold=self.threshold,
        )
        if action is KASIAction.CLARIFY:
            return KASIActionDecision(action, prompt="Please provide the missing information before acting.", reason="low_confidence")
        if action is KASIAction.CONFIRM:
            details = ", ".join(f"{call.name}({call.to_mapping()['arguments']!r})" for call in proposal.calls)
            return KASIActionDecision(action, calls=proposal.calls, prompt=f"Confirm action: {details}.", reason="unconfirmed_or_unknown_risk")
        if action is KASIAction.REFUSE:
            return KASIActionDecision(action, prompt="No registered and safe candidate action is available.", reason="unregistered_or_no_proposed_call")
        return KASIActionDecision(action, calls=proposal.calls, reason="approved_by_host_policy")
