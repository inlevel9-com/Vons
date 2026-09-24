import math
import unittest

from vons.contract import (
    DecisionResponse,
    OptionProbability,
    QuestionAnswer,
    ResponseStatus,
    derive_kasi_action,
)
from vons.kasi import KASIAdapter, KASICall, KASIProposal
from vons.models import ddim_step, linear_beta_schedule, require_torch

try:
    torch, _ = require_torch()
except RuntimeError:
    torch = None


class KASIRegressionTests(unittest.TestCase):
    def test_high_risk_proposal_derives_confirm_in_host_adapter(self) -> None:
        adapter = KASIAdapter(tool_policy={"unlock": "high"})
        proposal = KASIProposal(
            calls=(KASICall(name="unlock", arguments={"target": "lab"}),),
            confidence=0.99,
            risk="high",
        )

        decision = adapter.decide(proposal)

        self.assertEqual(decision.action.value, "confirm")
        self.assertEqual(decision.reason, "unconfirmed_or_unknown_risk")
        self.assertEqual(decision.calls, proposal.calls)

    def test_host_risk_cannot_be_lowered_by_model_risk(self) -> None:
        adapter = KASIAdapter(tool_policy={"unlock": "high"})
        proposal = KASIProposal(
            calls=(KASICall(name="unlock", arguments={"target": "lab"}),),
            confidence=0.99,
            risk="low",
        )
        self.assertEqual(adapter.decide(proposal).action.value, "confirm")

    def test_exact_call_consent_is_required_for_arguments(self) -> None:
        adapter = KASIAdapter(tool_policy={"unlock": "high"})
        first = KASICall(name="unlock", arguments={"target": "lab"})
        second = KASICall(name="unlock", arguments={"target": "vault"})
        proposal = KASIProposal(calls=(second,), confidence=0.99, risk="high")
        self.assertEqual(adapter.decide(proposal, confirmed_calls={first.fingerprint()}).action.value, "confirm")
        self.assertEqual(adapter.decide(proposal, confirmed_calls={second.fingerprint()}).action.value, "call")
        self.assertEqual(adapter.decide(proposal, confirmed_tools={"unlock"}).action.value, "confirm")

    def test_unregistered_call_is_refused(self) -> None:
        adapter = KASIAdapter()
        proposal = KASIProposal(calls=(KASICall(name="read", arguments={}),), confidence=0.99, risk="low")
        decision = adapter.decide(proposal)
        self.assertEqual(decision.action.value, "refuse")

    def test_invalid_confidence_fails_closed(self) -> None:
        self.assertEqual(
            derive_kasi_action(proposed_calls=({"name": "read"},), confidence=math.nan, risk="low", tool_policy={"read": "low"}).value,
            "clarify",
        )

    def test_malformed_tool_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KASICall(name="unlock\nnow", arguments={})

    def test_question_answer_invariants_and_response_parser(self) -> None:
        with self.assertRaises(ValueError):
            QuestionAnswer("q", "a", (OptionProbability("a", 0.5), OptionProbability("b", 0.5)), 0.9, ResponseStatus.ABSTAIN, "missing")
        with self.assertRaises(ValueError):
            QuestionAnswer("q", "c", (OptionProbability("a", 1.0),), 0.9, ResponseStatus.OK)
        response = DecisionResponse.from_mapping(
            {
                "answers": [
                    {
                        "question_id": "q",
                        "choice": "a",
                        "probabilities": [{"option": "a", "probability": 1.0}],
                        "confidence": 0.9,
                        "status": "ok",
                    }
                ],
                "backend": "direct",
                "model_id": "test",
            }
        )
        self.assertEqual(response.to_mapping()["answers"][0]["choice"], "a")

    def test_low_confidence_derives_clarify(self) -> None:
        adapter = KASIAdapter(tool_policy={"read": "low"})
        proposal = KASIProposal(
            calls=(KASICall(name="read", arguments={"path": "state.json"}),),
            confidence=0.2,
            risk="low",
        )

        decision = adapter.decide(proposal)

        self.assertEqual(decision.action.value, "clarify")
        self.assertEqual(decision.reason, "low_confidence")


@unittest.skipIf(torch is None, "optional torch dependency is unavailable")
class DiffusionRegressionTests(unittest.TestCase):
    def test_linear_beta_schedule_is_deterministic_and_increasing(self) -> None:
        first = linear_beta_schedule(8)
        second = linear_beta_schedule(8)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.all(first[1:] > first[:-1]).item()))

    def test_ddim_step_preserves_score_vector_shape(self) -> None:
        torch.manual_seed(17)
        sample = torch.randn(2, 4)
        predicted_noise = torch.randn(2, 4)
        alpha = torch.tensor(0.8)
        alpha_previous = torch.tensor(0.9)

        updated = ddim_step(sample, predicted_noise, alpha, alpha_previous)

        self.assertEqual(updated.shape, sample.shape)


if __name__ == "__main__":
    unittest.main()
