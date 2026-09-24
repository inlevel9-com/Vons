import unittest

from vons.contract import (
    Backend,
    DecisionRequest,
    Question,
    QuestionType,
    derive_kasi_action,
    validate_request,
)
from vons.kasi import KASIAdapter, KASIProposal


class ContractTests(unittest.TestCase):
    def test_valid_request(self) -> None:
        validate_request(DecisionRequest("A weather tool is available.", (Question("next", QuestionType.CHOICE, "Choose.", ("call", "clarify")),), Backend.DIRECT, 7))

    def test_duplicate_questions_rejected(self) -> None:
        request = DecisionRequest("state", (Question("q", QuestionType.CHOICE, "pick", ("a", "b")), Question("q", QuestionType.CHOICE, "pick", ("a", "b"))))
        with self.assertRaises(ValueError):
            validate_request(request)

    def test_kasi_confirm_is_wrapper_derived(self) -> None:
        self.assertEqual(derive_kasi_action(proposed_calls=({"name": "unlock"},), confidence=0.99, risk="high", confirmed_tools=set(), tool_policy={"unlock": "high"}).value, "confirm")
        self.assertEqual(derive_kasi_action(proposed_calls=({"name": "read"},), confidence=0.99, risk="low", confirmed_tools=set(), tool_policy={"read": "low"}).value, "call")

    def test_unregistered_tool_cannot_call(self) -> None:
        action = derive_kasi_action(proposed_calls=({"name": "read"},), confidence=0.99, risk="low", tool_policy={})
        self.assertEqual(action.value, "refuse")

    def test_kasi_response_and_low_confidence(self) -> None:
        adapter = KASIAdapter(tool_policy={"read_weather": "low"})
        response = adapter.decide(KASIProposal(response="The answer is ready.", confidence=0.9))
        self.assertEqual(response.action.value, "respond")
        clarify = adapter.decide(KASIProposal(calls=(), confidence=0.2))
        self.assertEqual(clarify.action.value, "clarify")


if __name__ == "__main__":
    unittest.main()
