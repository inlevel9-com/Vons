import hashlib
import unittest

from vons.contract import (
    Backend,
    DecisionRequest,
    Question,
    QuestionType,
    derive_kasi_action,
    validate_request,
)
from vons.kasi import KASIAdapter, KASICall, KASIProposal


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

    def test_kasi_fingerprint_matches_shared_canonical_json_number_vector(self) -> None:
        canonical = (
            r'''{"arguments":{"literals":[null,true,false],"numbers":'''
            r'''[333333333.3333333,4.5,0.002,0.000001,1e-7,1424953923781206.2,0],'''
            r'''"string":"€$\u000f\nA'B\"\\\\\"/"},"name":"read"}'''
        )
        call = KASICall(
            "read",
            {
                "numbers": [333333333.33333329, 4.5, 2e-3, 1e-6, 1e-7, 1424953923781206.25, -0.0],
                "string": "€$\u000f\nA'B\"\\\\\"/",
                "literals": [None, True, False],
            },
        )

        self.assertEqual(call.fingerprint(), hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    def test_kasi_fingerprint_matches_shared_utf16_key_order_and_rejects_invalid_values(self) -> None:
        canonical = (
            r'{"arguments":{"\r":"Carriage Return","1":"One","":"Control",'
            r'"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
            r'"😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"},'
            r'"name":"read"}'
        )
        arguments = {
            "€": "Euro Sign",
            "\r": "Carriage Return",
            "\ufb33": "Hebrew Letter Dalet With Dagesh",
            "1": "One",
            "😀": "Emoji: Grinning Face",
            "\u0080": "Control",
            "ö": "Latin Small Letter O With Diaeresis",
        }
        call = KASICall("read", arguments)
        self.assertEqual(call.fingerprint(), hashlib.sha256(canonical.encode("utf-8")).hexdigest())

        invalid_values = [float("nan"), float("inf"), 1 << 53, "\ud800"]
        for value in invalid_values:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                KASICall("read", {"value": value})

    def test_kasi_fingerprint_rejects_numeric_subclasses_that_can_change_json(self) -> None:
        class InjectingInt(int):
            def __str__(self) -> str:
                return '1,"injected":true'

        class InjectingFloat(float):
            def __repr__(self) -> str:
                return '1,"injected":true'

        values = (
            ("int subclass", InjectingInt(1)),
            ("float subclass", InjectingFloat(1.0)),
        )
        for label, value in values:
            with self.subTest(value=label), self.assertRaisesRegex(ValueError, "JSON-compatible"):
                KASICall("read", {"value": value}).fingerprint()


if __name__ == "__main__":
    unittest.main()
