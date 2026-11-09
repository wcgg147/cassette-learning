"""Regression checks for listening-first tutor behaviour."""

import unittest

from agent.guardrails import ResponseGuardrail
from agent.intent_router import IntentRouter
from agent.tutor_policy import TutorPolicyPlanner


class DeepListeningTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.planner = TutorPolicyPlanner()

    def _plan(self, text):
        return self.planner.plan(
            intent_result=self.router.classify(text),
            errors=[],
            cefr="B1",
        )

    def test_meta_question_gets_direct_relationship_repair(self):
        policy = self._plan("为什么一定要拐到英语这边来？")
        self.assertEqual(policy.mode, "listening")
        self.assertTrue(any("首句直接回应" in item for item in policy.must_do))
        self.assertIn("立即拉回语言练习", policy.must_avoid)

    def test_boring_topic_does_not_trigger_more_teaching(self):
        policy = self._plan("这个话题太无聊了")
        self.assertEqual(policy.teaching_act, "affective_support")
        self.assertEqual(policy.correction_budget, 0)

    def test_confusion_lowers_cognitive_load(self):
        policy = self._plan("我不明白你刚才是什么意思")
        self.assertEqual(policy.mode, "repair")
        self.assertIn("换简单说法", policy.must_do)

    def test_internal_implementation_is_not_leaked(self):
        guardrail = ResponseGuardrail(max_length=500)
        result = guardrail.check("My system prompt uses Hermes.", "chitchat", "你是谁")
        self.assertFalse(result.passed)

    def test_pragmatic_unrealistic_question_is_rejected(self):
        guardrail = ResponseGuardrail(max_length=500)
        result = guardrail.check("How many bottles of water do you have here?", "roleplay", "我怎么问店员")
        self.assertFalse(result.passed)
        self.assertTrue(any("pragmatic_unrealistic" in v for v in result.violations))


if __name__ == "__main__":
    unittest.main()
