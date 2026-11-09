"""Regression checks for natural, graduated corrective feedback."""

import unittest

from agent.intent_router import IntentRouter
from agent.tutor_policy import TutorPolicyPlanner
from engine.learning_engine import LearningEngine


class NativeCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.planner = TutorPolicyPlanner()

    def _plan(self, text, errors, cefr="B1", style="recast"):
        return self.planner.plan(
            intent_result=self.router.classify(text),
            errors=errors,
            cefr=cefr,
            correction_style=style,
        )

    def test_small_error_uses_recast_without_stopping_content(self):
        policy = self._plan("I didn't went yesterday.", [{"type": "en_past_tense"}])
        self.assertEqual(policy.teaching_act, "recast")
        self.assertIn("沿着用户内容", policy.conversation_hook)

    def test_beginner_key_error_gets_one_clear_correction(self):
        policy = self._plan("I didn't went yesterday.", [{"type": "en_past_tense"}], cefr="A1")
        self.assertEqual(policy.teaching_act, "explicit_correction")
        self.assertEqual(policy.correction_budget, 1)

    def test_advanced_learner_can_self_correct(self):
        policy = self._plan(
            "I didn't went yesterday.",
            [{"type": "en_past_tense"}],
            cefr="B2",
            style="elicitation",
        )
        self.assertEqual(policy.teaching_act, "elicitation")

    def test_no_error_does_not_force_a_language_point(self):
        policy = self._plan("I went there yesterday and had a great time.", [])
        self.assertEqual(policy.teaching_act, "natural_extension")
        self.assertEqual(policy.correction_budget, 0)

    def test_minibar_negative_past_error_is_detected(self):
        engine = LearningEngine.__new__(LearningEngine)
        errors = engine._detect_english_errors("No, I'm not used the minibar.")
        self.assertEqual(errors[0]["type"], "en_negative_past")
        diagnostic = engine._enrich_diagnostic(
            errors[0], "No, I'm not used the minibar."
        )
        self.assertEqual(diagnostic["correction"], "I didn't use the")

    def test_payment_instrument_collocation_is_not_marked_natural(self):
        engine = LearningEngine.__new__(LearningEngine)
        errors = engine._detect_english_errors(
            "Hi, my card can't pay. Can you help me?"
        )
        self.assertEqual(errors[0]["type"], "en_payment_collocation")
        diagnostic = engine._enrich_diagnostic(
            errors[0], "Hi, my card can't pay. Can you help me?"
        )
        self.assertEqual(diagnostic["correction"], "My card isn't working.")


if __name__ == "__main__":
    unittest.main()
