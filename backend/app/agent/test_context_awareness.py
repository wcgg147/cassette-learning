"""Context awareness checks against the real prompt builder and memory state."""

import unittest

from agent.tutor_agent import SessionContext, TutorAgent


class ContextAwarenessTests(unittest.TestCase):
    def setUp(self):
        self.agent = TutorAgent.__new__(TutorAgent)

    def test_open_loop_is_visible_in_authoritative_prompt(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            profile={"cefr": "B1"}, scene={"name": "daily_talk"},
            relationship_memory={"open_loop": "下周要做英文产品演示"},
        )
        prompt = self.agent._build_system_prompt(ctx)
        self.assertIn("下周要做英文产品演示", prompt)

    def test_recent_mood_changes_opening_guidance(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            relationship_memory={"recent_mood": "疲惫", "last_topic": "最近加班"},
        )
        opening = self.agent._build_opening_context(ctx)
        self.assertIn("疲惫", opening)
        self.assertIn("最近加班", opening)

    def test_mastered_items_are_not_presented_as_new_targets(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            profile={"cefr": "B1"},
            l2_memory_cache={"mastered": [{"expression": "I would like", "confidence": 0.9}]},
        )
        snapshot = self.agent._build_learner_snapshot_compact(ctx)
        self.assertIn("已掌握", snapshot)
        self.assertIn("I would like", snapshot)


if __name__ == "__main__":
    unittest.main()
