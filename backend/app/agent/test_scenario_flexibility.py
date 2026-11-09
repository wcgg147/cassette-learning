"""Scenario behaviour must remain flexible and learner-led."""

import unittest

from agent.intent_router import IntentRouter, TutorIntent
from agent.tutor_agent import SessionContext, TutorAgent
from agent.tutor_policy import TutorPolicyPlanner


class ScenarioFlexibilityTests(unittest.TestCase):
    def test_roleplay_request_is_recognized(self):
        result = IntentRouter().classify("我们来一遍咖啡店角色扮演")
        self.assertEqual(result.intent, TutorIntent.SCENARIO_ROLEPLAY)

    def test_normal_topic_change_is_not_a_hard_refusal(self):
        planner = TutorPolicyPlanner()
        result = IntentRouter().classify("换个话题，聊聊旅行吧")
        policy = planner.plan(intent_result=result, errors=[], cefr="B1")
        self.assertNotEqual(policy.teaching_act, "refusal")
        self.assertEqual(policy.correction_budget, 0)

    def test_options_are_only_for_scaffolding(self):
        planner = TutorPolicyPlanner()
        normal = planner.plan(
            intent_result=IntentRouter().classify("今天工作挺有意思"),
            errors=[], cefr="B1",
        )
        confused = planner.plan(
            intent_result=IntentRouter().classify("我不知道怎么说"),
            errors=[], cefr="B1",
        )
        self.assertIn("机械给2-3个选项", normal.must_avoid)
        self.assertIn("二选一", confused.conversation_hook)

    def test_opening_continues_relationship_not_role_script(self):
        agent = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            profile={"cefr": "B1"},
            scene={"name": "咖啡店", "goal": "自然点单"},
            relationship_memory={"open_loop": "用户下周要参加英文面试"},
        )
        opening = agent._build_opening_context(ctx)
        self.assertIn("英文面试", opening)
        self.assertIn("不要重复自我介绍", opening)


if __name__ == "__main__":
    unittest.main()
