# -*- coding: utf-8 -*-
"""闲聊盘摘掉导演（2026-10-21 用户拍板）的回归。

跑法：cd chanpinzhujian/api && PYTHONPATH=. python -m unittest agent.test_chat_tape_director_removal

守四件事：
1. 纯闲聊轮不再调导演；联网轮、建场景轮、练习盘、素材盘、场景中仍调。
2. 闲聊盘 prompt 不再注入陈旧的 [Conversation Director - authoritative]；
   联网轮的检索计划仍注入。
3. 空 content_need 不再产生"导演是本轮最高优先级"这句——旧写法会让模型
   服从一个空权威，是旧话题被拽回来的入口。
4. CHAT_TAPE_DIRECTOR_ENABLED 打开后回到旧行为（一键回滚）。
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, ".")

os.environ.setdefault("DATABASE_URL", "postgresql://llm_user:***@localhost:5432/llm_os")
os.environ.setdefault("MEMORY_BACKEND", "local")

import config as app_config  # noqa: E402
import agent.tutor_agent as tutor_agent  # noqa: E402
from agent.test_expression_contract import control_ctx, stable_prepared  # noqa: E402
from agent.tutor_policy import TutorPolicy  # noqa: E402


def _chat_ctx():
    agent, ctx = control_ctx(0)
    ctx.scene = {}
    ctx.scene_context_set = False
    ctx.scenario_role = ""
    ctx.cassette_mode = "chat"
    ctx.persona_state = {"mode": "companion"}
    return agent, ctx


def _prepared_with_director(grounded: bool = False) -> dict:
    prepared = stable_prepared("continue_roleplay_scene")
    plan = {
        "content_need": "上一轮没说完的那件事",
        "interaction_frame": "general",
        "response_need": "general_chat",
        "_meta": {"source": "cached_scene_plan", "reused_scene_plan": True},
    }
    if grounded:
        plan["grounding_need"] = "web_search"
        plan["_meta"] = {
            "source": "director_model",
            "reused_scene_plan": False,
            "web_search_used": True,
        }
    prepared["conversation_director"] = plan
    prepared["constraints"]["conversation_director"] = plan
    return prepared


class DirectorGateTests(unittest.TestCase):
    """第 1 条：谁还允许调导演。"""

    def test_plain_chat_skips_director(self):
        agent, ctx = _chat_ctx()
        self.assertFalse(
            agent._chat_tape_director_allowed(
                ctx, scene_pending=False, search_needed=False
            )
        )

    def test_search_turn_keeps_director(self):
        agent, ctx = _chat_ctx()
        self.assertTrue(
            agent._chat_tape_director_allowed(
                ctx, scene_pending=False, search_needed=True
            )
        )

    def test_scene_pending_keeps_director(self):
        agent, ctx = _chat_ctx()
        self.assertTrue(
            agent._chat_tape_director_allowed(
                ctx, scene_pending=True, search_needed=False
            )
        )

    def test_scene_tape_keeps_director(self):
        agent, ctx = _chat_ctx()
        ctx.scene_context_set = True
        self.assertTrue(
            agent._chat_tape_director_allowed(
                ctx, scene_pending=False, search_needed=False
            )
        )

    def test_expression_tape_keeps_director(self):
        agent, ctx = _chat_ctx()
        ctx.cassette_mode = "expression"
        self.assertTrue(
            agent._chat_tape_director_allowed(
                ctx, scene_pending=False, search_needed=False
            )
        )

    def test_material_tape_keeps_director(self):
        agent, ctx = _chat_ctx()
        ctx.material_context = {"material_id": "m1"}
        self.assertTrue(
            agent._chat_tape_director_allowed(
                ctx, scene_pending=False, search_needed=False
            )
        )

    def test_switch_restores_old_behavior(self):
        agent, ctx = _chat_ctx()
        with mock.patch.object(app_config, "CHAT_TAPE_DIRECTOR_ENABLED", True):
            self.assertTrue(
                agent._chat_tape_director_allowed(
                    ctx, scene_pending=False, search_needed=False
                )
            )


class PromptSectionTests(unittest.TestCase):
    """第 2 条：陈旧计划不得再以 authoritative 出现。"""

    def test_chat_prompt_drops_stale_director_section(self):
        agent, ctx = _chat_ctx()
        prompt = agent._build_control_prompt(ctx, _prepared_with_director(), "")
        self.assertNotIn("[Conversation Director - authoritative]", prompt)

    def test_chat_prompt_keeps_grounded_search_plan(self):
        agent, ctx = _chat_ctx()
        prompt = agent._build_control_prompt(
            ctx, _prepared_with_director(grounded=True), ""
        )
        self.assertIn("[Conversation Director - authoritative]", prompt)

    def test_roleplay_prompt_keeps_director_section(self):
        agent, ctx = _chat_ctx()
        ctx.scene_context_set = True
        ctx.scenario_role = "咖啡店店员"
        prompt = agent._build_control_prompt(ctx, _prepared_with_director(), "")
        self.assertIn("[Conversation Director - authoritative]", prompt)


class PolicyPriorityTests(unittest.TestCase):
    """第 3 条：空 content_need 不再声明最高优先级。"""

    def test_empty_content_need_drops_priority_line(self):
        policy = TutorPolicy()
        tutor_agent.TutorAgent._apply_director_policy(
            policy, {"content_need": "", "_meta": {"source": "cached_scene_plan"}}
        )
        joined = "\n".join(policy.must_do)
        self.assertNotIn("Conversation Director 是本轮最高优先级", joined)
        self.assertNotIn("先满足 content_need", joined)
        # 另外两条与导演无关的护栏必须还在。
        self.assertIn("只使用已知信息", joined)

    def test_real_content_need_keeps_priority_line(self):
        policy = TutorPolicy()
        tutor_agent.TutorAgent._apply_director_policy(
            policy,
            {"content_need": "他在问退货流程", "_meta": {"source": "director_model"}},
        )
        joined = "\n".join(policy.must_do)
        self.assertIn("Conversation Director 是本轮最高优先级", joined)


if __name__ == "__main__":
    unittest.main()
