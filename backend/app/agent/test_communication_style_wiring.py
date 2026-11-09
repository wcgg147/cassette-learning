# -*- coding: utf-8 -*-
"""表达习惯标签的读侧接线测试（plans/0919-03 §4.6）。

守两件事：
1. 标签真的进了权威 system prompt（不是只写进了 L1 就算完）；
2. L1 是整档覆盖写，标签不会被不带该字段的画像写入抹掉。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

os.environ.setdefault("DATABASE_URL", "postgresql://llm_user:llm_pass@localhost:5432/llm_os")
os.environ.setdefault("MEMORY_BACKEND", "local")

import agent.tutor_agent as tutor_agent  # noqa: E402
from agent.test_expression_contract import (  # noqa: E402
    control_ctx,
    stable_prepared,
)
from agent.tutor_agent import TutorAgent  # noqa: E402
from memory.l1_file import L1FileProfile  # noqa: E402


class Ctx:
    """_build_system_prompt 所需的最小 ctx（照 test_name_memory 的范式）。

    _build_system_prompt 会读几十个 ctx 字段；未列出的按"空值"兜底，
    免得每加一个 prompt 区块就要回来补一次测试桩。字典类字段必须显式
    给 {}——它们要接 .get()。
    """

    user_id = "style-user"
    relationship_memory = {}
    session_intent_resolution = ""
    session_memory_packet = {}
    l2_memory_cache = {}
    profile = {}
    scene = {}
    scene_context_set = False
    learner_role = ""
    scenario_role = ""
    scenario_brief = ""
    roleplay_turn = 0
    target_lang = "en"
    style_parameters = None
    session_facts = []
    recent_sessions = []
    cross_session_recall = {}
    memory_route = {}
    knowledge_chunks = []
    turn_knowledge = []
    active_learning_probe = None
    context_checkpoint = {}
    time_context = {}
    memory_entities = []
    cassette_mode = ""

    def __getattr__(self, _name):
        return ""


def _bare_tutor():
    """建一个只够跑 _build_system_prompt 的 TutorAgent，旁路无关渲染。"""
    tutor = TutorAgent.__new__(TutorAgent)
    tutor._entity_memory_lines = lambda _ctx: []
    tutor._persona_scoped_facts = lambda _ctx: []
    tutor._checkpoint_for_prompt = lambda _ctx: {}
    tutor._persona_scoped_entities = lambda _ctx: []
    return tutor


class CommunicationStylePromptTests(unittest.TestCase):
    def test_known_label_enters_authoritative_prompt(self):
        ctx = Ctx()
        ctx.profile = {"communication_style": "analytical"}
        prompt = _bare_tutor()._build_system_prompt(ctx)
        self.assertIn("[User Communication Style]", prompt)
        self.assertIn("讲事情带因果和条件", prompt)
        self.assertIn("给结构化解释", prompt)

    def test_two_labels_render_together(self):
        ctx = Ctx()
        ctx.profile = {"communication_style": "direct+hesitant"}
        prompt = _bare_tutor()._build_system_prompt(ctx)
        self.assertIn("说话简短直接", prompt)
        self.assertIn("常带犹豫词", prompt)

    def test_layering_guard_is_present(self):
        """只调表达、不改观点——分层护栏必须在同一段里（框架档 0919-01 §4.1）。"""
        ctx = Ctx()
        ctx.profile = {"communication_style": "direct"}
        prompt = _bare_tutor()._build_system_prompt(ctx)
        self.assertIn("不得据此推断他的性格、观点或立场", prompt)
        self.assertIn("不改变你自己的观点、边界与教学原则", prompt)

    def test_no_label_adds_nothing(self):
        ctx = Ctx()
        ctx.profile = {}
        self.assertNotIn("[User Communication Style]", _bare_tutor()._build_system_prompt(ctx))

    def test_unknown_label_adds_nothing(self):
        """脏数据不该被原样透进 prompt。"""
        ctx = Ctx()
        ctx.profile = {"communication_style": "mystery+direct"}
        prompt = _bare_tutor()._build_system_prompt(ctx)
        self.assertIn("[User Communication Style]", prompt)
        self.assertIn("说话简短直接", prompt)
        self.assertNotIn("mystery", prompt)

    def test_all_unknown_labels_adds_nothing(self):
        ctx = Ctx()
        ctx.profile = {"communication_style": "mystery+nonsense"}
        self.assertNotIn("[User Communication Style]", _bare_tutor()._build_system_prompt(ctx))


class L1RoundTripTests(unittest.IsolatedAsyncioTestCase):
    """L1 整档覆盖写的保留语义——否则刚算的标签会被下一次画像写入抹掉。"""

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.l1 = L1FileProfile(base_path=self._tmp.name)

    async def asyncTearDown(self):
        self._tmp.cleanup()

    async def test_label_survives_a_write_that_omits_it(self):
        await self.l1.update_profile("u1", {
            "cefr": "B1",
            "communication_style": "analytical",
        })
        self.assertEqual(
            (await self.l1.get_frozen_snapshot("u1"))["communication_style"], "analytical"
        )

        # main.py / session_manager 的画像写入不带这个字段
        await self.l1.update_profile("u1", {"cefr": "B2"})

        snapshot = await self.l1.get_frozen_snapshot("u1")
        self.assertEqual(snapshot["communication_style"], "analytical")
        self.assertEqual(snapshot["cefr"], "B2")

    async def test_explicit_empty_clears_the_label(self):
        """保留语义不能变成只增不减：显式传空串要能清掉。"""
        await self.l1.update_profile("u1", {"communication_style": "direct"})
        await self.l1.update_profile("u1", {"communication_style": ""})
        self.assertEqual((await self.l1.get_frozen_snapshot("u1"))["communication_style"], "")

    async def test_missing_label_reads_as_empty(self):
        await self.l1.update_profile("u1", {"cefr": "A2"})
        self.assertEqual((await self.l1.get_frozen_snapshot("u1"))["communication_style"], "")


class CommunicationStyleControlPromptTests(unittest.TestCase):
    """§4.6 的微调层必须与底层人格**同在回复路径**。

    回复文本由 `_build_control_prompt` 生成（两处调用点都在"正在生成这一轮回应"里）。
    原先 §4.6 只接在会话级 `_build_system_prompt`，会把用户说的
    "底层人格 + 底层之上的用户适应微调 = 一个具象的 AI 人物"拆成两条链路。
    """

    def _branches(self, label: str = "analytical") -> list:
        out = []
        original = tutor_agent._PROMPT_CACHE_ORDER
        prepared_base = stable_prepared("engage_general_topic")
        for authoritative in (True, False):
            for cache in (1, 0):
                prepared = {
                    **prepared_base,
                    "constraints": {**prepared_base["constraints"]},
                }
                if authoritative:
                    prepared["conversation_director"] = {
                        "_meta": {"source": "director_model"},
                        "response_need": "listen_and_continue",
                    }
                else:
                    prepared["conversation_director"] = {
                        "response_need": "listen_and_continue"}
                _agent, ctx = control_ctx(4)
                ctx.profile = {"communication_style": label} if label else {}
                try:
                    tutor_agent._PROMPT_CACHE_ORDER = cache
                    out.append((
                        "authoritative=%s/cache=%s" % (authoritative, cache),
                        _agent._build_control_prompt(ctx, prepared, ""),
                    ))
                finally:
                    tutor_agent._PROMPT_CACHE_ORDER = original
        return out

    def test_reaches_all_four_branches_of_the_reply_path(self):
        for label, prompt in self._branches():
            with self.subTest(branch=label):
                self.assertIn("[User Communication Style]", prompt)

    def test_sits_right_after_the_base_persona(self):
        """顺序即语义：底层人格在前、用户适应微调在后，两层相邻。"""
        for label, prompt in self._branches():
            with self.subTest(branch=label):
                self.assertLess(
                    prompt.index("[Disposition]"),
                    prompt.index("[User Communication Style]"),
                )

    def test_no_label_means_zero_increment(self):
        for label, prompt in self._branches(""):
            with self.subTest(branch=label):
                self.assertNotIn("[User Communication Style]", prompt)


if __name__ == "__main__":
    unittest.main()
