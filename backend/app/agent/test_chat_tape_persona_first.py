# -*- coding: utf-8 -*-
"""闲聊盘「人格先行」（2026-10-24）回归。

守五件事：
1. 开关关：闲聊盘 prompt 与开关开之前逐字一致（时间戳屏蔽后比较）。
2. 开关开：服务/教学/控制段与机器行、配额句数从闲聊盘 prompt 去掉；人格、
   诚实、防误听、元叙述红线段全部保留。
3. 练习盘（expression）不受开关影响。
4. cut_shelved_spans 按段头区间删，不误伤相邻人格段（相邻字面量会合并成同一元素）。
5. 开关开时输出约束/合同的剥离只删指定片段，其余文本原样。

跑法：cd chanpinzhujian/api && PYTHONPATH=. py -3.12 -m unittest agent.test_chat_tape_persona_first
"""

import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, ".")

os.environ.setdefault("DATABASE_URL", "postgresql://llm_user:***@localhost:5432/llm_os")
os.environ.setdefault("MEMORY_BACKEND", "local")

import config as app_config  # noqa: E402
import agent.tutor_agent as tutor_agent  # noqa: E402
from agent import chat_tape_policy as policy  # noqa: E402
from agent.test_expression_contract import control_ctx, stable_prepared  # noqa: E402

TRANSCRIPT = "今天加班到九点，累死了。"


def _chat_ctx():
    agent, ctx = control_ctx(0)
    ctx.scene = {}
    ctx.scene_context_set = False
    ctx.scenario_role = ""
    ctx.cassette_mode = "chat"
    ctx.persona_state = {"mode": "companion"}
    return agent, ctx


def _mask_time(text: str) -> str:
    """当前北京时间会随分钟变化，比较前屏蔽 HH:MM。"""
    return re.sub(r"\d{1,2}:\d{2}", "HH:MM", text)


# 人格/诚实/红线段：开关开时必须仍在。
KEEP_HEADS = (
    "[Becoming]",
    "[Disposition]",
    "[Stance - content] 对",
    "[Non-Judgment - language] 对",
    "[No Meta]",
    "[Reality]",
    "[Semantic Fidelity]",
    "[Companion social foundation]",
    "[Emotion]",
    "[Session Memory]",
)

# 服务/教学/控制段与机器行：开关开时必须不在。
DROP_TOKENS = (
    "[Order]",
    "[Frame]",
    "[Pragmatics]",
    "[Teaching] 一轮",
    "[Reply Contract]",
    "[Conversation Rhythm]",
    "[Turn Rhythm]",
    "[Policy]",
    "[Learning Move]",
    "[Learning State]",
    "[Learner State]",
    "[Teaching Method",
    "[Constraints]",
    "[Latest Turn Contract",
    "USER_INTENT",
    "TERM:",
    "1到3个",
    "每个短句必须独立完整",
    "不总递问题",
)


class PersonaFirstOffTests(unittest.TestCase):
    """开关关时：闲聊盘 prompt 与未开关时一致。"""

    def test_off_flag_is_noop_for_filters(self):
        agent, ctx = _chat_ctx()
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", False):
            sec = ["[Order] a", "[Role] b", ""]
            self.assertIs(policy.filter_stable_sections(sec, ctx), sec)
            tail = ["[Policy] x", "[Constraints] y"]
            self.assertIs(policy.filter_dynamic_tail(tail, ctx), tail)
            self.assertFalse(policy.persona_first_keep(ctx))
            self.assertEqual(
                tutor_agent._persona_first_output_contract(
                    tutor_agent._OUTPUT_CONTRACT_CACHE, ctx),
                tutor_agent._OUTPUT_CONTRACT_CACHE)
            self.assertEqual(
                tutor_agent._persona_first_contract_text(ctx),
                tutor_agent._CHAT_TAPE_OUTPUT_CONTRACT)
            text = "[Order] x [Turn Rhythm] y [Stance - content] z"
            self.assertEqual(policy.cut_shelved_spans(text, ctx), text)

    def test_off_chat_prompt_keeps_service_sections(self):
        agent, ctx = _chat_ctx()
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", False):
            prompt = agent._build_control_prompt(ctx, stable_prepared("continue_roleplay_scene"), TRANSCRIPT)
        self.assertIn("[Order]", prompt)
        self.assertIn("[Turn Rhythm]", prompt)
        self.assertIn("USER_INTENT", prompt)


class PersonaFirstOnTests(unittest.TestCase):
    """开关开时：闲聊盘去掉服务/教学/控制，保留人格。"""

    @classmethod
    def setUpClass(cls):
        agent, ctx = _chat_ctx()
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", True):
            cls.on = agent._build_control_prompt(ctx, stable_prepared("continue_roleplay_scene"), TRANSCRIPT)
        agent, ctx = _chat_ctx()
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", False):
            cls.off = agent._build_control_prompt(ctx, stable_prepared("continue_roleplay_scene"), TRANSCRIPT)

    def test_keeps_persona_and_guard_sections(self):
        for head in KEEP_HEADS:
            with self.subTest(head=head):
                self.assertIn(head, self.on)

    def test_drops_service_teaching_and_machine_rows(self):
        for token in DROP_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token, self.on)

    def test_keeps_honesty_and_pacing_clauses_in_output_contract(self):
        for token in ("【直接问题优先】", "【时间溯源】", "【推进优先·反复述】",
                      "不要写标题、列表、内部分析或元说明", "【表达变体】"):
            with self.subTest(token=token):
                self.assertIn(token, self.on)

    def test_on_is_shorter_than_off(self):
        self.assertLess(len(self.on), len(self.off))

    def test_only_removals_no_new_text_leaks(self):
            """开关开只能删，不能凭空加字：ON 必须是 OFF 的有序子序列。"""
            off_norm = _mask_time(self.off)
            on_norm = _mask_time(self.on)
            i = j = 0
            while i < len(on_norm) and j < len(off_norm):
                if on_norm[i] == off_norm[j]:
                    i += 1
                j += 1
            self.assertEqual(i, len(on_norm),
                             "ON 出现了 OFF 里没有的字符（应只删不加）")


class PracticeTapeUnaffectedTests(unittest.TestCase):
    """练习盘 / 素材盘不受开关影响。"""

    def test_expression_tape_prompt_identical_with_flag_on_and_off(self):
        def build(flag):
            agent, ctx = control_ctx(0)
            ctx.scene = {}
            ctx.scene_context_set = False
            ctx.scenario_role = ""
            ctx.cassette_mode = "expression"
            ctx.persona_state = {"mode": "companion"}
            with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", flag):
                return agent._build_control_prompt(ctx, stable_prepared("continue_roleplay_scene"), TRANSCRIPT)

        self.assertEqual(_mask_time(build(True)), _mask_time(build(False)))

    def test_material_tape_not_chat(self):
        agent, ctx = _chat_ctx()
        ctx.material_context = {"material_id": "m1"}
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", True):
            self.assertFalse(policy.persona_first_keep(ctx))


class CutShelvedSpansTests(unittest.TestCase):
    """区间删除：不误伤相邻人格段。"""

    def test_turn_rhythm_cut_keeps_adjacent_stance(self):
        text = (
            "[Conversation Rhythm] 普通聊天。"
            "[Turn Rhythm] 发起权转移：闲聊盘里问句预算。"
            "[Stance - content] 对内容你有自己的立场。"
            "[Non-Judgment - language] 对语言不评判。"
        )
        ctx = SimpleNamespace(scene_context_set=False, material_context=None,
                              cassette_mode="chat", persona_state={"mode": "companion"})
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", True):
            out = policy.cut_shelved_spans(text, ctx)
        self.assertNotIn("[Turn Rhythm]", out)
        self.assertNotIn("[Conversation Rhythm]", out)
        self.assertIn("[Stance - content] 对内容你有自己的立场。", out)
        self.assertIn("[Non-Judgment - language] 对语言不评判。", out)

    def test_cut_is_idempotent(self):
        text = "[Order] a [Becoming] b [Turn Rhythm] c [Disposition] d"
        ctx = SimpleNamespace(scene_context_set=False, material_context=None,
                              cassette_mode="chat", persona_state={"mode": "companion"})
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", True):
            once = policy.cut_shelved_spans(text, ctx)
            twice = policy.cut_shelved_spans(once, ctx)
        self.assertEqual(once, twice)


class ContractTrimTests(unittest.TestCase):
    """配额子句：开关开删、关不删。"""

    def test_quota_clauses_removed_only_when_on(self):
        agent, ctx = _chat_ctx()
        contract = tutor_agent._CHAT_TAPE_OUTPUT_CONTRACT
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", False):
            self.assertEqual(tutor_agent._persona_first_contract_text(ctx), contract)
        with mock.patch.object(app_config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", True):
            out = tutor_agent._persona_first_contract_text(ctx)
        self.assertNotIn("1到2个短句", out)
        # ③「不总递问题」不是配额、是姿态：搁置后实测 13 轮 8 轮以问句收尾（访谈主持人），
        # 2026-10-26 起保留。只有 ⑤ 的数字配额继续搁置。
        self.assertIn("不总递问题", out)
        # 保留：不复述、不道歉、不总递问题、说具体
        for keep in ("① 不复述", "② 不道歉", "③ 不总递问题", "④ 说具体"):
            self.assertIn(keep, out)


if __name__ == "__main__":
    unittest.main()
