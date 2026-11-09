# -*- coding: utf-8 -*-
"""`状态变化问句 ⇒ 允许一次有界检索` 这条路由分支的回归（plans/1017-02 §5 / §8.9）。

背景（2026-10-23 冻结集实测 `bugreports/data/ab_off_1023.json` / `ab_on_1023.json`）：
30 条回忆问法里只有 12 条进回忆（route 0.400）；被挡的 18 条大多是
「我那个房租的事后来怎么样了」这类**自然追问**——不含「上次/之前/还记得」，
于是走到 `global_only`，检索再好也轮不到出场。

`STATE_CHANGE_QUESTION` 就是为这批问法准备的**预算开关**（台架
`bugreports/eval_definite_reference_gate_1017.py`：召回 15/16、误报 1/50），
由 `HISTORY_RECALL_STATE_CHANGE_TRIGGER_ENABLED` 控制，默认 false。

守四件事：
1. 开关关 ⇒ 自然追问仍是 `global_only`（与改前逐字同义，一键回滚）。
2. 开关开 ⇒ 自然追问进 `episodic_recall`，且 reason 标明是预算闸而不是判据。
3. 已有显式标记、待确认、身份问句这些**更高优先级分支不被抢**。
4. 纯寒暄不因为开关打开就被拖进回忆。

跑法：cd chanpinzhujian/api && PYTHONPATH=. python -m unittest agent.test_state_change_recall_route
"""

import os
import sys
import unittest

sys.path.insert(0, ".")

os.environ.setdefault("DATABASE_URL", "postgresql://llm_user:***@localhost:5432/llm_os")
os.environ.setdefault("MEMORY_BACKEND", "local")

from agent.memory_orchestrator import MemoryOrchestrator  # noqa: E402

VAGUE_FOLLOWUPS = [
    "我那个房租的事后来怎么样了？",
    "年糕后来好点了吗？",
    "那个麻烦事最后怎么样了？",
    "How's your leg doing?",
    "Any better today?",
]


def route(text: str, on: bool, **kw):
    return MemoryOrchestrator.route_turn(
        text,
        fresh_context=kw.pop("fresh_context", False),
        turn_count=kw.pop("turn_count", 3),
        state_change_recall=on,
        **kw,
    )


class SwitchOffKeepsOldBehaviour(unittest.TestCase):
    def test_vague_followups_stay_global_only(self):
        for text in VAGUE_FOLLOWUPS:
            with self.subTest(text=text):
                self.assertEqual(route(text, on=False).mode, "global_only")


class SwitchOnRoutesVagueFollowups(unittest.TestCase):
    def test_vague_followups_route_to_episodic(self):
        for text in VAGUE_FOLLOWUPS:
            with self.subTest(text=text):
                got = route(text, on=True)
                self.assertEqual(got.mode, "episodic_recall")
                self.assertIn("state-change", got.reason)
                self.assertEqual(got.precision, "semantic")

    def test_query_is_stripped_for_retrieval(self):
        got = route("我那个房租的事后来怎么样了？", on=True)
        self.assertTrue(got.query)
        self.assertNotEqual(got.query, "我那个房租的事后来怎么样了？")


class HigherPriorityBranchesStillWin(unittest.TestCase):
    def test_explicit_recall_marker_wins(self):
        got = route("你还记得我房租那件事吗？", on=True)
        self.assertEqual(got.mode, "episodic_recall")
        self.assertIn("explicitly referenced", got.reason)

    def test_bare_remember_question_stays_global(self):
        # VAGUE_RECALL_REFERENCE.fullmatch 命中 ⇒ 不占预算（兜底句不该触发检索）
        self.assertEqual(route("你还记得吗？", on=True).mode, "global_only")

    def test_awaiting_confirmation_takes_precedence(self):
        got = route("后来怎么样了", on=True, awaiting_confirmation=True)
        self.assertEqual(got.mode, "pending_confirmation")

    def test_identity_question_unchanged(self):
        self.assertEqual(route("你叫什么名字？", on=True).mode, "global_only")


class SmallTalkIsNotDraggedIn(unittest.TestCase):
    def test_plain_smalltalk_stays_global(self):
        for text in ("今天天气不错", "Good morning! How are you today?", "早上好呀"):
            with self.subTest(text=text):
                self.assertEqual(route(text, on=True).mode, "global_only")

    def test_known_false_positive_is_documented(self):
        # config.py 注释里记的**唯一已知误报**：「你还好吗」与正例「你还打算辞职吗」
        # 词形同构（`还.{0,8}吗`）。这里把它钉成行为契约，不当作正确性——
        # 真要修就得改正则，届时本用例会跟着变，避免"悄悄改了"。
        self.assertEqual(route("你还好吗？", on=True).mode, "episodic_recall")
        self.assertEqual(route("你还好吗？", on=False).mode, "global_only")


if __name__ == "__main__":
    unittest.main()
