# -*- coding: utf-8 -*-
"""练习盘开场的**直出 + 兜底**回归（plans/0923-06 §4.1.3 / §4.1.4 / §5 第 3–4 条）。

练习盘开场自 0923-06 起**不走开场 LLM**：话题卡与任务都是人工审定好的，直接输出。
本文件用打桩 l2 覆盖：

  - 直出：`should_open=True, type='expression'`，文本就是库内某张卡的审定文本；
  - 确定性：同一 `conv_id` 反复请求是同一 `topic_id` / URL / 任务；
  - 排除：最近 5 盘用过的 `topic_id` 不重复；
  - 兜底：原 LLM/DB 异常、旧后端缺方法面 → 仍回到**有出处保底卡**（不是无源旧模板）；
  - 来源只读接口：可见 / 隐藏不可见 / 无记录（不补造"当时已看到"的出处）。

不联网、不连库、不调模型。

    cd chanpinzhujian/api/test_runner && PYTHONPATH=<api> python -m unittest test_expression_opening_topic -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from engine import expression_topics as et  # noqa: E402

USER = "zz-selftest-expr-opening"


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeL2:
    """只实现开场与来源查询会用到的方法面（含练习盘分配三方法）。"""

    def __init__(self, rows=None, hidden=(), fail=False):
        self.rows = {k: dict(v) for k, v in (rows or {}).items()}
        self.hidden = set(hidden)
        self.fail = fail

    # ── 开场通用 ──
    async def get_relationship_memory(self, user_id):
        return {}

    async def get_material_by_conv(self, user_id, conv_id):
        return {}

    async def get_last_archive_time(self, user_id):
        return {"created_at": (datetime.now(timezone.utc)
                               - timedelta(days=1)).isoformat()}

    async def list_recent_openings(self, user_id, limit=5):
        return []

    async def last_conv_message_at(self, user_id, conv_id):
        return ""

    # ── 后端就绪判据用（`_material_backend_ready`）──
    async def add_material(self, *a, **k):
        return {}

    async def save_conversation(self, *a, **k):
        return True

    # ── 练习盘分配 ──
    async def get_expression_topic_assignment(self, user_id, conv_id, include_hidden=False):
        if self.fail:
            raise RuntimeError("db down")
        if conv_id in self.hidden and not include_hidden:
            return {}
        return dict(self.rows.get(conv_id) or {})

    async def list_recent_expression_topic_assignments(self, user_id, limit=5,
                                                       include_hidden=True):
        if self.fail:
            raise RuntimeError("db down")
        return [dict(r) for r in list(self.rows.values())[:limit]]

    async def put_expression_topic_assignment(self, user_id, conv_id, assignment):
        if self.fail:
            raise RuntimeError("db down")
        self.rows.setdefault(conv_id, dict(assignment))
        return {"status": "assigned", "conv_id": conv_id, "inserted": True}


class LegacyL2(FakeL2):
    """旧后端：**没有**练习盘分配方法面（模拟未升级的 l2）。"""

    async def get_expression_topic_assignment(self, *a, **k):
        raise AttributeError("no such method")

    async def list_recent_expression_topic_assignments(self, *a, **k):
        raise AttributeError("no such method")

    async def put_expression_topic_assignment(self, *a, **k):
        raise AttributeError("no such method")


def _open(l2, conv_id="expr-selftest_A", scene="expression"):
    # `get_opening_line` 开头有总开关 CHAT_OPENING_ENABLED（2026-10-18 用户拍板
    # 「入口不说话」），config 默认 false。本文件测的是**开关打开后**的开场卡
    # 直出/兜底/排除逻辑，所以必须自己声明这个前置——不能靠改产品默认值。
    import config as _config
    with unittest.mock.patch.object(main, "l2", l2), \
         unittest.mock.patch.object(_config, "CHAT_OPENING_ENABLED", True):
        return _run(main.get_opening_line(
            user_id=USER, conv_id=conv_id, scene=scene,
            target_lang="en", current_user_id=USER))


class DirectOpeningTests(unittest.TestCase):
    def test_opening_is_direct_and_sourced(self) -> None:
        out = _open(FakeL2())
        self.assertTrue(out["should_open"])
        self.assertEqual("expression", out["type"])
        self.assertEqual("expression_topic", out["source"])
        topic = out.get("topic") or {}
        self.assertTrue(topic.get("has_source"), "开场没带来源元信息")
        self.assertTrue(str(topic.get("source_url")).startswith("https://"))
        # 文本必须能对上库里某张卡的审定文本（直出、不是现编）
        allowed = {et.opening_text(c) for c in et.TOPICS}
        self.assertIn(out["text"], allowed)

    def test_tts_text_never_contains_the_url(self) -> None:
        out = _open(FakeL2())
        self.assertNotIn("http://", out["text"])
        self.assertNotIn("https://", out["text"])
        # 但元数据里必须有（前端要显示"Source · 标题"）
        self.assertIn("https://", (out.get("topic") or {}).get("source_url", ""))

    def test_same_conv_is_the_same_topic(self) -> None:
        l2 = FakeL2()
        a = _open(l2, conv_id="expr-abc_A")
        b = _open(l2, conv_id="expr-abc_A")
        self.assertEqual(a["topic"]["topic_id"], b["topic"]["topic_id"])
        self.assertEqual(a["text"], b["text"])

    def test_recent_topics_are_excluded(self) -> None:
        used = [c["id"] for c in et.TOPICS[:6]]
        rows = {}
        for i, tid in enumerate(used):
            rows["expr-old-%d_A" % i] = {
                "topic_id": tid, "source_title": "x", "source_url": "https://x/",
                "source_date": "", "prompt_text": "x", "assigned_at": "2026-09-2%d" % i,
            }
        out = _open(FakeL2(rows=rows), conv_id="expr-new_A")
        self.assertNotIn(out["topic"]["topic_id"], used,
                         "抽到了最近用过的卡（没按最近 5 盘排除）")


class FallbackTests(unittest.TestCase):
    def test_db_failure_still_returns_sourced_fallback_card(self) -> None:
        out = _open(FakeL2(fail=True))
        self.assertTrue(out["should_open"])
        self.assertEqual("expression", out["type"])
        self.assertEqual(et.opening_text(et.fallback_topic()), out["text"])
        self.assertTrue((out.get("topic") or {}).get("has_source"))

    def test_legacy_backend_still_returns_sourced_fallback_card(self) -> None:
        out = _open(LegacyL2())
        self.assertTrue(out["should_open"])
        self.assertEqual("expression", out["type"])
        self.assertTrue((out.get("topic") or {}).get("has_source"))

    def test_fallback_helper_returns_library_card_not_old_template(self) -> None:
        text = main._expression_opening_fallback(1)
        self.assertEqual(et.opening_text(et.fallback_topic()), text)
        self.assertNotIn("Let's practice. Tell me about one thing", text)
        card = et.fallback_topic()
        self.assertTrue(et.has_valid_source(card), "保底卡没有真实来源")
        self.assertIn("https://", card["source_url"])

    def test_fallback_is_not_the_generic_pool(self) -> None:
        out = _open(FakeL2(fail=True))
        # 通用池的开场是"今天过得怎么样"式闲聊问句；这里必须不是
        self.assertNotEqual("template", out["source"])
        self.assertIn(et.OWN_TOPIC_PHRASE, out["text"])


class SourceRouteTests(unittest.TestCase):
    def _call(self, l2, conv_id="expr-selftest_A"):
        with unittest.mock.patch.object(main, "l2", l2):
            return _run(main.expression_topic_source(
                USER, conv_id=conv_id, current_user_id=USER))

    def _row(self, topic_id="classic-turing-test"):
        card = et.get_topic(topic_id)
        return et.assignment_snapshot(card, user_id=USER, conv_id="expr-selftest_A",
                                      assigned_at="2026-09-22T10:00:00+08:00")

    def test_visible_tape_returns_source(self) -> None:
        l2 = FakeL2(rows={"expr-selftest_A": self._row()})
        out = self._call(l2)
        self.assertTrue(out["has_source"])
        self.assertEqual("classic-turing-test", out["topic_id"])
        self.assertTrue(out["source_url"].startswith("https://"))
        self.assertEqual("", out["reason"])

    def test_hidden_tape_is_invisible(self) -> None:
        l2 = FakeL2(rows={"expr-selftest_A": self._row()},
                    hidden={"expr-selftest_A"})
        out = self._call(l2)
        self.assertFalse(out["has_source"])
        self.assertEqual("hidden", out["reason"])

    def test_tape_without_assignment_does_not_invent_a_source(self) -> None:
        out = self._call(FakeL2())
        self.assertFalse(out["has_source"])
        self.assertEqual("no_record", out["reason"])
        self.assertIn("此前话题无来源记录", out["note"])

    def test_missing_conv_id_is_rejected(self) -> None:
        out = self._call(FakeL2(), conv_id="")
        self.assertFalse(out["has_source"])
        self.assertEqual("no_conv_id", out["reason"])

    def test_db_failure_reports_unavailable_not_no_record(self) -> None:
        out = self._call(FakeL2(fail=True))
        self.assertFalse(out["has_source"])
        self.assertEqual("read_failed", out["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
