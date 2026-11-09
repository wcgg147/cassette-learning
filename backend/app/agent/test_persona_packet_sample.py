# -*- coding: utf-8 -*-
"""闲聊盘每轮带的"她自己的料"条数可调（2026-10-22，plans/1022-01 点5）。

默认仍是 6 条；`max_facts` 只在开关打开且闲聊盘时传入。
实测约束：`MAX_PACKET_CHARS=1600` 仍会截断长事实——条数上限放开不等于真能带满。
本测试用短事实验证参数本身生效。
"""

from __future__ import annotations

import unittest

from memory.companion_persona import MAX_PACKET_FACTS, build_persona_packet


def _facts(count: int) -> list[dict]:
    return [
        {
            "fact_id": f"f{index}",
            "kind": "preference",
            "status": "active",
            "value_text": f"short value {index}",
            "facet_key": f"topic-{index}",
            "recorded_at": "2026-01-01T00:00:00Z",
            "observed_at": "2026-01-01T00:00:00Z",
        }
        for index in range(count)
    ]


class PersonaPacketSampleTest(unittest.TestCase):
    def test_default_is_still_six(self) -> None:
        packet = build_persona_packet(_facts(10), "今天天气怎么样")
        self.assertEqual(len(packet["lines"]), MAX_PACKET_FACTS)

    def test_max_facts_can_lower_the_cap(self) -> None:
        packet = build_persona_packet(_facts(10), "今天天气怎么样", max_facts=3)
        self.assertEqual(len(packet["lines"]), 3)

    def test_max_facts_can_raise_the_cap(self) -> None:
        packet = build_persona_packet(_facts(10), "今天天气怎么样", max_facts=8)
        self.assertEqual(len(packet["lines"]), 8)


    def test_suffix_is_compact_and_keeps_only_date(self) -> None:
        """2026-10-22：后缀不再带 fact_id/source（全仓无人解析，占掉 47.8% 预算）。"""
        packet = build_persona_packet(_facts(2), "今天天气怎么样")
        joined = " ".join(packet["lines"])
        self.assertNotIn("fact_id=", joined)
        self.assertNotIn("source=", joined)
        self.assertIn("[2026-01-01]", joined)

    def test_overflowing_fact_is_skipped_not_truncated(self) -> None:
        """预算不够时装不下就跳过整行：绝不把半条自述喂进 prompt。

        单条正文被 _clean 截到 360 字符，所以用"多条接近满额"来制造溢出：
        4 条 × ~395 字符 ≈ 1580，第 5 条（~395）超出 1600 ⇒ 应被跳过而不是切半句。
        """
        facts = _facts(5)
        for fact in facts:
            fact["value_text"] = "y" * 360
        packet = build_persona_packet(facts, "今天天气怎么样", max_facts=5)
        self.assertEqual(len(packet["lines"]), 4)
        self.assertLessEqual(len(packet["text"]), 1600)
        for line in packet["lines"]:
            self.assertTrue(line.endswith("]"), line[-20:])

    def test_ten_facts_fit_in_the_same_budget(self) -> None:
        """压完后缀后，10 条真实长度（~130 字符/条）应能塞进 1600。"""
        facts = _facts(10)
        for fact in facts:
            fact["value_text"] = "y" * 120
        packet = build_persona_packet(facts, "今天天气怎么样", max_facts=10)
        self.assertEqual(len(packet["lines"]), 10)
        self.assertLessEqual(len(packet["text"]), 1600)


if __name__ == "__main__":
    unittest.main()
