# -*- coding: utf-8 -*-
"""天气工具单测（2026-10-22，plans/1022-01 点3）。

只测确定性部分：城市提取、问句判定、证据格式化。真实网络查询在探针里跑，
单测不依赖外网。
"""

from __future__ import annotations

import unittest

from agent.weather_tool import (
    describe_code,
    extract_city,
    format_evidence,
    wants_weather,
)


class WeatherToolTest(unittest.TestCase):
    def test_extract_city_chinese_and_english(self) -> None:
        self.assertEqual(extract_city("北京今天天气怎么样"), "北京")
        self.assertEqual(extract_city("上海明天会下雨吗"), "上海")
        self.assertEqual(extract_city("Beijing weather today"), "Beijing")
        self.assertIsNone(extract_city("今天天气怎么样"))
        self.assertIsNone(extract_city("我这边今天冷吗"))

    def test_wants_weather_only_questions(self) -> None:
        self.assertTrue(wants_weather("今天天气怎么样"))
        self.assertTrue(wants_weather("明天会下雨吗"))
        self.assertTrue(wants_weather("北京今天天气怎么样"))
        self.assertFalse(wants_weather("今天天气不错"))
        self.assertFalse(wants_weather("今天上班还行"))

    def test_describe_code(self) -> None:
        self.assertEqual(describe_code(3), "阴")
        self.assertEqual(describe_code(0), "晴")
        self.assertEqual(describe_code(999), "未知天气")

    def test_format_evidence_ok(self) -> None:
        text = format_evidence({
            "ok": True, "city": "北京", "condition": "阴", "current_c": 13.8,
            "observed_at": "2026-10-22T17:00",
            "today": {"min_c": 4.0, "max_c": 15.5, "precip_prob": 0},
            "tomorrow": {"min_c": 6.6, "max_c": 15.9, "precip_prob": 2},
        })
        self.assertIn("北京", text)
        self.assertIn("13.8", text)
        self.assertIn("15.5", text)
        self.assertIn("Open-Meteo", text)

    def test_format_evidence_needs_location(self) -> None:
        text = format_evidence({"ok": False, "needs_location": True})
        self.assertIn("哪个城市", text)

    def test_format_evidence_failure(self) -> None:
        text = format_evidence({"ok": False, "reason": "Timeout"})
        self.assertIn("不要编", text)


if __name__ == "__main__":
    unittest.main()
