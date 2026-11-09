# -*- coding: utf-8 -*-
"""练习盘「盘内趋势」层单测（plans/0919-02 §4.7）。

只测纯函数：读数提取、相邻折叠、趋势判定、有界措辞。
零 LLM、零 IO、零真实会话 —— 与 `test_expression_argument.py` 同一纪律。
"""
import io
import os
import unittest

from engine.expression_argument import DIMENSIONS
from engine import expression_progress as ep


def _event(turn_index, reading, created_at=""):
    return {
        "session_id": "s1", "turn_id": "t%d" % turn_index,
        "created_at": created_at,
        "metadata": {"turn_index": turn_index, ep.READING_KEY: reading},
    }


class ReadingFromFocusTests(unittest.TestCase):
    def test_only_ok_and_weak_are_collected(self):
        focus = {"dims": {"claim": "ok", "evidence": "weak", "warrant": "na",
                          "qualifier": "", "rebuttal": None}}
        self.assertEqual(ep.reading_from_focus(focus),
                         {"claim": "ok", "evidence": "weak"})

    def test_unknown_value_is_dropped_not_guessed(self):
        focus = {"dims": {dim: "strong" for dim in DIMENSIONS}}
        self.assertEqual(ep.reading_from_focus(focus), {})

    def test_junk_inputs_return_empty(self):
        for junk in (None, [], "x", 3, {}, {"dims": None}, {"dims": []}):
            self.assertEqual(ep.reading_from_focus(junk), {}, repr(junk))

    def test_all_na_returns_empty_so_caller_does_not_book_a_blank_row(self):
        focus = {"dims": {dim: "na" for dim in DIMENSIONS}}
        self.assertEqual(ep.reading_from_focus(focus), {})


class ReadingsFromEventsTests(unittest.TestCase):
    def test_reads_dict_metadata(self):
        events = [_event(1, {"claim": "ok"})]
        self.assertEqual(ep.readings_from_turn_events(events), [{"claim": "ok"}])

    def test_reads_json_string_metadata(self):
        events = [{"metadata": '{"turn_index": 1, "expr_arg": {"claim": "weak"}}'}]
        self.assertEqual(ep.readings_from_turn_events(events), [{"claim": "weak"}])

    def test_sorted_by_turn_index_regardless_of_input_order(self):
        # 三条读数两两不同签名，避免被折叠规则干扰 —— 本用例只验排序。
        events = [_event(3, {"claim": "ok"}), _event(1, {"claim": "ok"}),
                  _event(2, {"claim": "weak"})]
        self.assertEqual(
            ep.readings_from_turn_events(events),
            [{"claim": "ok"}, {"claim": "weak"}, {"claim": "ok"}],
        )

    def test_adjacent_identical_signature_collapses(self):
        # 同一次讲述被重放（缓存 / 记忆恢复路径）⇒ 只算一票。
        # 已知取舍：两次**内容不同**但五维签名恰好完全相同的讲述也会被并成一条
        # ⇒ 宁可**少算**、不算多（少算只是趋势晚一点出现，多算会让模型说错话）。
        events = [_event(1, {"claim": "ok"}), _event(2, {"claim": "ok"}),
                  _event(3, {"claim": "ok"})]
        self.assertEqual(ep.readings_from_turn_events(events), [{"claim": "ok"}])

    def test_non_adjacent_repeat_is_kept(self):
        events = [_event(1, {"claim": "ok"}), _event(2, {"claim": "weak"}),
                  _event(3, {"claim": "ok"})]
        self.assertEqual(len(ep.readings_from_turn_events(events)), 3)

    def test_events_without_reading_are_skipped(self):
        events = [{"metadata": {"turn_index": 1}}, {"metadata": None}, {}, None,
                  _event(2, {"claim": "ok"})]
        self.assertEqual(ep.readings_from_turn_events(events), [{"claim": "ok"}])

    def test_empty_reading_is_skipped(self):
        events = [_event(1, {"claim": "na"}), _event(2, {"claim": "ok"})]
        self.assertEqual(ep.readings_from_turn_events(events), [{"claim": "ok"}])

    def test_none_input_is_safe(self):
        self.assertEqual(ep.readings_from_turn_events(None), [])


class TrendsTests(unittest.TestCase):
    def _runs(self, value, times, dim="claim"):
        return [{dim: value} for _ in range(times)]

    def test_two_in_a_row_is_not_enough(self):
        self.assertEqual(ep.trends_from_readings(self._runs("ok", 2)), {})

    def test_three_in_a_row_is_a_trend(self):
        trends = ep.trends_from_readings(self._runs("ok", 3))
        self.assertEqual(trends, {"claim": {"streak": 3, "direction": "ok"}})

    def test_na_in_the_middle_breaks_the_chain(self):
        readings = [{"claim": "ok"}, {"claim": "ok"}, {}, {"claim": "ok"}]
        self.assertEqual(ep.trends_from_readings(readings), {})

    def test_na_breaks_even_when_the_dim_is_absent_from_that_row(self):
        readings = [{"claim": "ok"}, {"claim": "ok"}, {"evidence": "ok"},
                    {"claim": "ok"}]
        self.assertEqual(ep.trends_from_readings(readings), {})

    def test_direction_flip_breaks_the_chain(self):
        readings = [{"claim": "weak"}] + self._runs("ok", 3)
        self.assertEqual(ep.trends_from_readings(readings),
                         {"claim": {"streak": 3, "direction": "ok"}})

    def test_counts_backwards_from_the_newest(self):
        readings = self._runs("weak", 5) + self._runs("ok", 3)
        self.assertEqual(ep.trends_from_readings(readings),
                         {"claim": {"streak": 3, "direction": "ok"}})

    def test_each_dimension_is_counted_independently(self):
        readings = [{"claim": "ok", "evidence": "weak"}] * 3
        trends = ep.trends_from_readings(readings)
        self.assertEqual(trends["claim"], {"streak": 3, "direction": "ok"})
        self.assertEqual(trends["evidence"], {"streak": 3, "direction": "weak"})

    def test_junk_input_is_safe(self):
        for junk in (None, [], ["x"], [None]):
            self.assertEqual(ep.trends_from_readings(junk), {}, repr(junk))


class ProgressLineTests(unittest.TestCase):
    def test_no_trend_returns_empty_string(self):
        self.assertEqual(ep.progress_line({}), "")
        self.assertEqual(ep.progress_line(None), "")
        self.assertEqual(ep.progress_line("x"), "")

    def test_below_min_streak_returns_empty_string(self):
        self.assertEqual(ep.progress_line({"claim": {"streak": 2, "direction": "ok"}}), "")

    def test_ok_and_weak_read_differently(self):
        ok = ep.progress_line({"claim": {"streak": 3, "direction": "ok"}})
        weak = ep.progress_line({"claim": {"streak": 3, "direction": "weak"}})
        self.assertIn("先放一放", ok)
        self.assertIn("优先盯它", weak)

    def test_reports_at_most_max_dims(self):
        trends = {dim: {"streak": 3, "direction": "weak"} for dim in DIMENSIONS}
        line = ep.progress_line(trends)
        self.assertEqual(line.count("连着"), ep.MAX_REPORTED_DIMS)

    def test_longer_streak_is_reported_first(self):
        trends = {"claim": {"streak": 3, "direction": "weak"},
                  "evidence": {"streak": 5, "direction": "weak"}}
        line = ep.progress_line(trends)
        self.assertLess(line.index("证据"), line.index("主张"))

    def test_does_not_leak_academic_dim_keys(self):
        trends = {dim: {"streak": 3, "direction": "weak"} for dim in DIMENSIONS}
        line = ep.progress_line(trends)
        for key in ("claim", "warrant", "qualifier", "rebuttal", "evidence"):
            self.assertNotIn(key, line)

    def test_malformed_trend_entries_are_ignored(self):
        trends = {"claim": {"streak": "x", "direction": "ok"},
                  "evidence": None,
                  "warrant": {"streak": 4, "direction": "sideways"}}
        self.assertEqual(ep.progress_line(trends), "")


class ModuleDisciplineTests(unittest.TestCase):
    def test_no_llm_or_io_dependency(self):
        path = os.path.join(os.path.dirname(ep.__file__), "expression_progress.py")
        with io.open(path, encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in ("import requests", "import httpx", "llm_client",
                          "open(", "asyncio"):
            self.assertNotIn(forbidden, source, forbidden)

    def test_min_streak_is_conservative(self):
        # 轮内读数比会话末那次糙，所以阈值不许低于 3（与支架档位的"连续两次"不同轴）
        self.assertGreaterEqual(ep.MIN_STREAK, 3)

    def test_reading_key_matches_the_metering_whitelist(self):
        from memory.metering_contract import METADATA_FIELDS
        self.assertIn(ep.READING_KEY, METADATA_FIELDS)


if __name__ == "__main__":
    unittest.main()
