# -*- coding: utf-8 -*-
"""世界线的确定性部分（plans/1017-02 §8.79）。

这些用例来自真机/探针实测踩过的坑，别再退回去：
  · "那你还有别的爱好吗？" 漏判 ⇒ 没给材料 ⇒ 她现编"最近在琢磨做饭"
  · "你说我该怎么跟她说？" 误判 ⇒ 会把她自己的事塞进"给建议"的场景
"""
from __future__ import annotations

import unittest

from agent.life_line import asked_about_her, pick_items, render_line, score_item

HER_ASKED = [
    "你最近在忙什么呀？",
    "那你还有别的爱好吗？",
    "你平时都干嘛？",
    "你学过什么乐器吗？",
    "What have you been up to lately?",
    "你觉得这电影怎么样",
]
NOT_HER = [
    "你说我该怎么跟她说？",
    "今天加班到九点，累死了。",
    "我朋友下周从杭州来，还没想好带他吃什么。",
    "他不太能吃辣。",
]

ITEMS = [
    {"item_id": "a", "text": "在学用 Excel 做甘特图，卡在公式上"},
    {"item_id": "b", "text": "周三下雨，把窗台那盆多肉换了盆"},
    {"item_id": "c", "text": "在读《海边的卡夫卡》，读了三分之一"},
]


class AskedAboutHerTests(unittest.TestCase):
    def test_her_question_shapes(self):
        for text in HER_ASKED:
            with self.subTest(text=text):
                self.assertTrue(asked_about_her(text))

    def test_not_about_her(self):
        for text in NOT_HER:
            with self.subTest(text=text):
                self.assertFalse(asked_about_her(text))


class PickItemsTests(unittest.TestCase):
    """§8.79 改了契约：摆材料是代码的事，说不说是她的事。

    以前这里测的是"什么时候允许给"（他问才给、撞词才给、问过就不再给）。
    那套门已经被删掉——用代码替她决定"什么时候可以说自己的事"，
    等于把人格判断写成 if。现在只测边界：上限、去重、空池。
    """

    def test_always_offers_material(self):
        """他没问、也没撞词，材料照样摆出来。"""
        got = pick_items(ITEMS, "今天加班到九点，累死了。")
        self.assertTrue(got)
        self.assertLessEqual(len(got), 2)

    def test_skip_is_respected(self):
        """同一句话别在一个人面前说两遍——这是边界，不是判断。"""
        got = pick_items(ITEMS, "那你还有别的爱好吗？", skip_ids={"a", "b", "c"})
        self.assertEqual(got, [])

    def test_limit(self):
        got = pick_items(ITEMS, "随便聊聊", limit=2)
        self.assertLessEqual(len(got), 2)

    def test_empty_pool(self):
        self.assertEqual(pick_items([], "你好"), [])

    def test_score_is_bigram_overlap(self):
        self.assertGreater(score_item(ITEMS[0], "你那个甘特图做完了吗"), 0)


class AliasRecallTests(unittest.TestCase):
    """§8.92：换个说法也要撞得上——条目带写入时存的 aliases（近义抓手）。

    没有抓手时实测：`在学吉他` ← `最近迷上弹琴了` 二元组零交集、门永远不开；
    `在学吉他` ← `how is the guitar` 跨语言更是 0。
    """

    ITEM = {"item_id": "g", "text": "在学吉他，手指头疼",
            "aliases": "琴 弹琴 guitar instrument"}

    def test_paraphrase_hits_alias(self):
        self.assertGreater(score_item(self.ITEM, "最近迷上弹琴了"), 0)

    def test_cross_language_hits_alias(self):
        self.assertGreater(score_item(self.ITEM, "how is the guitar going"), 0)

    def test_unrelated_still_zero(self):
        self.assertEqual(score_item(self.ITEM, "明天要下雨"), 0)

    def test_pick_items_fires_on_alias_only(self):
        got = pick_items([self.ITEM], "最近迷上弹琴了")
        self.assertEqual([x["item_id"] for x in got], ["g"])

    def test_text_hit_not_diluted_by_alias(self):
        """原文撞得上时按原文算——别名只在撞不上时兜底。"""
        self.assertGreater(score_item(self.ITEM, "吉他练得手指疼"),
                           score_item(self.ITEM, "guitar"))

    def test_missing_alias_is_safe(self):
        """老条目没有 aliases 字段也不能炸。"""
        self.assertGreater(score_item(ITEMS[0], "你那个甘特图做完了吗"), 0)
        self.assertEqual(score_item({"item_id": "x", "text": "在学吉他"}, "迷上弹琴"), 0)


class RenderLineTests(unittest.TestCase):
    def test_line_is_fact_only(self):
        """渲染出来只能是陈述句；出现"应该/不要/必须"就是规则又回来了。"""
        line = render_line(ITEMS[:2], asked=True)
        self.assertIn("你最近的事", line)
        self.assertIn(ITEMS[0]["text"][:6], line)
        for word in ("挑一条", "不要", "必须", "应该", "别再提", "要带上", "一句就够"):
            with self.subTest(word=word):
                self.assertNotIn(word, line)

    def test_empty(self):
        self.assertEqual(render_line([], asked=True), "")


if __name__ == "__main__":
    unittest.main()
