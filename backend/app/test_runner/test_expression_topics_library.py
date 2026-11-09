# -*- coding: utf-8 -*-
"""练习盘外部话题库的静态验收（plans/0923-06 §4.1、§5 第 2 条）。

只测静态库本身：**不联网、不连库、不调模型、不碰真实用户**。覆盖

  - 库级自检：≥12 张、三类各 ≥3、ID 唯一、来源均可打开的 https 原始来源、
    热点卡带 `expires_at`；
  - 可用性判定：过期卡 / 缺来源卡**不参加新盘选择**；库只余保底卡仍选到有真实
    来源的保底卡；
  - 确定性选择：同一 `(user_id, conv_id)` 恒定同一张，排除集生效；
  - 开场文本形态：英文、≤45 词、无 URL、无序号／连接词清单／五维术语、
    每张都带"也可讲你自己的话题"退路；
  - 快照字段与「hook 是据此延展、非原文断言」的标注。

跑法：cd chanpinzhujian/api/test_runner && python -m unittest test_expression_topics_library -v
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import expression_topics as et  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))       # backend/app/test_runner
APP = os.path.dirname(HERE)                             # backend/app
MODULE_PY = os.path.join(APP, "engine", "expression_topics.py")

FAR_FUTURE = date(2099, 1, 1)
ANCIENT = date(2000, 1, 1)


def read(path: str) -> str:
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


class LibraryValidationTests(unittest.TestCase):
    """库级自检 + 规模 / 来源 / 过期。"""

    def test_self_validation_passes(self) -> None:
        self.assertEqual([], et.validate_topic_library(), "话题库自检没过")

    def test_at_least_twelve_cards_and_three_per_type(self) -> None:
        self.assertGreaterEqual(len(et.TOPICS), 12, "初始卡不足 12 张")
        counts = {}
        for card in et.TOPICS:
            counts[card["type"]] = counts.get(card["type"], 0) + 1
        for ctype in et.TOPIC_TYPES:
            self.assertGreaterEqual(counts.get(ctype, 0), 3,
                                    "类型 %s 少于 3 张" % ctype)

    def test_ids_unique(self) -> None:
        ids = [c["id"] for c in et.TOPICS]
        self.assertEqual(len(ids), len(set(ids)), "存在重复 topic_id")

    def test_every_source_is_openable_https(self) -> None:
        for card in et.TOPICS:
            url = card["source_url"]
            self.assertTrue(url.lower().startswith("https://"),
                            "%s: URL 不是 https" % card["id"])
            parsed = urlparse(url)
            self.assertEqual("https", parsed.scheme)
            self.assertTrue(parsed.netloc, "%s: URL 无主机" % card["id"])
            self.assertTrue(str(card.get("source_title") or "").strip())
            self.assertTrue(str(card.get("source_org") or "").strip())

    def test_hot_issue_cards_have_expiry_others_do_not(self) -> None:
        for card in et.TOPICS:
            if card["type"] == "issue":
                self.assertTrue(str(card.get("expires_at") or "").strip(),
                                "%s: 热点卡缺 expires_at" % card["id"])
            else:
                self.assertIsNone(card.get("expires_at"),
                                  "%s: 非热点卡不应设 expires_at" % card["id"])

    def test_source_date_label_marks_unknown(self) -> None:
        unknown = {"source_published": "", "source_revised": ""}
        self.assertEqual("发表时间未知", et.source_date_label(unknown))
        both = {"source_published": "2002-05-14", "source_revised": "2025-07-08"}
        self.assertIn("2002-05-14", et.source_date_label(both))
        self.assertIn("2025-07-08", et.source_date_label(both))


class UsabilityAndFallbackTests(unittest.TestCase):
    """过期 / 缺来源 不进新盘；库只余保底卡时仍得有来源的卡。"""

    def test_missing_source_card_is_unusable(self) -> None:
        broken = dict(et.all_topics()[0])
        broken["source_url"] = ""
        self.assertFalse(et.is_topic_usable(broken))
        broken2 = dict(et.all_topics()[0])
        broken2["source_url"] = "http://example.com/not-https"
        self.assertFalse(et.is_topic_usable(broken2))

    def test_expired_card_is_unusable_and_not_eligible(self) -> None:
        expired = {
            "id": "issue-expired-fixture", "type": "issue",
            "source_title": "X", "source_url": "https://example.com/x",
            "source_org": "X", "source_published": "", "source_revised": "",
            "verified_at": "2026-09-22", "expires_at": "2026-01-01",
            "hook": "Some people say one thing is better than another.",
            "task": "Tell me what you think, and give one reason.",
        }
        self.assertFalse(et.is_topic_usable(expired, now=date(2026, 9, 22)))
        # 库内 issue 卡在足够远的未来一律退出（只余经典/TED）
        far = et.eligible_topics(now=FAR_FUTURE)
        self.assertTrue(far, "远期不该把库清空——经典/TED 不过期")
        self.assertTrue(all(c["type"] != "issue" for c in far),
                        "过期热点卡混进了新盘可选集")

    def test_fallback_is_sourceful_when_library_all_consumed(self) -> None:
        all_ids = [c["id"] for c in et.TOPICS]
        # 排除掉除保底卡外的全部卡 → 可选集恰为保底卡
        others = [i for i in all_ids if i != et.FALLBACK_TOPIC_ID]
        pool = et.eligible_topics(exclude_ids=others)
        self.assertEqual([et.FALLBACK_TOPIC_ID], [c["id"] for c in pool])
        # 全部排除 → 回到保底卡（仍是库内、有真实来源）
        picked = et.select_topic("u1", "c1", exclude_ids=all_ids)
        self.assertEqual(et.FALLBACK_TOPIC_ID, picked["id"])
        self.assertTrue(et.has_valid_source(picked))

    def test_fallback_card_never_expires(self) -> None:
        fb = et.fallback_topic(now=FAR_FUTURE)
        self.assertTrue(et.has_valid_source(fb))
        self.assertIsNone(fb.get("expires_at"))

    def test_fallback_from_clean_library_is_a_classic_card(self) -> None:
        fb = et.fallback_topic()
        self.assertEqual("classic", fb["type"])
        self.assertEqual(et.FALLBACK_TOPIC_ID, fb["id"])


class DeterministicSelectionTests(unittest.TestCase):
    """确定性选择：同盘同题、排除集生效、结果有来源。"""

    def test_same_conv_is_stable_across_calls(self) -> None:
        a = et.select_topic("user-1", "conv-1")
        b = et.select_topic("user-1", "conv-1")
        c = et.select_topic("user-1", "conv-1")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(a["id"], c["id"])

    def test_different_convs_are_independent(self) -> None:
        ids = {et.select_topic("user-1", "conv-%d" % i)["id"] for i in range(40)}
        self.assertGreater(len(ids), 1, "所有盘都抽到同一张卡，选择函数没生效")

    def test_exclusion_is_respected(self) -> None:
        recent = [c["id"] for c in et.TOPICS[:5]]
        picked = et.select_topic("user-1", "conv-9", exclude_ids=recent)
        self.assertNotIn(picked["id"], recent, "选中了最近用过的卡")

    def test_selection_always_sourceful(self) -> None:
        for i in range(20):
            card = et.select_topic("u", "c%d" % i)
            self.assertTrue(et.has_valid_source(card), "选到无来源卡")


class OpeningTextTests(unittest.TestCase):
    """开场文本形态：英文、短、无框架、带自选退路。"""

    def test_every_opening_passes_own_quality_check(self) -> None:
        for card in et.TOPICS:
            text = et.opening_text(card)
            self.assertEqual("", et.validate_opening_text(text),
                             "%s: 开场质检不过" % card["id"])

    def test_opening_is_short_english_without_url(self) -> None:
        for card in et.TOPICS:
            text = et.opening_text(card)
            words = re.findall(r"[A-Za-z']+", text)
            self.assertLessEqual(len(words), et.OPENING_MAX_WORDS)
            self.assertIsNone(re.search(r"[\u4e00-\u9fff]", text))
            self.assertNotIn("http://", text)
            self.assertNotIn("https://", text)

    def test_opening_has_own_topic_escape(self) -> None:
        for card in et.TOPICS:
            self.assertIn(et.OWN_TOPIC_PHRASE, et.opening_text(card))

    def test_opening_has_no_framework_or_toulmin_jargon(self) -> None:
        jargon = re.compile(r"\b(claim|warrant|qualifier|rebuttal|toulmin)\b", re.I)
        for card in et.TOPICS:
            text = et.opening_text(card)
            self.assertIsNone(jargon.search(text), "%s: 开场带五维术语" % card["id"])
            self.assertIsNone(re.search(r"(^|\s)\d[.)]\s", text),
                              "%s: 开场带序号框架" % card["id"])

    def test_opening_is_not_a_greeting(self) -> None:
        for card in et.TOPICS:
            self.assertNotEqual("greets", et.validate_opening_text(et.opening_text(card)))

    def test_negative_fixtures_are_rejected(self) -> None:
        self.assertEqual("empty", et.validate_opening_text(""))
        self.assertEqual("cjk", et.validate_opening_text("今天很累"))
        self.assertEqual("greets", et.validate_opening_text("Hi! Tell me one thing."))
        self.assertEqual("has_url", et.validate_opening_text(
            "Read https://example.com and tell me what you think."))
        self.assertEqual("framework", et.validate_opening_text(
            "First give a claim, then evidence, then a rebuttal."))
        self.assertEqual("too_long", et.validate_opening_text("word " * 60))


class SnapshotAndHonestyTests(unittest.TestCase):
    """快照字段与「hook 是据此延展、非原文断言」的标注。"""

    def test_snapshot_fields_match_plan(self) -> None:
        card = et.get_topic("classic-turing-test")
        snap = et.assignment_snapshot(card, user_id="u", conv_id="c",
                                      assigned_at="2026-09-22T10:00:00+08:00")
        self.assertEqual(
            {"user_id", "conv_id", "topic_id", "source_title", "source_url",
             "source_date", "prompt_text", "assigned_at"},
            set(snap.keys()))
        self.assertEqual("classic-turing-test", snap["topic_id"])
        self.assertTrue(snap["source_url"].startswith("https://"))
        self.assertEqual(et.opening_text(card), snap["prompt_text"])

    def test_snapshot_is_frozen_copy_not_live_reference(self) -> None:
        card = et.get_topic("classic-free-will")
        snap = et.assignment_snapshot(card, user_id="u", conv_id="c", assigned_at="t")
        # 改库不改变旧盘快照（快照存的是字符串，不持有卡对象）
        self.assertIsInstance(snap["source_title"], str)
        self.assertNotEqual(id(snap), id(card))

    def test_hook_disclaimer_is_exposed(self) -> None:
        self.assertTrue(et.HOOK_DISCLAIMER.strip())
        payload = et.source_payload(et.get_topic("ted-single-story"))
        self.assertEqual(et.HOOK_DISCLAIMER, payload["hook_disclaimer"])
        self.assertTrue(payload["has_source"])

    def test_source_payload_marks_missing_source(self) -> None:
        payload = et.source_payload({"id": "x", "source_title": "", "source_url": ""})
        self.assertFalse(payload["has_source"])


class NoNetworkTests(unittest.TestCase):
    """库必须离线可判：源码里不许出现联网调用（§4.1「无网络请求」）。"""

    def test_module_source_has_no_network_calls(self) -> None:
        src = read(MODULE_PY)
        for banned in ("urllib.request", "import requests", "socket.", "http.client",
                       "aiohttp", "httpx", "urlopen"):
            self.assertNotIn(banned, src, "话题库出现联网调用：%s" % banned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
