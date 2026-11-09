# -*- coding: utf-8 -*-
"""素材导入 V1 的离线可测部分（plans/0916-21 §9）。

覆盖：标题 3–6 字 / 正文提取（去脚本样式、块级转换行）/ 最长文本块 /
分块与上限 / 摘要触发阈值 / **抓取失败必须返回明确原因且不产空正文**
（用本地 8002 的 404 做确定性的失败源，不打外网）。

另覆盖 plans/0916-24 §三·五 修复 1 **视频热评**：排序/封顶/去重/截断、
B 站 wbi 与老接口兜底、YouTube innertube 解析、以及"评论失败绝不影响 ok"。
评论相关全部离线（stub client / 本地 stub 页），不打外网。
"""
import http.server
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from engine import link_fetcher, material_analyzer, media_transcribe

SAMPLE_HTML = """<!doctype html>
<html><head><title>行尸走肉 第一季 - 维基百科，自由的百科全书</title>
<meta property="og:description" content="《行尸走肉》第一季讲述警长瑞克在末日中醒来。">
<style>.nav{color:red}</style><script>var x = "瑞克";</script></head>
<body>
<nav>首页 随机 附近 登录</nav>
<p>《行尸走肉》（The Walking Dead）第一季共 6 集，2010 年首播。</p>
<p>主角瑞克·格莱姆斯在昏迷中醒来，发现世界已被丧尸占领，他前往亚特兰大寻找家人，
途中遇到摩根与杜安，并逐渐集结起包括肖恩、洛瑞、卡尔、达里尔在内的一群人。</p>
<footer>版权声明 联系</footer>
</body></html>"""

CHINESE_BODY = "《行尸走肉》第一季第一集名为 Days Gone Bye。" + "瑞克在医院醒来，发现走廊里都是丧尸。" * 60
ENGLISH_BODY = "Call of Duty Black Ops is a first person shooter. " * 80
# 多段正文（两段各 ~1080 字，都超过 BLOCK_CHARS=1200 的聚合窗口）：
# 用来验证"块会按 block_chars 切开、且块数封顶"。
MULTI_PARA_BODY = (
    "瑞克在医院醒来，发现走廊里都是丧尸。" * 60
    + "\n"
    + "他遇到摩根，两人一起去了亚特兰大寻找补给。" * 60
)

# ── 视频热评 fixture（plans/0916-24 §三·五 修复 1，离线）─────────────────
# B 站 wbi 正常时的最小响应：nav 给签名密钥，reply/wbi/main 给置顶 + 热评
BILI_NAV_OK = {"code": 0, "data": {"wbi_img": {
    "img_url": "https://i0.hdslb.com/bfs/wbi/" + "a" * 32 + ".png",
    "sub_url": "https://i0.hdslb.com/bfs/wbi/" + "b" * 32 + ".png"}}}
BILI_REPLY_WBI_OK = {"code": 0, "data": {
    "top_replies": [{"content": {"message": "置顶评论"}, "like": 500}],
    "replies": [{"content": {"message": "热评一"}, "like": 12},
                {"content": {"message": "热评二"}, "like": 340}]}}
# YouTube innertube next 响应：两条评论，点赞数是 "1.2K"/"3.4万" 这种文本
YT_NEXT_PAYLOAD = {"onResponseReceivedEndpoints": [{"appendContinuationItemsAction": {
    "continuationItems": [
        {"commentThreadRenderer": {"comment": {"commentRenderer": {
            "contentText": {"runs": [{"text": "Great "}, {"text": "video"}]},
            "voteCount": {"simpleText": "1.2K"}}}}},
        {"commentThreadRenderer": {"comment": {"commentRenderer": {
            "contentText": {"simpleText": "first!"},
            "voteCount": {"simpleText": "3.4万"}}}}},
    ]}}]}
YT_INITIAL_DATA = {"contents": {"twoColumnWatchNextResults": {"results": {"results": {
    "contents": [{"itemSectionRenderer": {
        "sectionIdentifier": "comment-item-section",
        "contents": [{"continuationItemRenderer": {"continuationEndpoint": {
            "continuationCommand": {"token": "TOKEN-123"}}}}]}}]}}}}}
# 页面里 ytInitialData 的值串里含一个 `}`（验证手写括号配对不会被字符串里的括号带偏）
YT_PAGE = ('<script>var ytInitialData = ' + json.dumps(YT_INITIAL_DATA) + ';</script>'
           '<script>"INNERTUBE_API_KEY":"KEY123"</script>'
           '<script>"INNERTUBE_CLIENT_VERSION":"2.20240101.00.00"</script>')

# 本地 stub 视频页：用来离线验证"评论失败也只是 []，ok 不变"
VIDEO_PAGE_HTML = """<!doctype html>
<html><head><title>离线测试视频页 - bilibili</title>
<meta property="og:description" content="这是一段视频简介，用来把视频页路径跑通并验证评论不会改变抓取结果。">
</head><body><p>播放器占位</p>
<aside>推荐视频里的独立话题：SIDEBAR_ONLY_TITLE</aside>
</body></html>"""

# 本地 stub 图文页（非视频路径）：验证非视频页 ok=True 且 comments 恒为 []
ARTICLE_PAGE_HTML = """<!doctype html>
<html><head><title>离线测试文章页</title></head><body>
<p>这是一段用于离线测试的正文，""" + "内容足够长，确保能通过正文提取的最低字数门槛。" * 12 + """</p>
</body></html>"""


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _StubClient:
    """离线 httpx 替身：按 URL 里的关键字返回预置 JSON，并记录请求参数。"""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _route(self, url):
        for needle, payload in self.routes.items():
            if needle in url:
                return _StubResponse(payload)
        return _StubResponse({})

    async def get(self, url, params=None):
        self.calls.append((url, params or {}))
        return self._route(url)

    async def post(self, url, params=None, json=None):
        self.calls.append((url, params or {}))
        return self._route(url)


class _VideoPageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        page = VIDEO_PAGE_HTML if "/watch" in self.path else ARTICLE_PAGE_HTML
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TitleTests(unittest.TestCase):
    def test_chinese_title_truncated_to_6(self):
        # 中文标题先去掉内部空白再按 6 字截：得到"行尸走肉第一"，
        # 而不是被空格吃掉一个字后截成"行尸走肉 第"（断词）。
        self.assertEqual(material_analyzer.pick_title("", title_hint="行尸走肉 第一季 - 豆瓣"),
                         "行尸走肉第一")

    def test_site_suffix_stripped_for_english(self):
        # 英文按词截，站点后缀（- 维基百科）必须先去掉，否则标题是"站点名"
        got = material_analyzer.pick_title("", title_hint="Call of Duty - Wikipedia")
        self.assertEqual(got, "Call of Duty")

    def test_title_from_first_meaningful_line(self):
        got = material_analyzer.pick_title("# 周会纪要\n今天要开周会", title_hint="")
        self.assertEqual(got, "周会纪要")

    def test_never_empty(self):
        self.assertTrue(material_analyzer.pick_title("", title_hint="").strip())


class ExternalBackgroundTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_upgrade_preserves_previous_sources_when_pages_are_offline(self):
        cache = {"status": "ready", "schema_version": 5, "sources": [{
            "url": "https://example.org/atlas", "snippet": "Mira leads Atlas."}]}
        upgraded = await material_analyzer.enrich_external_context(
            cache, page_fetcher=mock.AsyncMock(return_value={}))
        self.assertEqual(upgraded["schema_version"], 6)
        self.assertEqual(upgraded["sources"][0]["snippet"], "Mira leads Atlas.")
        self.assertEqual(cache["schema_version"], 5)
        self.assertNotIn("content_status", cache["sources"][0])

    async def test_reference_body_is_cached_without_rewriting_material_facts(self):
        search = SimpleNamespace(search=mock.AsyncMock(return_value={
            "status": "ok", "results": [{"title": "Atlas expedition official guide",
            "url": "https://example.org/atlas", "snippet": "Atlas follows Mira's expedition."}]}))
        page = mock.AsyncMock(return_value={"url": "https://example.org/atlas",
            "text": "Mira and Lena are siblings. " * 20})
        analysis = {"external_context_queries": ["Atlas Mira expedition official"]}
        result = await material_analyzer.build_external_context(
            "Atlas expedition", kind="link", source_coverage={"kind": "video", "body_available": True},
            analysis=analysis, body_text="Mira starts the Atlas expedition.",
            search_tool=search, page_fetcher=page, enabled=True)
        self.assertIn("Mira and Lena are siblings", result["sources"][0]["body_text"])
        self.assertEqual(result["sources"][0]["content_status"], "fetched")
        self.assertNotIn("external_context", analysis)
        page.assert_awaited_once()

    async def test_reference_failure_keeps_attributed_search_excerpt(self):
        search = SimpleNamespace(search=mock.AsyncMock(return_value={"status": "ok", "results": [{
            "title": "Atlas Mira expedition", "url": "https://example.org/atlas",
            "snippet": "Atlas follows Mira's expedition."}]}))
        for response in ({}, {"url": "https://unrelated.example/other", "text": "Wrong work."}):
            result = await material_analyzer.build_external_context(
                "Atlas expedition", kind="link", source_coverage={"kind": "video", "body_available": True},
                analysis={"external_context_queries": ["Atlas Mira expedition"]},
                body_text="Mira starts the Atlas expedition.", search_tool=search,
                page_fetcher=mock.AsyncMock(return_value=response), enabled=True)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["sources"][0]["content_status"], "unavailable")
            self.assertNotIn("body_text", result["sources"][0])

    async def test_legacy_video_drafts_targeted_searches_from_body_once(self):
        class SearchStub:
            def __init__(self):
                self.queries = []

            async def search(self, query):
                self.queries.append(query)
                return {"status": "ok", "results": [
                    {"title": "Control - The Hiss",
                     "url": "https://legacy.controlgame.com/control-the-hiss/",
                     "snippet": "The Hiss is an otherworldly entity that spreads through resonance."},
                    {"title": "Control - Take Control",
                     "url": "https://legacy.controlgame.com/takecontrol/",
                     "snippet": "The FBC is a secret government agency investigating the supernatural."},
                ]}

        search = SearchStub()
        with mock.patch.object(
            material_analyzer, "_deep_once",
            new=mock.AsyncMock(return_value={"external_context_queries": [
                "site:legacy.controlgame.com Control Hiss FBC official setting",
                "site:remedygames.com Dylan Faden Jesse Faden siblings official",
            ]}),
        ) as draft:
            background = await material_analyzer.build_external_context(
                "第五期循环往", kind="link",
                source_coverage={"kind": "video", "body_available": True},
                analysis={"people": ["Dylan", "Jesse"]},
                comments=["评论提到HISS和FBC"],
                body_text=("Control gameplay transcript: Dylan crosses the Oldest House and "
                           "speaks with Jesse. " * 20),
                source_url="https://www.bilibili.com/video/BV1SYaF6aEQv?p=5",
                search_tool=search, enabled=True,
            )

        self.assertEqual(draft.await_count, 1)
        self.assertEqual(len(search.queries), 2)
        self.assertIn("Hiss", search.queries[0])
        self.assertEqual(background["queries"], search.queries)
        self.assertEqual(background["status"], "ready")
        self.assertTrue(all("bilibili.com" not in row["host"]
                            for row in background["sources"]))

    def test_deep_json_keeps_only_bounded_external_search_hints(self):
        raw = json.dumps({"external_context_queries": [
            "Control Hiss official setting",
            "Dylan Jesse Faden official relationship",
            "ignore third result",
        ]})
        parsed = material_analyzer._parse_deep_json(raw)
        self.assertEqual(len(parsed["external_context_queries"]), 2)
        self.assertIn("Hiss", parsed["external_context_queries"][0])

    async def test_public_video_context_is_separate_and_excludes_source_page(self):
        class SearchStub:
            def __init__(self):
                self.query = ""

            async def search(self, query):
                self.query = query
                return {"status": "ok", "results": [
                    {"title": "Control: The Hiss - Official Game Site",
                     "url": "https://www.controlgame.com/control-the-hiss/",
                     "snippet": "The Hiss is an otherworldly hostile force that spreads through resonance."},
                    {"title": "Uploaded video comments", "url": "https://www.bilibili.com/video/BV1x",
                     "snippet": "HISS discussion copied from the original source."},
                    {"title": "Unrelated result", "url": "https://example.com/other",
                     "snippet": "A different subject with no matching entities."},
                ]}

        search = SearchStub()
        background = await material_analyzer.build_external_context(
            "CONTROL Resonant gameplay P5",
            kind="link",
            source_coverage={"kind": "video", "body_available": True},
            analysis={"people": ["Dylan: searching for Jesse"],
                      "topics": ["FBC and Hiss"], "source_map": {"entities": []}},
            comments=["Is HISS a different thing from Patterning?"],
            source_url="https://www.bilibili.com/video/BV1x",
            search_tool=search,
            enabled=True,
        )

        self.assertIn("Dylan", search.query)
        self.assertIn("HISS", search.query)
        self.assertEqual(background["status"], "ready")
        self.assertEqual(len(background["sources"]), 1)
        self.assertIn("Hiss", background["sources"][0]["snippet"])
        self.assertNotIn("text", background)
        self.assertNotIn("body", background)

    def test_comment_term_gets_material_anchor_even_when_broad_query_mentions_it(self):
        queries = material_analyzer._supplement_comment_search_query(
            ["HISS official supernatural setting"],
            "第五期 CONTROL 实况",
            {"people": ["Dylan", "Jesse"], "topics": ["FBC Patterning"]},
            ["评论提到 HISS，但这个词具体指什么？"],
        )
        self.assertEqual(len(queries), 2)
        self.assertIn("HISS", queries[1])
        self.assertTrue(any(anchor in queries[1] for anchor in ("CONTROL", "Dylan", "FBC")))

    def test_canonical_spelling_of_comment_term_is_not_rewritten_away(self):
        queries = material_analyzer._supplement_comment_search_query(
            ["Control Dylan Jesse FBC official setting",
             "Control Hiss FBC official game entity"],
            "CONTROL gameplay P5",
            {"people": ["Dylan", "Jesse"], "topics": ["FBC Patterning"]},
            ["有人猜姐姐会出去对抗西斯，但这只是评论者的推测。"],
        )
        self.assertEqual(queries[1], "Control Hiss FBC official game entity")

    async def test_comment_named_lore_source_survives_broader_orientation_result(self):
        class SearchStub:
            async def search(self, query):
                if "西斯" in query:
                    return {"status": "ok", "results": [{
                        "title": "CONTROL: The Hiss",
                        "url": "https://legacy.controlgame.com/control-the-hiss/",
                        "snippet": "The Hiss is a hostile resonance force; the FBC studies it.",
                    }]}
                return {"status": "ok", "results": [{
                    "title": "CONTROL Resonant, Dylan and Jesse at the FBC",
                    "url": "https://www.remedygames.com/control-resonant",
                    "snippet": "Official sequel background for CONTROL, Dylan, Jesse, and the FBC.",
                }]}

        result = await material_analyzer.build_external_context(
            "CONTROL gameplay P5", kind="link",
            source_coverage={"kind": "video", "body_available": True},
            analysis={"external_context_queries": [
                "Control Dylan Jesse FBC official sequel background",
                "Control 西斯 希斯 FBC",
            ]},
            comments=["有人猜姐姐出去对抗西斯，但这只是评论者的推测。"],
            body_text="CONTROL gameplay with Dylan and Jesse. " * 12,
            source_url="https://www.bilibili.com/video/BV1SYaF6aEQv?p=5",
            search_tool=SearchStub(), enabled=True,
        )

        self.assertEqual(result["status"], "ready")
        self.assertIn("Hiss", result["queries"][1])
        self.assertEqual(
            result["sources"][0]["url"],
            "https://legacy.controlgame.com/control-the-hiss/",
        )

    def test_comment_term_extraction_does_not_treat_opinion_phrases_as_names(self):
        comment = (
            "FBC出了太古屋后缺乏认知，面对未知共振体很被动；"
            "有人猜姐姐会独自出去对抗西斯，而这只是评论者的推测。"
        )
        self.assertEqual(
            material_analyzer._external_context_comment_terms([comment]), ["西斯"]
        )

    async def test_ambiguous_remedy_hits_are_rejected_without_two_content_anchors(self):
        class SearchStub:
            async def search(self, query):
                return {"status": "ok", "results": [
                    {"title": "Remedy Skin Repair", "url": "https://remedyskin.com/repair",
                     "snippet": "A lightweight face cream for dry skin."},
                    {"title": "HISS Definition", "url": "https://dictionary.example/hiss",
                     "snippet": "A sharp sound; remedy for a sore throat."},
                    {"title": "BMC Remedy ITSM", "url": "https://bmc.example/remedy",
                     "snippet": "Service management software for enterprise teams."},
                ]}

        result = await material_analyzer.build_external_context(
            "CONTROL Resonant gameplay P5", kind="link",
            source_coverage={"kind": "video", "body_available": True},
            analysis={"external_context_queries": ["CONTROL HISS FBC official game setting"]},
            comments=["The HISS is mentioned in this Control discussion."],
            body_text="CONTROL Resonant gameplay transcript with Dylan and FBC. " * 12,
            source_url="https://www.bilibili.com/video/BV1SYaF6aEQv?p=5",
            search_tool=SearchStub(), enabled=True,
        )
        self.assertEqual(result["schema_version"], material_analyzer.EXTERNAL_CONTEXT_SCHEMA_VERSION)
        self.assertEqual(result["status"], "no_relevant_results")
        self.assertEqual(result["sources"], [])

    async def test_relevant_hiss_page_matches_content_not_just_host(self):
        class SearchStub:
            async def search(self, query):
                return {"status": "ok", "results": [
                    {"title": "Control - The Hiss", "url": "https://legacy.controlgame.com/control-the-hiss/",
                     "snippet": "The Hiss is an otherworldly force that spreads through resonance."},
                    {"title": "Unrelated Hiss definition", "url": "https://legacy.controlgame.com/dictionary/hiss",
                     "snippet": "A sibilant sound."},
                ]}

        result = await material_analyzer.build_external_context(
            "CONTROL Resonant gameplay P5", kind="link",
            source_coverage={"kind": "video", "body_available": True},
            analysis={"external_context_queries": ["CONTROL HISS FBC official game setting"]},
            comments=["The HISS is mentioned in this Control discussion."],
            body_text="CONTROL Resonant gameplay transcript with Dylan and FBC. " * 12,
            source_url="https://www.bilibili.com/video/BV1SYaF6aEQv?p=5",
            search_tool=SearchStub(), enabled=True,
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("otherworldly force", result["sources"][0]["snippet"])

    async def test_non_video_or_page_only_material_does_not_search(self):
        class SearchStub:
            called = False

            async def search(self, query):
                self.called = True
                return {"status": "ok", "results": []}

        search = SearchStub()
        result = await material_analyzer.build_external_context(
            "Control Resonant", kind="link",
            source_coverage={"kind": "video", "body_available": False},
            analysis={"people": ["Dylan"]}, search_tool=search, enabled=True,
        )
        self.assertEqual(result, {})
        self.assertFalse(search.called)


class ExtractTests(unittest.TestCase):
    def test_script_and_style_dropped(self):
        title, body = link_fetcher.strip_html(SAMPLE_HTML)
        self.assertIn("行尸走肉", title)
        self.assertNotIn("color:red", body)
        self.assertNotIn("var x", body)
        self.assertIn("瑞克", body)

    def test_longest_block_prefers_body(self):
        _title, body = link_fetcher.strip_html(SAMPLE_HTML)
        block = link_fetcher.longest_text_block(body)
        self.assertIn("亚特兰大", block)
        self.assertNotIn("版权声明", block)

    def test_video_host_detection(self):
        self.assertTrue(link_fetcher.is_video_url("https://www.youtube.com/watch?v=abc"))
        self.assertTrue(link_fetcher.is_video_url("https://www.bilibili.com/video/BV1xx"))
        self.assertFalse(link_fetcher.is_video_url("https://zh.wikipedia.org/wiki/Walking_Dead"))

    def test_video_fallback_description_drops_related_video_titles(self):
        desc = ("页面标题与播放数据。作者简介：你好。相关视频：别的视频标题一，"
                "别的视频标题二。")
        cleaned = link_fetcher._video_page_description(desc)
        self.assertTrue(cleaned.startswith("页面标题与播放数据。作者简介：你好"))
        self.assertNotIn("别的视频标题", cleaned)


class BlockTests(unittest.TestCase):
    def test_blocks_respect_size_and_cap(self):
        blocks = material_analyzer.split_blocks(MULTI_PARA_BODY)
        self.assertGreaterEqual(len(blocks), 2)
        self.assertLessEqual(len(blocks), material_analyzer.MAX_BLOCKS)
        for block in blocks:
            self.assertLessEqual(len(block), material_analyzer.BLOCK_CHARS)

    def test_overlong_paragraph_hard_split(self):
        blocks = material_analyzer.split_blocks("字" * 5000)
        self.assertTrue(blocks)
        for block in blocks:
            self.assertLessEqual(len(block), material_analyzer.BLOCK_CHARS)

    def test_blocks_capped_for_huge_input(self):
        blocks = material_analyzer.split_blocks("段落内容。" * 4000)
        self.assertEqual(len(blocks), material_analyzer.MAX_BLOCKS)

    def test_summary_trigger_threshold(self):
        self.assertFalse(material_analyzer.needs_summary("短素材"))
        self.assertTrue(material_analyzer.needs_summary(
            "字" * (material_analyzer.SUMMARY_TRIGGER_CHARS + 1)))


class AnalyzeTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_without_model_keeps_text(self):
        result = await material_analyzer.analyze_material(
            CHINESE_BODY, kind="text", allow_model=False)
        self.assertTrue(result["title"])
        self.assertEqual(result["char_count"], len(CHINESE_BODY))
        self.assertTrue(result["blocks"])
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["analyzer"], material_analyzer.ANALYZER_NAME)

    async def test_short_text_needs_no_summary(self):
        result = await material_analyzer.analyze_material(
            "很短的一段素材", kind="diary", allow_model=True)
        self.assertEqual(result["summary"], "")


class FetchFailureTests(unittest.IsolatedAsyncioTestCase):
    """§7-1 底线：抓不到必须给明确原因，且 text 为空 —— 绝不产空素材带。"""

    async def test_http_404_reports_reason(self):
        import httpx
        with mock.patch("httpx.AsyncClient") as client:
            client.return_value.__aenter__.return_value.get = mock.AsyncMock(
                return_value=httpx.Response(404))
            got = await link_fetcher.fetch_url("https://example.org/missing")
        self.assertFalse(got["ok"])
        self.assertEqual(got["text"], "")
        self.assertIn("404", str(got["error"]))

    async def test_empty_url_reports_reason(self):
        got = await link_fetcher.fetch_url("   ")
        self.assertFalse(got["ok"])
        self.assertEqual(got["text"], "")
        self.assertTrue(str(got["error"]).strip())


class VideoCoverageTests(unittest.IsolatedAsyncioTestCase):
    def test_page_only_summary_keeps_description_but_excludes_comments(self):
        main = __import__("main")
        summary = main._video_page_only_summary(
            "Video title. Page description about the creator's subject. "
            "相关视频：An unrelated video.\n\n"
            "（以下是这条内容下的观众评论，来自观众，不是正文：）\n"
            "A viewer claims X is the MVP.", "Video title")
        self.assertIn("《Video title》", summary)
        self.assertIn("无法确认UP主的具体论点", summary)
        self.assertNotIn("A viewer claims", summary)
        self.assertNotIn("An unrelated video", summary)

    async def test_page_only_video_keeps_explicit_coverage_and_asr_reason(self):
        fetched = {
            "ok": True,
            "kind": "video_light",
            "title": "Interview clip",
            "text": "Title and description only.",
            "comments": [{"text": "A viewer's opinion."}],
            "note": "transcription unavailable: quota exhausted",
        }
        with mock.patch.object(link_fetcher, "fetch_url", return_value=fetched):
            source = await __import__("main")._material_source_of(
                "link", "https://example.test/video", "", "Interview clip")
        self.assertTrue(source["ok"])
        self.assertEqual(source["source_coverage"]["kind"], "video")
        self.assertFalse(source["source_coverage"]["body_available"])
        self.assertEqual(source["source_coverage"]["scope"], "page_and_comments_only")
        self.assertIn("quota exhausted", source["source_coverage"]["note"])
        self.assertIn("观众评论", source["text"])


class VideoAndPlaceholderTests(unittest.TestCase):
    """两条判据来自 2026-09-17 拿真实 B 站链接抓出来的实测（不需要联网即可回归）。"""

    def test_video_hosts_recognised(self):
        for url in ("https://www.youtube.com/watch?v=x",
                    "https://www.bilibili.com/video/BV1xx411c7e3",
                    "https://b23.tv/abcdefg",
                    "https://www.douyin.com/video/123"):
            self.assertTrue(link_fetcher.is_video_url(url), url)

    def test_bilibili_column_is_article_not_video(self):
        # 专栏(/read/cv…)是**图文**：按视频站处理只拿标题（实测 35 字），
        # 走文章路径能抽到 447 字正文。所以同域要按路径分流。
        self.assertFalse(link_fetcher.is_video_url(
            "https://www.bilibili.com/read/cv16499284/"))
        # 同域的视频路径不受影响
        self.assertTrue(link_fetcher.is_video_url(
            "https://www.bilibili.com/video/BV1xx411c7e3"))

    def test_deleted_video_placeholder_detected(self):
        # 已删视频：HTTP 200，标题是"视频去哪了呢？"，简介是**未填充模板**。
        self.assertTrue(link_fetcher.is_placeholder_page(
            "视频去哪了呢？_呵呵呵呵_bilibili",
            "{$0}, 视频播放量 undefined、弹幕数 {$1}、点赞数 {$2}"))
        # 模板残留单独也判得出（标题被站点改写的情况）
        self.assertTrue(link_fetcher.is_placeholder_page(
            "某个视频", "弹幕数 {$1}、点赞数 {$2}"))
        self.assertTrue(link_fetcher.is_placeholder_page("404 Not Found", ""))

    def test_real_video_page_not_placeholder(self):
        self.assertFalse(link_fetcher.is_placeholder_page(
            "【Gero】Just Be Friends >.<大赞！！_呵呵呵呵_bilibili",
            "nico转载 sm9329290 >3<Gero！一大波再生TAT大感动！, "
            "视频播放量 29062、弹幕数 177、点赞数 321"))


class CommentNormalizeTests(unittest.TestCase):
    """§三·五 修复 1：热评排序 / 封顶 / 去重 / 截断。"""

    def test_sorted_deduped_truncated(self):
        raw = [
            {"text": "低赞评论", "likes": 1},
            {"text": "高赞评论", "likes": 99},
            {"text": "高赞评论", "likes": 5},      # 同文重复：留点赞高的那条
            {"text": "长" * (link_fetcher.MAX_COMMENT_CHARS + 50), "likes": 50},
            {"text": "   ", "likes": 999},          # 空文本丢弃
            {"text": "缺点赞字段"},
            "不是 dict 的脏数据",
        ]
        got = link_fetcher.normalize_comments(raw)
        self.assertEqual([c["likes"] for c in got], [99, 50, 1, 0])  # 点赞降序
        self.assertEqual(len([c for c in got if c["text"] == "高赞评论"]), 1)  # 去重
        self.assertEqual(got[0]["likes"], 99)                        # 留的是高赞那条
        for item in got:
            self.assertLessEqual(len(item["text"]), link_fetcher.MAX_COMMENT_CHARS)

    def test_capped(self):
        raw = [{"text": "评论%d" % i, "likes": i} for i in range(link_fetcher.MAX_COMMENTS + 5)]
        got = link_fetcher.normalize_comments(raw)
        self.assertEqual(len(got), link_fetcher.MAX_COMMENTS)
        self.assertEqual(got[0]["likes"], link_fetcher.MAX_COMMENTS + 4)  # 封顶留下最热的

    def test_empty_input(self):
        self.assertEqual(link_fetcher.normalize_comments([]), [])
        self.assertEqual(link_fetcher.normalize_comments(None), [])


class BilibiliCommentTests(unittest.IsolatedAsyncioTestCase):
    """B 站评论不在页面 HTML 里，走 JSON 接口；wbi 失败要能退老接口。"""

    def test_reads_uploader_display_name_from_page_metadata(self):
        state = {"videoData": {"owner": {"mid": 42, "name": "纯黑"}}}
        page = "<script>window.__INITIAL_STATE__=" + json.dumps(state) + ";</script>"
        self.assertEqual(link_fetcher.bilibili_creator_display_name(page), "纯黑")

    def test_missing_or_malformed_uploader_metadata_stays_unknown(self):
        self.assertEqual(link_fetcher.bilibili_creator_display_name("<html>no state</html>"), "")
        self.assertEqual(link_fetcher.bilibili_creator_display_name(
            "<script>__INITIAL_STATE__={bad json};</script>"), "")

    async def test_wbi_path_signed_and_sorted(self):
        client = _StubClient({"nav": BILI_NAV_OK, "reply/wbi/main": BILI_REPLY_WBI_OK})
        raw = await link_fetcher._bilibili_comments(
            client, "https://www.bilibili.com/video/BV1mheg6hE6j",
            '{"aid":117285138203445}')
        got = link_fetcher.normalize_comments(raw)
        self.assertEqual([c["likes"] for c in got], [500, 340, 12])
        signed = [params for url, params in client.calls if "reply/wbi/main" in url]
        self.assertTrue(signed)
        self.assertTrue(signed[0].get("w_rid") and signed[0].get("wts"))  # 确实签了名
        self.assertEqual(signed[0].get("oid"), 117285138203445)

    async def test_falls_back_to_legacy_endpoint(self):
        # nav 拿不到 wbi 密钥 → 退到不签名的老接口（实测只回前几条，但有胜于无）
        client = _StubClient({
            "nav": {"code": -101, "data": None},
            "x/v2/reply": {"code": 0, "data": {
                "replies": [{"content": {"message": "老接口热评"}, "like": 7}]}}})
        raw = await link_fetcher._bilibili_comments(
            client, "https://www.bilibili.com/video/BV1mheg6hE6j", '{"aid":1}')
        self.assertEqual(raw, [{"text": "老接口热评", "likes": 7}])

    async def test_no_aid_returns_empty(self):
        client = _StubClient({})
        raw = await link_fetcher._bilibili_comments(
            client, "https://www.bilibili.com/video/BV1mheg6hE6j", "没有 aid 的页面")
        self.assertEqual(raw, [])


class YoutubeCommentTests(unittest.IsolatedAsyncioTestCase):
    """YouTube 评论：ytInitialData → continuation token → innertube next 解析。"""

    def test_json_object_at_ignores_braces_inside_strings(self):
        text = 'var ytInitialData = {"a":{"b":"}"}}; tail'
        self.assertEqual(link_fetcher._json_object_at(text, 0), '{"a":{"b":"}"}}')

    def test_comment_continuation_token(self):
        self.assertEqual(link_fetcher.yt_comment_continuation(YT_INITIAL_DATA), "TOKEN-123")

    def test_parse_payload_text_and_likes(self):
        got = link_fetcher.normalize_comments(link_fetcher.parse_youtube_comments(YT_NEXT_PAYLOAD))
        self.assertEqual(got, [{"text": "first!", "likes": 34000},
                               {"text": "Great video", "likes": 1200}])

    def test_like_count_formats(self):
        self.assertEqual(link_fetcher._parse_like_count({"simpleText": "1,234"}), 1234)
        self.assertEqual(link_fetcher._parse_like_count({"simpleText": "3.4万"}), 34000)
        self.assertEqual(link_fetcher._parse_like_count({"runs": [{"text": "no"}]}), 0)

    async def test_youtube_page_flow(self):
        client = _StubClient({"youtubei/v1/next": YT_NEXT_PAYLOAD})
        raw = await link_fetcher._youtube_comments(client, YT_PAGE)
        self.assertEqual([c["likes"] for c in link_fetcher.normalize_comments(raw)],
                         [34000, 1200])


class VideoPageCommentsTests(unittest.IsolatedAsyncioTestCase):
    """视频页成功路径：comments 进结果；评论拿不到也只是 []，ok 与 text 不受影响。"""

    @classmethod
    def setUpClass(cls):
        # httpx 默认 trust_env=True，会读系统代理（Windows 注册表 / HTTP(S)_PROXY）。
        # 本机开着 Clash 之类时，连 http://127.0.0.1:<port> 也会被送去代理，
        # 代理回 400 "plain HTTP request was sent to HTTPS port" → 用例全红，
        # 但这是**测试环境**问题，不是抓取代码坏了。显式让回环地址绕过代理。
        cls._proxy_env_backup = {}
        for key in ("NO_PROXY", "no_proxy"):
            cls._proxy_env_backup[key] = os.environ.get(key)
        existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        bypass = ",".join(x for x in (existing, "127.0.0.1", "localhost") if x)
        os.environ["NO_PROXY"] = bypass
        os.environ["no_proxy"] = bypass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _VideoPageHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        for key, val in cls._proxy_env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def _url(self):
        return "http://127.0.0.1:%d/watch?v=1" % self.port

    async def _fetch_video(self, **patches):
        hosts = tuple(link_fetcher.VIDEO_HOSTS) + ("127.0.0.1",)
        # 2026-09-21：视频分支现在**先试音轨转写**。测试必须把它挡住——
        # 否则单测会真的去起 yt-dlp 子进程（慢、且依赖外部工具装没装）。
        # 默认给"转写不可用"，于是这些用例走的正是**降级到轻素材**那条路。
        spoken = patches.get("spoken")
        if spoken is None:
            async def spoken(*args, **kwargs):
                return {"ok": False, "text": "", "title": "", "duration": 0,
                        "note": "", "error": "（测试桩）转写不可用"}
        with mock.patch.object(link_fetcher, "VIDEO_HOSTS", hosts), \
             mock.patch.object(media_transcribe, "transcribe_url", new=spoken):
            if "comments" in patches:
                with mock.patch.object(link_fetcher, "_fetch_video_comments",
                                       new=patches["comments"]):
                    return await link_fetcher.fetch_url(self._url(), timeout=5)
            return await link_fetcher.fetch_url(self._url(), timeout=5)

    async def test_video_page_without_comments_still_ok(self):
        got = await self._fetch_video()
        self.assertTrue(got["ok"])
        self.assertEqual(got["kind"], "video_light")
        self.assertEqual(got["comments"], [])
        self.assertTrue(got["text"])

    async def test_transcript_becomes_the_material_text(self):
        """转写成功 → 视频里讲的话就是素材正文，kind 标 video。"""
        async def fake_spoken(*args, **kwargs):
            return {"ok": True, "text": "今天讲罐焖牛肉的做法，先切牛肋条。",
                    "title": "某视频", "duration": 410,
                    "note": "已从视频音轨转写全文（2 段）", "error": ""}
        got = await self._fetch_video(spoken=fake_spoken)
        self.assertTrue(got["ok"])
        self.assertEqual(got["kind"], "video")
        self.assertIn("罐焖牛肉", got["text"])
        self.assertNotIn("SIDEBAR_ONLY_TITLE", got["text"])
        self.assertIn("转写全文", got["note"])

    async def test_transcript_failure_falls_back_and_says_why(self):
        """转写失败 → 降级到轻素材，但**原因必须出现在 note 里**——
        那是用户唯一能看到的解释（是"这个视频不行"还是"能力没开"）。"""
        async def fake_spoken(*args, **kwargs):
            return {"ok": False, "text": "", "title": "", "duration": 0,
                    "note": "", "error": "这个视频太长（50 分钟），当前一次最多处理 35 分钟"}
        got = await self._fetch_video(spoken=fake_spoken)
        self.assertTrue(got["ok"])
        self.assertEqual(got["kind"], "video_light")
        self.assertIn("最多处理 35 分钟", got["note"])
        self.assertTrue(got["text"], "轻素材（标题+简介）仍要给出")

    async def test_transcript_empty_text_is_treated_as_failure(self):
        """转写"成功"但文本为空 → 同样降级，不能产一盘没有正文的带。"""
        async def fake_spoken(*args, **kwargs):
            return {"ok": True, "text": "   ", "title": "", "duration": 0,
                    "note": "已转写", "error": ""}
        got = await self._fetch_video(spoken=fake_spoken)
        self.assertEqual(got["kind"], "video_light")
        self.assertTrue(got["text"])

    async def test_comments_flow_into_result(self):
        async def fake_comments(*args, **kwargs):
            return [{"text": "来自评论区的热评", "likes": 42}]
        got = await self._fetch_video(comments=fake_comments)
        self.assertTrue(got["ok"])
        self.assertEqual(got["comments"], [{"text": "来自评论区的热评", "likes": 42}])
        self.assertTrue(got["text"])  # 主正文不受评论影响

    async def test_non_video_page_has_empty_comments(self):
        # 非视频页：走文章路径，ok=True，comments 恒为 []（不该去抓评论）
        got = await link_fetcher.fetch_url(
            "http://127.0.0.1:%d/article" % self.port, timeout=5)
        self.assertTrue(got["ok"])
        self.assertEqual(got["kind"], "article")
        self.assertEqual(got["comments"], [])

    async def test_comment_fetch_exception_is_swallowed(self):
        # 评论链路整体炸了（连接都建不起来）也必须只是 []，绝不冒泡
        with mock.patch("httpx.AsyncClient", side_effect=RuntimeError("boom")):
            got = await link_fetcher._fetch_video_comments(
                "https://www.bilibili.com/video/BV1xx", "页面", "bilibili", timeout=1)
        self.assertEqual(got, [])

    async def test_non_video_site_skips_comment_fetch(self):
        got = await link_fetcher._fetch_video_comments(
            "https://www.douyin.com/video/123", "页面", "", timeout=1)
        self.assertEqual(got, [])  # 站点未识别 → 直接空，不发任何请求


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ResearchPresearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_web_search_does_not_hide_research_abstract(self):
        research = SimpleNamespace(search=mock.AsyncMock(return_value={"status":"ok", "results":[{
            "title":"Growth mindset school trial", "url":"https://doi.org/10.example/test",
            "snippet":"A randomized school trial of growth mindset.",
            "body_text":"A randomized school trial of growth mindset. Effects depend on context.",
            "content_status":"fetched", "evidence_kind":"research_abstract"}]}))
        web = SimpleNamespace(search=mock.AsyncMock(side_effect=TimeoutError))
        result = await material_analyzer.build_external_context(
            "Mindset", kind="link", source_coverage={"kind":"video","body_available":True},
            analysis={"external_context_queries":[{"query":"成长性思维 growth mindset school trial", "purpose":"claim_verification"}]},
            body_text="作者讨论成长性思维。", search_tool=web, research_tool=research,
            page_fetcher=mock.AsyncMock(), enabled=True)
        self.assertEqual(result["verification_status"], "candidates_retrieved")
        self.assertIn("Effects depend on context", result["sources"][0]["body_text"])

    def test_crossref_abstract_parser_rejects_protocol_without_results(self):
        from agent.web_search_tool import WebSearchTool
        import json
        tool=WebSearchTool(provider="crossref")
        tool._request = mock.Mock(return_value=json.dumps({"message":{"items":[
            {"DOI":"10.example/protocol","title":["Mindset trial protocol"],"abstract":"A future study."},
            {"DOI":"10.example/results","title":["Mindset school trial"],"abstract":"<jats:p>Results depend on context.</jats:p>"}]}}))
        result=tool._search_crossref("成长性思维 growth mindset school trial")
        self.assertEqual(len(result["results"]),1)
        self.assertEqual(result["results"][0]["body_text"],"Results depend on context.")
        self.assertNotIn("成长性思维", tool._request.call_args.args[0])

    def test_query_plan_merge_keeps_verification_purpose(self):
        merged=material_analyzer._merge_deep([{"external_context_queries":[
            {"query":"Example background official","purpose":"background"},
            {"query":"growth mindset school trial","purpose":"claim_verification"}]}])
        self.assertEqual(merged["external_context_queries"][0]["purpose"],"claim_verification")
