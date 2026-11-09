# -*- coding: utf-8 -*-
"""本地文件 / 图片导入 + 深度分析层（plans/0916-24 V1.2）的确定性回归测试。

覆盖三块，全部离线（不调真模型、不连库、不碰真实用户数据）：
1. `engine/file_parser`：类型判定（扩展名优先、MIME 兜底）、解码降级、
   真 docx/epub 解析、**失败必须给明确中文原因**（§7-1 底线：解不出来不产空带）；
2. `engine/material_analyzer` 的深析层：六字段解析（含代码块/夹带文本/坏 JSON）、
   缺项空项不报错、**失败返回 {} 绝不抛**（增强失败不能连累导入）；
3. 端点：`/materials/{user_id}/file` 的文档与图片分派、失败路径（200 + ok=false）。

跑法：cd chanpinzhujian/api && python -m unittest test_runner.test_material_file -v
"""

from __future__ import annotations

import asyncio
import io
import sys
import unittest
import unittest.mock
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from engine import file_parser, material_analyzer  # noqa: E402


# ── 造真文件的最小构造器（不是假二进制：要真能过解析器）──────────────────

def make_docx(paragraphs, table_rows=None) -> bytes:
    import docx
    document = docx.Document()
    for line in paragraphs:
        document.add_paragraph(line)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, cell in enumerate(row):
                table.cell(r, c).text = cell
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def make_epub(chapters) -> bytes:
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier("zz-selftest-epub")
    book.set_title("ZZ Self Test")
    items = []
    for i, html in enumerate(chapters):
        item = epub.EpubHtml(title="ch%d" % i, file_name="ch%d.xhtml" % i, lang="en")
        item.content = "<html><body>%s</body></html>" % html
        book.add_item(item)
        items.append(item)
    book.toc = items
    book.spine = ["nav"] + items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    buf = io.BytesIO()
    epub.write_epub(buf, book)
    return buf.getvalue()


class KindOfTests(unittest.TestCase):
    """类型判定：用户手里的文件名比浏览器给的 MIME 可信。"""

    def test_extension_wins_over_generic_mime(self):
        self.assertEqual(file_parser.kind_of("notes.docx", "application/octet-stream"), "docx")
        self.assertEqual(file_parser.kind_of("photo.JPG", ""), "image")

    def test_mime_fallback_when_no_extension(self):
        self.assertEqual(file_parser.kind_of("", "application/pdf"), "pdf")
        self.assertEqual(file_parser.kind_of("download", "image/png"), "image")

    def test_unknown_returns_empty(self):
        for name in ("a.mov", "b.zip", "c.mp4", ""):
            self.assertEqual(file_parser.kind_of(name, "application/zip"), "", name)


class ExtractTests(unittest.TestCase):
    def test_text_prefers_utf8_then_gbk(self):
        self.assertEqual(file_parser.extract_text("你好，世界".encode("utf-8"),
                                                 "a.txt")["text"], "你好，世界")
        # 中文用户的 txt 里 GBK 很常见：不能整份失败，要降级解出来
        got = file_parser.extract_text("你好，世界".encode("gb18030"), "a.txt")
        self.assertTrue(got["ok"])
        self.assertIn("你好", got["text"])

    def test_utf8_bom_stripped(self):
        got = file_parser.extract_text("\ufeffThe Bear".encode("utf-8"), "a.md")
        self.assertEqual(got["text"], "The Bear")

    def test_docx_paragraphs_and_tables(self):
        data = make_docx(["Carmy runs the Beef.", "Sydney is the sous chef."],
                         table_rows=[["dish", "price"], ["short rib", "18"]])
        got = file_parser.extract_text(data, "notes.docx")
        self.assertTrue(got["ok"])
        self.assertIn("Carmy runs the Beef.", got["text"])
        # 菜单/对照表常常才是素材重点，表格内容必须进正文
        self.assertIn("short rib | 18", got["text"])

    def test_epub_chapters_joined_without_tags(self):
        data = make_epub(["<p>The first chapter.</p>", "<p>The second chapter.</p>"])
        got = file_parser.extract_text(data, "book.epub")
        self.assertTrue(got["ok"])
        self.assertIn("The first chapter.", got["text"])
        self.assertIn("The second chapter.", got["text"])
        self.assertNotIn("<p>", got["text"])

    def test_broken_docx_gets_a_readable_reason(self):
        got = file_parser.extract_text(b"not really a docx", "fake.docx")
        self.assertFalse(got["ok"])
        self.assertIn("打不开", got["error"])
        self.assertEqual(got["text"], "")

    def test_empty_file_rejected(self):
        got = file_parser.extract_text(b"", "empty.txt")
        self.assertFalse(got["ok"])
        self.assertIn("空", got["error"])

    def test_oversize_rejected_before_parsing(self):
        got = file_parser.extract_text(b"x" * (file_parser.MAX_FILE_BYTES + 1), "big.txt")
        self.assertFalse(got["ok"])
        self.assertIn("太大", got["error"])

    def test_text_file_with_no_readable_content(self):
        # 只有空白 → 不行（"解析成功但一个字都没有"对用户和模型都没意义）
        got = file_parser.extract_text(b"   \n\n  \t ", "blank.md")
        self.assertFalse(got["ok"])
        self.assertNotEqual(got["error"], "")

    def test_image_is_not_a_document_failure(self):
        # 图片不是"解析失败"，是"不走这条路"——分派由端点负责
        got = file_parser.extract_text(b"\x89PNG\r\n", "shot.png")
        self.assertFalse(got["ok"])
        self.assertEqual(got["kind"], "image")

    def test_garbage_pdf_says_something_actionable(self):
        got = file_parser.extract_text(b"%PDF-1.4 garbage", "broken.pdf")
        self.assertFalse(got["ok"])
        self.assertNotEqual(got["error"], "")


class DeepAnalysisTests(unittest.TestCase):
    """深析层：产物是**默认框架**（缺项空项正常），解析要抗模型的花式输出。"""

    def test_plain_json_parsed(self):
        raw = ('{"topics": ["菜单之争"], "people": ["Carmy — 主厨"], '
               '"viewpoints": ["老主顾觉得被抛下"], "hooks": ["餐厅该不该升级"]}')
        got = material_analyzer._parse_deep_json(raw)
        self.assertEqual(got["topics"], ["菜单之争"])
        self.assertEqual(got["people"], ["Carmy — 主厨"])
        self.assertEqual(got["hooks"], ["餐厅该不该升级"])
        self.assertNotIn("timeline", got)   # 没给的字段不硬凑

    def test_code_fence_and_prose_wrapped(self):
        raw = '下面是我的分析：\n```json\n{"gaps": ["第一集那个电话是谁打的"]}\n```\n希望有用。'
        self.assertEqual(material_analyzer._parse_deep_json(raw)["gaps"],
                         ["第一集那个电话是谁打的"])

    def test_string_instead_of_list_tolerated(self):
        got = material_analyzer._parse_deep_json('{"topics": "只有一个主题"}')
        self.assertEqual(got["topics"], ["只有一个主题"])

    def test_bad_json_returns_empty(self):
        for raw in ("", "无", "{坏掉的 json", "[1,2,3]", "null"):
            self.assertEqual(material_analyzer._parse_deep_json(raw), {}, raw)

    def test_items_deduped_trimmed_and_capped(self):
        raw = '{"topics": ["  同一个 ", "同一个", "", "%s"]}' % ("长" * 200)
        got = material_analyzer._parse_deep_json(raw)
        self.assertEqual(got["topics"][0], "同一个")
        self.assertEqual(len(got["topics"]), 2)
        self.assertLessEqual(len(got["topics"][1]), 80)

    def test_short_material_skips_deep_call(self):
        # 太短的素材不值得花一次"深读"调用（判据是确定性的，不调模型）
        self.assertFalse(material_analyzer.needs_deep_analysis("短短一句"))
        self.assertTrue(material_analyzer.needs_deep_analysis("字" * 1300))

    def test_analyze_deep_survives_model_failure(self):
        async def _boom(*a, **kw):
            raise RuntimeError("model down")
        fake = unittest.mock.Mock()
        fake.chat_completion = _boom
        with unittest.mock.patch("llm_client.robust_llm_client", fake):
            got = asyncio.run(material_analyzer.analyze_deep("字" * 1300, kind="text"))
        self.assertEqual(got, {})   # 增强失败不能让导入失败

    def test_comments_go_into_the_analysis_input(self):
        seen = {}

        async def _capture(messages, **kw):
            seen["prompt"] = messages[0]["content"]
            return '{"viewpoints": ["观众觉得结局仓促"]}'
        fake = unittest.mock.Mock()
        fake.chat_completion = _capture
        with unittest.mock.patch("llm_client.robust_llm_client", fake):
            got = asyncio.run(material_analyzer.analyze_deep(
                "字" * 1300, kind="link", title="某视频",
                comments=["这个结局太仓促了"]))
        self.assertEqual(got["viewpoints"], ["观众觉得结局仓促"])
        self.assertIn("这个结局太仓促了", seen["prompt"])
        self.assertIn("观众评论", seen["prompt"])


class ImageAnalysisTests(unittest.TestCase):
    def test_image_mime_from_extension_or_content_type(self):
        self.assertEqual(material_analyzer.image_mime("a.webp", ""), "image/webp")
        self.assertEqual(material_analyzer.image_mime("a", "image/jpeg"), "image/jpeg")
        self.assertEqual(material_analyzer.image_mime("a", ""), "image/png")

    def test_parse_image_json_with_ocr_and_description(self):
        got = material_analyzer._parse_image_json(
            '{"ocr_text": "第 3 季 第 2 集", "description": "一张剧照", "entities": ["The Bear"]}')
        self.assertEqual(got["ocr_text"], "第 3 季 第 2 集")
        self.assertEqual(got["entities"], ["The Bear"])

    def test_unstructured_output_still_usable(self):
        # 模型没给 JSON：把原文当描述，总比"图里什么都没有"对用户有用
        got = material_analyzer._parse_image_json("这看起来是一张菜单照片。")
        self.assertEqual(got["description"], "这看起来是一张菜单照片。")

    def test_empty_output_returns_empty(self):
        self.assertEqual(material_analyzer._parse_image_json(""), {})

    def test_oversize_image_rejected_without_calling_model(self):
        got = asyncio.run(material_analyzer.analyze_image(
            b"x" * (material_analyzer.MAX_IMAGE_BYTES + 1), filename="big.png"))
        self.assertFalse(got["ok"])
        self.assertIn("太大", got["error"])

    def test_model_failure_is_explicit(self):
        async def _boom(*a, **kw):
            raise RuntimeError("vision down")
        fake = unittest.mock.Mock()
        fake.chat_completion = _boom
        with unittest.mock.patch("llm_client.robust_llm_client", fake):
            got = asyncio.run(material_analyzer.analyze_image(b"\x89PNG", filename="a.png"))
        self.assertFalse(got["ok"])
        self.assertIn("读图失败", got["error"])
        self.assertEqual(got["text"], "")


class EntitiesTests(unittest.TestCase):
    """流水线③：深析产物 → memory_entities 行（"你导入过这部剧"要成为可引用的记忆）。"""

    def test_people_split_into_name_and_detail(self):
        rows = main._material_entities({"people": ["Carmy — 餐厅主厨", "Sydney：副主厨"]})
        self.assertEqual(rows[0], {"etype": "person", "name": "Carmy", "detail": "餐厅主厨"})
        self.assertEqual(rows[1]["name"], "Sydney")

    def test_plain_name_kept_whole(self):
        rows = main._material_entities({"people": ["The Bear"]})
        self.assertEqual(rows[0]["name"], "The Bear")
        self.assertEqual(rows[0]["detail"], "")

    def test_image_entities_are_things(self):
        rows = main._material_entities({}, extra=["The Bear", "Chicago"])
        self.assertEqual([r["etype"] for r in rows], ["thing", "thing"])

    def test_topics_never_become_entities(self):
        # topics 是话题不是实体：塞进实体表会把 F1 的兴趣取材灌成素材目录
        rows = main._material_entities({"topics": ["菜单之争", "厨房文化"]})
        self.assertEqual(rows, [])

    def test_empty_and_duplicate_input(self):
        self.assertEqual(main._material_entities({}), [])
        rows = main._material_entities({"people": ["A", "A", "", None]})
        self.assertEqual(len(rows), 1)


class FileEndpointTests(unittest.TestCase):
    """端点分派与失败契约（打桩 l2 + 模型，不碰真库真模型）。"""

    class FakeL2:
        def __init__(self):
            self.materials = []
            self.entities = []

        async def save_conversation(self, *a, **kw):
            return True

        async def add_material(self, user_id, kind, title, **kw):
            self.materials.append({"kind": kind, "title": title, **kw})
            return len(self.materials)

        async def save_entities(self, user_id, conv_id, entities):
            self.entities.extend(entities)
            return len(entities)

    def assert_no_tape(self, fake):
        """§7 底线：失败不建带。审计留痕行（status='failed'）必须有，带必须没有。"""
        self.assertTrue(fake.materials, "失败也要留一行明确原因给前端提示")
        for row in fake.materials:
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["material_conv_id"], "", "失败件绝不能带 conv_id")
        self.assertEqual(fake.entities, [])

    def call_file(self, data: bytes, filename: str, content_type: str = "",
                  l2=None, deep=None):
        class _Upload:
            def __init__(self, blob, name, ctype):
                self._blob, self.filename, self.content_type = blob, name, ctype

            async def read(self):
                return self._blob

        fake_l2 = l2 or self.FakeL2()

        async def _run():
            async def _deep(*a, **kw):
                return deep if deep is not None else {}
            with unittest.mock.patch.object(main, "l2", fake_l2), \
                 unittest.mock.patch.object(material_analyzer, "analyze_deep", _deep):
                return await main.import_material_file(
                    user_id="zz-selftest-u1",
                    file=_Upload(data, filename, content_type),
                    current_user_id="zz-selftest-u1")
        return asyncio.run(_run()), fake_l2

    def test_text_file_becomes_a_tape(self):
        result, fake = self.call_file(b"The Bear is a show about a chef.",
                                      "the-bear.txt")
        self.assertTrue(result["ok"])
        self.assertEqual(result["kind"], "file")
        self.assertTrue(result["conv_id"].endswith("_A"))
        self.assertEqual(fake.materials[0]["kind"], "file")
        self.assertIn("chef", fake.materials[0]["raw_text"])

    def test_docx_file_goes_through_the_parser(self):
        data = make_docx(["Sydney argues about the menu."])
        result, fake = self.call_file(data, "notes.docx")
        self.assertTrue(result["ok"])
        self.assertIn("Sydney", fake.materials[0]["raw_text"])

    def test_broken_file_is_ok_false_with_reason_and_no_tape(self):
        result, fake = self.call_file(b"nope", "broken.docx")
        self.assertFalse(result["ok"])
        self.assertIn("打不开", result["error"])
        self.assert_no_tape(fake)   # §7 底线：不建空带（留痕行 status='failed' 是有意的）

    def test_image_is_dispatched_to_vision(self):
        async def _vision(data, filename="", content_type=""):
            return {"ok": True, "text": "一张菜单照片\n\n图中文字：\nSHORT RIB 18",
                    "summary": "一张菜单照片", "entities": ["Short Rib"], "error": ""}
        with unittest.mock.patch.object(material_analyzer, "analyze_image", _vision):
            result, fake = self.call_file(b"\x89PNG fake", "menu.png")
        self.assertTrue(result["ok"])
        self.assertEqual(result["kind"], "image")
        self.assertIn("SHORT RIB", fake.materials[0]["raw_text"])
        # 视觉调用已经说清"这是什么"，不再花一次摘要；实体照挂（流水线③）
        self.assertEqual(fake.materials[0]["summary"], "一张菜单照片")
        self.assertEqual(fake.entities[0]["name"], "Short Rib")

    def test_failed_vision_says_why_and_builds_no_tape(self):
        async def _vision(data, filename="", content_type=""):
            return {"ok": False, "text": "", "summary": "", "entities": [],
                    "error": "这张图没能读出内容，换一张更清楚的试试"}
        with unittest.mock.patch.object(material_analyzer, "analyze_image", _vision):
            result, fake = self.call_file(b"\x89PNG", "blurry.png")
        self.assertFalse(result["ok"])
        self.assertIn("换一张", result["error"])
        self.assert_no_tape(fake)

    def test_deep_analysis_is_stored_and_entities_attached(self):
        result, fake = self.call_file(
            b"The Bear is a show about Carmy and Sydney.",
            "the-bear.txt",
            deep={"people": ["Carmy — 主厨", "Sydney — 副主厨"],
                  "hooks": ["餐厅要不要升级"]})
        self.assertEqual(result["analysis"]["hooks"], ["餐厅要不要升级"])
        self.assertEqual(fake.materials[0]["analysis"]["people"][0], "Carmy — 主厨")
        self.assertEqual(sorted(e["name"] for e in fake.entities), ["Carmy", "Sydney"])

    def test_unsupported_type_is_rejected_clearly(self):
        result, fake = self.call_file(b"movie bytes", "clip.mp4")
        self.assertFalse(result["ok"])
        self.assertIn("读不了", result["error"])
        self.assert_no_tape(fake)


if __name__ == "__main__":
    unittest.main(verbosity=2)
