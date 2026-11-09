"""Unit tests for the reply-path recall tool (read-only, flag-off by default)."""

from __future__ import annotations

import asyncio
import json
import unittest

from agent.recall_tool import RECALL_PROMPT_STANCE, RecallTool

LABEL = (
    "历史对话显示文本片段；用户气泡可能经过英文精修，不保证等于原始音频逐字内容。"
)


def window(user_text: str = "", ai_text: str = "", extra: list | None = None) -> dict:
    parts = []
    if user_text:
        parts.append(f"用户:{user_text}")
    if ai_text:
        parts.append(f"AI:{ai_text}")
    for item in extra or []:
        parts.append(item)
    return {"summary": f"{LABEL} " + " / ".join(parts), "created_at": "2026-10-01 12:00:00"}


class FakeSearch:
    def __init__(self, rows=None, *, delay: float = 0.0, boom: bool = False):
        self.rows = rows if rows is not None else []
        self.delay = delay
        self.boom = boom
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, user_id: str, query: str):
        self.calls.append((user_id, query))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.boom:
            raise RuntimeError("db down")
        return self.rows


class DefinitionTests(unittest.TestCase):
    def test_definition_is_openai_tool_shape(self):
        definition = RecallTool.definition()
        self.assertEqual(definition["type"], "function")
        function = definition["function"]
        self.assertEqual(function["name"], "recall_memory")
        self.assertEqual(function["parameters"]["required"], ["query"])
        self.assertFalse(function["parameters"]["additionalProperties"])
        self.assertIn("query", function["parameters"]["properties"])

    def test_disabled_unless_flag_says_so(self):
        self.assertFalse(RecallTool.enabled({}))
        self.assertFalse(RecallTool.enabled({"RECALL_TOOL_ENABLED": "false"}))
        self.assertFalse(RecallTool.enabled({"RECALL_TOOL_ENABLED": "maybe"}))
        for value in ("true", "TRUE", "1", "yes", "on"):
            self.assertTrue(RecallTool.enabled({"RECALL_TOOL_ENABLED": value}), value)


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_learner_segments_only(self):
        search = FakeSearch([window("房东说要涨房租 15%，我在跟他谈。", "I can help you write a reply.")])
        tool = RecallTool(search)
        result = await tool.search("user-1", "房租")
        self.assertTrue(result["found"])
        texts = [item["text"] for item in result["evidence"]]
        self.assertEqual(texts, ["房东说要涨房租 15%，我在跟他谈。"])
        self.assertNotIn("I can help", json.dumps(texts, ensure_ascii=False))
        self.assertEqual(result["evidence"][0]["when"], "2026-10-01")

    async def test_result_count_and_length_are_bounded(self):
        rows = [
            window(f"第{index}条关于房租的消息" + "长" * 600)
            for index in range(6)
        ]
        tool = RecallTool(FakeSearch(rows), max_results=2, snippet_chars=50)
        result = await tool.search("user-1", "房租")
        self.assertEqual(len(result["evidence"]), 2)
        for item in result["evidence"]:
            self.assertLessEqual(len(item["text"]), 50)

    async def test_rendered_payload_stays_small(self):
        rows = [window(f"房租第{index}次来说" + "长" * 900) for index in range(4)]
        tool = RecallTool(FakeSearch(rows), max_results=3, snippet_chars=220)
        payload = RecallTool.render(await tool.search("user-1", "房租"), snippet_chars=220)
        self.assertLessEqual(len(payload), 3 * 260)
        self.assertEqual(len(json.loads(payload)), 3)

    async def test_empty_lookup_is_neutral_and_not_a_report(self):
        tool = RecallTool(FakeSearch([]))
        result = await tool.search("user-1", "房租")
        self.assertFalse(result["found"])
        rendered = RecallTool.render(result)
        self.assertEqual(rendered, "[]")
        for tell in ("没找到", "没有找到", "记录", "查了", "搜索", "抱歉"):
            self.assertNotIn(tell, rendered)

    async def test_query_is_trimmed_and_missing_query_skips_the_lookup(self):
        search = FakeSearch([])
        tool = RecallTool(search, query_chars=20)
        await tool.search("user-1", "  房" + "租" * 100 + "  ")
        self.assertEqual(len(search.calls), 1)
        self.assertLessEqual(len(search.calls[0][1]), 20)
        self.assertNotIn("\n", search.calls[0][1])

        skipped = await tool.search("user-1", "   ")
        self.assertFalse(skipped["found"])
        self.assertEqual(skipped["reason"], "empty_query")
        self.assertEqual(len(search.calls), 1)
        await tool.search("", "房租")
        self.assertEqual(len(search.calls), 1)

    async def test_failures_never_escape_into_the_reply(self):
        boom = RecallTool(FakeSearch(boom=True))
        result = await boom.search("user-1", "房租")
        self.assertFalse(result["found"])
        self.assertEqual(result["reason"], "search_failed")
        self.assertEqual(RecallTool.render(result), "[]")

        slow = RecallTool(FakeSearch([window("房租")], delay=0.5), timeout_s=0.2)
        timed_out = await slow.search("user-1", "房租")
        self.assertFalse(timed_out["found"])
        self.assertEqual(timed_out["reason"], "timeout")

    async def test_bigram_noise_is_not_treated_as_evidence(self):
        # The backend probes Chinese by two-character shingles, so "英语计划"
        # also matches any message containing "英语". That alone is not evidence.
        tool = RecallTool(FakeSearch([window("我最近在学英语，挺费劲的。")]))
        self.assertFalse((await tool.search("user-1", "英语计划"))["found"])

    async def test_whole_word_match_is_accepted(self):
        tool = RecallTool(FakeSearch([window("英语计划我想改成每天写三句日记。")]))
        result = await tool.search("user-1", "英语计划 日记")
        self.assertTrue(result["found"])
        self.assertIn("日记", result["evidence"][0]["matched"])
        self.assertIn("英语计划", result["evidence"][0]["matched"])

    async def test_multi_word_query_needs_two_hits(self):
        tool = RecallTool(FakeSearch([window("我在学英语。")]))
        self.assertFalse((await tool.search("user-1", "英语 日记"))["found"])

    async def test_query_made_only_of_filler_words_skips_the_lookup(self):
        search = FakeSearch([window("什么都可以")])
        tool = RecallTool(search)
        result = await tool.search("user-1", "之前 那个 什么")
        self.assertFalse(result["found"])
        self.assertEqual(result["reason"], "query_not_specific")

    async def test_duplicate_snippets_are_not_repeated(self):
        rows = [window("我在跟房东谈房租。"), window("我在跟房东谈房租。")]
        tool = RecallTool(FakeSearch(rows))
        result = await tool.search("user-1", "房租")
        self.assertEqual(len(result["evidence"]), 1)

    async def test_from_memory_backend_requires_the_search_method(self):
        class Backend:
            async def search_message_chunks(self, user_id, query, limit=60):
                return [window("房租的事")]

        tool = RecallTool.from_memory_backend(Backend())
        self.assertTrue((await tool.search("user-1", "房租"))["found"])
        with self.assertRaises(TypeError):
            RecallTool.from_memory_backend(object())


class StanceTests(unittest.TestCase):
    def test_stance_tells_her_to_admit_without_a_service_voice(self):
        self.assertIn("没什么印象", RECALL_PROMPT_STANCE)
        self.assertIn("不道歉", RECALL_PROMPT_STANCE)
        self.assertIn("不要提", RECALL_PROMPT_STANCE)

    def test_render_is_the_only_thing_the_model_sees(self):
        empty = RecallTool.render({"found": False, "evidence": []})
        self.assertEqual(empty, RecallTool.EMPTY_RESULT)


if __name__ == "__main__":
    unittest.main()
