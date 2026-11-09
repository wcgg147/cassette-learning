"""Wiring tests for the flag-gated recall tool on the reply path.

Covers the contract that matters operationally:
  * flag off -> byte-for-byte the old call (no tools, usage metering intact);
  * flag on, model does not ask -> still one request;
  * flag on, model asks -> exactly one tool round, then a second request;
  * a failing lookup degrades to a plain reply instead of raising.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

from agent.recall_tool import RecallTool
from agent.tutor_agent import TutorAgent

LABEL = "历史对话显示文本片段；用户气泡可能经过英文精修，不保证等于原始音频逐字内容。"


class FakeBackend:
    def __init__(self, rows=None, *, boom: bool = False):
        self.rows = rows or []
        self.boom = boom
        self.calls = []

    async def search_message_chunks(self, user_id, query, limit=60):
        self.calls.append((user_id, query))
        if self.boom:
            raise RuntimeError("db down")
        return self.rows


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSelf:
    """Minimal stand-in: the helper only needs l2 and the metering parser."""

    _completion_content_and_metering = staticmethod(
        TutorAgent._completion_content_and_metering
    )

    def __init__(self, backend):
        self.l2 = backend


def window(text: str) -> dict:
    return {"summary": f"{LABEL} 用户:{text}", "created_at": "2026-10-01 12:00:00"}


def call(agent, *, user_id="user-1"):
    return TutorAgent._reply_completion_with_recall(
        agent, messages=[{"role": "user", "content": "聊聊之前说过的事"}],
        model="m", base_url="b", api_key="k", temperature=0.3, max_tokens=100,
        extra_body=None, timeout=5, user_id=user_id,
    )


class FlagOffTests(unittest.IsolatedAsyncioTestCase):
    async def test_flag_off_keeps_the_original_call(self):
        client = FakeClient([{"content": "hello", "metering": {
            "model": "m", "request_status": "success", "request_count": 1,
            "token_usage_status": "available"}}])
        with mock.patch.dict(os.environ, {"RECALL_TOOL_ENABLED": "false"}), \
                mock.patch("llm_client.robust_llm_client", client):
            content, metering, recall_meta = await call(FakeSelf(FakeBackend()))
        self.assertEqual(content, "hello")
        self.assertTrue(metering["token_usage_status"] == "available")
        self.assertFalse(recall_meta["enabled"])
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("tools", client.calls[0])
        self.assertTrue(client.calls[0]["return_usage"])


class FlagOnTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_offered_but_unused_costs_one_request(self):
        backend = FakeBackend([window("房租的事")])
        client = FakeClient([{"role": "assistant", "content": "reply text", "tool_calls": []}])
        with mock.patch.dict(os.environ, {"RECALL_TOOL_ENABLED": "true"}), \
                mock.patch("llm_client.robust_llm_client", client):
            content, metering, recall_meta = await call(FakeSelf(backend))
        self.assertEqual(content, "reply text")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tools"][0]["function"]["name"], RecallTool.NAME)
        self.assertEqual(client.calls[0]["tool_choice"], "auto")
        self.assertFalse(recall_meta["used"])
        self.assertEqual(backend.calls, [])
        self.assertEqual(metering["token_usage_status"], "unavailable")

    async def test_tool_used_runs_exactly_one_round(self):
        backend = FakeBackend([window("点外卖可以继续工作")])
        client = FakeClient([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": RecallTool.NAME,
                                          "arguments": json.dumps({"query": "工作"}, ensure_ascii=False)}}]},
            {"content": "记得一点——你说点外卖是为了继续工作。", "metering": {
                "model": "m", "request_status": "success", "request_count": 1,
                "token_usage_status": "available"}},
        ])
        with mock.patch.dict(os.environ, {"RECALL_TOOL_ENABLED": "true"}), \
                mock.patch("llm_client.robust_llm_client", client):
            content, metering, recall_meta = await call(FakeSelf(backend))
        self.assertIn("点外卖", content)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(backend.calls, [("user-1", "工作")])
        self.assertTrue(recall_meta["used"])
        self.assertTrue(recall_meta["found"])
        self.assertEqual(recall_meta["rounds"], 1)
        self.assertEqual(metering["request_count"], 2)
        follow_up = client.calls[1]["messages"]
        self.assertEqual(follow_up[-1]["role"], "tool")
        self.assertIn("点外卖", follow_up[-1]["content"])
        # the second request must not offer the tool again (no loops)
        self.assertNotIn("tools", client.calls[1])

    async def test_lookup_failure_degrades_to_a_plain_reply(self):
        client = FakeClient([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": RecallTool.NAME, "arguments": "{\"query\": \"工作\"}"}}]},
            {"content": "想不太起来了，你再说说？", "metering": {
                "model": "m", "request_status": "success", "request_count": 1,
                "token_usage_status": "available"}},
        ])
        with mock.patch.dict(os.environ, {"RECALL_TOOL_ENABLED": "true"}), \
                mock.patch("llm_client.robust_llm_client", client):
            content, _, recall_meta = await call(FakeSelf(FakeBackend(boom=True)))
        self.assertEqual(content, "想不太起来了，你再说说？")
        self.assertTrue(recall_meta["used"])
        self.assertFalse(recall_meta["found"])
        self.assertEqual(recall_meta["reason"], "search_failed")
        self.assertEqual(client.calls[1]["messages"][-1]["content"], "[]")

    async def test_malformed_arguments_do_not_break_the_turn(self):
        backend = FakeBackend([])
        client = FakeClient([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": RecallTool.NAME, "arguments": "not-json"}}]},
            {"content": "ok", "metering": {"model": "m", "request_status": "success",
                                           "request_count": 1, "token_usage_status": "available"}},
        ])
        with mock.patch.dict(os.environ, {"RECALL_TOOL_ENABLED": "true"}), \
                mock.patch("llm_client.robust_llm_client", client):
            content, _, recall_meta = await call(FakeSelf(backend))
        self.assertEqual(content, "ok")
        self.assertFalse(recall_meta["found"])
        self.assertEqual(recall_meta["reason"], "empty_query")


if __name__ == "__main__":
    unittest.main()
