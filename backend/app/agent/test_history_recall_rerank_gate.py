# -*- coding: utf-8 -*-
"""rerank 闸门单测（2026-10-22，plans/1022-01 点 1 第 3 步）。

闸门只做一件事：`_rerank_scores` 返回的分数低于 `MIN_SCORE` 的候选不返回。
所有失败路径（异常、超时、长度不符、没文本）都必须**放行**，不能吞掉回忆。
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import config
from memory import history_retrieval as hr


def _rows(*texts: str) -> list[dict]:
    return [{"summary": text, "candidate_id": f"c{index}"}
            for index, text in enumerate(texts)]


class RerankGateTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.retriever = hr.HybridHistoryRetriever.__new__(hr.HybridHistoryRetriever)
        self._patches = [
            patch.object(config, "HISTORY_RECALL_RERANK_GATE_ENABLED", True),
            patch.object(config, "HISTORY_RECALL_RERANK_MIN_SCORE", 0.74),
            patch.object(config, "HISTORY_RECALL_RERANK_TIMEOUT", 1.0),
            patch.object(config, "DASHSCOPE_API_KEY", "test-key"),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    async def test_disabled_returns_rows_without_calling_rerank(self) -> None:
        config.HISTORY_RECALL_RERANK_GATE_ENABLED = False
        rows = _rows("a", "b")
        fake = AsyncMock(return_value=[0.0, 0.0])
        with patch.object(hr, "_rerank_scores", fake):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual(kept, rows)
        fake.assert_not_awaited()

    async def test_keeps_rows_at_or_above_threshold(self) -> None:
        rows = _rows("a", "b", "c")
        with patch.object(hr, "_rerank_scores",
                          AsyncMock(return_value=[0.90, 0.74, 0.50])):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual([row["candidate_id"] for row in kept], ["c0", "c1"])

    async def test_all_below_threshold_returns_empty(self) -> None:
        rows = _rows("a", "b")
        with patch.object(hr, "_rerank_scores",
                          AsyncMock(return_value=[0.10, 0.20])):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual(kept, [])

    async def test_rerank_error_fails_open(self) -> None:
        rows = _rows("a", "b")
        with patch.object(hr, "_rerank_scores",
                          AsyncMock(side_effect=RuntimeError("boom"))):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual(kept, rows)

    async def test_timeout_fails_open(self) -> None:
        rows = _rows("a", "b")
        with patch.object(hr, "_rerank_scores",
                          AsyncMock(side_effect=asyncio.TimeoutError())):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual(kept, rows)

    async def test_score_length_mismatch_fails_open(self) -> None:
        rows = _rows("a", "b")
        with patch.object(hr, "_rerank_scores", AsyncMock(return_value=[0.9])):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual(kept, rows)

    async def test_empty_docs_fails_open_without_calling_rerank(self) -> None:
        rows = [{"candidate_id": "c0"}, {"candidate_id": "c1"}]
        fake = AsyncMock(return_value=[0.0, 0.0])
        with patch.object(hr, "_rerank_scores", fake):
            kept = await self.retriever._rerank_gate("q", rows)
        self.assertEqual(kept, rows)
        fake.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
