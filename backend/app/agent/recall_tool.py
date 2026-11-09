"""Bounded recall tool for the reply path.

The reply model may request this function through the OpenAI-compatible tool-call
protocol.  This module, not the model, performs the lookup: a read-only lexical
search over the learner's own past messages, returning a few short snippets.

Design constraints (from the 2026-10-19 probe, `reports/advice-vs-source-1020.md`):
  * Read-only.  This module never writes to memory and never calls another model,
    so enabling it adds no metering events and cannot pollute what Alice remembers.
  * User evidence only.  Snippets come from the learner's own messages, never from
    earlier assistant text, because replaying Alice's own words as "memory" is the
    self-echo failure the memory side already guards against.
  * Empty results must not read like a system report.  ``render`` returns an empty
    JSON array, not "nothing found"; the wording a learner hears lives in
    ``RECALL_PROMPT_STANCE``, because tool-result text such as "未找到相关记录"
    gets echoed into the reply and turns her into a search box.
  * Off unless asked.  ``enabled`` reads ``RECALL_TOOL_ENABLED`` (default false),
    so wiring this into a shared prompt is a no-op until the flag is set.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

try:  # reuse the repo's own stopword list so query words stay content-bearing
    from memory.entities import _ENTITY_SEARCH_STOPWORDS as _STOPWORDS
except Exception:  # pragma: no cover - import layout differs in some runners
    _STOPWORDS = {"什么", "怎么", "这个", "那个", "我们", "你们", "之前", "现在", "就是"}

EvidenceSearch = Callable[[str, str], Awaitable[List[Dict[str, Any]]]]


class RecallTool:
    """Read-only lookup of what the learner said before this session."""

    NAME = "recall_memory"
    ENV_FLAG = "RECALL_TOOL_ENABLED"
    # Neutral payload for "nothing matched": no "not found", no "records", no apology.
    EMPTY_RESULT = "[]"
    USER_PREFIXES = ("用户:", "用户：")

    def __init__(
        self,
        search: EvidenceSearch,
        *,
        max_results: int = 3,
        snippet_chars: int = 220,
        query_chars: int = 200,
        timeout_s: float = 2.0,
        min_term_hits: int = 2,
    ) -> None:
        if not callable(search):
            raise TypeError("RecallTool needs an async search callable")
        self._search = search
        self.max_results = max(1, min(5, int(max_results)))
        self.snippet_chars = max(40, min(400, int(snippet_chars)))
        self.query_chars = max(8, min(400, int(query_chars)))
        self.timeout_s = max(0.2, min(10.0, float(timeout_s)))
        self.min_term_hits = max(1, min(4, int(min_term_hits)))

    # ── wiring ────────────────────────────────────────────────────────────
    @classmethod
    def enabled(cls, env: Optional[Dict[str, str]] = None) -> bool:
        source = os.environ if env is None else env
        return str(source.get(cls.ENV_FLAG, "false")).strip().lower() in {"true", "1", "yes", "on"}

    CHUNKS_FLAG = "RECALL_CHUNKS_ENABLED"

    @classmethod
    def chunks_enabled(cls, env: Optional[Dict[str, str]] = None) -> bool:
        """Route episodic recall to message-level chunks (with the precision gate)
        instead of session-archive summaries. Default false: off = today's behavior."""
        source = os.environ if env is None else env
        return str(source.get(cls.CHUNKS_FLAG, "false")).strip().lower() in {"true", "1", "yes", "on"}

    @classmethod
    def from_memory_backend(cls, backend: Any, **kwargs: Any) -> "RecallTool":
        """Wire to a memory service exposing ``search_message_chunks``."""
        search = getattr(backend, "search_message_chunks", None)
        if not callable(search):
            raise TypeError("memory backend has no search_message_chunks")
        return cls(cast(EvidenceSearch, search), **kwargs)

    @classmethod
    def definition(cls) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.NAME,
                "description": (
                    "Look up what this learner told you in earlier conversations. "
                    "Call it when answering depends on details they shared before "
                    "(a plan, an event, a person, a worry). Do not call it for "
                    "ordinary chat, practice requests, or things they are saying now."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Short topical keywords for what to recall, e.g. the "
                                "thing itself plus one or two synonyms. Not a full sentence."
                            ),
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    # ── lookup ────────────────────────────────────────────────────────────
    async def search(self, user_id: str, query: str) -> Dict[str, Any]:
        """Return ``{"found", "evidence", "reason", "elapsed_ms"}``; never raises."""
        clean_query = " ".join(str(query or "").split())[: self.query_chars]
        started = time.monotonic()
        if not str(user_id or "").strip() or not clean_query:
            return self._result([], "empty_query", started)
        try:
            rows = await asyncio.wait_for(
                self._search(user_id, clean_query), timeout=self.timeout_s
            )
        except asyncio.TimeoutError:
            return self._result([], "timeout", started)
        except Exception:  # a lookup failure must never break the reply
            return self._result([], "search_failed", started)

        evidence: List[Dict[str, Any]] = []
        seen = set()
        words = self._query_words(clean_query)
        if not words:
            return self._result([], "query_not_specific", started)
        need = 1 if len(words) == 1 else min(self.min_term_hits, len(words))
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            when = self._when(row.get("created_at"))
            for snippet in self._user_snippets(row.get("summary")):
                matched = self._matched_terms(snippet, words)
                if len(matched) < need:
                    continue
                snippet = snippet[: self.snippet_chars].strip()
                if not snippet:
                    continue
                key = snippet.casefold()
                if key in seen:
                    continue
                seen.add(key)
                evidence.append({"when": when, "text": snippet, "matched": matched})
                if len(evidence) >= self.max_results:
                    break
            if len(evidence) >= self.max_results:
                break
        return self._result(evidence, "" if evidence else "no_match", started)

    def _result(self, evidence: List[Dict[str, str]], reason: str, started: float) -> Dict[str, Any]:
        return {
            "found": bool(evidence),
            "evidence": evidence,
            "reason": reason,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }

    # ── evidence shaping ──────────────────────────────────────────────────
    @staticmethod
    def _query_words(query: str) -> List[str]:
        """Content words of the model's query, used as a precision gate.

        The backend expands Chinese into two-character probes, so a query like
        ``英语计划`` also matches any message containing ``英语``.  A window that
        merely shares one broad probe is not evidence that the learner said the
        thing; feeding it back would have Alice assert a memory that never
        happened.  Single-word queries therefore must appear as a whole word,
        and multi-word queries must land at least ``min_term_hits`` of them.
        """
        words: List[str] = []
        seen = set()
        for token in re.split(r"[^0-9a-z\u4e00-\u9fff]+", str(query or "").casefold()):
            token = token.strip()
            if not token or token in seen:
                continue
            if re.fullmatch(r"[0-9a-z]+", token):
                if len(token) < 3:
                    continue
            elif len(token) < 2:
                continue
            if token in _STOPWORDS or token.isdigit():
                continue
            seen.add(token)
            words.append(token)
            if len(words) >= 8:
                break
        return words

    @staticmethod
    def _matched_terms(snippet: str, words: List[str]) -> List[str]:
        haystack = str(snippet or "").casefold()
        return [word for word in words if word in haystack]

    @classmethod
    def _user_snippets(cls, summary: Any) -> List[str]:
        """Keep only the learner's own segments, dropping assistant text."""
        body = str(summary or "")
        starts = [index for index in (body.find("用户:"), body.find("用户：")) if index >= 0]
        if not starts:
            return []
        body = body[min(starts):]
        snippets: List[str] = []
        for part in body.split(" / "):
            part = part.strip()
            prefix = next((p for p in cls.USER_PREFIXES if part.startswith(p)), "")
            if not prefix:
                continue
            text = " ".join(part[len(prefix):].split())
            if text:
                snippets.append(text)
        return snippets

    @staticmethod
    def _when(value: Any) -> str:
        text = str(value or "").strip()
        return text[:10] if len(text) >= 10 and text[4:5] == "-" else ""

    # ── rendering ─────────────────────────────────────────────────────────
    @classmethod
    def render(cls, result: Optional[Dict[str, Any]], *, snippet_chars: int = 220) -> str:
        """Bounded tool-result text. Empty lookup renders as an empty JSON array."""
        evidence = (result or {}).get("evidence") or []
        payload = [
            {"when": str(item.get("when") or ""), "text": str(item.get("text") or "")[:snippet_chars]}
            for item in evidence
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        return json.dumps(payload, ensure_ascii=False) if payload else cls.EMPTY_RESULT


RECALL_PROMPT_STANCE = (
    "你可以调用 recall_memory 查你和用户过去聊过的事。"
    "只要用户这句话提到你们之前聊过的某件事，就必须先调用 recall_memory 查一下再回答——"
    "哪怕没说“还记得”“上次”“之前”，比如“我那个麻烦事”“家里那个小家伙”“我那个新爱好”，"
    "也不许凭印象否认、换话题或反问对方在说什么。"
    "查询词给 2 到 4 个关键词，要包含同义词和这件事可能用到的说法"
    "（例如“麻烦事”要带上“房租 房东 涨租”，“新爱好”要带上“吉他 乐器”）。"
    "查不到时，你其实对这件事没有印象——像朋友那样自然地说你没什么印象，"
    "不要提“查了”“记录”“没找到”，默认不道歉，也不要求对方重讲一遍；"
    "接着顺着对方现在说的话聊，或只问一个具体的小问题。"
)
