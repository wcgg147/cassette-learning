"""Bounded web search tool for the conversation director.

The language model may request this function through the OpenAI-compatible
tool-call protocol.  This module, not the model, performs the network request.
Search queries use fixed provider endpoints. Optional known source URLs use
the separate bounded public HTTPS reader, never the material import pipeline.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple


class WebSearchTool:
    NAME = "web_search"
    _cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def __init__(
        self, *, provider: str = "bing_rss", api_key: str = "",
        timeout: int = 5, max_results: int = 5, cache_ttl: int = 1800,
    ) -> None:
        self.provider = (provider or "bing_rss").strip().lower()
        self.api_key = (api_key or "").strip()
        self.timeout = max(2, min(15, int(timeout)))
        self.max_results = max(1, min(8, int(max_results)))
        self.cache_ttl = max(60, min(86400, int(cache_ttl)))

    @classmethod
    def definition(cls) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.NAME,
                "description": (
                    "Search the public web only when a current, location-specific, "
                    "brand-specific, policy, regulation, or real-world process fact "
                    "is necessary to plan the next conversation step. Do not use it "
                    "for generic stable role-play flow or ordinary language teaching."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A concise search query containing the relevant place, "
                                "organization, process, and recency cue."
                            ),
                        },
                        "source_url": {"type": "string", "maxLength": 2048,
                            "description": "Optional exact public HTTPS source URL already known or supplied by the user. Use it to read the actual page; omit if unknown. Never invent a URL."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def explicit_page_url(user_text: str) -> str:
        if not re.search(r"\b(?:open|read|visit|check)\b|look at|打开|读取|读一下|看看", user_text or "", re.I):
            return ""
        matches = re.findall(r"https?://[^\s<>\"'()，。！？；：、]+|(?<![\w@])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}/[^\s<>\"'()，。！？；：、]*", user_text or "")
        urls = []
        for match in matches:
            value = match.rstrip(".,!?;:，。！？；：")
            if not value.startswith(("https://", "http://")):
                value = "https://" + value
            if value not in urls:
                urls.append(value)
        return urls[0] if len(urls) == 1 else ""

    @staticmethod
    def page_excerpts(text: str, query: str) -> List[Dict[str, Any]]:
        """Select bounded verbatim spans; relevance ranking is not verification."""
        words = set(re.findall(r"[a-z0-9]{3,}", str(query or "").lower())) - {
            "the", "and", "for", "please", "search", "tell", "what", "about", "https", "www", "com", "org", "official", "page"}
        positions = [m.start() for m in re.finditer(r"\b[a-z0-9]{3,}\b", text.lower()) if m.group() in words]
        candidates = {}
        for position in positions:
            start = max(0, position - 80)
            end = min(len(text), start + 280)
            span = text[start:end]
            score = sum(bool(re.search(r"\b" + re.escape(word) + r"\b", span, re.I)) for word in words)
            candidates[start] = (score, end)
        if not candidates and text.strip():
            candidates[0] = (0, min(280, len(text)))
        selected = []
        for start, (_, end) in sorted(candidates.items(), key=lambda row: (-row[1][0], row[0])):
            if any(start < row["source_end"] and end > row["source_start"] for row in selected):
                continue
            selected.append({"snippet": text[start:end], "source_start": start, "source_end": end})
            if len(selected) == 3:
                break
        return selected

    async def read_page(self, source_url: str, query: str) -> Dict[str, Any]:
        from .public_page_reader import read_public_page
        page = await read_public_page(source_url, timeout=self.timeout,
            resolver=os.getenv("PUBLIC_PAGE_READER_RESOLVER", "system"))
        record = {"query": self._normalize_query(query), "provider": "public_page_reader", "cached": False,
                  "status": page.get("status", "error"), "results": []}
        if page.get("status") != "ok":
            record["error_code"] = page.get("error_code", "page_read_failed")
            return record
        common = {key: page.get(key) for key in ("url", "title", "evidence_kind", "fetched_at", "text_truncated", "response_sha256")}
        common["title"] = str(common.get("title") or "")[:180]
        record["results"] = [{**common, "content_status": "fetched", **span}
                             for span in self.page_excerpts(page["text"], query)]
        record["page_metadata"] = {key: value for key, value in page.items() if key != "text"}
        record["page_metadata"]["title"] = common["title"]
        return record

    @staticmethod
    def page_model_content(record: Dict[str, Any]) -> str:
        """Bound page tool context without cutting serialized JSON mid-field."""
        payload = {key: value for key, value in record.items() if key != "page_metadata"}
        payload["results"] = list(record.get("results") or [])[:3]
        content = json.dumps(payload, ensure_ascii=False)
        while len(content) > 8000 and len(payload["results"]) > 1:
            payload["results"].pop()
            content = json.dumps(payload, ensure_ascii=False)
        return content

    async def search(self, query: str) -> Dict[str, Any]:
        normalized = self._normalize_query(query)
        if not normalized:
            return self._error("invalid_query", "The search query is empty.")

        cache_key = f"{self.provider}:{normalized.casefold()}:{self.max_results}"
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and cached[0] > now:
            result = dict(cached[1])
            result["cached"] = True
            return result

        try:
            if self.provider == "tavily":
                if not self.api_key:
                    return self._error("missing_api_key", "Tavily API key is not configured.")
                result = await asyncio.to_thread(self._search_tavily, normalized)
            elif self.provider == "bing_rss":
                result = await asyncio.to_thread(self._search_bing_rss, normalized)
            elif self.provider == "crossref":
                result = await asyncio.to_thread(self._search_crossref, normalized)
            else:
                return self._error("unsupported_provider", f"Unsupported provider: {self.provider}")
        except Exception as exc:
            result = self._error(type(exc).__name__[:64], "Web search request failed.")

        result["query"] = normalized
        result["provider"] = self.provider
        result["cached"] = False
        if result.get("status") == "ok":
            self._cache[cache_key] = (now + self.cache_ttl, dict(result))
        return result

    def _search_bing_rss(self, query: str) -> Dict[str, Any]:
        params = urllib.parse.urlencode({"q": query, "format": "rss", "setlang": "en-US"})
        body = self._request(
            f"https://www.bing.com/search?{params}",
            headers={"Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8"},
        )
        return {"status": "ok", "results": self._parse_bing_rss(body, self.max_results)}

    def _search_crossref(self, query: str) -> Dict[str, Any]:
        # A research query may retain a Chinese source anchor for grounding.
        # Use its English scientific terms for the international paper index.
        english = " ".join(re.findall(r"[A-Za-z][A-Za-z0-9'-]*", query))
        search_query = english if len(english.split()) >= 2 else query
        params = urllib.parse.urlencode({
            "query.bibliographic": search_query, "rows": self.max_results,
            "filter": "type:journal-article,has-abstract:true",
            "select": "DOI,title,abstract,URL,published,container-title,type",
        })
        for attempt in range(2):
            try:
                data = json.loads(self._request(
                    "https://api.crossref.org/works?" + params,
                    headers={"Accept": "application/json"}))
                break
            except urllib.error.HTTPError as exc:
                if attempt or exc.code not in (500, 502, 503, 504):
                    raise
                time.sleep(0.2)

        results = []
        for item in (data.get("message", {}).get("items") or [])[:self.max_results]:
            title = self._clean_text(" ".join(item.get("title") or []), 180)
            abstract = self._clean_text(item.get("abstract"), 12000)
            doi = str(item.get("DOI") or "").strip()
            if (not doi or not abstract or re.search(
                    r"\b(?:protocol|retraction|withdrawal|correction|erratum)\b", title, re.I)):
                continue
            results.append({
                "title": title, "url": "https://doi.org/" + urllib.parse.quote(doi, safe="/"),
                "snippet": abstract[:500], "body_text": abstract,
                "content_status": "fetched", "evidence_kind": "research_abstract",
                "doi": doi, "published": item.get("published"),
                "journal": " ".join(item.get("container-title") or [])[:180],
            })
        return {"status": "ok", "results": results}

    def _search_tavily(self, query: str) -> Dict[str, Any]:
        payload = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }).encode("utf-8")
        body = self._request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        decoded = json.loads(body)
        results = []
        for item in (decoded.get("results") or [])[:self.max_results]:
            url = self._safe_http_url(str(item.get("url") or ""))
            if not url:
                continue
            results.append({
                "title": self._clean_text(item.get("title"), 180),
                "url": url,
                "snippet": self._clean_text(item.get("content"), 500),
            })
        return {"status": "ok", "results": results}

    def _request(
        self, url: str, *, data: bytes | None = None,
        headers: Dict[str, str] | None = None,
    ) -> str:
        request_headers = {
            "User-Agent": "LanguageTutorDirector/1.0 (+bounded-web-search)",
            **(headers or {}),
        }
        request = urllib.request.Request(
            url, data=data, headers=request_headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(512_000)
            charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")

    @classmethod
    def _parse_bing_rss(cls, body: str, max_results: int) -> List[Dict[str, str]]:
        root = ET.fromstring(body)
        results: List[Dict[str, str]] = []
        for item in root.findall(".//item"):
            url = cls._safe_http_url(item.findtext("link") or "")
            if not url:
                continue
            results.append({
                "title": cls._clean_text(item.findtext("title"), 180),
                "url": url,
                "snippet": cls._clean_text(item.findtext("description"), 500),
            })
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _normalize_query(query: Any) -> str:
        value = " ".join(str(query or "").split())
        value = re.sub(r"[\x00-\x1f\x7f]", "", value)
        return value[:240]

    @staticmethod
    def _clean_text(value: Any, limit: int) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        return " ".join(text.split())[:limit]

    @staticmethod
    def _safe_http_url(value: str) -> str:
        url = value.strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return url[:1000]

    @staticmethod
    def _error(code: str, message: str) -> Dict[str, Any]:
        return {
            "status": "error", "error_code": code,
            "message": message, "results": [],
        }
