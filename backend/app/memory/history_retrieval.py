"""Bounded hybrid retrieval for cross-cassette learner memory.

The live voice path must not query long-term memory on every turn.  This
module is therefore used only after ``MemoryOrchestrator`` has selected an
episodic-recall route.  It combines a deterministic multilingual BM25 signal
with an optional semantic embedding signal and degrades to the local signal
when the embedding provider is unavailable or slow.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Any, Iterable


_ENGLISH_RECALL_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "as", "at", "by", "for", "from",
    "in", "into", "of", "on", "to", "with", "about", "regarding", "do", "does",
    "did", "have", "has", "had", "are", "is", "was", "were", "you", "your", "we",
    "our", "i", "me", "my", "he", "she", "they", "it", "this", "that", "what",
    "when", "where", "who", "why", "how", "remember", "recall", "talk", "talked",
    "chat", "chatted", "discuss", "discussed", "mention", "mentioned", "say", "said",
    "tell", "told", "earlier", "previous", "ago",
})
_CHINESE_RECALL_STOP_PHRASES = (
    "你还记得", "还记得", "记得吗", "记得", "我们之前聊过", "之前聊过", "聊过",
    "我们聊过", "我们之前说过", "之前说过", "说过", "谈过", "聊了什么", "谈了什么",
    "之前", "以前", "我们", "你们", "关于", "那个", "这个", "什么", "有没有",
)


def lexical_tokens(text: str) -> list[str]:
    """Return stable English tokens and Chinese character n-grams.

    Character n-grams avoid requiring a Chinese segmentation dependency and
    are also tolerant of many ASR spacing and punctuation differences.
    """

    lowered = (text or "").lower()
    english = re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", lowered)
    english = [
        token for token in english
        if len(token) > 1 and token not in _ENGLISH_RECALL_STOPWORDS
    ]
    english_bigrams = [
        f"{english[index]}_{english[index + 1]}"
        for index in range(max(0, len(english) - 1))
    ]
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    for phrase in _CHINESE_RECALL_STOP_PHRASES:
        chinese = chinese.replace(phrase, "")
    chinese = re.sub(r"(?:你|我|我们|你们|他|她|它|吗|呢|吧|啊|呀)+$", "", chinese)
    chinese_grams = [
        chinese[index:index + size]
        for size in (2, 3)
        for index in range(max(0, len(chinese) - size + 1))
    ]
    return english + english_bigrams + chinese_grams


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return 0.0
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def rank_history_sparse(
    candidates: Iterable[dict], query: str, top_k: int = 5
) -> list[dict]:
    """Rank bounded history with BM25 plus exact-overlap reliability signals."""

    rows = [dict(item) for item in candidates]
    query_tokens = lexical_tokens(query)
    if not rows or not query_tokens:
        return []

    docs = [lexical_tokens(str(row.get("searchable") or row.get("summary") or "")) for row in rows]
    counters = [Counter(tokens) for tokens in docs]
    doc_lengths = [max(1, len(tokens)) for tokens in docs]
    avg_len = sum(doc_lengths) / max(1, len(doc_lengths))
    query_terms = set(query_tokens)
    document_frequency = {
        term: sum(1 for counter in counters if term in counter)
        for term in query_terms
    }
    raw_bm25: list[float] = []
    k1, b = 1.2, 0.75
    for counter, doc_len in zip(counters, doc_lengths):
        score = 0.0
        for term in query_terms:
            frequency = counter.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (len(rows) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * doc_len / avg_len)
            score += idf * (frequency * (k1 + 1.0)) / denominator
        raw_bm25.append(score)

    max_bm25 = max(raw_bm25, default=0.0) or 1.0
    newest = max((_timestamp(row.get("created_at")) for row in rows), default=0.0)
    normalized_query = re.sub(r"\s+", "", (query or "").lower())
    ranked: list[dict] = []
    for row, doc_tokens, bm25 in zip(rows, docs, raw_bm25):
        doc_terms = set(doc_tokens)
        overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
        normalized_body = re.sub(
            r"\s+", "", str(row.get("searchable") or row.get("summary") or "").lower()
        )
        exact_bonus = 0.12 if len(normalized_query) >= 3 and normalized_query in normalized_body else 0.0
        age_seconds = max(0.0, newest - _timestamp(row.get("created_at")))
        recency_bonus = 0.05 * math.exp(-age_seconds / (90.0 * 86400.0)) if newest else 0.0
        handoff_bonus = 0.05 if row.get("source") == "cassette_handoff" else 0.0
        score = 0.62 * (bm25 / max_bm25) + 0.21 * overlap + exact_bonus + recency_bonus + handoff_bonus
        if bm25 <= 0 and overlap <= 0:
            continue
        row["sparse_score"] = round(score, 6)
        row["score"] = round(score, 6)
        row["lexical_overlap"] = round(overlap, 6)
        # 原始 BM25（已含 IDF），供绝对证据门槛用；归一化分数会掩盖弱池子。
        row["sparse_raw"] = round(bm25, 6)
        ranked.append(row)

    ranked.sort(
        key=lambda item: (item.get("sparse_score", 0.0), _timestamp(item.get("created_at"))),
        reverse=True,
    )
    return _dedupe(ranked)[: max(1, int(top_k))]


def _self_echo_ratio(query: str, candidate: str) -> float:
    """用户这句话的字符二元组，有多大比例也出现在候选里。

    用来识别「候选就是用户这句原话」（§8.23 实测记忆里 11/11 都是这样）。
    判的是**整句重合**：旧记录里提一句"脚踝"不会让比例冲到 0.95。
    """
    def pairs(text: str) -> set[str]:
        clean = "".join(ch for ch in str(text or "").lower() if ch.isalnum())
        return {clean[i:i + 2] for i in range(len(clean) - 1)}

    query_pairs = pairs(query)
    if not query_pairs:
        return 0.0
    return len(query_pairs & pairs(candidate)) / len(query_pairs)


def _is_archive_row(row: dict) -> bool:
    """这条候选是不是"一场对话的归档摘要"（区别于逐条消息块／回放块）。"""
    source = str(row.get("source") or "")
    if source in ("session_archive", "cassette_handoff"):
        return True
    return bool(row.get("summary")) and not row.get("seq") and not row.get("role")


def _candidate_text(row: dict) -> str:
    for key in ("searchable", "summary", "content", "text"):
        value = row.get(key)
        if value:
            return str(value)
    return ""

_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
_RERANK_MODEL = "qwen3.7-text-rerank"


async def _rerank_scores(query: str, docs: list[str], api_key: str) -> list[float]:
    """qwen rerank：返回与 docs 等长的 relevance_score。失败抛异常，由调用方放行。"""
    import httpx

    body = {
        "model": _RERANK_MODEL,
        "input": {"query": query, "documents": docs},
        "parameters": {"return_documents": False},
    }
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.post(
            _RERANK_URL, json=body,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
    results = (data.get("output", {}).get("results")
               or data.get("results") or [])
    scores = [0.0] * len(docs)
    for item in results:
        index = int(item.get("index") or 0)
        if 0 <= index < len(scores):
            scores[index] = float(item.get("relevance_score") or 0.0)
    return scores


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _dedupe(rows: Iterable[dict]) -> list[dict]:
    result = []
    seen = set()
    for row in rows:
        key = str(row.get("candidate_id") or row.get("session_id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(row)
    return result


# ── 被 embed 的文本：剔掉「结构性包装」（plans/1017-02 §8.5 / M2b）──────────
# 为什么需要：入库文本长这样 —— `场景=…；角色=…；当前步骤=… 用户:… / AI:…`。
# 那套字段标签**每一行都有、且与内容无关**，进向量就把所有旧行的向量往一处拉，
# 稀释判别力。台架实测：同一对语义相似度 0.553 → 加了包装后 0.464，
# 掉到稠密独中的 0.48 门以下 ⇒ 本该召回的那条被丢掉。
# 这里只处理**喂给 embedding 的那份文本**，`searchable` 本身一字不动
# （稀疏打分、注入 prompt、导出全不受影响）。
_EMBED_BOILERPLATE_RE = re.compile(
    r"历史对话显示文本片段[；;][^。]*。?"
)
_EMBED_LABEL_RE = re.compile(
    r"(?:旧磁带原始片段=|最近对话=|场景=|角色=|当前步骤=|目标=|下一动作=|待处理=|已完成=)"
)
_EMBED_ROLE_RE = re.compile(r"(?:用户|AI)\s*[:：]")


def embedding_surface(text: str, limit: int = 1000) -> str:
    """把入库文本剥成"只剩内容"：去样板句、去字段标签、去角色标记。

    标签去掉、**值留下**（`场景=咖啡店` → `咖啡店`）——值是有意义的语义内容，
    标签不是。空串 ⇒ 返回空串，由调用方回落原始文本。
    """
    value = str(text or "")
    if not value:
        return ""
    value = _EMBED_BOILERPLATE_RE.sub(" ", value)
    value = _EMBED_LABEL_RE.sub(" ", value)
    value = _EMBED_ROLE_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit] if value else ""


def build_message_chunk_candidates(
    rows: Iterable[dict], query: str, limit: int = 60
) -> list[dict]:
    """Build bounded, query-centered windows from matched archived messages.

    ``conversation_messages.content`` is the text saved for display. A user
    bubble may contain polished ASR, so these windows are useful for recalling
    meaning but must not be represented as verbatim audio evidence.
    """
    windows: dict[tuple[str, int], dict[int, dict]] = {}
    scores: dict[tuple[str, int], int] = {}
    timestamps: dict[tuple[str, int], Any] = {}
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        conv_id = str(raw.get("conv_id") or "").strip()
        try:
            seed_seq = int(raw.get("seed_seq"))
            seq = int(raw.get("seq"))
        except (TypeError, ValueError):
            continue
        if not conv_id:
            continue
        key = (conv_id, seed_seq)
        windows.setdefault(key, {})[seq] = dict(raw)
        scores[key] = max(scores.get(key, 0), int(raw.get("match_score") or 0))
        timestamps[key] = raw.get("seed_created_at") or raw.get("created_at")

    ordered = sorted(
        windows,
        key=lambda key: (scores.get(key, 0), _timestamp(timestamps.get(key))),
        reverse=True,
    )
    picked_ranges: dict[str, list[tuple[int, int]]] = {}
    candidates = []
    for conv_id, seed_seq in ordered:
        start, end = seed_seq - 2, seed_seq + 3
        ranges = picked_ranges.setdefault(conv_id, [])
        if any(start <= old_end and end >= old_start for old_start, old_end in ranges):
            continue
        ranges.append((start, end))
        messages = [windows[(conv_id, seed_seq)][seq]
                    for seq in sorted(windows[(conv_id, seed_seq)])]
        body = " / ".join(
            ("用户" if str(message.get("role") or "") == "user" else "AI")
            + ":" + " ".join(str(message.get("content") or "").split())[:360]
            for message in messages if str(message.get("content") or "").strip()
        )
        if not body:
            continue
        label = (
            "历史对话显示文本片段；用户气泡可能经过英文精修，不保证等于原始音频逐字内容。"
        )
        candidates.append({
            "candidate_id": f"{conv_id}#matched-message-{seed_seq}",
            "session_id": conv_id,
            "summary": f"{label} {body}"[:1800],
            "searchable": f"{label} {body}"[:2400],
            "error_count": 0,
            "created_at": timestamps.get((conv_id, seed_seq)),
            "source": "conversation_message_chunk",
        })
        if len(candidates) >= max(1, min(int(limit), 120)):
            break
    return rank_history_sparse(candidates, query, top_k=limit)


class HybridHistoryRetriever:
    """Fuse sparse and optional dense recall without entering every voice turn."""

    def __init__(
        self,
        store: Any,
        embedder: Any | None = None,
        *,
        semantic_timeout_seconds: float = 2.4,
        candidate_limit: int = 80,
        semantic_candidate_limit: int = 28,
        cache_size: int = 512,
        archive_store: Any | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.semantic_timeout_seconds = semantic_timeout_seconds
        self.candidate_limit = candidate_limit
        self.semantic_candidate_limit = semantic_candidate_limit
        # 宽语义池上限（plans/1017-02 §5 M2）：只在开关打开时用到。默认取
        # `candidate_limit` —— 即「宁可把已经取回来的字面候选都过一遍语义，
        # 也不让改写过的行因为字面分低而永远进不了稠密阶段」。
        # ⚠ 不新增构造参数（仓规：不动共用函数签名），这里内部推导。
        self.semantic_pool_limit = max(semantic_candidate_limit, candidate_limit)
        self.cache_size = cache_size
        self.archive_store = archive_store
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()

    async def search(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        if not hasattr(self.store, "list_history_candidates"):
            if hasattr(self.store, "search_history"):
                return await self.store.search_history(user_id, query, top_k=top_k)
            return []

        candidates = await self.store.list_history_candidates(
            user_id, limit=self.candidate_limit, query=query
        )
        # Handoff candidates and verbatim archive chunks serve different
        # recall needs. Do not let a full handoff pool crowd the transcript
        # chunks out before ranking; both pools are independently bounded.
        # 0915-14：conversation_message_chunk 是第三个独立池。显式回忆时优先
        # 按当前 query 从全量已存消息找匹配窗口；旧对话不随每轮进入 prompt。
        pools = [await self._archive_candidates(user_id), list(candidates)]
        if hasattr(self.store, "search_message_chunks"):
            try:
                pools.insert(0, list(await self.store.search_message_chunks(
                    user_id, query, limit=self.candidate_limit
                )))
            except Exception:
                pass
        elif hasattr(self.store, "list_message_chunks"):
            try:
                pools.insert(0, list(await self.store.list_message_chunks(user_id)))
            except Exception:
                pass
        candidates = _dedupe([row for pool in pools for row in pool])
        # ── 自回声过滤（plans/1017-02 §8.23，默认关）─────────────────────
        # 实测：真实 11 条回忆轮，第一名 **11/11 都是用户这句话本身**
        # （最近一条记忆就是这一轮刚写的摘要）。回忆不能把自己当前这句话
        # 当成"旧记忆"还给模型，否则任何相关性模型都会给它满分（§8.16）。
        try:
            from config import (
                HISTORY_RECALL_DROP_SELF_ECHO_ENABLED as _echo_gate_on,
                HISTORY_RECALL_SELF_ECHO_RATIO as _echo_ratio,
            )
        except Exception:
            _echo_gate_on, _echo_ratio = False, 0.95
        if _echo_gate_on and candidates:
            candidates = [
                row for row in candidates
                if _self_echo_ratio(query, _candidate_text(row)) < _echo_ratio
            ]
        sparse = rank_history_sparse(candidates, query, top_k=self.candidate_limit)
        # 绝对证据门槛（2026-10-22）：按池内 max_bm25 归一化会把弱池子的第一名
        # 也放大到 0.62 左右。弱池子（问句里只有“我上/次说”这类常见二元组）整池
        # 不过线就返回空，而不是硬塞一条旧消息。
        try:
            from config import HISTORY_RECALL_SPARSE_MIN_RAW as _sparse_min_raw
        except Exception:
            _sparse_min_raw = 0.0
        if _sparse_min_raw > 0 and sparse:
            sparse = [
                row for row in sparse
                if float(row.get("sparse_raw") or 0.0) >= _sparse_min_raw
            ]
        if not candidates:
            return []
        # ── 最近归档兜底（plans/1017-02 §8.25，默认关）─────────────────────
        # 「我们上次聊了什么」这类问句**没有内容词**，词面必然零命中；
        # 它要的是"最近那几场对话"，不是"含某个词的那一段"。
        try:
            from config import (
                HISTORY_RECALL_RECENT_ARCHIVE_FALLBACK_ENABLED as _recent_on,
                HISTORY_RECALL_RECENT_ARCHIVE_COUNT as _recent_n,
            )
        except Exception:
            _recent_on, _recent_n = False, 2
        if _recent_on and not sparse:
            archives = [row for row in candidates if _is_archive_row(row)]
            archives.sort(key=lambda row: _timestamp(row.get("created_at")),
                          reverse=True)
            if archives:
                return self._public_rows(
                    _dedupe(archives)[:max(1, int(_recent_n))], "recent_archive")
        if not self.embedder or not hasattr(self.embedder, "get_embeddings_batch"):
            return self._public_rows(sparse[:top_k], "sparse")

        # ── M6 短路（plans/1017-02 §8.14）：闸门要求「最终第一名有词面锚点」，
        # 而 `rank_history_sparse` **本来就只返回有词面重合的行** ⇒ 整池没有锚点
        # 等价于 sparse 为空 ⇒ 这次结果**必然是空**，不必付一次 embedding 往返。
        # 实测：一次 embedding 中位 ~93 ms，抖起来受 `semantic_timeout_seconds`
        # 2.4 s 约束（那是这一轮对话的等待时间）。50 句日常话里 **39 句**命中此短路。
        # ⛔ 与闸门**输出完全等价**（不是近似）；闸门关着时这段不执行。
        try:
            from config import (
                HISTORY_RECALL_MARGIN_GATE_ENABLED as _margin_gate_early,
            )
        except Exception:
            _margin_gate_early = False
        if _margin_gate_early and not any(
            float(row.get("lexical_overlap") or 0.0) > 0.0 for row in sparse
        ):
            return []

        # Embed the strongest sparse candidates and a bounded recent sample.
        # This keeps explicit recall responsive while still allowing semantic
        # recovery when the learner paraphrases an earlier conversation.
        recent = sorted(
            (dict(row) for row in candidates),
            key=lambda row: _timestamp(row.get("created_at")),
            reverse=True,
        )[:10]
        # 宽语义池（plans/1017-02 §5 M2）：默认关＝改前行为（只 embed 稀疏前列
        # ＋最近 10 条，再截到 `semantic_candidate_limit`）。打开后把**其余字面
        # 候选**也送进稠密阶段 —— 否则「字面不重合、又不是最近 10 条」的改写行
        # 永远进不了语义打分，无论它语义多像。**打分与融合逻辑一字未改。**
        try:
            from config import (
                HISTORY_RECALL_WIDE_SEMANTIC_ENABLED as _wide_semantic_pool,
            )
        except Exception:
            _wide_semantic_pool = False
        if _wide_semantic_pool:
            pool = _dedupe(
                list(sparse[: self.semantic_candidate_limit])
                + recent
                + list(candidates)
            )[: self.semantic_pool_limit]
        else:
            pool = _dedupe(list(sparse[: self.semantic_candidate_limit]) + recent)
            pool = pool[: self.semantic_candidate_limit]
        try:
            query_vector, candidate_vectors = await asyncio.wait_for(
                self._embed_query_and_candidates(query, pool),
                timeout=self.semantic_timeout_seconds,
            )
        except (asyncio.TimeoutError, Exception):
            return self._public_rows(sparse[:top_k], "sparse_fallback")
        if not query_vector or len(candidate_vectors) != len(pool):
            return self._public_rows(sparse[:top_k], "sparse_fallback")

        dense_rows = []
        all_similarities = []
        for row, vector in zip(pool, candidate_vectors):
            similarity = _cosine(query_vector, vector)
            all_similarities.append(similarity)
            if similarity < 0.34:
                continue
            dense = dict(row)
            dense["dense_score"] = similarity
            dense_rows.append(dense)
        dense_rows.sort(key=lambda item: item.get("dense_score", 0.0), reverse=True)

        from config import (
            HISTORY_RECALL_MARGIN_GATE_ENABLED as _margin_gate_on,
            HISTORY_RECALL_MARGIN_MIN as _margin_min,
        )
        if _margin_gate_on and dense_rows:
            # 环境边际（plans/1017-02 §8.11）：这个问句与**全池**相似度的中位数就是它的
            # "底噪"，只有明显高出底噪的行才值得注入。绝对余弦判不出"有没有这件事"
            # （§8.10：正负分布几乎重叠），但"比底噪高出多少"可以。
            _ordered = sorted(all_similarities)
            _mid = len(_ordered) // 2
            _ambient = (_ordered[_mid] if len(_ordered) % 2
                        else (_ordered[_mid - 1] + _ordered[_mid]) / 2.0)
            _floor = _ambient + _margin_min
            dense_rows = [row for row in dense_rows
                          if float(row.get("dense_score") or 0.0) >= _floor]

        sparse_rank = {
            self._candidate_key(row): index + 1 for index, row in enumerate(sparse)
        }
        dense_rank = {
            self._candidate_key(row): index + 1 for index, row in enumerate(dense_rows)
        }
        by_id = {
            self._candidate_key(row): dict(row)
            for row in list(sparse) + dense_rows
            if self._candidate_key(row)
        }
        fused = []
        for key, row in by_id.items():
            sparse_position = sparse_rank.get(key)
            dense_position = dense_rank.get(key)
            rrf = (1.0 / (60 + sparse_position) if sparse_position else 0.0)
            rrf += (1.0 / (60 + dense_position) if dense_position else 0.0)
            sparse_score = float(row.get("sparse_score") or 0.0)
            dense_score = float(row.get("dense_score") or 0.0)
            # A dense-only candidate needs a stronger similarity threshold;
            # this prevents a provider fallback or generic old topic from
            # being injected solely because it is recent.
            if not sparse_position and dense_score < 0.48:
                continue
            row["score"] = round(0.58 * sparse_score + 0.32 * dense_score + 6.0 * rrf, 6)
            fused.append(row)
        fused.sort(
            key=lambda item: (item.get("score", 0.0), _timestamp(item.get("created_at"))),
            reverse=True,
        )
        if _margin_gate_on:
            # 词面锚点只对**最终第一名**生效（plans/1017-02 §8.11）：
            # 闸门要回答的是"这次到底有没有一个真的对得上的旧片段"，
            # 而不是"能不能凑出三条"。第一名没有共同词 ⇒ 整次不注入。
            # ⛔ 实测教训：若对每一行都放行，「今天天气怎么样」会靠
            # 「今天早上的公交又晚了」共用一个「今天」混进来（11/50 vs 2/50）。
            if not fused:
                return []
            if float(fused[0].get("lexical_overlap") or 0.0) <= 0.0:
                return []
        # rerank 看前 10（plans/1022-01）：高分候选可能排在融合第 6~10 位，
        # 只看 top_k 会把它切掉。闸门关着时多出的行仍被 [:top_k] 截掉，
        # 与原行为逐字一致。
        final_rows = await self._rerank_gate(
            query, _dedupe(fused)[:max(top_k, 10)])
        return self._public_rows(final_rows[:top_k], "hybrid")

    async def _rerank_gate(self, query: str, rows: list[dict]) -> list[dict]:
        """按 rerank 分挡掉"低到不可能是用户问的那件事"的候选（plans/1022-01）。

        只过滤、不重排；失败/超时/配置缺失一律放行（fail-open）。
        """
        try:
            from config import (
                DASHSCOPE_API_KEY as api_key,
                HISTORY_RECALL_RERANK_GATE_ENABLED as enabled,
                HISTORY_RECALL_RERANK_MIN_SCORE as min_score,
                HISTORY_RECALL_RERANK_TIMEOUT as timeout,
            )
        except Exception:
            return rows
        if not enabled or not rows:
            return rows
        docs = [_candidate_text(row)[:500] for row in rows]
        if not any(docs):
            return rows
        try:
            scores = await asyncio.wait_for(
                _rerank_scores(query, docs, api_key),
                timeout=max(0.1, float(timeout)),
            )
        except Exception:
            return rows
        if len(scores) != len(rows):
            return rows
        return [
            row for row, score in zip(rows, scores)
            if float(score) >= float(min_score)
        ]

    @staticmethod
    def _candidate_key(row: dict) -> str:
        return str(row.get("candidate_id") or row.get("session_id") or "")

    async def _archive_candidates(self, user_id: str) -> list[dict]:
        """Build bounded transcript chunks only for an explicit recall turn.

        Cassette handoffs are executable checkpoints, not verbatim history. A
        learner may ask about an older word or sentence that fell outside the
        handoff's recent-message window, so exact recall also searches small
        chunks from the existing L3 archive. Only the matching chunk is later
        injected into generation; the full cassette is never replayed.
        """
        archive = self.archive_store
        if archive is None or not hasattr(archive, "list_sessions"):
            return []
        try:
            sessions = await archive.list_sessions(user_id)
        except Exception:
            return []
        sessions = sorted(
            (dict(item) for item in sessions),
            key=lambda item: str(item.get("archived_at") or ""),
            reverse=True,
        )[:20]
        rows: list[dict] = []
        seen_content: set[str] = set()
        for session in sessions:
            session_id = str(session.get("session_id") or "")
            if not session_id or not hasattr(archive, "retrieve_session"):
                continue
            try:
                payload = await archive.retrieve_session(user_id, session_id)
            except Exception:
                continue
            transcript = [
                {
                    "role": str(item.get("role") or ""),
                    "content": " ".join(str(item.get("content") or "").split())[:500],
                }
                for item in list((payload or {}).get("transcript") or [])
                if item.get("role") in {"user", "assistant"} and item.get("content")
            ]
            for start in range(0, len(transcript), 4):
                chunk = transcript[start:start + 6]
                if not chunk:
                    continue
                body = " / ".join(
                    ("用户" if item["role"] == "user" else "AI")
                    + ":" + item["content"]
                    for item in chunk
                )
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                if digest in seen_content:
                    continue
                seen_content.add(digest)
                rows.append({
                    "candidate_id": f"{session_id}#turns-{start}-{start + len(chunk) - 1}",
                    "session_id": session_id,
                    "summary": f"旧磁带原始片段={body}"[:1800],
                    "searchable": body[:2400],
                    "error_count": 0,
                    "created_at": session.get("archived_at") or (payload or {}).get("archived_at"),
                    "source": "session_transcript_chunk",
                })
                if len(rows) >= self.candidate_limit:
                    return rows
        return rows

    async def _embed_query_and_candidates(
        self, query: str, candidates: list[dict]
    ) -> tuple[list[float], list[list[float]]]:
        # M2b 开关：只影响"哪份文本被拿去算向量"，不影响 searchable 本身。
        try:
            from config import (
                HISTORY_RECALL_CLEAN_EMBED_TEXT_ENABLED as _clean_embed_text,
            )
        except Exception:
            _clean_embed_text = False
        candidate_keys = []
        candidate_vectors: list[list[float] | None] = []
        missing_texts = []
        missing_positions = []
        for index, row in enumerate(candidates):
            raw_text = str(row.get("searchable") or row.get("summary") or "")
            if _clean_embed_text:
                # M2b：只把"去了包装"的那份喂给 embedding；空串则回落原文。
                text = (embedding_surface(raw_text) or raw_text)[:1000]
            else:
                text = raw_text[:1000]
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            candidate_keys.append(key)
            cached = self._embedding_cache.get(key)
            candidate_vectors.append(cached)
            if cached is None:
                missing_texts.append(text)
                missing_positions.append(index)

        vectors = await self.embedder.get_embeddings_batch([query[:1000]] + missing_texts)
        if not isinstance(vectors, list) or len(vectors) != 1 + len(missing_texts):
            return [], []
        query_vector = self._validated_vector(vectors[0])
        if not query_vector:
            return [], []
        for position, raw_vector in zip(missing_positions, vectors[1:]):
            vector = self._validated_vector(raw_vector)
            if not vector:
                return [], []
            candidate_vectors[position] = vector
            self._cache(candidate_keys[position], vector)
        validated = [self._validated_vector(vector) for vector in candidate_vectors]
        if any(not vector or len(vector) != len(query_vector) for vector in validated):
            return [], []
        return query_vector, validated

    @staticmethod
    def _validated_vector(raw: Any) -> list[float]:
        if not isinstance(raw, (list, tuple)) or not raw:
            return []
        try:
            vector = [float(value) for value in raw]
        except (TypeError, ValueError):
            return []
        if not all(math.isfinite(value) for value in vector):
            return []
        if not any(abs(value) > 1e-12 for value in vector):
            return []
        return vector

    def _cache(self, key: str, vector: list[float]) -> None:
        self._embedding_cache[key] = vector
        self._embedding_cache.move_to_end(key)
        while len(self._embedding_cache) > self.cache_size:
            self._embedding_cache.popitem(last=False)

    @staticmethod
    def _public_rows(rows: list[dict], mode: str) -> list[dict]:
        result = []
        for raw in rows:
            row = dict(raw)
            row.pop("searchable", None)
            row["retrieval_mode"] = mode
            result.append(row)
        return result
