"""
Methodology RAG v4 - Pluggable embedding providers (siliconflow default).

Providers (set EMBEDDING_PROVIDER in .env):
  siliconflow - BGE-M3 via api.siliconflow.cn (domestic, free credits, CJK best)
  dashscope   - text-embedding-v3 via Alibaba Cloud (domestic, free tier)
  jina        - jina-embeddings-v3 (1M tokens/day free, global)
  fastembed   - BAAI/bge-small-zh-v1.5 (local ONNX, 130MB, fully offline)
  openai      - text-embedding-3-small (no free tier)

Default: siliconflow (domestic, cheap, runs BGE-M3 as API)
"""

import os
import re
import numpy as np
from typing import Any, Dict, Optional
# asyncpg 只在 PG 向量库路径用；本包默认本地 JSON 后端，
# 改成惰性导入，免得纯本地模式还要装一个 Postgres 驱动。
try:
    from ..config import (
        DATABASE_URL, BOOST_SCORE_CLAMP, EMBEDDING_PROVIDER, EMBEDDING_DIM,
        OPENAI_API_KEY, JINA_API_KEY, SILICONFLOW_API_KEY,
    )
except ImportError:
    from config import (
        DATABASE_URL, BOOST_SCORE_CLAMP, EMBEDDING_PROVIDER, EMBEDDING_DIM,
        OPENAI_API_KEY, JINA_API_KEY, SILICONFLOW_API_KEY,
    )


# ── 共享 HTTP 连接 ────────────────────────────────────────────
#
# 每次 embedding 都新建一个 httpx.AsyncClient，等于每个请求都重做一次
# TCP+TLS 握手。2026-09-14 对 siliconflow 实测：
#     每次新建客户端  2.139s / 2.104s / 2.087s
#     复用同一客户端  0.185s / 0.117s / 0.121s
# 也就是说那 2 秒里有约 1.95 秒纯粹是握手，复用后快约 18 倍。
#
# 为什么这不只是"快一点"：prepare_turn 每轮要 embed 两次（知识覆盖探针
# + 知识注入检索），光握手就白烧掉约 3.9 秒；加上回复生成，一轮总计会
# 顶穿 doubao-session-server.mjs 的 12s 兜底预算，用户听到的就是那句
# 固定的"抱歉，我还在追"。连接复用把这一项从 4.2s 压到约 0.24s。
#
# 按事件循环隔离存储：测试里会反复创建/关闭 loop，跨 loop 复用已关闭的
# 连接会直接抛错，按 loop key 存最省心。
_HTTP_CLIENTS: Dict[int, Any] = {}


def _shared_http_client():
    """返回当前事件循环可复用的 httpx.AsyncClient。"""
    import asyncio
    import httpx

    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        # 没有运行中的事件循环（同步调用场景）：退回一次性客户端，
        # 语义与改造前完全一致。
        return httpx.AsyncClient()
    client = _HTTP_CLIENTS.get(loop_key)
    if client is None or getattr(client, "is_closed", False):
        client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
        _HTTP_CLIENTS[loop_key] = client
    return client


async def aclose_shared_http_clients() -> None:
    """关闭所有共享连接（进程退出或测试收尾时调用）。"""
    for key in list(_HTTP_CLIENTS):
        client = _HTTP_CLIENTS.pop(key, None)
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


class _ReusedClient:
    """把 `async with httpx.AsyncClient() as c` 变成"复用共享连接"。

    刻意保留 async-with 的形状，只替换客户端来源：调用点的缩进和语义
    一行都不用改，diff 因此小到可以逐行审。退出时不关闭连接——它还要留
    给下一个请求用。
    """

    async def __aenter__(self):
        return _shared_http_client()

    async def __aexit__(self, *exc_info):
        return False


class MethodologyRAG:

    def __init__(self, pool=None):
        self._pool = pool
        self._dim = EMBEDDING_DIM
        self._provider = EMBEDDING_PROVIDER
        self._embedder = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        return self._pool

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    # -- Embedding router --

    async def get_embedding(self, text: str) -> list[float]:
        p = self._provider
        if p == "siliconflow": return await self._siliconflow_embed(text)
        if p == "dashscope": return await self._dashscope_embed(text)
        if p == "jina": return await self._jina_embed(text)
        if p == "fastembed": return await self._fastembed(text)
        if p == "openai": return await self._openai_embed(text)
        return self._hash_embedding(text)

    async def get_embeddings_batch(self, texts):
        p = self._provider
        if p == "siliconflow": return await self._siliconflow_embed_batch(texts)
        if p == "dashscope": return await self._dashscope_embed_batch(texts)
        if p == "jina": return await self._jina_embed_batch(texts)
        if p == "fastembed": return await self._fastembed_batch(texts)
        if p == "openai": return await self._openai_embed_batch(texts)
        return [self._hash_embedding(t) for t in texts]

    # -- SiliconFlow BGE-M3 (domestic, free credits, CJK best) --

    async def _siliconflow_embed(self, text):
        key = SILICONFLOW_API_KEY
        if not key or key.startswith("sk-your-"): return self._hash_embedding(text)
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://api.siliconflow.cn/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "BAAI/bge-m3", "input": [text]},
                    timeout=15.0,
                )
                if r.status_code == 200:
                    return r.json()["data"][0]["embedding"]
        except Exception: pass
        return self._hash_embedding(text)

    async def _siliconflow_embed_batch(self, texts):
        key = SILICONFLOW_API_KEY
        if not key or key.startswith("sk-your-"):
            return [self._hash_embedding(t) for t in texts]
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://api.siliconflow.cn/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "BAAI/bge-m3", "input": texts},
                    timeout=30.0,
                )
                if r.status_code == 200:
                    return [d["embedding"] for d in r.json()["data"]]
        except Exception: pass
        return [self._hash_embedding(t) for t in texts]

    # -- Alibaba DashScope text-embedding-v3 (domestic, free tier) --

    async def _dashscope_embed(self, text):
        key = os.getenv("DASHSCOPE_API_KEY", "")
        if not key: return self._hash_embedding(text)
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "text-embedding-v3", "input": {"texts": [text]}},
                    timeout=15.0,
                )
                if r.status_code == 200:
                    return r.json()["output"]["embeddings"][0]["embedding"]
        except Exception: pass
        return self._hash_embedding(text)

    async def _dashscope_embed_batch(self, texts):
        key = os.getenv("DASHSCOPE_API_KEY", "")
        if not key: return [self._hash_embedding(t) for t in texts]
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "text-embedding-v3", "input": {"texts": texts}},
                    timeout=30.0,
                )
                if r.status_code == 200:
                    return [e["embedding"] for e in r.json()["output"]["embeddings"]]
        except Exception: pass
        return [self._hash_embedding(t) for t in texts]

    # -- Jina AI v3 (1M tokens/day free) --

    async def _jina_embed(self, text):
        key = JINA_API_KEY
        if not key: return self._hash_embedding(text)
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "jina-embeddings-v3", "input": [text], "task": "retrieval.passage"},
                    timeout=15.0,
                )
                if r.status_code == 200:
                    return r.json()["data"][0]["embedding"]
        except Exception: pass
        return self._hash_embedding(text)

    async def _jina_embed_batch(self, texts):
        key = JINA_API_KEY
        if not key: return [self._hash_embedding(t) for t in texts]
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "jina-embeddings-v3", "input": texts, "task": "retrieval.passage"},
                    timeout=30.0,
                )
                if r.status_code == 200:
                    return [d["embedding"] for d in r.json()["data"]]
        except Exception: pass
        return [self._hash_embedding(t) for t in texts]

    # -- FastEmbed (local ONNX, fully offline) --

    def _load_fastembed(self):
        if self._embedder is not None: return self._embedder
        try:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
            return self._embedder
        except Exception: pass
        return None

    async def _fastembed(self, text):
        e = self._load_fastembed()
        if e:
            try: return list(e.embed([text]))[0].tolist()
            except Exception: pass
        return self._hash_embedding(text)

    async def _fastembed_batch(self, texts):
        e = self._load_fastembed()
        if e:
            try: return [r.tolist() for r in list(e.embed(texts))]
            except Exception: pass
        return [self._hash_embedding(t) for t in texts]

    # -- OpenAI (no free tier) --

    async def _openai_embed(self, text):
        key = OPENAI_API_KEY
        if not key or key.startswith("sk-your-"): return self._hash_embedding(text)
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"input": text, "model": "text-embedding-3-small"},
                    timeout=15.0,
                )
                if r.status_code == 200:
                    return r.json()["data"][0]["embedding"]
        except Exception: pass
        return self._hash_embedding(text)

    async def _openai_embed_batch(self, texts):
        key = OPENAI_API_KEY
        if not key or key.startswith("sk-your-"):
            return [self._hash_embedding(t) for t in texts]
        try:
            import httpx
            async with _ReusedClient() as c:
                r = await c.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"input": texts, "model": "text-embedding-3-small"},
                    timeout=30.0,
                )
                if r.status_code == 200:
                    return [d["embedding"] for d in r.json()["data"]]
        except Exception: pass
        return [self._hash_embedding(t) for t in texts]

    # -- Hash fallback --

    def _hash_embedding(self, text):
        text = text.lower()
        features = np.zeros(self._dim, dtype=np.float32)
        for i in range(len(text) - 1):
            idx = hash(text[i:i + 2]) % self._dim; features[idx] += 0.01
        for w in re.findall(r'\w+', text):
            idx = hash(w) % self._dim; features[idx] += 0.1
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff':
                idx = hash(ch) % self._dim; features[idx] += 0.05
        norm = np.linalg.norm(features)
        if norm > 0: features = features / norm
        return features.tolist()

    # -- Seed data --

    async def seed_methodologies(self):
        pool = await self._get_pool()
        methodologies = [
            {"name": "recast_feedback", "instruction": "When user makes a grammar error, naturally reformulate their sentence correctly in your response instead of pointing out the mistake.", "example_dialogue": 'User: "I goed to the store yesterday." -> AI: "Oh, you went to the store? What did you buy?"', "cefr_level": "A1", "scene_tags": ["en", "free_talk", "daily_life"]},
            {"name": "elicitation_en", "instruction": "Prompt user to self-correct by pausing or asking a question that makes them notice the error themselves.", "example_dialogue": 'User: "He go to school." -> AI: "Sorry, he...?"', "cefr_level": "B1", "scene_tags": ["en", "grammar_focus"]},
            {"name": "cn_ja_contrast", "instruction": "Use Chinese-Japanese contrast to highlight kanji meaning differences.", "example_dialogue": 'AI: "注意：中文的「勉強」是勉强的意思，但日语的「勉強（べんきょう）」是学习的意思。"', "cefr_level": "N5", "scene_tags": ["ja", "vocabulary", "kanji"]},
            {"name": "particle_drill", "instruction": "When user makes particle errors, model the correct particle naturally and explain the rule in Chinese.", "example_dialogue": 'User: "学校に勉強する" -> Tutor: "学校で勉強するんですね。动作场所用「で」，移动目的地用「に」。"', "cefr_level": "N5", "scene_tags": ["ja", "grammar_focus", "particles"]},
            {"name": "keigo_scaffolding", "instruction": "Introduce honorific and humble forms gradually. Start with desu/masu.", "example_dialogue": 'Tutor: "对老师说「先生がそうおっしゃいました」。おっしゃる是言う的尊敬语。"', "cefr_level": "N4", "scene_tags": ["ja", "keigo", "roleplay"]},
            {"name": "te_form_pattern", "instruction": "Teach te-form through common patterns: tekudasai, teimasu, temoiidesu.", "example_dialogue": 'Tutor: "買う的て形是買って。う结尾的五段动词变成って。"', "cefr_level": "N5", "scene_tags": ["ja", "grammar_focus", "te_form"]},
            {"name": "transitivity_pairs", "instruction": "Teach transitive/intransitive verb pairs. Rule: wo + transitive, ga + intransitive.", "example_dialogue": 'Tutor: "ドアが開いた（自动）。ドアを開けた（他动）。"', "cefr_level": "N4", "scene_tags": ["ja", "grammar_focus", "transitivity"]},
            {"name": "shadow_repeat", "instruction": "Model correct Japanese with natural rhythm. Have user shadow-repeat.", "example_dialogue": 'Tutor: "跟我重复：昨日、友達と映画を見に行きました。"', "cefr_level": "N5", "scene_tags": ["ja", "pronunciation", "free_talk"]},
            {"name": "conversation_gambits", "instruction": "Teach natural Japanese fillers: soudesune, naruhodo, eeto, anou.", "example_dialogue": 'Tutor: "日本人用「あのう」来停顿思考，像中文的那个。"', "cefr_level": "N4", "scene_tags": ["ja", "conversation", "free_talk"]},
        ]
        async with pool.acquire() as conn:
            for m in methodologies:
                emb = await self.get_embedding(f"{m['name']} {m['instruction']}")
                await conn.execute(
                    """INSERT INTO methodology_vectors
                       (name, instruction, example_dialogue, cefr_level, scene_tags, embedding, source)
                       VALUES (, , , , , ::vector, )
                       ON CONFLICT DO NOTHING""",
                    m["name"], m["instruction"], m.get("example_dialogue", ""),
                    m["cefr_level"], m["scene_tags"], emb,
                    f"embedding_{self._provider}",
                )

    # -- Retrieval --

    async def retrieve(self, query_embedding, cefr_level=None, scene_tag=None, top_k=3):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            conditions = ["TRUE"]; params = [query_embedding]; idx = 2
            if cefr_level:
                conditions.append(f"cefr_level = "); params.append(cefr_level); idx += 1
            if scene_tag:
                conditions.append(f" = ANY(scene_tags)"); params.append(scene_tag); idx += 1
            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"""SELECT id, name, instruction, example_dialogue, cefr_level,
                    scene_tags, boost_score,
                    1 - (embedding <=> ::vector) AS similarity
                    FROM methodology_vectors WHERE {where}
                    ORDER BY (1 - (embedding <=> ::vector)) * (1 + boost_score) DESC
                    LIMIT """,
                *params, top_k,
            )
            return [dict(r) for r in rows]

    async def search_by_text(self, query_text, cefr_level=None, scene_tag=None, top_k=3):
        pool = await self._get_pool()
        tsq = " & ".join(w for w in re.findall(r'\w+', query_text) if len(w) > 2) or "english"
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, name, instruction, example_dialogue, cefr_level,
                   scene_tags, boost_score, 0.5::float as similarity
                   FROM methodology_vectors
                   WHERE (::varchar IS NULL OR cefr_level = )
                     AND (::varchar IS NULL OR  = ANY(scene_tags))
                     AND to_tsvector('english', instruction || ' ' || name)
                         @@ to_tsquery('english', )
                   ORDER BY boost_score DESC LIMIT """,
                cefr_level, scene_tag, tsq, top_k,
            )
            if rows: return [dict(r) for r in rows]
            rows = await conn.fetch(
                """SELECT id, name, instruction, example_dialogue, cefr_level,
                   scene_tags, boost_score, 0.3::float as similarity
                   FROM methodology_vectors
                   WHERE (::varchar IS NULL OR cefr_level = )
                     AND (::varchar IS NULL OR  = ANY(scene_tags))
                     AND (instruction ILIKE '%%' ||  || '%%'
                          OR name ILIKE '%%' ||  || '%%')
                   ORDER BY boost_score DESC LIMIT """,
                cefr_level, scene_tag, query_text, top_k,
            )
            return [dict(r) for r in rows]

    async def record_call(self, user_id, mid, score, sid):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO methodology_calls (user_id, methodology_id, cooperation_score, session_id)
                   VALUES (::uuid, , , ::uuid)""",
                user_id, mid, score, sid,
            )

    async def evolve_boosts(self, user_id):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT methodology_id, AVG(cooperation_score) as avg_coop
                   FROM methodology_calls WHERE user_id = ::uuid
                   GROUP BY methodology_id""",
                user_id,
            )
            lo, hi = BOOST_SCORE_CLAMP
            for row in rows:
                delta = 0.1 * (row["avg_coop"] - 0.5)
                await conn.execute(
                    """UPDATE methodology_vectors
                       SET boost_score = GREATEST(, LEAST(, boost_score + ))
                       WHERE id = """,
                    delta, lo, hi, row["methodology_id"],
                )
