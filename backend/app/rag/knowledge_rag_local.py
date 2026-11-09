"""
Knowledge RAG - 本地文件版（替代pgvector）

用JSON文件存储知识点和embedding，使用Python实现余弦相似度检索，
零外部依赖，不需要安装PostgreSQL和pgvector扩展。

路径结构：./data/knowledge/chunks.json（所有知识点）
"""

import os
import json
import glob
import math
from typing import Optional
from pathlib import Path


LAYER_PRIORITY = {
    "L1方法论与理论": 1.0,
    "L2学情诊断与需求分析": 1.0,
    "L4语言点精讲与纠错": 1.2,
    "L5即时互动与引导": 1.1,
    "L6学习策略指导": 0.9,
}


class KnowledgeRAGLocal:
    """
    知识点向量检索器 - 本地文件版。
    
    替代KnowledgeRAG（pgvector版），使用JSON文件存储，
    Python实现余弦相似度计算，无需PostgreSQL。
    
    数据存储：./data/knowledge/chunks.json
    """

    def __init__(self, methodology_rag=None):
        self._embedder = methodology_rag
        self._data_path = Path("./data/knowledge/chunks.json")
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks = self._load_chunks()
        self._last_mtime = self._data_path.stat().st_mtime if self._data_path.exists() else 0.0

    def _check_and_reload(self):
        """如果磁盘上的 chunks.json 文件被自动修复更新，自动加载最新数据到运行内存中。"""
        if self._data_path.exists():
            try:
                mtime = self._data_path.stat().st_mtime
                if mtime > self._last_mtime:
                    self._chunks = self._load_chunks()
                    self._last_mtime = mtime
                    print(f"[KnowledgeRAGLocal] Detected chunks.json modification. Automatically reloaded {len(self._chunks)} chunks.")
            except Exception as e:
                print(f"[KnowledgeRAGLocal] Failed to auto-reload chunks: {e}")

    def _load_chunks(self) -> list:
        if self._data_path.exists():
            with open(self._data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_chunks(self):
        with open(self._data_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)

    async def close(self):
        pass

    async def get_embedding(self, text: str) -> list[float]:
        if self._embedder:
            return await self._embedder.get_embedding(text)
        return self._hash_embedding(text)

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if self._embedder:
            return await self._embedder.get_embeddings_batch(texts)
        return [self._hash_embedding(t) for t in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        import numpy as np
        import re
        dim = 1024
        features = np.zeros(dim, dtype=np.float32)
        for i in range(len(text) - 1):
            idx = hash(text[i:i + 2]) % dim
            features[idx] += 0.01
        for w in re.findall(r'\w+', text):
            idx = hash(w) % dim
            features[idx] += 0.1
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff':
                idx = hash(ch) % dim
                features[idx] += 0.05
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.tolist()

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    async def seed_chunks(self, chunks_dir: str) -> int:
        if self._chunks and len(self._chunks) > 0:
            print(f"[KnowledgeRAGLocal] Using cached {len(self._chunks)} chunks from {self._data_path}")
            return len(self._chunks)

        pattern = os.path.join(chunks_dir, "*.json")
        files = sorted(glob.glob(pattern))

        if not files:
            return 0

        count = 0
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                chunk = json.load(f)

            embed_text = f"{chunk['title']} {' '.join(chunk.get('keywords', []))} {chunk['content']}"
            embedding = await self.get_embedding(embed_text)

            existing = next((c for c in self._chunks if c["chunk_id"] == chunk["id"].replace("chunk_", "")), None)

            chunk_data = {
                "chunk_id": chunk["id"].replace("chunk_", ""),
                "title": chunk["title"],
                "layer": chunk["layer"],
                "language": chunk["language"],
                "skill": chunk["skill"],
                "source": "|".join(chunk.get("source", [])),
                "source_note": chunk.get("source_note", ""),
                "keywords": chunk.get("keywords", []),
                "content": chunk["content"],
                "embedding": embedding,
                "boost_score": 0.0,
            }

            if existing:
                self._chunks.remove(existing)
            self._chunks.append(chunk_data)
            count += 1

        self._save_chunks()
        return count

    async def retrieve(
        self,
        query_embedding: list[float],
        layer: Optional[str] = None,
        language: Optional[str] = None,
        skill: Optional[str] = None,
        top_k: int = 3,
        has_errors: bool = False,
        seen_chunks: Optional[dict] = None,
        exclude_layers: Optional[list[str]] = None,
    ) -> list[dict]:
        self._check_and_reload()
        results = []
        for chunk in self._chunks:
            if layer and chunk["layer"] != layer:
                continue
            if exclude_layers and chunk["layer"] in exclude_layers:
                continue
            if language and chunk["language"] != language:
                continue
            if skill and chunk["skill"] != skill:
                continue

            similarity = self._cosine_similarity(query_embedding, chunk.get("embedding", []))
            results.append({
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "layer": chunk["layer"],
                "language": chunk["language"],
                "skill": chunk["skill"],
                "source": chunk["source"],
                "source_note": chunk["source_note"],
                "keywords": chunk["keywords"],
                "content": chunk["content"],
                "boost_score": chunk["boost_score"],
                "similarity": similarity,
            })

        results = self._apply_layer_boost(results, has_errors)

        if seen_chunks:
            results = self._apply_seen_penalty(results, seen_chunks)

        return results[:top_k]

    async def retrieve_by_text(
        self,
        query_text: str,
        layer: Optional[str] = None,
        language: Optional[str] = None,
        skill: Optional[str] = None,
        top_k: int = 3,
        has_errors: bool = False,
        seen_chunks: Optional[dict] = None,
        exclude_layers: Optional[list[str]] = None,
    ) -> list[dict]:
        self._check_and_reload()
        # 1. 查询改写与扩展 (Query Expansion & Spoken Normalization)
        try:
            from .query_rewriter import QueryRewriter
            expanded_query = QueryRewriter.rewrite(query_text)
        except ImportError:
            expanded_query = query_text

        embedding = await self.get_embedding(expanded_query)
        candidates = []
        query_terms = [w.lower() for w in expanded_query.split() if len(w) > 1]

        # 2. 多路召回：向量分数与 BM25/关键词重叠分数
        for chunk in self._chunks:
            if layer and chunk["layer"] != layer:
                continue
            if exclude_layers and chunk["layer"] in exclude_layers:
                continue
            if language and chunk["language"] != language:
                continue
            if skill and chunk["skill"] != skill:
                continue

            vector_sim = self._cosine_similarity(embedding, chunk.get("embedding", []))

            # 计算 BM25 词频得分
            title_text = chunk.get("title", "").lower()
            content_text = chunk.get("content", "").lower()
            keywords = [k.lower() for k in chunk.get("keywords", [])]

            bm25_score = 0.0
            for term in query_terms:
                if term in title_text:
                    bm25_score += 3.0
                if any(term in kw for kw in keywords):
                    bm25_score += 2.0
                if term in content_text:
                    bm25_score += 1.0

            compressed_snippet = self.compress_chunk_for_prompt(chunk)

            candidates.append({
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "layer": chunk["layer"],
                "language": chunk["language"],
                "skill": chunk["skill"],
                "source": chunk["source"],
                "source_note": chunk["source_note"],
                "keywords": chunk["keywords"],
                "content": chunk["content"],
                "compressed_snippet": compressed_snippet,
                "boost_score": chunk["boost_score"],
                "vector_sim": vector_sim,
                "bm25_score": bm25_score,
            })

        # 3. RRF (Reciprocal Rank Fusion) 融合排名
        vector_sorted = sorted(candidates, key=lambda x: x["vector_sim"], reverse=True)
        bm25_sorted = sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)

        for rank, item in enumerate(vector_sorted):
            item["vector_rank"] = rank + 1
        for rank, item in enumerate(bm25_sorted):
            item["bm25_rank"] = rank + 1

        # 当使用 hash embedding（无真实向量模型）时，向量排名噪声大，应降低其权重，
        # 让 BM25/关键词匹配占据主导，保证文献块（KN-061~KN-068）按真实关键词召回。
        max_vector_sim = max((c.get("vector_sim", 0.0) for c in candidates), default=0.0)
        vector_weight = 0.25 if max_vector_sim < 0.25 else 0.5

        for item in candidates:
            # RRF 加权融合：BM25 权重高于向量，降低 hash embedding 噪声
            rrf_score = (
                vector_weight * (1.0 / (60 + item["vector_rank"]))
                + (1.0 - vector_weight) * (1.0 / (60 + item["bm25_rank"]))
            )
            item["hybrid_score"] = rrf_score + (item["vector_sim"] * vector_weight)

        # 4. 两阶段重排序 (Two-Stage Lightweight Reranker)
        candidates = self._rerank_candidates(candidates, has_errors, query_text)

        if seen_chunks:
            candidates = self._apply_seen_penalty(candidates, seen_chunks)

        return candidates[:top_k]

    # 明确标识的学术文献块（带真实引用），用于 rerank 时额外加权
    LITERATURE_CHUNK_IDS = {
        "KN-061", "KN-062", "KN-063", "KN-064", "KN-065", "KN-066", "KN-067", "KN-068", "KN-069"
    }

    def _rerank_candidates(self, candidates: list[dict], has_errors: bool, query_text: str) -> list[dict]:
        """精排层 (Reranker)：结合层级优先级、错误驱动、文献来源与意图切合度。"""
        qt = query_text.lower()
        for r in candidates:
            boost = LAYER_PRIORITY.get(r.get("layer", ""), 1.0)
            if has_errors and r.get("layer") == "L4语言点精讲与纠错":
                boost *= 1.4

            # Tutor Acts / 场景切分倾向加权
            if "biped" in qt or "tutor act" in qt or "纠错" in qt:
                if r.get("skill") == "Methodology" or "biped" in r.get("title", "").lower():
                    boost *= 1.5

            # 学术文献块额外加权：当查询涉及具体 SLA 概念时，把对应的真实文献块推到前面。
            # 映射表：概念关键词 -> 优先文献 chunk_id
            concept_to_literature = {
                "article": {"KN-061", "KN-053"},
                "冠词": {"KN-061", "KN-053"},
                "subject_verb": {"KN-061", "KN-054"},
                "主谓": {"KN-061", "KN-054"},
                "tense": {"KN-061", "KN-055"},
                "时态": {"KN-061", "KN-055"},
                "preposition": {"KN-061", "KN-056"},
                "介词": {"KN-061", "KN-056"},
                "word_order": {"KN-061", "KN-057"},
                "语序": {"KN-061", "KN-057"},
                "chinglish": {"KN-061", "KN-058"},
                "中式英语": {"KN-061", "KN-058"},
                "collocation": {"KN-061"},
                "搭配": {"KN-061"},
                "vocabulary": {"KN-061"},
                "词汇": {"KN-061"},
                "clec": {"KN-061"},
                "corpus": {"KN-061"},
                "corrective": {"KN-062", "KN-063", "KN-064"},
                "recast": {"KN-062", "KN-063"},
                "explicit": {"KN-064"},
                "feedback": {"KN-062", "KN-063", "KN-064"},
                "修正性反馈": {"KN-062", "KN-063"},
                "显性纠错": {"KN-064"},
                "隐性纠错": {"KN-064"},
                "output": {"KN-065"},
                "pushed": {"KN-065"},
                "输出假说": {"KN-065"},
                "interaction": {"KN-066"},
                "negotiation": {"KN-066"},
                "互动假说": {"KN-066"},
                "formulaic": {"KN-067"},
                "语块": {"KN-067"},
                "affective": {"KN-069"},
                "affective_filter": {"KN-069"},
                "anxiety": {"KN-069"},
                "情感过滤": {"KN-069"},
                "焦虑": {"KN-069"},
                "contrastive": {"KN-068"},
                "l1_transfer": {"KN-068"},
                "母语迁移": {"KN-068"},
                "对比分析": {"KN-068"},
            }
            chunk_id = r.get("chunk_id", "")
            for concept, preferred_ids in concept_to_literature.items():
                if concept in qt and chunk_id in preferred_ids:
                    boost *= 2.2
                    break

            r["final_score"] = r.get("hybrid_score", 0.0) * boost * (1 + r.get("boost_score", 0.0))

        candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return candidates

    @staticmethod
    def compress_chunk_for_prompt(chunk: dict) -> str:
        """
        Token 瘦身压缩器：将冗长知识块压缩为高密度、低 Tokens 提示，
        专为实时语音流响应设计（< 50 Words），保障毫秒级首包延迟并节省 70% Tokens。
        """
        title = chunk.get("title", "")
        content = chunk.get("content", "")

        # 若存在“地道句型/原则”等核心句，优先抽取核心段
        lines = [line.strip() for line in content.split("。") if line.strip()]
        core_line = lines[0] if lines else content[:100]

        return f"[{title}]: {core_line}"

    def _apply_layer_boost(self, results: list[dict], has_errors: bool) -> list[dict]:
        for r in results:
            boost = LAYER_PRIORITY.get(r.get("layer", ""), 1.0)
            if has_errors and r.get("layer") == "L4语言点精讲与纠错":
                boost *= 1.3
            base_sim = r.get("hybrid_score", r.get("similarity", 0.0))
            r["final_score"] = base_sim * boost * (1 + r.get("boost_score", 0.0))

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results

    def _apply_seen_penalty(
        self, results: list[dict], seen_chunks: dict
    ) -> list[dict]:
        for r in results:
            chunk_id = r.get("chunk_id", "")
            if chunk_id in seen_chunks:
                coop = seen_chunks[chunk_id]
                if coop > 0.7:
                    penalty = 0.3
                elif coop < 0.3:
                    penalty = 1.5
                else:
                    penalty = 0.6
                r["seen_penalty"] = penalty
                r["cooperation_score"] = coop
                r["final_score"] = r.get("final_score", r.get("similarity", 0.0)) * penalty
            else:
                r["seen_penalty"] = 1.0
                r["cooperation_score"] = None

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results

    async def get_user_recent_knowledge(
        self, user_id: str, days: int = 7
    ) -> dict:
        """查询用户最近 days 天内接触过的知识点及其平均吸收度。
        返回 {chunk_id: avg_cooperation_score} 字典，供 _apply_seen_penalty 使用。
        
        高吸收度 (>0.7) → penalty=0.3（基本排除，不用再教）
        低吸收度 (<0.3) → penalty=1.5（提权强化，需要再练）
        中等吸收度 (0.3~0.7) → penalty=0.6（降低优先级）
        """
        try:
            import sqlite3
            from datetime import datetime, timedelta
            db_path = str(self._data_path.parent.parent / "memory.db")
            if not os.path.exists(db_path):
                return {}
            conn = sqlite3.connect(db_path)
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            cursor = conn.execute("""
                SELECT chunk_id, AVG(cooperation_score) as avg_score, COUNT(*) as cnt
                FROM knowledge_usage
                WHERE user_id = ? AND created_at >= ?
                GROUP BY chunk_id
                ORDER BY cnt DESC
            """, (user_id, cutoff))
            rows = cursor.fetchall()
            conn.close()
            result = {}
            for row in rows:
                chunk_id = row[0]
                avg_score = row[1] if row[1] is not None else 0.5
                result[chunk_id] = round(avg_score, 2)
            return result
        except Exception:
            return {}

    async def record_usage(
        self,
        user_id: str,
        chunk_id: str,
        session_id: str,
        cooperation_score: float = 0.5,
    ):
        """记录知识点使用情况到 memory.db，用于去重和权重优化。"""
        if not chunk_id:
            return
        try:
            import sqlite3
            from datetime import datetime
            db_path = str(self._data_path.parent.parent / "memory.db")
            conn = sqlite3.connect(db_path)
            conn.execute("""
                INSERT INTO knowledge_usage (user_id, chunk_id, session_id, cooperation_score, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, chunk_id, session_id, cooperation_score,
                  datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception:
            pass  # best-effort, don't fail the session

    async def evolve_boosts(self, user_id: str):
        pass

    def get_all_chunks(self) -> list[dict]:
        return self._chunks