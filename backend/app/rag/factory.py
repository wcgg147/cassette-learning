"""RAG 工厂 —— 开源包只有一种形态：本地文件版知识点检索器。

原仓这里按 MEMORY_BACKEND 分叉出 pgvector 版（rag/knowledge_rag.py）。
本包不含 PG/Redis 后端，那个分支已随包移除；四盘与复读机用的都是本地 JSON 版。
"""
try:
    from ..config import MEMORY_BACKEND
except ImportError:
    from config import MEMORY_BACKEND


def create_knowledge_rag(methodology_rag=None):
    """创建知识点检索器（本地 JSON 版，零外部依赖）。"""
    if MEMORY_BACKEND != "local":
        raise RuntimeError("MEMORY_BACKEND 只支持 local；本开源包不含 pgvector 后端。")
    from .knowledge_rag_local import KnowledgeRAGLocal
    return KnowledgeRAGLocal(methodology_rag)
