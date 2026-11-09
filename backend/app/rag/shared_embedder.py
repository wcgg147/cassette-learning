"""Shared embedding singleton for non-RAG consumers (plans/0915-06).

MethodologyRAG bundles pool + RAG query logic; history search only needs
`get_embedding`. Reuse the same provider router (siliconflow BGE-M3 etc.)
through one lazily-created instance so config stays single-sourced.
"""

from __future__ import annotations

from typing import List

from .methodology_rag import MethodologyRAG

_shared: MethodologyRAG | None = None


def get_shared_embedder() -> MethodologyRAG:
    global _shared
    if _shared is None:
        _shared = MethodologyRAG()
    return _shared


async def embed_text(text: str) -> List[float]:
    """Embed one text with the configured provider. May raise — callers guard."""
    return await get_shared_embedder().get_embedding(text)
