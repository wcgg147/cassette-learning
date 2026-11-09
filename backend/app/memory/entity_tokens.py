# -*- coding: utf-8 -*-
"""实体内容判据（2026-10-25 实体层大修）：把 detail/描述转成可比对的内容词元集。

供两处共用（判据必须同源，否则写入合并与存量合并会对不上）：
- memory/entities.normalize_archive_entity_graph 的内容兜底匹配（写入侧）
- 存量合并脚本（一次性数据修复）

判据语义：两条描述的**独有内容词**互为 <3（双向）⇒ 同一事件的措辞版。
复用 entity_search_terms 的分词与停用词，不另立一套。
"""
from __future__ import annotations

from .entities import entity_search_terms


def content_token_set(text: str) -> set[str]:
    """文本 → 内容词元集（去停用词；英文词 + 中文 bigram，与实体检索同源）。

    剔除 ≥5 字的长拼接词元（entity_search_terms 会产出"用户想带他去馆子
    喝酒"这类整串）：比对"是不是同一事件的措辞版"靠的是短内容词重合，
    长整串对措辞差异逐字不同，会把微差异放大成"不相似"。
    """
    return {
        term for term in entity_search_terms(str(text or ""), max_terms=96)
        if len(term) < 5 or not all("\u4e00" <= ch <= "\u9fff" for ch in term)
    }


def is_same_event_variants(a: str, b: str, *, min_unique: int = 3) -> bool:
    """双向独有内容词都 < min_unique ⇒ 视为同一事件的措辞版。"""
    ta = content_token_set(a)
    tb = content_token_set(b)
    if not ta or not tb:
        return False
    return len(ta - tb) < min_unique and len(tb - ta) < min_unique