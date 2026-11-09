"""Choose one evidence-backed next speaking task for the learner.

This module is intentionally deterministic.  It turns existing learner memory
into a clear C-end entry point without asking an LLM to invent a learning need.
"""

from __future__ import annotations

from typing import Any


_ITEM_LABELS = {
    "en_articles": "冠词",
    "en_plurals": "名词单复数",
    "en_prepositions": "介词",
    "en_present_perfect": "现在完成时",
    "ja_wa_ga": "助词 は／が",
    "ja_ni_de": "助词 に／で",
    "ja_te_form": "て 形",
    "ja_ta_form": "た 形",
    "ja_keigo": "敬语",
    "ja_cn_misuse": "汉字误用",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _topic(profile: dict) -> str:
    topics = profile.get("preferred_topics") or []
    if isinstance(topics, str):
        topics = [part.strip() for part in topics.split(",") if part.strip()]
    return _text(topics[0]) if topics else ""


def build_next_session_brief(
    profile: dict | None,
    due_reviews: list[dict] | None,
    relationship: dict | None,
    target_lang: str = "en",
) -> dict:
    """Return exactly one next-best session, ordered by learning value."""
    profile = profile or {}
    relationship = relationship or {}
    due_reviews = due_reviews or []
    language = "日语" if target_lang.lower().startswith("ja") else "英语"
    base = {"estimated_minutes": 5, "target_lang": target_lang}

    if due_reviews:
        item = due_reviews[0]
        item_type = _text(item.get("item_type"))
        label = _ITEM_LABELS.get(item_type, item_type or "上次的易错表达")
        return {
            **base,
            "mode": "due_review",
            "title": f"把「{label}」真正说顺",
            "reason": "这项内容已经到复习时间，趁现在回忆更容易形成长期记忆。",
            "session_goal": f"在不同语境中独立、自然地使用{label}",
            "scene": "review_transfer",
        }

    open_loop = _text(relationship.get("open_loop"))
    if open_loop:
        return {
            **base,
            "mode": "open_loop",
            "title": "接着上次没聊完的事",
            "reason": open_loop,
            "session_goal": f"围绕这件事完成一次真实的{language}表达：{open_loop}",
            "scene": "continuity_talk",
        }

    weakness = _text(profile.get("recent_weakness"))
    if weakness:
        label = _ITEM_LABELS.get(weakness, weakness)
        return {
            **base,
            "mode": "weakness",
            "title": f"用 5 分钟攻克「{label}」",
            "reason": "这是近期反复出现的薄弱点，本次只练一个目标。",
            "session_goal": f"在真实对话中独立使用{label}",
            "scene": "weakness_transfer",
        }

    topic = _topic(profile)
    if topic:
        return {
            **base,
            "mode": "interest",
            "title": f"聊聊你感兴趣的「{topic}」",
            "reason": "从熟悉的话题开始，更容易进入自然表达状态。",
            "session_goal": f"围绕{topic}完成一段自然的{language}对话",
            "scene": "interest_talk",
        }

    return {
        **base,
        "mode": "discovery",
        "title": "第一次先聊你真正需要什么",
        "reason": "外教会用轻松对话了解你的目标和当前表达水平。",
        "session_goal": f"通过自然对话发现最值得优先提升的{language}能力",
        "scene": "goal_discovery",
    }
