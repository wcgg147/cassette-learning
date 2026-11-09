"""Deterministic routing for learner memory at conversation boundaries.

The memory store answers *what is known*.  This router answers *when it is
allowed to affect the current cassette*.  Keeping that decision deterministic
prevents an old scene from silently becoming the new scene.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MemoryRoute:
    mode: str = "global_only"
    reason: str = "new cassette inherits learner state, not old dialogue"
    query: str = ""
    pending_intent: str = ""
    precision: str = "semantic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "query": self.query,
            "pending_intent": self.pending_intent,
            "precision": self.precision,
        }


class MemoryOrchestrator:
    """Small policy layer between L1/L2/L3 memory and the dialogue director."""

    # 身份问句（plans/0915-21）：问"我叫什么/你叫什么/还记得我名字吗"是**身份
    # 事实**问题，不是"回忆旧磁带"。实测："Do you remember what my name is?" 命中
    # 下面的回忆正则 → 走 episodic_recall → 检索回来的都是别的主题摘要 →
    # 模型按回忆合同答"我记忆里没有你的名字"。身份必须先判出来，走 global_only，
    # 由【身份记忆】+ Episode Ledger 的 known_facts 直接作答。
    IDENTITY_QUESTION = re.compile(
        r"(?:我|本人)(?:的)?(?:名字|姓名|称呼)|我(?:叫|是)什么|(?:记得|知道).{0,6}(?:我|俺)(?:的)?名字|"
        r"who\s+am\s+i|what(?:'s|\s+is)\s+my\s+name|(?:remember|know)\s+(?:my|the\s+learner'?s)\s+name|"
        r"my\s+name\b|"
        r"(?:你|AI|助手)(?:的)?(?:名字|称呼)|你(?:叫|是)什么|what(?:'s|\s+is)\s+your\s+name|"
        r"(?:remember|know)\s+your\s+name|your\s+name\b",
        re.IGNORECASE,
    )
    CROSS_SESSION_REFERENCE = re.compile(
        r"(?:上(?:一|个)?(?:次|盘|段|个会话)|之前(?:那|的)?(?:次|盘|段|个会话)?|"
        r"别的磁带|另一盘|其他磁带|我们(?:之前|上次)聊过|还记得|记不记得|之前聊过|继续上次|"
        r"last\s+(?:time|session|conversation)|previous\s+(?:session|conversation)|"
        # Natural recap questions do not always say "remember" or "last time".
        # Without these forms, a request such as "Have we talked about any other
        # shows?" stayed global_only and never called the episodic retriever.
        r"what\s+(?:else\s+)?(?:do|can)\s+you\s+know\s+about\s+me|"
        r"what\s+(?:else\s+)?(?:have|did)\s+we\s+(?:ever\s+)?(?:talked|chatted|discussed|mentioned)\b|"
        r"(?:have|did)\s+we\s+(?:ever\s+)?(?:talked|chatted|discussed|mentioned)\b|"
        r"what\s+(?:[a-z'-]+\s+){0,3}(?:tv\s+)?shows?\s+(?:was|were)\s+i\s+(?:just\s+)?watching\b|"
        r"what\s+(?:else\s+)?have\s+i\s+told\s+you\s+about\b|"
        # A natural progressive-recall form: asking what the learner previously
        # said about a named topic must search old cassettes even without the
        # words "remember", "before", or "last time".
        r"what\s+(?:did|have)\s+i\s+(?:say|said|mention|mentioned|tell\s+you|told\s+you)\s+(?:anything\s+)?(?:else\s+)?(?:about|regarding|on)\b|"
        r"continue\s+(?:from|our)\s+last|"
        # 英文回忆问句（0915 实测 "Do you remember what we talked about before?"
        # 未命中旧触发词表 → episodic_recall 没路由 → 检索没跑）：
        r"do\s+you\s+remember|remember\s+(?:what|when|where|that|my\s+name)|"
        r"what\s+did\s+we\s+(?:talk|chat|discuss|do)|what\s+were\s+we\s+(?:talking|doing))",
        re.IGNORECASE,
    )
    VAGUE_RECALL_REFERENCE = re.compile(
        r"^(?:(?:hey|hi|alice)[,，\s]*)*(?:do\s+you\s+remember|remember)"
        r"(?:\s+(?:anything|something|what|when|where|that|this|it|we|us|me|"
        r"one|thing|stuff|everything|about))?[?？!！.。\s]*$|"
        r"^(?:(?:Alice|你)[,，\s]*)?(?:你还记得|还记得|记不记得)"
        r"(?:(?:我|我们|你|吗|呢|啊|呀|不|这个|那个|这件事|那件事|这东西|那东西))*"
        r"[?？!！.。\s]*$",
        re.IGNORECASE,
    )
    # ── 「状态变化问句」：M0 的**预算开关**（plans/1017-02 §5 / §8.9）─────────
    # ⛔ 这不是判据，只是「允许花一次有界检索」的许可；**注入与否由检索分数决定**。
    # 台架 `bugreports/eval_definite_reference_gate_1017.py`（16 正例／50 负例）：
    #   * 只靠定指成分   召回 12/16，误报 21/50 ⇒ 废
    #   * 只靠本正则     召回 15/16，误报  1/50 ⇒ 三者中唯一可用
    #   * 两者相与       召回 12/16，误报  1/50 ⇒ 加定指反而砍召回（中文常省略定指）
    # 唯一误报「你还好吗」与正例「你还打算辞职吗」词形同构 ⇒ 只能靠检索分数兜。
    STATE_CHANGE_QUESTION = re.compile(
        r"怎么样了|好点|好些|修好|成没成|成了没|后来|还.{0,8}吗|有没有.{0,10}[吗呢]|"
        r"还(?:在|是|打算|老)|"
        r"\bstill\b|\bany better\b|\bfall through\b|\bgo up\b|\bwent up\b|\blately\b|"
        r"\bhow(?:'s|\s+is)\b.{0,24}\bdoing\b|\bever\b|\bend up\b|\bturned out\b",
        re.IGNORECASE,
    )
    EXACT_DETAIL_RECALL = re.compile(
        r"\bwhat\s+(?:did|have)\s+i\s+(?:say|said|mention|mentioned|tell\s+you|told\s+you)\s+(?:anything\s+)?(?:else\s+)?(?:about|regarding|on)\b|"
        r"\b(?:tell|remind)\s+me\s+(?:what|which|how|why)\b.{0,60}\b(?:said|mentioned|talked|discussed)\b|"
        r"(?:哪个|哪一个|什么).{0,12}(?:单词|短语|句子|表达|发音)|"
        r"(?:原话|具体).{0,10}(?:说|问|回答)|"
        r"(?:说成|读成|听成).{0,12}(?:什么|哪个|哪一个)|"
        r"(?:最后|当时).{0,12}(?:解释|说|问)(?:的|了)?(?:什么|哪个|哪一个)",
        re.IGNORECASE,
    )
    DELEGATES_PLANNING = re.compile(
        r"(?:你来安排|你安排(?:一下)?|继续练(?:英语)?(?:吧)?|今天练什么|"
        r"你来定|随便练点什么|继续学(?:英语)?|you\s+(?:choose|decide)|"
        r"what\s+should\s+we\s+practice)",
        re.IGNORECASE,
    )
    ACCEPTS_PENDING = re.compile(
        r"^(?:好|好的|可以|行|就这个|就按这个|从这个开始|继续这个|yes|ok(?:ay)?|sure)"
        r"[，,呀啊吧。.!！ ]*(?:就这个|就按这个|继续这个)?[呀啊吧。.!！ ]*$",
        re.IGNORECASE,
    )
    DECLINES_PENDING = re.compile(
        r"(?:不要|不想|换一个|换个|别练这个|今天不练|新的任务|其他(?:的)?|"
        r"no\b|not\s+today|something\s+else)",
        re.IGNORECASE,
    )
    CONCRETE_NEW_TASK = re.compile(
        r"(?:今天|这次|这一盘).{0,10}(?:想|要|练|聊|模拟)|"
        r"(?:我想|我要|我们来)(?:练|聊|模拟|学习)|"
        r"(?:today|this time).{0,20}(?:practice|talk|roleplay)",
        re.IGNORECASE,
    )

    @classmethod
    def build_session_packet(
        cls,
        *,
        profile: Dict[str, Any],
        relationship: Dict[str, Any],
        assessment: Dict[str, Any],
        target_lang: str,
        style: Any = None,
    ) -> Dict[str, Any]:
        """Build the bounded facts every new cassette may inherit."""
        error_focus = (
            assessment.get("priority_focus")
            or assessment.get("focus")
            or profile.get("recent_weakness")
            or ""
        )
        response_mode = {
            "preferred_tone": relationship.get("preferred_tone", ""),
            "avoid_patterns": list(relationship.get("avoid_patterns") or [])[:6],
            "correction_style": getattr(style, "correction_style", "") if style else "",
            "preferred_length": getattr(style, "preferred_length", None) if style else None,
        }
        response_mode = {key: value for key, value in response_mode.items() if value not in (None, "", [])}
        return {
            "inherit_always": {
                "target_language": target_lang,
                "cefr": str(profile.get("cefr") or "A1"),
                "learner_profile": {
                    key: profile[key]
                    for key in ("goals", "interests", "learning_preferences")
                    if profile.get(key)
                },
                "error_profile_focus": error_focus,
                "self_reported_weakness": cls._distill_weakness(
                    relationship.get("self_reported_weakness", "")
                ),
                "stable_learning_preference": cls._distill_preference(
                    relationship.get("learning_preference", "")
                ),
                "response_mode": response_mode,
            },
            "pending_confirmation": relationship.get("next_session_intent", ""),
            "excluded_by_default": [
                "old_dialogue", "old_scene", "old_roles", "old_step", "old_episode_ledger",
            ],
        }

    @staticmethod
    def _distill_preference(value: str) -> str:
        """从记忆原文提取学习偏好子句，去掉场景具体物件与废话。"""
        text = " ".join(str(value or "").split())
        if not text:
            return ""
        marker_match = re.search(
            r"(以后|下次|接下来|想练|多练|要练|重点练|希望练|喜欢学|更想练)[^。！？]*",
            text,
        )
        if marker_match and len(marker_match.group(0).strip()) >= 4:
            return marker_match.group(0).strip()[:60]
        first = re.split(r"[。；]", text, maxsplit=1)[0]
        first = re.sub(r"^(我)?(?:明确)?(?:记一下|记住)?[:：]?\s*", "", first)
        return first.strip()[:60]

    @staticmethod
    def _distill_weakness(value: str) -> str:
        """只保留真正的薄弱点表述；无法确认时返回空（不注入垃圾）。"""
        text = " ".join(str(value or "").split())
        if not text:
            return ""
        if not re.search(
            r"(薄弱|不擅长|英语.{0,4}(不好|差)|口语.{0,4}(不好|差)|语法.{0,4}(不好|不会)|"
            r"发音.{0,4}(不好|不准)|过去式|过去时|时态|struggle|weak\s+(at|in)|"
            r"容易说错|老是说错|总说错|经常说错)",
            text, re.IGNORECASE,
        ):
            return ""
        clause = re.search(
            r"((?:英语|口语|语法|发音|时态|过去式)?[^。！？]{0,20}"
            r"(?:薄弱|不好|不擅长|不会说|说错|struggle|weak|差)[^。！？]{0,24})",
            text, re.IGNORECASE,
        )
        return (clause.group(1) if clause else text)[:60].strip()

    @classmethod
    def route_turn(
        cls,
        text: str,
        *,
        fresh_context: bool,
        turn_count: int,
        cassette_mode: str = "resume",
        pending_intent: str = "",
        awaiting_confirmation: bool = False,
        state_change_recall: bool = False,
    ) -> MemoryRoute:
        clean = " ".join((text or "").split())
        # 身份问句优先于回忆路由：名字是持久身份事实，不是旧磁带检索
        if cls.IDENTITY_QUESTION.search(clean) and not re.search(
            r"(?:上(?:一|个)?(?:次|盘|段)|之前|那次|last\s+(?:time|session)|previous)",
            clean, re.IGNORECASE,
        ):
            return MemoryRoute(
                mode="global_only",
                reason="identity question (name) answered from authoritative identity memory",
                pending_intent=pending_intent,
            )
        if (cls.CROSS_SESSION_REFERENCE.search(clean)
                and not cls.VAGUE_RECALL_REFERENCE.fullmatch(clean)):
            return MemoryRoute(
                mode="episodic_recall",
                reason="user explicitly referenced another cassette or earlier conversation",
                query=cls._history_query(clean),
                pending_intent=pending_intent,
                precision=(
                    "exact_detail" if cls.EXACT_DETAIL_RECALL.search(clean)
                    else "semantic"
                ),
            )
        if cassette_mode == "continuation" and turn_count == 0:
            return MemoryRoute(
                mode="linked_continuation",
                reason="the learner explicitly opened a linked continuation cassette",
            )
        if awaiting_confirmation:
            if cls.ACCEPTS_PENDING.search(clean):
                return MemoryRoute(
                    mode="pending_accepted",
                    reason="user confirmed the proposed continuation",
                    pending_intent=pending_intent,
                )
            if cls.DECLINES_PENDING.search(clean) or len(clean) >= 8:
                return MemoryRoute(
                    mode="pending_replaced",
                    reason="user declined or supplied a different concrete task",
                    pending_intent=pending_intent,
                )
            return MemoryRoute(
                mode="pending_confirmation",
                reason="confirmation is still unresolved",
                pending_intent=pending_intent,
            )
        if cassette_mode == "continuation_candidate" and turn_count == 0 and pending_intent and cls.DELEGATES_PLANNING.search(clean):
            return MemoryRoute(
                mode="pending_confirmation",
                reason="a prior cassette left a next-step intention; ask before activating it",
                pending_intent=pending_intent,
            )
        if cassette_mode == "continuation_candidate" and turn_count == 0 and pending_intent and cls.CONCRETE_NEW_TASK.search(clean):
            return MemoryRoute(
                mode="pending_replaced",
                reason="the learner supplied a concrete new-cassette task, which overrides the old suggestion",
                pending_intent=pending_intent,
            )
        # ── M0：状态变化问句 ⇒ **允许**花一次有界检索（预算开关，不是判据）──────
        # 位置在所有既有分支**之后**：显式标记、入戏续接、待确认、候选新盘全部优先，
        # 只在水面最平静（本来会 `global_only`）时才开。开关默认关 ⇒ 与改前逐字同义。
        # ⛔ 这里只说"可以查"；**要不要注入由检索自己的分数决定**（否则又是词表当判据）。
        if (state_change_recall
                and cls.STATE_CHANGE_QUESTION.search(clean)
                and not cls.VAGUE_RECALL_REFERENCE.fullmatch(clean)):
            return MemoryRoute(
                mode="episodic_recall",
                reason=("state-change question about something mentioned earlier "
                        "(budget gate; the retriever's own score decides injection)"),
                query=cls._history_query(clean),
                pending_intent=pending_intent,
                precision="semantic",
            )
        return MemoryRoute()

    @classmethod
    def _history_query(cls, text: str) -> str:
        focused_recall = re.search(
            r"(?:like|for\s+example|比如|例如)[,:，\s]*"
            r"(what\s+(?:[a-z'-]+\s+){0,3}(?:tv\s+)?shows?\s+(?:was|were)\s+i\s+(?:just\s+)?watching\b)",
            text, re.IGNORECASE,
        )
        if focused_recall:
            return focused_recall.group(1)[:240]
        query = cls.CROSS_SESSION_REFERENCE.sub(" ", text)
        query = re.sub(r"(?:你知道吗|你记得吗|的内容|聊了什么|说了什么)", " ", query)
        query = re.sub(r"^(?:(?:like|well|also|and|so|then)[,:\s]*)+", " ", query,
                       flags=re.IGNORECASE)
        query = re.sub(r"[^\w\u4e00-\u9fff]+", " ", query, flags=re.UNICODE)
        query = " ".join(query.split()).strip()
        # Removing question scaffolding can leave only punctuation (e.g. the
        # whole "what show was I watching" template). Keep the original query
        # in that case; an empty-token query otherwise silently returns no hits.
        searchable = re.sub(
            r"\b(?:like|well|also|and|so|then|what|which|show|shows|series|was|were|"
            r"i|we|did|have|do|you|know|about|me|just|watching|talked|chatted|"
            r"discussed|mentioned|else|other|any|ever)\b",
            " ", query, flags=re.IGNORECASE,
        )
        if not re.search(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2}", searchable):
            query = text
        return query[:240]
