# -*- coding: utf-8 -*-
"""她问过、但用户没答的问题（plans/1017-02 §8.76）。

⛔ 用户明确担心的事："存下来之后时不时拿出来说，不合时宜"。
所以这个模块的纪律是**门要窄**：

1. **不主动开口**：没有定时、不在开场、不在第一轮；只有用户这句话里出现了
   相关词（≥2 分），她才有资格提一句。
2. **一条、三天**：同一时间最多三条，三天没机会就过期。
3. **不催**：措辞是"上次我问你 X，你还没说"，不是"你怎么不回答"。

零模型调用：抽取和匹配都是字符串规则。
"""

from __future__ import annotations

import re
import uuid

from agent.open_threads import _zh_bigrams, _en_words

_QUESTION_END = re.compile(r"[?？]\s*$")
# 只收"真在问他事"的问句；这些开头的多是她自己的感慨/反问，不该追。
_SKIP_PREFIX = ("你觉得", "你说呢", "是不是", "对吧", "好吗", "行吗", "要不要我")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s*", str(text or ""))
    return [p.strip() for p in parts if p.strip()]


def last_question(text: str, *, min_chars: int = 10) -> str:
    """从她的一段话里取最后一句问句（没有就空串）。"""
    for sentence in reversed(_sentences(text)):
        if not _QUESTION_END.search(sentence):
            continue
        body = sentence.strip()
        if len(body) < min_chars:
            continue
        if any(body.startswith(p) for p in _SKIP_PREFIX):
            continue
        return body[:160]
    return ""


def score_question(question: str, user_text: str) -> int:
    """用户这句话跟那个问题有多相关（中文二元组 + 英文长词）。"""
    q = str(question or "")
    text = str(user_text or "")
    return len(_zh_bigrams(q) & _zh_bigrams(text)) + 2 * len(_en_words(q) & _en_words(text))


def question_record(text: str, topic: str = "") -> dict | None:
    """把模型判定"她问了、他没答"的那一句收成记录（模型判断，规则只做校验）。

    ⛔ 为什么不用关键词判"答没答"：他答"He loves fish"，词面上和
    "What does your friend like to eat" 一个词都对不上，会被判成没答——
    那正是用户担心的"不合时宜地拿出来说"。
    """
    body = " ".join(str(text or "").split())[:160]
    if len(body) < 10 or not _QUESTION_END.search(body):
        return None
    qid = "q_" + uuid.uuid5(uuid.NAMESPACE_URL, body.casefold()).hex[:12]
    # topic 是**双语**短标签（模型给），只用于匹配；展示仍用原问句。
    label = " ".join(str(topic or "").split())[:80]
    return {"question_id": qid, "text": body, "topic": label}


def pick_shown(questions: list[dict], user_text: str, *, threshold: int = 1,
               skip_ids: set[str] | None = None) -> dict | None:
    """本轮要不要提：只有用户的话把门打开才返回那一条。

    门槛 1（不是 2）：问句本来就短——"朋友来杭州吃什么"对他那句
    "我朋友明天就到了"只有 1 分（共同词只有"朋友"）。护栏是**别的**：
    同时最多三条待问、3 天过期、一盘只提一次；而且只有话题词撞上才可能到 1。
    """
    skip = skip_ids or set()
    best = None
    best_score = 0
    for item in questions or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("question_id") or "") in skip:
            continue
        # 有双语标签就按标签匹配（跨语言对得上）；没有才退回原问句。
        score = score_question(
            str(item.get("topic") or item.get("text") or ""), user_text)
        if score >= threshold and score > best_score:
            best, best_score = item, score
    return best


def render_line(question: dict) -> str:
    """给 prompt 的一行。措辞留退路——不催、不追。"""
    text = str(question.get("text") or "").strip()[:160]
    return (
        "\n[上次我问过、他还没说的事] 你之前问过：" + text +
        "。他这一句正好又在聊这件事——这一轮**先接一句你还在等他说**（一句就够，别追问）；"
        "他要是岔开了，就别再追，也不要重复问。\n"
    )
