"""Turn-level recall evidence packet.

Pure functions only: no model calls, no DB calls, no imports from the reply path.
The caller supplies candidates; this module decides how many may be shown (0-2),
adds provenance/status, drops self-echo, and renders the use contract.

Division of labour:
  * code: candidate budget, provenance, status, dedupe, self-echo, rendering;
  * reply model: whether this utterance refers to the past, which item, whether
    to mention it, and how to say it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

MAX_ITEMS = 2
SNIPPET_CHARS = 220
ECHO_RATIO = 0.95


@dataclass(frozen=True)
class Evidence:
    content: str
    source: str
    when: str = ""
    status: str = "unknown"
    reason: str = "candidate"

    def to_dict(self) -> Dict[str, str]:
        return {
            "content": self.content,
            "source": self.source,
            "when": self.when,
            "status": self.status,
            "reason": self.reason,
        }


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _bigrams(text: str) -> set[str]:
    text = _norm(text)
    if len(text) < 2:
        return set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def coverage(query: str, text: str) -> float:
    """Fraction of the query's character bigrams present in the candidate.

    Used only to drop the current utterance echoed back as "memory"; it is not a
    relevance score. 0.95 means the candidate is essentially this sentence.
    """
    q = _bigrams(query)
    if not q:
        return 0.0
    return len(q & _bigrams(text)) / len(q)


def should_offer(
    *,
    route_mode: str = "",
    candidate_count: int = 0,
    related_entity_count: int = 0,
    open_thread_count: int = 0,
    user_text: str = "",
    topic_overlap: bool = False,
) -> bool:
    """Whether the caller has any candidates worth putting in front of the model.

    This is an availability/budget gate, not a semantic judgement: it never asks
    "does this sentence refer to the past". The model makes that call from the
    packet. A false offer costs a few hundred prompt characters; a false skip
    costs the user the feeling of not being remembered.

    ``looks_like_reference_question`` is a broader *supply* cue for indexical
    asks ("我那个麻烦事后来怎么样了"), where the lexical router has no content
    word to match. It only decides whether to show the topic index; the model
    still decides whether the ask refers to any of it.
    """
    if str(route_mode or "") == "episodic_recall":
        return True
    if _is_immediate_context(user_text):
        return False
    if max(int(candidate_count or 0), int(related_entity_count or 0)) > 0:
        return True
    if topic_overlap:
        return True
    if not _has_question(user_text):
        return False
    return bool(_content_terms(user_text))


# Supply cues only: a question shape plus a demonstrative/past marker. They do
# not decide that the user is asking about memory; they decide whether the
# compact topic index is worth attaching. The use decision stays in the model.
_QUESTION_CUES = ("？", "?", "吗", "呢", "什么", "哪", "怎么", "为什么",
                  "是不是", "有没有", "还记得", "记得吗", "来着")
_REFERENCE_CUES = ("那个", "那件", "那家", "那个人", "那事", "那些",
                   "上次", "之前", "以前", "后来", "还记得", "记得",
                   "来着", "小家伙", "麻烦事")
_EN_REFERENCE_CUES = ("that one", "that thing", "the one", "the thing",
                      "remember", "last time", "little one", "what was that",
                      "did i tell you", "again")
# Immediate-context references point at the previous sentence / this session,
# not at long-term memory. They must not pull the archive index in; c05-c08 in
# the frozen set showed that an indiscriminate gate makes the model invent a
# "previous statement" instead of asking which word was meant.
_IMMEDIATE_CUES = ("刚才", "上一句", "这句", "这个词", "这个说法", "前面那句",
                   "刚说", "last sentence", "that word", "just said")
_LONGTERM_CUES = ("上次", "之前", "以前", "还记得", "记得", "last time", "remember")


def _is_immediate_context(text: Any) -> bool:
    t = _norm(text)
    if not t:
        return False
    low = t.lower()
    return (
        any(cue in t or cue in low for cue in _IMMEDIATE_CUES)
        and not any(cue in t or cue in low for cue in _LONGTERM_CUES)
    )


def _has_question(text: Any) -> bool:
    t = _norm(text)
    return bool(t) and any(cue in t for cue in _QUESTION_CUES)


def _has_reference(text: Any) -> bool:
    t = _norm(text)
    low = t.lower()
    return (any(cue in t for cue in _REFERENCE_CUES)
            or any(cue in low for cue in _EN_REFERENCE_CUES))


def looks_like_reference_question(text: Any) -> bool:
    t = _norm(text)
    if not t or _is_immediate_context(t):
        return False
    return _has_question(t) and _has_reference(t)


# Two-character Chinese terms / three-letter English words that carry no topic.
_STOP_TERMS = {
    "最后", "后来", "怎么", "什么", "那个", "这个", "那家", "这家", "那样",
    "这样", "现在", "已经", "还是", "就是", "觉得", "知道", "记得", "我们",
    "你们", "他们", "她们", "自己", "一个", "一下", "可以", "是不是", "有没有",
    "哪里", "为什么", "事情", "东西", "时候", "地方", "的话", "一样", "有点",
    "最近", "还在", "然后", "而且", "但是", "因为", "所以", "如果", "真的",
    "不是", "没有", "什么", "多少", "怎么样", "怎样", "这样", "那样",
    "你好", "还好", "好吗", "你还", "今天", "天气", "么样", "天天",
}
_EN_STOP = {"the", "and", "for", "you", "your", "that", "this", "was", "were",
            "are", "with", "what", "when", "how", "did", "does", "have", "has",
            "had", "about", "again", "still", "there", "here", "from", "they"}


def _content_terms(text: Any) -> set:
    t = _norm(text)
    terms = set()
    for i in range(len(t) - 1):
        w = t[i:i + 2]
        if w in _STOP_TERMS:
            continue
        if all("\u4e00" <= c <= "\u9fff" for c in w):
            terms.add(w)
    for w in re.findall(r"[a-z]{3,}", t.lower()):
        if w not in _EN_STOP:
            terms.add(w)
    return terms


def index_overlap(text: Any, archives: Any) -> bool:
    """True when the utterance names a topic that appears in the index.

    This is a supply signal only: it says "there is a candidate line with this
    word", not "the user is asking about the past". The model still decides.
    """
    terms = _content_terms(text)
    if not terms:
        return False
    blob = " ".join(
        str(a.get("summary") or "") + " " + str(a.get("title") or "")
        + " " + str(a.get("last_user_text") or "")
        for a in (archives or []) if isinstance(a, dict)
    )
    if not blob:
        return False
    low = blob.lower()
    return any(t in blob or t in low for t in terms)


def _topic_line(summary: Any, limit: int = 90) -> str:
    text = _norm(summary)
    if "具体事件：" in text:
        text = text.split("具体事件：", 1)[1].split("|")[0]
    return text.strip()[:limit].strip()


def build_topic_index(
    archives: Iterable[Dict[str, Any]],
    *,
    max_items: int = 12,
) -> List[Evidence]:
    """A compact map of known topics, one line per archived session.

    It is not a list of facts to recite; it exists so the model can resolve an
    indexical ask ("那个小家伙") to a topic. Provenance is carried per line.
    """
    limit = max(0, min(int(max_items or 0), 16))
    out: List[Evidence] = []
    seen: set[str] = set()
    for item in archives or ():
        if not isinstance(item, dict):
            continue
        source = "archive:" + str(item.get("session_id") or item.get("conv_id") or "")[:80]
        if source in seen:
            continue
        content = _topic_line(item.get("summary") or item.get("text") or "")
        if not content:
            continue
        last = " ".join(str(item.get("last_user_text") or "").split())
        if last and last not in content:
            content = f"{content}；最后：{last[:70]}"
        seen.add(source)
        out.append(Evidence(
            content=content,
            source=source,
            when=str(item.get("created_at") or item.get("archived_at") or "")[:40],
            status="unknown",
            reason="index",
        ))
        if len(out) >= limit:
            break
    return out


def render_topic_index(items: Iterable[Evidence]) -> str:
    """Render the topic index plus its use contract. Empty -> empty string."""
    rows = [item for item in (items or []) if item and item.content]
    if not rows:
        return ""
    lines = ["[你知道的话题索引·不是待办]"]
    for item in rows:
        meta = " · ".join(part for part in (item.when, item.source) if part)
        lines.append(f"- {item.content}" + (f"（{meta}）" if meta else ""))
    lines.append(
        "如果他在问一件具体的事，先在索引里找；找到就用那件事，用自己的话说；"
        "找不到就说你没什么印象，不要拿别的事顶上；不要罗列索引。"
        "只用索引行里写明的事；不要补充行为、地点、数量或感受；"
        "如果这一行没有他问的那个方面，就说你只记得这件事的概况，不要编。"
        "同一行里有先后状态时，用最后/最新的那个；"
        "如果他问的是还没发生的结果，就说计划或到哪一步，不要否认；"
        "如果他只是泛问最近聊了什么，挑一两件自然的说，不要念清单。"
    )
    return "\n".join(lines)


def _from_archive(item: Dict[str, Any], reason: str) -> Optional[Evidence]:
    content = _norm(item.get("summary") or item.get("text") or item.get("content"))
    if not content:
        return None
    return Evidence(
        content=content[:SNIPPET_CHARS],
        source="archive:" + str(item.get("session_id") or item.get("conv_id") or "")[:80],
        when=str(item.get("created_at") or item.get("observed_at") or "")[:40],
        status=str(item.get("status") or "unknown")[:24],
        reason=reason,
    )


def _from_entity(item: Dict[str, Any], reason: str) -> Optional[Evidence]:
    name = _norm(item.get("name") or item.get("entity_name"))
    detail = _norm(item.get("detail") or item.get("value"))
    content = "：".join(part for part in (name, detail) if part)
    if not content:
        return None
    return Evidence(
        content=content[:SNIPPET_CHARS],
        source="entity:" + str(item.get("entity_id") or item.get("id") or "")[:80],
        when=str(item.get("created_at") or item.get("observed_at") or "")[:40],
        status=str(item.get("status") or "unknown")[:24],
        reason=reason,
    )


def _from_thread(item: Dict[str, Any], reason: str) -> Optional[Evidence]:
    topic = _norm(item.get("topic") or item.get("title"))
    detail = _norm(item.get("detail") or item.get("text"))
    content = "：".join(part for part in (topic, detail) if part)
    if not content:
        return None
    return Evidence(
        content=content[:SNIPPET_CHARS],
        source="thread:" + str(item.get("thread_id") or item.get("id") or "")[:80],
        when=str(item.get("last_seen_at") or item.get("created_at") or "")[:40],
        status=str(item.get("status") or "open")[:24],
        reason=reason,
    )


def build_evidence(
    *,
    user_text: str = "",
    route_mode: str = "",
    cross_session_recall: Iterable[Dict[str, Any]] = (),
    related_entities: Iterable[Dict[str, Any]] = (),
    open_threads: Iterable[Dict[str, Any]] = (),
    max_items: int = MAX_ITEMS,
) -> List[Evidence]:
    """Pack 0..N candidates into at most ``max_items`` evidence rows.

    Order: episodic recall first (the retriever already ranked it), then open
    threads, then entity hits. The current utterance echoed back is dropped.
    """
    limit = max(0, min(int(max_items or 0), 3))
    if limit == 0:
        return []
    ordered: List[Evidence] = []
    if route_mode == "episodic_recall":
        for item in cross_session_recall or ():
            if isinstance(item, dict):
                ev = _from_archive(item, "explicit")
                if ev:
                    ordered.append(ev)
    for item in open_threads or ():
        if isinstance(item, dict):
            ev = _from_thread(item, "open_thread")
            if ev:
                ordered.append(ev)
    if route_mode != "episodic_recall":
        for item in cross_session_recall or ():
            if isinstance(item, dict):
                ev = _from_archive(item, "candidate")
                if ev:
                    ordered.append(ev)
    for item in related_entities or ():
        if isinstance(item, dict):
            ev = _from_entity(item, "entity")
            if ev:
                ordered.append(ev)

    seen: set[tuple[str, str]] = set()
    out: List[Evidence] = []
    for ev in ordered:
        if coverage(user_text, ev.content) >= ECHO_RATIO:
            continue
        key = (ev.source, ev.content[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
        if len(out) >= limit:
            break
    return out


def render_evidence(items: Iterable[Evidence]) -> str:
    """Render the packet plus the use contract. Empty packet -> empty string."""
    rows = [item for item in (items or []) if item and item.content]
    if not rows:
        return ""
    lines = [
        "[可能相关的旧事·不是待办]",
    ]
    for item in rows:
        meta = " · ".join(part for part in (item.when, item.source,
                                            "" if item.status in ("", "unknown") else item.status)
                          if part)
        lines.append(f"- {item.content}" + (f"（{meta}）" if meta else ""))
    lines.extend([
        "只有当他这句话确实在指其中一件事时才用；用你自己的话说，不报出处、不罗列。",
        "如果他在问一件具体的事，而这里没有那件事，就说你没什么印象，不要拿另一件事顶上。",
    ])
    return "\n".join(lines)
