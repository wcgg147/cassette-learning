# -*- coding: utf-8 -*-
"""未闭环的事：她记得"上次没说完的那件事"，但**只在话题真的碰到时**才提。

为什么需要（plans/1017-02 §8.56）：
12 轮实录里她的回复语法都对、也都接得上，但**没有具体的东西**——没有"上次你说要换工作，
后来呢"。这不是加规则能补的，缺的是**连续性材料**。

⛔ 三条纪律（都是用户明确否过的坑）：
  1. **不是每条回复带一行记忆**：只有本轮话题命中才注入，且**一盘最多一次**；
  2. **不写成清单**：只给一句"她自己的好奇"，由她决定要不要问；
  3. **零额外模型调用**：匹配是字符串比对，注入只是拼一行 prompt。

代价：会话末抽一次（1 次模型调用，不在语音路径上）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# 换行用常量：本文件经常被脚本改写，转义换行最容易在半路被吃掉。
NL = chr(10)

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_RE = re.compile(r"[a-z]{4,}")
_STOP_EN = {
    "that", "this", "with", "have", "been", "about", "what", "when", "where",
    "which", "there", "their", "them", "then", "than", "from", "into", "just",
    "like", "some", "more", "very", "will", "would", "could", "should", "your",
    "yours", "ours", "they", "were", "was", "his", "her", "she", "him",
}
_STOP_ZH = {"这个", "那个", "什么", "怎么", "我们", "你们", "他们", "自己", "一个",
            "现在", "后来", "事情", "东西", "觉得", "时候", "还是", "就是"}


def _zh_bigrams(text: str) -> set[str]:
    """中文按二元组切——不引分词器，够用且零依赖。"""
    chars = [c for c in text if _ZH_RE.match(c)]
    grams = {a + b for a, b in zip(chars, chars[1:])}
    return {g for g in grams if g not in _STOP_ZH}


def _en_words(text: str) -> set[str]:
    return {w for w in _EN_RE.findall(text.lower()) if w not in _STOP_EN}


def _thread_terms(thread: dict) -> tuple[set[str], set[str]]:
    blob = " ".join(str(thread.get(k) or "") for k in ("topic", "detail"))
    return _zh_bigrams(blob), _en_words(blob)


def score_thread(thread: dict, user_text: str) -> int:
    """话题命中分：中文二元组重合 + 英文长词重合 + 原词直接出现。

    短话题（"体检""换工作"）只有 1-2 个二元组，光靠二元组会漏；
    原词整串出现在用户这句话里，直接算强命中。
    """
    text = user_text or ""
    topic = str(thread.get("topic") or "").strip()
    zh_terms, en_terms = _thread_terms(thread)
    zh_hits = len(zh_terms & _zh_bigrams(text))
    en_hits = len(en_terms & _en_words(text))
    # 英文命中权重更高：中文二元组容易误撞（"工作""事情"这类太泛的词已在停用表里）。
    bonus = 3 if topic and topic in text else 0
    return zh_hits + en_hits * 2 + bonus


def best_match(threads: list[dict], user_text: str, *, threshold: int = 2,
               skip_ids: set[str] | None = None) -> dict | None:
    """挑一条真的相关的；不够像就返回 None（宁可不提，也别硬接）。"""
    skip = skip_ids or set()
    scored = [
        (score_thread(t, user_text), t) for t in (threads or [])
        if isinstance(t, dict) and str(t.get("thread_id") or "") not in skip
    ]
    scored = [(s, t) for s, t in scored if s >= threshold]
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    return scored[0][1]


def due_thread(threads: list[dict], now, *, window_days: int = 3,
               skip_ids: set[str] | None = None) -> dict | None:
    """到点该问的那件事：`due_at` 已过、且在 window 天内。

    为什么单独一条路（§8.59）：话题匹配只能覆盖"他自己又提起"的情况；
    人不会天天提，但**事情已经发生了**——这时候该由她先问一句。
    ⛔ 只认 `YYYY-MM-DD`；解析不出来的一律不当"到点"，绝不按乱日期提醒。
    """
    from datetime import date, timedelta

    skip = skip_ids or set()
    today = now.date() if hasattr(now, "date") else date.today()
    best = None
    for thread in threads or []:
        if not isinstance(thread, dict) or str(thread.get("thread_id") or "") in skip:
            continue
        # 回写闭环：最近 3 天问过的，别再提（存在库里，重启也记得）。
        asked = str(thread.get("asked_at") or "")[:10]
        if asked:
            try:
                if (today - date.fromisoformat(asked)).days < window_days:
                    continue
            except ValueError:
                pass
        raw = str(thread.get("due_at") or "").strip()[:10]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            continue
        try:
            due = date.fromisoformat(raw)
        except ValueError:
            continue
        if not (today - timedelta(days=window_days) <= due <= today):
            continue
        if best is None or raw > str(best.get("due_at") or ""):
            best = dict(thread)
    return best


def pick_thread(threads: list[dict], user_text: str, *, now=None,
                skip_ids: set[str] | None = None, threshold: int = 2,
                lone_recent_days: int = 3) -> tuple[dict | None, bool]:
    """"到点"优先，其次"话题命中"。返回 (thread, 是否到点)。

    顺序的理由：话题命中只是锦上添花；**事情过去了没人问**才是真损失。
    """
    due = due_thread(threads, now, skip_ids=skip_ids) if now is not None else None
    if due:
        return due, True
    hit = best_match(threads, user_text, threshold=threshold, skip_ids=skip_ids)
    if hit:
        return hit, False
    # §8.74：他说的是**指代**（"我朋友""那个"），二元组只拿 1 分，够不到 2。
    # 只在"三天内恰好只有一条对得上"时放行 1 分——两条以上宁可不说，别猜错人。
    # 实测：'我朋友明天到了' 对 '杭州的朋友来访' = 1 分（2026-10-18）。
    if now is None:
        return None, False
    skip = skip_ids or set()
    recent_hits: list[dict] = []
    for thread in threads or []:
        if not isinstance(thread, dict) or str(thread.get("thread_id") or "") in skip:
            continue
        if score_thread(thread, user_text) < 1:
            continue
        seen = str(thread.get("last_seen_at") or thread.get("created_at") or "")[:19]
        try:
            seen_at = datetime.fromisoformat(seen.replace("Z", "+00:00"))
            if seen_at.tzinfo is None:
                seen_at = seen_at.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (now - seen_at).days <= lone_recent_days:
            recent_hits.append(thread)
    # 同一次会话里抽出来的多条线程算**一件事**（实测一个"朋友来访"会被抽成
    # "杭州的朋友来访" + "带朋友吃什么"两条）。歧义要按"不同的事"判，不是按行数。
    clusters: dict[str, list[dict]] = {}
    for thread in recent_hits:
        key = str(thread.get("source_conv_id") or thread.get("thread_id") or "")
        clusters.setdefault(key, []).append(thread)
    if len(clusters) == 1:
        group = next(iter(clusters.values()))
        group.sort(key=lambda t: -score_thread(t, user_text))
        return group[0], False
    return None, False


def render_line(thread: dict, *, due: bool = False, explicit: bool = False) -> str:
    """给 prompt 的一句“她自己的好奇”。

    ⛔ 措辞必须留退路：话题只是擦边时她可以完全不提；否则会变成每条都追问。
    """
    topic = str(thread.get("topic") or "").strip()
    detail = str(thread.get("detail") or "").strip()
    body = f"{topic} — {detail}" if detail else topic
    if due:
        # 到点那句要**主动但留退路**：问一句结果，不追问细节。
        return (
            NL + "[到点了的事] 他之前提过：" + body[:150] +
            "，时间就是这两天。这一轮**先**问一句结果怎么样（一句就够，别追问细节），"
            "再顺着他的话往下；如果他明显不想聊，就放下。" + NL
        )
    if explicit:
        # 他**这一轮自己提到了**那件事（不是擦边）。实测“顺口问一句”从来不会发生，
        # 只把措辞抬到“接一句你知道的”——仍是顺序，不是硬门。
        return (
            NL + "[他刚提到的那件事] 他之前提过：" + body[:160] +
            "。这一轮**接一句你知道的**（比如问一句后来怎样），一句就够；"
            "他如果不接这个话头，就顺着他的往下，不要再提第二次。" + NL
        )
    return (
        NL + "[上次没说完的事] 他之前提过：" + body[:160] +
        "。只有这一轮真的在聊这件事，才顺口问一句后来怎样；"
        "别的地方不要提，也不要把它当谈资复述。" + NL
    )


def extract_payload(parsed: dict) -> tuple[list[dict], list[str]]:
    """把模型输出收成 (要存的 thread, 要关掉的 thread_id)。

    只认 `open` 状态的；用户已经说清结果的走 `resolved`。
    """
    opened: list[dict] = []
    for item in (parsed.get("open") or []):
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        if not topic:
            continue
        opened.append({
            "topic": topic[:80],
            "detail": str(item.get("detail") or "").strip()[:240],
            "quote": str(item.get("quote") or "").strip()[:200],
        })
    closed = [str(x).strip() for x in (parsed.get("resolved") or []) if str(x).strip()]
    return opened[:5], closed[:5]
