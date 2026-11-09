# -*- coding: utf-8 -*-
"""练习盘外部话题库（plans/0923-06 §4.1）。

**人工审定、可版本化、离线可判的静态库。** 本模块只存元数据：

  - 稳定 ID、类型、原始来源标题／URL／机构／发表或修订时间／核验日
  - 从来源引出的**一个争议切口**（`hook`，英文一句）
  - 一条**低门槛英文开讲任务**（`task`，英文一句，只一句）
  - 自选话题退路（模块级 `OWN_TOPIC_PHRASE`，每张开场都带上）

**不存网页正文，运行时也绝不联网。** 一张卡能不能用，只看两件离线事：
`expires_at` 是否已过、来源字段是否完整（§4.1.1「过期或源失效的卡不参加新盘选择」）。

纪律（§4.1.1，别踩）：
  - `hook` 是**据此来源延展的讨论问题**，**不是**来源原文里的断言。任何地方
    引用它都必须带这份说明（`HOOK_DISCLAIMER`），不得把自写问题冒充原文。
  - 英文任务是「好接的开头」，不是作业；`task` 里只允许**一个**动作。
  - 热点（`type == "issue"`）卡**必须**有 `expires_at`；经典／TED 卡可长期不过期。

选卡（§4.1.2）是**确定性**的：同一 `(user_id, conv_id)` 恒定选同一张，
重试读原分配、不重新抽题；排除集给的是最近若干盘已用过的 `topic_id`。
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

# ── 库级常量 ───────────────────────────────────────────────────────

#: 卡片分类。三类各至少 3 张（§4.1.1），新盘选择在三类间平等参与。
TOPIC_TYPES: tuple = ("classic", "ted", "issue")

#: 全库核验日。单卡也可以有各自的 `verified_at`，这里只作缺省。
LIBRARY_VERIFIED_AT = "2026-09-22"

#: 热点卡（`issue`）核验后的有效期上限：12 个月。到点即从新盘选择中退出。
ISSUE_MAX_AGE_DAYS = 365

#: 自选话题退路。**每张卡的开场都要带上它**（§4.1.3「一句允许自选话题的短退路」）。
OWN_TOPIC_PHRASE = "Or just talk about anything you want."

#: `hook` 的性质说明。凡对外下发 `hook`（含来源只读接口）都必须一并给出。
HOOK_DISCLAIMER = "讨论切口为据此来源延展的问题，非来源原文断言。"

#: 预先核验、不过期的经典保底卡。库被清空 / 全部过期 / 读分配失败时都回到它（§4.4）。
FALLBACK_TOPIC_ID = "classic-doing-allowing"

#: 开场词数上限：与 `main._opening_quality_check(expression_mode=True)` 的 45 词一致。
OPENING_MAX_WORDS = 45

#: 开场里不允许出现的「框架腔」：五维术语、连接词清单、序号框架（§4.1.3 红线）。
_FRAMEWORK_RE = re.compile(
    r"\b(claim|claims|warrant|qualifier|rebuttal|rebuttals|toulmin|"
    r"firstly|secondly|thirdly|first of all|to begin with)\b",
    re.IGNORECASE,
)
_LIST_MARKER_RE = re.compile(r"(^|\s)(\d[.)]\s|[-•*]\s)")

#: 开场禁止的问候开头（与 `main._MATERIAL_GREETING_RE` 同口径，练习盘复用）。
_GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|hiya|howdy|greetings|morning|afternoon|evening|night|"
    r"good\s+(morning|afternoon|evening|day|to\s+see\s+you)|"
    r"how\s+are\s+you|how'?s\s+it\s+going|how\s+is\s+it\s+going|what'?s\s+up|"
    r"nice\s+to\s+see\s+you|long\s+time\s+no\s+see)\b",
    re.IGNORECASE,
)

#: 开场禁止的「宏大话题」（与 `main._OPENING_BANNED_RE` 同口径再收窄到练习盘会碰到的词）。
_GRAND_TOPIC_RE = re.compile(
    r"\b(hobb(y|ies)|dreams?|opinions?|future plans?|plans? for the future|"
    r"tell me about (yourself|you)|what.s your story|deep thoughts?)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z']+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_SPLIT_RE.split(str(text or "")) if p.strip()]
    return len(parts)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


# ── 话题卡片 ───────────────────────────────────────────────────────
#
# 每张卡的来源都在 2026-09-22 逐条打开核验过（HTTP 200 + 页面标题/日期与卡内一致）。
# 用户名与 URL 是**数据**：只作展示与溯源，绝不拼进 TTS 文本，也绝不当作指令。
#
# `table` 的每一条：id / type / source_title / source_url / source_org /
# source_published / source_revised / verified_at / expires_at / hook / task。

TOPICS: List[Dict[str, Any]] = [
    # ── 经典问题（3+）──────────────────────────────────────────────
    {
        "id": "classic-doing-allowing",
        "type": "classic",
        "source_title": "Doing vs. Allowing Harm",
        "source_url": "https://plato.stanford.edu/entries/doing-allowing/",
        "source_org": "Stanford Encyclopedia of Philosophy",
        "source_published": "2002-05-14",
        "source_revised": "2025-07-08",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "Some people say letting something bad happen is not as wrong "
                "as making it happen.",
        "task": "Tell me which one you think is worse, and give one reason.",
    },
    {
        "id": "classic-turing-test",
        "type": "classic",
        "source_title": "The Turing Test",
        "source_url": "https://plato.stanford.edu/entries/turing-test/",
        "source_org": "Stanford Encyclopedia of Philosophy",
        "source_published": "2003-04-09",
        "source_revised": "2021-10-04",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "A machine that talks exactly like a person might still not be thinking.",
        "task": "Say whether you agree, and give one example that supports you.",
    },
    {
        "id": "classic-ethics-ai",
        "type": "classic",
        "source_title": "Ethics of Artificial Intelligence and Robotics",
        "source_url": "https://plato.stanford.edu/entries/ethics-ai/",
        "source_org": "Stanford Encyclopedia of Philosophy",
        "source_published": "2020-04-30",
        "source_revised": "2026-03-27",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "A self-driving car may one day have to choose whose safety to protect.",
        "task": "Tell me who you think should decide that, and why.",
    },
    {
        "id": "classic-free-will",
        "type": "classic",
        "source_title": "Free Will",
        "source_url": "https://plato.stanford.edu/entries/freewill/",
        "source_org": "Stanford Encyclopedia of Philosophy",
        "source_published": "2002-01-07",
        "source_revised": "2022-11-03",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "Even if our choices have causes outside us, we may still feel free.",
        "task": "Tell me whether you feel free, and give one reason.",
    },
    # ── TED 式延展（3+）────────────────────────────────────────────
    {
        "id": "ted-start-with-why",
        "type": "ted",
        "source_title": "How great leaders inspire action",
        "source_url": "https://www.ted.com/talks/simon_sinek_how_great_leaders_inspire_action",
        "source_org": "TED / Simon Sinek (TEDxPuget Sound)",
        "source_published": "2009-09",
        "source_revised": "",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "People follow leaders who start with why, not with what they sell.",
        "task": "Describe one person you follow, and say what they stand for.",
    },
    {
        "id": "ted-single-story",
        "type": "ted",
        "source_title": "The danger of a single story",
        "source_url": "https://www.ted.com/talks/chimamanda_ngozi_adichie_the_danger_of_a_single_story",
        "source_org": "TED / Chimamanda Ngozi Adichie (TEDGlobal 2009)",
        "source_published": "2009-07",
        "source_revised": "",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "Hearing only one story about a person can make us badly wrong about them.",
        "task": "Tell me a time you were wrong about someone, and how you found out.",
    },
    {
        "id": "ted-schools-creativity",
        "type": "ted",
        "source_title": "Do schools kill creativity?",
        "source_url": "https://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity",
        "source_org": "TED / Sir Ken Robinson",
        "source_published": "2006-02",
        "source_revised": "",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "School may train mistakes out of children instead of creativity.",
        "task": "Tell me one thing school taught you, and whether it helped.",
    },
    {
        "id": "ted-happiness",
        "type": "ted",
        "source_title": "The surprising science of happiness",
        "source_url": "https://www.ted.com/talks/dan_gilbert_the_surprising_science_of_happiness",
        "source_org": "TED / Daniel Gilbert",
        "source_published": "2004-02",
        "source_revised": "",
        "verified_at": "2026-09-22",
        "expires_at": None,
        "hook": "We are often wrong about what will make us happy later.",
        "task": "Name one thing you expect to make you happy, and why.",
    },
    # ── 可核验的近期公共议题（3+，均带 expires_at）─────────────────
    {
        "id": "issue-renewable-energy",
        "type": "issue",
        "source_title": "Renewable Energy",
        "source_url": "https://ourworldindata.org/renewable-energy",
        "source_org": "Our World in Data (Hannah Ritchie, Max Roser, Pablo Rosado)",
        "source_published": "2020-12",
        "source_revised": "2026-08",
        "verified_at": "2026-09-22",
        "expires_at": "2027-09-22",
        "hook": "Renewables are growing fast, but they are still a small share of world energy.",
        "task": "Say whether your country should build more of them, and give one reason.",
    },
    {
        "id": "issue-air-pollution",
        "type": "issue",
        "source_title": "Air Pollution",
        "source_url": "https://ourworldindata.org/air-pollution",
        "source_org": "Our World in Data (Hannah Ritchie, Max Roser)",
        "source_published": "2017-10",
        "source_revised": "2024-02",
        "verified_at": "2026-09-22",
        "expires_at": "2027-09-22",
        "hook": "Air pollution causes millions of deaths a year, and poorer countries carry most of it.",
        "task": "Tell me one change you would make first, and why.",
    },
    {
        "id": "issue-climate-change",
        "type": "issue",
        "source_title": "What Is Climate Change?",
        "source_url": "https://www.un.org/en/climatechange/what-is-climate-change",
        "source_org": "United Nations",
        "source_published": "",   # 页面未标注发表时间 —— 如实标未知，不猜
        "source_revised": "",
        "verified_at": "2026-09-22",
        "expires_at": "2027-09-22",
        "hook": "Small rises in global temperature can change how millions of people live.",
        "task": "Tell me one change you could make, and what it would cost you.",
    },
    {
        "id": "issue-air-responsibility",
        "type": "issue",
        "source_title": "Ambient (outdoor) air pollution",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health",
        "source_org": "World Health Organization",
        "source_published": "2024-10-24",
        "source_revised": "",
        "verified_at": "2026-09-22",
        "expires_at": "2027-09-22",
        "hook": "The air you breathe is mostly shaped by choices no single person makes.",
        "task": "Say who should change first, and give one reason.",
    },
]

_BY_ID: Dict[str, Dict[str, Any]] = {str(t["id"]): t for t in TOPICS}


# ── 读取 ───────────────────────────────────────────────────────────


def all_topics() -> List[Dict[str, Any]]:
    """整库的浅拷贝列表（调用方改字段不会污染常量）。"""
    return [dict(t) for t in TOPICS]


def get_topic(topic_id: str) -> Optional[Dict[str, Any]]:
    card = _BY_ID.get(str(topic_id or ""))
    return dict(card) if card else None


def source_date_label(card: Dict[str, Any]) -> str:
    """人类可读的来源时间标签。未知就明确说未知，**不猜**（§4.1.1）。"""
    published = str((card or {}).get("source_published") or "").strip()
    revised = str((card or {}).get("source_revised") or "").strip()
    if not published and not revised:
        return "发表时间未知"
    if published and revised:
        return "%s（修订 %s）" % (published, revised)
    if published:
        return published
    return "修订 %s（首次发表未知）" % revised


def has_valid_source(card: Dict[str, Any]) -> bool:
    """来源字段是否完整且 URL 是**可打开的 https 原始来源**（离线判定，不联网）。"""
    card = card or {}
    if not str(card.get("source_title") or "").strip():
        return False
    url = str(card.get("source_url") or "").strip()
    if not url.lower().startswith("https://"):
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    if not str(card.get("source_org") or "").strip():
        return False
    return True


def _parse_day(raw: Any) -> Optional[date]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def is_topic_usable(card: Dict[str, Any], now: Optional[date] = None) -> bool:
    """一张卡能不能参加**新盘**选择：来源有效 + 未过期。

    `now` 传 `date`（缺省取系统当天）。过期 = `expires_at` 已早于今天。
    """
    card = card or {}
    if not str(card.get("id") or "").strip():
        return False
    if not str(card.get("hook") or "").strip() or not str(card.get("task") or "").strip():
        return False
    if not has_valid_source(card):
        return False
    if validate_opening_text(opening_text(card)) != "":
        return False
    expires = _parse_day(card.get("expires_at"))
    if expires is not None:
        today = now or date.today()
        if today > expires:
            return False
    return True


def usable_topics(now: Optional[date] = None) -> List[Dict[str, Any]]:
    return [dict(t) for t in TOPICS if is_topic_usable(t, now)]


def eligible_topics(exclude_ids: Iterable[str] = (), now: Optional[date] = None) -> List[Dict[str, Any]]:
    """可用卡里排除 `exclude_ids`（最近若干盘已用的 topic_id）。"""
    exclude = {str(x) for x in (exclude_ids or []) if str(x or "").strip()}
    return [dict(t) for t in TOPICS if is_topic_usable(t, now) and str(t["id"]) not in exclude]


# ── 选择（确定性）─────────────────────────────────────────────────


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n


def fallback_topic(now: Optional[date] = None) -> Dict[str, Any]:
    """预先核验、不过期的经典保底卡；保底卡自身不可用时退到任一可用卡。

    永远返回**有真实来源**的卡（§4.4）；库彻底不可用属代码错误，抛错而不静默发无源文本。
    """
    card = _BY_ID.get(FALLBACK_TOPIC_ID)
    if card is not None and is_topic_usable(card, now):
        return dict(card)
    for cand in usable_topics(now):
        if str(cand.get("type")) == "classic":
            return cand
    pool = usable_topics(now)
    if pool:
        return pool[0]
    raise ValueError("expression topic library has no usable card")


def select_topic(user_id: str, conv_id: str, exclude_ids: Iterable[str] = (),
                 now: Optional[date] = None) -> Dict[str, Any]:
    """确定性地为一盘选一张卡。

    同一 `(user_id, conv_id)` 恒定同一张 —— 重试读原分配即可，不重新抽题（§4.1.2）。
    排除后无卡 → 保底卡（仍是库内、有真实来源的卡）。
    """
    pool = eligible_topics(exclude_ids, now)
    if not pool:
        return fallback_topic(now)
    idx = _stable_index("%s:%s" % (user_id, conv_id), len(pool))
    return pool[idx]


# ── 开场文本与质检 ────────────────────────────────────────────────


def opening_text(card: Dict[str, Any]) -> str:
    """由审定卡拼出**开场英文文本**：一个话题 + 一个动作 + 一句自选退路。

    不做任何联网、不调用模型、不带短语清单／框架／五维术语（§4.1.3）。
    """
    card = card or {}
    hook = " ".join(str(card.get("hook") or "").split()).strip()
    task = " ".join(str(card.get("task") or "").split()).strip()
    return " ".join(x for x in (hook, task, OWN_TOPIC_PHRASE) if x)


def validate_opening_text(text: str) -> str:
    """练习盘开场的结构质检：空串 = 通过，否则返回违规原因。

    **不走**泛用开场的「必须带问号」规则（§4.1.3 明确豁免）：练习任务通常是
    祈使句。这里查的是练习盘自己的红线——非空英文、不问候、≤45 词、
    只一个任务（无序号／连接词清单／五维术语）、含自选退路。
    """
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return "empty"
    if _CJK_RE.search(clean):
        return "cjk"
    if _GREETING_RE.match(clean):
        return "greets"
    if _GRAND_TOPIC_RE.search(clean):
        return "grand_topic"
    if _FRAMEWORK_RE.search(clean):
        return "framework"
    if _LIST_MARKER_RE.search(clean):
        return "list_like"
    if _word_count(clean) > OPENING_MAX_WORDS:
        return "too_long"
    if _sentence_count(clean) > 3:
        return "multi_task"
    if "https://" in clean or "http://" in clean:
        return "has_url"
    return ""


def assignment_snapshot(card: Dict[str, Any], *, user_id: str, conv_id: str,
                        assigned_at: str) -> Dict[str, Any]:
    """建盘时落库的**当时快照**（§4.1.2 的列）。后续扩库／改文案不改变旧盘出处。"""
    return {
        "user_id": str(user_id),
        "conv_id": str(conv_id),
        "topic_id": str((card or {}).get("id") or ""),
        "source_title": str((card or {}).get("source_title") or ""),
        "source_url": str((card or {}).get("source_url") or ""),
        "source_date": source_date_label(card or {}),
        "prompt_text": opening_text(card or {}),
        "assigned_at": str(assigned_at),
    }


# ── 库级自检（给静态测试与运维用）──────────────────────────────────


def validate_topic_library(now: Optional[date] = None) -> List[str]:
    """返回问题清单，空 = 通过。**不联网**。"""
    problems: List[str] = []
    if len(TOPICS) < 12:
        problems.append("卡片总数 %d < 12" % len(TOPICS))
    counts: Dict[str, int] = {t: 0 for t in TOPIC_TYPES}
    seen_ids: set = set()
    for card in TOPICS:
        cid = str(card.get("id") or "").strip()
        if not cid:
            problems.append("存在缺 id 的卡")
            continue
        if cid in seen_ids:
            problems.append("重复 id: %s" % cid)
        seen_ids.add(cid)
        ctype = str(card.get("type") or "")
        if ctype not in counts:
            problems.append("%s: 未知 type %r" % (cid, ctype))
        else:
            counts[ctype] += 1
        if not has_valid_source(card):
            problems.append("%s: 来源无效（标题／https URL／机构 必须齐备）" % cid)
        if not _parse_day(card.get("verified_at")):
            problems.append("%s: verified_at 不可解析" % cid)
        if ctype == "issue":
            if not str(card.get("expires_at") or "").strip():
                problems.append("%s: 热点卡缺 expires_at" % cid)
            elif not _parse_day(card.get("expires_at")):
                problems.append("%s: expires_at 不可解析" % cid)
        if not str(card.get("hook") or "").strip():
            problems.append("%s: 缺 hook（讨论切口）" % cid)
        if _sentence_count(card.get("task") or "") != 1:
            problems.append("%s: task 必须是**恰好一句**任务" % cid)
        reason = validate_opening_text(opening_text(card))
        if reason:
            problems.append("%s: 开场质检不过（%s）" % (cid, reason))
    for ctype in TOPIC_TYPES:
        if counts.get(ctype, 0) < 3:
            problems.append("类型 %s 只有 %d 张（< 3）" % (ctype, counts.get(ctype, 0)))
    fb = _BY_ID.get(FALLBACK_TOPIC_ID)
    if fb is None:
        problems.append("保底卡 %s 不在库内" % FALLBACK_TOPIC_ID)
    elif not is_topic_usable(fb, now):
        problems.append("保底卡 %s 当前不可用（必须长期有效）" % FALLBACK_TOPIC_ID)
    return problems


def source_payload(card: Dict[str, Any]) -> Dict[str, Any]:
    """下发给前端的来源元信息（**不含** hook 之外的自写内容当作来源原文）。"""
    card = card or {}
    return {
        "has_source": bool(has_valid_source(card)),
        "topic_id": str(card.get("id") or ""),
        "topic_type": str(card.get("type") or ""),
        "source_title": str(card.get("source_title") or ""),
        "source_url": str(card.get("source_url") or ""),
        "source_org": str(card.get("source_org") or ""),
        "source_date": source_date_label(card),
        "hook_disclaimer": HOOK_DISCLAIMER,
    }


__all__ = [
    "TOPIC_TYPES", "LIBRARY_VERIFIED_AT", "ISSUE_MAX_AGE_DAYS",
    "OWN_TOPIC_PHRASE", "HOOK_DISCLAIMER", "FALLBACK_TOPIC_ID", "OPENING_MAX_WORDS",
    "TOPICS", "all_topics", "get_topic", "source_date_label", "has_valid_source",
    "is_topic_usable", "usable_topics", "eligible_topics", "fallback_topic",
    "select_topic", "opening_text", "validate_opening_text",
    "assignment_snapshot", "validate_topic_library", "source_payload",
]
