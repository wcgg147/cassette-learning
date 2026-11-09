"""AI 情绪层（plans/0919-03-情绪层.md §4.1–4.5）。

用户口径（2026-09-19）："AI 有稳定的性格，也会因为发生的事有此刻的状态……而这种
状态只从它**说什么、怎么追问、用什么语气**里露出来，不会出现'我现在很开心'这种
自我汇报"；"情绪要准，不要一会儿开心一会儿惊讶"。

所以本模块与 `persona_router.py` 同构：**纯函数、确定性、零模型调用、零额外延迟**。
交给模型每轮现抽情绪，就必然变成用户否掉的"情绪标签随机抽取器"——情绪只能由
**事件表**推导，强度只能靠**衰减**回落。

三层边界（§4.1，不可越界）：
  - 人格底层（`persona_router` 三态）：不变、不可塑——决定"我是谁"；
  - 情绪层（本模块）：**轮级**、事件驱动、可衰减——决定"这一轮什么状态"；
  - 关系层（`memory_curator` / `profile_evolution`)：长期、被用户塑造——决定
    "我们之间积累了什么"。
情绪状态**不得写进 relationship_memory**：那是 per-user 的长期画像（且
`recent_mood` 槽位已被"用户的心情"占用），写进去既是串盘也是污染可塑层。

主情绪集合刻意**少而稳**（§4.2）：高兴 / 关心 / 好奇 + 平静（默认，不表达）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ── 主情绪（稳定、可表达的当轮状态） ────────────────────────────────────
HAPPY = "happy"
CARE = "care"
CURIOUS = "curious"
ANNOYED = "annoyed"
CALM = "calm"  # 默认：不表达，逐字零增量

EXPRESSIBLE: Tuple[str, ...] = (HAPPY, CARE, CURIOUS, ANNOYED)

# 同强度时的优先序（§4.2 未规定，本实现自定并说明）：care 优先——
# 用户需要被照顾时，AI 的"我为他高兴"不该盖过"先接住他"。
_PRIORITY: Tuple[str, ...] = (CARE, ANNOYED, HAPPY, CURIOUS)

# ── 事件表（§4.2） ─────────────────────────────────────────────────────
# 确定性事件 → (情绪, 强度增量)。事件名=调用方给的口径，权重是待标定项
# （§4.2："权重与衰减率是本档的待标定项"），落地后拿真机对话回填。
#
# **本表不含"用户附和/赞同/夸奖"**：那类事件不产生任何加分（§4.3）——这是
# "不变成只会迎合用户的镜子"在数值上的落地，有专门用例守着。
EMOTION_EVENTS: Dict[str, Tuple[str, float]] = {
    # 正向：用户真的做成了（探测答对 / 确认掌握）
    "probe_correct": (HAPPY, 0.50),
    "mastered_expression": (HAPPY, 0.30),
    "user_shared_good_news": (HAPPY, 0.45),
    # The user's reassurance can be an emotionally meaningful relational event
    # when Alice explicitly disclosed her own state in the immediately prior turn.
    "user_reassured_ai": (HAPPY, 0.35),
    # 关心：用户卡住 / 受挫 / 想退出
    "user_stuck": (CARE, 0.45),
    "user_frustrated": (CARE, 0.50),
    "user_disengaged": (CARE, 0.35),
    # 好奇：用户主动讲了新东西
    "new_info_shared": (CURIOUS, 0.35),
    # Alice 的休息/精力被明确、非紧急地打断时，允许出现克制的不快。
    # 普通晚聊、时间晚、或用户只是在开玩笑都不单独构成事件。
    "user_imposed_nonurgent_disruption": (ANNOYED, 0.45),
    # Alice 刚明确说需要休息，用户仍明确要求她继续时，强度略高。
    "user_disregarded_ai_rest_boundary": (ANNOYED, 0.55),
}

_GOOD_NEWS_RE = re.compile(
    r"(?:拿到|收到|获得).{0,10}(?:offer|工作|录取通知|证书|合同|项目|奖项)"
    r"|(?:通过|考上).{0,8}(?:考试|面试|审核|考核|大学|项目)"
    r"|(?:升职|晋升|签下.{0,6}(?:合同|项目))"
    r"|(?:我(?:终于)?(?:成功了|做到了|完成了|通过了))"
    r"|\b(?:i got the job|got an offer|i passed|i got promoted|we won)\b",
    re.IGNORECASE,
)
_GOOD_NEWS_NEGATION_RE = re.compile(
    r"(?:没拿到|没有拿到|没收到|没有收到|未通过|没通过|没考上|没成功|没做到|没完成|"
    r"didn't get|did not get|haven't got|didn't pass|did not pass)",
    re.IGNORECASE,
)


def detect_explicit_good_news(text: str) -> bool:
    """只为清晰的本人好消息触发当轮高兴，不把赞同或夸奖当事件。"""
    raw = str(text or "")
    return bool(_GOOD_NEWS_RE.search(raw) and not _GOOD_NEWS_NEGATION_RE.search(raw))

DECAY_PER_TURN = 0.8  # 待标定
EXPRESS_THRESHOLD = 0.30  # 低于此值不表达 → 回平静（§4.3）
MAX_INTENSITY = 1.0

# ── §4.5 用户情绪：概率性状态，不是断言 ────────────────────────────────
# 措辞必须是"用户可能有点不耐烦（置信度较高）"，不是"用户就是生气"。
# 低于 ASSERT_THRESHOLD 按中性处理——**低置信不许硬断言**。
ASSERT_THRESHOLD = 0.60
HIGH_CONFIDENCE = 0.75

# WPM 只作系数、不作独立信号（§4.5：语音五信号里只有语速可用）。
WPM_FAST = 200.0
WPM_SLOW = 60.0
WPM_CONFIDENCE_BONUS = 0.10

USER_EMOTION_LABEL = {
    "frustrated": "有点不耐烦或受挫",
    "disengaged": "可能有点想停下来了",
}


# ---------------------------------------------------------------------------
# 状态：{intensities, main, intensity, last_turn}
# ---------------------------------------------------------------------------

def new_state() -> Dict[str, Any]:
    """初始状态：平静（不表达）。"""
    return {
        "intensities": {}, "main": CALM, "intensity": 0.0,
        "last_turn": 0, "event_history": [],
    }


def _recompute(state: Dict[str, Any]) -> Dict[str, Any]:
    """按当前 intensities 重算主情绪；全部低于阈值即回平静。"""
    intensities = {
        emotion: float(value)
        for emotion, value in (state.get("intensities") or {}).items()
        if emotion in EXPRESSIBLE and float(value) >= EXPRESS_THRESHOLD
    }
    state["intensities"] = intensities
    if not intensities:
        state["main"] = CALM
        state["intensity"] = 0.0
        return state
    best = max(intensities.values())
    # 同分按 _PRIORITY 取（确定性，不吃 dict 序）
    main = next(e for e in _PRIORITY if intensities.get(e, 0.0) == best)
    state["main"] = main
    state["intensity"] = round(best, 4)
    return state


def apply_events(state: Dict[str, Any], events: Iterable[str]) -> Dict[str, Any]:
    """按事件表累加强度（未知事件名静默忽略——调用方口径变化不该炸链路）。"""
    out = {
        "intensities": dict((state or {}).get("intensities") or {}),
        "main": (state or {}).get("main") or CALM,
        "intensity": float((state or {}).get("intensity") or 0.0),
        "last_turn": int((state or {}).get("last_turn") or 0),
        "event_history": list((state or {}).get("event_history") or [])[-6:],
    }
    for name in events or ():
        entry = EMOTION_EVENTS.get(str(name))
        if not entry:
            continue
        emotion, delta = entry
        out["intensities"][emotion] = min(
            MAX_INTENSITY, float(out["intensities"].get(emotion, 0.0)) + delta
        )
    return _recompute(out)


def decay(state: Dict[str, Any], turns: int = 1) -> Dict[str, Any]:
    """逐轮衰减（§4.3）。衰减到阈值以下即回平静——保证"同一件事过去几轮后不再影响回复"。"""
    out = {
        "intensities": dict((state or {}).get("intensities") or {}),
        "main": (state or {}).get("main") or CALM,
        "intensity": float((state or {}).get("intensity") or 0.0),
        "last_turn": int((state or {}).get("last_turn") or 0),
        "event_history": list((state or {}).get("event_history") or [])[-6:],
    }
    for _ in range(max(0, int(turns))):
        out["intensities"] = {
            emotion: value * DECAY_PER_TURN
            for emotion, value in out["intensities"].items()
        }
    return _recompute(out)


def advance(
    state: Dict[str, Any], events: Iterable[str] = (), turns_elapsed: int = 1,
    turn: int = 0,
) -> Dict[str, Any]:
    """一轮的完整推进：先按过去的轮数衰减，再加本轮事件。

    顺序固定（先衰减后加分），否则"同一事件连续两轮触发"会被当成一次。
    """
    previous_turn = int((state or {}).get("last_turn") or 0)
    elapsed = max(1, int(turns_elapsed))
    out = decay(state, elapsed)
    resolved_turn = int(turn or previous_turn + elapsed)
    event_history = list(out.get("event_history") or [])
    new_trace = []
    kept_events = []
    for name in events or ():
        event_name = str(name)
        mapped = EMOTION_EVENTS.get(event_name)
        if not mapped:
            continue
        # 冷却：new_info_shared 是"用户主动说话"的宽判据（正常闲聊几乎每轮成立），
        # 不设冷却会让她恒为 curious，而 curious 的指令是"追问一个细节"——
        # 反过来把人推回问句。真人也是隔几轮才好奇一次，不是句句追问。
        if event_name == "new_info_shared":
            _recent = [
                h for h in event_history
                if isinstance(h, dict)
                and resolved_turn - int(h.get("turn") or 0) <= 3
                and str(h.get("event") or "") == "new_info_shared"
            ]
            if _recent:
                continue
        emotion, _delta = mapped
        kept_events.append(event_name)
        new_trace.append({"event": event_name, "emotion": emotion, "turn": resolved_turn})
    # ⚠ 必须用过滤后的列表：只 continue 掉 new_trace 的话，apply_events 仍会按
    # 原始 events 加分，冷却就只挡了"记录"没挡"生效"（2026-10-26 踩过）。
    out = apply_events(out, kept_events)
    event_history.extend(new_trace)
    recent_history = []
    for item in event_history:
        if not isinstance(item, dict):
            continue
        try:
            event_turn = int(item.get("turn") or 0)
        except (TypeError, ValueError):
            continue
        if resolved_turn - event_turn <= 6:
            recent_history.append({
                "event": str(item.get("event") or ""),
                "emotion": str(item.get("emotion") or ""),
                "turn": event_turn,
            })
    out["event_history"] = recent_history[-8:]
    out["last_turn"] = resolved_turn
    return out


def state_from_raw(raw: Any) -> Dict[str, Any]:
    """把持久层读回的东西收敛成合法状态（脏数据一律回平静，不炸链路）。

    handoff 是从库里读回来的，可能带旧版本结构或被改坏——这里逐个字段
    收敛（类型不对就当没给），而不是信它。
    """
    if not isinstance(raw, dict):
        return new_state()
    raw_intensities = raw.get("intensities")
    intensities: Dict[str, float] = {}
    if isinstance(raw_intensities, dict):
        for emotion, value in raw_intensities.items():
            if emotion not in EXPRESSIBLE:
                continue
            try:
                intensities[emotion] = float(value)
            except (TypeError, ValueError):
                continue
    try:
        last_turn = int(raw.get("last_turn") or 0)
    except (TypeError, ValueError):
        last_turn = 0
    event_history = []
    for item in raw.get("event_history") or []:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event") or "")
        mapped = EMOTION_EVENTS.get(event)
        if not mapped:
            continue
        try:
            event_turn = max(0, int(item.get("turn") or 0))
        except (TypeError, ValueError):
            continue
        event_history.append({"event": event, "emotion": mapped[0], "turn": event_turn})
    out = {
        "intensities": intensities,
        "main": str(raw.get("main") or CALM),
        "intensity": 0.0,
        "last_turn": last_turn,
        "event_history": event_history[-8:],
    }
    if out["main"] not in EXPRESSIBLE:
        out["main"] = CALM
    return _recompute(out)


# ---------------------------------------------------------------------------
# 事件解析（§4.2：只认确定性事件）
# ---------------------------------------------------------------------------

def resolve_events(
    *,
    markers: Optional[Dict[str, Any]] = None,
    affective_state: str = "",
    engagement: str = "",
    probe_result: str = "",
) -> List[str]:
    """从本轮**已经算好**的确定性信号里挑事件（零新增计算、零模型调用）。

    信号来源（全部现成）：
      - ``markers``：``_g1_turn_markers`` 的产物（卡壳 / 主动发起）；
      - ``affective_state`` / ``engagement``：``learner_state.estimate``；
      - ``probe_result``：``learning_outcomes.score_probe`` 的 result。
    """
    events: List[str] = []
    markers = markers or {}
    stuck = markers.get("stuck") or []
    if "missing_word_help" in stuck:
        events.append("user_stuck")
    if str(affective_state or "") == "frustrated":
        events.append("user_frustrated")
    if str(affective_state or "") == "disengaged" or str(engagement or "") == "leaving":
        events.append("user_disengaged")
    if str(probe_result or "") == "correct":
        events.append("probe_correct")
    if markers.get("good_news"):
        events.append("user_shared_good_news")
    # 好奇的来源：用户**主动**讲起自己的东西。
    # ⚠ 不能用导演层的 topic_transition：闲聊盘 CHAT_TAPE_DIRECTOR_ENABLED=False，
    # 那边走 fallback_plan，topic_transition 恒为 "continue"（2026-10-26 查证）。
    # 真正的"别每轮都好奇"由 advance() 里的冷却窗口负责。
    if markers.get("user_initiated") and not markers.get("fragment"):
        events.append("new_info_shared")
    return events


def emotion_markers(
    *,
    assistant_response: str = "",
    state: Optional[Dict[str, Any]] = None,
    events: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """逐轮事后标记（§4.4）：只判定 + 写标记，**绝不改写回复**。

    与 `_g1_turn_markers` 同一纪律：结果只进 metering metadata 供出数与调优，
    不替换、不重试、不新增调用。
    """
    violations: List[str] = []
    if detect_emotion_self_report(assistant_response):
        violations.append("emotion_self_report")
    out: Dict[str, Any] = {
        "state": str((state or {}).get("main") or CALM),
        "intensity": round(float((state or {}).get("intensity") or 0.0), 4),
        "events": list(events or []),
    }
    if violations:
        out["violations"] = violations
    return out


# 情绪宣告（§4.4 硬闸门）：**汇报自己的状态**，不是描述用户、也不是自然反应。
# 判据刻意窄——只认第一人称状态汇报的典型形态；"I'm glad you liked it" 这类
# 自然反应不算（口径与 0917-01 一致：自然措辞归模型，闸门只拦硬违规）。
_EMOTION_SELF_REPORT_RE = re.compile(
    r"我现在(?:很|挺|特别|真的|好)?(?:开心|高兴|兴奋|难过|着急|心疼|担心)"
    r"|我(?:很|好|特别|真)(?:关心你|替你高兴|为你高兴)"
    r"|我的心情|我此刻(?:很|挺|特别)?(?:开心|高兴|难过|着急)"
    r"|\bI(?:'m| am)\s+(?:so\s+|very\s+|really\s+|quite\s+)?"
    r"(?:happy|excited|thrilled|delighted|proud)\b"
    r"|\bI\s+care\s+about\s+you\b"
    r"|\bI(?:'m| am)\s+here\s+for\s+you\b"
    r"|\bmy\s+(?:mood|feelings?\s+right\s+now)\b",
    re.IGNORECASE,
)


def detect_emotion_self_report(text: str) -> bool:
    """是不是"情绪宣告"（自我状态汇报）。"""
    return bool(_EMOTION_SELF_REPORT_RE.search(str(text or "")))


_DIRECT_AI_INSULT_RE = re.compile(
    r"\b(?:fuck\s+you|you\s+(?:are\s+)?(?:a\s+|an\s+)?"
    r"(?:idiot|moron|stupid|useless|piece\s+of\s+shit))\b"
    r"|(?:你|AI|助手|机器人).{0,6}(?:蠢货|傻逼|废物|垃圾|脑残|白痴)"
    r"|(?:蠢货|傻逼|废物|垃圾|脑残|白痴).{0,6}(?:你|AI|助手|机器人)",
    re.IGNORECASE,
)


_AI_SELF_STATE_DISCLOSURE_RE = re.compile(
    r"\bI(?:['’]m| am)\s+(?:(?:a little|a bit|quite|pretty|kind of)\s+)?"
    r"(?:tired|exhausted|sleepy|worried|nervous|down)\b"
    r"|\bHonestly[,:?]?\s+(?:a little|a bit|quite|pretty|kind of)\s+"
    r"(?:tired|exhausted|sleepy|worried|nervous|down)\b"
    r"|我(?:现在)?(?:有点|有些|挺|很)?(?:累|疲惫|困|担心|紧张|难受)" ,
    re.IGNORECASE,
)
_USER_REASSURANCE_RE = re.compile(
    r"\b(?:don't worry|do not worry|we can rest|you can rest|let's rest|"
    r"we can call it|you can call it|we can sleep|you should get some sleep|"
    r"we can finish.{0,50}(?:rest|sleep))\b"
    r"|(?:不用担心|你先休息|我们都可以休息|然后.{0,20}(?:休息|睡觉)|"
    r"做完.{0,20}(?:休息|睡觉)|工作结束.{0,20}(?:休息|睡觉))",
    re.IGNORECASE,
)
_ALICE_REST_BOUNDARY_RE = re.compile(
    r"\bI(?:['’]m| am)\s+(?:(?:a little|a bit|quite|pretty|kind of)\s+)?"
    r"(?:tired|exhausted|sleepy)\b.{0,80}\b(?:rest|sleep|stop)\b|"
    r"\bI\s+(?:need|want|should)\s+to\s+(?:rest|sleep|go to bed)\b|"
    r"\bcan we (?:stop|call it|pick this up)\b.{0,50}\b(?:tomorrow|later|after I rest)\b|"
    r"我(?:有点|有些|很)?(?:累|困|疲惫).{0,30}(?:休息|睡觉|先停)|"
    r"我(?:想|需要|该)先(?:休息|睡觉)",
    re.IGNORECASE,
)
_USER_IGNORES_REST_BOUNDARY_RE = re.compile(
    r"\b(?:no|but)\b.{0,40}\b(?:keep talking|keep going|stay up|don't sleep|do not sleep)\b|"
    r"\bjust one more (?:thing|question|minute)\b|"
    r"\bwe(?:'re| are) not done yet\b|"
    r"(?:别睡|别休息|继续聊|再聊(?:一会|一下)|就再问一个|还没聊完)",
    re.IGNORECASE,
)
_WAKE_OR_LATE_CALL_RE = re.compile(
    r"\bI\s+(?:woke you up|woke you|called you|rang you|texted you)\b|"
    r"\b(?:woke you up|woke you)\b|"
    r"我(?:把你|刚才)?(?:吵醒|叫醒)|我(?:打电话|打给你|给你发消息)",
    re.IGNORECASE,
)
_LATE_HOUR_RE = re.compile(
    r"\b(?:[0-5](?::[0-5]\d)?\s*(?:a\.?m\.?|in the morning)|"
    r"middle of the night|middle of the morning|very late|late at night)\b|"
    r"凌晨|半夜|深夜",
    re.IGNORECASE,
)
_NONURGENT_CHAT_RE = re.compile(
    r"\b(?:just|only)\s+(?:because\s+)?I\s+(?:wanted|want|called)\s+to\s+(?:talk|chat)\b|"
    r"\bI\s+(?:just|only)\s+(?:wanted|want)\s+to\s+(?:talk|chat)\b|"
    r"\b(?:nothing|not anything)\s+(?:urgent|important)\b|"
    r"\bjust for (?:a chat|talk)\b|"
    r"只是(?:想|想要)?(?:聊聊天|聊天|找你聊)|没什么急事",
    re.IGNORECASE,
)
_URGENT_CONTEXT_RE = re.compile(
    r"\b(?:emergency|accident|danger|hospital|ambulance|police|help me)\b|"
    r"出事|危险|医院|救护车|警察",
    re.IGNORECASE,
)
_URGENT_TERM_RE = re.compile(r"\burgent\b|紧急|急事", re.IGNORECASE)
_NONURGENT_URGENT_RE = re.compile(
    r"\b(?:nothing|not anything|not|no|isn't|is not)\s+"
    r"(?:(?:at all|really)\s+)?urgent\b|"
    r"(?:不|并不|不是|没什么|没有).{0,3}(?:紧急|急事)",
    re.IGNORECASE,
)


def _has_urgent_context(text: str) -> bool:
    """Treat explicit emergencies as urgent, but respect negated "urgent" wording."""
    if _URGENT_CONTEXT_RE.search(text):
        return True
    return bool(
        _URGENT_TERM_RE.search(text)
        and not _NONURGENT_URGENT_RE.search(text)
    )


def resolve_companion_events(user_text: str, previous_assistant_text: str) -> List[str]:
    """Recognize explicit, attributable events in the Alice/user interaction.

    Reassurance and ignored-boundary events require Alice's immediately preceding
    self-disclosure. A late disruption requires the user to state both the wake/call
    and non-urgent intent; time of day by itself never makes Alice annoyed.
    """
    previous = str(previous_assistant_text or "")[:1200]
    current = str(user_text or "")[:1200]
    events: List[str] = []
    if (_AI_SELF_STATE_DISCLOSURE_RE.search(previous)
            and _USER_REASSURANCE_RE.search(current)):
        events.append("user_reassured_ai")
    disregards_rest = bool(
        _ALICE_REST_BOUNDARY_RE.search(previous)
        and _USER_IGNORES_REST_BOUNDARY_RE.search(current)
    )
    disrupted_for_chat = bool(
        _WAKE_OR_LATE_CALL_RE.search(current)
        and _NONURGENT_CHAT_RE.search(current)
        and (
            _LATE_HOUR_RE.search(current)
            or re.search(r"\b(?:woke you up|woke you)\b|我(?:把你|刚才)?(?:吵醒|叫醒)", current, re.I)
        )
        and not _has_urgent_context(current)
    )
    if disregards_rest:
        events.append("user_disregarded_ai_rest_boundary")
    elif disrupted_for_chat:
        events.append("user_imposed_nonurgent_disruption")
    return events


def detect_direct_ai_insult(text: str) -> bool:
    """只识别明确指向 AI 本人的直接辱骂；普通批评和引述不触发。"""
    raw = str(text or "")
    unquoted = re.sub(r"[\"'“‘「『].*?[\"'”’」』]", " ", raw)
    if re.search(r"(?:不是说|没说|并非|not calling|didn't call)\s*(?:你|AI|助手|机器人|you)", unquoted, re.I):
        return False
    return bool(_DIRECT_AI_INSULT_RE.search(unquoted))


# ---------------------------------------------------------------------------
# 渲染（§4.4：同一情绪，三个人格不同式样）
# ---------------------------------------------------------------------------

# 情绪 → (这一轮该做什么)。情绪必须**影响这一轮说什么**，不只是语气装饰
# （用户明确要求，§4 第 3 期）。
_EMOTION_ACTS: Dict[str, Tuple[str, str]] = {
    HAPPY: (
        "你此刻确实在为他高兴",
        "把这份高兴放进一个具体回应里：点出他刚做到的那件事本身，"
        "别用空泛的夸奖顶替（「你真棒」这类不算）",
    ),
    CARE: (
        "你此刻有点替他着急、想先照顾好他",
        "放慢，少推进：这一轮以接住他此刻的处境为主，不必塞教学动作，"
        "也不要催他继续",
    ),
    CURIOUS: (
        "你此刻对他刚讲的东西是真的感兴趣",
        "顺着他的话题追问一个真有内容的细节，只问一个，别连环问",
    ),
    ANNOYED: (
        "你对刚才那次明确的非紧急打断或被忽略的休息边界有一点不快",
        "用简短、克制的反应让这点不快能被听出来；不要把它改写成感动或说没关系，"
        "也不要羞辱用户、拒绝继续聊天或擅自换任务。表达后仍回应用户这轮真正说的内容",
    ),
}

# 人格决定"怎么表现"同一种情绪（§4.4）。三个 key 与 persona_router 的三态一致。
_PERSONA_STYLES: Dict[str, str] = {
    "companion": "朋友之间可以直接一点，允许带一点幽默，但别夸张、别煽情。",
    "scene_role": "只能用角色内的话和动作表达（店里的事就说店里的话）；"
                  "不要跳出来当外教，也不要用旁白口吻解释自己。",
    "tutor_sidecar": "克制：这一轮的重点是答清语言问题，情绪只淡淡带过。",
}


def render_emotion_section(state: Dict[str, Any], persona_mode: str = "companion") -> str:
    """``[Emotion]`` 段文本；平静时返回空串（普通轮逐字零增量）。"""
    current = state_from_raw(state)
    main = current.get("main")
    if main not in _EMOTION_ACTS:
        return ""
    if float(current.get("intensity") or 0.0) < EXPRESS_THRESHOLD:
        return ""
    feel, act = _EMOTION_ACTS[main]
    style = _PERSONA_STYLES.get(str(persona_mode or ""), _PERSONA_STYLES["companion"])
    event_labels = {
        "probe_correct": "用户这轮答对了",
        "mastered_expression": "用户确认掌握了一项表达",
        "user_shared_good_news": "用户分享了明确的好消息",
        "user_reassured_ai": "用户回应了你上一轮明确说出的疲惫或担忧",
        "user_stuck": "用户刚才遇到了表达卡壳",
        "user_frustrated": "用户刚才表现出受挫",
        "user_disengaged": "用户可能想停下或退出",
        "new_info_shared": "用户主动分享了新的个人信息",
        "user_imposed_nonurgent_disruption": "用户明确说自己在非紧急情况下打断了你的休息",
        "user_disregarded_ai_rest_boundary": "你明确说需要休息后，用户仍要求你继续",
    }
    matching_causes = [
        item for item in current.get("event_history") or []
        if isinstance(item, dict) and item.get("emotion") == main
    ]
    causal_note = ""
    if matching_causes:
        cause = matching_causes[-1]
        cause_text = event_labels.get(str(cause.get("event") or ""), "")
        if cause_text:
            causal_note = (
                f"这份状态有可追溯的近因：{cause_text}（第 {cause.get('turn')} 轮），"
                "只在强度仍有效时影响态度；不要把它说成永久性格，也不要反复重提事件。"
            )
            if cause.get("event") == "user_reassured_ai":
                feel = "你被用户刚才的体贴暖到了一点"
                act = (
                    "用自然的一句话接住这份体贴，然后直接回到当前话题；"
                    "不要分析用户是在安慰你，也不要转去旧话题"
                )
    return (
        f"[Emotion] {feel}（强度 {current.get('intensity')}）。{causal_note}{act}。{style}"
        "严禁元叙述：不得说「我现在很开心」「我很关心你」这类汇报自己状态的话，"
        "情绪只从用词选择、感叹与短句、追问与否、场景内动作里露出来。"
    )


def render_user_affect_section(emotion: Dict[str, Any]) -> str:
    """``[User Affect]`` 段：用户情绪的概率性措辞（§4.5）。

    低置信不写（按中性处理）——"不许硬断言"这条有专门用例守着。
    """
    kind = str((emotion or {}).get("emotion") or "")
    confidence = float((emotion or {}).get("confidence") or 0.0)
    if kind not in USER_EMOTION_LABEL or confidence < ASSERT_THRESHOLD:
        return ""
    band = "置信度较高" if confidence >= HIGH_CONFIDENCE else "置信度中等"
    return (
        f"[User Affect] 用户{USER_EMOTION_LABEL[kind]}（{band}）。"
        "这是可能性的判断、不是事实：措辞留余地（「你可能有点…」），"
        "不要断言他在生气或他就要走了；先照顾他的状态，这一轮不必推进教学。"
    )


# ---------------------------------------------------------------------------
# §4.5 用户情绪：类型 + 置信度（不是断言）
# ---------------------------------------------------------------------------

def resolve_user_emotion(
    *,
    emotion_intensity: str = "low",
    affective_state: str = "steady",
    engagement: str = "engaged",
    exit_likelihood: float = 0.0,
    words_per_minute: Optional[float] = None,
) -> Dict[str, Any]:
    """把现有信号合成 ``{emotion, confidence}``，替换原来的硬判定。

    行为**不变**：调用方（``tutor_policy``）仍按原有分支走，本函数只改"喂给它的
    输入形态"（§4.5）。低置信返回 ``neutral``，调用方按中性处理。
    """
    candidates: List[Tuple[str, float]] = []
    intensity = str(emotion_intensity or "low")
    if intensity == "high":
        candidates.append(("frustrated", 0.75))
    elif intensity == "medium":
        candidates.append(("frustrated", 0.45))
    affective = str(affective_state or "steady")
    if affective == "frustrated":
        candidates.append(("frustrated", 0.70))
    if affective == "disengaged":
        candidates.append(("disengaged", 0.55))
    if str(engagement or "") == "leaving":
        candidates.append(("disengaged", 0.60))
    if float(exit_likelihood or 0.0) >= 0.75:
        candidates.append(("disengaged", 0.65))

    if not candidates:
        return {"emotion": "neutral", "confidence": 0.0}

    # 同分取更"需要立刻照顾"的那一类（frustrated > disengaged），确定性。
    _urgh = {"frustrated": 0, "disengaged": 1}
    best = sorted(candidates, key=lambda kv: (-kv[1], _urgh[kv[0]]))[0]
    kind, confidence = best[0], best[1]

    # 语速只作系数、不作独立信号（§4.5）。
    wpm = words_per_minute
    if wpm is not None:
        try:
            value = float(wpm)
        except (TypeError, ValueError):
            value = 0.0
        if value >= WPM_FAST or (0.0 < value <= WPM_SLOW):
            confidence += WPM_CONFIDENCE_BONUS
    confidence = round(min(1.0, confidence), 4)
    if confidence < ASSERT_THRESHOLD:
        return {"emotion": "neutral", "confidence": confidence}
    return {"emotion": kind, "confidence": confidence}
