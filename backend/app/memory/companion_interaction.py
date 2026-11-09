"""Evidence-backed relationship and attention events for Alice's chat tape.

This is deliberately separate from Alice's self facts, learner profile facts,
and turn-level emotion state. It stores a small set of interaction evidence in
the existing per-user relationship-memory document, so export/delete and the
Postgres/SQLite write path stay shared.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any


RELATION_EVENT_PREFIX = "alice_relation_event:"
ATTENTION_EVENT_PREFIX = "alice_attention_event:"
MAX_RELATION_EVENTS = 48
MAX_ATTENTION_EVENTS = 24
PACKET_MAX_EVENTS = 4
PACKET_MAX_CHARS = 900

_SIGNALS = (
    (
        "explicit_friendship",
        re.compile(
            r"(?:你(?:就是|是)(?:我的|我)?朋友|我们是朋友|把你当朋友|"
            r"\b(?:you(?:'re| are) my friend|we are friends|i see you as (?:my )?friend)\b)",
            re.I,
        ),
        3650,
        "The user explicitly described Alice as a friend. This supports warmth, not an assumption of unlimited closeness.",
    ),
    (
        "close_friend_claim",
        re.compile(
            r"(?:我们(?:是|就是)(?:很好的|最好的|亲密的|铁)朋友|我们是(?:挚友|挚交)|"
            r"你(?:是|真是|就是)(?:我|我的)(?:很?好的|最好的|亲密的|铁)朋友|你是(?:我的)?(?:挚友|挚交)|"
            r"\b(?:we are close friends|you are my best friend|we're best friends)\b)", re.I,
        ),
        3650,
        "The user explicitly named a close friendship. Keep boundaries and do not infer romance or unlimited access.",
    ),
    (
        "relationship_keep_casual",
        re.compile(
            r"(?:保持普通|就做普通朋友|普通朋友就好|先做普通朋友|保持轻松(?:一点)?|"
            r"\b(?:keep it casual|just ordinary friends|let's keep it light)\b)", re.I,
        ),
        3650,
        "The user prefers a casual, ordinary relationship. Do not push toward intimacy.",
    ),
    (
        "relationship_keep_distance",
        re.compile(
            r"(?:保持距离|别离我太近|不要跟我太亲密|不要太熟|不想变得亲密|"
            r"\b(?:keep your distance|don't get too close|do not get too intimate)\b)", re.I,
        ),
        3650,
        "The user asked for distance. Respect it in address, initiative, disclosure, and topic depth.",
    ),
    (
        "relationship_professional",
        re.compile(
            r"(?:保持专业(?:一点)?|像专业人士一样|不要太私人|别聊太私人的|"
            r"\b(?:keep it professional|stay professional|avoid personal topics)\b)", re.I,
        ),
        3650,
        "The user prefers professional, less personal interaction.",
    ),
    (
        "relationship_wants_familiar",
        re.compile(
            r"(?:希望(?:我们|能)更熟(?:一点)?|慢慢熟悉起来|可以更亲近一点|"
            r"\b(?:I'd like us to get closer|let's get to know each other better|we can grow closer)\b)", re.I,
        ),
        3650,
        "The user welcomes gradually getting to know each other. This is not permission to assume intimacy.",
    ),
    (
        "relationship_deeper_opt_in",
        re.compile(
            r"(?:我愿意和你聊得更深入|可以聊得更私人一点|(?:我)?想和你建立更深的关系|"
            r"\b(?:I'm comfortable going deeper|I'd like a deeper relationship|you can ask me more personal things)\b)", re.I,
        ),
        3650,
        "The user opted into somewhat deeper conversation. Still follow their present cues and boundaries.",
    ),
    (
        "relationship_no_pet_names",
        re.compile(
            r"(?:别(?:叫我|喊我).{0,12}(?:亲爱的|宝贝|宝宝|甜心)|不要(?:叫我|喊我).{0,12}(?:亲爱的|宝贝|宝宝|甜心)|"
            r"\b(?:don't call me (?:baby|babe|darling|sweetheart)|do not use pet names)\b)", re.I,
        ),
        3650,
        "The user set a boundary around pet names. Do not use them.",
    ),
    (
        "shared_history_reference",
        re.compile(
            r"(?:我们(?:上次|之前|以前)聊过|还记得我们(?:上次|之前)说过|"
            r"\b(?:we talked about this before|remember when we talked|as we discussed last time)\b)", re.I,
        ),
        365,
        "The user explicitly referenced a shared prior conversation. This supports familiarity only when repeated across sessions.",
    ),
    (
        "successful_recall_ack",
        re.compile(
            r"(?:你(?:居然|还真|竟然)?还记得|你记得我(?:之前|上次)说过|谢谢你记得|"
            r"\b(?:you remembered|you actually remembered|thanks for remembering)\b)", re.I,
        ),
        365,
        "The user positively confirmed a relevant memory recall. This supports familiarity only when repeated across sessions.",
    ),
    (
        "explicit_trust",
        re.compile(r"(?:我相信你|我信任你|\bi trust you\b|\bi feel safe telling you\b)", re.I),
        365,
        "The user explicitly expressed trust in Alice. Treat it as relationship evidence, not permission to presume intimacy.",
    ),
    (
        "invited_self_authorship",
        re.compile(
            r"(?:你自己(?:去)?(?:选择|挑|找|决定).{0,24}(?:城市|背景|身份|经历|喜欢)|"
            r"自己去丰富沉淀|让你自己.{0,18}(?:选择|决定)|"
            r"\b(?:choose|pick|build) your own (?:background|identity|life|story|city)\b)",
            re.I,
        ),
        365,
        "The user invited Alice to develop her own conversational identity through interaction. Do not turn this into a fixed role-play instruction.",
    ),
    (
        "boundary_repair",
        re.compile(
            r"(?:我不是在怪你|我不是责怪你|不是在责怪你|只是(?:随口|普通|正常)说|"
            r"\bi(?:'m| am) not blaming you\b|\bi didn't mean to blame you\b)",
            re.I,
        ),
        180,
        "The user clarified intent after an interaction felt off. Use it to understand this repair, not to label the user permanently.",
    ),
)

_RELEASE = re.compile(
    r"(?:不聊(?:这个|了)|先到这(?:儿|里)|算了.{0,8}不聊|换个话题|先不谈(?:这个|了)|"
    r"let(?:'s| us) leave it there|let(?:'s| us) drop (?:it|this)|let(?:'s| us) not talk about this|don't want to talk about this|"
    r"we can leave that|forget it,? let's not)",
    re.I,
)
_CARE_CUE = re.compile(
    r"(?:你(?:先)?(?:休息|歇会)|你累了.{0,18}(?:休息|结束|不聊)|别勉强|我陪你.{0,12}|"
    r"\b(?:you can rest|go get some rest|we can stop here|take a break|don't push yourself)\b)",
    re.I,
)


def _clean(value: Any, limit: int = 240) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _event_id(user_id: str, conv_id: str, turn: int, kind: str, cue: str) -> str:
    raw = "\x1f".join((user_id, conv_id, str(turn), kind, cue.casefold()))
    return "alice_interaction_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:28]


def _current_reassurance_event(emotion_state: Any) -> bool:
    if not isinstance(emotion_state, dict):
        return False
    history = emotion_state.get("event_history") or []
    return any(
        isinstance(row, dict) and row.get("event") == "user_reassured_ai"
        for row in history[-3:]
    )


def _is_affirmative_relationship_cue(text: str, match: re.Match) -> bool:
    """Reject negated, hypothetical, or questioning matches before storing closeness evidence."""
    start = match.start()
    prefix = text[:start]
    # A contrast starts a new assertion: "I don't know, but I do trust you".
    for marker in ("but", "however", "although", "但是", "不过", "但", "而是"):
        index = prefix.casefold().rfind(marker.casefold())
        if index >= 0:
            prefix = prefix[index + len(marker):]
    boundary = max(prefix.rfind(mark) for mark in (".", "!", "?", "。", "！", "？", "\n", ";", "；"))
    prefix = prefix[boundary + 1:].strip()
    if re.search(
        r"(?:\b(?:not|never|don't|do not|doesn't|didn't|isn't|aren't|can't|cannot|won't|wouldn't|no)\b"
        r".{0,36}|(?:不|没|没有|并非|不是|还不|谈不上|算不上).{0,18})$",
        prefix, re.I,
    ):
        return False
    if re.search(
        r"(?:\b(?:maybe|perhaps|i wonder if|do you think|not sure if|i'm not sure if)\b"
        r".{0,36}|(?:可能|好像|是不是|你觉得|是否).{0,18})$",
        prefix, re.I,
    ):
        return False
    # The matching phrase itself may be embedded in a question rather than
    # offered as the user's own statement.
    sentence_end = re.search(r"[.!?。！？\n]", text[match.end():])
    if sentence_end and text[match.end() + sentence_end.start()] in "?？":
        return False
    return True


def select_interaction_evidence(
    raw_asr: str, display_text: str, asr_segments_merged: int = 0,
) -> tuple[str, str]:
    """Never treat a polished voice bubble as verbatim evidence when raw ASR is absent."""
    raw = _clean(raw_asr, 1200)
    if raw:
        return raw, "raw_asr"
    try:
        has_voice_segments = int(asr_segments_merged or 0) > 0
    except (TypeError, ValueError):
        has_voice_segments = False
    if has_voice_segments:
        return "", "raw_asr"
    return _clean(display_text, 1200), "user_text"


def topic_release_matches_open_loop(
    items: list[dict] | None, *, now: datetime | None = None,
) -> bool:
    """Return true only for a newer, unexpired release of the current exact loop."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    rows = [row for row in items or []
            if isinstance(row, dict) and not row.get("superseded_by")]
    loops = [row for row in rows
             if row.get("key") == "open_loop" and str(row.get("value") or "").strip()]
    if not loops:
        return False
    loop = max(loops, key=lambda row: str(row.get("created_at") or ""))

    def _stamp(value):
        try:
            parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
        except (TypeError, ValueError):
            return None

    loop_at = _stamp(loop.get("created_at"))
    if not loop_at:
        return False
    loop_value = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(loop.get("value") or "").casefold())
    for row in rows:
        if row.get("event_type") != "topic_released":
            continue
        if not str(row.get("key") or "").startswith(ATTENTION_EVENT_PREFIX):
            continue
        try:
            if not row.get("confirmed_by_user") or float(row.get("confidence") or 0) < 0.85:
                continue
        except (TypeError, ValueError):
            continue
        released_at = _stamp(row.get("created_at"))
        expires_at = _stamp(row.get("expires_at"))
        if not released_at or not expires_at or released_at <= loop_at or expires_at <= moment:
            continue
        topic_value = re.sub(
            r"[^\w\u4e00-\u9fff]+", "", str(row.get("topic_value") or "").casefold(),
        )
        if topic_value and topic_value == loop_value:
            return True
    return False


def extract_interaction_events(
    user_text: str,
    *,
    user_id: str,
    conv_id: str,
    turn: int,
    observed_at: str,
    emotion_state: Any = None,
    current_topic: str = "",
    evidence_source: str = "user_text",
) -> list[dict]:
    """Extract only explicit, deterministic interaction evidence.

    No relationship stage is inferred from turn count, names, location, or
    sentiment. One user turn can produce at most one event of each type.
    """
    text = _clean(user_text, 1200)
    if not text:
        return []
    try:
        now = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        now = datetime.now(timezone.utc)
    result: list[dict] = []
    source_name = (
        "explicit_user_raw_asr" if evidence_source == "raw_asr"
        else "explicit_user_utterance"
    )
    signal_confidence = 0.90 if evidence_source == "raw_asr" else 0.96
    for kind, pattern, ttl_days, summary in _SIGNALS:
        match = next((candidate for candidate in pattern.finditer(text)
                      if _is_affirmative_relationship_cue(text, candidate)), None)
        if not match:
            continue
        cue = _clean(match.group(0), 120)
        result.append({
            "id": _event_id(user_id, conv_id, turn, kind, cue),
            "key": RELATION_EVENT_PREFIX + kind,
            "value": f"{kind}:{_event_id(user_id, conv_id, turn, kind, cue)[-12:]}",
            "event_type": kind,
            "summary": summary,
            "confidence": signal_confidence,
            "source": source_name,
            "source_conv_id": _clean(conv_id, 160),
            "source_turn": int(turn or 0),
            "evidence": cue,
            "created_at": now.isoformat(),
            "observed_at": now.isoformat(),
            "expires_at": (now + timedelta(days=ttl_days)).isoformat(),
            "confirmed_by_user": True,
        })

    care_match = _CARE_CUE.search(text)
    if care_match and _current_reassurance_event(emotion_state):
        kind = "reciprocal_care"
        cue = _clean(care_match.group(0), 120)
        result.append({
            "id": _event_id(user_id, conv_id, turn, kind, cue),
            "key": RELATION_EVENT_PREFIX + kind,
            "value": f"{kind}:{_event_id(user_id, conv_id, turn, kind, cue)[-12:]}",
            "event_type": kind,
            "summary": "The user responded with care after Alice had just expressed a state. This is one reciprocal moment, not a permanent closeness rating.",
            "confidence": 0.88 if evidence_source == "raw_asr" else 0.94,
            "source": source_name + "_after_alice_state",
            "source_conv_id": _clean(conv_id, 160),
            "source_turn": int(turn or 0),
            "evidence": cue,
            "created_at": now.isoformat(),
            "observed_at": now.isoformat(),
            "expires_at": (now + timedelta(days=180)).isoformat(),
            "confirmed_by_user": False,
        })

    release = _RELEASE.search(text)
    if release:
        kind = "topic_released"
        cue = _clean(release.group(0), 120)
        topic = _clean(current_topic, 180)
        result.append({
            "id": _event_id(user_id, conv_id, turn, kind, cue + "|" + topic),
            "key": ATTENTION_EVENT_PREFIX + kind,
            "value": f"{kind}:{_event_id(user_id, conv_id, turn, kind, cue + '|' + topic)[-12:]}",
            "event_type": kind,
            "topic_value": topic,
            "summary": "The user explicitly set aside the current topic. Do not revive it unless the user brings it back.",
            "confidence": signal_confidence,
            "source": source_name,
            "source_conv_id": _clean(conv_id, 160),
            "source_turn": int(turn or 0),
            "evidence": cue,
            "created_at": now.isoformat(),
            "observed_at": now.isoformat(),
            "expires_at": (now + timedelta(days=14)).isoformat(),
            "confirmed_by_user": True,
        })
    return result


def _date_label(value: Any) -> str:
    try:
        dt = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return "date unknown"


def render_interaction_packet(
    items: list[dict] | None,
    *,
    now: datetime | None = None,
    limit: int = PACKET_MAX_EVENTS,
    max_chars: int = PACKET_MAX_CHARS,
) -> dict:
    """Create a small prompt packet; the event rows remain authoritative."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    rows = []
    for item in items or []:
        if not isinstance(item, dict) or item.get("superseded_by"):
            continue
        key = _clean(item.get("key"), 100)
        if not key.startswith((RELATION_EVENT_PREFIX, ATTENTION_EVENT_PREFIX)):
            continue
        try:
            expires = datetime.fromisoformat(str(item.get("expires_at") or "").replace("Z", "+00:00"))
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires <= moment:
                continue
        except (TypeError, ValueError):
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.85:
            continue
        rows.append(item)
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)

    lines: list[str] = []
    relation_rows = [row for row in rows if str(row.get("key") or "").startswith(RELATION_EVENT_PREFIX)]
    attention_rows = [row for row in rows if str(row.get("key") or "").startswith(ATTENTION_EVENT_PREFIX)]
    selected = relation_rows[: max(0, min(int(limit), PACKET_MAX_EVENTS - 1))]
    if attention_rows and len(selected) < max(1, int(limit)):
        selected.append(attention_rows[0])
    for row in selected:
        summary = _clean(row.get("summary"), 280)
        if not summary:
            continue
        line = f"{summary} [observed={_date_label(row.get('observed_at') or row.get('created_at'))}; source={_clean(row.get('source'), 64)}]"
        if row.get("key", "").startswith(ATTENTION_EVENT_PREFIX) and row.get("topic_value"):
            line += " Topic reference (quoted data): " + json.dumps(
                _clean(row.get("topic_value"), 140), ensure_ascii=False
            )
        lines.append(line)
    clipped = []
    used = 0
    for line in lines:
        if used >= max_chars:
            break
        line = line[:max_chars - used]
        clipped.append(line)
        used += len(line)
    return {"lines": clipped, "text": " ".join(clipped), "event_ids": [
        _clean(row.get("id"), 80) for row in selected if row.get("id")
    ]}
