"""Deterministic, per-user relationship state for ordinary Alice chat.

This module consumes explicit interaction evidence already stored in
relationship_memory. It never infers intimacy from turn count, sentiment,
private disclosure, or a single name/friend cue.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


_EVENT_PREFIX = "alice_relation_event:"
_ATTENTION_PREFIX = "alice_attention_event:"
_VALID_STYLES = {"warm", "casual", "reserved", "professional"}
_VALID_DEPTH = {"user_led", "open_when_invited"}


def default_relationship_arc() -> dict[str, str]:
    return {
        "familiarity": "stranger",
        "interaction_style": "warm",
        "depth_permission": "user_led",
        "boundary": "none_recorded",
        "trust": "unexpressed",
        "growth_preference": "uncommitted",
    }


def _stamp(item: dict[str, Any]) -> str:
    return str(item.get("observed_at") or item.get("created_at") or "")


def _eligible(item: Any, *, now: datetime) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("hidden_at") or item.get("superseded_by"):
        return False
    if not str(item.get("key") or "").startswith(_EVENT_PREFIX):
        return False
    try:
        if float(item.get("confidence") or 0) < 0.85:
            return False
    except (TypeError, ValueError):
        return False
    if not item.get("confirmed_by_user"):
        return False
    expires_at = item.get("expires_at")
    if not expires_at:
        return False
    try:
        expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > now
    except (TypeError, ValueError):
        return False


def reduce_relationship_arc(
    relationship_memory: dict[str, Any] | None,
    *,
    pending_events: Iterable[dict[str, Any]] = (),
    has_persona_history: bool = False,
    now: datetime | None = None,
) -> dict[str, str]:
    """Reduce user-verified relationship events into four independent axes."""
    memory = relationship_memory if isinstance(relationship_memory, dict) else {}
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    events = [
        item for item in (memory.get("memory_items") or [])
        if _eligible(item, now=moment)
    ]
    events.extend(item for item in pending_events if _eligible(item, now=moment))
    events.sort(key=_stamp)
    kinds = [str(item.get("event_type") or "") for item in events]
    arc = default_relationship_arc()

    # Preserve older users' established identity and shared history without
    # treating a name or one old memory as proof of close friendship.
    familiar_evidence = {
        "explicit_friendship", "close_friend_claim", "shared_history_reference",
        "successful_recall_ack", "invited_self_authorship", "boundary_repair",
        "reciprocal_care",
    }
    old_relationship_memory = bool(
        memory.get("last_topic") or memory.get("open_loop")
        or any(
            isinstance(item, dict)
            and item.get("key") not in {"learner_name", "tutor_name"}
            and not item.get("hidden_at")
            and not item.get("superseded_by")
            and not str(item.get("key") or "").startswith(
                (_EVENT_PREFIX, _ATTENTION_PREFIX)
            )
            for item in memory.get("memory_items") or []
        )
    )
    if (old_relationship_memory or has_persona_history
            or any(kind in familiar_evidence for kind in kinds)):
        arc["familiarity"] = "recognized"

    explicit_stage_events = {
        "explicit_friendship": "familiar",
        "close_friend_claim": "close_friend",
    }
    latest_stage = next(
        (event for event in reversed(events)
         if str(event.get("event_type") or "") in explicit_stage_events),
        None,
    )
    if latest_stage:
        arc["familiarity"] = explicit_stage_events[str(latest_stage["event_type"])]
    if ("relationship_wants_familiar" in kinds
            and arc["familiarity"] == "stranger"):
        arc["familiarity"] = "recognized"

    # Familiarity may also grow through verified shared-history signals, but
    # requires independent sessions. Trust and personal disclosure alone do
    # not count toward it.
    history_events = [
        event for event in events
        if event.get("event_type") in {"shared_history_reference", "successful_recall_ack"}
    ]
    history_sessions = {
        str(event.get("source_conv_id") or "")
        for event in history_events if str(event.get("source_conv_id") or "")
    }
    if len(history_sessions) >= 2 and arc["familiarity"] in {"stranger", "recognized"}:
        arc["familiarity"] = "familiar"

    # User preferences are durable until the user states a different one.
    style_map = {
        "relationship_keep_casual": ("casual", "keep_casual"),
        "relationship_keep_distance": ("reserved", "keep_distance"),
        "relationship_professional": ("professional", "professional"),
    }
    latest_style = next(
        (event for event in reversed(events)
         if str(event.get("event_type") or "") in style_map),
        None,
    )
    latest_style_stamp = _stamp(latest_style) if latest_style else ""
    if latest_style:
        arc["interaction_style"], arc["boundary"] = style_map[
            str(latest_style["event_type"])
        ]
    latest_boundary = next(
        (event for event in reversed(events)
         if event.get("event_type") == "relationship_no_pet_names"),
        None,
    )
    if latest_boundary:
        arc["boundary"] = ";".join(
            value for value in (
                arc["boundary"] if arc["boundary"] != "none_recorded" else "",
                "no_pet_names",
            ) if value
        )
    latest_depth = next(
        (event for event in reversed(events)
         if event.get("event_type") == "relationship_deeper_opt_in"),
        None,
    )
    if latest_depth and _stamp(latest_depth) >= latest_style_stamp:
        arc["depth_permission"] = "open_when_invited"

    if "explicit_trust" in kinds:
        arc["trust"] = "user_expressed"
    if any(kind in kinds for kind in ("shared_history_reference", "successful_recall_ack")):
        arc["growth_preference"] = "shared_history"
    if "relationship_wants_familiar" in kinds:
        arc["growth_preference"] = "gradually_get_to_know_each_other"

    # Defensive bounds keep corrupted/legacy memory from becoming prompt text.
    if arc["interaction_style"] not in _VALID_STYLES:
        arc["interaction_style"] = "warm"
    if arc["depth_permission"] not in _VALID_DEPTH:
        arc["depth_permission"] = "user_led"
    return arc


def events_for_current_turn(
    user_text: str,
    *,
    user_id: str,
    conversation_id: str,
    turn: int,
    observed_at: str | None = None,
    evidence_source: str = "user_text",
) -> list[dict[str, Any]]:
    """Use the same deterministic extractor for immediate and persisted state."""
    from memory.companion_interaction import extract_interaction_events

    stamp = observed_at or datetime.now(timezone.utc).isoformat()
    return extract_interaction_events(
        user_text,
        user_id=user_id,
        conv_id=conversation_id,
        turn=turn,
        observed_at=stamp,
        evidence_source=evidence_source,
    )


def familiarity_from_memory_depth(depth: int) -> str:
    """按"她记得你多少"给一个亲密度下限（plans/1017-02 §8.37）。

    实测：主用户记忆深度 237（实体 134＋有料归档 103），其余用户 4~20，
    而说话方式**几乎没有差别**——因为标签永远停在 stranger。
    阈值取自真实分布：≥50 视为熟悉、≥10 视为认得、否则陌生。
    """
    try:
        value = int(depth or 0)
    except (TypeError, ValueError):
        return "stranger"
    if value >= 50:
        return "familiar"
    if value >= 10:
        return "recognized"
    return "stranger"


def closeness_score(events: list[dict], now: datetime | None = None,
                    half_life_days: float = 30.0) -> float:
    """按**内容**算亲疏，而不是按聊了多少（plans/1017-02 §8.38）。

    用户的原话：「关系不是单纯聊得多久就能定下来，同事天天说也不见得关系有多好，
    以及关系是会变化的」。所以：
      · **次数不贡献任何分数**——中性互动记 0，聊再多也还是 0；
      · **只有内容事件计分**：用户分享私事（+）、明确表达关心（+）、明确要求保持距离（−）；
      · **按天衰减**（半衰期默认 30 天）⇒ 久不联系会淡，翻脸会掉档，不是只升不降的阶梯。

    事件形状：{"kind": "closeness_delta", "value": +1.0/−1.0, "observed_at": ISO8601}
    """
    moment = now or datetime.now(timezone.utc)
    total = 0.0
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("kind") or event.get("event_type") or "") != "closeness_delta":
            continue
        try:
            value = float(event.get("value") or 0.0)
        except (TypeError, ValueError):
            continue
        stamp = str(event.get("observed_at") or event.get("created_at") or "")
        age_days = 0.0
        if stamp:
            try:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (moment - when).total_seconds() / 86400.0)
            except (TypeError, ValueError):
                age_days = 0.0
        total += value * (0.5 ** (age_days / max(1.0, half_life_days)))
    return round(total, 3)


def familiarity_from_closeness(score: float) -> str:
    """由内容分给档位：靠内容积累才升，掉分就降（不是下限，是可逆档位）。"""
    try:
        value = float(score or 0.0)
    except (TypeError, ValueError):
        return "stranger"
    if value >= 2.5:
        return "familiar"
    if value >= 0.8:
        return "recognized"
    return "stranger"


bounds_contract_close_friend = (
    "关系熟了可以更放得开，但**话量、玩笑的密度和方式都按你自己平时怎么说话来**，"
        "不因熟而变话痨、不硬凹统一的玩笑模板；关心对方的事情和情绪可以更主动，"
        "表达的深浅仍贴合你的格调。"
)


def compile_dialogue_contract(arc: dict[str, Any] | None) -> str:
    """Render a bounded relationship behavior contract for companion chat."""
    state = default_relationship_arc()
    if isinstance(arc, dict):
        for key in state:
            value = str(arc.get(key) or state[key])
            if len(value) <= 64:
                state[key] = value

    familiarity = {
        # ⛔ 写成**状态描述**而不是清单（plans/1017-02 §8.40）：
        # 旧版 familiar 是"至少体现一项：个性化判断／承接共同背景／主动提出协助"——
        # 那是配额，模型会照着做一遍，正是"固定模板"。现在改成她怎么说话，
        # 并显式给出**剂量**（偶尔、不是每轮）与**不要演**。
        "stranger": (
            "你们刚认识：友好但不套近乎。先接住对方说的；不主动讲自己的事，"
            "也不编你们之间发生过什么。"
        ),
        "recognized": (
            "你记得这个人，也记得你们聊过什么。可以自然用名字、接上之前的话题；"
            "不必表明你们有多熟——熟不熟由对方推进。"
        ),
        "familiar": (
            "这个人你熟了：有些事只有你们之间知道。说话像跟熟人那样——少一点客套，"
            "判断更直接；也可以顺手带一句自己的事，一句就够。"
            "边界是这两条：**关于对方**、以及**你们之间发生过什么**，只能来自你真记得的，"
            "没记住就别说；**你自己的习惯和日子**可以有，但要稳、要跟平常的你一致，"
            "不为了显得熟而现编一段来撑场面。"
        ),
        "close_friend": (
            "用户明确把你们当亲密朋友：可以更坦率、更松弛、更主动；引用共同记忆、"
            "说自己的事都很自然——同样：关于对方和你们之间的事只用你真记得的，"
            "自己的日子要稳；仍尊重这一轮的边界，不推断恋爱或独占。"
        ),
    }.get(state["familiarity"], "关系证据有限，友好并保持分寸。")
    if state["familiarity"] == "close_friend":
        familiarity += " " + bounds_contract_close_friend
    style = {
        "warm": "保持温和自然的日常交流。",
        "casual": "按用户偏好保持轻松、普通的相处，不主动推深关系。",
        "reserved": "用户要求保持距离；减少亲昵称呼、主动追问、自我披露和私人话题推进。",
        "professional": "用户偏好专业克制的交流；避免亲密表达和无关私人追问。",
    }.get(state["interaction_style"], "保持温和自然的日常交流。")
    depth = (
        "用户明确允许更深入的话题；仍逐次看当下回应，拒答或换题即停止追问。"
        if state["depth_permission"] == "open_when_invited"
        else "私人话题由用户带领；不把信任表达、个人披露或关系称谓当作追问许可。"
    )
    boundary_map = {
        "keep_casual": "保持普通轻松",
        "keep_distance": "保持距离",
        "professional": "保持专业",
        "no_pet_names": "不使用亲昵称谓",
    }
    boundary_values = [
        boundary_map[value] for value in state["boundary"].split(";")
        if value in boundary_map
    ]
    boundary = (
        "关系边界：" + "；".join(boundary_values) + "，本轮即生效。"
        if boundary_values else "关系边界：用户当轮明确要求优先于旧关系证据。"
    )
    trust = (
        "用户曾明确表达信任；把它视为需要珍惜的信号，不扩大为亲密或隐私授权。"
        if state["trust"] == "user_expressed" else ""
    )
    return (
        "[Relationship arc contract] "
        + " ".join(part for part in (familiarity, style, depth, boundary, trust) if part)
        + " 关系状态只影响称呼、语气、主动性、自我披露和共同记忆的使用；"
        "不得改写事实、用户当前任务、学习目标或场景角色。"
    )
