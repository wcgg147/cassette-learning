"""Deterministic conflict handling for evidence-bearing relationship memory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable


SINGULAR_RELATIONSHIP_KEYS = {
    "last_topic", "open_loop", "recent_mood", "preferred_tone",
    "learning_preference", "self_reported_weakness", "next_session_intent",
    "next_session_intent_consumed",
    # 身份记忆（plans/0915-04）：名字是单人槽位——用户改名时新名字覆盖旧名字，
    # 且绝不堆积多条同键记忆。
    "learner_name", "tutor_name",
    # AI 朋友人格自身已形成的背景；作为单一权威槽位，冲突的新自述不能静默覆盖。
    "tutor_persona_origin", "tutor_persona_residence",
}

# 角色扮演别名（plans/0915-16 方案B）：刻意不在 SINGULAR_KEYS 里——
# 戏剧别名可以多条共存（dedup 由非 singular 分支的 (key,value) 去重负责）。
ROLEPLAY_ALIAS_KEYS = {"learner_roleplay_alias", "tutor_roleplay_alias"}
TUTOR_PERSONA_STABLE_KEYS = {"tutor_persona_origin", "tutor_persona_residence"}
TUTOR_PERSONA_VERSIONED_PREFIXES = (
    "tutor_persona_ability:", "tutor_persona_preference:", "tutor_persona_trait:",
    "tutor_persona_language_environment:",
)
TUTOR_PERSONA_GRADUAL_PREFIXES = (
    "tutor_persona_preference:", "tutor_persona_trait:",
)
ALICE_INTERACTION_PREFIXES = ("alice_relation_event:", "alice_attention_event:")


def _is_tutor_persona_key(key: str) -> bool:
    return str(key or "").startswith("tutor_persona_")


def _is_tutor_persona_versioned_key(key: str) -> bool:
    return str(key or "").startswith(TUTOR_PERSONA_VERSIONED_PREFIXES)


def partition_name_updates(
    existing_names: dict, names: dict, roleplay_names: dict, *,
    allow_rename: bool = False,
) -> dict:
    """场景闸门（plans/0915-16 方案B）：真名层与角色扮演别名分流。

    用户的想法：AI 可以大量角色扮演、戏里用别的身份，但名字层面（真名层
    learner_name/tutor_name）必须保持真实确立的称呼，不被戏里的舞台名顶掉。

    - ``names``：meta 层面确立的称呼（用户自报真名 / 给 AI 起名 / 明确改名）。
      ``allow_rename=True``（归档 LLM 路径，看得到全场上下文，能判定"这是
      真改名"）时允许覆盖真名槽位；``allow_rename=False``（盘内正则路径，
      只有单句、无法判定语境）时，与既有真名冲突的候选一律隔离进别名槽位。
    - ``roleplay_names``：整场是虚构角色扮演时确立的舞台名（如"你演
      Sherlock 我演 Watson"），只进别名槽位，永不触碰真名层。

    返回 {key: value} 待写项；空 dict 表示什么都不用写。
    """
    existing = {k: str(v or "").strip() for k, v in dict(existing_names or {}).items()}
    out: dict = {}
    for slot in ("learner", "tutor"):
        name_key, alias_key = f"{slot}_name", f"{slot}_roleplay_alias"
        current = existing.get(name_key, "")
        new_name = str((names or {}).get(slot) or "").strip()
        rp_name = str((roleplay_names or {}).get(slot) or "").strip()
        if new_name:
            if allow_rename or not current or current == new_name:
                out[name_key] = new_name
            else:
                # 无语境路径的保守隔离：疑似戏中名，不顶真名
                out.setdefault(alias_key, new_name)
        if rp_name and rp_name != current:
            out.setdefault(alias_key, rp_name)
    return out


def _created_at(item: dict) -> datetime:
    raw = str(item.get("created_at") or "").strip()
    if raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def merge_relationship_items(
    existing: Iterable[dict], incoming: Iterable[dict], *,
    max_items: int = 30, max_history: int = 24,
) -> list[dict]:
    """Merge new evidence without letting a late, older task overwrite it.

    Singular slots use newest-evidence-wins semantics for the live view.
    0915-12（Mem0 v3 思路，《AI-Agents-in-Depth》§3.3.4）：被顶替的旧值不再
    物理删除，而是打上 ``superseded_by`` 标记留底——一次错误更新（如泛称
    "用户/AI" 覆盖真名 Jason/Alice）不再不可逆；读侧过滤 superseded 得到
    当前有效值，导出/审计仍能看到全部历史。``max_history`` 限制留底规模。
    """
    now = datetime.now(timezone.utc)

    merged = [dict(item) for item in existing if isinstance(item, dict)]
    for raw in incoming:
        item = dict(raw) if isinstance(raw, dict) else {}
        key, value = str(item.get("key") or ""), str(item.get("value") or "")
        if not key or not value:
            continue
        if key in TUTOR_PERSONA_STABLE_KEYS:
            established = [
                old for old in merged
                if old.get("key") == key and not old.get("superseded_by")
            ]
            if established:
                current_origin = max(established, key=_created_at)
                # First established origin is stable across concurrent sessions too.
                # A later model self-disclosure cannot silently rewrite Alice's past.
                if str(current_origin.get("value") or "").strip() != value.strip():
                    continue
        if key in SINGULAR_RELATIONSHIP_KEYS or _is_tutor_persona_versioned_key(key):
            previous = [old for old in merged if old.get("key") == key]
            if key.startswith(TUTOR_PERSONA_GRADUAL_PREFIXES):
                active_versions = [
                    old for old in previous
                    if not old.get("superseded_by")
                    and old.get("change_status") != "candidate"
                ]
                prior_same_value = [
                    old for old in previous
                    if not old.get("superseded_by") and old.get("value") == value
                ]
                try:
                    support_count = max(
                        int(item.get("support_count") or 0),
                        max((int(old.get("support_count") or 0) for old in prior_same_value), default=0),
                    )
                except (TypeError, ValueError):
                    support_count = int(item.get("support_count") or 0)
                conflicting_active = [
                    old for old in active_versions
                    if str(old.get("value") or "") != value
                ]
                if conflicting_active and not item.get("confirmed_by_user") and support_count < 2:
                    # Keep one disclosure as a dated candidate; a long-term trait
                    # or preference changes only after corroboration or explicit
                    # user confirmation. Do not let a single generated sentence
                    # erase the established Alice narrative.
                    item["change_status"] = "candidate"
                    item["candidate_for"] = [
                        str(old.get("id") or "") for old in conflicting_active
                        if old.get("id")
                    ][-8:]
                    merged = [
                        old for old in merged
                        if not (old.get("key") == key and old.get("value") == value
                                and old.get("change_status") == "candidate"
                                and not old.get("superseded_by"))
                    ]
                    merged.append(item)
                    continue
                if conflicting_active:
                    stamp = str(item.get("created_at") or "") or now.isoformat()
                    superseded_ids = []
                    for old in conflicting_active:
                        old["superseded_by"] = str(item.get("id") or "")
                        old["superseded_at"] = stamp
                        old["valid_to"] = stamp
                        old.setdefault("time_basis", "recorded_at")
                        if old.get("id"):
                            superseded_ids.append(str(old["id"]))
                    item["change_status"] = "confirmed"
                    item.setdefault("valid_from", stamp)
                    item.setdefault("time_basis", "recorded_at")
                    if superseded_ids:
                        item["supersedes"] = list(dict.fromkeys(
                            list(item.get("supersedes") or []) + superseded_ids
                        ))[-8:]
            newest_previous = max(previous, key=_created_at, default=None)
            if newest_previous and _created_at(item) < _created_at(newest_previous):
                continue
            if newest_previous and str(newest_previous.get("value") or "") == value:
                reinforced = dict(newest_previous)
                reinforced.update(item)
                reinforced["confidence"] = max(
                    float(newest_previous.get("confidence") or 0.0),
                    float(item.get("confidence") or 0.0),
                )
                item = reinforced
                merged = [old for old in merged if old is not newest_previous]
            else:
                # append-only：旧值打标留底，不删除
                stamp = str(item.get("created_at") or "") or now.isoformat()
                superseded_ids = []
                for old in previous:
                    if str(old.get("value") or "") == value:
                        continue
                    old["superseded_by"] = str(item.get("id") or "")
                    old["superseded_at"] = stamp
                    if _is_tutor_persona_versioned_key(key):
                        old["valid_to"] = stamp
                        old.setdefault("time_basis", "recorded_at")
                        item.setdefault("valid_from", stamp)
                        item.setdefault("time_basis", "recorded_at")
                    if old.get("id"):
                        superseded_ids.append(str(old["id"]))
                if superseded_ids:
                    item["supersedes"] = list(dict.fromkeys(
                        list(item.get("supersedes") or []) + superseded_ids
                    ))[-8:]
            merged.append(item)
        else:
            merged = [
                old for old in merged
                if (old.get("key"), old.get("value")) != (key, value)
            ]
            merged.append(item)
    live = [i for i in merged if not i.get("superseded_by")]
    history = [i for i in merged if i.get("superseded_by")]
    # Established Alice location identity is user-specific and permanent. Keep
    # those live slots outside the rolling cap so ordinary learner memories
    # cannot evict them after enough conversations.
    persona_identity = [
        item for item in live
        if item.get("key") in TUTOR_PERSONA_STABLE_KEYS
    ]
    alice_interaction_live = [
        item for item in live
        if str(item.get("key") or "").startswith(ALICE_INTERACTION_PREFIXES)
    ]
    persona_current = [
        item for item in live
        if _is_tutor_persona_key(str(item.get("key") or ""))
        and item.get("key") not in TUTOR_PERSONA_STABLE_KEYS
    ]
    rolling_live = [
        item for item in live
        if (not _is_tutor_persona_key(str(item.get("key") or ""))
            and not str(item.get("key") or "").startswith(ALICE_INTERACTION_PREFIXES))
    ]
    persona_history = [
        item for item in history
        if _is_tutor_persona_key(str(item.get("key") or ""))
    ]
    alice_interaction_history = [
        item for item in history
        if str(item.get("key") or "").startswith(ALICE_INTERACTION_PREFIXES)
    ]
    other_history = [
        item for item in history
        if not _is_tutor_persona_key(str(item.get("key") or ""))
    ]
    return (
        rolling_live[-max(1, int(max_items)):]
        + persona_identity
        + persona_current[-48:]
        + alice_interaction_live[-48:]
        + other_history[-max(0, int(max_history)):]
        + persona_history[-48:]
        + alice_interaction_history[-48:]
    )
