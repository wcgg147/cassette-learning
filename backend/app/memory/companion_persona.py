"""Evidence-backed, per-user memory for Alice's conversational persona.

This module deliberately keeps Alice's dialogue-persona facts separate from
the learner profile, user relationship facts, and turn-level emotion state.
Snapshots are derived views only; facts and their source turns remain the
authority.
"""
from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import Any


PERSONA_KEY = "alice"
LEGACY_BACKFILL_VERSION = "alice-persona-v2"
# v3 excludes views of the user from self-growth evidence. Retain old stored
# summaries, but rebuild them through the existing cold pass before reuse.
PERSONA_SNAPSHOT_VERSION = "alice-reflection-v3"
MAX_PACKET_FACTS = 6
MAX_PACKET_CHARS = 1600
_GROWTH_KINDS = {"trait", "preference", "ability", "experience", "language_environment"}
_PERSONA_SELF_FACT_KINDS = _GROWTH_KINDS | {"current_status"}
_CURRENT_STATUS_EDUCATION_RE = re.compile(
    r"^(?:i(?:['’]m| am)\s+(?:(?:still|currently|already|not|no longer)\s+)?"
    r"(?:in school|a student|studying\b|at (?:school|college|university)\b)|"
    r"i\s+(?:currently\s+)?(?:graduated\b|left school\b)|"
    r"我(?:现在)?(?:还在|仍在|在读|正在)?(?:上学|读书|读大学|读高中|读研)|"
    r"我(?:现在)?(?:已经)?(?:毕业|不在上学|不再上学))",
    re.IGNORECASE,
)
_CURRENT_STATUS_EMPLOYMENT_RE = re.compile(
    r"^(?:i(?:['’]m| am)\s+(?:(?:still|currently|not)\s+)?"
    r"(?:working|employed|unemployed)\b|"
    r"i\s+(?:currently\s+)?(?:work\b|have a job\b|do have work\b)|"
    r"我(?:现在)?(?:已经)?(?:参加工作|在工作|有工作|没工作))",
    re.IGNORECASE,
)

_LEGACY_PREFIXES = (
    "tutor_persona_origin",
    "tutor_persona_residence",
    "tutor_persona_experience",
    "tutor_persona_language_environment",
    "tutor_persona_ability",
    "tutor_persona_preference",
    "tutor_persona_trait",
)
_STOPWORDS = {
    "about", "and", "are", "did", "do", "does", "for", "from", "have",
    "he", "her", "how", "i", "is", "it", "me", "my", "of", "the",
    "this", "to", "was", "we", "what", "when", "where", "who", "with",
    "you", "your", "yourself", "alice", "tell", "remember", "talked", "know", "聊过", "记得",
    "之前", "我们", "关于", "你", "什么", "还有", "说过",
}


def _stable_id(*parts: Any) -> str:
    raw = "\x1f".join(str(part or "") for part in parts)
    return "alice_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _clean(value: Any, limit: int = 1000) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def current_status_facet(fact: dict) -> str:
    """Identify explicit present-life-stage records, including older experience rows.

    Current status is useful continuity after one clear self-disclosure. It is
    not evidence of a stable personality trait, so it stays outside growth
    promotion and cold-pass snapshots.
    """
    if not isinstance(fact, dict) or fact.get("hidden_at"):
        return ""
    kind = _clean(fact.get("kind"), 80)
    value = _clean(fact.get("value_text") or fact.get("value"), 1000)
    if kind not in {"current_status", "experience"}:
        return ""
    if _CURRENT_STATUS_EDUCATION_RE.search(value):
        return "education_status"
    if _CURRENT_STATUS_EMPLOYMENT_RE.search(value):
        return "employment_status"
    if kind == "current_status":
        facet = _clean(fact.get("facet_key"), 180).casefold()
        return facet if facet in {"education_status", "employment_status", "life_status"} else "life_status"
    return ""


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _fact_time_key(fact: dict) -> float:
    value = _clean(fact.get("observed_at") or fact.get("recorded_at"), 80)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _list(value: Any) -> list:
    if isinstance(value, list):
        return value
    try:
        from engine.chunk_evidence import loads_json_list
        return loads_json_list(value)
    except Exception:
        return []


def is_legacy_persona_memory_key(value: Any) -> bool:
    """Identify Alice facts kept in relationship_memory before cutover.

    Adapters still read the raw JSON column for the one-time backfill, while
    normal relationship-memory reads must not expose these rows as a second
    runtime authority.
    """
    key = _clean(value, 180)
    return any(key == prefix or key.startswith(prefix + ":") for prefix in _LEGACY_PREFIXES)


def legacy_items_to_facts(items: list[dict]) -> list[dict]:
    """Convert legacy relationship-memory rows without guessing missing data."""
    facts: list[dict] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = _clean(item.get("key"), 180)
        if not is_legacy_persona_memory_key(key):
            continue
        legacy_id = _clean(item.get("id"), 180)
        value = _clean(item.get("value"), 1000)
        if not value:
            continue
        facet = key.split(":", 1)[1] if ":" in key else key.rsplit("_", 1)[-1]
        kind = "identity" if key in {"tutor_persona_origin", "tutor_persona_residence"} else key.split(":", 1)[0].removeprefix("tutor_persona_")
        status = "superseded" if item.get("superseded_by") else (
            "candidate" if item.get("change_status") == "candidate" else "active"
        )
        fact_id = legacy_id or _stable_id("legacy", key, value, item.get("created_at"))
        if fact_id in seen:
            continue
        seen.add(fact_id)
        facts.append({
            "fact_id": fact_id,
            "persona_key": PERSONA_KEY,
            "kind": kind,
            "facet_key": facet,
            "value_text": value,
            "status": status,
            "source_kind": _clean(item.get("source") or "legacy_unknown", 80),
            "source_conv_id": _clean(item.get("source_conv_id"), 160) or None,
            "source_turn": _int_or_none(item.get("source_turn")),
            "source_message_ref": _clean(item.get("source_message_ref"), 200),
            "evidence_quote": _clean(item.get("evidence") or item.get("value"), 1000),
            "confidence": _float(item.get("confidence"), 0.5),
            "observed_at": _clean(item.get("observed_at") or item.get("created_at"), 80),
            "valid_from": _clean(item.get("valid_from") or item.get("created_at"), 80),
            "valid_to": _clean(item.get("valid_to"), 80),
            "recorded_at": _clean(item.get("recorded_at") or item.get("created_at"), 80),
            "time_basis": _clean(item.get("time_basis") or "recorded_at", 40),
            "supersedes": _clean(item.get("supersedes") or item.get("superseded_by"), 180),
            "supporting_fact_ids": _list(item.get("supporting_fact_ids")),
            "related_entity_ids": _list(item.get("related_entity_ids")),
            "legacy_fact_id": legacy_id,
            "source_occurrence_id": "legacy:" + fact_id,
        })
    return facts


def extract_explicit_place_facts(text: str) -> dict[str, str]:
    """Extract only Alice's explicit first-person origin/residence statements."""
    body = _clean(text, 2000)
    result = {"origin": "", "residence": ""}
    origin = re.search(
        r"\b(?:I'm from|I am from|I come from|I grew up in|I was born in)\s+"
        r"([A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*){0,2})",
        body,
    )
    if origin:
        place = re.sub(r"\s+(?:actually|really|originally)$", "", origin.group(1), flags=re.I)
        result["origin"] = place.strip(" ,.;:!?—-")[:80]
    else:
        # Preserve the prior narrow demonym exception. Never infer a place from
        # language, food, cultural stereotypes, or a place mentioned in passing.
        demonym = re.search(r"\bI(?:['’]m| am)\s+(?:a\s+)?Singaporean\b", body, re.I)
        if demonym:
            result["origin"] = "Singapore"
    residence = re.search(
        r"\bI (?:currently )?(?:live|stay) in\s+"
        r"([A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*){0,2})",
        body,
    )
    if residence:
        result["residence"] = residence.group(1).strip(" ,.;:!?—-")[:80]
    return result


def legacy_confirmed_identity_to_facts(
    rows: list[dict], *, user_id: str,
) -> list[dict]:
    """Backfill only archived Alice place claims that the user explicitly confirmed.

    The archive is not blindly treated as an Alice biography: a traceable
    first-person assistant statement and an immediately following user
    confirmation naming the same place are both required.
    """
    facts: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        conv_id = _clean(row.get("conv_id"), 160)
        assistant_seq = _int_or_none(row.get("seq"))
        confirmation_seq = _int_or_none(row.get("confirmed_by_seq"))
        assistant_text = _clean(row.get("content"), 2000)
        confirmation = _clean(row.get("confirmation"), 1000)
        observed_at = _clean(row.get("created_at"), 80)
        if not conv_id or assistant_seq is None or confirmation_seq is None:
            continue
        places = extract_explicit_place_facts(assistant_text)
        for facet, place in places.items():
            place = _clean(place, 80)
            if not place:
                continue
            escaped = re.escape(place)
            named_place = re.search(rf"\b{escaped}\b", confirmation, re.I)
            if not named_place:
                continue
            before = confirmation[max(0, named_place.start() - 48):named_place.start()]
            after = confirmation[named_place.end():named_place.end() + 64]
            if facet == "origin":
                claim = re.search(
                    r"\b(?:you(?:['’]re| are)\s+(?:from|a))\s*$",
                    before, re.I,
                )
            else:
                claim = re.search(
                    r"\b(?:you(?:['’]re| are)\s+(?:currently\s+)?(?:living|staying)|"
                    r"you\s+(?:live|stay))\s+(?:in\s+)?$",
                    before, re.I,
                )
            explicit_confirmation = re.search(
                r"\b(?:right|correct|is that right|did I get that right)\b|"
                r"(?:对吧|是吧|对吗|是吗|没错)", after, re.I,
            )
            leading_check_question = bool(
                re.match(r"^\s*(?:so|then)\b", confirmation, re.I)
                and re.search(r"[?？]", after)
            )
            confirmation_marker = explicit_confirmation or leading_check_question
            if not claim or not confirmation_marker:
                continue
            occurrence = _stable_id(
                "archive-confirmed-identity", user_id, conv_id,
                assistant_seq, confirmation_seq, facet, place,
            )
            facts.append({
                "fact_id": occurrence,
                "persona_key": PERSONA_KEY,
                "kind": "identity",
                "facet_key": facet,
                "value_text": place,
                "status": "active",
                "source_kind": "archived_assistant_disclosure_user_confirmed",
                "source_conv_id": conv_id,
                "source_turn": assistant_seq,
                "source_message_ref": (
                    f"assistant_seq:{assistant_seq};user_confirmed_seq:{confirmation_seq}"
                ),
                "evidence_quote": assistant_text,
                "confidence": 0.96,
                "observed_at": observed_at,
                "valid_from": observed_at,
                "valid_to": "",
                "recorded_at": observed_at,
                "time_basis": "archived_message_created_at",
                "supersedes": "",
                "supporting_fact_ids": [],
                "related_entity_ids": [],
                "legacy_fact_id": "",
                "source_occurrence_id": occurrence,
            })
    return facts


def observation_to_fact(
    fact: dict, *, user_id: str, conv_id: str, turn: int,
    observed_at: str, evidence_quote: str = "", confidence: float = 0.62,
    interaction_context: str = "", interaction_context_source: str = "",
) -> dict:
    """Build an idempotent row from one explicit Alice self-disclosure."""
    category = _clean(fact.get("category"), 100)
    facet = _clean(fact.get("facet") or fact.get("value"), 160).casefold()
    value = _clean(fact.get("value"), 1000)
    occurrence = _stable_id(user_id, conv_id, turn, category, facet, value)
    return {
        "fact_id": occurrence,
        "persona_key": PERSONA_KEY,
        "kind": category.removeprefix("tutor_persona_") or "self_disclosure",
        "facet_key": facet,
        "value_text": value,
        "status": "active",
        "source_kind": "assistant_self_disclosure",
        "source_conv_id": _clean(conv_id, 160) or None,
        "source_turn": int(turn or 0),
        "source_message_ref": "",
        "evidence_quote": _clean(evidence_quote or value, 1000),
        # This is the immediately preceding conversational cue, not evidence
        # that the user's described event happened to Alice. Preserve its
        # provenance so cold reflection can distinguish raw ASR from display text.
        "interaction_context": _clean(interaction_context, 500),
        "interaction_context_source": (
            _clean(interaction_context_source, 40)
            if interaction_context_source in {"raw_asr", "user_text"} else ""
        ),
        "confidence": _float(confidence, 0.62),
        "observed_at": _clean(observed_at, 80),
        "valid_from": _clean(observed_at, 80),
        "valid_to": "",
        "recorded_at": _clean(observed_at, 80),
        "time_basis": "observed_at",
        "supersedes": "",
        "supporting_fact_ids": [],
        "related_entity_ids": [],
        "legacy_fact_id": "",
        "source_occurrence_id": occurrence,
    }


def query_terms(query: str) -> set[str]:
    return {
        term for term in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", _clean(query, 500).casefold())
        if len(term) > 1 and term not in _STOPWORDS
    }


def wants_persona_recall(query: str) -> bool:
    text = _clean(query, 500).casefold()
    # 10-12R12：**求翻译类问句永远不是"问她自己"** —— 被引用的那句是素材，不是提问。
    # R7~R11 只用 `(?<!['\"])` 挡"短语正好在引号后面"的情形（How do you say "your morning
    # routine"…）；当引用的整句以 What's/Tell me 开头时（How do you say "What's your
    # morning routine?" in Chinese?）那个 lookbehind 够不着 ⇒ 假命中。这里补一条窄守卫。
    # 只挡"怎么说/怎么读/翻译"这一类明确的翻译请求，不挡"怎么说呢"这种语气词。
    if re.search(
        r"\bhow\s+(?:do|would|can|did|to)\s+(?:you|i|we|one)\s+(?:say|translate|write)\b|"
        r"\btranslate\s+(?:this|that|it|these|those)\b|"
        r"怎么(?:说|读|翻译)(?!呢)|用(?:英语|英文|中文|汉语)怎么(?:说|读)|翻译(?:成|为)",
        text,
    ):
        return False
    return bool(re.search(
        r"\b(?:what do you know about yourself|tell me about yourself|your background|"
        r"your experience|your personality|what are you like|where are you from|"
        r"what languages do you speak|what do you like|what did you tell me about your|"
        r"how have you changed|how did you change|what were you like before)\b|"
        r"\b(?:do|did)\s+you\s+(?:still\s+)?(?:have|keep)\b.{0,120}\b(?:you|alice)\s+(?:mentioned|said|told me about)\b|"
        r"\b(?:what|which|where|when|how)\b.{0,100}\b(?:you|your|alice)\b.{0,80}\b(?:said|told me|mentioned)\b|"
        # Natural follow-ups can point back first, then ask a separate detail
        # question: "You mentioned that soup. What was in it again?"
        r"\b(?:you|alice)\s+(?:just\s+)?(?:said|told me|mentioned)\b.{0,180}\b(?:what|which|where|when|how)\b|"
        r"\bremember\b.{0,80}\b(?:you|your|yourself)\b|"
        # 10-12R6：补"问她自己生活/计划"的正向分支。这类问句与她的自述词面
        # 零重叠 ⇒ 旧门 off 时逐轮 packet 只剩 2 条强制行（identity+current_status），
        # 她拿不到自己的生活事实、只能当场编（"镜像锚定/悬浮"的读侧一半）。
        # 命中/误伤实测：正例 8/8、负例 0/9（见 REPORT-读侧召回缺口-20261012R5.md）。
        r"\b(?:what|how)\s+(?:have|are)\s+you\s+(?:been\s+)?(?:up\s+to|doing|keeping\s+busy)\b"
        r"(?!\s+(?:this|that|the|my|your)\s+(?:exercise|lesson|task|sentence|word|phrase|"
        r"question|quiz|homework|assignment|drill|unit|chapter|worksheet|translation)\b)|"
        r"\banything\s+of\s+your\s+own\b|"
        r"\byour\s+own\s+(?:plans?|goals?|life|days?|weekends?|evenings?|routine)\b|"
        r"\bhow\s+do\s+you\s+(?:usually\s+)?spend\s+your\s+(?:days?|time|evenings?|weekends?)\b|"
        r"\bwhat\s+do\s+you\s+(?:usually\s+)?do\s+(?:for\s+fun|on\s+weekends?|in\s+your\s+free\s+time)\b|"
        r"你(?:自己)?(?:平时|最近)?(?:在)?(?:忙什么|做什么|怎么过)|"
        r"你自己(?:的)?(?:生活|计划|打算)|你有什么(?:计划|打算)|"
        # 10-12R7：EN/ZH 对齐，补"问她靠什么生活/做什么工作"这一族。
        # 现象：EN 侧整族 miss（What do you do for a living? / What's your job? /
        # Do you have a job? …），ZH 侧当"是/的"夹在中间时也 miss（你是做什么的？）
        # ——而 `你做什么工作？` 命中 ⇒ 同一语义中英结果不同。门 miss 时逐轮 packet
        # 只剩 2 条强制行（identity+current_status），她想聊自己的生活却拿不到素材。
        # 命中/误伤实测：正例 16/16、负例 0/15（见 _probe_1012s/t/u.py）。
        r"(?<!['\"])\bwhat\s+do\s+you\s+do\s+(?:for\s+(?:a\s+living|a\s+job|work|a\s+career)|all\s+day|these\s+days)\b|"
        r"(?<!['\"])\bwhat(?:'s|\s+is)\s+your\s+(?:day\s+)?(?:job|work|profession|occupation)\b|"
        r"\bdo\s+you\s+have\s+a\s+(?:steady\s+)?(?:job|profession|career)\b(?!\s+for\s+(?:me|us)\b)|"
        r"你是做什么的|你的工作(?:是什么)?|你有工作(?:吗|么)|你靠什么(?:生活|谋生)|"
        # 10-12R8：补 EN/ZH 两侧"问她的日常/兴趣/近况"更自然的说法。复现：本门原先
        # 只认 R6/R7 那几种措辞 ⇒ `_probe_1012w.py` 实测 **14/20 真措辞全 miss**
        # （What do you do? / What are your hobbies? / What's your daily routine? /
        #  How's your day going? / What did you do today? / 你平时喜欢做什么？ /
        #  你有什么爱好？ / 你周末一般做什么？ …）。这些问句与她的自述**零词面重叠**
        # ⇒ 门 off 时逐轮 packet 又退回 2 条强制行（identity+current_status），
        # 「存了但用不上」。命中/误伤实测（`_probe_1012x.py`）：正例 30/30、负例 0/30。
        r"(?<!['\"])\bwhat\s+do\s+you\s+do\s*(?:,?\s*(?:alice|then))?\s*[?.!]*\s*$|"
        r"(?<!['\"])\b(?:do\s+you\s+have\s+(?:any\s+)?hobbies|what\s+are\s+your\s+hobbies|your\s+hobbies?)\b|"
        r"(?<!['\"])\byour\s+free\s+time\b|"
        r"(?<!['\"])\byour\s+(?:daily|typical)\s+(?:routine|day)\b"
        r"(?!.{0,60}\b(?:material|lesson|passage|text|exercise|course|article|worksheet|unit|chapter|cassette)\b)|"
        r"\btypical\s+day\b[^?.!]{0,40}\bfor\s+you\b|"
        r"\bhow(?:'s|\s+is|\s+was)\s+your\s+(?:day|morning|afternoon|evening|night|week|weekend)\b|"
        r"\bwhat\s+did\s+you\s+do\s+(?:today|yesterday|last\s+night|this\s+(?:morning|afternoon|evening|week|weekend))\b|"
        r"\byour\s+plans?\s+for\s+(?:the\s+|this\s+|next\s+)?(?:weekend|week|holiday|summer|tonight|break)\b|"
        r"你(?:自己)?(?:平时|平常|周末|一般|最近|每天|通常|今天|现在|打算|都|在|会|喜欢|爱){0,3}(?:做什么|做些什么|干什么|干些什么|干嘛|玩什么|忙什么|怎么过|过得怎么样|过得如何)|"
        r"你(?:有|喜欢)什么(?:爱好|兴趣)|你的(?:爱好|兴趣)|"
        # 10-12R9：补"反问/互问""近况问候""问她喜好"三族。复现（`_probe_1012z.py`，
        # 改前）：**POS 漏 37/56** —— 反问族（What about you? / 你呢？）整族 miss，
        # 而它正是本产品最高频的"用户说完自己 → 请她讲她自己"的句式；近况问候
        # （How have you been? / What's new with you? / 你最近怎么样？）与问她喜好
        # （What's your favorite food? / 你最喜欢什么？）同样 miss ⇒ 门 off 时逐轮
        # packet 只剩 2 条强制行（identity+current_status），她拿不到自己的记录。
        # 命中/误伤实测（`_probe_1012z.py`）：正例 0 漏、负例 0 误伤（含
        # "How have you been teaching me?" / "What's new in this lesson?" /
        # "What's your favorite example for this grammar point?" / "这个句子呢？" /
        # "I like tea and you like coffee." / "这个项目最近怎么样？"）。
        # 同批顺带收紧 R6 的 `doing`（`How are you doing this exercise?` 原为假命中）。
        r"(?<!['\"])(?:what|how)\s+about\s+(?:you|yourself|alice)\b"
        r"(?!\s+(?:explain|teach|show|give|write|read|say|tell|try|use|translate|correct|fix|do)\b)|"
        r"\band\s+you\b\s*(?:,\s*alice)?\s*[?.!]*$|"
        r"你\s*呢\s*[?？!！。]*\s*$|"
        r"\bhow(?:'ve|\s+have)\s+you\s+been\b"
        r"(?:\s+(?:lately|these\s+days|recently|doing|keeping|getting\s+on|feeling))?\s*[?.!]*$|"
        r"\bhow(?:'s|\s+is|\s+are)\s+(?:life|things)\b|"
        r"\bwhat(?:'s|\s+is)\s+new\s+with\s+you\b|"
        r"\b(?:anything|something)\s+new\s+with\s+you\b|"
        r"\bwhat\s+about\s+your\s+(?:day|week|weekend|evening|life|hobbies?|routine|plans?)\b|"
        r"\bwhat(?:'s|\s+is)\s+your\s+favou?rite\s+"
        r"(?:food|dish|meal|drink|color|colour|season|movie|film|book|song|music|band|artist|"
        r"place|city|country|animal|pet|hobby|sport|subject|flower|tea|coffee|dessert|breakfast|"
        r"weather|time\s+of\s+day|day\s+of\s+the\s+week|thing\s+to\s+do)\b|"
        r"\bdo\s+you\s+have\s+a\s+favou?rite\s+"
        r"(?:food|dish|meal|drink|color|colour|season|movie|film|book|song|music|band|artist|"
        r"place|city|country|animal|pet|hobby|sport|subject|tea|coffee|dessert)\b|"
        r"\bwhat\s+kind\s+of\s+\w+\s+do\s+you\s+like\b|"
        r"\bwhat\s+are\s+you\s+into\b|"
        r"\bwhat\s+do\s+you\s+enjoy\s+doing\b|"
        r"你最喜欢(?:的)?(?:什么|啥|哪|吃|喝|做|玩|颜色|季节|电影|书|歌|音乐|动物|地方|运动|食物|菜)|"
        r"你(?:最近|近来|现在)?(?:过得)?(?:怎么样|还好吗|好吗|如何)|"
        r"(?:^|[，,。!？?\s])最近(?:过得)?(?:怎么样|如何)|"
        r"你(?:最近|近来)?有(?:什么|啥)(?:新鲜事|新计划|新打算|新想法)|"
        # 10-12R10：补"问她喜好/经历/日常细节"四族。复现（`_probe_1012ab.py`，
        # 改前 **POS 漏 23/39**）：① `Do you like X?`（Do you like coffee? /
        # Do you like your job? / 你喜欢咖啡吗？ / 你喜欢住在哪儿？）；
        # ② `Have you ever …?`（Have you ever been to Japan? / 你吃过榴莲吗？）；
        # ③ 她的日常细节（Where do you live? / Do you have any pets? /
        # What do you usually have for breakfast? / What are you reading these days? /
        # Are you a morning person? / 你养宠物吗？ / 你平时几点起床？）；
        # ④ "请她自述"的遗漏措辞（Tell me something about yourself. / What's your story? /
        # What do you do when you're not working?）。这些问句与她的自述**零词面重叠**
        # ⇒ 门 off 时逐轮 packet 又退回 2 条强制行（identity+current_status）。
        # 负向守卫（把"问我写的东西 / 产品对象"的问句挡在外面）：
        #  `do you like` 后紧跟 my/our/this/that/the/its/his/her/it ⇒ 不命中
        #   （Do you like my writing? / Do you like this sentence?）；
        #  `have you ever` 后 40 字内出现 this/that/the/my/your + 产品名词 ⇒ 不命中
        #   （Have you ever seen this word? / …taught this lesson?）；
        #  中文 `你喜欢` 后紧跟 我/这/那/它 ⇒ 不命中（你喜欢这个句子吗？/ 你喜欢我写的作文吗？）；
        #  `你去过…` 后紧跟 这/那/该 ⇒ 不命中（你玩过这个游戏吗？）。
        # 命中/误伤实测（`_probe_1012ab.py ab`）：POS 漏 23/39 → 0/39、NEG 0/32 → 0/32；
        # R8/R9 既有语料回归 0 漏 0 误伤（`_probe_1012x.py ab` / `_probe_1012z.py ab`）。
        r"\btell\s+me\s+(?:something|a\s+little|a\s+bit|more)\s+about\s+yourself\b|"
        r"\bwhat(?:'s|\s+is)\s+your\s+story\b|"
        r"\bwhere\s+do\s+you\s+live\b(?!\s+in\s+(?:this|that)\b)|"
        r"\bdo\s+you\s+(?:have|got)\s+(?:any\s+)?(?:pets?|cats?|dogs?|kids?|children|siblings?|brothers?|sisters?)\b|"
        r"\bwhat\s+do\s+you\s+(?:usually\s+)?(?:have|eat)\s+for\s+(?:breakfast|lunch|dinner|supper)\b|"
        r"\bwhat\s+are\s+you\s+(?:reading|watching|listening\s+to|playing)\b(?!\s+in\s+(?:this|that)\b)|"
        r"\bwhat\s+do\s+you\s+do\s+when\s+you(?:'re|\s+are)\s+not\s+(?:working|teaching|studying|busy)\b|"
        r"\bare\s+you\s+a\s+(?:morning|night|cat|dog|people|coffee|tea|city|country|beach|"
        r"winter|summer|book|music)\s+(?:person|owl|lover|kind|type)\b|"
        r"\bdo\s+you\s+like\s+(?!\s*(?:my|our|this|that|the|its|his|her|it)\b)\S|"
        r"\bhave\s+you\s+ever\b(?![^.?!]{0,40}\b(?:this|that|the|my|your)\s+"
        r"(?:word|phrase|sentence|expression|grammar|lesson|exercise|example|translation|"
        r"material|chapter|unit|quiz|drill|worksheet|point|rule)\b)|"
        r"你(?:养|有)(?:宠物|猫|狗)(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:平时|一般|通常)?几点(?:起床|睡|睡觉|下班|上班|吃饭|回家)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:去过|吃过|喝过|住过|养过|玩过|试过)(?!这|那|该)|"
        r"你喜欢(?!我|这|那|它)|"
        # 10-12R11：补 ZH 侧"她的居住/家庭/宠物/三餐/在看在读/周末安排/目标/作息"。
        # 这一族 EN 侧 R10 已覆盖（where do you live / any pets / breakfast /
        # what are you reading / are you a morning person），但 **ZH 侧整族 miss**
        # ⇒ 中文用户问同一件事时，门 off ⇒ 逐轮 packet 退回 2 条强制行
        # （identity+current_status），她拿不到自己的记录（＝§7 D 悬浮的读侧一半）。
        # R7 已证明 EN/ZH 会不对称，本轮把 R10 那一族在 ZH 补齐，并补 EN 无 your 的
        # "周末计划"说法。复现（`_probe_1012ad.py` 对 R10 源码）：**POS 漏 29/51**。
        # 负向守卫：①`(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))` 挡"…怎么说"
        #   这类求翻译（本产品高频：你养宠物的英文怎么说？/ 你住在哪儿的英文怎么说？）；
        #   ②`你` 后不接 我/这/那/该（问的是产品对象而非她自己）。
        # 命中/误伤实测（`_probe_1012ad.py ab`）：见 REPORT-G门R11-中文日常族-20261012.md。
        r"你(?:现在|目前)?住(?:在)?(?!这|那|该)(?:哪儿|哪里|哪|什么地方|什么城市|哪个城市|哪个区)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:有|养)(?!这|那|该)(?:了)?(?:什么|啥|几只|多少)?(?:宠物|猫|狗|花|植物)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你家(?:里)?(?:有|养)(?!这|那|该)(?:什么|啥)?(?:宠物|猫|狗)|"
        r"你(?:有|要)(?:几个)?(?:兄弟姐妹|兄弟|姐妹|哥哥|姐姐|弟弟|妹妹|孩子|小孩|儿女)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:早饭|早餐|午饭|午餐|晚饭|晚餐|早上|中午|晚上)(?:一般|通常|平时)?(?:都)?"
        r"(?:爱|喜欢)?(?:吃|喝)(?:什么|啥|点什么)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:平时|最近|最近在|现在)?(?:在)?(?:看|读|追)(?:什么|啥)(?:书|小说|剧|电影|电视|漫画)|"
        r"(?:你(?:这|下)?(?:个)?周末|(?:这|下)?(?:个)?周末你)"
        r"(?:有|要|打算|准备)?(?:什么|啥)?(?:安排|计划|打算|做什么|干什么)|"
        r"(?<!['\"])\bany\s+plans?\s+for\s+(?:the\s+|this\s+|next\s+)?(?:weekend|week|holiday|break|summer)\b|"
        r"(?<!['\"])\bdo\s+you\s+have\s+any\s+plans?\s+(?:for\s+)?(?:the\s+|this\s+|next\s+)?"
        r"(?:weekend|week|holiday|break|summer)\b|"
        r"你(?:的)?(?:目标|梦想|理想)(?:是什么|是啥)|你有什么(?:目标|梦想|理想)|"
        r"(?<!['\"])\bwhat(?:'s|\s+is|\s+are)\s+your\s+(?:dream|goal|goals|ambition|aspiration|plans?)\b"
        r"(?!\s+for\s+(?:the\s+|this\s+|my\s+|our\s+)?"
        r"(?:lesson|class|course|exercise|task|homework|assignment|quiz|test|exam)\b)|"
        r"你是(?:个)?(?:早起|早睡|夜猫子|喜欢熬夜|爱睡懒觉)(?:的)?(?:人|家伙|那种人)?|"
        # 10-12R12：补 **EN 侧"她的日常"三族**（作息时间 / 日常 routine / "讲述你的一天"）
        #        ＋今晚计划 ＋ ZH 两处补漏。复现（`_probe_1012af.py` 对 R11 源码）：**POS 漏 18/43**。
        # 现象：R10/R11 把"日常细节"只补到 **ZH 侧**（`你几点起床`）和 **EN 的 where/reading/breakfast**，
        #       而 EN 侧更自然的说法整族 miss：`What time do you get up?` / `What's your morning routine?` /
        #       `Tell me about your day.` / `Do you have any plans tonight?` /
        #       `What do you usually do in the evening?` ⇒ **又一处 EN/ZH 不对称**（同 R7/R11）。
        #       门 miss ⇒ 逐轮 packet 退回 2 条强制行（identity+current_status），她拿不到自己的记录
        #       （＝§7 D 悬浮的读侧一半）。
        # 负向守卫：①`(?<!['\"])` 挡"引用整句求翻译"（How do you say "What's your morning routine?" in Chinese?）；
        #   ②`你的周末一般怎么过` 等 ZH 行带 `(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))` 挡求翻译；
        #   ③`tell me about your day` 不带 `in this/that`；`morning routine` 不带 `for the exam/lesson…`；
        #   ④`你` 开头（"我们周末…"不命中）。
        # 命中/误伤实测（`_probe_1012af.py ab`）：见 REPORT-G门R12-EN日常三族-20261012.md。
        r"(?<!['\"])\bwhat\s+time\s+do\s+you\s+(?:usually\s+|normally\s+|typically\s+)?"
        r"(?:get\s+up|wake\s+up|go\s+to\s+bed|go\s+to\s+sleep|have\s+breakfast|lunch|dinner)\b|"
        r"(?<!['\"])\bwhen\s+do\s+you\s+(?:usually\s+|normally\s+|typically\s+)?"
        r"(?:get\s+up|wake\s+up|go\s+to\s+bed|go\s+to\s+sleep|have\s+(?:breakfast|lunch|dinner))\b|"
        r"(?<!['\"])\byour\s+(?:morning|evening|night|weekend|weekday|daily|typical)\s+routine\b"
        r"(?!\s+for\s+(?:the\s+|this\s+|my\s+|our\s+)?(?:exam|test|lesson|class|course|exercise|task|quiz)\b)|"
        r"(?<!['\"])\btell\s+me\s+about\s+your\s+(?:day|week|weekend|morning|evening|night|life|lifestyle)\b"
        r"(?!\s+in\s+(?:this|that|the)\b)|"
        r"(?<!['\"])\b(?:do\s+you\s+have\s+)?any\s+plans?\s+(?:for\s+)?"
        r"(?:tonight|this\s+(?:evening|weekend|week)|the\s+(?:evening|night|weekend))\b|"
        r"(?<!['\"])\bwhat\s+do\s+you\s+(?:usually\s+|normally\s+)?do\s+"
        r"(?:in\s+the\s+(?:morning|afternoon|evening|evenings)|at\s+night|after\s+work|on\s+weekdays?)\b|"
        # ZH 补漏：R11 的"周末安排"行只认 `你周末…安排/计划/做什么`，挡不住"你的周末一般怎么过？"
        #   （中间多了"的"）；"你是哪里人 / 你老家在哪"也整族缺（EN 侧 R0 已有 where are you from）。
        r"你(?:的)?(?:每个)?(?:周末|假期|假日|晚上)(?:一般|通常|平时)?(?:都)?"
        r"(?:怎么过|干什么|做什么|做些啥|怎么安排|有什么安排)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:是哪里人|老家(?:在)?(?:哪|哪里|哪儿))"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        # 10-12R14：补 **"How's your <生活域> going / coming along?"** 这一族 ——
        # 它是"用户顺着她自己某件事追问进展"的最自然说法（真链路 R13 用过
        # `How is your running going these days?`），但 R6~R13 只覆盖了
        # `how's your (day|week|weekend|…)` 与 `your … routine`，**具体生活域名词整族 miss**。
        # 复现（`_probe_1012al.py` 对 R13 live 源码）：**POS 33 例中漏 23** ——
        # How's your running going? / How's your marathon training going? /
        # How's your new job going? / How's your book coming along? /
        # 你的跑步最近怎么样？ / 你的训练进展如何？ / 你的新工作还顺利吗？ …
        # 门 miss ⇒ 逐轮 packet 退回 2 条强制行（identity+current_status），
        # 她明明有 running-goal 这类事实却拿不到（＝§7 D 悬浮的读侧一半）。
        # 负向守卫：① 主语必须是 `your|the`（`my/our/this/that` 不命中
        #   ⇒ How's my running going? ✗）；② 生活域走**白名单**，刻意**不含**
        #   english/chinese/homework/exercise/translation 等产品对象词
        #   ⇒ How's your English homework going? / How's your exercise going? ✗；
        #   ③ 顶层"求翻译"守卫已挡 `How do you say …` 与 `…怎么说`。
        r"(?<!['\"])\bhow(?:'s|\s+is|\s+are)\s+(?:your|the)\s+"
        r"(?:(?:new|side|little|other|own|latest)\s+)?"
        r"(?:running|jogging|training|marathon|workout|swimming|cycling|yoga|garden|gardening|"
        r"studio|music|band|guitar|piano|book|reading|writing|novel|art|painting|pottery|cooking|"
        r"baking|studies|study|research|project|job|career|business|shop|bookshop|blog|podcast|"
        r"channel|health|sleep|recovery|diet|trip|travel)\b"
        r"(?:\s+\w+){0,2}\s+(?:going|coming\s+along|doing|treating\s+you)\b|"
        r"(?<!['\"])\bwhat\s+have\s+you\s+been\s+(?:reading|watching|listening\s+to|playing)\b|"
        r"\b(?:anything|something)\s+new\s+(?:on\s+your\s+side|on\s+your\s+end)\b|"
        # ZH 同族（R14）：`你的<生活域>(最近)怎么样了 / 进展如何 / 还顺利吗`。
        # 求翻译守卫同 R11（`…怎么说/的英文` 不命中）。
        r"你(?:的)?(?:(?:新|小|副)?(?:工作|事业|项目|副业)|跑步|晨跑|长跑|马拉松|训练|健身|"
        r"锻炼|游泳|骑行|瑜伽|花园|菜园|书|小说|写作|画画|陶艺|做菜|烘焙|学习|研究|博客|播客|"
        r"音乐|吉他|钢琴|身体|睡眠|恢复|饮食)"
        r"(?:最近|现在|最近这阵子|这阵子)?"
        r"(?:跑得|进行得|看得|读得|练得|搞得|做|准备)?"
        r"(?:怎么样了|怎么样啦|怎么样|如何|进展如何|进展怎么样|还顺利吗|顺利吗|还好吗|有没有进展)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:最近|现在)?在忙(?:的)?(?:那件事|事情|项目)(?:的)?(?:进展|情况)?(?:如何|怎么样)|"
        # 10-12R15：补"问她工作/职业/谋生"这一族 —— **ZH 侧几乎整族 miss**，EN 侧只差 3 种说法。
        # R7 只对齐了 EN（`what do you do for a living` / `what's your job` / `do you have a job`）
        # 与 ZH 的 `你是做什么的` / `你的工作(是什么)`；而 ZH 最自然的一族整族 miss：
        #   `你是做什么工作的？` / `你的职业是什么？` / `你从事什么工作？` / `你在哪儿工作？` /
        #   `你上班吗？` / `你平时上班累不累？`；EN 也仍缺 `what line of work are you in` /
        #   `do you work` / `are you working`。
        # 复现（`_probe_1012an.py` 对 R14 live 源码）：**POS 漏 18/35**，本族独占 9 条。
        # 门 miss ⇒ 逐轮 packet 退回 2 条强制行（identity+current_status），她明明有
        # `work`/`current_status` 类事实却拿不到（＝§7 D 悬浮的读侧一半、D2 事业线的读侧）。
        # 负向守卫：①ZH 各行沿用 R11 的 `(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))`
        #   挡求翻译（`你做什么工作的英文怎么说？` / `你的职业用英语怎么说？` / `你上班的英文怎么说？`）；
        #   ②`你` 后只接"上班/工作/从事/做"⇒ `我的工作是什么？` / `我们公司是做什么的？` /
        #   `这个练习是做什么的？` 不命中；③EN `do you work` / `are you working` 后挡
        #   `on|through|with` 与 `for me/us`（`Do you work on this exercise with me?` ✗）。
        # 命中/误伤实测（`_probe_1012an.py ab`）：见 REPORT-G门R15-工作职业族-20261012.md。
        r"(?<!['\"])\bwhat\s+line\s+of\s+work\s+are\s+you\s+in\b|"
        r"(?<!['\"])\bdo\s+you\s+work\b(?!\s+(?:on|through|with)\b)(?!\s+for\s+(?:me|us)\b)|"
        r"(?<!['\"])\bare\s+you\s+(?:still\s+|currently\s+)?working\b"
        r"(?!\s+(?:on|through|with)\b)(?!\s+for\s+(?:me|us)\b)|"
        r"你是做什么工作的(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你的(?:职业|行业)(?:是什么|是啥)?(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:从事|做)(?:什么|啥)(?:的)?(?:工作|职业|行业)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你在(?:哪儿|哪里|哪)(?:上班|工作)(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:平时|平常|一般|现在|最近)?(?:上班|工作)(?:吗|忙|累|辛苦|顺心)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        # 10-12R16：补"问她自己的生活细节"这一族 —— 业余时间/看电影/理想的一天、
        #   家庭婚姻、近况问候变体、自我介绍变体。复现（`_probe_1012aq.py` 对 R15 live
        #   源码，47 正 + 20 负）：**POS 漏 27/47**，本族独占 27 条：
        #     `你平时业余时间做什么？` / `你下班后一般做什么？` / `你爱看什么电影？` /
        #     `Do you watch movies?` / `What movies do you like?` /
        #     `What's your idea of a good weekend?` / `你理想的一天是什么样的？` /
        #     `Do you have a family?` / `Are you married?` / `你结婚了吗？` / `你有老公吗？` /
        #     `How have things been with you?` / `What's going on with you?` /
        #     `Introduce yourself.` …
        #   门 miss ⇒ 逐轮 packet 退回 2 条强制行（identity+current_status），她明明有
        #   自己的记录却拿不到（＝§7 D 悬浮的读侧一半）。
        # 负向守卫：①ZH 各行沿用 R11 的 `(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))`
        #   挡求翻译（你爱看什么电影的英文怎么说？/ 你结婚了吗的英文怎么说？）；
        #   ②EN `introduce yourself` 前加 `(?<!['\"])`，且顶层求翻译守卫已挡 `How do you say …`；
        #   ③EN `do you have a …` 后挡 `for me/us` 与 `for the/this/my/our lesson…`（产品对象）；
        #   ④EN `what's going on` 必须是 `with/for you`（`What's going on with this sentence?` ✗）；
        #   ⑤EN 看电影行挡 `in/for/with this/that/the course|lesson|material…`；
        #   ⑥"业余/理想"只认她的生活域（`good weekend/day/…`，不含 `good answer`）；
        #   ⑦ZH 自我介绍行要求"介绍"后紧跟"你"（`介绍一下这个单词` ✗）。
        # 命中/误伤实测（`_probe_1012aq.py ab`）：见 REPORT-G门R16-生活细节族-20261012.md。
        r"你(?:平时|平常)?(?:的)?(?:业余|空闲|休闲|休息)时间(?:都|一般|通常|平时|会)?(?:在)?"
        r"(?:做|干|忙)(?:什么|啥|点啥|些啥)?|"
        r"你(?:下班后|下班之后|下了班)(?:一般|通常|平时|都)?(?:在)?(?:做|干|忙)(?:什么|啥|点啥)?|"
        r"你(?:休息的?时候|有空的?时候|闲下来的?时候|不上班的?时候)"
        r"(?:一般|通常|平时|都|喜欢|爱)?(?:在)?(?:做|干|玩)(?:什么|啥|点啥)?|"
        r"\bwhat\s+do\s+you\s+(?:usually\s+|normally\s+)?do\s+in\s+your\s+(?:spare|free|idle|off)\s+time\b|"
        r"\bhow\s+do\s+you\s+(?:usually\s+|normally\s+)?spend\s+your\s+(?:spare|free|idle|off)\s+time\b|"
        r"\bwhat\s+do\s+you\s+(?:like|love|enjoy)\s+to\s+do\s+in\s+your\s+(?:spare|free|idle|off)\s+time\b|"
        r"\bwhat\s+do\s+you\s+do\s+(?:after\s+work|when\s+you(?:'re|\s+are)\s+off\s+work)\b|"
        r"你(?:平时|平常|最近|一般|通常)?(?:喜欢|爱|最爱|常|经常|都)?(?:看|追)(?:什么|啥|哪)"
        r"(?:类型|种类|部)?(?:的)?"
        r"(?:电影|影片|片子|剧|电视剧|电视|综艺|节目|美剧|韩剧|英剧|动漫|纪录片)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"\b(?:what|which)\s+(?:movies?|films?|shows?|tv\s+shows?|series|tv\s+series)\s+"
        r"do\s+you\s+(?:like|watch|enjoy|prefer)\b|"
        r"\bdo\s+you\s+(?:like\s+to\s+|enjoy\s+)?watch\s+(?:movies?|films?|tv|shows?|series)\b"
        r"(?!.{0,40}\b(?:in|for|with)\s+(?:this|that|the)\s+"
        r"(?:course|lesson|material|exercise|video|clip|class)\b)|"
        r"\bwhat(?:'s|\s+is)\s+your\s+idea\s+of\s+a\s+(?:good|great|perfect|ideal|nice)\s+"
        r"(?:weekend|day|evening|night|holiday|vacation|break|trip|morning|sunday)\b|"
        r"你(?:理想|理想中|心里|心目中)(?:的)?(?:一天|周末|假期|生活|日子)"
        r"(?:是什么|是啥|是什么样的|是怎么样的|什么样)|"
        r"\bare\s+you\s+(?:married|single|divorced|engaged|in\s+a\s+relationship)\b"
        r"(?!\s+to\s+(?:that|this|it)\b)|"
        r"\bdo\s+you\s+have\s+(?:a\s+|any\s+)?"
        r"(?:family|husband|wife|partner|boyfriend|girlfriend|spouse|fiancee?|sons?|daughters?)\b"
        r"(?!\s+for\s+(?:me|us)\b)(?!.{0,24}\bfor\s+(?:the|this|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|homework)\b)|"
        r"\bdo\s+you\s+have\s+(?:a\s+)?significant\s+other\b|"
        r"你(?:已经)?结婚(?:了)?(?:吗|没有|没|么|了没)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:有|交|找了?)(?:老公|老婆|丈夫|妻子|媳妇|对象|男朋友|女朋友|男票|女票|伴侣)"
        r"(?:吗|没)?(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:成家|有对象)(?:了)?(?:吗|没)?|"
        r"\bhow(?:'ve|\s+have)\s+things\s+been\b|"
        r"\bhow\s+has\s+(?:things|life|everything)\s+been\b|"
        r"\bhow(?:'s|\s+is)\s+everything\s+(?:going\s+)?(?:with|for)\s+you\b|"
        r"\bwhat(?:'s|\s+is)\s+going\s+on\s+(?:with|for)\s+you\b|"
        r"\bhow(?:'s|\s+is)\s+life\s+(?:going\s+)?(?:with|for)\s+you\b|"
        r"(?<!['\"])\b(?:can\s+you\s+|could\s+you\s+|please\s+)?introduce\s+yourself\b|"
        r"(?:你)?(?:能|可以)?(?:介绍一下|介绍下)你(?:自己)?|"
        r"自我介绍一下|做个自我介绍|"
        # 10-12R18：补"问她自己的生活/状态"最后一批自然措辞 —— 生活状态追问、
        #   典型一天/周末、"怎么放松"、独居、作息（"什么时候"而非"几点"）、最近在忙什么。
        #   复现（`_probe_1012as.py` 对 R16 源码，44 正 + 20 负）：**POS 漏 17/44**：
        #     How's life treating you? / What's going on in your life? /
        #     What keeps you busy these days? / How are you spending your days lately? /
        #     What's your typical weekend like? / What does your average day look like? /
        #     How do you unwind? / How do you relax? / What do you do to relax? /
        #     What are you passionate about? / 你一个人住吗？ / 你一般什么时候睡觉？ /
        #     你平时怎么放松？ / 你最近在忙些什么？ …
        #   门 miss ⇒ 逐轮 packet 退回 2 条强制行（identity+current_status），
        #   她明明有自己的记录却拿不到（＝§7 D 悬浮的读侧一半）。
        # 负向守卫：①EN 各行带 `(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+
        #   (?:this|that|the|my|our)\s+(?:lesson|class|course|exercise|material|unit|
        #   chapter|passage|text|book|topic|video|clip|quiz|task)\b)`
        #   挡产品对象（How do you unwind after this lesson? ✗ /
        #   What does a typical day look like in this lesson? ✗ /
        #   What are you passionate about in this passage? ✗）；
        #   ②ZH 各行沿用 R11 求翻译守卫（…怎么说/的英文 不命中）；
        #   ③`你` 后不接 我/这/那/该。
        # 命中/误伤实测（`_probe_1012as.py ab`）：见 REPORT-G门R18-生活状态族-20261012.md。
        r"\bhow(?:'s|\s+is)\s+life\s+(?:been\s+)?treating\s+you\b|"
        r"\bwhat(?:'s|\s+is)\s+going\s+on\s+in\s+your\s+life\b|"
        r"\bwhat\s+keeps\s+you\s+busy\b"
        r"(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+(?:this|that|the|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|book|topic|video|clip|quiz|task)\b)|"
        r"\bhow\s+are\s+you\s+spending\s+your\s+(?:days?|time|weekends?|evenings?|summer|holiday)\b"
        r"(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+(?:this|that|the|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|book|topic|video|clip|quiz|task)\b)|"
        r"\bhow\s+do\s+you\s+(?:usually\s+|normally\s+)?(?:unwind|relax)\b"
        r"(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+(?:this|that|the|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|book|topic|video|clip|quiz|task)\b)|"
        r"\bwhat\s+do\s+you\s+do\s+to\s+relax\b"
        r"(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+(?:this|that|the|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|book|topic|video|clip|quiz|task)\b)|"
        r"\bwhat(?:'s|\s+is)\s+your\s+(?:typical|average|usual|normal)\s+"
        r"(?:day|weekend|week|morning|evening)\s+like\b|"
        r"\bwhat\s+does\s+(?:your|a)\s+(?:typical|average|usual|normal)\s+"
        r"(?:day|weekend|week)\s+(?:look\s+like|involve|consist\s+of)\b"
        r"(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+(?:this|that|the|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|book|topic|video|clip|quiz|task)\b)|"
        r"\bwhat\s+are\s+you\s+passionate\s+about\b"
        r"(?!.{0,30}\b(?:in|for|with|on|after|before|during|about)\s+(?:this|that|the|my|our)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|book|topic|video|clip|quiz|task)\b)|"
        r"你(?:一个人|自己|独自)住(?:吗|么|嘛|不)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:(?:平时|平常|一般|通常|每天)\s*)?什么时候"
        r"(?:睡觉|睡|起床|起来|下班|上班|吃饭|回家)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:平时|平常|一般|通常|最近)?(?:都是)?(?:怎么|如何)(?:放松|减压|解压|消遣|休息)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:(?:平时|平常|一般|通常)\s*)?有什么(?:爱好|兴趣)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:最近|现在)(?:都|在)?忙(?:些|点)?(?:什么|啥)|"
        # 10-12R22：补"时点族"——她的今天/今晚/明天/这两天这几天/本周的状态与安排。
        #   复现（`_probe_1012au.py` 对 R21 live 源码，25 正 + 12 负）：**POS 漏 21/25**，
        #   本族独占 21 条：`你今晚有什么安排？` / `你今晚干嘛？` / `你明天有什么打算？` /
        #   `你今天怎么样？` / `你今天忙吗？` / `你这几天在忙什么？` / `你这周过得怎么样？` /
        #   `你本周有什么安排？` / `你昨晚睡得好吗？` / `Did you have a good weekend?` /
        #   `How did you sleep?` …（门 miss ⇒ 逐轮 packet 退回 2 条强制行，＝§7 D 悬浮的读侧一半）。
        #   注：`你今天过得怎么样？` / `你今天怎么过？` / `What are you up to tonight?` /
        #   `How was your weekend?` 在 R8/R6 既有行已命中（回归项），本块不重复。
        # 负向守卫：①各行沿用 R11 求翻译守卫（…怎么说/的英文 不命中）；
        #   ②"忙/累/怎么样"行要求 时点紧跟 你（`做这个练习你累不累？` / `你今天的课怎么样？` /
        #   `你觉得这节课怎么样？` / `这个单词你今天记住了吗？` ✗——中间隔了别的词就不命中）；
        #   ③安排/打算行后接的必须是 安排|打算|计划|行程（`你今天有什么语法问题吗？` ✗）；
        #   ④EN good-day 行走名词白名单 + 挡 with/in/on/during/after + this/that/the + 产品名词。
        r"你(?:今天|今晚|今晚上|明晚|明天|这两天|这几天|这周|本周)(?:一般|通常|都|在){0,2}"
        r"(?:忙|干|做)(?:个|点|些)?(?:什么|啥|嘛)(?:呢|啊|呀)?"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:今天|今晚|今晚上|明晚|明天|这两天|这几天|这周|本周)(?:有什么|有啥)?(?:安排|打算|计划|行程)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:今天|今晚|这两天|这几天|这周|本周)(?:过得)?"
        r"(?:怎么样|如何|还好吗|好吗|累不累|忙不忙|忙吗|累吗)(?:呢|啊|呀)?"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"你(?:昨晚|昨夜|昨天晚上|今天早上|早上)?睡得(?:怎么样|好吗|好不好|香不香|香吗)"
        r"(?!.{0,6}(?:怎么说|怎么读|的英文|的英语|用英语))|"
        r"\bdid\s+you\s+(?:have\s+a\s+good\s+(?:day|weekend|night|morning|evening|time|rest|break|holiday|trip)|"
        r"sleep\s+(?:well|ok|okay)|rest\s+well)\b"
        r"(?!.{0,30}\b(?:with|in|on|during|after)\s+(?:this|that|the)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|quiz|task|video|reading)\b)|"
        r"\bhow\s+did\s+you\s+sleep\b"
        r"(?!.{0,30}\b(?:with|in|on|during|after)\s+(?:this|that|the)\s+"
        r"(?:lesson|class|course|exercise|material|unit|chapter|passage|text|quiz|task|video)\b)|"
        r"(你记得.{0,30}(?:你自己|你之前|我们之前聊过你)|我们之前聊过你|"
        r"你之前告诉我|你之前说过|你从哪里来|你是什么样的人|你有什么经历|"
        r"你会说什么语言|你喜欢什么|你的背景|你的经历|你的性格)", text,
    ))


def is_recalled_persona_echo(
    query: str, candidate_quote: str, facts: list[dict], kind: str = "",
) -> bool:
    """Detect a prior Alice disclosure repeated in a later response.

    A user does not need to ask "do you remember?" for Alice to reuse an old
    self-disclosure. Exact or near-exact repetitions are useful for continuity,
    but are not independent growth observations. Explicit recall answers may
    also add material, so their existing broader match remains available.
    """
    explicit_recall = wants_persona_recall(query)
    candidate = _clean(candidate_quote, 1000).casefold()
    if len(candidate) < 16:
        return False
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", candidate).strip()
    candidate_terms = query_terms(candidate)
    generic = {
        "mine", "were", "small", "too", "that", "thing", "kind", "those",
        "little", "what", "which", "where", "when", "how", "week", "finally",
        "got", "through", "said", "told", "mentioned", "about", "just", "really",
        "something", "similar",
    }
    candidate_terms -= generic
    if not candidate_terms:
        return False
    expected_kind = _clean(kind, 80).casefold()
    for fact in facts or []:
        if not isinstance(fact, dict) or fact.get("hidden_at"):
            continue
        if fact.get("status") not in {"active", "superseded"}:
            continue
        if expected_kind and _clean(fact.get("kind"), 80).casefold() != expected_kind:
            continue
        source = _clean(fact.get("evidence_quote") or fact.get("value_text"), 1000)
        if len(source) < 16:
            continue
        source_normalized = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+", " ", source.casefold(),
        ).strip()
        if source_normalized and (
            source_normalized == normalized
            or (explicit_recall and (
                source_normalized in normalized or normalized in source_normalized
            ))
        ):
            return True
        source_terms = query_terms(source.casefold()) - generic
        overlap = candidate_terms & source_terms
        if len(overlap) < 5:
            continue
        source_coverage = len(overlap) / max(1, len(source_terms))
        candidate_coverage = len(overlap) / max(1, len(candidate_terms))
        if explicit_recall and source_coverage >= 0.8:
            return True
        if source_coverage >= 0.8 and candidate_coverage >= 0.8:
            return True
    return False


def wants_persona_history(query: str) -> bool:
    text = _clean(query, 500).casefold()
    return bool(re.search(r"\b(?:used to|before|changed|change|growth|how have you become)\b|以前|过去|变化|怎么变|成长|后来", text))


def exact_persona_detail_recall_reply(query: str, facts: list[dict]) -> str:
    """Return only Alice's own stored words for an explicit, topic-linked detail ask.

    Lightweight recall stays with the normal conversational generator. This
    narrow path is for a user who names Alice/you and asks to elaborate on an
    earlier personal disclosure. The source sentence is the reply: the model
    cannot turn a retrieved memory into plausible but unsupported biography.
    """
    text = _clean(query, 500)
    lowered = text.casefold()
    asks_which_part = bool(re.search(
        r"\b(?:what|which) part\b.{0,48}\b(?:hardest|most difficult|roughest)\b|"
        r"\bwhat was (?:the )?(?:hardest|most difficult|roughest) part\b|"
        r"(?:哪一部分|哪部分).{0,20}(?:最难|最辛苦|最难熬)",
        lowered,
    ))
    asks_for_detail = bool(re.search(
        r"\b(?:tell me more|elaborate|go into more detail|what was it like|"
        r"what did it (?:look|feel|sound|smell|taste) like|why (?:do|did) you|"
        r"how did you|what happened|what else do you remember|"
        r"what part (?:of that (?:was )?|was )?(?:the )?(?:hardest|most difficult|roughest)|"
        r"what was (?:the )?(?:hardest|most difficult|roughest) part|"
        r"which part was (?:the )?(?:hardest|most difficult|roughest)|"
        r"how did that affect you|what did you mean by that|"
        r"what (?:was|were) (?:in|inside) (?:it|that)|"
        r"what did (?:it|that) (?:include|contain|have)|"
        # A natural follow-up can ask for a named detail without saying
        # "tell me more" (for example, "What was that little morning ritual
        # you mentioned?"). Treat it as a topic-linked recall so the selected
        # source fact, rather than a plausible new biography, supplies the answer.
        r"what (?:was|were) (?:that|the)\b[^?!.]{0,96}\b(?:you|alice)\s+"
        r"(?:just\s+)?(?:mentioned|said|told me about)\b)\b|"
        r"(?:具体|详细|展开|为什么|怎么回事|是什么样|当时发生了什么)",
        lowered,
    ))
    asks_for_one_example = bool(re.search(
        r"\b(?:what|which)\b.{0,48}\b(?:one of them|one of those|one of the things|"
        r"one of the objects|an example)\b|"
        r"\bwhat was one\b.{0,24}\b(?:them|those|things|objects)\b|"
        r"\b(?:do|did)\s+you\s+(?:still\s+)?(?:have|keep)\b.{0,120}\b(?:you|alice)\s+(?:mentioned|said|told me about)\b|"
        r"(?:其中|里面|里头).{0,16}(?:一个|哪件|什么东西)|举个例子",
        lowered,
    ))
    asks_for_contents = bool(re.search(
        r"\bwhat (?:was|were) (?:in|inside) (?:it|that)\b|"
        r"\bwhat did (?:it|that) (?:include|contain|have)\b|"
        r"(?:里面|其中|里头).{0,12}(?:有什么|是什么)",
        lowered,
    ))
    addresses_alice = bool(re.search(
        r"\b(?:you|your|alice)\b|你|爱丽丝|艾丽丝", lowered,
    ))
    if not (asks_for_detail or asks_for_one_example) or not addresses_alice:
        return ""

    query_tokens = query_terms(text)
    if not query_tokens:
        return ""
    selected = rank_facts(facts, text, limit=6)
    if asks_for_contents:
        # A contents question should first answer from the directly named item,
        # rather than concatenating adjacent but non-answering life details.
        # Keep Alice's stored wording and only normalize sentence boundaries.
        for fact in selected:
            if (
                fact.get("source_kind") != "assistant_self_disclosure"
                or _clean(fact.get("kind"), 80) != "preference"
            ):
                continue
            value = _clean(fact.get("value_text") or fact.get("value"), 1000)
            if not (query_tokens & query_terms(value)):
                continue
            quote = _clean(fact.get("evidence_quote") or value, 500)
            if not quote:
                continue
            quote = quote[0].upper() + quote[1:]
            quote = re.sub(r"[,;:\s]+$", "", quote)
            if not re.search(r"[.!?]$", quote):
                quote += "."
            # The wording of the question sets the answer scope. Even if the
            # same disclosure also contains an emotional backstory, a direct
            # contents question should return the requested item only.
            return quote
    if asks_for_one_example:
        # A natural back-reference can ask for one remembered example without
        # saying "elaborate" (for example, "You mentioned keeping little
        # things that carry memories. What was one of them?"). Return one
        # source disclosure, not a model-invented substitute or a list of all
        # nearby Alice memories.
        for fact in selected:
            if (
                fact.get("source_kind") != "assistant_self_disclosure"
                or _clean(fact.get("kind"), 80) not in _PERSONA_SELF_FACT_KINDS
            ):
                continue
            quote = _clean(fact.get("evidence_quote") or fact.get("value_text"), 500)
            if not quote:
                continue
            if not re.search(r"[.!?]$", quote):
                quote += "."
            return quote
    quotes: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for fact in selected:
        if fact.get("source_kind") != "assistant_self_disclosure":
            continue
        if _clean(fact.get("kind"), 80) not in _PERSONA_SELF_FACT_KINDS:
            continue
        searchable = (
            _clean(fact.get("value_text") or fact.get("value"), 1000)
            + " " + _clean(fact.get("facet_key"), 180)
        )
        overlap = query_tokens & query_terms(searchable)
        # Require a real topic match, not generic words such as "why" or
        # "like". Two terms handle short English topics; one distinctive
        # topic token is enough when a separate follow-up refers back to that
        # already named topic (e.g. "the soup ... what was in it?").
        distinctive = any(
            len(term) >= 4 and re.search(r"[a-z]", term)
            or re.search(r"[\u4e00-\u9fff]", term)
            for term in overlap
        )
        if len(overlap) < 2 and not distinctive:
            continue
        quote = _clean(fact.get("evidence_quote") or fact.get("value_text"), 500)
        key = quote.casefold()
        if not quote or key in seen or "\n" in quote:
            continue
        if total_chars + len(quote) > 900:
            continue
        seen.add(key)
        quotes.append(quote)
        total_chars += len(quote)
        if len(quotes) >= 2:
            break
    if not quotes:
        return ""
    if asks_which_part:
        evidence = " ".join(quotes)
        has_explicit_part_answer = bool(re.search(
            r"\b(?:hardest|most difficult|roughest) part\b.{0,60}\b(?:was|is|felt|came from)\b|"
            r"\b(?:the part|what was hardest)\b.{0,60}\b(?:was|is|felt|came from)\b",
            evidence.casefold(),
        ))
        if not has_explicit_part_answer:
            if re.search(r"[\u4e00-\u9fff]", text):
                return f"我当时只说过：“{quotes[0]}”，没有说清具体哪一部分最难熬。"
            return f"I only said, “{quotes[0]}” — I didn't say which part was hardest."
    return " ".join(quotes)


def persona_recall_draft_status(query: str, draft: str, facts: list[dict]) -> str:
    """Classify an existing Director draft before the sourced quote fallback.

    The Director already receives the same bounded Alice evidence packet. A
    relevant, grounded draft should remain the spoken answer; replacing it with
    the stored sentence makes memory recall sound like a transcript replay.
    Exact/near-exact copying is routed through the existing response-generation
    fallback so Alice can phrase the sourced detail naturally. If the draft is
    unsupported, callers may still use ``exact_persona_detail_recall_reply``
    as the conservative evidence-only fallback.
    """
    if not exact_persona_detail_recall_reply(query, facts):
        return ""
    candidate = _clean(draft, 1200)
    if not candidate:
        return "ungrounded"
    candidate_norm = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+", " ", candidate.casefold(),
    ).strip()
    candidate_tokens = re.findall(
        r"[a-z0-9]+|[\u4e00-\u9fff]+", candidate.casefold(),
    )
    candidate_terms = query_terms(candidate)
    if not candidate_terms:
        return "ungrounded"

    for fact in rank_facts(facts, query, limit=6):
        if (fact.get("source_kind") != "assistant_self_disclosure"
                or _clean(fact.get("kind"), 80) not in _PERSONA_SELF_FACT_KINDS):
            continue
        source = _clean(fact.get("evidence_quote") or fact.get("value_text"), 800)
        if not source:
            continue
        source_norm = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+", " ", source.casefold(),
        ).strip()
        source_tokens = re.findall(
            r"[a-z0-9]+|[\u4e00-\u9fff]+", source.casefold(),
        )
        if not source_tokens:
            continue

        # A quoted source sentence, even with a short conversational preface,
        # is still transcript replay unless the user asked for exact wording.
        if source_norm and source_norm in candidate_norm:
            return "verbatim_echo"
        if len(source_tokens) >= 5 and SequenceMatcher(
            None, source_tokens, candidate_tokens,
        ).ratio() >= 0.84:
            return "verbatim_echo"

        source_terms = query_terms(source)
        # A reply can quote/paraphrase the requested detail correctly and then
        # attach a separate, unsupported autobiographical claim. Check each
        # clause so one grounded sentence cannot make an invented add-on look
        # sourced merely because the whole response has enough token overlap.
        for clause in re.split(r"[.!?;；\n]+|[—–]", candidate):
            clause_terms = query_terms(clause)
            if len(clause_terms) >= 3 and not (clause_terms & source_terms):
                return "ungrounded"
        overlap = candidate_terms & source_terms
        required = 1 if min(len(candidate_terms), len(source_terms)) <= 2 else 2
        if (len(overlap) >= required
                and len(overlap) / max(1, len(candidate_terms)) >= 0.22):
            return "grounded"
    return "ungrounded"


def has_explicit_causal_reflection(text: str) -> bool:
    return bool(re.search(
        r"\b(?:because|made me|taught me|changed (?:the way|how))\b|"
        r"\b(?:you|our conversations|talking with you)\b.{0,80}\b(?:helped|taught|made) me\b|"
        r"因为.{0,40}(?:让我|使我|我变得)|"
        r"(?:你|和你聊天|跟你聊天|我们聊这些).{0,24}(?:让我|教会我|使我).{0,16}(?:变得|开始|学会|意识到|改变)|"
        r"让我变得|改变了我|以前.{0,12}现在",
        _clean(text, 500), re.I,
    ))


def rank_facts(
    facts: list[dict], query: str, limit: int = MAX_PACKET_FACTS,
    include_history: bool = False,
) -> list[dict]:
    terms = query_terms(query)
    scored = []
    seen_values: set[tuple[str, str, str, bool]] = set()
    for fact in facts or []:
        if not isinstance(fact, dict) or fact.get("hidden_at"):
            continue
        if str(fact.get("facet_key") or "").casefold() == "impression_of_user":
            continue
        is_history = fact.get("status") == "superseded" or bool(fact.get("valid_to"))
        if fact.get("status") != "active" and not (include_history and is_history):
            continue
        if is_history and not include_history:
            continue
        value = _clean(fact.get("value_text") or fact.get("value"), 1000)
        dedupe_key = (
            _clean(fact.get("kind"), 80),
            _clean(fact.get("facet_key"), 180).casefold(),
            value.casefold(),
            is_history,
        )
        if dedupe_key in seen_values:
            continue
        seen_values.add(dedupe_key)
        searchable = (value + " " + _clean(fact.get("facet_key"), 180)).casefold()
        fact_terms = query_terms(searchable)
        overlap = len(terms & fact_terms)
        if terms and overlap == 0:
            continue
        kind = _clean(fact.get("kind"), 80)
        score = overlap * 10
        if kind == "identity" and not is_history:
            score += 3
        if fact.get("source_kind") == "assistant_self_disclosure":
            score += 1
        if is_history:
            score += 2
        scored.append((score, _clean(fact.get("recorded_at") or fact.get("observed_at"), 80), fact))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [dict(row[2]) for row in scored[:max(0, limit)]]


def _supported_growth_groups(facts: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Return self-fact groups with enough evidence to shape lasting persona.

    A stored self-disclosure is still recallable as something Alice said. It
    becomes default behavioral context only after it recurs in distinct
    conversations, or Alice explicitly describes why/how she changed. Grouping
    by kind and facet prevents unrelated facts from jointly creating a trait.
    Superseded versions remain useful evidence for a supported trajectory.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for fact in facts or []:
        if not isinstance(fact, dict) or fact.get("hidden_at"):
            continue
        if fact.get("status") not in {"active", "superseded"}:
            continue
        kind = _clean(fact.get("kind"), 80)
        facet = _clean(fact.get("facet_key"), 180).casefold()
        if facet == "impression_of_user":
            continue
        if kind not in _GROWTH_KINDS or not facet:
            continue
        grouped.setdefault((kind, facet), []).append(fact)

    supported: dict[tuple[str, str], list[dict]] = {}
    for key, rows in grouped.items():
        source_conversations = {
            _clean(row.get("source_conv_id"), 160)
            for row in rows if _clean(row.get("source_conv_id"), 160)
        }
        has_causal_reflection = any(
            has_explicit_causal_reflection(
                _clean(row.get("evidence_quote") or row.get("value_text"), 500)
            )
            for row in rows
        )
        # 第三种证据：同一倾向被独立观察到多次（support_count 由合并累加）。
        # 一条话出现一次不算人格；同一维度上反复出现、且内容相近，才算。
        repeated = max((int(row.get("support_count") or 1) for row in rows), default=1) >= 2
        if len(source_conversations) >= 2 or has_causal_reflection or repeated:
            supported[key] = rows
    return supported


def supported_growth_fact_ids(facts: list[dict]) -> set[str]:
    return {
        str(row.get("fact_id") or "")
        for rows in _supported_growth_groups(facts).values()
        for row in rows if row.get("fact_id")
    }


def _topic_label_hit(query_tokens: set[str], facet_key) -> bool:
    """True when the user's own wording points at one of Alice's topic labels.

    Used only to *order* the broad packet, never to drop anything. Allows simple
    word-form drift (walk/walks, evening/evenings, rain/raining) so a natural
    follow-up still lands on the stored `facet_key`; a miss costs nothing.
    """
    for facet_term in query_terms(_clean(facet_key, 180)):
        for term in query_tokens:
            if term == facet_term:
                return True
            if len(term) >= 4 and len(facet_term) >= 4 and term[:4] == facet_term[:4]:
                return True
    return False


def build_persona_packet(
    facts: list[dict], query: str = "", snapshot: dict | None = None,
    *, broad: bool | None = None, seed_offset: int = 0, seed_window: int = 3,
    max_facts: int | None = None,
) -> dict:
    """Render one small evidence packet shared by Director and Control."""
    limit = max(1, int(max_facts or MAX_PACKET_FACTS))
    include_history = wants_persona_history(query)
    broad = (wants_persona_recall(query) or include_history) if broad is None else bool(broad)
    active = [f for f in facts or [] if isinstance(f, dict)
              and str(f.get("facet_key") or "").casefold() != "impression_of_user"
              and f.get("status") in ({"active", "superseded"} if include_history else {"active"})
              and not f.get("hidden_at")
              and (include_history or not f.get("valid_to"))]
    identities_by_facet: dict[str, dict] = {}
    for fact in active:
        if fact.get("kind") != "identity":
            continue
        facet = _clean(fact.get("facet_key"), 180).casefold()
        prior = identities_by_facet.get(facet)
        if prior is None or _clean(fact.get("recorded_at"), 80) < _clean(prior.get("recorded_at"), 80):
            identities_by_facet[facet] = fact
    identities = list(identities_by_facet.values())
    # The earliest schema stored explicit school/work-stage statements under
    # the broader `experience` kind. Recognize those old rows too, and keep
    # only the newest active statement for each mutable status facet.
    current_statuses: dict[str, dict] = {}
    for fact in active:
        facet = current_status_facet(fact)
        if not facet:
            continue
        prior = current_statuses.get(facet)
        fact_time = _clean(fact.get("observed_at") or fact.get("recorded_at"), 80)
        prior_time = _clean((prior or {}).get("observed_at") or (prior or {}).get("recorded_at"), 80)
        if prior is None or fact_time >= prior_time:
            current_statuses[facet] = fact
    current_status_rows = list(current_statuses.values())
    if not include_history:
        # Old schemas stored life stages as experiences. Keep their source rows,
        # but only the latest semantic state may enter an ordinary turn packet.
        # Filter before ranking/limits so an obsolete state does not consume space.
        active = [fact for fact in active
                  if not current_status_facet(fact)
                  or fact == current_statuses[current_status_facet(fact)]]
    if query and not broad:
        selected = rank_facts(active, query, limit, include_history)
        # 2026-10-14 修复（读侧"存了用不上"的真机制）：上面的 rank_facts 在
        # `terms and overlap == 0` 时会把**整包**事实丢掉（见本文件 :1088）。
        # 而门 off 的普通闲聊轮走的正是这条分支 ⇒ 用户问"你中午吃的啥 /
        # Where are you yourself?"这类与她自述**零词面重叠**的问题时，selected
        # 为空，只剩 identity + current_status 两条强制行 ⇒ 她拿不到自己说过的
        # 话、只能当场编。本次真人对话 18 句里 17 句命中此路径。
        # 改法＝把"排序"与"清空"分开：query 命中仍排在最前（纯命中轮结果逐字
        # 不变），但用全量顺序**补齐到同一上限**，覆盖面与 broad 路径一致 ——
        # 这也是本模块 broad 分支自己写下的原则（"仍取满 MAX_PACKET_FACTS，
        # 不缩小覆盖面"）。上限仍是 MAX_PACKET_FACTS / MAX_PACKET_CHARS，不放大注入。
        if len(selected) < limit:
            seen_ids = {str(row.get("fact_id") or "") for row in selected}
            for row in rank_facts(active, "", limit, include_history):
                if len(selected) >= limit:
                    break
                fact_id = str(row.get("fact_id") or "")
                if fact_id and fact_id in seen_ids:
                    continue
                selected.append(row)
                seen_ids.add(fact_id)
    elif broad:
        # 10-12R13：broad 的语义是"不再按 query 收敛"，但**不等于丢弃 query 信号**。
        # 实测（一个真实账号，14 条 fact）：`You mentioned taking a walk. When do
        # you usually go for a walk?` / `It rained here last night. Do you like rainy
        # evenings?` / `Tell me about your daily routine.` 这类**跨轮话题式追问**命中
        # 门（R6~R12 放宽所致）⇒ 切 broad ⇒ query 被清空 ⇒ 那条最该召回的
        # `daily-walks` / `rainy-evening-weather` 事实被挤出前 6 ⇒ 提问者反而看不到
        # 自己刚提到的那件事（违反"存了必须用得上"）。
        # 改法：仍取满 MAX_PACKET_FACTS（**不缩小覆盖面**），只把"query 词命中**她自己的
        # 话题标签 facet_key**"的事实排到最前 ⇒ 话题事实保证入选，其余按原广度顺序补齐。
        # 话题锚点只认 facet_key（不是 value 的任意词面重叠），避免 `been`/`like`/`about`
        # 这类虚词造成无意义重排 ⇒ 非话题问句结果与改前**逐字一致**。
        broad_rows = rank_facts(active, "", limit, include_history)
        if query:
            q_terms = query_terms(query)
            topic_rows = [
                row for row in rank_facts(active, query, limit, include_history)
                if _topic_label_hit(q_terms, row.get("facet_key"))
            ]
            seen_ids = {str(row.get("fact_id") or "") for row in topic_rows}
            selected = (topic_rows + [
                row for row in broad_rows
                if str(row.get("fact_id") or "") not in seen_ids
            ])[:limit]
        else:
            selected = broad_rows
    else:
        supported_ids = supported_growth_fact_ids(facts)
        current_status_ids = {
            str(f.get("fact_id") or "") for f in current_status_rows
        }
        durable = [
            f for f in active
            if f.get("kind") == "identity"
            or (current_status_facet(f)
                and str(f.get("fact_id") or "") in current_status_ids)
            or (f.get("kind") in _GROWTH_KINDS
                and str(f.get("fact_id") or "") in supported_ids)
        ]
        selected = rank_facts(durable, "", limit)
        # 开场兜底（plans/1017-02 §8.31，默认关）：
        # 实测主用户 39 条自述（偏好/性格/能力/语言环境/经历）里**没有一条**是
        # identity 或 current_status ⇒ `durable` 为空 ⇒ **开场渲染 0 行**，
        # 她每盘都从白纸开始，自述要等用户问到才出现（§8.31 三处接缝之①）。
        # 开：durable 为空时，按原顺序取前 MAX_PACKET_FACTS 条 active 自述。
        if not selected:
            try:
                from config import PERSONA_OPENING_RECALL_ENABLED as _opening_on
            except Exception:
                _opening_on = False
            if _opening_on:
                selected = rank_facts(active, "", limit)

    lines: list[str] = []
    used: set[str] = set()
    for fact in [*identities, *current_status_rows]:
        if fact not in selected:
            selected.insert(0, fact)
    snapshot_text = _clean((snapshot or {}).get("narrative"), 600)
    snapshot_ids = [str(value) for value in _list((snapshot or {}).get("supporting_fact_ids"))]
    derived_view_ids = {
        str(f.get("fact_id") or "") for f in facts or [] if isinstance(f, dict)
        and str(f.get("facet_key") or "").casefold() == "impression_of_user"
    }
    cites_user_view = any(
        value in derived_view_ids or value == "impression_of_user_v1"
        or value.startswith("impression_of_user_v2:") for value in snapshot_ids
    )
    # Snapshot save and source-hide/purge methods validate and invalidate these
    # citations in L2. A valid trajectory may cite a superseded version, which
    # is intentionally absent from the session's current-facts cache.
    if (snapshot_text and snapshot_ids
            and not cites_user_view
            and _clean((snapshot or {}).get("snapshot_version"), 80) == PERSONA_SNAPSHOT_VERSION
            and not query):
        lines.append("Alice's sourced self-reflection: " + snapshot_text)
        used.update(snapshot_ids)

    for fact in selected:
        if len(lines) >= limit:
            break
        fact_id = str(fact.get("fact_id") or "")
        if fact_id in used:
            continue
        value = _clean(fact.get("value_text") or fact.get("value"), 360)
        if not value:
            continue
        kind = _clean(fact.get("kind"), 80)
        if fact.get("status") == "superseded" or fact.get("valid_to"):
            text = "Earlier Alice self-background: " + value
        elif kind == "identity":
            text = "Established Alice self-background: " + value
        elif current_status_facet(fact):
            latest = current_statuses[current_status_facet(fact)]
            prefix = "Alice's current stated status: " if fact == latest else "Earlier Alice self-background: "
            text = prefix + value
        elif fact.get("source_kind") == "assistant_self_disclosure":
            text = "Alice's exact recorded self-disclosure: " + value
        elif kind in {"trait", "preference", "ability", "experience", "language_environment"}:
            if fact.get("source_kind") == "reading":
                # 阅读后产生的想法，不是亲口对用户说的——不能说成"Alice previously said"（背稿）。
                text = "Alice's take from something she read: " + value
            else:
                text = "Alice previously said: " + value
        else:
            text = "Alice's supported personal context: " + value
        source = _clean(fact.get("source_kind") or "unknown", 60)
        when = _clean(fact.get("observed_at") or fact.get("recorded_at"), 40)
        # 2026-10-22（plans/1022-01 点 5）：后缀从 122 字符压到 ~12。
        # 实测真实池 6 条 1532 字符里 732（47.8%）是这个后缀，而全仓没有任何代码
        # 解析它；前缀已经说明来源（"exact recorded self-disclosure" 等），
        # fact_id 也不进 prompt 的读侧。只保留日期，给模型判断新旧用。
        # 收益：同样 1600 预算，10 条料 1419 字符（原 6 条就要 1532）。
        _when_short = when[:10] if when else ""
        if _when_short:
            text += f" [{_when_short}]"
        lines.append(text)
        used.add(fact_id)

    # ── 作者种子（plans/1017-02 §8.35，默认关）────────────────────────
    # 591 条回复只有 13% 在讲她自己、可核验自述只剩 4 句 ⇒ 对话生不出"她的生活"。
    # 种子**只进 prompt、不写库**；只要她自己 active 自述覆盖同一 facet，种子让位。
    try:
        from config import PERSONA_SEED_ENABLED as _seed_on
    except Exception:
        _seed_on = False
    if _seed_on:
        try:
            from memory.persona_seed import PERSONA_SEED_FACTS as _seed_facts
        except Exception:
            _seed_facts = []
        covered = {
            _clean(fact.get("facet_key"), 180).casefold()
            for fact in active if fact.get("status") == "active"
        }
        _seed_life: list[str] = []
        _seed_manner: list[str] = []
        for item in _seed_facts:
            facet = _clean(item.get("facet_key"), 180).casefold()
            if facet and facet in covered:
                continue
            # ⛔ 只喂英文版：中文原文会被她整句照抄（实测「雨天最适合待在家里，
            # 什么都不干也不心虚」原样出现在英文回复里，违反 D011）。
            _seed_line = _clean(item.get("value_en") or "", 220)
            if not _seed_line:
                continue
            if str(item.get("role") or "life") == "manner":
                _seed_manner.append(_seed_line)  # 见下：语气不进包，改走 prompt 结尾
            else:
                _seed_life.append(_seed_line)
        # 语气（manner）**不进这个包**：包是"证据/素材"，上限 6 行 1600 字符，
        # 4 行语气会把她的真实事实挤出去（实测新用户 life 种子被挤掉 1 条）。
        # 语气改放 prompt 结尾的固定段（`chat_tape_policy.prompt_tail`）。
        # 素材（life）：轮换 + 覆盖（§8.57）。
        # 轮换——12 条全塞进去她会在同一盘里反复说同一件（实测雨天 3/12）；
        # 覆盖——她自己的经历攒够了（真实主用户 29 条 active）就用她的，不用我编的。
        _story_kinds = {"preference", "habit", "experience", "ability"}
        _learned = sum(
            1 for fact in active
            if str(fact.get("kind") or "") in _story_kinds
            and str(fact.get("facet_key") or "").casefold() not in {
                _clean(item.get("facet_key"), 180).casefold() for item in _seed_facts
            }
        )
        _seed_window = 3 if _learned < 3 else (1 if _learned < 6 else 0)
        if _seed_life and _seed_window:
            _start = int(seed_offset or 0) % len(_seed_life)
            _window = (_seed_life[_start:] + _seed_life[:_start])[:min(
                max(1, int(seed_window or 3)), _seed_window)]
            for _line in _window:
                lines.append("Alice's own way (authored scaffolding; when you have your own "
                             "real material from earlier conversations, prefer that; say this "
                             "in your own English words, never quote): " + _line)

    clipped: list[str] = []
    total = 0
    for line in lines:
        if len(clipped) >= limit:
            break
        remaining = MAX_PACKET_CHARS - total
        if remaining <= 0:
            break
        if len(line) > remaining:
            # 2026-10-22（plans/1022-01 点 5）：装不下就跳过整行，**不再切半句**。
            # 旧行为把最后一行截到预算边，模型会读到半条自述（"我上周去了…" 戛然而止），
            # 比不注入更糟。跳过则保证每行进 prompt 的料都是完整的。
            continue
        clipped.append(line)
        total += len(line)
    return {
        "persona_key": PERSONA_KEY,
        "snapshot_version": _clean((snapshot or {}).get("snapshot_version"), 80),
        "fact_ids": sorted(used),
        "lines": clipped,
        "text": " ".join(clipped),
    }


def cold_pass_eligible(facts: list[dict]) -> bool:
    """Require support for one coherent personal facet, not unrelated facts."""
    return bool(_supported_growth_groups(facts))


def arbitrate_version_change(
    incoming: dict, active_versions: list[dict], same_value_sources: list[str],
) -> dict:
    """Return a deterministic append-only transition for an evolving facet.

    A single contrary statement remains a candidate. A second observation from
    a different conversation, or an explicit reflection about the change,
    closes the previous version and promotes the new one. Stable identity slots
    are intentionally outside this arbitration and remain immutable here.
    """
    fact = dict(incoming or {})
    kind = _clean(fact.get("kind"), 80)
    facet = _clean(fact.get("facet_key"), 180)
    value = _clean(fact.get("value_text") or fact.get("value"), 1000)
    if kind not in {"trait", "preference", "ability", "language_environment", "current_status"} or not facet or not value:
        return {"status": str(fact.get("status") or "active"), "supersede_ids": [], "promote_ids": []}
    conflicting_active = [row for row in active_versions or []
                          if row.get("status") == "active"
                          and _clean(row.get("value_text"), 1000).casefold() != value.casefold()]
    if not conflicting_active:
        # §8.41：没有冲突时也要能升格——docstring 写的就是"第二次、不同会话出现即升"，
        # 但原实现只在有冲突时走仲裁，candidate 同值重复会永远停在 candidate。
        same_value_rows = [
            row for row in (active_versions or [])
            if _clean(row.get("value_text"), 1000).casefold() == value.casefold()
        ]
        other_candidate_ids = [
            str(row.get("fact_id")) for row in (active_versions or [])
            if row.get("status") == "candidate"
            and row.get("fact_id")
            and _clean(row.get("value_text"), 1000).casefold() != value.casefold()
        ]
        incoming_conv = _clean(fact.get("source_conv_id"), 160)
        independent = {str(source) for source in same_value_sources or [] if str(source)}
        if incoming_conv:
            independent.add(incoming_conv)
        if len(independent) >= 2:
            change_time = _clean(fact.get("observed_at") or fact.get("recorded_at"), 80)
            return {
                "status": "active",
                "promote_ids": [str(row.get("fact_id")) for row in same_value_rows
                                if row.get("status") == "candidate" and row.get("fact_id")],
                "supersede_candidate_ids": other_candidate_ids,
                "change_time": change_time,
                "support_count": len(independent),
            }
        return {"status": str(fact.get("status") or "active"), "supersede_ids": [],
                "promote_ids": [], "support_count": max(1, len(independent))}
    if kind == "current_status":
        # A current education/work stage is a mutable fact, not a personality
        # interpretation. A later explicit self-disclosure may update it once;
        # retain the former value as history instead of requiring trait-style
        # repetition before Alice can remember that her life stage changed.
        incoming_time = _clean(fact.get("observed_at") or fact.get("recorded_at"), 80)
        prior_times = [
            _clean(row.get("observed_at") or row.get("valid_from"), 80)
            for row in conflicting_active
        ]
        if incoming_time and all(prior and incoming_time > prior for prior in prior_times):
            supersede_ids = [str(row.get("fact_id")) for row in conflicting_active if row.get("fact_id")]
            return {
                "status": "active",
                "valid_from": _clean(fact.get("valid_from"), 80) or incoming_time,
                "supersedes": ",".join(supersede_ids)[:180],
                "supersede_ids": supersede_ids,
                "promote_ids": [],
                "change_time": incoming_time,
            }
        return {"status": "candidate", "supersede_ids": [], "promote_ids": []}
    incoming_conv = _clean(fact.get("source_conv_id"), 160)
    independent = {str(source) for source in same_value_sources or [] if str(source)}
    if incoming_conv:
        independent.add(incoming_conv)
    causal = has_explicit_causal_reflection(
        _clean(fact.get("evidence_quote") or value, 500)
    )
    if len(independent) >= 2 or causal:
        change_time = _clean(fact.get("observed_at") or fact.get("recorded_at"), 80)
        other_candidate_ids = [
            str(row.get("fact_id")) for row in (active_versions or [])
            if row.get("status") == "candidate"
            and row.get("fact_id")
            and _clean(row.get("value_text"), 1000).casefold() != value.casefold()
        ]
        return {
            "status": "active",
            "valid_from": _clean(fact.get("valid_from"), 80) or change_time,
            "supersedes": ",".join(str(row.get("fact_id") or "") for row in conflicting_active if row.get("fact_id"))[:180],
            "supersede_ids": [str(row.get("fact_id")) for row in conflicting_active if row.get("fact_id")],
            "promote_ids": [str(row.get("fact_id")) for row in (active_versions or [])
                            if str(row.get("value_text") or "").casefold() == value.casefold()
                            and row.get("status") == "candidate" and row.get("fact_id")],
            "supersede_candidate_ids": other_candidate_ids,
            "change_time": change_time,
            "support_count": max(2, len(independent)),
        }
    return {"status": "candidate", "supersede_ids": [], "promote_ids": [], "support_count": max(1, len(independent))}


def merge_shared_and_user_persona_facts(
    shared_facts: list[dict] | None, user_facts: list[dict] | None,
) -> list[dict]:
    """Overlay user-specific Alice growth on top of shared '通用爱丽丝' base persona.

    User-conversation facts (`user_facts`) represent how Alice evolved specifically
    with this user ('决定基因的'), so they take precedence on any overlapping facet_key.
    """
    own = [dict(f) for f in (user_facts or []) if isinstance(f, dict)]
    base = [dict(f) for f in (shared_facts or []) if isinstance(f, dict)]
    if not base:
        return own
    if not own:
        return base
    covered_facets = {
        _clean(f.get("facet_key"), 180).casefold()
        for f in own if _clean(f.get("facet_key"), 180)
    }
    covered_values = {
        _clean(f.get("value_text") or f.get("value"), 1000).casefold()
        for f in own if _clean(f.get("value_text") or f.get("value"), 1000)
    }
    merged = list(own)
    for item in base:
        fk = _clean(item.get("facet_key"), 180).casefold()
        val = _clean(item.get("value_text") or item.get("value"), 1000).casefold()
        if (fk and fk in covered_facets) or (val and val in covered_values):
            continue
        merged.append(item)
    return merged


def build_cold_pass_payload(facts: list[dict], limit: int = 48) -> list[dict]:
    supported = _supported_growth_groups(facts)
    rows = [fact for group in supported.values() for fact in group]
    # Keep the newest bounded evidence, then present it chronologically so the
    # cold pass can distinguish a sequence from a pile of unordered claims.
    rows.sort(key=_fact_time_key, reverse=True)
    rows = rows[:limit]
    rows.sort(key=_fact_time_key)
    return [{
        "fact_id": _clean(f.get("fact_id"), 180),
        "kind": _clean(f.get("kind"), 80),
        "facet_key": _clean(f.get("facet_key"), 120),
        "value_text": _clean(f.get("value_text"), 360),
        "source_conv_id": _clean(f.get("source_conv_id"), 160),
        "observed_at": _clean(f.get("observed_at"), 80),
        "source_kind": _clean(f.get("source_kind"), 80),
        "confidence": _float(f.get("confidence"), 0.0),
        "evidence_quote": _clean(f.get("evidence_quote"), 300),
        "interaction_context": _clean(f.get("interaction_context"), 500),
        "interaction_context_source": _clean(f.get("interaction_context_source"), 40),
        "status": _clean(f.get("status"), 24),
        "valid_from": _clean(f.get("valid_from"), 80),
        "valid_to": _clean(f.get("valid_to"), 80),
    } for f in rows if f.get("fact_id") and f.get("value_text")]


def _reflection_grounded_in_self_quotes(text: str, rows: list[dict]) -> bool:
    """Keep user context from becoming proof of Alice's biography or growth."""
    candidate_terms = query_terms(text) - {
        "alice", "feel", "felt", "think", "thought", "said", "says",
        "become", "became", "gradually", "lately", "before", "now",
        "because", "made", "helped", "taught", "changed", "change",
    }
    source_terms: set[str] = set()
    for row in rows:
        # Deliberately exclude interaction_context: it identifies the topic
        # discussed, but is not evidence about Alice's own experience.
        quote = _clean(row.get("evidence_quote") or row.get("value_text"), 500)
        source_terms.update(query_terms(quote))
    overlap = candidate_terms & source_terms
    required = 1 if len(candidate_terms) <= 2 else 2
    if len(overlap) < required:
        return False
    for clause in re.split(r"[.!?;；\n]+|[—–]", text):
        clause_terms = query_terms(clause)
        if len(clause_terms) >= 3 and not (clause_terms & source_terms):
            return False
    return True


def validate_snapshot(raw: Any, facts: list[dict]) -> dict:
    if not isinstance(raw, dict):
        return {}
    eligible_rows = build_cold_pass_payload(facts)
    allowed = {str(f.get("fact_id")) for f in eligible_rows if f.get("fact_id")}
    fact_by_id = {
        str(f.get("fact_id")): f for f in facts
        if isinstance(f, dict) and str(f.get("fact_id") or "") in allowed
    }

    def validate_reflection(text: Any, raw_ids: Any, kind: Any = "", facet: Any = "") -> tuple[str, list[str]]:
        sentence = _clean(text, 180)
        ids = list(dict.fromkeys(
            str(value) for value in _list(raw_ids)
            if str(value) in allowed
        ))
        if not sentence or not ids or len(ids) > 6:
            return "", []
        support_rows = [fact_by_id[value] for value in ids if value in fact_by_id]
        if not _reflection_grounded_in_self_quotes(sentence, support_rows):
            return "", []
        support_groups = {
            (_clean(row.get("kind"), 80), _clean(row.get("facet_key"), 180).casefold())
            for row in support_rows
        }
        if len(support_groups) != 1:
            return "", []
        group_kind, group_facet = next(iter(support_groups))
        if ((kind and _clean(kind, 80) != group_kind)
                or (facet and _clean(facet, 180).casefold() != group_facet)):
            return "", []
        source_sessions = {
            str(row.get("source_conv_id") or "") for row in support_rows
            if row.get("source_conv_id")
        }
        causal = any(has_explicit_causal_reflection(
            _clean(row.get("evidence_quote") or row.get("value_text"), 500)
        ) for row in support_rows)
        if len(source_sessions) < 2 and not causal:
            return "", []
        return sentence, ids

    # The cold pass may summarize several separately supported facets in one
    # compact packet. Each sentence remains independently grounded in exactly
    # one facet; unrelated facts cannot be blended into a made-up causal story.
    reflections = raw.get("reflections")
    if isinstance(reflections, list):
        accepted_text: list[str] = []
        accepted_ids: list[str] = []
        seen_groups: set[tuple[str, str]] = set()
        for item in reflections[:4]:
            if not isinstance(item, dict):
                continue
            sentence, ids = validate_reflection(
                item.get("narrative"), item.get("supporting_fact_ids"),
                item.get("kind"), item.get("facet_key"),
            )
            if not sentence:
                continue
            rows = [fact_by_id[value] for value in ids]
            group = (
                _clean(rows[0].get("kind"), 80),
                _clean(rows[0].get("facet_key"), 180).casefold(),
            )
            if group in seen_groups or len(set(accepted_ids + ids)) > 6:
                continue
            seen_groups.add(group)
            accepted_text.append(sentence)
            accepted_ids.extend(ids)
        narrative = _clean(" ".join(accepted_text), 600)
        ids = list(dict.fromkeys(accepted_ids))
        return ({"narrative": narrative, "supporting_fact_ids": ids}
                if narrative and ids else {})

    # Backward compatibility for already persisted / older cold-pass output.
    sentence, ids = validate_reflection(
        raw.get("narrative"), raw.get("supporting_fact_ids"),
    )
    return {"narrative": sentence, "supporting_fact_ids": ids} if sentence else {}


def snapshot_input_hash(facts: list[dict]) -> str:
    rows = build_cold_pass_payload(facts)
    serialized = "\n".join(
        f"{row['fact_id']}|{row['kind']}|{row['facet_key']}|{row['value_text']}|"
        f"{row['source_conv_id']}|{row['observed_at']}|{row['status']}|"
        f"{row['valid_from']}|{row['valid_to']}|{row['interaction_context_source']}|"
        f"{row['interaction_context']}"
        for row in rows
    )
    # Eligibility changes must also invalidate the unchanged-source cold-pass
    # shortcut, so a retained older snapshot can be rebuilt without new facts.
    return hashlib.sha256((PERSONA_SNAPSHOT_VERSION + "\n" + serialized).encode("utf-8")).hexdigest()
