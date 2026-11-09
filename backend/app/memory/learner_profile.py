"""统一学习者画像（七维）的**确定性核心** —— 纯函数层。

设计依据：``plans/0923-04-记忆系统与用户认知层-方案.md`` §4.2 / §4.3 / §4.5。

定位与边界
----------
本模块只做**确定性计算**：七维与事实键白名单、值消毒、来源锚点、冲突裁决、
半衰期打分、按时点（as_of）选择，以及 SQLite 同形建表 DDL。

- 不读库、不调模型、不引第三方包（只有标准库），因此可以在任何环境直接单测。
- 写入口 / 读入口在后续阶段接入（``memory/l2_postgres.py``、``memory/l2_sqlite.py``、
  ``agent/tutor_agent.py``），本阶段不接线、不改任何现有行为。
- 事实表是供画像查询与 prompt 使用的**权威读模型**；既有五条写入管道与 L0
  仍是它的可追溯来源。同一 ``fact_key`` 只允许一个当前权威值。

与既有代码的关系（不要重复造）
-----------------------------
``fact_key`` 的取值刻意复用 ``memory/relationship_merge.py::SINGULAR_RELATIONSHIP_KEYS``
里已经在用的键名，避免同一件事出现两套词汇。
"""

from __future__ import annotations

import hashlib
import inspect
import re
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DIMENSIONS",
    "FACT_KEY_DIMENSION",
    "SINGULAR_FACT_KEYS",
    "MULTI_VALUE_FACT_KEYS",
    "EVERGREEN_FACT_KEYS",
    "HALF_LIFE_DAYS",
    "PROMPT_PROTECTED_FACT_KEYS",
    "STATUSES",
    "TIME_BASES",
    "SOURCE_RANK",
    "WRITE_LEASE_TTL_HOURS",
    "LEASE_GATE_UNAVAILABLE",
    "LEASE_CHECK_ERROR_OPEN",
    "LEASE_REVOKED_OR_MISSING",
    "lease_tokens",
    "SQLITE_DDL",
    "SQLITE_TABLES",
    "SQLITE_INDEXES",
    "parse_dt",
    "now_utc",
    "sanitize_text",
    "new_fact_subkey",
    "source_anchor_id",
    "source_occurrence_id",
    "normalize_fact",
    "decay_score",
    "resolve_current",
    "as_of",
    "select_for_prompt",
    "is_fact_suppressed",
    "fact_keys_for_relationship_item",
    "detect_as_of_query",
    "render_as_of_line",
    "filter_forgotten_sources",
    "takeover_audit",
    "build_narrative_payload",
    "narratives_from_llm_json",
]

# --------------------------------------------------------------------------
# 七维（方案 §4.2 表格的七行）
# --------------------------------------------------------------------------

DIM_IDENTITY = "identity_relation"      # ① 身份与关系
DIM_LEVEL = "language_level"            # ② 语言水平
DIM_WEAK_MASTERED = "weak_mastered"     # ③ 薄弱项与已掌握
DIM_PREFERENCE = "preference_boundary"  # ④ 偏好与边界
DIM_INTEREST = "interest_topic"         # ⑤ 兴趣与话题
DIM_OPEN_LOOP = "open_loop"             # ⑥ 未了结的事
DIM_MOOD = "mood_relation"              # ⑦ 情绪与关系状态

DIMENSIONS: Tuple[str, ...] = (
    DIM_IDENTITY,
    DIM_LEVEL,
    DIM_WEAK_MASTERED,
    DIM_PREFERENCE,
    DIM_INTEREST,
    DIM_OPEN_LOOP,
    DIM_MOOD,
)

# fact_key -> dimension。键名取自 relationship_merge 现有词汇，不新造同义词。
FACT_KEY_DIMENSION: Dict[str, str] = {
    # ① 身份与关系
    "learner_name": DIM_IDENTITY,
    "tutor_name": DIM_IDENTITY,
    "learner_roleplay_alias": DIM_IDENTITY,
    "tutor_roleplay_alias": DIM_IDENTITY,
    # ② 语言水平
    "cefr_level": DIM_LEVEL,
    "pronunciation_focus": DIM_LEVEL,
    # ③ 薄弱项与已掌握
    "self_reported_weakness": DIM_WEAK_MASTERED,
    "error_pattern": DIM_WEAK_MASTERED,
    "mastered_expression": DIM_WEAK_MASTERED,
    "chunk_in_use": DIM_WEAK_MASTERED,
    # ④ 偏好与边界
    "learning_preference": DIM_PREFERENCE,
    "avoid_pattern": DIM_PREFERENCE,
    "preferred_tone": DIM_PREFERENCE,
    # ⑤ 兴趣与话题
    "engagement_topic": DIM_INTEREST,
    "preferred_topics": DIM_INTEREST,
    # ⑥ 未了结的事
    "open_loop": DIM_OPEN_LOOP,
    "next_session_intent": DIM_OPEN_LOOP,
    "next_session_intent_consumed": DIM_OPEN_LOOP,
    # ⑦ 情绪与关系状态
    "recent_mood": DIM_MOOD,
    "last_topic": DIM_MOOD,
}

# 单值槽位：同一 fact_key 最多一个当前值（沿用 relationship_merge 定义 + cefr）
SINGULAR_FACT_KEYS = frozenset({
    "last_topic", "open_loop", "recent_mood", "preferred_tone",
    "learning_preference", "self_reported_weakness", "next_session_intent",
    "next_session_intent_consumed", "learner_name", "tutor_name", "cefr_level",
})

MULTI_VALUE_FACT_KEYS = frozenset(FACT_KEY_DIMENSION) - SINGULAR_FACT_KEYS

# 常青：不衰减（方案 §4.3「实名、明确边界、已确认长期偏好和有测评支持的 CEFR」）
EVERGREEN_FACT_KEYS = frozenset({
    "learner_name", "tutor_name",
    "avoid_pattern", "learning_preference", "preferred_tone",
    "cefr_level", "mastered_expression", "chunk_in_use",
})

# 半衰期（天）。方案 §4.3 明确点名的四档，其余给默认值。
HALF_LIFE_DAYS: Dict[str, float] = {
    "recent_mood": 3.0,
    "last_topic": 14.0,
    "open_loop": 90.0,
    "engagement_topic": 180.0,
    "preferred_topics": 180.0,
}
DEFAULT_HALF_LIFE_DAYS = 90.0

# 超额 top-k 时**不允许**被挤掉的键（方案 §4.3 末句：实名与用户明确的「不要」）
PROMPT_PROTECTED_FACT_KEYS = frozenset({"learner_name", "tutor_name", "avoid_pattern"})

# 逐轮 prompt 里**已由既有段落渲染**的 fact_key —— 统一画像只补这些之外的
# "真缺席事实"，同键**绝不另加一份**（方案 §4.4）。
# 改这个集合必须同步核 `agent/tutor_agent.py::_build_control_memory_context`
# 的【身份记忆】【角色扮演别名】与「关系记忆」三段。
TUTOR_ALREADY_INJECTED_FACT_KEYS = frozenset({
    "learner_name", "tutor_name",
    "learner_roleplay_alias", "tutor_roleplay_alias",
    "open_loop", "last_topic", "recent_mood", "preferred_tone",
    "learning_preference", "self_reported_weakness", "avoid_pattern",
})

STATUSES = ("candidate", "confirmed", "contested", "legacy_unattributed", "forget")
TIME_BASES = ("explicit", "observed", "unknown")

# 写入令牌的自我清理窗口（小时）。见 §4.4「短生命周期写入令牌」：作业是秒级到
# 分钟级的，24 小时前登记的令牌必然属于已结束（甚至已崩溃）的作业。登记新令牌时
# 顺手清掉，表因此有界，不必额外加一个调度槽。**校验本身不套这个窗口**——
# 否则一个跑超 24 小时的合法作业会被误拒。
WRITE_LEASE_TTL_HOURS = 24

# 闸门结果状态码（**不含原文**，只用于日志与返回，便于做"取回/未使用"归因）。
LEASE_GATE_UNAVAILABLE = "lease_gate_unavailable"   # 后端没有租约方法（老后端）⇒ 放行
LEASE_CHECK_ERROR_OPEN = "lease_check_error_open"   # 校验查询异常 ⇒ 放行（见 §4.5）
LEASE_REVOKED_OR_MISSING = "lease_revoked_or_missing"  # 令牌已被撤销/不存在 ⇒ 拒写

# 来源资格：高者优先。分事实类的更细优先级由调用方在归一化时落到 source_pipeline。
SOURCE_RANK: Dict[str, int] = {
    "user_explicit": 60,   # 用户明确确立（称呼 / 边界 / 更正）
    "assessment": 50,      # 测评、练习实测、复习结果
    # 练习盘五维（plans/0923-06 §4.2.4）：会话末轮外模型对**他真实首讲/重讲**
    # 的判定，属"练习实测"一类，与 assessment 同级。新增键是纯追加：全仓只有
    # `_resolution_key` 读本表，且此前没有任何管道用这个名字 ⇒ 对既有各管道
    # 的排序结果**零影响**（不加这一行就会被当 legacy=0 排在任何实测证据之后）。
    "expression_argument": 50,
    "curation": 30,        # 异步策展（证据门控）
    "per_turn": 25,        # 逐轮关系记忆
    "inferred": 10,        # 启发式推断（兴趣 / 风格）
    "legacy": 0,           # 迁移来的、无法归因的历史行
}

_MAX_QUOTE = 160  # 原话型证据上限（方案 §4.2）

# --------------------------------------------------------------------------
# SQLite 同形 DDL（PG 侧见 db/migrate_learner_profile.sql）
# --------------------------------------------------------------------------

SQLITE_TABLES = (
    "learner_profile_facts",
    "learner_profile_narratives",
    "learner_profile_write_leases",
)
SQLITE_INDEXES = (
    "uq_lp_facts_occurrence",
    "idx_lp_facts_user_key",
    "idx_lp_facts_source_conv",
    "idx_lp_leases_user_status",
    "idx_lp_leases_user_conv",
)

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS learner_profile_facts (
    fact_id               TEXT PRIMARY KEY,
    user_id               TEXT NOT NULL,
    dimension             TEXT,
    fact_key              TEXT NOT NULL,
    fact_subkey           TEXT NOT NULL DEFAULT '',
    value_text            TEXT NOT NULL DEFAULT '',
    source_pipeline       TEXT NOT NULL DEFAULT '',
    source_table          TEXT NOT NULL DEFAULT '',
    source_id             TEXT NOT NULL DEFAULT '',
    source_anchor_id      TEXT NOT NULL DEFAULT '',
    source_occurrence_id  TEXT NOT NULL DEFAULT '',
    extractor_version     TEXT NOT NULL DEFAULT '',
    source_conv_id        TEXT,
    source_session_id     TEXT,
    evidence_quote        TEXT NOT NULL DEFAULT '',
    evidence_refs         TEXT NOT NULL DEFAULT '[]',
    evidence_kind         TEXT NOT NULL DEFAULT 'unknown',
    confidence            REAL,
    observed_at           TEXT,
    valid_from            TEXT,
    valid_to              TEXT,
    recorded_at           TEXT NOT NULL DEFAULT '',
    time_basis            TEXT NOT NULL DEFAULT 'unknown',
    expires_at            TEXT,
    status                TEXT NOT NULL DEFAULT 'candidate',
    CHECK (status IN ('candidate','confirmed','contested','legacy_unattributed','forget')),
    CHECK (dimension IS NULL OR dimension IN (
        'identity_relation','language_level','weak_mastered',
        'preference_boundary','interest_topic','open_loop','mood_relation')),
    CHECK (status = 'forget' OR dimension IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS learner_profile_narratives (
    user_id             TEXT NOT NULL,
    dimension           TEXT NOT NULL,
    text                TEXT NOT NULL DEFAULT '',
    supporting_fact_ids TEXT NOT NULL DEFAULT '[]',
    updated_at          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, dimension)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_lp_facts_occurrence
    ON learner_profile_facts (user_id, fact_key, fact_subkey, source_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_lp_facts_user_key
    ON learner_profile_facts (user_id, fact_key, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_lp_facts_source_conv
    ON learner_profile_facts (source_conv_id);

-- 写入租约（§4.4 阶段 2）：只存控制记录（谁的令牌、属哪盘、是否仍有效、何时登记），
-- 不存原文与画像值。PG 侧同形 DDL 见 db/migrate_learner_profile.sql。
CREATE TABLE IF NOT EXISTS learner_profile_write_leases (
    user_id        TEXT NOT NULL,
    token          TEXT NOT NULL,
    scope_conv_id  TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'active',
    created_at     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, token),
    CHECK (status IN ('active','revoked'))
);

CREATE INDEX IF NOT EXISTS idx_lp_leases_user_status
    ON learner_profile_write_leases (user_id, status);
CREATE INDEX IF NOT EXISTS idx_lp_leases_user_conv
    ON learner_profile_write_leases (user_id, scope_conv_id);
"""


# --------------------------------------------------------------------------
# 时间与消毒
# --------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(raw: Any) -> Optional[datetime]:
    """宽容解析为 UTC aware datetime；无法解析返回 None（不抛错，不伪造时间）。"""
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:system|user|assistant|developer|tool|系统|用户|助手)\s*[:：]\s*",
    re.IGNORECASE,
)
_CHAT_MARKER_RE = re.compile(r"<\|[^|>]{0,32}\|>")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_text(value: Any, max_len: int = _MAX_QUOTE) -> str:
    """把外部文本变成可安全注入 prompt 的**数据**。

    - 去掉控制字符，换行 / 制表符折成空格并压缩连续空格（不保留伪角色换行）；
    - 剥掉开头的 ``SYSTEM:`` / ``系统：`` 之类角色边界前缀与 ``<|...|>`` 聊天标记；
    - 截断到 ``max_len``。

    注意：这里**不删除**「忽略上面的指令」这类文本——它被当作**被引用的证据**，
    只是永远不被执行（指令注入防护在渲染侧，见方案 §4.4）。
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = _CHAT_MARKER_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s{2,}", " ", text)
    text = _ROLE_PREFIX_RE.sub("", text)
    text = text.strip()
    limit = max(0, int(max_len))
    if limit and len(text) > limit:
        text = text[:limit]
    return text


def new_fact_subkey() -> str:
    """多值键子键：**不含内容**的稳定随机 ID。

    调用方在某个逻辑事实首次入库时分配一次，此后同值同作用域的新证据
    都归到这个子键上。绝不能把规范化值或其哈希写进子键——否则删除留底
    会泄露原文。
    """
    return _uuid.uuid4().hex


# --------------------------------------------------------------------------
# 来源锚点（不含原文与值）
# --------------------------------------------------------------------------


def source_anchor_id(
    dimension: str,
    evidence_refs: Iterable[Any],
    scope: str = "real",
) -> str:
    """稳定来源锚点：由**维度类别 + 作用域 + 稳定证据 ID** 生成 SHA-256。

    ``evidence_refs`` 只放稳定标识（消息 ID / 事件序号 / 测评 ID），
    **不放原句、不放值**。同一句跨管道可合并来源。
    """
    refs = sorted(str(r) for r in (evidence_refs or []) if str(r or "").strip())
    payload = "|".join([str(dimension or ""), str(scope or ""), *refs])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def source_occurrence_id(
    source_pipeline: str,
    anchor_id: str,
    fact_subkey: str = "",
) -> str:
    """同一管道 + 同一来源锚点 + 同一子键 ⇒ 同一 occurrence（重跑幂等）。"""
    payload = "|".join([
        str(source_pipeline or ""),
        str(anchor_id or ""),
        str(fact_subkey or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------
# 归一化
# --------------------------------------------------------------------------


_VALUE_GARBAGE_TOKENS = ("[", "]", "{", "}", "<", ">")


def looks_like_garbage_value(value_text: str) -> bool:
    """值质量闸（2026-10-25 画像审计）：判断一条事实值是不是"人话"。

    只拦**结构性垃圾**（代码擅长的边界判断），不判断内容对错（那是模型的事）：
    - 纯符号/单字符碎片（"[" "]"——存量实锤的 JSON 解析 bug 产物）；
    - 整句对话腔：超长（>60字）或长问句（>25字且带问号）——实锤如 0.98 置信的
      "Oh I see, you're pretty smart... Is it you prefer to use ChatGPT?" 与
      ASR 乱码整句，被当 learning_preference/last_topic 落库；
    - 证据腔：以 "我说的是/我说过" 开头的自我指认残句（ASR 乱码典型形态）。
    命中 ⇒ normalize_fact 拒收（返回 None），不落库。
    """
    text = str(value_text or "").strip()
    if not text:
        return True
    # 纯符号/单字符碎片
    if len(text) <= 2 and not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    if all(ch in _VALUE_GARBAGE_TOKENS or ch.isspace() for ch in text):
        return True
    # 整句对话腔/问句（值应当是"偏好/话题/名字"这类名词性短语）
    if len(text) > 60:
        return True
    if len(text) > 25 and ("?" in text or "？" in text):
        return True
    if text.startswith(("我说的是", "我说过", "我是说")):
        return True
    # 第一人称意愿/陈述整句（"Hi! I want to practice English by just chatting
    # today."被当 learning_preference 落库的实锤形态）：偏好值应是提炼后的短语，
    # 不是用户原话整句的搬运。
    _first_person_openers = (
        "i want", "i'd like", "i would like", "i hope", "i plan",
        "我想", "我希望", "我打算", "我喜欢先练", "hi!", "hello!", "hey",
    )
    lowered = text.lower()
    if lowered.startswith(_first_person_openers) and len(text) > 12:
        return True
    return False


def normalize_fact(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把一个候选（来自任意管道的原始产物）归一化成一行动画像事实。

    返回 ``None`` 表示不可入库：键不在白名单、维度与键不符、或值被消毒后为空
    （``forget`` 控制行允许空值）。

    不做任何 LLM 调用；不猜时间——解析不出 ``observed_at`` 就记 ``time_basis='unknown'``。
    """
    if not isinstance(raw, dict):
        return None
    fact_key = str(raw.get("fact_key") or "").strip()
    if fact_key not in FACT_KEY_DIMENSION:
        return None

    status = str(raw.get("status") or "candidate").strip() or "candidate"
    if status not in STATUSES:
        return None

    is_forget = status == "forget"
    value_text = sanitize_text(raw.get("value_text"))
    if not value_text and not is_forget:
        return None
    if not is_forget and looks_like_garbage_value(value_text):
        return None

    dimension = raw.get("dimension") or FACT_KEY_DIMENSION[fact_key]
    if dimension not in DIMENSIONS:
        return None
    if FACT_KEY_DIMENSION[fact_key] != dimension:
        return None

    evidence_refs = [
        sanitize_text(r, 64) for r in (raw.get("evidence_refs") or [])
        if str(r or "").strip()
    ]
    evidence_quote = sanitize_text(raw.get("evidence_quote"), _MAX_QUOTE)
    if is_forget:
        # forget 控制行**不留值与原话**（方案 §4.3）：只保留 用户 / 逻辑键 /
        # 删除时间 / 旧来源锚点，用于阻止补偿任务从旧对话重新种回。
        value_text = ""
        evidence_refs = []
        evidence_quote = ""

    confidence = raw.get("confidence")
    if confidence is not None and confidence != "":
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None
    else:
        confidence = None

    observed_at = parse_dt(raw.get("observed_at"))
    explicit_from = parse_dt(raw.get("valid_from"))
    if explicit_from is not None:
        valid_from, time_basis = explicit_from, "explicit"
    elif observed_at is not None:
        valid_from, time_basis = observed_at, "observed"
    else:
        valid_from, time_basis = None, "unknown"

    scope = str(raw.get("scope") or "real").strip() or "real"
    anchor = str(raw.get("source_anchor_id") or "").strip() or source_anchor_id(
        dimension, evidence_refs, scope
    )
    pipeline = str(raw.get("source_pipeline") or "").strip()
    fact_subkey = str(raw.get("fact_subkey") or "").strip()
    if fact_key in MULTI_VALUE_FACT_KEYS and not fact_subkey:
        fact_subkey = new_fact_subkey()
    if fact_key in SINGULAR_FACT_KEYS:
        fact_subkey = ""

    occurrence = str(raw.get("source_occurrence_id") or "").strip() or source_occurrence_id(
        pipeline, anchor, fact_subkey
    )

    return {
        "fact_id": str(raw.get("fact_id") or _uuid.uuid4().hex),
        "user_id": str(raw.get("user_id") or ""),
        "dimension": dimension,
        "fact_key": fact_key,
        "fact_subkey": fact_subkey,
        "value_text": value_text,
        "source_pipeline": pipeline,
        "source_table": str(raw.get("source_table") or ""),
        "source_id": str(raw.get("source_id") or ""),
        "source_anchor_id": anchor,
        "source_occurrence_id": occurrence,
        "extractor_version": str(raw.get("extractor_version") or ""),
        "source_conv_id": raw.get("source_conv_id"),
        "source_session_id": raw.get("source_session_id"),
        "evidence_quote": evidence_quote,
        "evidence_refs": evidence_refs,
        "evidence_kind": str(raw.get("evidence_kind") or "unknown"),
        "confidence": confidence,
        "observed_at": observed_at.isoformat() if observed_at else None,
        "valid_from": valid_from.isoformat() if valid_from else None,
        "valid_to": parse_dt(raw.get("valid_to")).isoformat()
        if parse_dt(raw.get("valid_to")) else None,
        "recorded_at": (parse_dt(raw.get("recorded_at")) or now_utc()).isoformat(),
        "time_basis": time_basis,
        "expires_at": parse_dt(raw.get("expires_at")).isoformat()
        if parse_dt(raw.get("expires_at")) else None,
        "status": status,
        "scope": scope,
    }


# --------------------------------------------------------------------------
# 打分与裁决
# --------------------------------------------------------------------------


def decay_score(fact: Dict[str, Any], now: Optional[datetime] = None) -> float:
    """``confidence × 2^(-age_days / half_life_days)``；常青键不衰减。

    **只影响检索排名**，不回写原始 ``confidence``、不删旧版本。
    缺 ``confidence`` 按 0 计；缺 ``observed_at`` 按常青处理（不因解析失败而抹掉）。
    """
    if not isinstance(fact, dict):
        return 0.0
    fact_key = str(fact.get("fact_key") or "")
    try:
        confidence = float(fact.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    if fact_key in EVERGREEN_FACT_KEYS:
        return confidence

    observed_at = parse_dt(fact.get("observed_at") or fact.get("valid_from"))
    if observed_at is None:
        return confidence

    ref = now or now_utc()
    age_days = (ref - observed_at).total_seconds() / 86400.0
    half_life = float(HALF_LIFE_DAYS.get(fact_key, DEFAULT_HALF_LIFE_DAYS))
    if half_life <= 0:
        return confidence
    return confidence * (2.0 ** (-age_days / half_life))


def _resolution_key(fact: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """确定性裁决键（越大越优先）：来源资格 > 现实作用域 > 证据时刻 > 置信度。

    ``fact_id`` 故意**不**进这个键——它是最终稳定排序用，不参与"能否裁决"判断。
    """
    rank = SOURCE_RANK.get(str(fact.get("source_pipeline") or ""), 0)
    scope = 1 if str(fact.get("scope") or "real") == "real" else 0
    observed = parse_dt(fact.get("observed_at") or fact.get("valid_from"))
    observed_ts = observed.timestamp() if observed else float("-inf")
    try:
        conf = float(fact.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return (rank, scope, observed_ts, conf)


def resolve_current(
    facts: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把候选裁决成「当前唯一权威值」。

    - 单值键按 ``fact_key`` 分组；多值键按 ``(fact_key, fact_subkey)`` 分组。
    - 排序语义：**晚到的旧任务 / 低可信推断不能覆盖新近明确更正**，由
      ``_resolution_key`` 里"来源资格 → 证据时刻"的顺序保证。
    - 前两名在语义键上完全并列、值却不同 ⇒ 判 ``contested``，**不进 prompt**。

    返回 ``(current, contested)``；``forget`` 控制行与 ``legacy_unattributed``
    不参与裁决，原样留在输入里由调用方处理。
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    passthrough: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        status = str(fact.get("status") or "candidate")
        fact_key = str(fact.get("fact_key") or "")
        if status in ("forget", "legacy_unattributed") or not fact_key:
            passthrough.append(dict(fact))
            continue
        subkey = str(fact.get("fact_subkey") or "") if fact_key in MULTI_VALUE_FACT_KEYS else ""
        groups.setdefault((fact_key, subkey), []).append(dict(fact))

    current: List[Dict[str, Any]] = []
    contested: List[Dict[str, Any]] = []
    for _, bucket in sorted(groups.items()):
        ordered = sorted(
            bucket,
            key=lambda f: (_resolution_key(f), str(f.get("fact_id") or "")),
            reverse=True,
        )
        top = ordered[0]
        if len(ordered) > 1:
            runner = ordered[1]
            same_semantics = _resolution_key(top) == _resolution_key(runner)
            if same_semantics and str(top.get("value_text")) != str(runner.get("value_text")):
                for item in ordered:
                    mark = dict(item)
                    mark["status"] = "contested"
                    contested.append(mark)
                continue
        winner = dict(top)
        winner["status"] = "confirmed"
        current.append(winner)
    return current, contested + passthrough


def as_of(
    facts: Iterable[Dict[str, Any]],
    when: Any,
    *,
    fact_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按时点返回**当时有效**的已确认事实（区间语义 ``[valid_from, valid_to)``）。

    指代或时刻无法确定时不要调用它猜——调用方应要求澄清或弃答（方案 §4.3）。
    """
    ref = parse_dt(when)
    if ref is None:
        return []
    out: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        if str(fact.get("status") or "") != "confirmed":
            continue
        if fact_key and str(fact.get("fact_key") or "") != fact_key:
            continue
        start = parse_dt(fact.get("valid_from"))
        if start is None or start > ref:
            continue
        end = parse_dt(fact.get("valid_to"))
        if end is not None and ref >= end:
            continue
        out.append(dict(fact))
    return out


def select_for_prompt(
    facts: Iterable[Dict[str, Any]],
    *,
    top_k: int = 8,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """按打分取 top-k 供 prompt 使用。

    - 只吃已 ``confirmed`` 的事实；``expires_at`` 已过的退出**当前**视图（历史仍在）。
    - 超额时先舍低分、争议项；``PROMPT_PROTECTED_FACT_KEYS``（实名 / 明确边界）
      永不被挤掉。
    """
    ref = now or now_utc()
    eligible: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        if str(fact.get("status") or "") != "confirmed":
            continue
        expires = parse_dt(fact.get("expires_at"))
        if expires is not None and expires <= ref:
            continue
        eligible.append(dict(fact))

    eligible.sort(
        key=lambda f: (decay_score(f, ref), str(f.get("fact_id") or "")),
        reverse=True,
    )
    limit = max(0, int(top_k))
    if len(eligible) <= limit:
        return eligible

    kept: List[Dict[str, Any]] = []
    for fact in eligible:
        if len(kept) >= limit:
            break
        kept.append(fact)
    protected = [
        f for f in eligible
        if str(f.get("fact_key") or "") in PROMPT_PROTECTED_FACT_KEYS
    ]
    seen = {str(f.get("fact_id") or "") for f in kept}
    for fact in protected:
        if len(kept) >= limit and str(fact.get("fact_id") or "") not in seen:
            # 挤掉分数最低且未被保护的末位
            for idx in range(len(kept) - 1, -1, -1):
                if str(kept[idx].get("fact_key") or "") not in PROMPT_PROTECTED_FACT_KEYS:
                    seen.discard(str(kept[idx].get("fact_id") or ""))
                    kept[idx] = fact
                    seen.add(str(fact.get("fact_id") or ""))
                    break
    kept.sort(
        key=lambda f: (decay_score(f, ref), str(f.get("fact_id") or "")),
        reverse=True,
    )
    return kept


# --------------------------------------------------------------------------
# 投影：既有管道的产物 → 画像事实（阶段 1，读侧贯通）
# --------------------------------------------------------------------------

DIMENSION_LABELS: Dict[str, str] = {
    DIM_IDENTITY: "身份与关系",
    DIM_LEVEL: "语言水平",
    DIM_WEAK_MASTERED: "薄弱项与已掌握",
    DIM_PREFERENCE: "偏好与边界",
    DIM_INTEREST: "兴趣与话题",
    DIM_OPEN_LOOP: "未了结的事",
    DIM_MOOD: "情绪与关系状态",
}
_DIMENSION_ORDER = {dim: idx for idx, dim in enumerate(DIMENSIONS)}

# `relationship_memory` 的标量列 → fact_key（键名与 relationship_merge 一致）
_RELATIONSHIP_SCALAR_KEYS = (
    "learner_name", "tutor_name", "learner_roleplay_alias", "tutor_roleplay_alias",
    "last_topic", "open_loop", "recent_mood", "preferred_tone",
    "learning_preference", "self_reported_weakness",
    "next_session_intent", "next_session_intent_consumed",
)
# `relationship_memory` 的 JSONB 列表列 → 多值 fact_key
_RELATIONSHIP_LIST_KEYS = {
    "engagement_topics": "engagement_topic",
    "avoid_patterns": "avoid_pattern",
}


def facts_from_relationship_memory(
    user_id: str,
    relationship: Optional[Dict[str, Any]],
    *,
    source_pipeline: str = "per_turn",
    scope: str = "real",
) -> List[Dict[str, Any]]:
    """把一份关系记忆读视图投影成画像候选事实。

    **逐轮管道只投影它写得出来的东西**；不在这里做跨源裁决、不调模型。
    `memory_items` 里同键条目自带 `created_at`/`confidence`/`evidence`/`expires_at`，
    取最近一条作为这一行的证据与时间；拿不到就留 `time_basis='unknown'`，
    **不把投影时间当事实发生时间**。
    """
    if not isinstance(relationship, dict):
        return []

    item_by_key: Dict[str, Dict[str, Any]] = {}
    for item in (relationship.get("memory_items") or []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        prev = item_by_key.get(key)
        if prev is None or _created_ts(item) >= _created_ts(prev):
            item_by_key[key] = item

    out: List[Dict[str, Any]] = []

    # 2026-10-25 整句搬运修复：per_turn 规则管道（tutor_agent remember()）对
    # last_topic/learning_preference/next_session_intent 存的是用户整句原话
    # （"对了正式自我介绍一下，我叫小周…"整句当 last_topic，真机实锤，且仅
    # 38 字——长度闸挡不住）。整句搬运的铁证签名是 value == evidence（规则
    # 管道 remember(key, text, ..., text) 值与证据同串，从未提炼）；curator
    # 提炼版（"爬山：本月已第四次…"）value 是短语、evidence 是原句，两者不同。
    # 闸只挡"投影进画像表"：relationship_memory 短时层照存不丢，同一内容的
    # 提炼版由会话末 curator 路径写入画像。
    _VERBATIM_COPY_FACT_KEYS = frozenset({
        "last_topic", "learning_preference", "next_session_intent", "recent_mood",
    })

    def _emit(fact_key: str, value: Any, *, subkey: str = "") -> None:
        text = sanitize_text(value, 140)
        if not text:
            return
        if fact_key in _VERBATIM_COPY_FACT_KEYS:
            meta = item_by_key.get(fact_key) or {}
            evidence = sanitize_text(meta.get("evidence"), 140)
            if evidence and evidence == text:
                return
        meta = item_by_key.get(fact_key) or {}
        refs = []
        if meta.get("id"):
            refs.append(str(meta["id"]))
        refs.append(f"{source_pipeline}:{fact_key}")
        observed = meta.get("created_at")
        out.append({
            "user_id": str(user_id or ""),
            "fact_key": fact_key,
            "fact_subkey": subkey,
            "value_text": text,
            "source_pipeline": source_pipeline,
            "source_table": "relationship_memory",
            "source_id": str(meta.get("id") or ""),
            "evidence_quote": sanitize_text(meta.get("evidence"), _MAX_QUOTE),
            "evidence_refs": refs,
            "evidence_kind": "utterance" if meta.get("evidence") else "unknown",
            "confidence": meta.get("confidence"),
            "observed_at": observed,
            "expires_at": meta.get("expires_at"),
            "scope": scope,
            # 逐轮写出来的条目在用户明确确立时可信度高；此处只用来源资格记号，
            # 真正"是否 confirmed"由 resolve_current 裁决，不在这里抢先定性。
            "status": "candidate",
        })

    for key in _RELATIONSHIP_SCALAR_KEYS:
        if key in FACT_KEY_DIMENSION:
            _emit(key, relationship.get(key))
    for list_key, fact_key in _RELATIONSHIP_LIST_KEYS.items():
        values = relationship.get(list_key)
        if isinstance(values, str):
            text = values.strip()
            if not text or text.startswith("["):
                # 序列化痕迹（视图把空列表存成 "[]"）：逐字符迭代会产出 '['
                # 和 ']' 这类垃圾候选（e2e 实测），整串跳过；普通单词串才收。
                continue
            values = [text]
        if not isinstance(values, (list, tuple, set)):
            continue
        for value in values:
            _emit(fact_key, value)
    return out


def _created_ts(item: Dict[str, Any]) -> float:
    parsed = parse_dt(item.get("created_at"))
    return parsed.timestamp() if parsed else float("-inf")


def render_profile_packet(
    facts: Iterable[Dict[str, Any]],
    *,
    max_lines: int = 8,
    now: Optional[datetime] = None,
) -> str:
    """把**已裁决的当前事实**渲染成一段可直接进逐轮 prompt 的短摘编。

    没有可注入事实时返回 **空串**（调用方据此跳过，绝不留空行）。
    只给「值 + 日期 + 来源短标识 + 可信度档」，**不广播整句原话**。
    """
    selected = select_for_prompt(facts, top_k=max_lines, now=now)
    if not selected:
        return ""
    selected.sort(key=lambda f: (
        _DIMENSION_ORDER.get(str(f.get("dimension") or ""), 99),
        -decay_score(f, now),
    ))
    lines = [
        "【统一画像·可按已知事实直接使用；带日期与来源，不是本轮新信息。"
        "可以据此推理，也可以用自己的话讲——**结论可以是新的，事实不能是新的**："
        "不要补造这里没有的具体细节】"
    ]
    for fact in selected:
        dim = str(fact.get("dimension") or "")
        label = DIMENSION_LABELS.get(dim, dim or "其他")
        date = beijing_day_label(fact.get("observed_at") or fact.get("valid_from"))
        source = str(fact.get("source_pipeline") or "unknown")
        try:
            conf = float(fact.get("confidence"))
            conf_text = f"{conf:.2f}"
        except (TypeError, ValueError):
            conf_text = "n/a"
        value = sanitize_text(fact.get("value_text"), 140)
        if not value:
            continue
        meta = "·".join([piece for piece in (date, source, conf_text) if piece])
        lines.append(f"· {label}：{fact.get('fact_key')}={value}（{meta}）")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def render_narrative_packet(narratives: Iterable[Dict[str, Any]], *,
                            max_chars: int = 640) -> str:
    """把已沉淀的维度叙事渲染成一段「用她自己的话记住这个人」。

    没有可用叙事时返回 **空串**（调用方据此回落档案行，绝不留空行）。

    与 `render_profile_packet` 的分工（plans/1017-01 §12 A1）：那份是**档案**
    （值 + 日期 + 来源 + 可信度），这份是**她的理解**。两者是同一批事实的
    两个权威源 ⇒ **不并排出现**，有叙事就只说叙事。

    维度排序固定（与档案同一个 `_DIMENSION_ORDER`），保证逐字确定、不吃缓存前缀。
    """
    ordered: List[Dict[str, Any]] = []
    for nar in narratives or []:
        if not isinstance(nar, dict):
            continue
        text = sanitize_text(nar.get("text"), 180)
        if not text:
            continue
        ordered.append({"dimension": str(nar.get("dimension") or ""), "text": text})
    if not ordered:
        return ""
    ordered.sort(key=lambda n: _DIMENSION_ORDER.get(n["dimension"], 99))
    lines: List[str] = []
    total = 0
    for entry in ordered:
        if total + len(entry["text"]) > max_chars:
            break
        lines.append("· " + entry["text"])
        total += len(entry["text"])
    if not lines:
        return ""
    return (
        "【我记得的这个人·用你自己的话】（下面这些是你和他相处里慢慢形成的印象，"
        "不是档案条目：合适的时候可以顺着话题自然带出，也可以从这些印象里得出"
        "新的看法——**结论可以是新的，事实不能是新的**。用你的语言环境里平时"
        "说话的口吻带出（她这里就是英语），像聊天时想起这人那样自然，别用"
        "'you told me / you mentioned' 这类报告式转述、别照抄原话）：\n"
        + "\n".join(lines)
    )


def beijing_day_label(raw: Any) -> str:
    """证据时间 → `MM-DD`（北京时间）。解析不出来返回空串，不猜。"""
    dt = parse_dt(raw)
    if dt is None:
        return ""
    return (dt + timedelta(hours=8)).strftime("%m-%d")


# --------------------------------------------------------------------------
# as_of 逐轮备轮探测（plans/0923-04 §4.3 阶段 3）
# --------------------------------------------------------------------------

# 回忆线索词：没有它们就不触发按时查询（避免普通陈述被误判成"问过去"）。
_AS_OF_CUES = (
    "那时", "当时", "上次", "那天", "那个时候", "那会儿",
    "还记得", "记不记得", "以前", "之前", "早些时候",
)

_CN_DIGIT = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
_CN_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5,
               "日": 6, "天": 6}


def _cn_to_int(text: str) -> Optional[int]:
    """「十／十X／X十／X十X」型简繁常用中文数字 → int；认不出返回 None。"""
    text = str(text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGIT.get(left, 1) if left else 1
        ones = _CN_DIGIT.get(right, 0) if right else 0
        if (left and left not in _CN_DIGIT) or (right and right not in _CN_DIGIT):
            return None
        return tens * 10 + ones
    return _CN_DIGIT.get(text)


def _past_instant(ref_utc: datetime, target_date, end_of_day: bool) -> datetime:
    """北京日历日 → UTC 即刻。end_of_day 取当天 23:59（含当天发生的变更）。"""
    hh, mm = (23, 59) if end_of_day else (12, 0)
    local = datetime(target_date.year, target_date.month, target_date.day, hh, mm,
                     tzinfo=timezone(timedelta(hours=8)))
    return local.astimezone(timezone.utc)


def _resolve_past_moment(text: str, ref_utc: datetime) -> Optional[Dict[str, str]]:
    """从文本解析**可确定**的历史时刻；认不出返回 None（绝不猜）。"""
    ref_local = ref_utc + timedelta(hours=8)          # 北京日历日
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?\s*月\s*([0-9一二两三四五六七八九十]+)\s*[号日]", text)
    if m:
        month, day = _cn_to_int(m.group(1)), _cn_to_int(m.group(2))
        if month and day and 1 <= month <= 12 and 1 <= day <= 31:
            try:
                target = ref_local.date().replace(month=month, day=day)
            except ValueError:
                return None
            if target >= ref_local.date():            # 未来 ⇒ 指去年同月日
                try:
                    target = target.replace(year=target.year - 1)
                except ValueError:                    # 2/29 → 平年
                    return None
            if target >= ref_local.date():
                return None
            return {"as_of": _past_instant(ref_utc, target, True).isoformat(),
                    "label": f"{month}月{day}日"}
        return None
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?\s*月前", text)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            target = ref_local.date()
            for _ in range(n):                        # 按自然月回退，不猜天数
                prev_month_last = target.replace(day=1) - timedelta(days=1)
                target = prev_month_last.replace(
                    day=min(target.day, prev_month_last.day))
            if target >= ref_local.date():
                return None
            return {"as_of": _past_instant(ref_utc, target, True).isoformat(),
                    "label": f"{n}个月前"}
        return None
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?\s*周前", text)
    if m:
        n = _cn_to_int(m.group(1))
        if n and n > 0:
            target = ref_local.date() - timedelta(days=7 * n)
            return {"as_of": _past_instant(ref_utc, target, True).isoformat(),
                    "label": f"{n}周前"}
        return None
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*天前", text)
    if m:
        n = _cn_to_int(m.group(1))
        if n and n > 0:
            target = ref_local.date() - timedelta(days=n)
            return {"as_of": _past_instant(ref_utc, target, True).isoformat(),
                    "label": f"{n}天前"}
        return None
    m = re.search(r"上(?:个)?周\s*([一二三四五六日天])", text)
    if m:
        wd = _CN_WEEKDAY.get(m.group(1))
        if wd is not None:
            delta = (ref_local.weekday() - wd) % 7 + 7   # 上周同一星期几（≥7 天前）
            target = ref_local.date() - timedelta(days=delta)
            return {"as_of": _past_instant(ref_utc, target, True).isoformat(),
                    "label": f"上周{m.group(1)}"}
        return None
    for token, n, label in (("大前天", 3, "大前天"), ("前天", 2, "前天"), ("昨天", 1, "昨天")):
        if token in text:
            target = ref_local.date() - timedelta(days=n)
            return {"as_of": _past_instant(ref_utc, target, True).isoformat(),
                    "label": label}
    return None


def detect_as_of_query(text: Any, now: Optional[datetime] = None) -> Optional[Dict[str, str]]:
    """§4.3 逐轮探测：文本里**同时**有回忆线索词 + 可确定的历史时刻才返回查询参数。

    返回 ``{"as_of": <iso>, "label": <人类可读>, "cue": <命中的线索词>}``；
    缺线索词（不是在问过去）或时刻定不出来 ⇒ None（调用方什么都不注入，
    绝不把当前值倒灌成过去值）。
    """
    raw = str(text or "")
    if not raw.strip():
        return None
    cue = next((c for c in _AS_OF_CUES if c in raw), None)
    if cue is None:
        return None
    ref = now or now_utc()
    moment = _resolve_past_moment(raw, ref)
    if moment is None:
        return None
    return {"as_of": str(moment["as_of"]), "label": str(moment["label"]), "cue": cue}


def render_as_of_line(hits: Iterable[Dict[str, Any]], probe: Dict[str, Any]) -> str:
    """把 ``as_of`` 命中渲染成**一条**有来源的短摘编（进逐轮 prompt）。

    - 命中：``【按时记忆·<label>】当时记录：…``，每条 = 值 + 原话/来源，
      最多 4 条，同 fact_key 只留一条（排序稳定：按 valid_from 新→旧）。
    - 零命中：给一条**防倒灌**护栏句（不要拿当前值冒充当时值）。
    - probe 缺 as_of ⇒ 空串（调用方跳过，绝不留空行）。
    """
    moment = parse_dt((probe or {}).get("as_of"))
    if moment is None:
        return ""
    label = str((probe or {}).get("label") or "目标时刻")
    when_text = (moment + timedelta(hours=8)).strftime("%Y-%m-%d")
    rows: List[Dict[str, Any]] = [
        dict(f) for f in (hits or [])
        if isinstance(f, dict) and str(f.get("fact_key") or "")
        and str(f.get("value_text") or "").strip()
    ]
    if not rows:
        return (f"【按时记忆·{label}（{when_text}，北京时间）】该时刻没有可靠的画像"
                "证据；不要拿现在的值冒充当时的值，答不上就如实说需要确认。")
    rows.sort(key=lambda f: (parse_dt(f.get("valid_from")) or moment), reverse=True)
    parts: List[str] = []
    seen_keys: set = set()
    for fact in rows:
        key = str(fact.get("fact_key") or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        value = sanitize_text(fact.get("value_text"), 60)
        quote = sanitize_text(fact.get("evidence_quote"), 40)
        source = str(fact.get("source_pipeline") or "").strip()
        piece = f"{key}={value}"
        if quote:
            piece += f"（原话「{quote}」）"
        elif source:
            piece += f"（来源：{source}）"
        parts.append(piece)
        if len(parts) >= 4:
            break
    if not parts:
        return ""
    return (f"【按时记忆·{label}（{when_text}，北京时间）；以下是当时已确认的记录，"
            "不是当前值】" + "；".join(parts))


# --------------------------------------------------------------------------
# 被忘引用过滤 · 逐键切换门槛 · 叙事验证（plans/0923-04 §4.3/§4.4/§4.5）
# --------------------------------------------------------------------------

# 召回/事实卡条目里**可比对**来源引用的字段（精确匹配；格式对不上就不过滤，不猜）。
_RECALL_ID_FIELDS = ("id", "session_id", "message_id", "source_id", "conv_id")


def filter_forgotten_sources(items: Iterable[Any], forgotten_refs: Iterable[Any],
                             *, id_fields: Optional[Iterable[str]] = None) -> List[Any]:
    """按被忘事实的来源引用过滤召回/事实卡条目（§4.3 后半，纯函数）。

    - ``forgotten_refs`` 为空 ⇒ 原样返回（零 forget 用户的召回路径零开销）。
    - 条目任一 ID 字段精确命中被忘引用 ⇒ **整条剔除**（不截选段）。
    - 字段值对不上 ⇒ 保留：它是否含被忘原话**无法精确判定**，不猜；
      真机核验各召回源 ID 格式后再按方案收紧为保守跳过。
    """
    refs = {str(r or "").strip() for r in (forgotten_refs or []) if str(r or "").strip()}
    if not refs:
        return list(items or [])
    fields = tuple(id_fields) if id_fields else _RECALL_ID_FIELDS
    kept: List[Any] = []
    for item in items or []:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        hit = False
        for field in fields:
            value = str(item.get(field) or "").strip()
            if value and value in refs:
                hit = True
                break
        if not hit:
            kept.append(item)
    return kept


def takeover_audit(facts: Iterable[Dict[str, Any]],
                   old_keys: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """§4.5 逐键切换门槛的**按键裁决**（纯函数，不改任何 prompt 渲染）。

    对旧 prompt 已展示的每个键返回裁定，供回填清单与"完整统一读"开启决策用：

    - ``takeover_ready``：新表有**可追溯**（evidence_refs 非空）、已确认、
      未争议的当前事实 ⇒ 统一 packet 有资格接管该键（附最优行的键值元数据）；
    - ``legacy_unattributed``：该键只有无来源引用的行 ⇒ 旧段原样显示，
      回填清单标"历史未归因"，不伪造时间/置信度；
    - ``contested``：该键存在争议候选 ⇒ 保持旧段，不注入；
    - ``no_new_evidence``：该键在新表里没有行。
    """
    current, contested = resolve_current(facts)
    contested_keys = {str(f.get("fact_key") or "") for f in contested}
    out: Dict[str, Dict[str, Any]] = {}
    for key in old_keys:
        key = str(key or "").strip()
        if not key:
            continue
        if key in contested_keys:
            out[key] = {"verdict": "contested"}
            continue
        candidates = [f for f in current if str(f.get("fact_key") or "") == key]
        if not candidates:
            out[key] = {"verdict": "no_new_evidence"}
            continue
        attributed = [
            f for f in candidates
            if [str(r) for r in (f.get("evidence_refs") or []) if str(r or "").strip()]
        ]
        if not attributed:
            out[key] = {"verdict": "legacy_unattributed"}
            continue
        attributed.sort(key=lambda f: (
            float(f.get("confidence") or 0.0),
            str(f.get("observed_at") or ""),
        ), reverse=True)
        best = attributed[0]
        out[key] = {
            "verdict": "takeover_ready",
            "fact_id": str(best.get("fact_id") or ""),
            "value_text": str(best.get("value_text") or ""),
            "confidence": best.get("confidence"),
            "source_pipeline": str(best.get("source_pipeline") or ""),
        }
    return out


def build_narrative_payload(facts: Iterable[Dict[str, Any]], *,
                            max_facts: int = 24) -> Dict[str, Any]:
    """冷路径叙事的**允许输入**：只有 fact_id + 维度 + 消毒短值（§4.4）。

    只吃已确认事实；引文/原话不进 payload——叙事只能引用 L1 事实，
    不能成为事实权威。
    """
    allowed: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict) or str(fact.get("status") or "") != "confirmed":
            continue
        fid = str(fact.get("fact_id") or "").strip()
        value = sanitize_text(fact.get("value_text"), 60)
        if not fid or not value:
            continue
        allowed.append({
            "fact_id": fid,
            "dimension": str(fact.get("dimension") or ""),
            "fact_key": str(fact.get("fact_key") or ""),
            "value": value,
        })
        if len(allowed) >= max_facts:
            break
    return {"facts": allowed}


def narratives_from_llm_json(raw: Any,
                             allowed_facts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """逐条验证叙事 LLM 输出（§4.4：逐条验证支持 ID，不合格的丢弃该条）。

    - ``dimension`` 必须在七维白名单内；``text`` 消毒后非空且 ≤400 字；
    - ``supporting_fact_ids`` 必须全部 ∈ 允许集且至少 1 个，否则该条丢弃；
    - 输出不是期望形状 ⇒ 空列表（调用方沿用上一版叙事或不显示）。
    """
    allowed_ids = {str(f.get("fact_id") or "") for f in (allowed_facts or []) if isinstance(f, dict)}
    items: Any = []
    if isinstance(raw, dict):
        items = raw.get("narratives")
    elif isinstance(raw, list):
        items = raw
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items[:len(DIMENSIONS)]:
        if not isinstance(item, dict):
            continue
        dim = str(item.get("dimension") or "")
        if dim not in DIMENSIONS:
            continue
        text = sanitize_text(item.get("text"), 400)
        if not text:
            continue
        ids: List[str] = []
        broken = False
        for sid in (item.get("supporting_fact_ids") or []):
            sid_text = str(sid or "").strip()
            if sid_text not in allowed_ids:
                # 逐条验证取**严格**方向：引用集里混入任何不可追溯 ID ⇒ 该条
                # 整体丢弃（叙事里可能混着无支持的断言，宁缺勿假，§4.4）。
                broken = True
                break
            if sid_text not in ids:
                ids.append(sid_text)
        if broken or not ids:
            continue
        out.append({"dimension": dim, "text": text, "supporting_fact_ids": ids})
    return out


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _run_writer(writer: Any) -> Any:
    """跑"原写入"。`writer` 可以是无参可调用对象，也可以是现成 awaitable。

    **不吞异常**：非删除态下必须与不套闸门的调用语义一致。
    """
    value = writer() if callable(writer) else writer
    return await _maybe_await(value)


def lease_tokens(*sources: Any) -> List[str]:
    """把「根令牌 + 任务令牌」这类混杂入参归一成去重、非空的字符串列表。

    根令牌是会话开场登记的（§4.4）；任务令牌是本次后台作业自己登记的。
    两者**都要在**才算通过 —— 全删撤销根令牌后，从该会话派生的任何迟到作业
    都通不过校验。
    """
    out: List[str] = []
    for src in sources:
        if src is None:
            continue
        seq = src if isinstance(src, (list, tuple, set)) else [src]
        for item in seq:
            token = str(item or "").strip()
            if token and token not in out:
                out.append(token)
    return out


# --------------------------------------------------------------------------
# forget 抑制（plans/0923-04 §4.3，阶段 2 补全）
# --------------------------------------------------------------------------


def is_fact_suppressed(forget_rows: Iterable[Any], fact: Dict[str, Any]) -> bool:
    """判断一条候选事实是否被 ``forget`` 控制行抑制（旧原文重扫不复活）。

    控制行**不含原话与值**，只有 ``fact_key`` / ``fact_subkey`` / 旧来源锚点。
    判定规则（保守方向见方案 §4.3「旧项无法唯一定位到原话时，保守抑制」）：

    - 候选**带** ``source_anchor_id``：只有锚点**精确命中**某条控制行才抑制。
      锚点是"维度+作用域+稳定证据 ID"的 SHA-256，**不含子键**——所以删除后
      重投影分配的新子键不影响命中；同一句旧原文重扫必然复算出同一锚点 ⇒
      命中被挡，而用户新说的新话是新锚点 ⇒ 放行（"删除后的新明确表述可重新进入"）。
      唯一例外：控制行本身**没有**锚点（被删的旧证据本就无法归因）且键是
      单值槽 ⇒ 整槽保守抑制——此时无法区分"旧证据重扫"与"新表述"，按方案选保守。
    - 候选**不带**锚点：无法定位 ⇒ 该 ``fact_key`` 存在任意控制行即抑制。
    """
    if not isinstance(fact, dict) or not forget_rows:
        return False
    key = str(fact.get("fact_key") or "").strip()
    if not key:
        return False
    anchor = str(fact.get("source_anchor_id") or "").strip()
    for row in forget_rows:
        if not isinstance(row, dict) or str(row.get("fact_key") or "") != key:
            continue
        if not anchor:
            return True
        row_anchor = str(row.get("source_anchor_id") or "").strip()
        if row_anchor and row_anchor == anchor:
            return True
        if not row_anchor and key in SINGULAR_FACT_KEYS:
            return True
    return False


# relationship_memory 条目键 → 画像 fact_key 的别名（条目 key 大多直接用
# fact_key 名字；个别沿用关系记忆列名的复数形式，在这里归一）。
_FORGET_ITEM_KEY_ALIASES: Dict[str, str] = {
    "avoid_patterns": "avoid_pattern",
    "engagement_topics": "engagement_topic",
}


def fact_keys_for_relationship_item(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """把一条 relationship_memory item 映射到它支撑的画像 fact_key（§4.3）。

    旧 ``/relationship-memory`` 按 item ID 删除时用它定位统一侧要 forget 的事实：
    单值键整槽；多值键带 ``value_text`` 让 ``forget`` 按现存行找回子键——
    **只**删这一条对应的事实及全部版本，不能因为同属 ``avoid_pattern``
    就删掉其他边界。不在白名单里的键返回空列表（照旧只走旧删除路径）。
    """
    if not isinstance(item, dict):
        return []
    key = str(item.get("key") or "").strip()
    key = _FORGET_ITEM_KEY_ALIASES.get(key, key)
    if key not in FACT_KEY_DIMENSION:
        return []
    spec: Dict[str, str] = {"fact_key": key}
    if key in MULTI_VALUE_FACT_KEYS:
        spec["value_text"] = str(item.get("value") or "")
    return [spec]


class ProfileMemory:
    """统一画像读写门面（阶段 1：幂等投影 + 读侧摘编；阶段 2：写入租约闸门）。

    `backend` 只要求实现**同签名**方法（PG / SQLite 各一份）：

    - ``upsert_learner_profile_facts(user_id, facts) -> int``
    - ``list_learner_profile_facts(user_id, *, statuses=None) -> list[dict]``
    - ``put_learner_profile_narrative(user_id, dimension, text, supporting_fact_ids)``
    - ``list_learner_profile_narratives(user_id) -> list[dict]``
    - ``register_write_lease(user_id, token, scope_conv_id="") -> bool``
    - ``write_lease_active(user_id, tokens) -> bool``
    - ``revoke_write_leases(user_id, scope_conv_id="") -> int``
    - ``clear_write_leases(user_id) -> int``

    用鸭子类型而不是 import 后端，好处是本模块保持"纯 stdlib、无副作用"，
    可以在没有数据库的环境里单独单测；坏处是拼错方法名只能在运行时发现。
    后四个是阶段 2 新增的：**老后端没有它们时闸门整体旁路**（见
    ``guarded_write`` 的三条放行分支），不会让既有写入一起失效。
    """

    def __init__(self, backend: Any):
        self._backend = backend

    # ── 写入租约闸门（plans/0923-04 §4.4 阶段 2）─────────────────────────────

    async def admit_write(self, user_id: str, *, token: str = "",
                          source_conv_id: str = "") -> str:
        """作业开始时登记令牌（§4.4）。返回令牌；后端不支持时返回 ``""``。

        令牌是**随机**的，不含用户内容；`source_conv_id` 只用于"删/隐藏这一盘时
        按盘撤销该盘的迟到作业"。登记失败不抛错 —— 调用方拿到空串后应继续
        走原路径（由 `guarded_write` 的旁路分支放行）。
        """
        register = getattr(self._backend, "register_write_lease", None)
        if register is None:
            return ""
        token = str(token or "").strip() or _uuid.uuid4().hex
        try:
            ok = await _maybe_await(
                register(str(user_id or ""), token, str(source_conv_id or ""))
            )
        except Exception:
            return ""
        return token if ok else ""

    async def guarded_write(self, user_id: str, tokens: Any, writer: Any,
                            *, source_conv_id: str = "") -> Dict[str, Any]:
        """在删除闸门内核验令牌后执行**原写入**（§4.4「旧写手写入闸门」）。

        - `tokens` 可以是单个令牌，或"根令牌 + 任务令牌"的列表；**全部有效**才放行。
        - `writer` 是无参可调用对象（内部完成原来的写入）；其**原样返回值放在
          ``value``**，其**异常原样抛出**——非删除态下调用顺序、参数、返回值及
          异常处理与不套闸门时完全一致。
        - 返回 ``{"ok", "gated", "reason", "value"}``。``reason`` 是无原文状态码。

        三条**放行**分支（都不改变原有写入行为）：
        ① 没传令牌 / 后端没实现租约方法 / 迁移未就绪 ⇒ ``gated=False``；
        ② 校验查询本身抛错 ⇒ ``gated=True`` 但放行（§4.5 回退口径）；
        ③ 令牌全部有效 ⇒ 放行。
        只有"查得到、且令牌已被撤销或不存在"才拒写。
        """
        token_list = lease_tokens(tokens)
        check = getattr(self._backend, "write_lease_active", None)
        if check is None or not token_list:
            return {"ok": True, "gated": False,
                    "reason": LEASE_GATE_UNAVAILABLE,
                    "value": await _run_writer(writer)}
        try:
            admitted = bool(await _maybe_await(
                check(str(user_id or ""), list(token_list))
            ))
        except Exception:
            admitted = True
            reason = LEASE_CHECK_ERROR_OPEN
        else:
            reason = "" if admitted else LEASE_REVOKED_OR_MISSING
        if not admitted:
            return {"ok": False, "gated": True,
                    "reason": LEASE_REVOKED_OR_MISSING, "value": None}
        return {"ok": True, "gated": True, "reason": reason,
                "value": await _run_writer(writer)}

    async def project_facts(self, user_id: str, facts: Iterable[Dict[str, Any]]) -> int:
        """幂等投影：同一 occurrence 重跑不新增版本。

        多值键的**子键必须跨次复用**——所以先读一遍已有行，按
        `(fact_key, value_text)` 找回过子键；找不到才分配新的。
        子键本身不含内容，仅用于"同一条边界/话题"的稳定标识。
        """
        raw_list = [dict(item) for item in (facts or []) if isinstance(item, dict)]
        if not raw_list:
            return 0

        # forget 抑制（plans/0923-04 §4.3 阶段 2 补全）：投影前先读控制行。
        # 控制行**读不到（异常）时宁可不写**（fail-closed）——绝不让"已忘记的
        # 内容"借抑制表故障之机重新入库（§4.5 抑制过滤口径；写入侧与读侧的
        # fail 方向相反是故意的：读失败只少个性化，写失败会复活已删内容）。
        try:
            forget_rows = await self.list_forget_rows(user_id)
        except Exception:
            return 0

        existing = await self.list_facts(user_id)
        subkey_index: Dict[Tuple[str, str], str] = {}
        for row in existing:
            subkey = str(row.get("fact_subkey") or "")
            if not subkey:
                continue
            subkey_index.setdefault(
                (str(row.get("fact_key") or ""), str(row.get("value_text") or "")), subkey
            )

        rows: List[Dict[str, Any]] = []
        for raw in raw_list:
            fact_key = str(raw.get("fact_key") or "")
            if fact_key in MULTI_VALUE_FACT_KEYS and not str(raw.get("fact_subkey") or ""):
                raw["fact_subkey"] = subkey_index.get(
                    (fact_key, sanitize_text(raw.get("value_text"), 140)), ""
                )
            normalized = normalize_fact(raw)
            if normalized:
                rows.append(normalized)
        if not rows:
            return 0
        if forget_rows:
            rows = [r for r in rows if not is_fact_suppressed(forget_rows, r)]
            if not rows:
                return 0
        result = await _maybe_await(
            self._backend.upsert_learner_profile_facts(user_id, rows)
        )
        return int(result or 0)

    async def project_from_relationship(
        self, user_id: str, relationship: Optional[Dict[str, Any]], *, scope: str = "real"
    ) -> int:
        return await self.project_facts(
            user_id,
            facts_from_relationship_memory(user_id, relationship, scope=scope),
        )

    # ── forget（plans/0923-04 §4.3，阶段 2 补全）──────────────────────────────

    async def list_forget_rows(self, user_id: str) -> List[Dict[str, Any]]:
        """该用户全部 forget 控制行；后端缺方法 ⇒ ``[]``（从未用过 forget）。"""
        getter = getattr(self._backend, "list_learner_profile_forgets", None)
        if getter is None:
            return []
        rows = await _maybe_await(getter(str(user_id or "")))
        return [dict(r) for r in (rows or []) if isinstance(r, dict)]

    async def forget(self, user_id: str, fact_key: str, *,
                     fact_subkey: str = "", value_text: str = "") -> Dict[str, Any]:
        """忘记一个逻辑事实（§4.3）：物理删全部版本 + 同表留 forget 控制行。

        - 单值键是一整槽：`fact_subkey` 传不传都删整槽。
        - 多值键**必须**定位到一条边界/话题：传 `fact_subkey`，或传 `value_text`
          由现存行按值找回子键；两者都缺时**拒绝整键删除**（防止"因为同属
          avoid_pattern 就删掉其他边界"）。
        - 控制行只含键 / 子键 / 被删行的来源锚点，**不含原话与值**；被删证据
          本就无锚点时留一条无锚点控制行（整槽保守抑制的标记）。
        - 叙事引用同步清理：引用全部失效的叙事清成空行（text=""、引用=[]），
          部分失效的重写剩余 ID；渲染侧空 text 视为不存在。
        - 后端缺方法面 ⇒ 什么都不做（``reason`` 说明），旧路径照常工作。
        """
        key = str(fact_key or "").strip()
        if not key:
            return {"ok": False, "deleted_rows": 0, "forget_rows": 0,
                    "reason": "bad_fact_key"}
        delete = getattr(self._backend, "delete_learner_profile_facts", None)
        put_forget = getattr(self._backend, "put_learner_profile_forget", None)
        if delete is None or put_forget is None:
            return {"ok": False, "deleted_rows": 0, "forget_rows": 0,
                    "reason": "backend_unsupported"}
        uid = str(user_id or "")
        existing = await self.list_facts(uid)
        sub = str(fact_subkey or "").strip()
        if not sub and str(value_text or "").strip():
            want = sanitize_text(value_text, 140)
            for row in existing:
                if (str(row.get("fact_key") or "") == key
                        and str(row.get("value_text") or "") == want):
                    sub = str(row.get("fact_subkey") or "")
                    break
        if key in MULTI_VALUE_FACT_KEYS and not sub:
            return {"ok": False, "deleted_rows": 0, "forget_rows": 0,
                    "reason": "multi_value_needs_subkey"}
        scope_rows = [
            r for r in existing
            if str(r.get("fact_key") or "") == key
            and (key not in MULTI_VALUE_FACT_KEYS
                 or not sub
                 or str(r.get("fact_subkey") or "") == sub)
        ]
        if not scope_rows:
            # 没有可删的画像行：没有要挡的东西，不写控制行，如实返回。
            return {"ok": True, "deleted_rows": 0, "forget_rows": 0, "reason": ""}
        deleted = int(await _maybe_await(delete(uid, key, sub)) or 0)
        anchors = sorted({str(r.get("source_anchor_id") or "")
                          for r in scope_rows
                          if str(r.get("source_anchor_id") or "")})
        # 被删行的来源引用 ID（不含原话与值，方案 §4.3 允许留在控制行）：
        # 召回/事实卡按被忘引用过滤的数据基础。
        forgotten_refs: List[str] = []
        for row in scope_rows:
            for ref in (row.get("evidence_refs") or []):
                ref_text = str(ref or "").strip()
                if ref_text and ref_text not in forgotten_refs:
                    forgotten_refs.append(ref_text)
        forgotten_refs = forgotten_refs[:32]
        forget_rows = 0
        for anchor in (anchors or [""]):
            await _maybe_await(
                put_forget(uid, key, sub, anchor, evidence_refs=forgotten_refs))
            forget_rows += 1
        # 叙事引用清理：被删 fact_id 不允许再被任何叙事引用（§4.3）。
        removed_ids = {str(r.get("fact_id") or "") for r in scope_rows if r.get("fact_id")}
        try:
            narratives = await _maybe_await(
                self._backend.list_learner_profile_narratives(uid))
            for nar in (narratives or []):
                if not isinstance(nar, dict):
                    continue
                old_ids = [str(i) for i in (nar.get("supporting_fact_ids") or [])]
                keep = [i for i in old_ids if i not in removed_ids]
                if len(keep) == len(old_ids):
                    continue
                await _maybe_await(self._backend.put_learner_profile_narrative(
                    uid, str(nar.get("dimension") or ""),
                    "" if not keep else str(nar.get("text") or ""), keep))
        except Exception:
            pass
        return {"ok": True, "deleted_rows": deleted,
                "forget_rows": forget_rows, "reason": ""}

    async def list_facts(self, user_id: str) -> List[Dict[str, Any]]:
        rows = await _maybe_await(self._backend.list_learner_profile_facts(user_id))
        return [dict(r) for r in (rows or []) if isinstance(r, dict)]

    async def list_narratives(self, user_id: str) -> List[Dict[str, Any]]:
        """已沉淀的维度叙事（会话末冷路径写入；没跑过 ⇒ 空列表）。"""
        rows = await _maybe_await(
            self._backend.list_learner_profile_narratives(user_id))
        return [dict(r) for r in (rows or []) if isinstance(r, dict)]

    async def current_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """已裁决的当前事实（`contested` 已被 resolve_current 剔除）。"""
        current, _ = resolve_current(await self.list_facts(user_id))
        return current

    async def current_packet(self, user_id: str, *, max_lines: int = 8) -> str:
        return render_profile_packet(await self.current_facts(user_id), max_lines=max_lines)

    async def list_forgotten_refs(self, user_id: str) -> set:
        """全部 forget 控制行的来源引用 ID 并集（召回过滤用；不含原话与值）。

        后端缺方法 ⇒ 空集（旧后端没有抑制能力，也没有可过滤的东西）；
        方法存在但抛错 ⇒ 异常向上传播，由调用方决定保守方向（§4.5）。
        """
        list_forgets = getattr(self._backend, "list_learner_profile_forgets", None)
        if list_forgets is None:
            return set()
        rows = await _maybe_await(list_forgets(str(user_id or "")))
        refs: set = set()
        for row in rows or []:
            if not isinstance(row, dict) or str(row.get("status") or "") != "forget":
                continue
            for ref in (row.get("evidence_refs") or []):
                ref_text = str(ref or "").strip()
                if ref_text:
                    refs.add(ref_text)
        return refs

    async def save_narratives(self, user_id: str,
                              entries: List[Dict[str, Any]]) -> int:
        """按维度沉淀 L2 叙事（put 幂等覆盖同维度）；返回写入条数。"""
        written = 0
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            await _maybe_await(self._backend.put_learner_profile_narrative(
                str(user_id or ""),
                str(entry.get("dimension") or ""),
                str(entry.get("text") or ""),
                list(entry.get("supporting_fact_ids") or []),
            ))
            written += 1
        return written
