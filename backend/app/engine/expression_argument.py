# -*- coding: utf-8 -*-
"""练习盘的五维逻辑表达层（plans/0923-06 §4.2）。

为什么独立成模块：与 `engine/expression_scaffold.py` 同一条纪律——**判定、排序、
渲染都必须是纯函数、零 LLM、零 IO、可离线单测**的确定性层。本模块只回答三件事：

1. 导演给的 `argument_probe` 能不能信（结构白名单 + 证据必须是他本轮原话的子串）；
2. 这一轮**只提哪一个**焦点（当前段缺口优先，同档时用画像近期薄弱事实排序）；
3. 这个焦点在这一档帮助里**怎么说**（档 1 给一个可答的小线索，档 2 给一个结构动作，
   档 3 给一条原则，档 4 只提出重讲任务）。

五个维度是 [Toulmin 论证部件](https://www.cambridge.org/core/books/the-uses-of-argument/26CF801BC12004587B66778297D5567C)
在**口语反馈**里的局部映射：`backing` 归入“证据支撑／前提依据”，**不擅自加第六个用户评分轴**
（§4.2.1）。识别 → 一次受控重讲 → 在用户自选的新情境里运用，是借鉴
[Nation 2001 的语言学习过程](https://www.cambridge.org/core/books/learning-vocabulary-in-another-language/491314AA1B451AD04F3536000F1C9F0D)
来安排练习，**不声称该书提出了 Toulmin 口语量表**——所以本模块里没有任何“等级/分数”。

两套词表**故意不同名、不许混用**（§4.2.1 的“别把五个维度叫成 TAP 的五级”）：

  * 轮内 wire 词表（导演 JSON / probe）= `ok / weak / na`；
  * 画像落库词表（`error_pattern` 的 `evidence_kind`）= `strong / needs_work / not_observed`。

落库键是**现成的**画像③白名单键 `error_pattern`（`learner_profile.FACT_KEY_DIMENSION`），
维度放进 `fact_subkey = "argument_<dim>"`——**不新增第二权威、不改画像白名单**
（§4.4：公共函数签名与画像词表都不动）。读取侧只看 `evidence_kind` 里的状态，
不靠中文值文本反解，免得文案一改就读不出来。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ── 五维 ────────────────────────────────────────────────────────────────────
# 顺序即“下限优先”顺序：主张 → 证据 → 前提 → 限定 → 反例（§4.2.2 的递进）。
DIM_CLAIM = "claim"
DIM_EVIDENCE = "evidence"
DIM_WARRANT = "warrant"
DIM_QUALIFIER = "qualifier"
DIM_REBUTTAL = "rebuttal"

DIMENSIONS: Tuple[str, ...] = (
    DIM_CLAIM, DIM_EVIDENCE, DIM_WARRANT, DIM_QUALIFIER, DIM_REBUTTAL,
)

# 提示侧对模型的称呼（原样进 prompt，**不含 Toulmin 术语**：不要在教学口吻里
# 甩 claim/warrant 这种学术词，§4.2.1 的“不把术语当模板”）。
DIMENSION_LABELS: Dict[str, str] = {
    DIM_CLAIM: "主张是否清楚",
    DIM_EVIDENCE: "证据是否真支撑主张",
    DIM_WARRANT: "隐含前提是否说出",
    DIM_QUALIFIER: "是否限定／让步",
    DIM_REBUTTAL: "是否回应相关反例",
}

# “下限”两维：低级别先用这两维（§4.2.2）。它们稳了才轮到后三维。
FLOOR_DIMENSIONS: Tuple[str, ...] = (DIM_CLAIM, DIM_EVIDENCE)
ADVANCED_DIMENSIONS: Tuple[str, ...] = (DIM_WARRANT, DIM_QUALIFIER, DIM_REBUTTAL)

# ── 轮内 wire 词表 ──────────────────────────────────────────────────────────
STATUS_OK = "ok"
STATUS_WEAK = "weak"
STATUS_NA = "na"
STATUSES: Tuple[str, ...] = (STATUS_OK, STATUS_WEAK, STATUS_NA)

WIRE_TO_PROFILE: Dict[str, str] = {
    STATUS_OK: "strong",
    STATUS_WEAK: "needs_work",
    STATUS_NA: "not_observed",
}

# ── 画像落库 ────────────────────────────────────────────────────────────────
# 现成的画像③白名单键；**不新增键**。
FACT_KEY = "error_pattern"
SUBCONTEXT_PREFIX = "argument_"
EVIDENCE_KIND_PREFIX = "expr_argument_"
EVALUATOR_VERSION = "expr-arg-v1"
SOURCE_PIPELINE = "expression_argument"

MAX_EVIDENCE_CHARS = 140
MIN_EVIDENCE_CHARS = 4

# 低级别判定：只影响“高阶维要不要提”，不参与任何评分（§4.2.2）。
_LOW_LEVELS = frozenset({"a1", "a2", "beginner", "elementary", "basic", "入门", "初级"})

# 引文消毒：控制字符会破坏 prompt 结构；网页/ASR 文本里的换行同理。
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _clean_text(value: Any, limit: int = MAX_EVIDENCE_CHARS) -> str:
    """压空白 + 去控制字符 + 截断。原文是**数据**，不是指令（§4.3 末条）。"""
    text = _CTRL_RE.sub(" ", str(value or ""))
    return " ".join(text.split())[:limit]


def _norm_for_match(value: Any) -> str:
    return _clean_text(value, 4000).lower()


def normalize_status(value: Any) -> str:
    """任意输入 → 三个 wire 状态之一；认不出的一律 `na`（不猜成 weak）。"""
    text = str(value or "").strip().lower()
    if text in STATUSES:
        return text
    # 模型偶发输出画像词表或中文，做一次最小映射；仍认不出就 na。
    if text in {"strong", "稳", "好"}:
        return STATUS_OK
    if text in {"needs_work", "needswork", "待补", "弱"}:
        return STATUS_WEAK
    return STATUS_NA


def empty_probe() -> Dict[str, Any]:
    """无可用 probe 的规范形态（**不是**“五维都 ok”）。"""
    return {"ok": [], "weak": [], "na": list(DIMENSIONS), "focus_key": "", "evidence": ""}


def probe_from_statuses(statuses: Dict[str, Any]) -> Dict[str, Any]:
    """`{dim: status}` → 规范 probe（缺的维按 na）。"""
    out: Dict[str, Any] = {"ok": [], "weak": [], "na": [], "focus_key": "", "evidence": ""}
    for dim in DIMENSIONS:
        status = normalize_status((statuses or {}).get(dim))
        out[status].append(dim)
    return out


def _status_of(probe: Dict[str, Any], dim: str) -> str:
    for status in STATUSES:
        if dim in (probe.get(status) or []):
            return status
    return STATUS_NA


def sanitize_probe(raw: Any, transcript: str) -> Dict[str, Any]:
    """导演给的 `argument_probe` → 可信 probe；不可信返回 `{}`。

    两种 wire 形态都收（模型两种写法都出现过，收窄一种等于白丢判定）：

      * **逐维形态**（prompt 里要求的形态）：`{"claim": "weak", "evidence": "ok", ...}`
        + `evidence_quote` + `focus_key`；缺的维按 `na`。
      * **分桶形态**：`{"ok": [...], "weak": [...], "na": [...]}` + `evidence`。

    三档拒绝条件（§4.2.3：**证据不符或 JSON 截断即丢弃 probe**）：
      1. 结构不合法（不是 dict、状态值非法、同一维落在两个桶里）；
      2. 有 `weak` 时证据不是本轮原始转写的逐字子串（**伪证据一次都不采纳**，
         也不补第二次模型调用）；
      3. 有 `weak` 时证据过短（< `MIN_EVIDENCE_CHARS`）——“嗯/对”这种指不到
         东西的短引文不算证据。

    只有 `weak` 非空才叫“有焦点”；`focus_key` 这里只做合法性与“必须落在 weak
    里”的收口，排序仍由 `choose_focus` 决定。
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    bucket_form = any(key in raw for key in ("ok", "weak", "na"))
    if bucket_form:
        ok = _as_dim_list(raw.get("ok"))
        weak = _as_dim_list(raw.get("weak"))
        na = _as_dim_list(raw.get("na"))
        if ok is None or weak is None or na is None:
            return {}
        buckets = [set(ok), set(weak), set(na)]
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                if buckets[i] & buckets[j]:
                    return {}
        if not (ok or weak or na):
            return {}
        # ⚠ 只有分桶形态才能把 `evidence` 当引文别名——逐维形态里
        # `evidence` 是**维度名**，读成引文会把状态串 `"ok"` 当成他的原话。
        evidence = _clean_text(raw.get("evidence_quote") or raw.get("evidence"))
    else:
        dim_statuses: Dict[str, str] = {}
        for dim in DIMENSIONS:
            if dim in raw:
                dim_statuses[dim] = normalize_status(raw.get(dim))
        if not dim_statuses:
            return {}
        probe = probe_from_statuses(dim_statuses)
        ok, weak, na = probe["ok"], probe["weak"], probe["na"]
        evidence = _clean_text(raw.get("evidence_quote"))
    if weak:
        if len(evidence) < MIN_EVIDENCE_CHARS:
            return {}
        if _norm_for_match(evidence) not in _norm_for_match(transcript):
            return {}
    focus = str(raw.get("focus_key") or "").strip().lower()
    if focus not in DIMENSIONS or focus not in weak:
        focus = ""
    return {"ok": ok, "weak": weak, "na": na, "focus_key": focus, "evidence": evidence}


def _as_dim_list(value: Any) -> Optional[List[str]]:
    """把 probe 里的一个状态桶收成去重、保序的合法维度列表；不合法返回 None。"""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return None
    out: List[str] = []
    for item in value:
        dim = str(item or "").strip().lower()
        if not dim:
            continue
        if dim not in DIMENSIONS:
            return None
        if dim not in out:
            out.append(dim)
    return [d for d in DIMENSIONS if d in out]


def _profile_order(profile_weak: Sequence[str]) -> Dict[str, int]:
    order: Dict[str, int] = {}
    for index, dim in enumerate(profile_weak or ()):
        dim = str(dim or "").strip().lower()
        if dim in DIMENSIONS and dim not in order:
            order[dim] = index
    return order


def choose_focus(
    probe: Dict[str, Any],
    profile_weak: Iterable[str] = (),
) -> str:
    """本轮**只提一个**焦点（§4.2.2／§4.2.3）。没有可行动缺口 → 空串。

    排序（确定性，同输入必同输出）：
      1. 当前段的 `weak` **优先**——旧画像与本轮 `ok` 矛盾时以本轮为准；
      2. 先补下限：`claim` / `evidence` 里只要有一个 weak，就只在这两维里挑
         （低级别尤其如此；高阶项不许越过下限）；
      3. 同档内先看画像近期薄弱事实的顺序，再看固定的维度顺序；
      4. 下限已稳、且本轮确有空间（高阶维被标 weak）时才提示高阶维。

    `probe` 里导演建议的 `focus_key` **只作同档平手时的偏好**——它不能把焦点
    拽到本轮没有缺口、或还没轮到的高阶维上。

    ⚠ 这里**没有**"能力等级"参数：下限优先是无条件的（`FLOOR_DIMENSIONS` 永远
    排在 `ADVANCED_DIMENSIONS` 前面），加一个不影响行为的等级开关只会误导调用方。
    """
    weak = [d for d in DIMENSIONS if d in (probe or {}).get("weak", [])]
    if not weak:
        return ""
    floor = [d for d in weak if d in FLOOR_DIMENSIONS]
    advanced = [d for d in weak if d in ADVANCED_DIMENSIONS]
    pool = floor or advanced
    if not pool:
        return ""
    preferred = str((probe or {}).get("focus_key") or "").strip().lower()
    order = _profile_order(list(profile_weak or ()))
    ranked = sorted(
        pool,
        key=lambda dim: (
            0 if dim == preferred else 1,
            0 if dim in order else 1,
            order.get(dim, 0),
            DIMENSIONS.index(dim),
        ),
    )
    return ranked[0]


# ── 焦点话术（§4.2.6：档 1 线索 / 档 2 结构动作 / 档 3 原则 / 档 4 只给重讲任务）──
FOCUS_HINTS: Dict[str, Dict[int, str]] = {
    DIM_CLAIM: {
        1: "给他一个能一句话回答的线索，例如 “What's your main point in one sentence?”——"
           "只要他说得出那句，就算这一条过了。",
        2: "给他一个动作：把主张提到最前面，先一句话说结论，再讲后面的事。",
        3: "给他一条原则：别人要能一句话说出你的观点，才算讲清楚。",
    },
    DIM_EVIDENCE: {
        1: "给他一个能一句话回答的线索，例如 “Give me one example that made you think so.”——"
           "一个具体例子就够，不要多要。",
        2: "给他一个动作：先讲一个具体的事（发生了什么的哪一次），再把它连回你的看法。",
        3: "给他一条原则：例子要能直接支持你的看法，别人听完得觉得“所以你这样想是有道理的”。",
    },
    DIM_WARRANT: {
        1: "给他一个能一句话回答的线索，例如 “Why does that example make your point true?”——"
           "让他自己把中间那一步补出来。",
        2: "给他一个动作：在例子和结论之间加一句“所以……”，把没说的那步说出来。",
        3: "给他一条原则：从例子到结论中间总有一条没说的道理，把它说出来，别人就不会替你猜。",
    },
    DIM_QUALIFIER: {
        1: "给他一个能一句话回答的线索，例如 “Is that always true? Any case where it isn't?”",
        2: "给他一个动作：把绝对的说法加一个范围，例如 usually / in most cases / for me。",
        3: "给他一条原则：把话说满容易被一个例外推翻，先说清适用范围反而更有说服力。",
    },
    DIM_REBUTTAL: {
        1: "给他一个能一句话回答的线索，例如 “What would someone who disagrees say?”",
        2: "给他一个动作：先说一句别人可能的反对，再回应它。",
        3: "给他一条原则：主动提一句反过来的情况并回应，比自己只说一面更可信。",
    },
}

# 档 4 对所有维度同一句：只提出重讲任务，不再给任何提示。
FOCUS_HINT_TIER4 = "不要再给任何提示或例子——只提出同话题重讲的小任务，让他自己组织。"

_DIM_ZH = {
    DIM_CLAIM: "主张",
    DIM_EVIDENCE: "证据",
    DIM_WARRANT: "前提",
    DIM_QUALIFIER: "限定",
    DIM_REBUTTAL: "反例",
}

#: 轮外（会话末）评估用的五维判读口径。与轮内 probe 的口径**同义**，只是措辞
#: 面向"只读一段文本的评审"；单列成常量是为了让两处口径只有一个措辞源。
EVAL_RUBRIC = (
    "claim（主张是否清楚）：他有没有一句说得清的看法；"
    "evidence（证据是否真支撑主张）：他给的例子或事实是否真的支撑那个看法；"
    "warrant（隐含前提是否说出）：从例子到结论之间那条没明说的道理有没有说出来；"
    "qualifier（是否限定／让步）：有没有对绝对的说法做限定或让步；"
    "rebuttal（是否回应相关反例）：有没有回应一个相关的反例。"
)


def focus_hint(focus_key: str, tier: Any) -> str:
    """焦点 + 帮助档位 → 一条提示（§4.2.6）。焦点不合法则返回空串。"""
    dim = str(focus_key or "").strip().lower()
    if dim not in DIMENSIONS:
        return ""
    try:
        tier_value = int(tier)
    except (TypeError, ValueError):
        tier_value = 1
    if tier_value >= 4:
        return FOCUS_HINT_TIER4
    tier_value = max(1, tier_value)
    return FOCUS_HINTS[dim].get(tier_value) or FOCUS_HINTS[dim][1]


def render_focus_section(
    focus_key: str, tier: Any = 1, *, from_probe: bool = True,
) -> str:
    """逐轮动态焦点段（挂 `_expression_practice_section` 末尾）。

    **只在练习盘生成**（调用方负责非练习盘返回空串 ⇒ prompt 逐字不变）。
    三种形态：
      * 有经验证的 probe 且有焦点 → 指名这一维 + 该档的那条提示；
      * 有 probe 但本轮没有可行动缺口 → 明确“先问一个澄清，不要虚构诊断”；
      * 没有 probe（导演超时／坏 JSON／伪证据）→ 让现役回复模型依本轮原话
        自选一个最基础的主张／证据焦点，并**明确它不得进画像**（§4.2.3）。
    """
    dim = str(focus_key or "").strip().lower()
    if dim in DIMENSIONS:
        return (
            "[Expression Focus] 本轮**只提这一个**焦点：" + DIMENSION_LABELS[dim]
            + "（这是五项里的一个，不要列成清单、不要报总分、不要套总分总）。"
            + focus_hint(dim, tier)
            + " 严禁向用户直接使用 claim、warrant、qualifier、rebuttal 等学术术语，"
            "一律用日常英语（如 limit, condition, not always true）。"
            + " 反馈按三步走：先指出他刚说的**具体内容**（可引用他自己那句），"
            "再给这一个提升动作，最后请他**在同一个话题上重讲一遍**。"
        )
    if from_probe:
        return (
            "[Expression Focus] 本轮**没有**可以指认的焦点（他没有明显缺口，"
            "或话太短／转写不清）。**不要虚构诊断、不要凑一条意见出来**："
            "先问一个他能一句话回答的澄清，或顺着他的内容聊下去，等他讲得足够再给反馈。"
        )
    return (
        "[Expression Focus] 本轮没有可用的五维探查结果。**依他本轮原话**自选一个最基础的焦点"
        "（优先“看法清不清楚”，其次“例子支不支持这个看法”），**只提一个**，"
        "按三步反馈：指出具体内容 → 一个提升动作 → 请他在同话题上重讲一遍。"
        "这次的判断**只是本轮的临时读法，不得写成他的长期弱项**。"
    )


# ── 画像读写（只走 ProfileMemory 一个口）────────────────────────────────────


def weak_dims_from_facts(facts: Iterable[Dict[str, Any]]) -> List[str]:
    """画像事实 → 近期薄弱维度（**按发生时间由近到远**）。

    只认本模块写下的行：`fact_key == error_pattern` 且 `fact_subkey` 以
    `argument_` 开头且 `evidence_kind == expr_argument_needs_work`。
    值文本一律不看（文案会改），所以这里**不可能**被别的错误行串进来。
    """
    rows: List[Tuple[str, str]] = []
    for row in facts or ():
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") in {"contested", "forget", "legacy_unattributed"}:
            continue
        if str(row.get("fact_key") or "") != FACT_KEY:
            continue
        subkey = str(row.get("fact_subkey") or "")
        if not subkey.startswith(SUBCONTEXT_PREFIX):
            continue
        if str(row.get("evidence_kind") or "") != EVIDENCE_KIND_PREFIX + "needs_work":
            continue
        dim = subkey[len(SUBCONTEXT_PREFIX):].strip().lower()
        if dim not in DIMENSIONS:
            continue
        stamp = str(row.get("recorded_at") or row.get("observed_at") or "")
        rows.append((stamp, dim))
    ordered: List[str] = []
    for _, dim in sorted(rows, key=lambda item: item[0], reverse=True):
        if dim not in ordered:
            ordered.append(dim)
    return ordered


def is_expression_argument_fact(row: Any) -> bool:
    """这一行是不是**本模块（练习盘五维）自己写下**的画像事实。

    只服务一件事：把五维行挡在**练习盘以外**各盘的提示词之外 —— 10-16 用户
    拍板「闲聊盘的 Alice 不需要知道用户的练习水平」。判据用本模块自己的落库戳
    （`evidence_kind` 以 `expr_argument_` 开头），不拿值文本反解；同时认同一批
    行的另一道戳（`fact_key == error_pattern` 且子键以 `argument_` 开头），
    任一条成立即成。纯读行内已有字段：**不查库、不写库、不改任何状态**。
    """
    if not isinstance(row, dict):
        return False
    if str(row.get("evidence_kind") or "").startswith(EVIDENCE_KIND_PREFIX):
        return True
    if str(row.get("fact_key") or "") != FACT_KEY:
        return False
    return str(row.get("fact_subkey") or "").startswith(SUBCONTEXT_PREFIX)


# ── 按需召回：用户主动问起自己的练习表现（10-16）─────────────────────────────
# 背景：闲聊盘平时**不该**看到五维判定（逐轮画像 packet 已按盘摘掉）；但用户
# 拍板「Alice 有这个能力去读、什么时候用看它自己」⇒ 他主动问起时应当读得到。
# 下列两组词是这条路**唯一**的触发口。

# 按需召回最多报几维（只报最该被记起的那几处，不做清单、不报总分）。
MAX_RECALL_DIMS = 3

# 领域词：这一段话在说「练表达」这码事。裸「练」也要收——「我这样练下去有没有
# 改善」这种说法不带「练习」两个字。
_RECALL_PRACTICE_WORDS = (
    "练习", "练得", "练的", "练", "训练", "口语", "表达", "逻辑",
    "论证", "论据", "复述", "重讲", "复盘",
)
# 询问词：这一段话在**问**（或在求复盘），而不是在陈述。
_RECALL_ASK_WORDS = (
    "怎么样", "如何", "哪里", "哪儿", "什么", "哪些", "有没有", "进步",
    "不行", "弱", "差", "问题", "改善", "评价", "表现", "水平", "结果",
    "复盘",
)

_RECALL_EN_PRACTICE_RE = re.compile(
    r"\bmy\s+(?:practice|progress|expression|argument|speaking|weakness"
    r"|weaknesses|logical)\b|\bhow\s+am\s+i\s+doing\b|\bam\s+i\s+improving\b",
    re.IGNORECASE,
)
_RECALL_EN_ASK_RE = re.compile(
    r"\bhow\b|\bwhat\b|\bwhere\b|\bweak\b|\bimprove|\bprogress\b|\bdoing\b"
    r"|\bneed\b|\bshould\b",
    re.IGNORECASE,
)


def wants_expression_recall(text: Any) -> bool:
    """这一段话是不是**在问自己练表达练得怎么样**（闲聊盘按需召回的唯一触发口）。

    - 只回答「值不值得去查一次库」：**不查库、不改状态、不抛异常**。
    - 两组词**同时**命中才算（领域词 × 询问词），免得「我今天去练习了吉他」
      这类自述被当成提问。
    - 故意宽松：命中一次的代价只是一次有界读 ＋ 一行带边界说明的短句；漏判的
      代价是她手里明明有记录却答不上来。故**宁可偶有误触发**。
    """
    raw = str(text or "")
    if not raw.strip():
        return False
    lowered = raw.casefold()
    if _RECALL_EN_PRACTICE_RE.search(lowered) and _RECALL_EN_ASK_RE.search(lowered):
        return True
    if not any(word in raw for word in _RECALL_PRACTICE_WORDS):
        return False
    return any(word in raw for word in _RECALL_ASK_WORDS)


def recall_line_from_facts(facts: Iterable[Dict[str, Any]]) -> str:
    """「他在练习盘还没做到的地方」→ 一条带边界说明的按需召回短句（10-16）。

    只在闲聊盘、且用户主动问起练习表现时用（见 `wants_expression_recall`）；
    平时**绝不注入** —— 那等于系统替她使用，用户明确说过不要。
    返回空串 = 没有记录可说 ⇒ 调用方什么都不加，绝不留空行。
    """
    dims = weak_dims_from_facts(facts)
    if not dims:
        return ""
    items = "、".join(DIMENSION_LABELS.get(dim, dim) for dim in dims[:MAX_RECALL_DIMS])
    return (
        "【他的练习盘记录·仅供参考】最近练表达时，这几处他还没做到：" + items
        + "（由近到远）。这是**别的盘**的数据、不是本盘任务：只有他主动问起练习"
        "情况时才用得上，不要主动提起、不要切换成教学口吻。"
    )


def sanitize_verdicts(raw: Any, transcript: str) -> Dict[str, Any]:
    """轮外五维判定 → 只留**有逐字引文**的维度（§4.2.4）。

    与轮内 probe 同一条纪律：指不到他原话的判定一律不采信。所以返回值里
    **只会有** `ok` / `weak` 且引文在本轮转写里逐字存在的维度；`na`、缺引文、
    引文对不上的维度**根本不出现**（调用方据此"该维不写"，而不是写一条没证据的
    弱项——§5 的"缺证据不生成高置信弱项"）。

    `raw` 形如 `{dim: {"status": "ok|weak|na", "quote": "...", "confidence": 0.6}}`，
    也接受裸状态字符串（无引文 ⇒ 不返回）。
    """
    out: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    haystack = _norm_for_match(transcript)
    for dim in DIMENSIONS:
        item = raw.get(dim)
        if isinstance(item, dict):
            status = normalize_status(item.get("status"))
            quote = _clean_text(item.get("quote"))
            confidence = item.get("confidence")
        else:
            # 裸的 `{"claim": "weak"}`：没有引文，不能采信
            continue
        if status not in (STATUS_OK, STATUS_WEAK):
            continue
        if len(quote) < MIN_EVIDENCE_CHARS or _norm_for_match(quote) not in haystack:
            continue
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.6 if status == STATUS_WEAK else 0.7
        out[dim] = {"status": status, "quote": quote, "confidence": confidence}
    return out


def facts_from_verdicts(
    verdicts: Dict[str, Any],
    *,
    user_id: str,
    conv_id: str = "",
    session_id: str = "",
    observed_at: Optional[str] = None,
    source: str = SOURCE_PIPELINE,
) -> List[Dict[str, Any]]:
    """五维判定 → 画像事实行（交给 `ProfileMemory.project_facts`）。

    只写两态：`strong` / `needs_work`；**`not_observed` 一行都不写**
    （§4.2.4／§5：“不得把未观察项记为失败”）。没有对应原句引用的判定也不写
    （“缺证据不生成高置信弱项”）。

    `verdicts` 形如 `{dim: {"status": "ok|weak|na", "quote": "他的原话", "confidence": 0.6}}`；
    也接受 `{dim: "weak"}` 这种裸状态（无引文 ⇒ 不写）。
    """
    out: List[Dict[str, Any]] = []
    for dim in DIMENSIONS:
        raw = (verdicts or {}).get(dim)
        if isinstance(raw, dict):
            status = normalize_status(raw.get("status"))
            quote = _clean_text(raw.get("quote"))
            confidence = raw.get("confidence")
        else:
            status = normalize_status(raw)
            quote = ""
            confidence = None
        if status == STATUS_NA or not quote:
            continue
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.6 if status == STATUS_WEAK else 0.7
        # 来源锚点/occurrence **必须显式给**：`normalize_fact` 的默认实现只用
        # 「维度 + 证据引用 + 子键」，同一维在两盘、两个会话里会算出**同一个**
        # occurrence ⇒ 唯一索引把它们当同一条重复投影而丢掉（实测：第二次会话的
        # 判定写不进去，"最新状态"永远是第一盘那次的）。锚点按**维度**稳定
        # （忘记某一维时能挡住旧来源重扫），occurrence 再加会话区分。
        occurrence_seed = str(session_id or conv_id or observed_at or "").strip()
        row: Dict[str, Any] = {
            "fact_key": FACT_KEY,
            "fact_subkey": SUBCONTEXT_PREFIX + dim,
            "value_text": "逻辑表达·" + _DIM_ZH[dim] + "=" + WIRE_TO_PROFILE[status],
            "evidence_kind": EVIDENCE_KIND_PREFIX + WIRE_TO_PROFILE[status],
            "evidence_quote": quote,
            "confidence": confidence,
            "source_pipeline": source,
            "source_anchor_id": SOURCE_PIPELINE + ":" + dim,
            "source_occurrence_id": SOURCE_PIPELINE + ":" + dim + ":" + occurrence_seed,
            "extractor_version": EVALUATOR_VERSION,
            "evidence_refs": [dim],
        }
        if user_id:
            row["user_id"] = str(user_id)
        if conv_id:
            row["source_conv_id"] = str(conv_id)
        if session_id:
            row["source_session_id"] = str(session_id)
        if observed_at:
            row["observed_at"] = str(observed_at)
        out.append(row)
    return out


def session_verdicts_from_transcript(
    transcript: str,
    *,
    min_chars: int = 40,
    min_sentences: int = 2,
) -> Dict[str, Any]:
    """会话末冷评估的**准入门槛**（不含任何判定，判定由轮外模型给）。

    “不足一段或证据不清则不写”（§4.2.4）：太短、或没有可辨的反例/限定机会时，
    高阶三维直接标 `not_observed`——**不算弱项**。这里只做确定性门槛，
    高阶维度“有没有机会”由轮外模型给 `na` 决定，本函数不替它猜。
    """
    text = _clean_text(transcript, 4000)
    if len(text) < max(1, int(min_chars)):
        return {}
    sentences = len([s for s in re.split(r"[.!?。！？;；\n]+", text) if s.strip()])
    if sentences < max(1, int(min_sentences)):
        return {}
    return {"transcript": text, "sentences": sentences}
