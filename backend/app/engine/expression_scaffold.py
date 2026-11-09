# -*- coding: utf-8 -*-
"""练习盘的支架递减层（plans/0919-02 §4.5）。

为什么独立成模块：档位判定必须是**纯函数、零 LLM 调用、可离线单测**的确定性层
（§4.5「档位 = f(该维度的掌握度)」，项目原则：代码算出来才权威）。本模块只管
"该给什么帮助 / 什么时候升降"，读写落库在 `agent/tutor_agent.py`（l2 只在那边）。

术语（**看数先看这句**）：`help_tier` 是"帮助由多到少的序号"——
  * 1 = 帮助最多（给三步骨架 + 几个固定连接词）
  * 4 = 帮助最少（完全不给提示）
所以"帮助减少一档" = `help_tier` **+1**。§4.5 表里的"档 1/2/3/4"就是本字段。
§5 验收那句"连讲两次达标后档位下降一档（帮助减少）"指的是**帮助**下降，
即本字段数值 +1。

升降判据（§4.5 定案，实现原样照做）：
  * 同一维度**连续两次达标** → 帮助减少一档（`help_tier + 1`，封顶 4）；
  * **连续两次不达标** → 帮助增加一档（`help_tier - 1`，封底 1）；
  * 初始档 = 1（帮助最多）。
  * **数据不可用既不算达标也不算不达标**（`met=None`）：连胜/连败都不动——与
    §4.2「拿不到就标 unavailable，不伪造」同一条纪律，不许把"量不到"读成"没做到"。

达标由谁判：**代码**。判据 = **密度可算 且 密度上升 且 details 不下降**，
取数来自现役 F14 对比（`engine/chunk_evidence.compare_expression_metrics` 的
`verdict.available` + `delta.density` + `delta.details`）。
模型不参与档位判定，也不许在回复里报档位数字。

⚠ **2026-09-20 判据修订**（原判据的缺陷与理由见 `REASON_DENSITY_UP` 上方的注释块，
方案档 `plans/0920-02-支架达标判据修订.md`）：原判据直接取
`verdict.concise_progress`，而盘内首遍提示明确邀请用户"再多讲点细节"——
**两条口径互相矛盾，方向是反的**。修订只去掉"句数必须下降"这一个额外条件，
比率制与"details 不下降"两条护栏原样保留。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

# `learning_item_states` 的 item 名（§4.5 落点推荐：把表达维度作为 item，
# 间隔/门禁/成功计数都现成）。一个用户一行，档位与连胜/连败存在 last_evidence 里。
SCAFFOLD_ITEM_TYPE = "expression_practice"

SCAFFOLD_MIN_TIER = 1
SCAFFOLD_MAX_TIER = 4

# ── 档 1 的固定帮助（2026-09-20 定案，plans/0920-02 §8.6）────────────────────
# 为什么**由代码定死**、不给模型发挥空间（两条独立理由，都实测过）：
#
# ① **会喂饱判据**（硬伤）。原档 1 指示是"给 3-5 个能直接用上的关键词或词组"，
#    样例写着 `street food / a queue / ... and then / on my way back` —— 内容词与
#    衔接词混在一起。实测（`bugreports/probe_opening_keywords_metrics_0920.py`）：
#    开场那 5 个词**全部进度量**（`on my way`/`ran into`/`a bit` 各 +1 `details`，
#    `and then`/`in the end` 各 +1 衔接词），同一件事两遍的密度 **2.00 → 3.00**，
#    送进判据得 `met=True / density_up` ⇒ **"达标"里混进了"他把我们给的词念了一遍"**。
#    而 `details` 是达标的**分子**、密度是**判据本体** ⇒ 档 1 等于在给判据送分，
#    与"逻辑表达为主、语言水平是附加"的定位方向相反（语言层帮助推着逻辑层判据走）。
# ② **随机**（用户 2026-09-20 提出）。让模型每次现编，用户看到的东西每次都不一样，
#    无法形成"这一档就是这几个词"的稳定预期，也没法静态验证它不会塞内容词。
#    ⚠ 关键：一旦收成"只给连接词"，**连接词与话题无关** ⇒ "贴他讲的内容"这个需求
#    在档 1 消失 ⇒ 由代码给固定词表既更确定、也不损失任何贴合度。
#
# 所以档 1 只发**不进 `details` 分子**的东西：
#   · 骨架 = 中文元提示（与档 2 的样例同一风格；要"用上去"的才是目标语言）
#   · 连接词 = 目标语言，且**每一个都在 `chunk_evidence` 的衔接词表里** ⇒
#     衔接词只作观测量、不进 `details`/密度 ⇒ 发出去也不会给判据送分。
#
# ⚠ **选型不是语感，是重算出来的**（第一版我按语感挑了 `in the end`，当场被测试打红）：
#   `in the end` / `by the way` 送分（`end`/`way` 是内容词）、`later`/`next` 送分
#   （在 `_TIME_TOKENS` 里）、`second` 送分（在 `_NUMBER_TOKENS` 里）、`as soon as` 送分。
#   35 个候选里干净 29 个，现役 4 个取自其中：3 个时序 + 1 个因果，正好配那三步骨架。
# 契约测试 `test_expression_contract.ExpressionScaffoldTierOneTests` 逐个重算钉住这点
# （含"有人往词表里塞内容词/时间词"这一步）。
SCAFFOLD_TIER1_FRAME = "先说什么时候/在哪 → 再说发生了什么 → 最后怎么结束、你什么感受"
SCAFFOLD_TIER1_CONNECTIVES = ("at first", "and then", "so", "finally")

# 四档话术（§4.5 表：完全不给提示 ← 给原则 ← 给结构 ← 给骨架+连接词）。
# 档 1 那两样**由代码定死、模型只许原样搬运**，理由见 `SCAFFOLD_TIER1_CONNECTIVES`。
# `instruction` 是"这一档你只给什么"，会原样进 prompt；`example` 是 §4.5 的具体样例。
SCAFFOLD_TIERS: Dict[int, Dict[str, str]] = {
    1: {
        "label": "骨架 + 连接词",
        "summary": "帮助最多：给一个三步骨架，再加几个连接词",
        "instruction": (
            "把下面两样**原样照抄**在任务后面：① 结构提示（中文，一个箭头一步）；"
            "② 连接词（目标语言，**只给连接词**）。不许增删、不许改写、"
            "不许给任何名词/动词/内容词、不许给例句、不许讲解"
        ),
        "example": (
            "结构提示：" + SCAFFOLD_TIER1_FRAME + "；连接词："
            + " / ".join(SCAFFOLD_TIER1_CONNECTIVES)
        ),
    },
    2: {
        "label": "结构",
        "summary": "给一个可套用的结构",
        "instruction": (
            "给一个可套用的结构（一句话说清先后顺序），不要给具体词、不要给原则"
        ),
        "example": '比如："先说结论，再说怎么发生的"',
    },
    3: {
        "label": "原则",
        "summary": "给一条原则",
        "instruction": (
            "给一条原则（一句话，让他自己组织内容），不要给词、不要给结构模板"
        ),
        "example": '比如："讲之前先想一句最想让我记住的话"',
    },
    4: {
        "label": "无提示",
        "summary": "帮助最少：完全不给提示",
        "instruction": (
            "完全不给任何提示——只给任务本身，让他自己组织；"
            "不要给词、不要给结构、不要给原则"
        ),
        "example": "（这一档只有任务，没有帮助）",
    },
}

# 连续几次算"连续"（§4.5：两次）。
TIER_DOWN_STREAK = 2   # 连续达标两次 → 帮助减少一档
TIER_UP_STREAK = 2     # 连续不达标两次 → 帮助增加一档

# 不可用原因（进 last_evidence，方便复盘"为什么这轮没动档"）。
REASON_NO_COMPARE = "no_compare"
REASON_VERDICT_UNAVAILABLE = "verdict_unavailable"
# 达标/不达标的**原因串**（2026-09-20 判据修订后由本模块自己产出，不再借用
# F14 那套 verdict reason —— 那套的语义是"简洁进步"，与现在的判据不同名了）。
REASON_DENSITY_UP = "density_up"
REASON_DENSITY_FLAT = "density_flat"
REASON_DENSITY_DOWN = "density_down"
REASON_DETAILS_DROPPED = "details_dropped"

# 判据层真正读的两维（`met_and_reason_from_compare` 的输入）：**只有改这两维才会推动档位**。
# 键 = `compare['delta']` 的字段名；值 = 提示侧对模型的称呼（原样出现在 prompt 里）。
# 其余维度只写进提示给模型看（衔接词 / 背景要素 / 重点前置）——可以提，但**代码不判它们、
# 档位也不因它们动**。⇒ 练习盘"只挑一个点"时**优先挑这两维**（他重讲后代码能当场作证）；
# 提示侧那段优先级话术与本表对账（跨文件静态契约），别在别处再数一遍键名。
TIER_DRIVING_DIM_LABELS: Dict[str, str] = {
    "details": "details（细节）",
    "density": "密度",
}

# ══════════ 达标判据（2026-09-20 修订，plans/0920-02）══════════════════════
# **修订前的缺陷**（有据可查，不是口味问题）：
#   原判据 = `verdict.concise_progress` = `sentences_down and details_not_down`
#   （还要求两边密度都可算）。但盘内首遍提示**明确邀请用户"再多讲点细节"**
#   ——`agent/tutor_agent.py` 首遍 `_fb_hint` 原文就有
#   `'Tell me it again — this time, add more details.'`。于是：
#     · 用户**照邀请做** → details 变多 → 句数多半不降 →
#       `sentences_down=False` → 判"不达标" → 连续两次不达标**增加**帮助；
#     · 用户反向**压缩句数** → 实测样本里 details 也一起降（§9.3 英文组
#       5 句/26 细节 → 2 句/22 细节）→ `details_dropped` → 同样"不达标"。
#   ⇒ 两个自然方向都不通，"连续两次达标"这个降档条件**实质不可达**，档位冻结。
#
# **修订后**：`met = 密度可算 and 密度上升 and details 不下降`。
#   · 密度 = `details ÷ sentences`（比率制不变）⇒ 仍然**不奖励啰嗦**；
#   · 去掉"句数必须下降"⇒ 与"再多讲点细节"这条邀请不再打架
#     （5句/26细节 → 6句/40细节 密度 5.20→6.67，判达标）；
#   · 保留"details 不下降"⇒ 仍**不奖励"讲少了"**，对齐 §9.3 那条判据的价值观；
#   · 不可用（跨口径 / 密度算不出）仍为 `None` ⇒ 不推进档位、不伪造。
# 边界：三态语义与 §4.5 定案完全一致，改的只是"什么算达标"这一条判据，
# 升降判据（连续两次 / 封顶封底 / 不可用不动）一个字没动。


def clamp_tier(value: Any) -> int:
    """把任意输入夹到 [1, 4]；非数字/缺失 → 初始档 1。"""
    try:
        tier = int(value)
    except (TypeError, ValueError):
        return SCAFFOLD_MIN_TIER
    return max(SCAFFOLD_MIN_TIER, min(SCAFFOLD_MAX_TIER, tier))


def tier_row(tier: Any) -> Dict[str, str]:
    """取某一档的话术行（永远有值，越界自动夹档）。"""
    return SCAFFOLD_TIERS[clamp_tier(tier)]


def initial_state() -> Dict[str, Any]:
    """初始档位状态（§4.5：初始档 1）。"""
    return {
        "help_tier": SCAFFOLD_MIN_TIER,
        "met_streak": 0,
        "miss_streak": 0,
        "last_reason": "",
        "observations": 0,
    }


def _as_dict(value: Any) -> Dict[str, Any]:
    """last_evidence 的两种后端形态（sqlite 存 TEXT、pg 存 jsonb）都能吃。"""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, bytes)) and value:
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def state_from_row(row: Any) -> Dict[str, Any]:
    """`learning_item_states` 行 → 档位状态（缺失/坏数据一律退回初始档）。"""
    row = row if isinstance(row, dict) else {}
    evidence = _as_dict(row.get("last_evidence"))
    state = initial_state()
    if "help_tier" in evidence:
        state["help_tier"] = clamp_tier(evidence.get("help_tier"))
    for key in ("met_streak", "miss_streak", "observations"):
        try:
            state[key] = max(0, int(evidence.get(key) or 0))
        except (TypeError, ValueError):
            state[key] = 0
    state["last_reason"] = str(evidence.get("last_reason") or "")
    return state


def evidence_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """档位状态 → 存库的 last_evidence（只放本模块的键，不污染别的字段）。"""
    state = state or {}
    return {
        "help_tier": clamp_tier(state.get("help_tier")),
        "met_streak": max(0, int(state.get("met_streak") or 0)),
        "miss_streak": max(0, int(state.get("miss_streak") or 0)),
        "last_reason": str(state.get("last_reason") or ""),
        "observations": max(0, int(state.get("observations") or 0)),
        "kind": SCAFFOLD_ITEM_TYPE,
    }


def decide_tier(state: Dict[str, Any], met: Optional[bool],
                reason: str = "") -> Dict[str, Any]:
    """按一次"再讲一遍"的结果推进档位（§4.5 升降判据）。

    `met=True/False` = 代码判定的达标/不达标；`met=None` = **数据不可用**，
    既不推进也不清连胜（见模块 docstring 第三条）。

    返回**新状态**（不就地改入参），`last_reason` 记本次判定的原因字符串
    （`concise_progress` / `details_dropped` / ... 或本模块的两个不可用原因）。
    """
    base = state_from_row({"last_evidence": evidence_from_state(state or {})})
    tier = base["help_tier"]
    met_streak = base["met_streak"]
    miss_streak = base["miss_streak"]
    out = {
        "help_tier": tier, "met_streak": met_streak, "miss_streak": miss_streak,
        "last_reason": str(reason or ""),
        "observations": base["observations"] + (1 if met is not None else 0),
    }
    if met is None:
        return out
    if met:
        met_streak += 1
        miss_streak = 0
        if met_streak >= TIER_DOWN_STREAK and tier < SCAFFOLD_MAX_TIER:
            tier += 1
            met_streak = 0      # 已降档，连胜从零重算（封顶时也不许攒着）
        elif met_streak >= TIER_DOWN_STREAK:
            met_streak = 0      # 已在最末档：帮助没法再少，连胜清零免得反复触发
    else:
        miss_streak += 1
        met_streak = 0
        if miss_streak >= TIER_UP_STREAK and tier > SCAFFOLD_MIN_TIER:
            tier -= 1
            miss_streak = 0
        elif miss_streak >= TIER_UP_STREAK:
            miss_streak = 0     # 已在第一档：帮助没法再多，连败清零
    out.update({"help_tier": tier, "met_streak": met_streak, "miss_streak": miss_streak})
    return out


def met_and_reason_from_compare(compare: Any) -> Tuple[Optional[bool], str]:
    """现役 F14 对比结果 → 达标三态 **+ 原因串**（2026-09-20 修订版判据）。

    判据（判决理由见 `REASON_DENSITY_UP` 上方那段注释）：
        `met = 密度可算 and 密度上升 and details 不下降`

    `True`  = 达标（`REASON_DENSITY_UP`）；
    `False` = 有可用判定但没达标（`REASON_DENSITY_FLAT` / `REASON_DENSITY_DOWN` /
              `REASON_DETAILS_DROPPED`）；
    `None`  = 密度不可用/跨口径 → **不算达标也不算不达标**（不推进档位），
              原因串透传 F14 那边的 `caliber_mismatch` / `density_unavailable` 等，
              便于复盘"这轮为什么没动档"。

    取数全部来自 `compare`（`verdict.available` / `delta.density` / `delta.details`），
    本函数**纯函数、零 IO、零模型**，可离线单测。
    """
    if not isinstance(compare, dict):
        return None, REASON_NO_COMPARE
    verdict = compare.get("verdict") or {}
    if not isinstance(verdict, dict) or not verdict.get("available"):
        return None, verdict_reason_of(compare) or REASON_VERDICT_UNAVAILABLE
    delta = compare.get("delta") or {}
    if not isinstance(delta, dict):
        return None, REASON_VERDICT_UNAVAILABLE
    density_delta = delta.get("density")
    if density_delta is None:
        # `verdict.available=True` 时 F14 一定会给密度差；真拿不到就如实按不可用，
        # **不猜**（拿"句数下降"凑一个判定出来就是伪造）。
        return None, REASON_VERDICT_UNAVAILABLE
    try:
        density_delta = float(density_delta)
    except (TypeError, ValueError):
        return None, REASON_VERDICT_UNAVAILABLE
    try:
        details_delta = int(delta.get("details") or 0)
    except (TypeError, ValueError):
        details_delta = 0
    # 先判"细节有没有丢"：密度上升但靠**少讲内容**换来的，不算达标（§9.3 的价值观）。
    if details_delta < 0:
        return False, REASON_DETAILS_DROPPED
    # 浮点噪声容差：density 是 details/sentences 的商，"持平"不该被判成上升/下降。
    if density_delta > 1e-9:
        return True, REASON_DENSITY_UP
    if density_delta < -1e-9:
        return False, REASON_DENSITY_DOWN
    return False, REASON_DENSITY_FLAT


def met_from_compare(compare: Any) -> Optional[bool]:
    """[兼容壳] 只取达标三态，丢弃原因串。

    新代码请用 `met_and_reason_from_compare` —— 档位写库时要记 `last_reason`，
    只拿布尔值会让"为什么这轮没动档"查不出来。
    """
    return met_and_reason_from_compare(compare)[0]


def verdict_reason_of(compare: Any) -> str:
    """对比结果里的原因串（进 last_evidence，便于复盘）。"""
    if not isinstance(compare, dict):
        return REASON_NO_COMPARE
    verdict = compare.get("verdict") or {}
    reason = str((verdict or {}).get("reason") or "") if isinstance(verdict, dict) else ""
    return reason or REASON_VERDICT_UNAVAILABLE


def render_scaffold_section(state: Dict[str, Any]) -> str:
    """盘内合同里"当前档位"那一段（§4.4：开场给任务时携带当前档位）。

    非练习盘不调用本函数（调用方负责空串），故这里永远返回非空文本。
    """
    tier = clamp_tier((state or {}).get("help_tier"))
    row = tier_row(tier)
    return (
        "[Expression Scaffold] 当前支架档位 = 第 " + str(tier) + " 档（"
        + row["summary"] + "）。这一档你**只给这一种帮助**：" + row["instruction"]
        + "。" + row["example"] + "。"
        "档位由代码按他这两遍的达标情况算出（连续两次达标帮助减少一档、"
        "连续两次不达标帮助增加一档），**不得自行升降、也不要在回复里报档位数字**；"
        "给完这一档的帮助就停，不要叠加更多提示。"
    )


def render_opening_scaffold(state: Dict[str, Any]) -> str:
    """开场规则里"当前档位"那半句（main.py 的 otype='expression' 分支用）。

    §4.4：开场**给第一个任务**，并携带当前支架档位。开场走的是 main.py 自己那条
    生成链（不是 _build_control_prompt），所以这里另给一个短版本。
    """
    tier = clamp_tier((state or {}).get("help_tier"))
    row = tier_row(tier)
    return (
        "当前支架档位 = 第 " + str(tier) + " 档（" + row["summary"] + "）：给完任务后，"
        + row["instruction"] + "。" + row["example"] + "。"
        "档位由代码算出，不得自行升降、不要报档位数字。"
    )
