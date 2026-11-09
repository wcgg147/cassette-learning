# -*- coding: utf-8 -*-
"""练习盘的五维「盘内趋势」层（plans/0919-02 §4.7 能力画像的最小落地）。

为什么独立成模块：与 `engine/expression_argument.py`、`engine/expression_scaffold.py`
同一条纪律 —— 判定必须是**纯函数、零 LLM、零 IO、可离线单测**的确定性层。
本模块只回答两件事：

1. 这一轮的 `ctx.argument_focus` 里有没有**已验证**的五维读数；
2. 把这一盘已经攒下的若干条读数，折成**每一维「最近怎么走」**
   （连续同向几次 + 方向）。

## 为什么需要它（要解决的缺口）

`0923-06` 把五维证据写进了统一画像③，但读回来**只用于「同档平手时排序」**
（`expression_argument.choose_focus`），而且**只在会话末写一次**。于是：

* **盘内**：练十轮和练一轮，画像对练习的影响一样 —— 「帮你哪块」在盘内不变；
* 能变的只有「帮你多少」（支架档位，它逐轮存）。

本模块补的就是「帮你哪块」的**盘内累积**：读数逐轮攒、趋势由代码算。

## 术语与边界（三条纪律）

* **只收 `ok` / `weak`**：`na`（没观察到）一律不进统计 —— 与
  `expression_argument.facts_from_verdicts` 的「未观察项不记失败」同源。
  某一维中间断了一次，它的链就**断在那里**，不跳过、不猜。
* **读数只当「一张票」**：偶发一次口误不算趋势。必须**连续 `MIN_STREAK` 次同向**
  才被认作趋势。轮内读数只看当前一轮，比会话末那次（看整盘）糙，所以宁可要更长的
  连胜才认，免得误伤。
* **隐私**：本模块**不碰任何原话**。防「同一次讲述被重复计票」用的是
  「**相邻读数签名完全相同即折叠**」——不需要引文、不需要哈希、不落文本。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from engine.expression_argument import DIMENSIONS, DIMENSION_LABELS

#: 落在 `metering_events.metadata` 里的键。**必须**同时登记进
#: `memory/metering_contract.METADATA_FIELDS`，否则会被静默丢弃（0916-14 踩过）。
READING_KEY = "expr_arg"

STATUS_OK = "ok"
STATUS_WEAK = "weak"

#: 连续几次同向才算趋势。比支架档位的「连续两次」更保守（轮内读数更糙）。
MIN_STREAK = 3

#: 一句话里最多报几维（有界，照 §4.2.4 的预算纪律）。
MAX_REPORTED_DIMS = 2


def reading_from_focus(focus: Any) -> Dict[str, str]:
    """`ctx.argument_focus` → 一条读数 `{维度: ok|weak}`（只收这两态）。

    没有可用读数（非练习盘、无 probe、全 `na`）一律返回**空 dict** ——
    调用方据此「不记账」，而不是记一条空账。
    """
    if not isinstance(focus, dict):
        return {}
    dims = focus.get("dims")
    if not isinstance(dims, dict):
        return {}
    out: Dict[str, str] = {}
    for dim in DIMENSIONS:
        value = str(dims.get(dim) or "").strip().lower()
        if value in (STATUS_OK, STATUS_WEAK):
            out[dim] = value
    return out


def _metadata_of(event: Any) -> Dict[str, Any]:
    """metering 事件的 metadata（两种后端形态都吃：dict 或 JSON 字符串）。"""
    if not isinstance(event, dict):
        return {}
    meta = event.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _order_key(event: Dict[str, Any]) -> Tuple[int, str]:
    """排序键：先用 metadata 里的 `turn_index`，缺失再退回 `created_at`。"""
    meta = _metadata_of(event)
    try:
        turn_index = int(meta.get("turn_index") or 0)
    except (TypeError, ValueError):
        turn_index = 0
    return (turn_index, str(event.get("created_at") or ""))


def readings_from_turn_events(events: Iterable[Any]) -> List[Dict[str, str]]:
    """一串 metering 事件 → 按时间排好的读数序列。

    **相邻读数签名完全相同即折叠**：那多半是同一次讲述被重放（会话内缓存 / 记忆
    恢复路径），不重复计票。不比较原话、不做哈希。
    """
    rows: List[Tuple[Tuple[int, str], Dict[str, str]]] = []
    for event in events or ():
        if not isinstance(event, dict):
            continue
        raw = _metadata_of(event).get(READING_KEY)
        if not isinstance(raw, dict):
            continue
        reading: Dict[str, str] = {}
        for dim in DIMENSIONS:
            value = str(raw.get(dim) or "").strip().lower()
            if value in (STATUS_OK, STATUS_WEAK):
                reading[dim] = value
        if not reading:
            continue
        rows.append((_order_key(event), reading))
    rows.sort(key=lambda item: item[0])
    out: List[Dict[str, str]] = []
    for _, reading in rows:
        if out and out[-1] == reading:
            continue
        out.append(reading)
    return out


def trends_from_readings(readings: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """读数序列 → `{维度: {streak, direction}}`（只有够 `MIN_STREAK` 的才返回）。

    「最近怎么走」= 从**最新一条往回数**，同一维连续同向的条数。中间：
      * 该维是 `na`（这一轮没观察到）→ 链**断**，不越过它继续数；
      * 方向翻转 → 链断。
    都不猜、不补。
    """
    ordered = [row for row in (readings or ()) if isinstance(row, dict)]
    out: Dict[str, Dict[str, Any]] = {}
    for dim in DIMENSIONS:
        streak = 0
        direction = ""
        for reading in reversed(ordered):
            value = str(reading.get(dim) or "").strip().lower()
            if value not in (STATUS_OK, STATUS_WEAK):
                break
            if not direction:
                direction = value
            if value != direction:
                break
            streak += 1
        if direction and streak >= MIN_STREAK:
            out[dim] = {"streak": streak, "direction": direction}
    return out


def progress_line(trends: Any) -> str:
    """趋势 → 一句**有界**的盘内提示；没有趋势返回**空串**（调用方据此不加段）。

    措辞只走 `expression_argument.DIMENSION_LABELS`（同一处权威措辞源），
    不另造第二套维度名。次数照实报，不四舍五入。
    """
    if not isinstance(trends, dict) or not trends:
        return ""
    valid: List[Tuple[str, int]] = []
    for dim in DIMENSIONS:
        info = trends.get(dim)
        if not isinstance(info, dict):
            continue
        try:
            streak = int(info.get("streak") or 0)
        except (TypeError, ValueError):
            continue
        direction = str(info.get("direction") or "").strip().lower()
        if direction in (STATUS_OK, STATUS_WEAK) and streak >= MIN_STREAK:
            valid.append((dim, streak))
    if not valid:
        return ""
    valid.sort(key=lambda item: (-item[1], DIMENSIONS.index(item[0])))
    parts: List[str] = []
    for dim, streak in valid[:MAX_REPORTED_DIMS]:
        label = DIMENSION_LABELS.get(dim, dim)
        if trends[dim]["direction"] == STATUS_OK:
            parts.append("「" + label + "」连着 " + str(streak) + " 次都做到了，这一条可以先放一放")
        else:
            parts.append("「" + label + "」连着 " + str(streak) + " 次都没做到，优先盯它")
    return (
        "[Expression Progress] 这一盘已经攒出的趋势（**由代码统计**，不是你的判断）："
        + "；".join(parts)
        + "。以上只作你挑焦点时的参考，**不要在回复里报次数、不要提「我在统计你」**。"
    )
