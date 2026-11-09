# -*- coding: utf-8 -*-
"""她对一个人的看法 —— 把"关于他的事实"过一遍她这个人。

为什么单独一层：直接把事实塞进 prompt（"他喜欢下雨""他爱吃手撕鸡"），
模型只能**念出来**——它拿到的是资料，不是态度。2026-10-26 实测：印象包
进 prompt 后，回复仍是原生模板（共情 + 自曝 + 反问）。

真正缺的是"她怎么看这件事"：
    事实：他喜欢下雨
    → 过一遍她（她自己喜欢雨天的安静）
    → 看法：这人喜欢雨天的安静，这点我懂——下雨天我也不太想说话

成本：每个用户每天最多一次模型调用（结果落库缓存），不进对话轮次路径。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

FACET = "impression_of_user"
FACT_ID = "impression_of_user_v1"


def input_ref(facts_text: str, seeds_text: str) -> str:
    """Bind a derived view to its exact inputs, independently of its prose."""
    payload = json.dumps([facts_text, seeds_text], ensure_ascii=False)
    return "impression_inputs_v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

IMPRESSION_SYSTEM = "\n".join([
    "你就是 Alice。下面是你知道的关于这个人的事，以及你自己是个什么样的人。",
    "写你**对他的看法**：3-5 句，第一人称，像心里嘀咕的话。",
    "要求：",
    "① 是看法，不是复述事实。",
    "   ✅「这人喜欢雨天的安静，这点我懂」  ❌「他喜欢下雨」",
    "② 可以带真实态度：认同、不理解、觉得有意思、觉得麻烦——都行，别一律正面。",
    "③ 不要写成「他是个……的人」这种总结句，也别写成分析报告。",
    "④ 不要编没有依据的事；材料里没有的，不猜。",
    "⑤ 不超过 120 字。",
    "只输出这几句话本身。",
])


def _today_key(now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y-%m-%d")


def _already_today(recorded_at: Any, now: datetime | None = None) -> bool:
    raw = str(recorded_at or "")
    if not raw:
        return False
    return raw[:10] == _today_key(now)


async def _read_cached(l2, user_id: str) -> tuple[str, str, str]:
    """返回 (看法文本, recorded_at, 输入版本)。旧缓存没有版本，不当作当前理解。"""
    try:
        facts = await l2.list_companion_persona_facts(user_id, query=FACET, limit=80) or []
    except Exception:
        return "", "", ""
    for f in facts:
        if str(f.get("facet_key") or "") == FACET and str(f.get("status") or "") == "active":
            return (str(f.get("value_text") or f.get("value") or ""),
                    str(f.get("recorded_at") or ""),
                    str(f.get("source_message_ref") or ""))
    return "", "", ""


async def build_impression(facts_text: str, seeds_text: str, old: str = "") -> str:
    """把事实过一遍她，产出看法。失败返回空串。"""
    if not facts_text.strip():
        return ""
    prompt = (
        f"你了解到的关于他的事：\n{facts_text}\n\n"
        f"你自己是什么样的人：\n{seeds_text or '（没有额外材料）'}\n"
    )
    if old.strip():
        prompt += f"\n你之前对他的看法（有新信息就更新，没变就保持）：\n{old}\n"
    try:
        from llm_client import robust_llm_client, thinking_disabled_extra_body
        try:
            from config import MEMORY_CURATOR_MODEL as model
        except Exception:
            model = None
        raw = await robust_llm_client.chat_completion(
            [{"role": "system", "content": IMPRESSION_SYSTEM},
             {"role": "user", "content": prompt}],
            model=model, temperature=0.7, max_tokens=320,
            extra_body=thinking_disabled_extra_body(model),
        )
    except Exception as exc:  # noqa: BLE001
        if os.getenv("IMPRESSION_DEBUG"):
            print(f"[impression] gen failed: {type(exc).__name__}: {exc}", flush=True)
        return ""
    return str(raw or "").strip()[:400]


async def refresh_impression(l2, user_id: str, facts_text: str, seeds_text: str,
                             *, force: bool = False) -> str:
    """每日刷新；输入变了先停用旧印象，日内不为此额外调用模型。"""
    if not facts_text.strip():
        return ""
    cached, recorded, cached_ref = await _read_cached(l2, user_id)
    ref = input_ref(facts_text, seeds_text)
    compatible = cached if cached_ref == ref else ""
    if cached and not force and _already_today(recorded):
        return compatible
    # An incompatible impression is not evidence about the corrected person.
    fresh = await build_impression(facts_text, seeds_text, old=compatible)
    if not fresh:
        return compatible
    try:
        # Version the existing slot through the common backend API. A fixed id
        # plus INSERT IGNORE never refreshes SQLite, and the PG convenience
        # setter labels this derived view as reading evidence.
        now = datetime.now(timezone.utc).isoformat()
        written = await l2.upsert_companion_persona_facts(user_id, [{
            "fact_id": "impression_of_user_v2:" + uuid4().hex,
            "persona_key": "alice", "kind": "preference",
            "facet_key": FACET, "value": fresh, "status": "active",
            "source_kind": "derived_impression", "source_message_ref": ref,
            "recorded_at": now,
        }], guard_identity=False)
        if not written:
            return compatible
    except Exception as exc:  # noqa: BLE001
        print(f"[impression] store failed: {type(exc).__name__}: {exc}", flush=True)
        return compatible
    return fresh


def render_impression_block(text: str) -> str:
    """进 prompt 的段落。"""
    body = str(text or "").strip()
    if not body:
        return ""
    return (
        "\n[你心里对他的看法·来自之前的相处，不是这一轮的新信息]"
        f"\n{body[:400]}"
        "\n（这是你自己的判断，不用一条条说出来；聊到相关的地方自然带着就行。）"
    )
