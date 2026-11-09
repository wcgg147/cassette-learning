# -*- coding: utf-8 -*-
"""她自己的日子（世界线）的**生成**部分 —— plans/1017-02 §8.79。

为什么单独一个文件：这层只负责"一周写一次她的生活"，和对话无关，
放在 `profile_evolution` 里会让那个 3000 行的文件更难读。

成本：**每个用户约一周一次**模型调用（只在"一条有效条目都没有"时触发）。
"""
from __future__ import annotations

import json
import os

LIFE_LINE_SYSTEM = "\n".join([
    "你在过自己的日子。写 3-5 条这周你在做的事，用第一人称、中文。",
    "要求：",
    "① 都是具体的小事（在学什么、在做什么、买了什么、去哪里转了），不是心情总结；",
    "② 和你的设定一致，不许冲突；",
    "③ 不涉及用户，不写「陪谁聊天」这类；",
    "④ 每条 8-20 字。",
    "⑤ 每条给 8-12 个「别人可能用来指这件事」的抓手（aliases），中英都要，口语、要短、不许整句。**下面四类每类至少 2 个，缺一类就算不合格**：",
    "   ① 同义词/换动词：「学吉他」→ 琴,弹琴,练琴,guitar,practice",
    "   ② 相关物品/场景：「手冲咖啡」→ 咖啡豆,滤杯,泡一杯,brew,beans",
    "   ③ 身体/情绪感受：「学吉他」→ 手指疼,茧,blisters,sore fingers",
    "   ④ 词序不同的说法：「戒夜宵」也要给「宵夜」",
    "   例：「在学吉他」→ 琴,弹琴,练琴,乐器,guitar,practice,茧,sore fingers",
    "只输出 JSON：{\"items\":[{\"text\":\"...\",\"kind\":\"work|life|plan\",\"aliases\":[\"...\",\"...\"]}]}",
])


def compose_user_message(previous: list[dict] | None = None) -> str:
    """这周从上周长出来：能接的接上，别重复原句（§8.80）。"""
    texts = [str(x.get("text") or "").strip() for x in (previous or [])]
    texts = [x for x in texts if x][:5]
    if not texts:
        return "写这周的。"
    return (
        "上周你在做这些：\n- " + "\n- ".join(texts)
        + "\n\n接着写这周（3-5 条）：能延续的就写它的下文（同一件事的进展、结果、或者又失败了），"
        "别把原话重复一遍；也可以新开一两件新的。"
    )


async def refresh_life_line(l2, user_id: str) -> int:
    """一条有效条目都没有时才生成——约一周一次。返回写入条数。"""
    try:
        from config import COMPANION_LIFE_LINE_ENABLED as enabled
    except Exception:
        enabled = False
    if not enabled or l2 is None or not hasattr(l2, "replace_life_items"):
        return 0
    try:
        # 只数"过日子"的条目：她自己读/看来的记录也是 active，
        # 拿它当"生活线还新鲜"的证据，会让日常永远不刷新、慢慢只剩阅读笔记。
        _active = await l2.list_life_items(user_id, limit=40) or []
        if any(str(x.get("kind") or "") != "reading" for x in _active):
            return 0
    except Exception:
        return 0

    seeds = ""
    try:
        from memory.persona_seed import PERSONA_SEED_FACTS as facts
        seeds = "；".join(str(x)[:60] for x in list(facts or [])[:8])
    except Exception:
        seeds = ""

    previous: list[dict] = []
    try:
        if hasattr(l2, "list_recent_life_items"):
            previous = await l2.list_recent_life_items(user_id, limit=5) or []
    except Exception:
        previous = []

    try:
        from llm_client import robust_llm_client, thinking_disabled_extra_body
        try:
            from config import MEMORY_CURATOR_MODEL as model
        except Exception:
            model = None
        raw = await robust_llm_client.chat_completion(
            [
                {"role": "system", "content": LIFE_LINE_SYSTEM
                 + ("\n你的设定：" + seeds if seeds else "")},
                {"role": "user", "content": compose_user_message(previous)},
            ],
            model=model, temperature=0.9, max_tokens=300,
            extra_body=thinking_disabled_extra_body(model),
        )
        parsed = json.loads(str(raw or "{}"))
    except Exception as exc:  # noqa: BLE001
        if os.getenv("PERSONA_DEBUG_PACKET"):
            print(f"[DBG] life-line gen failed: {type(exc).__name__}: {exc}", flush=True)
        return 0

    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return 0
    clean = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split())[:160]
        if len(text) < 4:
            continue
        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a).strip() for a in aliases if str(a).strip()][:6]
        clean.append({"text": text, "kind": str(item.get("kind") or "life")[:40],
                      "aliases": " ".join(aliases)})
    if not clean:
        return 0
    try:
        return await l2.replace_life_items(user_id, clean, days=7)
    except Exception:
        return 0
