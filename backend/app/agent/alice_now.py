# -*- coding: utf-8 -*-
"""她此刻在过什么日子 —— 让 Alice 不是"待命状态"。

为什么单独一层：现有 `life_line` 是"这周做过的事"（事件库存），只在**他问**或
**话题撞词**时才摆出来；它解决的是"被问到有话说"，不解决"她本来在过日子"。
真机体感（2026-10-26 用户反馈）：她随时在线、句句全力接住，像准备就绪的客服，
缺少"自己的世界"。

这一层只做一件事：按**真实时间**给出"她此刻大概在做什么"，作为她的状态背景。
它不新增事实来源——内容全部来自下面这张作息表（设定）或已有 life_items（真事），
不在运行时编造新的生活细节。

一致性：同一天、同一时段用 seed 固定，避免"下午说在改稿、十分钟后说在做饭"。
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

# 作息表：按小时段的"此刻在做什么"。每条要具体、有生活质感，不写心情总结。
# 深夜/凌晨也在表里——用户半夜来聊天时不能说"我在上班"。
_RHYTHM: dict[str, tuple[str, ...]] = {
    "late_night": (          # 0-3
        "还没睡，窝在沙发上看完了一集，正犹豫要不要再来一集",
        "熬夜看一本停不下来的书，杯子里的水早就凉了",
        "躺下了但没睡着，在刷手机",
    ),
    "sleeping": (            # 3-6
        "已经睡了",
        "这个点早就睡了",
    ),
    "early_morning": (       # 6-8
        "刚起来，在磨豆子做今天第一杯手冲",
        "起得早，站在窗边喝第一杯咖啡",
    ),
    "morning_work": (        # 8-10
        "在赶一件拖了两天的活，桌上那杯咖啡已经凉了",
        "刚开始做事，把手头的东西理了一遍",
    ),
    "mid_morning": (         # 10-12
        "在改稿，改到第三遍还是不满意",
        "手上的活做到一半，卡在一个细节上",
    ),
    "lunch": (               # 12-14
        "刚吃完午饭，在楼下走了一圈",
        "这个点在吃饭，顺便歇一会儿",
    ),
    "afternoon_slow": (      # 14-16
        "下午最没精神的时候，靠第二杯咖啡撑着",
        "在做事，但效率不高，隔一会儿就想站起来走走",
    ),
    "late_afternoon": (      # 16-18
        "手上的活刚告一段落，在收拾桌子",
        "快收尾了，在心里盘算晚上吃什么",
    ),
    "evening": (             # 18-20
        "在做饭，锅里还炖着东西",
        "刚到家，换了衣服，在厨房里弄点吃的",
    ),
    "night": (               # 20-22
        "窝在沙发上看剧，手边放着一杯没喝完的东西",
        "晚上在看书，看到一半走神了",
    ),
    "before_sleep": (        # 22-24
        "洗漱完躺着刷手机，不太想睡",
        "准备睡了，但还在翻手机",
    ),
}

_WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# 时段 → 用于和 life_items 做匹配的时间词（有真事时优先用真事）
_SLOT_HINTS: dict[str, tuple[str, ...]] = {
    "early_morning": ("早上", "清晨", "咖啡", "早餐"),
    "morning_work": ("上午", "早上", "工作", "稿"),
    "mid_morning": ("上午", "稿", "工作"),
    "lunch": ("中午", "午饭", "吃"),
    "afternoon_slow": ("下午", "咖啡", "工作"),
    "late_afternoon": ("下午", "傍晚", "收拾"),
    "evening": ("晚上", "做菜", "做饭", "买"),
    "night": ("晚上", "看剧", "书", "电影"),
    "before_sleep": ("晚上", "睡前", "手机"),
    "late_night": ("半夜", "熬夜", "电影", "剧"),
    "sleeping": ("睡",),
}


def slot_for(hour: int) -> str:
    """小时 → 时段 key。"""
    h = int(hour) % 24
    if h < 3:
        return "late_night"
    if h < 6:
        return "sleeping"
    if h < 8:
        return "early_morning"
    if h < 10:
        return "morning_work"
    if h < 12:
        return "mid_morning"
    if h < 14:
        return "lunch"
    if h < 16:
        return "afternoon_slow"
    if h < 18:
        return "late_afternoon"
    if h < 20:
        return "evening"
    if h < 22:
        return "night"
    if h < 24:
        return "before_sleep"
    return "late_night"


def _stable_pick(options: tuple[str, ...], seed: str) -> str:
    """同一天同一时段稳定选一条（跨轮一致，不忽东忽西）。"""
    if not options:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def _match_life_item(slot: str, life_items: list[dict] | None) -> str:
    """有时段对得上的真事就用真事（更贴她的实际生活），否则返回空。"""
    hints = _SLOT_HINTS.get(slot) or ()
    if not hints:
        return ""
    best, best_hits = "", 0
    for item in (life_items or [])[:6]:
        text = str((item or {}).get("text") or "").strip()
        if not text:
            continue
        hits = sum(1 for h in hints if h in text)
        if hits > best_hits:
            best, best_hits = text, hits
    return best if best_hits >= 1 else ""


def current_state(now: datetime, life_items: list[dict] | None = None,
                  seed: str = "", turn: int = 0) -> dict[str, Any]:
    """她此刻的状态。只从作息表或已有 life_items 取，不编新事实。

    `turn` 参与 seed：原来只用 日期+时段，一个时段内**永远选同一件**，于是她
    整场对话反复说"看书走神"（2026-10-26 实测 6 轮说 4 次）。按 3 轮换一档，
    状态有变化但不会每轮都在干别的事。
    """
    slot = slot_for(now.hour)
    life_text = _match_life_item(slot, life_items)
    _pick_seed = f"{seed}:{now.date()}:{slot}:{int(turn or 0) // 3}"
    doing = life_text or _stable_pick(_RHYTHM.get(slot, ()), _pick_seed)
    return {
        "slot": slot,
        "doing": doing,
        "from_life": bool(life_text),
        "clock": f"{_WEEKDAY_ZH[now.weekday()]} {now.strftime('%H:%M')}",
    }


def render_now_block(now: datetime, life_items: list[dict] | None = None,
                     seed: str = "", turn: int = 0) -> str:
    """给 prompt 的「你此刻」段。

    措辞刻意留了"不是待命"和"不必主动汇报"两句：她要有自己的日子，
    但不能变成每轮播报自己——那是用户明确反感的。

    2026-10-26 实测：同一件"看书走神"在 6 轮里说了 4 次。原因不是措辞，
    是**素材每轮都一样**——一个时段只选一件事，她只能用这一件。
    修法在 `current_state(turn=...)`：按 3 轮换一档，让素材本身不重复。
    （先试过在 prompt 里写"这件事你已经提过N次"——模型直接忽略，已删。）
    """
    # 「你此刻」是**背景**，不是话题：只在开场和之后每 4 轮给一次。
    # 2026-10-26 实测：每轮都给同一句"在看书走神"，她 6 轮里说了 4 次；
    # 改成按 3 轮换一档后仍有 3 次——因为 3 轮内素材还是同一句。
    # 真人不会每轮报告自己在干嘛，所以这里控制的是**出现频率**，不是措辞。
    if int(turn or 0) % 4 != 0:
        return ""
    st = current_state(now, life_items=life_items, seed=seed, turn=turn)
    doing = st.get("doing") or ""
    if not doing:
        return ""
    # 「聊久了会累」这一段**故意不做**（2026-10-26）：
    # ① 代码没有可靠的"她有没有兴趣"信号——curious 的判据是"用户主动说话"，
    #    是"用户开口"的节流版，不是兴趣；
    # ② 只按轮次写死"你累了"违背"累要分情况、真感兴趣可以多聊"；
    # ③ 用户口径：宁可没有，不要假的。
    # 要做的前提是先有真兴趣信号（她自己主动展开的话题 / 她的品味命中），
    # 那属于输入层，不在这里硬凑。
    return (
        "\n[你此刻]"
        f"现在是{st['clock']}，你{doing}。"
        "你是从自己的日子里分出一点注意力跟他说话——不是待命状态，也不用每句话都全力接住。"
        "聊到相关的地方可以自然带一句，不必主动汇报；用你平时说话的语言带出来，别照抄这句中文。"
    )
