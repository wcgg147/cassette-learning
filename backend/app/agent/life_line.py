# -*- coding: utf-8 -*-
"""她自己的日子与外部冲浪经历（世界线）——§8.79 + 外部输入与真实动作接收。

只做三件事，都确定性、零模型调用、零逐轮查库：
  1. 判断"他是不是在问她"（你呢 / 你最近 / 你平时刷B站吗 …）；
  2. 挑出最合拍的一两条经历（话题/近义词/UP主/互动动作命中优先；多轮追问锁住上一轮焦点）；
  3. 渲染成陈述句实据（被问深或话题撞上时附上 UP 主、点赞投币收藏关注留评实况与内容备忘）。

⛔ 素材由后台定时段沉淀，这里不写内容、不编事实。
⛔ 闲聊未撞词时遵循 `skip_ids` 不重复；但用户明确追问刚才那条或直接点名某话题时允许展开细节。
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone

ASK_ABOUT_HER_ZH = ("你呢", "那你呢", "你最近", "你平时", "你学过", "你会不会", "你喜欢什么",
                    "你做什么", "你周末", "你也在", "你有过", "你怎么办", "你有在",
                    "那你还有", "还有别的", "还有什么", "你的爱好", "你一般", "你经常",
                    "你有没有", "你爱", "你干嘛", "你都干嘛", "业余",
                    "你在干嘛", "你今天干嘛", "你最近忙", "你在忙什么", "你最近在做什么",
                    "你都做什么", "刷b站", "看b站", "在b站", "哪个up", "哪位up",
                    "刷推", "看推特", "在推特", "推特上", "在x上", "刷x", "推文",
                    "小红书", "刷抖音", "在抖音", "哪个平台", "哪些平台", "什么平台", "哪个app", "哪些app", "什么软件",
                    "看小说", "追书", "书架", "推书", "追更", "起点", "番茄小说", "本章说", "章评", "段评",
                    "看直播", "直播间", "看比赛",
                    "关注了", "收藏夹", "加书签", "投过币", "发过评论", "留了什么", "留评")
ASK_ABOUT_HER_EN = ("how about you", "what about you", "do you ", "have you ", "are you ",
                    "what do you", "your week", "you ever", "what else do you",
                    "any hobbies", "your hobbies", "about yourself",
                    "watch any", "seen any", "on twitter", "on x", "any tweets",
                    "did you bookmark", "did you save")

# 另一条路：他在"问你"的形式千变万化，光列关键词必然漏（实测漏过
# "那你还有别的爱好吗？" ⇒ 没给材料 ⇒ 她现编"最近在琢磨做饭"）。
# 所以再加一条组合判据：**带问询标记 + 句子里有"你"** 就算在问她。
_ASK_MARKERS_ZH = ("吗", "呢", "?", "？", "什么", "怎么", "哪个", "哪儿", "多少", "是不是", "有没有")
_ASK_THIS_HER = ("你", "妳")

_EN_STOPWORDS = frozenset({
    "the", "and", "for", "you", "are", "did", "any", "has", "was", "not",
    "but", "how", "who", "why", "can", "out", "one", "got", "its", "her",
    "his", "our", "all", "too", "now", "new", "day", "off", "see", "saw",
    "say", "get", "let", "put", "set", "try", "use", "way", "may", "had",
    "have", "that", "this", "with", "from", "what", "when", "your", "been",
    "into", "just", "more", "some", "than", "them", "then", "they", "were",
    "will", "also", "back", "even", "good", "here", "know", "like", "look",
    "make", "most", "much", "need", "only", "over", "said", "same", "such",
    "take", "tell", "time", "very", "want", "well", "work", "year", "ever",
    "about", "after", "again", "could", "doing", "every", "first", "going",
    "great", "never", "other", "right", "still", "their", "there", "these",
    "thing", "think", "those", "today", "under", "where", "which", "while",
    "would", "lately", "really", "recently", "anything", "something", "seen",
})

_ZH_STOP_BIGRAMS = frozenset({
    "最近", "今天", "昨天", "明天", "现在", "平时", "时候", "什么", "怎么",
    "为什", "哪个", "哪些", "这个", "那个", "这些", "那些", "自己", "我们",
    "你们", "他们", "一个", "一条", "一下", "一点", "一些", "一直", "一般",
    "一样", "已经", "可以", "可能", "应该", "觉得", "感觉", "知道", "看到",
    "刷到", "看过", "刷过", "没有", "不是", "就是", "还是", "或者", "因为",
    "所以", "但是", "不过", "如果", "虽然", "然后", "而且", "其实", "真的",
    "真是", "特别", "非常", "比较", "有点", "出来", "起来", "下去", "下来",
    "进去", "过来", "过去", "那里", "这里", "哪儿", "这儿", "那你", "你呢",
    "你在", "你有", "你会", "你最", "都在", "在忙", "忙什", "干嘛", "做什",
    "看什", "看点", "忙点", "闲下", "好多", "在聊", "视频", "那期", "这期",
    "有意", "意思", "好玩", "好看", "有趣", "除了", "有没", "或者",
})

_ASK_OTHER_CUES = ("别的", "其他", "其它", "另外", "还有什么", "还有呢", "what else", "anything else", "other hobbies")

_FOLLOWUP_FOCUS_CUES = (
    "那个视频", "那条视频", "那期", "这期", "那条推", "那条推文", "那个博主",
    "他讲", "里面讲", "具体讲", "展开讲",
    "细说", "怎么说", "评论区", "弹幕", "留了什么", "留评", "留评论", "发评论",
    "收藏了吗", "书签了吗", "加书签了吗", "投币了吗", "点过赞", "关注了吗", "漏了什么", "缺了什么",
    "that video", "that tweet", "that post", "that thread", "that one", "that review",
    "did it ", "what did it", "miss any", "bookmark", "save it", "saved it", "comment on", "tell me more",
)


def asked_about_her(text: str) -> bool:
    low = (text or "").lower()
    if any(k in low for k in ASK_ABOUT_HER_ZH):
        return True
    if any(k in low for k in ASK_ABOUT_HER_EN):
        return True
    # 兜底组合判据：问询标记 + 有"你" + **没有"我"**（有"我"的多半在问他的事，
    # 例如"你说我该怎么跟我妈说"——那种不该塞她自己的事）。
    if any(m in text for m in _ASK_MARKERS_ZH) and any(p in text for p in _ASK_THIS_HER)             and not any(s in text for s in ("我", "俺")):
        return True
    return False


def _bigrams(text: str) -> set[str]:
    clean = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9 ]", " ", text or "")
    out: set[str] = set()
    for token in re.findall(r"[\u4e00-\u9fff]+", clean):
        for i in range(len(token) - 1):
            bg = token[i:i + 2]
            if bg not in _ZH_STOP_BIGRAMS:
                out.add(bg)
    for w in re.findall(r"[a-zA-Z0-9]{2,}", clean):
        lw = w.lower()
        if lw not in _EN_STOPWORDS and not lw.isdigit():
            out.add(lw)
    return out


def _parse_actions(item: dict) -> dict:
    raw = (item or {}).get("actions_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    act = (item or {}).get("actions")
    return dict(act) if isinstance(act, dict) else {}


def _action_tags(item: dict) -> set[str]:
    act = _parse_actions(item)
    tags: set[str] = set()
    for k in ("like", "coin", "fav", "follow"):
        if act.get(k):
            tags.add(k)
    if act.get("comments"):
        tags.add("comments")
    return tags


def score_item(item: dict, user_text: str) -> int:
    """撞上多少：条目原文 + 写入时存的近义抓手/标题/UP主/备忘 + 互动动作意图（§8.92）。

    原文命中权重高于别名兜底（保证 `test_text_hit_not_diluted_by_alias`），
    同时支持英文跨语言召回与"关注/投币/收藏/评论"动作追问，查询侧零模型调用。
    """
    if not isinstance(item, dict) or not (user_text or "").strip():
        return 0
    said = _bigrams(user_text)
    text_hit = len(_bigrams(str(item.get("text") or "")) & said) * 3
    meta_str = " ".join(
        str(item.get(k) or "") for k in ("aliases", "title", "author", "topic")
    )
    meta_hit = len(_bigrams(meta_str) & said) * 2
    detail_str = str(item.get("detail") or "")
    detail_hit = len(_bigrams(detail_str) & said) if detail_str else 0

    low = (user_text or "").lower()
    plat = detect_item_platform(item)
    ask_x = any(w in low for w in ("刷推", "推特", "推文", "在x上", "x上", "刷x", "或者x", "在x或", "twitter", "tweet", "on x"))
    ask_xhs = any(w in low for w in ("小红书", "红书", "xhs", "xiaohongshu"))
    ask_dy = any(w in low for w in ("刷抖音", "看抖音", "在抖音", "抖音上", "抖音", "douyin", "tiktok"))
    ask_rd = any(w in low for w in ("reddit", "subreddit"))
    ask_qd = any(w in low for w in ("起点", "起点中文网", "起点的书", "本章说", "投推荐票"))
    ask_fq = any(w in low for w in ("番茄", "番茄小说", "番茄的书", "段评"))
    ask_novel = any(w in low for w in ("看小说", "小说", "追书", "书架", "推书", "追更", "弃坑", "修仙", "玄幻", "网文"))
    ask_live = any(w in low for w in ("直播", "看播", "主播", "直播间", "弹幕", "解说", "比赛", "赛事", "livestream", "live stream", "stream"))
    excluded_bili = bool(re.search(r"除了.{0,6}(?:b站|bilibili)", low))
    ask_bili = (not excluded_bili) and any(
        w in low for w in ("刷b站", "看b站", "在b站", "b站", "bilibili", "哪个up", "哪位up")
    )
    act = _parse_actions(item)
    action_hit = 0
    if ask_x and plat == "x" and not ask_bili:
        action_hit += 8
    elif ask_x and plat == "x":
        action_hit += 5
    elif ask_xhs and plat == "xiaohongshu":
        action_hit += 8
    elif ask_dy and plat == "douyin":
        action_hit += 8
    elif ask_rd and plat == "reddit":
        action_hit += 8
    elif ask_qd and plat == "qidian":
        action_hit += 8
    elif ask_fq and plat == "fanqie":
        action_hit += 8
    elif ask_novel and plat in ("qidian", "fanqie"):
        action_hit += 5
    elif ask_live and plat == "bili_live":
        action_hit += 8
    elif ask_bili and plat == "bilibili" and not ask_x:
        action_hit += 3
    if act:
        if any(w in low for w in ("关注", "哪个up", "哪位up", "博主", "follow", "subscribe")) and act.get("follow"):
            action_hit += 5
        if any(w in low for w in ("投币", "投过币", "几个币", "coin")) and act.get("coin"):
            action_hit += 5
        if any(w in low for w in ("收藏", "书签", "马住", "bookmark", "save it", "saved it")) and act.get("fav"):
            action_hit += 4
        if any(w in low for w in ("评论", "留言", "留评", "留了什么", "发过评", "回过", "comment", "reply")) and (act.get("comments") or []):
            action_hit += 5
        if any(w in low for w in ("点赞", "点过赞", "liked")) and act.get("like"):
            action_hit += 3
        if ((ask_bili and plat == "bilibili") or ("光看不动手" in low)) and (
            any(act.get(k) for k in ("like", "coin", "fav", "follow")) or act.get("comments")
        ):
            action_hit += 2

    secondary = (meta_hit + detail_hit) if text_hit == 0 else min(meta_hit + detail_hit, 2)
    return text_hit + secondary + action_hit


def _is_followup_about_focus(user_text: str) -> bool:
    low = (user_text or "").lower()
    if any(c in low for c in _ASK_OTHER_CUES):
        return False
    return any(c in low for c in _FOLLOWUP_FOCUS_CUES)


# 遗忘曲线（Anderson & Schooler 1991）：幂律，不是指数。
POWER_D = 0.7
SCALE_DAYS = 1.0


def _parse_ts(value) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def storage_strength(item: dict) -> float:
    """存储强度：学得多牢。只增不减，缺字段按 1.0 算。"""
    try:
        return max(0.05, float((item or {}).get("strength") or 1.0))
    except (TypeError, ValueError):
        return 1.0


def retrieval_strength(item: dict, now=None) -> float:
    """提取强度：此刻容不容易想起来（0-1）。这才是"忘没忘"。"""
    now = now or datetime.now(timezone.utc)
    last = _parse_ts((item or {}).get("last_recalled_at")) or _parse_ts(
        (item or {}).get("created_at"))
    if last is None:
        return 1.0
    age_days = max(0.0, (now - last).total_seconds() / 86400.0)
    return (1.0 + age_days / (storage_strength(item) * SCALE_DAYS)) ** (-POWER_D)


# 兼容旧名字：外部（测试/旧调用）还在用 memory_strength
memory_strength = retrieval_strength


def recall(pool: list[dict], k: int, rng=None,
           weights: list[float] | None = None) -> list[dict]:
    """按剩余分量抽 k 条：反复接触过的常被想起，一次性的多数时候想不起来。"""
    rng = rng or random
    items = list(pool or [])
    k = max(0, min(int(k), len(items)))
    base = list(weights or [1.0] * len(items))
    w = [max(0.01, retrieval_strength(x)) * max(0.01, b)
         for x, b in zip(items, base)]
    out: list[dict] = []
    for _ in range(k):
        total = sum(w)
        if total <= 0:
            break
        pick = rng.uniform(0.0, total)
        acc = 0.0
        for i, weight in enumerate(w):
            acc += weight
            if pick <= acc:
                out.append(items.pop(i))
                w.pop(i)
                break
    return out


def pick_items(items: list[dict], user_text: str, *,
               skip_ids: set[str] | None = None, limit: int = 2,
               active_focus_id: str = "") -> list[dict]:
    """把此刻最该想起来的材料摆出来：
    1. 话题/实体/动作直接撞词（score > 0）：按匹配分优先，跨动作类型互补；
       若用户正追问刚聊过的那条（score >= 4 或命中焦点追问词），允许突破 skip_ids 展开细节；
    2. 未撞词的日常泛聊：严格遵守 skip_ids 不重复，生活条目保底 + 阅读经历互补。
    """
    skip = skip_ids or set()
    limit = max(0, int(limit))
    if limit == 0:
        return []
    valid = [x for x in (items or []) if isinstance(x, dict)
             and str(x.get("text") or "").strip()]
    if not valid:
        return []

    low = (user_text or "").lower()
    asking_other = any(c in low for c in _ASK_OTHER_CUES)

    # 1. 显式命中话题 / UP主 / 别名 / 互动动作
    scored: list[tuple[int, float, dict]] = []
    for x in valid:
        sc = score_item(x, user_text)
        if sc <= 0:
            continue
        iid = str(x.get("item_id") or "")
        # 已提过的条目只在用户明确追问（sc >= 4 且非"还有别的吗"）时才允许二次展开详情
        if iid in skip and (sc < 4 or asking_other):
            continue
        scored.append((sc, retrieval_strength(x), x))

    if scored:
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        first = {**scored[0][2], "_matched": True}
        out = [first]
        if limit >= 2 and len(scored) > 1:
            covered = _action_tags(first)
            best_second = None
            best_eff = -1.0
            for sc, rs, cand in scored[1:]:
                uncovered = len(_action_tags(cand) - covered)
                eff = sc + uncovered * 3.0 + rs * 0.1
                if eff > best_eff:
                    best_eff = eff
                    best_second = (sc + uncovered * 3, cand)
            # 只有第二条也具备明确相关性（有效分 >= 6，例如同时问了投币+关注+评论）才并列给出，
            # 避免单点追问某条视频时顺带塞进无关条目分散注意力。
            if best_second is not None and best_second[0] >= 6:
                out.append({**best_second[1], "_matched": True})
        return out[:limit]

    # 2. 上一轮刚提过某条经历，这一轮顺着追问细节（"那个视频具体讲了啥/你收藏了吗"）
    if active_focus_id and _is_followup_about_focus(user_text):
        for x in valid:
            if str(x.get("item_id") or "") == active_focus_id:
                return [{**x, "_matched": True}]

    # 3. 泛问近况或日常闲聊：严格排除已提过的 skip_ids
    pool = [x for x in valid if str(x.get("item_id") or "") not in skip]
    if not pool:
        return []
    daily = [x for x in pool if str(x.get("kind") or "") != "reading"]
    reading = [x for x in pool if str(x.get("kind") or "") == "reading"]
    picked = recall(daily, 1, weights=[6.0] * len(daily)) if daily else []
    if len(picked) < limit:
        # 第二槽优先给有深度的外部冲浪经历（兼顾她的线下日常与线上冲浪世界）
        rem_pool = reading if (picked and reading) else [x for x in pool if x not in picked]
        if rem_pool:
            rem_weights = [
                max(1.0, float(x.get("depth") or 1.0)) * (1.5 if _action_tags(x) else 1.0)
                for x in rem_pool
            ]
            picked += recall(rem_pool, limit - len(picked), weights=rem_weights)
    return picked[:limit]


_PLATFORM_META = {
    "bilibili": {
        "platform": "B站",
        "creator": "B站UP主",
        "work": "视频",
        "follow_who": "这位UP主",
        "fav_done": "加了收藏",
        "fav_no": "未收藏",
        "reply_where": "评论区留过话",
        "reply_no": "未留评",
        "has_coin": True,
    },
    "bili_live": {
        "platform": "B站直播",
        "creator": "B站主播",
        "work": "直播",
        "follow_who": "这位主播",
        "fav_done": "关注了直播间",
        "fav_no": "未关注",
        "reply_where": "弹幕里说过话",
        "reply_no": "没发弹幕，只挂着听",
        "has_coin": False,
    },
    "x": {
        "platform": "X(Twitter)",
        "creator": "X(Twitter)博主",
        "work": "推文",
        "follow_who": "这位博主",
        "fav_done": "加了书签",
        "fav_no": "未加书签",
        "reply_where": "推文下回过话",
        "reply_no": "未回复",
        "has_coin": False,
    },
    "xiaohongshu": {
        "platform": "小红书",
        "creator": "小红书博主",
        "work": "笔记",
        "follow_who": "这位博主",
        "fav_done": "加了收藏",
        "fav_no": "未收藏",
        "reply_where": "评论区留过话",
        "reply_no": "未留评",
        "has_coin": False,
    },
    "douyin": {
        "platform": "抖音",
        "creator": "抖音创作者",
        "work": "短视频",
        "follow_who": "这位创作者",
        "fav_done": "加了收藏",
        "fav_no": "未收藏",
        "reply_where": "评论区留过话",
        "reply_no": "未留评",
        "has_coin": False,
    },
    "reddit": {
        "platform": "Reddit",
        "creator": "Reddit网友",
        "work": "帖子",
        "follow_who": "这位答主",
        "fav_done": "加了书签",
        "fav_no": "未加书签",
        "reply_where": "帖子下回过话",
        "reply_no": "未回复",
        "has_coin": False,
    },
    "qidian": {
        "platform": "起点中文网",
        "creator": "起点作家",
        "work": "作品",
        "follow_who": "这位作家",
        "fav_done": "加入了书架",
        "fav_no": "未入书架",
        "reply_where": "章评区留过言",
        "reply_no": "未留评",
        "has_coin": False,
    },
    "fanqie": {
        "platform": "番茄小说",
        "creator": "番茄作者",
        "work": "作品",
        "follow_who": "这位作者",
        "fav_done": "加入了书架",
        "fav_no": "未入书架",
        "reply_where": "段评区留过言",
        "reply_no": "未留评",
        "has_coin": False,
    },
}


def detect_item_platform(item: dict) -> str:
    """识别外部冲浪经历来自哪个平台（bilibili / x / xiaohongshu / douyin / reddit / qidian / fanqie / web / ''）。"""
    if not isinstance(item, dict):
        return ""
    src = str(item.get("source") or "").strip().lower()
    ref_raw = str(item.get("ref") or "").strip()
    ref_low = ref_raw.lower()
    iid = str(item.get("item_id") or "").strip().lower()
    kind = str(item.get("kind") or "").strip().lower()

    if src in ("x", "twitter") or "x.com/" in ref_low or "twitter.com/" in ref_low or iid.startswith("x_"):
        return "x"
    if src in ("qidian", "qd") or "qidian.com/" in ref_low or iid.startswith("qd_"):
        return "qidian"
    if src in ("fanqie", "fq") or "fanqienovel.com/" in ref_low or iid.startswith("fq_"):
        return "fanqie"
    if src in ("xiaohongshu", "xhs") or "xiaohongshu.com/" in ref_low or "xhslink.com/" in ref_low or iid.startswith("xhs_"):
        return "xiaohongshu"
    if src in ("douyin", "tiktok") or "douyin.com/" in ref_low or "tiktok.com/" in ref_low or iid.startswith("dy_"):
        return "douyin"
    if src == "reddit" or "reddit.com/" in ref_low or iid.startswith("rd_"):
        return "reddit"
    if src in ("bili_live", "bilibili_live") or "live.bilibili.com/" in ref_low or iid.startswith("live_"):
        return "bili_live"
    if src == "bilibili" or "bilibili.com/" in ref_low or "b23.tv/" in ref_low or ref_raw.startswith("BV"):
        return "bilibili"
    if src == "web" or ref_low.startswith(("http://", "https://")):
        return "web"
    if kind == "reading":
        aliases_low = str(item.get("aliases") or "").lower()
        act = _parse_actions(item)
        if ("文章" in aliases_low or "网页" in aliases_low) and "bilibili" not in aliases_low and "b站" not in aliases_low and "coin" not in act:
            return "web"
        return "bilibili"
    return ""


def _format_reading_meta(item: dict, *, include_detail: bool) -> str:
    parts: list[str] = []
    author = str(item.get("author") or "").strip()
    title = str(item.get("title") or "").strip()
    plat = detect_item_platform(item)
    pm = _PLATFORM_META.get(plat)

    if pm:
        if author and title:
            parts.append(f"来源：{pm['creator']}「{author}」{pm['work']}《{title[:28]}》")
        elif author:
            parts.append(f"来源：{pm['creator']}「{author}」")
        elif title and str(item.get("kind") or "") == "reading":
            parts.append(f"来源：{pm['platform']}{pm['work']}《{title[:28]}》")
    elif plat == "web":
        if author and title:
            parts.append(f"来源：网页文章（{author}）《{title[:28]}》")
        elif title:
            parts.append(f"来源：网页文章《{title[:28]}》")
        elif author:
            parts.append(f"来源：网页文章（{author}）")
    elif title and str(item.get("kind") or "") == "reading":
        parts.append(f"标题：《{title[:28]}》")

    act = _parse_actions(item)
    if act:
        act_pm = pm or _PLATFORM_META["bilibili"]
        has_coin = bool(act_pm["has_coin"]) and plat == "bilibili"
        done: list[str] = []
        not_done: list[str] = []
        if act.get("like"):
            done.append("点过赞")
        if has_coin:
            if act.get("coin"):
                done.append("投了币")
            elif include_detail:
                not_done.append("未投币")
        if act.get("fav"):
            done.append(act_pm["fav_done"])
        elif include_detail:
            not_done.append(act_pm["fav_no"])
        if act.get("follow"):
            done.append(f"关注了{act_pm['follow_who']}")
        comments = [str(c).strip() for c in (act.get("comments") or []) if str(c).strip()]
        if comments:
            done.append(f"在{act_pm['reply_where']}：「{comments[-1][:72]}」")
        elif include_detail:
            not_done.append(act_pm["reply_no"])
        act_bits = []
        if done:
            act_bits.append("、".join(done))
        if include_detail and not_done:
            act_bits.append("（" + "、".join(not_done) + "）")
        if act_bits:
            parts.append("你的互动：" + "".join(act_bits))

    if include_detail:
        detail = str(item.get("detail") or "").strip()
        if detail:
            parts.append(f"内容与评论区实据：{detail[:220]}")
    return "（" + "；".join(parts) + "）" if parts else ""


def render_line(items: list[dict], *, asked: bool = False) -> str:
    """只摆事实与亲历实据，不写任何行为指令词。"""
    rendered: list[str] = []
    for x in (items or [])[:2]:
        if not isinstance(x, dict):
            continue
        base = str(x.get("text") or "").strip().rstrip("。；;，,")
        if not base:
            continue
        meta = _format_reading_meta(x, include_detail=bool(x.get("_matched")))
        rendered.append(base + meta)
    if not rendered:
        return ""
    return "\n[你最近的事] " + "；".join(rendered) + "\n"
