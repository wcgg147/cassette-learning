"""回复质量律（plans/0916-27）：律 0 对题 + 语言增量 / 承认收窄 / 禁复述。

本模块是**纯函数 + 静态词表**，零模型调用、零网络往返——判定发生在回复文本
拿到之后、播出之前的那一段，不占延迟预算（口径同 engine/chunk_evidence.py）。

用户口径（2026-09-17）："我说我昨天吃的什么，他就以为我在聊前天的事情——
这是精确度的问题"。**答非所问是第一位的**，风格/口吻/气泡/延迟都在其后。

判定产物只有两类，都不改写文本：
  ① 违规标记（字符串）→ 进 ``_controlled_reply_violations`` 既有族，交给既有
     修复链与观测；
  ② 对题裁定（``alignment_state``）→ 进 ``turn_contract.alignment``，让模型在
     **生成侧**就知道"本轮要对准哪一件事"，这才是主修法（事后拦截只是兜底）。

诚实的边界（本项目惯例，先写清楚再写代码）：
  - 「核心对象对齐」只能做**同语言词面**比对。用户的转写是中文而回复是英文时，
    没有翻译层就验不了"是不是同一件东西"，这里**不假装能验**（跳过，不误报）。
  - 「时间锚对齐」只认闭集时间词（昨天/前天/今天/明天/上周…及其英文同义词），
    自造时间说法（"我上上个月"）不在集内，不判。
  - 「问句必须被回答」只拦"整段只有反问"这一形态；"答了半句+反问"不判违规。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 与 tutor_agent 同一口径的英文词/汉字识别（这里自持一份，避免循环 import：
# tutor_agent 要 import 本模块）。
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 英文功能词（不参与"新内容""复述"判定）。
_STOP_TOKENS = {
    "the", "a", "an", "and", "or", "but", "so", "if", "then", "than", "that",
    "this", "these", "those", "there", "here", "it", "its", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "done", "have",
    "has", "had", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "not", "no", "yes", "nor", "you", "your", "yours", "i",
    "me", "my", "mine", "we", "us", "our", "they", "them", "their", "he",
    "she", "him", "her", "his", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "whether", "about", "with", "without",
    "for", "to", "of", "in", "on", "at", "by", "from", "up", "down", "out",
    "off", "over", "under", "again", "just", "now", "then", "also", "too",
    "very", "really", "right", "okay", "ok", "well", "like", "any", "some",
    "all", "both", "each", "every", "one", "two", "get", "got", "make", "made",
    "let", "put", "take", "took", "give", "gave", "know", "knew", "think",
    "thought", "see", "seen", "look", "looked", "say", "said", "saying",
    "tell", "told", "mean", "means", "meant", "thing", "things", "stuff",
    "want", "wanted", "need", "needed", "ask", "asked", "asking", "talk",
    "talked", "sorry", "please", "thanks", "thank", "sure", "yeah", "hmm",
}

# 中文功能词/高频虚词（不参与中文侧内容词判定）。
_CJK_STOP = {
    "什么", "怎么", "怎样", "哪个", "哪个", "哪里", "为什么", "多少", "几点",
    "我们", "你们", "他们", "她们", "自己", "这个", "那个", "这些", "那些",
    "一个", "一下", "可以", "不是", "没有", "还有", "就是", "然后", "因为",
    "所以", "但是", "如果", "已经", "现在", "时候", "东西", "事情", "意思",
    "今天", "昨天", "前天", "明天", "刚才", "刚刚", "上周", "上个月", "去年",
}

# ---------------------------------------------------------------------------
# 时间锚（律 0 判据①）
# ---------------------------------------------------------------------------

# 闭集：canonical -> (中文形态, 英文形态)。canonical 用于"同锚"比较，
# ``day`` 组内互斥（昨天/前天/今天/明天 说错一个就是答非所问）。
_TIME_ANCHORS: List[tuple] = [
    ("yesterday", ("昨天", "昨儿", "昨晚", "昨夜"), ("yesterday", "last night")),
    ("day_before", ("前天", "前儿"), ("the day before yesterday",)),
    ("today", ("今天", "今日", "今晚", "今儿"), ("today", "tonight")),
    ("tomorrow", ("明天", "明日", "明儿"), ("tomorrow",)),
    ("last_week", ("上周", "上个星期", "上星期"), ("last week",)),
    ("last_month", ("上个月", "上月"), ("last month",)),
    ("last_year", ("去年", "上一年"), ("last year",)),
    ("this_morning", ("今早", "今天早上", "今天上午"), ("this morning",)),
    ("just_now", ("刚才", "刚刚"), ("just now", "a moment ago", "earlier today")),
    ("last_weekend", ("上周末", "上个周末"), ("last weekend",)),
]
# 互斥组：回复落在这组里的**另一个**锚上（且完全没提用户那个锚）= 违规。
_DAY_LEVEL_ANCHORS = {"yesterday", "day_before", "today", "tomorrow"}

# 明确更正：用户的时间锚被说错后，"你是说昨天？"这种纠正问句算合规处置。
_TIME_CORRECTION_RE = re.compile(
    r"(你是说|你指的是|did you mean|you mean|just to make sure|"
    r"correct me if|让我确认|确认一下)",
    re.IGNORECASE,
)

# 自述理解过程（律 0 判据④）：用户要的是答案，不是听 AI 复盘自己怎么想错的。
_SELF_PROCESS_RE = re.compile(
    r"(我(之前)?理解(成|错|偏)|我以为你(说|指)的是|我把它理解|我搞错(了)?(你|你的)意思|"
    r"我(之前)?弄错(了)?(你|你的)意思|重新说(一下)?我的理解|"
    r"i\s+(?:thought|assumed)\s+you\s+(?:meant|were\s+saying)|"
    r"i\s+(?:misread|misunderstood)\s+(?:what\s+)?you|"
    r"my\s+understanding\s+was|"
    r"i\s+was\s+(?:confused|wrong)\s+about\s+(?:what\s+)?you)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


# 中文虚词/单字功能词：切开用（"我昨天吃的什么" → 昨天 / 吃）。
# 不做完整分词，只保证切出来的片段是实义块而不是"天吃"这种碎片。
_CJK_SPLIT_RE = re.compile(
    r"[我你他她它您咱我们你们他们她们的了着过是在有和跟给对把被这那"
    r"什么怎么哪里哪个谁为什么多少吗呢吧啊哦嗯就都也还又很不没别]"
)


def content_tokens(text: str) -> set:
    """英文实义词词元（小写、去功能词、极简词尾还原）。"""
    tokens = set()
    for token in _WORD_RE.findall(_normalize(text).lower()):
        stem = _stem(token)
        if len(stem) >= 3 and stem not in _STOP_TOKENS:
            tokens.add(stem)
    return tokens


def _stem(token: str) -> str:
    tok = token
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
            tok = tok[: -len(suffix)]
            break
    if len(tok) > 3 and tok[-1] == tok[-2] and tok[-1] not in "lsz":
        tok = tok[:-1]
    return tok


def cjk_content_tokens(text: str) -> set:
    """中文内容词块：先按虚词/单字功能词切开，只留 2 字以上的实义片段。

    中文没有分词器，这里不假装能分——按高频虚词切完取整段，**不做滑窗**
    （滑窗会产出"天吃""我昨"这类碎片，既可能碰撞出假的"有交集"，也让
    违规理由读起来没有信息量）。切不出东西时就是没抓到对象，宁可不判。
    """
    tokens = set()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", _normalize(text)):
        for piece in re.split(_CJK_SPLIT_RE, run):
            piece = piece.strip()
            if len(piece) >= 2 and piece not in _CJK_STOP:
                tokens.add(piece[:8])
    return tokens


def detect_time_anchors(text: str) -> List[str]:
    """按时序返回文本命中的时间锚 canonical 列表（去重）。

    长形态先匹配并**就地抹掉**：英文 "the day before yesterday" 里含
    "yesterday"，不抹掉就会同时命中两个互斥锚，"昨天 vs 前天"这类判定直接失效
    （真机用例："我说的昨天、他答前天"正是靠这条判出来的）。
    """
    working = _normalize(text).lower()
    hits: List[str] = []
    ordered = sorted(
        _TIME_ANCHORS,
        key=lambda item: max(
            [len(form) for form in item[1] + item[2]] or [0]
        ),
        reverse=True,
    )
    for key, zh_forms, en_forms in ordered:
        matched = False
        for form in list(zh_forms) + list(en_forms):
            if form and form in working:
                working = working.replace(form, " ")
                matched = True
        if matched and key not in hits:
            hits.append(key)
    return hits


_ZH_QUESTION_RE = re.compile(r"[?？]|吗|呢|吧|什么|怎么|哪里|哪儿|哪个|谁|为什么|几")
_EN_QUESTION_RE = re.compile(
    r"\?|\b(what|when|where|who|whom|whose|which|why|how|did|do|does|"
    r"can|could|would|will|should|is|are|was|were|have|has|had)\b",
    re.IGNORECASE,
)


def is_question_turn(text: str) -> bool:
    raw = _normalize(text)
    if not raw:
        return False
    return bool(_ZH_QUESTION_RE.search(raw) or _EN_QUESTION_RE.search(raw))


def alignment_state(user_text: str, raw_text: str = "") -> Dict[str, Any]:
    """本轮"要对准什么"的确定性裁定（进 turn_contract.alignment）。

    ``user_text`` 是规范化英文（模型实际看到的那份），时间锚与英文对象从它上面
    取；中文对象从 ``raw_text``（ASR 原文）上取，只在回复也含中文时参与比对。
    """
    user_norm = _normalize(user_text)
    raw_norm = _normalize(raw_text)
    anchors = detect_time_anchors(user_norm)
    for key in detect_time_anchors(raw_norm):
        if key not in anchors:
            anchors.append(key)
    # 时间词本身不当"核心对象"：它已经由 time_anchor 单独管，重复计入会把
    # 对象集撑大、让"通篇不谈对象"这条判据变钝。
    time_words = {
        form.lower()
        for key, zh_forms, en_forms in _TIME_ANCHORS
        for form in list(zh_forms) + list(en_forms)
    } | {key for key, _, _ in _TIME_ANCHORS}
    state: Dict[str, Any] = {
        "time_anchors": anchors,
        "core_objects": sorted(
            tok for tok in content_tokens(user_norm) if tok not in time_words
        ),
        "core_objects_zh": sorted(
            tok for tok in cjk_content_tokens(raw_norm) if tok not in _CJK_STOP
        ),
        "is_question": is_question_turn(raw_norm or user_norm),
    }
    return state


# ---------------------------------------------------------------------------
# 律 0：对题（四判据，其中"更正不许复燃"并轨 0916-22，见 tutor_agent 既有判据）
# ---------------------------------------------------------------------------

def topic_alignment_violations(
    user_text: str, reply: str, raw_text: str = ""
) -> List[str]:
    """用户这一轮所问的那件事，回复是否真的对准了。"""
    violations: List[str] = []
    text = _normalize(reply)
    if not text:
        return violations
    lowered = text.lower()
    state = alignment_state(user_text, raw_text)
    anchors = list(state.get("time_anchors") or [])
    if anchors and not _TIME_CORRECTION_RE.search(text):
        reply_anchors = detect_time_anchors(text)
        missing = [a for a in anchors if a not in reply_anchors]
        crossed = [
            a for a in reply_anchors
            if a in _DAY_LEVEL_ANCHORS and a not in anchors
        ]
        # 只要用户那天的锚一个都没提，而回复落在互斥组里的**另一天**，就是
        # "我说昨天、他答前天"这一类。同锚出现即放行（哪怕同时提了别的）。
        if missing and crossed and any(a in _DAY_LEVEL_ANCHORS for a in anchors):
            violations.append(
                "对题失败·时间锚: 用户说的是 "
                + "/".join(missing)
                + "，回复却落在 "
                + "/".join(crossed)
                + "；必须谈用户说的那个时间，或先用一句明确更正（Did you mean …?），"
                "不得换成另一个时间"
            )
    if _SELF_PROCESS_RE.search(text):
        violations.append(
            "对题失败·自述理解过程: 用户要的是那件事的回应，不是听你复盘自己怎么理解错的；"
            "直接用一句话确认要点后回答内容"
        )
    # 核心对象对齐：只做同语言词面比对（中↔英没有翻译层，不假装能验）。
    reply_en = content_tokens(text)
    user_en = set(state.get("core_objects") or [])
    if len(user_en) >= 2 and reply_en and not (user_en & reply_en):
        user_stems = {_stem(tok) for tok in user_en}
        if not (user_stems & {_stem(tok) for tok in reply_en}):
            violations.append(
                "对题失败·核心对象: 用户这轮谈的是 "
                + "/".join(sorted(user_en)[:4])
                + "，回复通篇没有出现其中任何一个（也没出现同指代）；必须回到用户说的那件事上"
            )
    if _CJK_RE.search(text):
        reply_zh = cjk_content_tokens(text)
        user_zh = set(state.get("core_objects_zh") or [])
        if len(user_zh) >= 2 and reply_zh and not (user_zh & reply_zh):
            violations.append(
                "对题失败·核心对象(中文): 回复谈的不是用户这轮说的 "
                + "/".join(sorted(user_zh)[:4])
                + "；先对准那件事再往下说"
            )
    # 问句必须被回答：整段只有反问、没有任何陈述句 = 用反问代替回答。
    if state.get("is_question"):
        sentences = [s for s in re.split(r"(?<=[.!?。！？])\s*", text) if s.strip()]
        if sentences and all(s.rstrip().endswith(("?", "？")) for s in sentences):
            violations.append(
                "对题失败·只反问不回答: 用户本轮在问问题，回复整段都是反问句；"
                "必须先给实质答案（最多一句反问跟在答案后面）"
            )
    return violations


# ---------------------------------------------------------------------------
# 三律（按我们的产品目标改写，形态随人格变）
# ---------------------------------------------------------------------------

_ACK_ONLY_RE = re.compile(
    r"^(?:got\s+it|okay|ok|right|i\s+see|sure|alright|understood|makes\s+sense|"
    r"i\s+understand|nice|good|great|hmm+|ah+|oh+|"
    r"好的|好|嗯+|哦+|明白|明白了|知道了|收到|了解|是的|对)[\s,.!。！？~、，]*$",
    re.IGNORECASE,
)
# 认错开场（承认律：只说 AI 自己的错，且必须紧跟修正）。
_SELF_EXCUSE_OPEN_RE = re.compile(
    r"^\s*(?:"
    r"(?:我)?(?:没有|没|查不到|找不到)(?:任何)?(?:记录|纪录|相关记录|相关纪录)|"
    r"我(?:这边)?(?:没有|没)(?:看到|查到|找到)(?:任何)?(?:记录|纪录|相关)|"
    r"i\s+(?:don['’]?t|do\s+not|can['’]?t|cannot)\s+(?:have|see|find|recall)\s+"
    r"(?:any\s+)?(?:record|records|record\s+of|that\s+in\s+my\s+records)|"
    r"i\s+(?:don['’]?t|do\s+not)\s+have\s+(?:that|any)\s+information|"
    r"there\s+(?:is|are)\s+no\s+records?\s+of"
    r")",
    re.IGNORECASE,
)
# 用户明确指出"AI 理解错了/我纠正 AI 的误解"（承认律的作用域只在 AI 自身错误）。
_AI_MISREAD_USER_RE = re.compile(
    r"(我没(?:有)?说|我没讲过|我不是(?:说|那个意思|这个意思)|这不是我的意思|"
    r"你(?:理解|搞|弄)错|你(?:说|想)歪|我说的是|我(?:的意思)?是(?:说)?|"
    r"i\s+did\s+not\s+say|i\s+didn['’]?t\s+say|i\s+never\s+said|"
    r"that['’]?s\s+not\s+what\s+i\s+(?:said|meant)|you\s+got\s+it\s+wrong|"
    r"what\s+i\s+meant\s+was)",
    re.IGNORECASE,
)


def reply_quality_violations(
    user_text: str,
    reply: str,
    *,
    raw_text: str = "",
    scene_role: bool = False,
    confirmation_turn: bool = False,
) -> List[str]:
    """三律判定（语言增量 / 承认收窄 / 禁复述）。

    ``scene_role=True``（戏内角色人格）时，"增量"是推进情节而不是教书，
    短台词（"Here you are."）是合法的——只保留"整段复述用户的话"这一条，
    零信息短句不判。
    ``confirmation_turn=True``（低置信轮，plans/0916-26 修复3）时跳过复述判据：
    那一轮**被合同要求**先复述确认，再拿"复述"去罚它就是自相矛盾。
    """
    violations: List[str] = []
    text = _normalize(reply)
    if not text:
        return violations
    reply_tokens = content_tokens(text)
    reply_zh = cjk_content_tokens(text)
    user_tokens = content_tokens(user_text)
    user_zh = cjk_content_tokens(raw_text or user_text)

    # ① 零信息过渡句（禁纯过渡/纯附和）：全篇没有承载任何内容词。
    if not scene_role and not reply_tokens and not reply_zh and len(text) <= 60:
        if _ACK_ONLY_RE.match(text) or len(_WORD_RE.findall(text)) <= 5:
            violations.append(
                "增量为零·纯过渡句: 回复只是附和/过渡，没有给出任何具体内容；"
                "必须至少带一个具体信息（一个说法、一件事、一个细节），"
                "并把话头交回用户"
            )

    # ② 复述当回应（两套人格都保留）：整段就是用户原话的换词重排——**一个
    #    用户没说过的新内容词都没有**（"So you mean you went to the beach with
    #    your brother?" 就是这一类：占了他的开口预算、零增量）。低置信确认轮
    #    例外（那是合同要求的）。
    if not confirmation_turn and reply_tokens and user_tokens:
        if not (reply_tokens - user_tokens) and len(reply_tokens) >= (2 if scene_role else 3):
            violations.append(
                "增量不足·复述当回应: 回复把用户的话换个说法抛了回去，占用他的开口预算且零增量；"
                "要么给一个新的说法/更地道的表达，要么顺着他的话往下推进并留出接话位"
            )
    elif not confirmation_turn and reply_zh and user_zh and not scene_role:
        overlap = len(reply_zh & user_zh) / float(len(reply_zh))
        if overlap >= 0.9 and len(reply_zh) >= 3:
            violations.append(
                "增量不足·复述当回应(中文): 回复只是重复用户的话；给出新的信息或把话头交回用户"
            )

    # ③ 承认收窄：只对"AI 自身错误"认错，且不许以"我没记录…"这类
    #    自证清白开头（真机事故：用户纠正后，回复以"我没有记录…"开头）。
    if _AI_MISREAD_USER_RE.search(raw_text or user_text or "") and _SELF_EXCUSE_OPEN_RE.match(text):
        violations.append(
            "承认律·自证清白开场: 用户在纠正我的理解，回复却以"
            "“我没有记录/我没看到相关记录”开头自证清白；先一句认下"
            "（我理解错了），再立刻给出对准他原意的修正"
        )
    return violations


# ---------------------------------------------------------------------------
# 兜底：宁可不硬答，也不答错题（方案的"退化为一次简短确认"）
# ---------------------------------------------------------------------------

_ANCHOR_EN = {
    "yesterday": "yesterday",
    "day_before": "the day before yesterday",
    "today": "today",
    "tomorrow": "tomorrow",
    "last_week": "last week",
    "last_month": "last month",
    "last_year": "last year",
    "this_morning": "this morning",
    "just_now": "just now",
    "last_weekend": "last weekend",
}


def alignment_confirm_reply(
    user_text: str, reply: str, raw_text: str = ""
) -> Optional[str]:
    """律 0 命中且没有确定性修法时，退化成的**一句简短确认**（不硬答）。

    只在"时间锚被换成了另一天"这一最高置信形态上启用——对象/问句两类判据
    的误报代价更高（同义词、简答都是正常语言），只标记不替换。
    """
    anchors = alignment_state(user_text, raw_text).get("time_anchors") or []
    day_anchors = [a for a in anchors if a in _DAY_LEVEL_ANCHORS]
    if not day_anchors:
        return None
    violations = [
        v for v in topic_alignment_violations(user_text, reply, raw_text)
        if v.startswith("对题失败·时间锚")
    ]
    if not violations:
        return None
    anchor_en = _ANCHOR_EN.get(day_anchors[0], day_anchors[0])
    return (
        f"Just to make sure I got you right — you mean {anchor_en}? "
        "Let me answer that."
    )
