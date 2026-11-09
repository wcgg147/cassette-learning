# -*- coding: utf-8 -*-
"""词块闭环的确定性层（plans/0916-11）。

为什么独立成模块：F10 归纳闸门（§4.2.1「代码硬闸门」）与后续 F13 匹配器 /
F14 对比器都是**纯函数、零 LLM 调用、可离线单测**的判定层。生成层（模型）
只负责提候选，"什么能进库"由这里说了算——项目原则：代码算出来才权威。

一期（本文件目前范围）：
  - normalize_chunk / chunk_word_count    归一化与词数
  - gate_chunk_candidates                 归纳闸门（§4.2 + §4.2.1 全部硬规则）
  - attach_chunk_source                   来源指针（来自哪盘 / 当时在聊什么）

硬规则（用户口径 2026-09-16，勿删）：
  * 「本库只收多词单位」——单词一律不进库（单词的释义需求已由点词卡片覆盖，
    本子补的是点词卡覆盖不了的那层：搭配/句型/话语标记/短语动词）；
  * 可迁移性必须自证——候选必须带「另一个情境的例句」，给不出或例句
    仍是本场对话的内容 → 直接丢；
  * 误报比漏报更伤（提前恭喜会稀释信任）——判据从紧、宁可漏。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

# ── kind 枚举 ────────────────────────────────────────────────────────
# 方案 §4.1 列了 collocation|frame|marker|phrasal_verb 四种；§4.2.1 的
# 进库样例里有 "It depends（固定表达）"，四分类装不下 → 增补 fixed。
# （实现报告里记一笔偏离。）
CHUNK_KINDS = {"collocation", "frame", "marker", "phrasal_verb", "fixed"}

# 状态机（§4.1）：candidate → saved → activated → mastered；dismissed = 用户
# 明确不想要（区别于撤销收藏退回 candidate）。append-only：所有流转进
# status_history 留底（照 0915-12 superseded 形状）。
CHUNK_STATUSES = {"candidate", "saved", "activated", "mastered", "dismissed"}

# 词数硬闸门（§4.2.1）：≥2 词（单词不进库）且 ≤8 词（禁整句独白）。
_CHUNK_MIN_WORDS = 2
_CHUNK_MAX_WORDS = 8

# 尾部悬空虚词：以这些词结尾的候选几乎必然是截断切片（"I went to the"）。
# 刻意**不含 "of"**——"a couple of" / "a bit of" / "kind of" 都以 of 收尾
# 且是标准词块；名单只收决定词/物主代词/宾格代词这类悬空后无法收尾的词，
# 外加连不成的连词（没有词块以 and/but/or 收尾）。
_TRAILING_DANGLING = {
    "the", "a", "an",
    "my", "your", "his", "her", "its", "our", "their",
    "him", "them", "us", "me", "and", "but", "or",
}
# 首部悬空虚词：以连接词/系动词开头的候选是句子切片，不是预制单位。
# 刻意**不含冠词 "a/the"**——"a touch of" / "a couple of" 以冠词开头且是
# 标准词块；"to be honest" 的首词 to 同理（to 在固定形式里）。
_LEADING_DANGLING = {
    "and", "but", "or", "because", "so",
    "was", "were", "is", "are", "am",
}

_NUM_WORD_RE = re.compile(r"^[a-z0-9']+$")

# ── 垃圾形态闸门（plans/0916-16 修复项 4）──────────────────────────────────
#
# 真机实测：用户打开词块本看到 16 条，其中至少 5 条是垃圾——
#   · `Do you know what story this game tells`（**AI 自己的问句**被当词块收走）
#   · `What does... mean?`（占位符句型）
#   · `I was asking about Black Ops 6, not other games.`（碎片被"整理成"整句）
# 这些条目用户直接看得见，所以必须在**代码层**拦掉（prompt 只是初筛，代码才是权威）。
_QUESTION_TAIL_RE = re.compile(r"[?？]\s*$")
# 助动词/系动词开头 + 主语：`Do you know…` / `Can I get…`。刻意不拦 `have a look`
# 这类（have 后面跟的是冠词/名词，不是主语代词），它们是标准词块。
_QUESTION_OPEN_RE = re.compile(
    r"^(do|does|did|can|could|would|should|will|shall|may|might|must|"
    r"is|are|was|were|am|have|has|had)\s+"
    r"(you|i|we|they|he|she|it|there)\b", re.IGNORECASE)
# 疑问词开头 + 助动词/主语：`What does ... mean` / `How do you ...`。
# 同样避开 `how come` / `what's up` / `how about you` 这类真词块。
_WH_QUESTION_RE = re.compile(
    r"^(what|why|how|when|where|who|which)\s+"
    r"(do|does|did|is|are|was|were|can|could|would|will|should|you|i|we|they|it|there)\b",
    re.IGNORECASE)
# 占位符/省略号：模型把"句型模板"当成词块报上来的典型形态。
_PLACEHOLDER_RE = re.compile(r"\.\.\.|…|___|\bxxx\b|某某")
# 整句独白：≥7 词且"主语 + 限定动词"齐全（碎片被整理成整句的形态）。
_SENTENCE_SV_RE = re.compile(
    r"\b(i|you|we|they|he|she|it|there)\s+"
    r"(was|were|am|is|are|have|has|had|do|does|did|"
    r"don't|doesn't|didn't|won't|can|could|should|would|will)\b",
    re.IGNORECASE)
_FULL_SENTENCE_WORDS = 7

# 虚词表（只用于"有没有实义词"这一个判定：全是虚词的串不值得进库）
_FUNCTION_WORDS = {
    "the", "and", "for", "with", "you", "your", "his", "her", "its", "our", "their",
    "that", "this", "these", "those", "have", "has", "had", "was", "were", "are",
    "is", "am", "not", "but", "or", "so", "if", "of", "to", "in", "on", "at", "by",
    "as", "it", "he", "she", "they", "we", "do", "does", "did", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "there", "here",
    "what", "when", "where", "which", "who", "why", "how", "than", "then", "too",
    "very", "just", "also", "from", "about", "into", "over", "out", "up", "down",
}


def looks_like_asr_garbage(text: str) -> bool:
    """ASR 疑似乱码 → 放弃采集（plans/0916-16 修复项 4-3）。**宁可漏，不可污染。**

    真机反例：用户说的一句被识别成 `l i i was asking that game of cod…`，
    孤立单字母成串；这种串被归纳侧一"整理"，就成了库里一条看着挺顺的句子。
    英文里孤立单字母几乎只出现在乱码或逐字母拼读中，所以按占比判；另外
    整串没有一个实义词的也丢掉。
    """
    tokens = normalize_chunk(text).split()
    if not tokens:
        return True
    singles = [t for t in tokens if len(t) == 1 and t.isalpha()]
    if len(singles) >= 2 and len(singles) / len(tokens) >= 0.3:
        return True
    return not [t for t in tokens if len(t) >= 3 and t not in _FUNCTION_WORDS]


def normalize_chunk(text: str) -> str:
    """归一化形（查重用）：小写、去标点、折叠空白。"""
    lowered = " ".join(str(text or "").lower().split())
    return "".join(ch if _NUM_WORD_RE.match(ch) else (" " if not ch.isalnum() else ch)
                   for ch in lowered).strip()


def chunk_word_count(text: str) -> int:
    """词数：按归一化后的空白切分。"""
    return len([t for t in normalize_chunk(text).split() if t])


def _token_subseq_in(normalized_chunk: str, normalized_text: str) -> bool:
    """词块是否作为连续 token 子序列出现在 normalized_text 里。"""
    c, t = normalized_chunk.split(), normalized_text.split()
    if not c or len(c) > len(t):
        return False
    return any(t[i:i + len(c)] == c for i in range(len(t) - len(c) + 1))


def chunk_junk_reason(text: str, word_count: int = 0) -> str:
    """这条候选是不是"用户看得见的垃圾"？返回原因，空串 = 不垃圾。

    抽成纯函数是为了**两个入口共用同一套判据**：归纳（对话归档时）与素材导入
    （plans/0916-24 流水线⑤：从素材文本里抽词块）。两处若各写一套，迟早分叉——
    真机上"词块本里有垃圾"就是这么来的。
    """
    body = " ".join(str(text or "").split())
    if not body:
        return "empty"
    wc = word_count or chunk_word_count(body)
    norm = normalize_chunk(body)
    if wc < _CHUNK_MIN_WORDS or wc > _CHUNK_MAX_WORDS:
        return "word_count"
    if _QUESTION_TAIL_RE.search(body) or _QUESTION_OPEN_RE.match(norm) \
            or _WH_QUESTION_RE.match(norm):
        return "question"
    if _PLACEHOLDER_RE.search(body):
        return "placeholder"
    if wc >= _FULL_SENTENCE_WORDS and _SENTENCE_SV_RE.search(norm):
        return "whole_sentence"
    tokens = norm.split()
    if tokens and (tokens[-1] in _TRAILING_DANGLING or tokens[0] in _LEADING_DANGLING):
        return "dangling"
    return ""


def gate_chunk_candidates(raw: list, messages: list,
                          known_expressions: Iterable[str] = (),
                          max_candidates: int = 3) -> list[dict]:
    """归纳闸门（§4.2 五条 + §4.2.1 三判据的代码可验部分）。

    参数
    ----
    raw                LLM 产出的候选列表（dict：text/kind/gloss/example/slot/reason/usable）
    messages           本盘消息（[{role, content}]，原始转写，保真原则的输入）
    known_expressions  用户已会的表达（mastered_expressions + 已收藏词块的 norm），用于去重
    max_candidates     单盘候选上限（V9：≤3 条，不给垃圾）

    闸门（全部确定性，不花模型）：
      1. 会话用户发言 < 4 轮 → 一律不抽；
      2. 只收多词单位：2–8 词，单词一律丢弃；
      3. 非整句/非切片：首尾悬空虚词丢弃；必须带「另一个情境的例句」，
         例句是本场对话原文的子串 → 视为"仍是本场那件事"，丢弃；
      4. 与已会表达 / 已收藏词块去重（归一化比较）；
      5. LLM 自评 usable=False 的丢弃；kind 越界的丢弃。
      6. 真机实测的三类垃圾形态（plans/0916-16 修复项 4）：疑问句、占位符/省略号句型、
         整句独白（≥7 词且主语+限定动词齐全）、以及"AI 整句被当词块"。
    通过全部闸门的候选按原顺序返回，最多 max_candidates 条。
    """
    if not isinstance(raw, list):
        return []
    user_turns = [m for m in (messages or [])
                  if isinstance(m, dict) and str(m.get("role") or "") == "user"
                  and str(m.get("content") or "").strip()]
    # 闸门 1：太短没料就不出（V9 原话）。
    if len(user_turns) < 4:
        return []

    transcript_norms = [normalize_chunk(str(m.get("content") or "")) for m in messages]
    ai_utterances = [normalize_chunk(str(m.get("content") or "")) for m in (messages or [])
                     if isinstance(m, dict) and str(m.get("role") or "") == "assistant"]
    known_norms = {normalize_chunk(str(k)) for k in (known_expressions or ()) if str(k or "").strip()}
    known_norms.discard("")

    out: list[dict] = []
    seen_norms: set[str] = set()
    for cand in raw:
        if not isinstance(cand, dict) or len(out) >= max_candidates:
            break
        text = " ".join(str(cand.get("text") or "").split())
        if not text:
            continue
        # 闸门 5a：LLM 自评不可用 → 丢弃（照 0915 自动补库 usable 闸门先例）。
        if cand.get("usable") is False:
            continue
        # 闸门 5b：kind 越界 → 丢弃。
        kind = str(cand.get("kind") or "").strip().lower()
        if kind not in CHUNK_KINDS:
            continue
        # 闸门 2：只收多词单位（用户口径：单词不值得建库）。
        wc = chunk_word_count(text)
        if wc < _CHUNK_MIN_WORDS or wc > _CHUNK_MAX_WORDS:
            continue
        norm = normalize_chunk(text)
        if not norm or norm in seen_norms or norm in known_norms:
            continue
        # 闸门 3a：切片/悬空首尾。
        tokens = norm.split()
        if tokens[-1] in _TRAILING_DANGLING or tokens[0] in _LEADING_DANGLING:
            continue
        # 闸门 3b：可迁移性自证——必须带另一个情境的例句。
        example = " ".join(str(cand.get("example") or "").split())
        if not example:
            continue
        ex_norm = normalize_chunk(example)
        # 例句是本场原文的逐字子串 → "仍是本场那件事"，不是词块。
        if ex_norm and any(ex_norm in tn or _token_subseq_in(ex_norm, tn)
                           for tn in transcript_norms if tn):
            continue
        # 闸门 4：用户已能主动说出的过滤（整场原文里以连续 token 出现）。
        # 注：AI 说过的不拦——F10 取材优先级②就是"AI 用过的地道表达"。
        if any(_token_subseq_in(norm, tn) for tn in
               (normalize_chunk(str(m.get("content") or ""))
                for m in user_turns) if tn):
            continue
        # 闸门 6（plans/0916-16 修复项 4）：真机上"用户看得见的垃圾"三类形态。
        # 判据在 chunk_junk_reason 里（素材导入那条链共用同一套，避免两处判据分叉）。
        if chunk_junk_reason(text, word_count=wc):
            continue
        # 6d：**AI 的整句**不得成为词块。注意与闸门 4 的分寸：AI 用过的地道**短语**
        # 仍放行（F10 取材优先级②），这里只拦"AI 那一整句话被原样搬进库"。
        if any(norm == ai_norm for ai_norm in ai_utterances if ai_norm):
            continue
        seen_norms.add(norm)
        out.append({
            "text": text[:80],
            "norm": norm,
            "kind": kind,
            "gloss": " ".join(str(cand.get("gloss") or "").split())[:60],
            "example": example[:160],
            "slot": " ".join(str(cand.get("slot") or "").split())[:30],
            "reason": " ".join(str(cand.get("reason") or "").split())[:120],
        })
    return out


def attach_chunk_source(candidates: list[dict], messages: list,
                        conv_id: str = "", source_type: str = "tape") -> list[dict]:
    """给候选补来源指针（§4.3：来自哪盘磁带 · 当时在聊什么）。

    context_snippet 取**包含该词块的那句话**：优先用户说的（卡壳点语境），
    其次 AI 说的（地道表达语境）；source_turn 是该句在消息流里的 1-based 序号。
    找不到包含句就用首条用户消息（仍能唤起"我在聊什么"的记忆）。
    """
    for cand in candidates or []:
        norm = str(cand.get("norm") or "")
        snippet, turn = "", 0
        for pass_role in ("user", "assistant"):
            for idx, m in enumerate(messages or []):
                if not isinstance(m, dict):
                    continue
                content = " ".join(str(m.get("content") or "").split())
                if not content or (m.get("role") or "") != pass_role:
                    continue
                if norm and _token_subseq_in(norm, normalize_chunk(content)):
                    snippet, turn = content[:160], idx + 1
                    break
            if snippet:
                break
        if not snippet:
            for idx, m in enumerate(messages or []):
                if isinstance(m, dict) and (m.get("role") or "") == "user" \
                        and str(m.get("content") or "").strip():
                    snippet, turn = " ".join(str(m["content"]).split())[:160], idx + 1
                    break
        cand["source_type"] = source_type
        cand["source_id"] = str(conv_id or "")
        cand["source_turn"] = turn
        cand["context_snippet"] = snippet
    return candidates


# ── D2 候选质量闸（plans/0923-05 §4.1）────────────────────────────────
# 形态闸（gate_chunk_candidates）只保证「不是垃圾形态」；用户口径要的是
# 「值得整块带走的表达」。两者互补、都要过：先过质量闸（有证据才收），
# 再过形态闸。判据是**方法借鉴**（Martinez 2013 频率与语义非透明性 /
# Boers & Lindstromberg 2009 固定性与可记忆性 / Simpson-Vlach & Ellis 2010
# 教学价值），线上不伪算 MI 与语料频次，只认归档模型给出的保守分类。

# frequency_band 只认这一档；uncertain / rare 一律不收（误报比漏报更伤）。
_QUALITY_COMMON_BAND = "common"

# 空泛自夸：写了等于没写，按「缺证据」拒绝（§4.1）。
_VAGUE_EVIDENCE = frozenset({
    "很常用", "常用", "很常用表达", "高频", "高频表达", "常见", "经常用到",
    "日常常用", "口语常用", "值得学", "值得学习", "很有用", "有用", "很实用",
    "实用", "地道", "很地道", "地道表达", "值得收", "值得收藏", "值得记",
    "推荐", "好短语", "好表达", "不错", "很好", "可以学", "建议学习", "要多用",
    "useful", "very useful", "common", "very common", "frequently used",
    "high frequency", "worth learning", "worth remembering", "worth knowing",
    "good phrase", "nice phrase", "good expression", "important", "handy",
    "useful phrase", "common phrase",
})


def _strict_true(value) -> bool:
    """只有真布尔 True 或字符串 "true" 算真（§4.1：布尔字符串不按真值放行）。"""
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _is_specific_evidence(text) -> bool:
    """非空、且不是空泛自夸——空泛的等同没给证据。"""
    body = " ".join(str(text or "").split())
    if not body:
        return False
    return body.strip().lower() not in _VAGUE_EVIDENCE


def _appears_in_conversation(text: str, messages) -> bool:
    """该文本是否就是本场对话的内容（归一化后包含 / 连续 token 子序列）。"""
    norm = normalize_chunk(text)
    if not norm:
        return False
    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        tn = normalize_chunk(str(m.get("content") or ""))
        if tn and (norm in tn or _token_subseq_in(norm, tn)):
            return True
    return False


def chunk_quality_reject_reason(cand: dict, messages=()) -> str:
    """候选未过 D2 三筛的原因；空串 = 通过。纯函数、零模型调用。

    三筛（§4.1，缺一不可）：
      ① `frequency_band` 严格为 `common`；
      ② `transfer_scene` 有具体内容且指向**不同于本场对话**的场景，
         且既有 `example` 不是本场原文；
      ③ `teaching_worth` 严格为真、`teaching_reason` 有具体内容，且
         `nontransparent_or_conventional` / `fixed_pattern` / 具体 `memory_hook`
         三项至少占一项。
    """
    if not isinstance(cand, dict):
        return "not_dict"
    # ① 频率档：只收 common（uncertain / rare / 缺失一律不收）。
    band = str(cand.get("frequency_band") or "").strip().lower()
    if band != _QUALITY_COMMON_BAND:
        return "frequency_band"
    # ② 换场景可迁移：场景说明要具体，且不能再是本场对话。
    scene = " ".join(str(cand.get("transfer_scene") or "").split())
    if not scene:
        return "transfer_scene_missing"
    if not _is_specific_evidence(scene):
        return "transfer_scene_vague"
    if _appears_in_conversation(scene, messages):
        return "transfer_scene_same_conversation"
    example = " ".join(str(cand.get("example") or "").split())
    if not example:
        return "example_missing"
    if _appears_in_conversation(example, messages):
        return "example_same_conversation"
    # ③ 教学价值有据：worth 为真 + 理由具体 + 三项证据至少一项。
    if not _strict_true(cand.get("teaching_worth")):
        return "teaching_worth"
    if not _is_specific_evidence(cand.get("teaching_reason")):
        return "teaching_reason_missing"
    hook = " ".join(str(cand.get("memory_hook") or "").split())
    if not (_strict_true(cand.get("nontransparent_or_conventional"))
            or _strict_true(cand.get("fixed_pattern"))
            or _is_specific_evidence(hook)):
        return "no_concrete_evidence"
    return ""


def filter_chunk_quality(raw: list, messages=()) -> list[dict]:
    """D2 质量闸 + 直译性闸：只留下有证据支撑、且**不能靠直译猜出意思**的候选。

    在 `save_conversation` 的词块支路**先**跑本函数、**再**跑形态闸
    `gate_chunk_candidates`。卡壳即时回收（用户自己说出的需求）不走这条路。

    两道闸都要过（保序、不改字段）：
      · `chunk_quality_reject_reason` —— D2 三筛（常见 / 跨场景 / 值得专门教）；
      · `chunk_transparency_reject_reason` —— 直译性（2026-09-24，用户口径）。
    """
    if not isinstance(raw, list):
        return []
    return [c for c in raw
            if isinstance(c, dict)
            and not chunk_quality_reject_reason(c, messages)
            and not chunk_transparency_reject_reason(c)]


# ── 直译性闸（2026-09-24，用户口径）──────────────────────────────────
#
# 用户原话："'think too much' 这种……人家只要学过这三个单词，组合起来他都知道是啥
# 意思，一眼就能直译过来（想太多）——**没有必要收**"；"by the way / according to
# 这种，你不学、或者说你不知道，你就不知道它到底是怎样的意思——**你不能光靠单独
# 拆开来的直译就知道它的意思**"（这种才收）。
#
# 为什么代码只能**复核**、不能**判定**：直译推不推得出意思是**语义判断**，纯词面
# 信号分不开——"think too much" 与 "by the way" 的成分全是最高频词。所以分工是：
#   **模型给证据**：`literal_gloss`（逐词直译）+ `gloss`（实际意思/用法）
#     + `nontransparent_or_conventional`（自报结论）；
#   **代码做确定性复核**：自报不透明是**硬门槛**（不再是 D2 三筛里三选一的"或"，
#     那道口子让 "think too much" 带个记忆钩子就能进）、缺证据拒、
#     **直译与实际意思高度相似 = 证据自相矛盾 → 拒**。
# 诚实边界：复核只能抓"模型自己的证据互相矛盾"；模型判断本身错了（给了不老实
# 的 literal_gloss）代码看不出来——所以判据要写得足够具体（用户原话例子进了
# 两条提取提示词当锚点），测试钉住这三类样本。

_TRANSPARENT_SIMILARITY = 0.80
# 逐字覆盖率阈值：直译里的字有 2/3 以上出现在实际意思里 ⇒ 意思就是词义的拼装。
# 选字符覆盖而不是词面相似度，是因为真实的"一眼直译"往往带着改写——
# "想太多" 的实际意思会写成 "想得太多、过度思考"，字符二元组相似度只有 0.14，
# 但 想/太/多 三个字**全部**出现在释义里；而 "by the way"（在这条路旁边 →
# 顺便说一句）一个字都不沾。这个对比正是"透明 vs 不透明"的词面形状。
_LITERAL_CHAR_COVERAGE = 0.67


def _gloss_similarity(a: str, b: str) -> float:
    """两条释义的相似度（字符二元组 Jaccard，0~1）。有一方空串按 0 处理。"""
    def grams(s: str) -> set:
        body = re.sub(r"\s+", "", str(s or ""))
        if not body:
            return set()
        if len(body) == 1:
            return {body}
        return {body[i:i + 2] for i in range(len(body) - 1)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _literal_char_coverage(literal: str, gloss: str) -> float:
    """直译里的有效字符有多大比例出现在实际意思里（0~1）。

    只数 CJK/字母/数字（标点和空白不算证据）；直译为空按 0 处理。
    """
    body = re.sub(r"[^\w\u4e00-\u9fff]", "", str(literal or ""), re.UNICODE)
    if not body:
        return 0.0
    pool = set(re.sub(r"[^\w\u4e00-\u9fff]", "", str(gloss or ""), re.UNICODE))
    if not pool:
        return 0.0
    hit = sum(1 for ch in body if ch in pool)
    return hit / len(body)


def chunk_transparency_reject_reason(cand: dict) -> str:
    """这条词块的意思能不能**靠直译猜出来**？能猜出来的不收。空串 = 通过。

    纯函数、零模型调用；与 归纳 / 素材库 / 素材盘 三条入库路径共用（判据只写这一份，
    两处各写一套迟早分叉——真机上"词块本里有垃圾"就是这么来的）。
    """
    if not isinstance(cand, dict):
        return "not_dict"
    # ① 硬门槛：模型自报"意思能靠字面推出来" ⇒ 这条就是用户说的"没必要收"。
    #    （原 D2 三筛里它与 fixed_pattern / memory_hook 三选一——那是口子。）
    if not _strict_true(cand.get("nontransparent_or_conventional")):
        return "not_nontransparent"
    # ② 没给逐词直译 = 没法复核 ⇒ 按缺证据拒（与 D2 的"拿不准就别报"同一口径）。
    literal = " ".join(str(cand.get("literal_gloss") or "").split())
    if not literal:
        return "missing_literal_gloss"
    gloss = " ".join(str(cand.get("gloss") or "").split())
    if not gloss:
        return ""
    # ③ 证据自相矛盾：直译 ≈ 实际意思 ⇒ "一眼就能直译过来"，正是用户点名的形态。
    #    三条规则（任一命中即拒）：整条包含、"直译逐字覆盖率"过高（意思=词义拼装）、
    #    字符二元组几乎重合。详见 `_literal_char_coverage` 的注释——真实的"一眼直译"
    #    往往带着改写，只有覆盖率这一条抓得住 "想太多 → 想得太多、过度思考"。
    if literal in gloss or gloss in literal:
        return "literal_matches_meaning"
    if _literal_char_coverage(literal, gloss) >= _LITERAL_CHAR_COVERAGE:
        return "literal_matches_meaning"
    if _gloss_similarity(literal, gloss) >= _TRANSPARENT_SIMILARITY:
        return "literal_matches_meaning"
    return ""


# ── D6 词块本分桶与分页（plans/0923-05 §4.4）────────────────────────
# 依据 Lindstromberg & Boers (2008) 的记忆与循环原则——把待复现的内容
# 保持可见。**这不是新的间隔调度**：只改读侧展示分组，不改 F12 可注入
# 资格、F13 计数、存储状态，也不删除任何行。

NOTEBOOK_BUCKETS = ("candidate_recent", "candidate_quiet", "owned", "mastered")
NOTEBOOK_RECENT_DAYS = 30
NOTEBOOK_PAGE_MAX = 50


def _parse_utc_ts(raw):
    """把 created_at / updated_at / last_use_at / status_history.at 统一解析成
    **带时区的 UTC datetime**；解析不了返回 None（不抛错＝按"时间不可用"处理）。

    两种后端形态都要吃：SQLite 给 "YYYY-MM-DD HH:MM:SS"（CURRENT_TIMESTAMP，
    已是 UTC），PG 给 timestamptz datetime，status_history 里是 ISO 串。
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except (ValueError, TypeError):
            dt = None
        if dt is None:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except (ValueError, TypeError):
                    continue
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def notebook_candidate_since(row):
    """该候选**进入 candidate 状态**的时间（§4.4）。

    只认 status_history 里最后一次「`from != 'candidate'` 且 `to == 'candidate'`」
    的 `at`；没有则退回 `created_at`。重复归档会刷新 `updated_at`、同状态重复写
    也可能留历史，**都不能**把旧候选算成新候选。
    """
    latest = None
    for item in loads_json_list(row.get("status_history")):
        if not isinstance(item, dict):
            continue
        if str(item.get("to") or "").strip().lower() != "candidate":
            continue
        if str(item.get("from") or "").strip().lower() == "candidate":
            continue
        ts = _parse_utc_ts(item.get("at"))
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    if latest is not None:
        return latest
    return _parse_utc_ts(row.get("created_at"))


def notebook_recent_use(row, now) -> bool:
    """`last_use_at` 是否落在 now 往前 30 天内；空值/无效值＝未用
    （**不拿 `updated_at` 代替**，§4.4）。"""
    ts = _parse_utc_ts(row.get("last_use_at"))
    if ts is None:
        return False
    delta = now - ts
    return timedelta(0) <= delta < timedelta(days=NOTEBOOK_RECENT_DAYS)


def notebook_bucket_of(row, now=None) -> str:
    """该行落在哪个分组；`dismissed` / 未知状态返回 ""（不出现在本子清单）。"""
    now = now or datetime.now(timezone.utc)
    status = str(row.get("status") or "").strip().lower()
    if status in ("saved", "activated"):
        return "owned"
    if status == "mastered":
        return "mastered"
    if status != "candidate":
        return ""
    if notebook_recent_use(row, now):
        return "candidate_recent"
    since = notebook_candidate_since(row)
    if since is not None and (now - since) < timedelta(days=NOTEBOOK_RECENT_DAYS):
        return "candidate_recent"
    return "candidate_quiet"


def _desc_ts(ts):
    """时间倒序键：有效时间在前（越大越前），None 排末。"""
    return (1, 0.0) if ts is None else (0, -ts.timestamp())


def _desc_id(row):
    """id 倒序键：数字 id 用数值倒序，非数字退回字符串倒序（稳定、同型可比较）。"""
    raw = str(row.get("id") or "")
    return (0, -int(raw)) if raw.isdigit() else (1, raw)


def _notebook_sort_key(row, bucket, now):
    """组内排序键。时间戳无效一律排末，绝不抛错（§4.4/§5 D6 时间边界）。"""
    if bucket == "candidate_recent":
        # 先排最近 30 天有使用记录的，再按 candidate_since 新到旧，最后 id
        return (0 if notebook_recent_use(row, now) else 1,
                _desc_ts(notebook_candidate_since(row)),
                _desc_id(row))
    if bucket == "candidate_quiet":
        return (_desc_ts(notebook_candidate_since(row)), _desc_id(row))
    # owned / mastered：沿用旧 list_chunks 的"新收/新动优先"
    return (_desc_ts(_parse_utc_ts(row.get("updated_at"))), _desc_id(row))


def group_notebook_chunks(rows, now=None) -> dict:
    """把某个用户的所有行按 §4.4 分桶并组内排序；返回 {bucket: [row, ...]}。

    `dismissed` 不进任何桶（库行与历史保留）。纯函数：不读库、不写库。
    """
    now = now or datetime.now(timezone.utc)
    out = {b: [] for b in NOTEBOOK_BUCKETS}
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        bucket = notebook_bucket_of(row, now)
        if bucket:
            out[bucket].append(row)
    for bucket in NOTEBOOK_BUCKETS:
        out[bucket].sort(key=lambda r, b=bucket: _notebook_sort_key(r, b, now))
    return out


def page_notebook_chunks(rows, bucket: str, offset: int = 0,
                         limit: int = NOTEBOOK_PAGE_MAX, now=None) -> dict:
    """单个分组的切页：返回 `{chunks, total, next_offset}`。

    `limit <= 0` 只返回计数（折叠组只要数字）；取行时**一页最多
    NOTEBOOK_PAGE_MAX 条**。分桶、排序、切页都在这个**两后端共用**的纯函数里
    做——旧 `list_chunks` 的 200/500 截断在这里不适用，"放久的候选"逐页取回，
    第 201 条也能到。非法 `bucket` 当空组（不抛错、不泄露别的分组）。
    """
    now = now or datetime.now(timezone.utc)
    grouped = group_notebook_chunks(rows, now)
    if bucket not in NOTEBOOK_BUCKETS:
        return {"chunks": [], "total": 0, "next_offset": None}
    items = grouped[bucket]
    total = len(items)
    try:
        safe_offset = max(0, int(offset))
    except (TypeError, ValueError):
        safe_offset = 0
    raw_limit = NOTEBOOK_PAGE_MAX if limit is None else limit
    try:
        safe_limit = max(0, min(int(raw_limit), NOTEBOOK_PAGE_MAX))
    except (TypeError, ValueError):
        safe_limit = NOTEBOOK_PAGE_MAX
    if safe_limit == 0:
        return {"chunks": [], "total": total, "next_offset": None}
    page = items[safe_offset:safe_offset + safe_limit]
    next_offset = safe_offset + len(page)
    return {
        "chunks": page,
        "total": total,
        "next_offset": next_offset if next_offset < total else None,
    }


# ── D3 原句核验（plans/0923-05 §4.2）────────────────────────────────
# 用户要看的是「当时的原句」，不是「来自第几句」。普通候选的「原句」必须是
# **真含该词块**的句子；历史候选的 context_snippet 可能只是首条用户话（不含
# 词块）——那就按 source_id(conv_id) + source_turn 回查 conversation_messages，
# **核不到就明确标"原句暂缺"，绝不生成假例句**（诚实缺口 > 编造）。

def _stumble_english_context(norm: str, msgs) -> str:
    """在整盘消息里找**真含该词块**的英文句子，优先 AI（assistant）那句精修英文。

    0924-09③（用户口径）：卡壳条目的注释要用全英文精修文本，不再铺用户
    中英混杂的原话。AI 回复里含词块的那句话本来就是精修过的英文——取句子
    （不是整段回复），找不到返回 ''，上层退回用户原话，**绝不编造**。
    """
    fallback = ""
    for m in (msgs or []):
        if not isinstance(m, dict):
            continue
        content = " ".join(str(m.get("content") or "").split())
        if not content or not norm:
            continue
        if not _token_subseq_in(norm, normalize_chunk(content)):
            continue
        sent = next(
            (s for s in re.split(r"(?<=[.!?…。？！])\s+", content)
             if _token_subseq_in(norm, normalize_chunk(s))),
            "",
        )
        if not sent:
            continue
        if str(m.get("role") or "") == "assistant":
            return sent[:160]
        if not fallback:
            fallback = sent[:160]
    return fallback


def resolve_original_sentences(chunks: list, messages_by_conv=None) -> list:
    """就地给词块本的条目附 `sentence_kind` / `orig_sentence` / `sentence_missing`。

    `messages_by_conv`：调用方**按页按盘批量取好**的 `{conv_id: [{role,content}, ...]}`
    （须按 `seq` 升序）。传空表只做"上下文自带核验"这一遍——第二遍再填回查结果。
    卡壳条目（`source_type='stumble'`）：snippet 是用户当时想说的话（中英混杂）
    ⇒ 先按 `wanted` 落位；第二遍带消息后若整盘里找得到**真含该词块的英文句**
    （优先 AI 那句精修英文），升级为 `original`——用户口径 0924-09③：注释
    要全英文精修文本。找不到才保留用户原话，绝不编造。
    """
    messages_by_conv = messages_by_conv or {}
    for c in (chunks or []):
        if not isinstance(c, dict):
            continue
        snippet = " ".join(str(c.get("context_snippet") or "").split())
        if str(c.get("source_type") or "") == "stumble":
            c["sentence_kind"] = "wanted"
            c["orig_sentence"] = snippet[:160]
            c["sentence_missing"] = not bool(snippet)
            conv = str(c.get("source_id") or "").strip()
            msgs = messages_by_conv.get(conv) if conv else None
            if isinstance(msgs, list) and msgs:
                sent = _stumble_english_context(str(c.get("norm") or ""), msgs)
                if sent:
                    c["sentence_kind"] = "original"
                    c["orig_sentence"] = sent
                    c["sentence_missing"] = False
            continue
        c["sentence_kind"] = "original"
        norm = str(c.get("norm") or "")
        if norm and snippet and _token_subseq_in(norm, normalize_chunk(snippet)):
            c["orig_sentence"] = snippet[:160]
            c["sentence_missing"] = False
            continue
        c["orig_sentence"] = ""
        c["sentence_missing"] = True
        conv = str(c.get("source_id") or "").strip()
        try:
            turn = int(c.get("source_turn") or 0)
        except (TypeError, ValueError):
            turn = 0
        msgs = messages_by_conv.get(conv) if conv else None
        if not isinstance(msgs, list) or turn <= 0 or turn - 1 >= len(msgs):
            continue
        content = " ".join(str((msgs[turn - 1] or {}).get("content") or "").split())
        if content and norm and _token_subseq_in(norm, normalize_chunk(content)):
            c["orig_sentence"] = content[:160]
            c["sentence_missing"] = False
    return chunks


def sentence_lookup_keys(chunks: list) -> list:
    """需要回查消息的 `(conv_id, turn)` 去重列表（turn > 0 才算）。

    配合 `resolve_original_sentences` 二次调用：第一遍不带消息、第二遍带
    按盘批量取好的消息。**不修改入参**。
    0924-09③：卡壳条目**也回查**——整盘消息里可能藏着 AI 那句含词块的
    精修英文（注释升级用）；其余类型仍只在原句标缺时回查。
    """
    keys, seen = [], set()
    for c in (chunks or []):
        if not isinstance(c, dict):
            continue
        is_stumble = str(c.get("source_type") or "") == "stumble"
        if not is_stumble and not c.get("sentence_missing"):
            continue
        conv = str(c.get("source_id") or "").strip()
        try:
            turn = int(c.get("source_turn") or 0)
        except (TypeError, ValueError):
            turn = 0
        if conv and turn > 0 and (conv, turn) not in seen:
            seen.add((conv, turn))
            keys.append((conv, turn))
    return keys


def loads_json_list(raw) -> list:
    """use_events / status_history 之类 JSON 串的容错读取（PG jsonb 与
    SQLite TEXT 两种形态都可能是 list 或 str）。"""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, TypeError):
            pass
    return []


# ══ 二/三期（plans/0916-11 §4.4–§4.6）══════════════════════════════════

def token_subsequence(normalized_chunk: str, normalized_text: str) -> bool:
    """词块是否作为连续 token 子序列出现在 normalized_text 里（公开别名）。"""
    return _token_subseq_in(normalized_chunk, normalized_text)


def evaluate_chunk_use(user_text: str, chunks: list,
                       last_assistant_text: str = "",
                       offered_norms: Iterable[str] = ()) -> list[dict]:
    """F13 会用确认的匹配层（§4.5，纯函数，逐轮在**用户原始转写**上跑）。

    判据（误报比漏报更伤，从紧、宁可漏）：
      1. 词块以连续 token 子序列出现在用户转写里（近似但非同一词块不误报）；
      2. 「自己发起」：该词块**不得出现在紧邻上一条 AI 回复里**——否则是
         学舌/AI 先说了，整条跳过不记；
      3. via_opportunity：上一轮 AI 注入过 [Chunk Practice] 的词块，
         用户这一轮说出来 = 接住机会（练过），不算自发起；
      4. 同一轮重复多次天然只记 1 次（每词块至多一条结果）。
    只匹配 status ∈ {saved, activated, mastered} 的行；**例外（plans/0916-12
    §6-B）**：卡壳条目（source_type='stumble'）的 candidate 也参与——那是他
    自己的缺口，"用出来"就是比收藏更强的信号，不用先收进本子；AI 归纳产生的
    candidate（tape）用户没选过，仍不参与（避免 AI 确认自己塞的词）。
    """
    norm_user = normalize_chunk(user_text)
    if not norm_user:
        return []
    norm_ai = normalize_chunk(last_assistant_text or "")
    offered = {normalize_chunk(str(n or "")) for n in (offered_norms or [])}
    offered.discard("")
    out: list[dict] = []
    seen: set[str] = set()
    for row in chunks or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status == "candidate":
            if str(row.get("source_type") or "") != "stumble":
                continue
        elif status not in {"saved", "activated", "mastered"}:
            continue
        norm = normalize_chunk(str(row.get("norm") or row.get("text") or ""))
        if not norm or norm in seen:
            continue
        if not token_subsequence(norm, norm_user):
            continue
        seen.add(norm)
        # 学舌排除：AI 上一轮刚说过 → 不记（宁漏勿误报）。
        if norm_ai and token_subsequence(norm, norm_ai):
            continue
        out.append({
            "chunk_id": str(row.get("id") or ""),
            "text": str(row.get("text") or ""),
            "norm": norm,
            "kind": str(row.get("kind") or ""),
            "status": str(row.get("status") or ""),
            "via_opportunity": norm in offered,
        })
    return out


# 本盘机会注入上限（§4.4：每盘最多 2 次机会）
OPPORTUNITY_MAX_PER_CONV = 2
# 同一词块 7 天内不重复塞（§4.4）——用 last_use_at 兜底（机会未被接住时
# 无持久痕迹，此约束以"用过"为准；本盘内用 chunk_offered_norms 精确去重）。
OPPORTUNITY_REPEAT_DAYS = 7
# 卡壳条目待回收池上限（plans/0916-12 §4.1/§6-B：待回收 ≤12 条）。rows 按新
# 在前传入，只取最近 12 条 stumble candidate 有资格；超出的最老条目自然
# 沉底淘汰，不需要物理删除。
STUMBLE_POOL_MAX = 12

CHUNK_PRACTICE_RULE = (
    "上面是学习者收进本子的词块（他主动收藏、想学会用的表达）。"
    "顺着当前话题**自然创造一次让他用得上它的时机**：问一个自然的问题、"
    "或让局面自然走到那一步——他需要那句话才能接下去。"
    "判据：话题不合适就这一条完全不用，绝不硬塞、绝不为了用词块改变话题走向；"
    "绝不解释词块本身、绝不说'练习/词块/试试这个表达'等元话术；"
    "一个词块一个机会，本轮最多制造一次。"
)


def select_opportunity_chunks(rows: list, offered_session_norms: Iterable[str] = (),
                              last_turn_offered: Iterable[str] = (),
                              now=None) -> list[dict]:
    """F12 机会选择层（§4.4，纯函数）：从本子挑本轮可注入的词块，≤2 条。

    节奏约束（全部确定性）：
      - 只挑 status ∈ {saved, activated}（mastered 已会用，不用再造机会）；
        **例外（plans/0916-12 §6-B）**：卡壳条目（source_type='stumble'）的
        candidate **按权利注入**——那是他自己要的东西，不需要先收藏；
        AI 归纳产生的 candidate 仍不注入（没被收藏就不主动塞）；
      - 卡壳候选只取最近 STUMBLE_POOL_MAX 条（rows 新在前，最老的自然淘汰）；
      - 同一盘里已注入过的词块不再注入（offered_session_norms）；
      - **不连续两轮塞**（上一轮注入过任何词块 → 本轮直接不注入）；
      - 同一词块 7 天内被真实用过 → 不再塞（last_use_at 距今 < 7 天）。
    话题是否合适不由代码硬判（交给生成侧 rule 把关），这里只做节奏闸门。
    """
    offered_session = {normalize_chunk(str(n or "")) for n in (offered_session_norms or ())}
    offered_session.discard("")
    if list(last_turn_offered or []):
        return []  # 不连续两轮塞
    from datetime import datetime, timezone, timedelta
    now = now or datetime.now(timezone.utc)
    picked: list[dict] = []
    stumble_quota = 0
    for row in rows or []:
        if len(picked) >= OPPORTUNITY_MAX_PER_CONV:
            break
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status not in {"saved", "activated"}:
            if status == "candidate" \
                    and str(row.get("source_type") or "") == "stumble" \
                    and stumble_quota < STUMBLE_POOL_MAX:
                stumble_quota += 1   # 前 12 条卡壳候选有资格，再老的直接淘汰
            else:
                continue
        norm = normalize_chunk(str(row.get("norm") or row.get("text") or ""))
        if not norm or norm in offered_session:
            continue
        last_use = str(row.get("last_use_at") or "").strip()
        if last_use:
            try:
                lu = datetime.fromisoformat(last_use.replace("Z", "+00:00"))
                if lu.tzinfo is None:
                    lu = lu.replace(tzinfo=timezone.utc)
                if now - lu < timedelta(days=OPPORTUNITY_REPEAT_DAYS):
                    continue
            except (ValueError, TypeError):
                pass
        picked.append({
            "text": str(row.get("text") or "")[:80],
            "norm": norm,
            "kind": str(row.get("kind") or ""),
            "gloss": str(row.get("gloss") or "")[:60],
            "example": str(row.get("example") or "")[:160],
            "source_type": str(row.get("source_type") or ""),
        })
    return picked


# ── F14 表达反馈（§4.6：数字必须由代码算，同 enforce_exact_clock 原则）──

# 衔接词词表（§4.6 原话的词表计数；匹配按 token 边界，词组按 token 序列）
_CONNECTIVE_TOKENS = {
    "because", "so", "but", "then", "also", "however", "although", "though",
    "when", "while", "before", "after", "since", "finally", "actually",
    "besides", "therefore", "anyway", "instead", "later", "first", "next",
    "second", "third", "lastly", "eventually", "meanwhile", "plus", "and",
}
_CONNECTIVE_SEQS = [
    ("after", "that"), ("as", "soon", "as"), ("and", "then"),
    ("at", "first"), ("in", "the", "end"), ("by", "the", "way"),
]
# 时间词（细节数的一部分）
_TIME_TOKENS = {
    "today", "yesterday", "tomorrow", "morning", "afternoon", "evening",
    "night", "noon", "midnight", "week", "weeks", "month", "months", "year",
    "years", "hour", "hours", "minute", "minutes", "second", "seconds",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "o'clock",
    "am", "pm", "last", "next", "ago", "now", "later", "early", "late",
    "weekend", "birthday", "holiday",
}
# 英文数词（数字细节的兜底；digits 另计）
_NUMBER_TOKENS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    "hundred", "thousand", "million", "half", "couple", "few", "several",
    "dozen",
}


# 内容词口径用的功能词表（plans/0919-02 §4.1 B）。两条纪律：
#   1. **不改 _FUNCTION_WORDS 本体**——它同时被 looks_like_asr_garbage 消费，
#      动它会连带改 ASR 乱码闸门的判定；这里只做并集。
#   2. **只补最扎眼的缺口**（冠词、第一/三人称代词、be 形），**不收副词与实义词**：
#      停用词表一旦往实义词方向长大，就变成了变相关键词表（HANDOFF §5 硬规则 2）。
_CONTENT_FUNCTION_WORDS = _FUNCTION_WORDS | {
    "a", "an", "i", "me", "my", "mine", "us", "him", "them",
    "be", "been", "being",
}

# 句末标点（plans/0919-02 §4.1 A）：豆包 ASR 对中文吐全角 `。！？`，旧正则
# `[.!?]` 只认 ASCII → 纯中文讲述 sentences 恒为 1（F14 首遍门槛是 >=3，
# 于是"再讲一遍"闭环在真机上基本空转）。全角半角一并认。
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？…]+")
_SENTENCE_END_RE = re.compile(r"[.!?。！？…]")


def _content_word_tokens(tokens: list[str]) -> set[str]:
    """内容词去重集：用功能词表反向筛出的近似"实词"（零模型、确定性）。

    **近似口径，如实标注**：没有 POS、也不看大小写——ASR 原文普遍全小写
    （`im i didnt say that thing`），任何依赖大小写的判据都会空转。因此这里
    只做"现成功能词反向筛"这一件可靠的事，不硬造专名检测（那只会退化成
    变相关键词表），也**禁止**把全小写文本强行恢复大小写再判专名——那是
    伪造成分，与 engine/speech_assessment.py"拿不到就标 unavailable"的纪律冲突。

    边界：中文没有分词、词表也是英文的，所以只判纯 ASCII 字母 token——
    中文整段在转写里是一个不带空格的 run，按 token 数计会得出荒谬值，
    此处不参与（中文度量要等第 2 期有分词口径再说）。纯数字 token 归 numbers 源。
    """
    out: set[str] = set()
    for tok in tokens:
        if not tok or not tok.isascii() or not any(c.isalpha() for c in tok):
            continue
        if tok.isdigit():
            continue
        if tok in _CONTENT_FUNCTION_WORDS or tok in _TIME_TOKENS \
                or tok in _NUMBER_TOKENS or tok in _CONNECTIVE_TOKENS:
            continue
        out.add(tok)
    return out


def _content_words_available(tokens: list[str]) -> bool:
    """内容词口径这一半**这段文本能不能判**。

    判不了的情形：一个 ASCII 字母 token 都没有（纯中文讲述）。中文没有分词、
    功能词表也是英文的 → 这一半**拿不到**。按 engine/speech_assessment.py 的既有
    纪律（拿不到的维度标 unavailable，不伪造），此时如实标不可用，
    **绝不填 0 冒充"这段没有内容词"**——那会让模型以为学习者什么都没讲。
    """
    return any(t.isascii() and any(c.isalpha() for c in t) for t in tokens)


# ══════════════ §4.2 比率制：分母换 sentences（2026-09-19 定案）══════════════
# 信息密度 = `details ÷ sentences`；简洁进步 = `sentences` 下降 且 `details` 不下降。
#
# 为什么分母是 sentences 而不是 words（§4.2「第 3 项定案」）：
#   ① sentences 在第 0 期改成全角半角通吃后，中英两种转写下都可靠；
#   ② "信息密度"的直觉含义本来就是**每句话装了多少细节**——用句数当分母天然
#      免疫中文无分词的问题，也不必造"英文词数 + 中文字符数"这种混算单位的假口径。
# `words` 仍留在 metrics 里作**长度观测**，但不进任何比率、不进对比提示。
#
# 密度至少要有 2 句才算数：无标点讲述会退化成 `sentences=1`，此时 `details÷1`
# 就是 details 本身、并不是"每句装多少"，报它等于编数（§4.2 定案："无标点讲述
# 不报密度"）。这个下限同时是"简洁进步"能否成立的前提。
_DENSITY_MIN_SENTENCES = 2
#
# **分子也必须真的被量到**（A 批新增闸门、本批收紧，理由见 §9.2 偏离 1 与 §9.5）：
# 纯中文讲述下 `details` 的内容词这一半是 `content_words_available=false`（§4.1 B），
# `details÷sentences` 会算出 **0.00** —— 那不是"每句 0 个细节"，是**这一半没被量到**。
# A 批的处置是整段级闸门（有没有 ASCII 字母 token）+ 中文如实降级；本批按 §4.2
# 「中文密度的分子口径」把中文的分子补上，并把闸门**从整段收紧到逐句**
# （`_measured_sentence_count`）——A 批自曝的边界是真机(2)①：整段只有 `one.two`
# 两个英文数字词也放行，`details=2` 全是数字、算出 `density=0.25`。
#
# 分子口径分两支（同一遍只会用其中一支，`detail_caliber` 标明用了哪支）：
#   · `en`：数字 + 时间词 + 内容词（§4.1 B，英文尺子）；
#   · `zh`：汉字数字 + 中文时间词 + 人物/地点指称（§4.2 第二版定案，中文尺子）。
# **两支的"抓得到的东西"本来就不同 → 中文密度天然低于英文密度**，所以密度只能
# 在同一人、**同一语言**的两遍之间比（跨口径在 `compare_expression_metrics` 里
# 直接标 `caliber_mismatch` 不报）。
_DENSITY_MIN_SENTENCES = 2


def _unavailable_density(reason: str, caliber: str = "en") -> dict:
    """不可用密度的固定形状（`text`/`fraction` 一律空串 —— 下游据此**不报这个数**）。"""
    return {"available": False, "value": None, "text": "", "fraction": "",
            "reason": reason, "caliber": caliber}


def _density_of(details: int, sentences: int, details_measured: bool = True,
                caliber: str = "en") -> dict:
    """信息密度 = details ÷ sentences（比率制，§4.2 定案）。

    返回形状固定；`available=False` 时 `value`/`text`/`fraction` 全空——下游据此
    **不报这个数**，而不是报 0、也不是拿 details 冒充密度。

    两道闸门：`sentences >= 2`（无标点讲述退化不算密度）、
    `details_measured`（分子**逐句**真的被量到，见上方注释与
    `_measured_sentence_count`）。`caliber` 标明这一遍用的是哪把尺子。
    """
    sentences = int(sentences or 0)
    details = int(details or 0)
    if not sentences:
        return _unavailable_density("empty", caliber)
    if sentences < _DENSITY_MIN_SENTENCES:
        return _unavailable_density("no_sentence_punctuation", caliber)
    if not details_measured:
        return _unavailable_density("numerator_too_thin", caliber)
    value = details / sentences
    return {
        "available": True,
        "value": value,
        # 小数由**代码**算到两位：模型只许原样引用，不得自己重算、不得四舍五入。
        "text": "%.2f" % value,
        # 精确分数（零舍入），小数形式的无损备份。
        "fraction": "%d/%d" % (details, sentences),
        "reason": "",
        "caliber": caliber,
    }


def _measured_sentence_count(sentence_parts: list[str], caliber: str) -> int:
    """逐句分子计数：**有几句话里真的数到了分子**（§4.2 第二版定案收紧的闸门）。

    A 批的闸门是整段级的 `content_words_available`（"这段有没有 ASCII 字母 token"），
    它自己就自曝了边界：真机(2)① 整段只有 `one.two` 两个英文数字词也放行，
    `details=2` 全是数字，算出 `density=0.25` —— 那不是"每句 0.25 个细节"，
    是**分子没量到**。改成逐句判，两支口径各自"逐句内容词计数"、形状一致：
      · 英文口径 = 逐句数**内容词**（`_content_word_tokens`；数字/时间词/衔接词都不算，
        这正是"内容词计数"四个字的字面含义）；
      · 中文口径 = 逐句数**中文分子项**（汉字数字/时间词/人物/地点指称）。
    """
    def _en_has_items(sentence: str) -> bool:
        return bool(_content_word_tokens(normalize_chunk(sentence).split()))

    def _zh_has_items(sentence: str) -> bool:
        return any(_zh_detail_hits(sentence).values())

    has_items = _zh_has_items if caliber == "zh" else _en_has_items
    return sum(1 for s in sentence_parts if s.strip() and has_items(s))


def _pick_detail_caliber(sentence_parts: list[str],
                         tokens: list[str]) -> tuple[str, bool]:
    """选分子口径，并报"逐句分子够不够"（§4.2 第二版定案）。

    优先能**真正量到**的那一支：≥2 句数到了内容词 → 英文口径；否则 ≥2 句数到了
    中文分子项 → 中文口径。两支都不够时回落到"这类文本本该用哪把尺子"（有 ASCII
    内容词层就英文，否则中文），此时 `measured=False` → 密度不报。

    这样中英混说不会被"混着的几个英文词"带偏到英文尺子上：真机(2)① 整段中文里
    只有 `one.two` 两个英文数字词 → 英文那支 0 句过硬、中文那支也不够 → 不报密度
    （这正是 A 批自曝边界的修复）。
    """
    if _measured_sentence_count(sentence_parts, "en") >= _DENSITY_MIN_SENTENCES:
        return "en", True
    if _measured_sentence_count(sentence_parts, "zh") >= _DENSITY_MIN_SENTENCES:
        return "zh", True
    return ("en" if _content_words_available(tokens) else "zh"), False


def _caliber_of(metrics: dict) -> str:
    """这份 metrics 的分子口径。

    老载荷（A 批写下的 pending metrics）没有 `detail_caliber` 键 → 按 A 批口径回落：
    内容词层可用即英文、否则中文。
    """
    caliber = (metrics or {}).get("detail_caliber")
    if caliber in ("en", "zh"):
        return str(caliber)
    return "en" if _detail_parts_of(metrics)["content_words_available"] else "zh"


def _numerator_measured_of(metrics: dict) -> bool:
    """这份 metrics 的分子有没有**逐句**被量到。

    老载荷回落成 A 批的整段闸门（`content_words_available`）——不在读取侧凭空把
    历史载荷从"可用"改成"不可用"。
    """
    flag = (metrics or {}).get("numerator_measured")
    if isinstance(flag, bool):
        return flag
    return _detail_parts_of(metrics)["content_words_available"]


def density_of_metrics(metrics: dict) -> dict:
    """从一份 metrics 取信息密度（`_density_of` 的取数壳）。"""
    metrics = metrics or {}
    return _density_of(
        metrics.get("details"), metrics.get("sentences"),
        _numerator_measured_of(metrics), _caliber_of(metrics),
    )


def looks_like_a_passage(metrics: dict) -> bool:
    """够不够"一段话"（§4.2 定案：**对比分支**的中性门槛）。

    旧门槛是 `metrics["words"] >= 4`（`agent/tutor_agent.py` 对比分支），而中文
    转写没有空格分词 → 三段中文讲述实测 `words=3`，第二遍**永远落回首遍分支**、
    拿不到【表达对比·代码权威】（§7.5 开口项 1）。换成不依赖分词的判据：
    `sentences >= 2`（有句末标点的一段话）或 `details >= 1`（至少交代了一个细节）。

    这只是**能不能进对比分支**的门槛，不是"讲得好不好"的评价。
    """
    metrics = metrics or {}
    return (int(metrics.get("sentences") or 0) >= 2
            or int(metrics.get("details") or 0) >= 1)


# ── 候选判据①：重点提炼（首句/前两句是否含全篇最高频实词）─────────────────────
# **k = 1（本批实测从 2 收紧到 1）**：离线出数表里"重点在后"的反例被判成"是"——
# 因为 top-2 里排第二的 `took` 恰好落在首句。只取**最高的那一个**实词才与
# "全篇最高频实词"字面一致，也才不把排在第二的词混进来当"重点"。
# 这不是为了凑维度放松，是标定后**收紧**（证据见 plans/0919-02 §9 数表）。
_KEYPOINT_TOP_K = 1
# 至少要两句才谈得上"重点在不在前"：一句话没有"前"与"后"。
_KEYPOINT_MIN_SENTENCES = 2


def _sentence_parts_of(text: str) -> tuple[list[str], int]:
    """切句（与 `count_expression_metrics` **同一把尺子**，口径不许分叉）。"""
    raw = " ".join(str(text or "").split())
    if not raw:
        return [], 0
    parts = [s for s in _SENTENCE_SPLIT_RE.split(raw) if s.strip()]
    return parts, (len(parts) if _SENTENCE_END_RE.search(raw) else 1)


def assess_key_point_front(text: str) -> dict:
    """候选判据①（§4.2 表）：重点提炼——首句/前两句是否含**全篇最高频实词**。

    口径（复用第 0 期修好的内容词口径取 top-k，零模型、确定性、可复算）：

      1. 切句；**至少两句**才判（否则谈不上"重点在不在前"）。
      2. 逐句取内容词（`_content_word_tokens`：功能词反向筛、只判 ASCII 字母
         token、全程不看大小写）。**至少两句有内容词**才判——纯中文讲述拿不到
         内容词（无分词、词表是英文的），此时标不可用、不硬判。
      3. 「全篇最高频实词」= 全篇出现 >= 2 次的实词，按（出现句数 ↓, 出现次数 ↓,
         字母序 ↑）排序取前 `_KEYPOINT_TOP_K` 个。**全篇每个实词都只出现一次时
         不报**：那样谁排第一只能靠字母序，是抛硬币而不是"最高频"。

    `hit_first_sentence` / `hit_first_two` 分开给：§4.2 写的是"首句/前两句"，两种
    宽严都留出来，交给实测标定选哪个（本批先按 `hit_first_two` 报）。
    """
    parts, sentences = _sentence_parts_of(text)
    out = {
        "available": False, "reason": "empty",
        "sentences": sentences, "judgeable_sentences": 0,
        "top_content_words": [], "hit_first_sentence": False, "hit_first_two": False,
    }
    if not parts:
        return out
    if sentences < _KEYPOINT_MIN_SENTENCES:
        out["reason"] = "not_a_passage"
        return out
    per_sentence = [
        _content_word_tokens(normalize_chunk(p).split()) for p in parts
    ]
    judgeable = sum(1 for s in per_sentence if s)
    out["judgeable_sentences"] = judgeable
    if judgeable < _KEYPOINT_MIN_SENTENCES:
        out["reason"] = "content_words_unavailable"
        return out
    doc_freq: dict[str, int] = {}
    for words in per_sentence:
        for word in words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    all_content = _content_word_tokens(
        [t for p in parts for t in normalize_chunk(p).split()])
    token_freq: dict[str, int] = {}
    for part in parts:
        for tok in normalize_chunk(part).split():
            if tok in all_content:
                token_freq[tok] = token_freq.get(tok, 0) + 1
    repeated = [w for w in all_content if token_freq.get(w, 0) >= 2]
    if not repeated:
        # 实词全是 hapax → 没有"最高频"可言。**不报，不猜。**
        out["reason"] = "no_repeated_content_word"
        return out
    repeated.sort(key=lambda w: (-doc_freq.get(w, 0), -token_freq.get(w, 0), w))
    top = repeated[:_KEYPOINT_TOP_K]
    first_two = per_sentence[0] | (
        per_sentence[1] if len(per_sentence) > 1 else set())
    out.update({
        "available": True, "reason": "",
        "top_content_words": top,
        "hit_first_sentence": bool(per_sentence[0] & set(top)),
        "hit_first_two": bool(first_two & set(top)),
    })
    return out


# ── 候选判据②：必要信息补充（背景要素命中数）─────────────────────────────────
# 词表**只服务这一个判据**，与 details 用的 `_TIME_TOKENS` 互不影响（不动既有口径）。
# 中文侧只能做**子串匹配**（没有分词），所以中文表一律收**多字词**、不收单字——
# 单字会误命中（"家"命中"大家"、"他"命中"其他"）。英文侧按 token 精确匹配。
#
# **刻意排除第一人称单数**（`我` / i, me, my, mine, myself）：讲述者自己必然出现，
# 把它算成"交代了人物"就等于这一格恒命中——那是凑维度，不是判据。第一人称复数
# （we / 我们）保留：它隐含"还有别人"。
#
# **第二人称也一并排除**（you / your / yours / 你们，本批实测后加的）：离线出数表里
# 真机(3) 靠转写中的 `do you know` 这一句命中了 person —— 而这一轮学习者是在**对 AI
# 说话**，"you" 指的不是故事里的任何人，是结构性误报。本产品的用户每轮都在对 AI
# 讲话，留着第二人称就等于这一格半恒命中。排除它，背景要素才回到"故事里交代了谁"。
_EN_PERSON_REFS = {
    "we", "us", "our", "ours",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "friend", "friends", "family", "brother", "sister", "mother", "mom", "mum",
    "father", "dad", "parents", "son", "daughter", "child", "children", "kid",
    "kids", "boss", "colleague", "coworker", "classmate", "teacher", "student",
    "neighbour", "neighbor", "waiter", "waitress", "clerk", "customer",
    "stranger", "guy", "girl", "boy", "man", "woman", "person", "people",
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody",
}
_EN_PLACE_REFS = {
    "home", "house", "apartment", "flat", "office", "school", "university",
    "college", "classroom", "library", "park", "hospital", "clinic", "airport",
    "station", "supermarket", "market", "store", "shop", "mall", "restaurant",
    "cafe", "bar", "hotel", "bank", "gym", "street", "road", "city", "town",
    "village", "country", "abroad", "downtown", "upstairs", "downstairs",
    "yard", "garden", "theatre", "theater", "cinema", "museum", "zoo", "beach",
    "mountain", "river", "lake", "room", "kitchen", "bathroom", "bedroom",
    "floor", "counter", "lobby", "entrance",
}
_ZH_TIME_REFS = (
    "今天", "昨天", "明天", "前天", "后天", "当天", "那天", "有一天",
    "早上", "上午", "中午", "下午", "晚上", "夜里", "半夜", "凌晨",
    "以前", "之前", "后来", "最近", "当时", "小时候",
    "去年", "今年", "明年", "上周", "上个月", "周末", "星期",
    "生日", "假期", "放假",
)
_ZH_PERSON_REFS = (
    "朋友", "同学", "同事", "老师", "学生", "老板", "店员", "服务员",
    "邻居", "家人", "妈妈", "爸爸", "父母", "儿子", "女儿", "孩子",
    "哥哥", "姐姐", "弟弟", "妹妹", "爷爷", "奶奶", "姥姥", "姥爷",
    "叔叔", "阿姨", "女朋友", "男朋友", "老婆", "老公", "客人",
    "大家", "别人", "对方", "有人", "他们", "她们", "我们",
)
_ZH_PLACE_REFS = (
    "家里", "公司", "学校", "医院", "公园", "机场", "车站", "地铁",
    "超市", "商店", "餐厅", "饭馆", "饭店", "咖啡店", "咖啡厅", "咖啡馆",
    "图书馆", "办公室", "教室", "路上", "街上", "楼下", "楼上",
    "门口", "城市", "附近", "外地", "国外", "国内", "广场", "酒店",
    "宾馆", "银行", "商场", "宿舍", "食堂", "操场", "电影院",
    "博物馆", "动物园", "海边", "房间", "厨房", "客厅", "卧室",
    "洗手间", "厕所", "柜台", "前台",
)


# ── 中文分子的第一类：汉字数字（§4.2「中文密度的分子口径」第二版定案）──────────
# 中文没有分词、只能子串匹配，而**单个数字字直配会立刻误报**（"一"命中 一起/一直/
# 一定/一样/一般/一些/一边/一点…，全是高频常用词）→ 所以不建"数字字集合"，
# 只认两种**数字形态**，两种都自带"这是数量"的证据：
#   ① **数字 + 量词**（三杯 / 两遍 / 五号 / 十岁）——量词把语义钉住；
#   ② **连写数字串**（≥2 个数字字：二十 / 一百 / 三十五）——单个数字字不算。
# 量词表**刻意不收 `点` 与 `分`**：`一点`/`有点`/`十分`/`几分` 都是程度副词，
# 收了就是稳误报；代价是 `三点`（三点钟）会漏 —— 按 §4.2 硬约束 2「宁可漏，
# 不错报」接受，并在此写明（不写清，后来人会当成漏词去"补"，然后引入误报）。
# 量词本身也只收**语义明确**的那些，不收 `番/味/头/时/式/般/下/点/分/度/边/面/手`
# 这类会同形于常用词的（一番/一味/一头/一时/一式/一般/一下/一点/十分/一度/一边/一面/一手）。
_ZH_NUMBER_CLASSIFIERS = (
    "个只杯碗次遍份张条台号块元岁斤米里步人天月年日"
    "家件位口层场道棵朵封页门本把双对"
    # 容器（一包咖啡豆 / 一瓶水 / 一盒饼干）
    "包袋瓶罐盒盘碟桶壶盆"
    # 其余安全量词（一幅画 / 一趟 / 一群 / 一座 / 一副 / 一排 / 一辆 / 一篇）
    "幅片支顿趟阵群批座间所套副顶丝刻串束排半架辆艘匹"
    # 文书/时间/比例（一部 / 一句 / 一轮 / 一秒 / 一倍 / 一回）
    "部篇段章句轮名秒倍回"
)
_ZH_NUMBER_CHARS = "零一二三四五六七八九十百千万两"
# **序数不算数量**：`第二天` / `第一杯` / `初一` 里的数字是序数，不是"几个" →
# 用 `(?<![第初])` 把紧跟"第/初"的数字形态排掉（不加这条，`第二天` 会被切成
# `二天` 命中，是稳误报）。
_ZH_NUMBER_RE = re.compile(
    r"(?<![第初])[" + _ZH_NUMBER_CHARS + r"]+[" + _ZH_NUMBER_CLASSIFIERS + r"]"
    r"|(?<![第初])[" + _ZH_NUMBER_CHARS + r"]{2,}"
)
# 与数字**同形的常见非数量词**（连写规则会命中，必须按**整个命中串**排掉）：
#   千万（务必）/ 万一（如果）/ 一一（逐一列举）/ 一二（约数的口头语）
# 按整串排除，所以"一千万"（命中串是"一千万"）不受影响，只有光杆"千万"被排掉。
_ZH_NUMBER_STOPWORDS = {"千万", "万一", "一一", "一二"}


def _zh_detail_hits(text: str) -> dict:
    """中文口径分子的四类命中（各自是**去重后的 surface 集合**，零模型、可复算）。

    后三类**直接复用候选判据②那三张已标定的中文表**（`_ZH_TIME_REFS` /
    `_ZH_PERSON_REFS` / `_ZH_PLACE_REFS`，真机 6 段验证无误报），不新造词表；
    也就继承了那两处收紧：**排除第一人称单数**（"我"必然出现，算上就恒命中）、
    **排除第二人称**（"你们"；本产品里学习者每轮都在对 AI 说话，那是结构性误报）。

    去重口径与英文侧同形：`details` = 去重后的证据**条数**，不是出现次数——
    两遍之间才可比。
    """
    raw = str(text or "")
    numbers = {m.group(0) for m in _ZH_NUMBER_RE.finditer(raw)}
    numbers -= _ZH_NUMBER_STOPWORDS
    return {
        "numbers": numbers,
        "time_words": {w for w in _ZH_TIME_REFS if w in raw},
        "person_refs": {w for w in _ZH_PERSON_REFS if w in raw},
        "place_refs": {w for w in _ZH_PLACE_REFS if w in raw},
    }


def _zh_detail_parts(text: str) -> dict:
    """中文分子四类各自的原样条数（与英文 `detail_parts` 同形，供下游分来源取用）。"""
    return {k: len(v) for k, v in _zh_detail_hits(text).items()}


def _zh_details(text: str) -> int:
    """中文口径的 details：四类命中的**并集**去重数（与英文并集口径同形）。"""
    merged: set[str] = set()
    for hits in _zh_detail_hits(text).values():
        merged |= hits
    return len(merged)


def assess_background_elements(text: str) -> dict:
    """候选判据②（§4.2 表）：必要信息补充——背景要素命中数。

    背景要素三类各算一格：**时间锚**（什么时候）、**人物指称**（涉及谁）、
    **地点指称**（在哪儿）。命中数 0–3，纯词表、零模型、确定性、可复算。

    英文按 token 精确匹配（用 `normalize_chunk` 归一化后），中文按**子串**匹配
    （没有分词）；两侧粗细不同，如实标注。`hits` 给出命中的是哪几类，供提示侧
    原样引用（"2/3，缺地点"），不让模型自己判断缺了什么。

    人称词表的取舍见 `_EN_PERSON_REFS` 上方注释：第一人称单数与第二人称都排除
    （前者必然出现、后者在本产品里指的是 AI 而不是故事里的人）。
    """
    raw = " ".join(str(text or "").split())
    out = {
        "available": bool(raw), "reason": "" if raw else "empty",
        "time": False, "person": False, "place": False, "count": 0, "hits": [],
    }
    if not raw:
        return out
    tokens = set(normalize_chunk(raw).split())
    time_hit = bool(tokens & _TIME_TOKENS) or any(w in raw for w in _ZH_TIME_REFS)
    person_hit = bool(tokens & _EN_PERSON_REFS) or any(
        w in raw for w in _ZH_PERSON_REFS)
    place_hit = bool(tokens & _EN_PLACE_REFS) or any(
        w in raw for w in _ZH_PLACE_REFS)
    hits = [name for name, flag in
            (("time", time_hit), ("person", person_hit), ("place", place_hit))
            if flag]
    out.update({
        "time": time_hit, "person": person_hit, "place": place_hit,
        "count": len(hits), "hits": hits,
    })
    return out


def _expression_signals(text: str) -> dict:
    """两个候选判据的合并入口（一遍讲述各算一次，形状永远稳定）。"""
    return {
        "key_point_front": assess_key_point_front(text),
        "background_elements": assess_background_elements(text),
    }


def count_expression_metrics(text: str) -> dict:
    """一遍讲述的确定性度量（F14）：句数/词数/细节数/衔接词数。

    细节数 `details` = **去重后的证据条数**，按**这一遍实际能用的尺子**（`detail_caliber`）
    取一支（§4.2 定案 + 第二版「中文密度的分子口径」）：
      · `en`（英文尺子）＝ 数字 + 时间词 + **内容词** 三个来源的并集：
        `numbers` / `time_words` 词表判定、不依赖大小写；`content_words` 是**近似口径**
        （见 `_content_word_tokens`），带 `content_words_available` 可用性标记。
      · `zh`（中文尺子）＝ 汉字数字 + 中文时间词 + 人物/地点指称 的并集
        （见 `_zh_detail_hits`；后三类复用候选判据②那三张已标定的表）。
    两支的明细分别放在 `detail_parts` / `zh_detail_parts` —— 下游要分来源取用时读它们，
    不要让 `details` 一个数承担两种语义。**两支抓得到的东西本来就不同**，所以
    `details` **只能同口径相比**（`compare_expression_metrics` 跨口径直接不报密度）。
    全部是词表与形态规则，无模型参与——两遍口径一致，差值才有意义。
    切句认全角半角；语音转写常无句末标点：完全没有句末标点时整段按 1 句计。
    """
    raw = " ".join(str(text or "").split())
    if not raw:
        return {
            "sentences": 0, "words": 0, "details": 0, "connectives": 0,
            "detail_parts": {"numbers": 0, "time_words": 0, "content_words": 0,
                             "content_words_available": True},
            "zh_detail_parts": {"numbers": 0, "time_words": 0,
                                "person_refs": 0, "place_refs": 0},
            "detail_caliber": "en", "numerator_measured": False,
            "density": _density_of(0, 0, False, "en"),
            "signals": _expression_signals(""),
        }
    # 句数：全角半角句末标点一并切（省略号/连续标点折叠）；
    # 完全没有句末标点按 1 句（学习者的短句不能因为没标点就被算成 0）。
    sentence_parts = [s for s in _SENTENCE_SPLIT_RE.split(raw) if s.strip()]
    sentences = len(sentence_parts) if _SENTENCE_END_RE.search(raw) else 1
    tokens = normalize_chunk(raw).split()
    number_tokens = {t for t in tokens if t.isdigit() or t in _NUMBER_TOKENS}
    time_tokens = {t for t in tokens if t in _TIME_TOKENS}
    content_tokens = _content_word_tokens(tokens)
    content_available = _content_words_available(tokens)
    en_details = len(number_tokens | time_tokens | content_tokens)
    # 分子口径与**逐句**闸门（§4.2 第二版定案）：见 `_pick_detail_caliber`。
    caliber, measured = _pick_detail_caliber(sentence_parts, tokens)
    details = _zh_details(raw) if caliber == "zh" else en_details
    # 序列匹配吃掉的 token 不再按单词重复计（"after that" 是 1 个衔接词，
    # 不是 after + that 各一次；"and then" 同理只计 1）。
    consumed: set[int] = set()
    connective_seqs = 0
    for i in range(len(tokens)):
        for seq in _CONNECTIVE_SEQS:
            if tokens[i:i + len(seq)] == list(seq):
                connective_seqs += 1
                consumed.update(range(i, i + len(seq)))
    connective_set = {
        tokens[i] for i in range(len(tokens))
        if tokens[i] in _CONNECTIVE_TOKENS and i not in consumed
    }
    return {
        "sentences": sentences,
        "words": len(tokens),
        "details": details,
        # 英文尺子的三个来源各自的原样数（各自诚实：details 只是它们的并集大小）
        "detail_parts": {
            "numbers": len(number_tokens),
            "time_words": len(time_tokens),
            "content_words": len(content_tokens),
            # 纯中文讲述→False：这一半拿不到，下游别把 0 当结论用
            "content_words_available": content_available,
        },
        # 中文尺子的四类各自的原样条数（§4.2 第二版定案）
        "zh_detail_parts": _zh_detail_parts(raw),
        # 这一遍用的是哪把尺子，以及分子有没有**逐句**被量到
        "detail_caliber": caliber,
        "numerator_measured": measured,
        "connectives": len(connective_set) + connective_seqs,
        # §4.2 比率制：密度由代码算好（句数<2 或分子逐句没量到时 available=False，
        # 下游据此**不报这个数**）
        "density": _density_of(details, sentences, measured, caliber),
        # §4.2 两个候选判据（重点提炼 / 必要信息补充），各自带可用性标记
        "signals": _expression_signals(raw),
    }


def _detail_parts_of(metrics: dict) -> dict:
    """从一份 metrics 里取三个来源的原样数（缺字段按 0，形状永远稳定）。

    缺 `content_words_available` 字段时按 True 处理（老载荷没有这个标记，
    不能在读取侧凭空把它变成"不可用"）。
    """
    parts = (metrics or {}).get("detail_parts")
    parts = parts if isinstance(parts, dict) else {}
    return {
        "numbers": int(parts.get("numbers") or 0),
        "time_words": int(parts.get("time_words") or 0),
        "content_words": int(parts.get("content_words") or 0),
        "content_words_available": bool(parts.get("content_words_available", True)),
    }


def _zh_detail_parts_of(metrics: dict) -> dict:
    """从一份 metrics 取**中文尺子**四类的原样条数（老载荷缺这个键 → 全 0，形状稳定）。

    中文分子的明细只在 `detail_caliber == "zh"` 时有意义；英文那一遍它会是 0，
    那是"这一遍没用中文尺子"，**不是**"中文一项都没有"——提示侧只允许在中文口径下
    引用它，别把英文那遍的 0 读成结论。
    """
    parts = (metrics or {}).get("zh_detail_parts")
    parts = parts if isinstance(parts, dict) else {}
    return {
        "numbers": int(parts.get("numbers") or 0),
        "time_words": int(parts.get("time_words") or 0),
        "person_refs": int(parts.get("person_refs") or 0),
        "place_refs": int(parts.get("place_refs") or 0),
    }


def _signals_of(metrics: dict) -> dict:
    """从一份 metrics 取两个候选判据的结果；老载荷没这个键时返回不可用形状。"""
    signals = (metrics or {}).get("signals")
    if isinstance(signals, dict) and signals:
        return signals
    return _expression_signals("")


def compare_expression_metrics(first: dict, second: dict) -> dict:
    """两遍讲述的差值（F14「再讲一遍」）：全部由代码算出，模型只许引用。

    §4.2 比率制（2026-09-19 定案）：对外只说**比率**或**同一人两遍的差**——
      信息密度 = `details ÷ sentences`（两边都 >= 2 句才成立）；
      简洁进步 = `sentences` 下降 且 `details` 不下降。
    §4.2 第二版（中文分子的口径 + 逐句闸门）：`details` 有 `en`/`zh` 两支尺子，
    **两支抓得到的东西本来就不同**（中文只数得到汉字数字/中文时间词/人物/地点指称，
    抓不到抽象内容词）→ 两遍口径不一致时密度**一律不报**（`caliber_mismatch`）：
    跨语言比密度会把"中文口径天然低"读成"这篇讲得差"。
    `words` 仍原样带出来（长度观测），但**不进任何比率、不进对比提示**。
    """
    first = first or {}
    second = second or {}
    c_first = _caliber_of(first)
    c_second = _caliber_of(second)
    same_caliber = c_first == c_second
    if same_caliber:
        d_first = density_of_metrics(first)
        d_second = density_of_metrics(second)
    else:
        # 跨口径不比密度（§4.2「密度只能在同一人、**同一语言**的两遍之间比」）
        d_first = _unavailable_density("caliber_mismatch", c_first)
        d_second = _unavailable_density("caliber_mismatch", c_second)
    delta_details = int(second.get("details") or 0) - int(first.get("details") or 0)
    delta_sentences = (int(second.get("sentences") or 0)
                       - int(first.get("sentences") or 0))
    delta_density = None
    delta_density_text = ""
    if d_first["available"] and d_second["available"]:
        delta_density = d_second["value"] - d_first["value"]
        delta_density_text = "%+.2f" % delta_density
    # 简洁进步 = 句数下降 且 details 不下降。**两边密度都可算**才认：无标点那一遍
    # 会退化成 1 句，此时"句数下降"是标点造成的假象，不是讲得更简洁（§4.2 定案
    # "无标点讲述不报密度"，这里同口径不判简洁进步——收紧，不是放松）。
    sentences_down = delta_sentences < 0
    details_not_down = delta_details >= 0
    density_ready = bool(d_first["available"] and d_second["available"])
    concise = bool(density_ready and sentences_down and details_not_down)
    if not same_caliber:
        verdict_reason = "caliber_mismatch"
    elif not density_ready:
        verdict_reason = "density_unavailable"
    elif concise:
        verdict_reason = "concise_progress"
    elif not sentences_down and not details_not_down:
        verdict_reason = "longer_and_less_detail"
    elif not sentences_down:
        verdict_reason = "not_shorter"
    else:
        verdict_reason = "details_dropped"
    return {
        "first": {
            "sentences": int(first.get("sentences") or 0),
            "words": int(first.get("words") or 0),
            "details": int(first.get("details") or 0),
            "connectives": int(first.get("connectives") or 0),
            "detail_parts": _detail_parts_of(first),
            "zh_detail_parts": _zh_detail_parts_of(first),
            "caliber": c_first,
            "density": d_first,
            "signals": _signals_of(first),
        },
        "second": {
            "sentences": int(second.get("sentences") or 0),
            "words": int(second.get("words") or 0),
            "details": int(second.get("details") or 0),
            "connectives": int(second.get("connectives") or 0),
            "detail_parts": _detail_parts_of(second),
            "zh_detail_parts": _zh_detail_parts_of(second),
            "caliber": c_second,
            "density": d_second,
            "signals": _signals_of(second),
        },
        "caliber_mismatch": not same_caliber,
        "delta": {
            "details": delta_details,
            "connectives": int(second.get("connectives") or 0) - int(first.get("connectives") or 0),
            "sentences": delta_sentences,
            "words": int(second.get("words") or 0) - int(first.get("words") or 0),
            # 比率制的差（两边都可算才有值；否则 None + 空串 = 下游不报）
            "density": delta_density,
            "density_text": delta_density_text,
        },
        "verdict": {
            "available": density_ready,
            "concise_progress": concise,
            "sentences_down": sentences_down,
            "details_not_down": details_not_down,
            "reason": verdict_reason,
        },
        "rule": (
            "进步必须用这里的原样数字表达（如 'you packed two more details into "
            "each sentence this time'）——密度 = details ÷ sentences，是**比率**，"
            "只用于与学习者自己的上一遍对比，**不当水平评价**；details 是**近似口径**"
            "（靠词表与形态规则算，无模型参与、不看大小写），且分两支尺子："
            "英文口径 = 数字 + 时间词 + 内容词；中文口径 = 汉字数字 + 中文时间词 + "
            "人物/地点指称（**中文只数得到这几类，密度天然低于英文口径**）。"
            "**禁止跨语言比密度**：密度只能在同一人、**同一语言**的两遍之间比，"
            "两遍口径不同时（caliber_mismatch）不报密度、不判简洁进步、也不要比 details；"
            "任一遍不足 2 句时密度**不可用**（无标点讲述会退化成 1 句），"
            "不可用时不要说密度、也不要判简洁进步；"
            "数字由代码计算得出，不得更改、不得四舍五入、不得自行估计；"
            "delta 为 0 或负数时如实说，不编造进步。"
        ),
    }


def asks_for_expression_feedback(text: str) -> bool:
    """用户主动要求表达反馈（§4.6 触发条件之二）。"""
    return bool(re.search(
        r"(帮我看看|给我.{0,6}反馈|讲得怎么样|说得怎么样|表达.{0,4}(反馈|评价)|"
        r"怎么样.{0,4}改进|如何改进|feedback|how (did|does) it sound|"
        r"any advice|what should i improve)",
        str(text or ""), re.IGNORECASE,
    ))
