"""素材分析：真标题 / 摘要 / 分块（plans/0916-21 D4/F3）。

⚠ **这是全项目唯一允许出现"素材分析模型"的地方。**
用户 2026-09-17 明确："我后续会用别的大模型做素材分析，豆包只是当前实现。"
所以：换模型 / 换 endpoint / 换提示词，**只改本文件**。抓取
（`engine/link_fetcher.py`）、存储（l2 的 `material_items`）、注入
（`tutor_agent` 的 `[Material]` 段）**都不得出现模型名**。

分层理由：抓取解析是确定性的、可测的；摘要/分块是模型相关的、注定要换的。
把"可变的部分"关进一个盒子 —— 这就是架构的价值（用户原话的落地）。

失败姿势：摘要**失败不阻塞导入**（原文还在，探讨照样能基于原文块进行），
返回空摘要即可；标题永远给得出（纯确定性，不依赖模型）。
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from engine.material_claim_scope import MATERIAL_CLAIM_SCOPE_RULE

from typing import Any, Dict, List
from urllib.parse import urlparse

# ── 换模型只需动这里（+ summarize_material 里的 prompt）────────────────────
# 2026-09-18：原来叫 `doubao-material-analyzer-v1` —— 当时模型确实在方舟（豆包）。
# 主端点切到 DeepSeek 官方后这个名字就**说错了供应商**（而且模型本来就可用
# MATERIAL_ANALYZER_MODEL / MATERIAL_VISION_MODEL 覆盖），改成不绑定厂商的名字。
# 只被 analyze_material() 回填给调用方、并在 test_material_import.py 里按常量断言，
# 没有别处按它过滤，所以改名安全。
ANALYZER_NAME = "material-analyzer-v1"
ANALYZER_MODEL_ENV = "MATERIAL_ANALYZER_MODEL"   # 空则用 llm_client 的默认模型

SUMMARY_TRIGGER_CHARS = 6000   # 超过才摘要：导入不在轮次路径上，一次调用可接受
SUMMARY_INPUT_CHARS = 12000    # 送进模型的原文上限（超出部分截断，摘要只求"够用"）
BLOCK_CHARS = 1200             # 分块大小（探讨时按轮注入相关块）
MAX_BLOCKS = 12                # 分块上限：注入有长度上限（方案 §7-5），块数必须封顶
TITLE_MAX_CHARS = 6            # 方案 §2：标题 3–6 字

# 页面标题常带站点后缀（"XXX - 维基百科"/"XXX | 豆瓣"），去掉才像真标题
_SITE_SUFFIX = re.compile(r"\s*[-|_–—·]\s*[^-|_–—·]{2,24}$")
_KEEP_CHARS = re.compile(r"[0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff]+")
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def pick_title(raw_text: str, title_hint: str = "", fallback: str = "") -> str:
    """3–6 字真标题（确定性，不花模型调用）。

    优先级：页面/用户给的标题线索 → 正文第一行有意义的内容。
    中文按字截，英文按词截 —— 两者都要"像标题"而不是被硬切断。
    """
    candidate = _clean(title_hint) or _clean(fallback)
    if not candidate:
        for line in str(raw_text or "").splitlines():
            stripped = _clean(line).lstrip("#*>-・•").strip()
            # 坑：len(findall) 数的是**段数**不是字数——带 `+` 的模式会把一整串中文
            # 合成一个匹配，"周会纪要" 只算 1 段，于是这里永远判不过、标题全退成"素材"
            # （实测踩到）。要的是"有几个可读字符"，所以把匹配拼起来再量长度。
            if len("".join(_KEEP_CHARS.findall(stripped))) >= 2:
                candidate = stripped
                break
    if not candidate:
        return "素材"
    candidate = _SITE_SUFFIX.sub("", candidate).strip() or candidate
    if re.search(r"[\u4e00-\u9fff]", candidate):
        # 中文：先去掉内部空白再按字截。否则 3–6 字的预算会被空格吃掉一个字符，
        # 截出来是"行尸走肉 第"这种断词；去空白后是"行尸走肉第一"，才像真标题。
        dense = re.sub(r"\s+", "", candidate)
        return dense[:TITLE_MAX_CHARS]
    words = candidate.split()
    short = " ".join(words[:4]).strip()
    return (short or candidate)[:32]


def split_blocks(raw_text: str, block_chars: int = BLOCK_CHARS,
                 max_blocks: int = MAX_BLOCKS) -> List[str]:
    """把正文切成若干块（按段落聚合，超长段落硬切）。

    探讨时按轮注入**相关**块，所以切分要尽量不切断段落语义。
    `max_blocks <= 0` = 不封顶（plans/0923-08 的全量存储走这条；注入侧仍按预算裁剪，
    详见 `select_blocks`）。
    """
    text = str(raw_text or "").strip()
    if not text:
        return []
    paragraphs: List[str] = []
    for para in re.split(r"\n{1,}", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > block_chars:      # 单段超长 → 硬切，否则一块永远装不下
            paragraphs.append(para[:block_chars])
            para = para[block_chars:]
        if para:
            paragraphs.append(para)

    capped = int(max_blocks or 0) > 0
    blocks: List[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > block_chars:
            blocks.append(current)
            if capped and len(blocks) >= max_blocks:
                current = ""
                break
            current = para
        else:
            current = (current + "\n" + para) if current else para
    if current and (not capped or len(blocks) < max_blocks):
        blocks.append(current)
    return blocks


def split_all_blocks(raw_text: str, *, block_chars: int = BLOCK_CHARS) -> List[str]:
    """不封顶的全量分块（plans/0923-08 §4.2）。

    与 `split_blocks` **同一套段落边界**——只是不施加 `MAX_BLOCKS=12` 的存储上限。
    12 块是**注入**预算的产物，不是存储上限；拿它当存储上限会让长文后段
    「原文明明有、检索永远找不到」。
    """
    return split_blocks(raw_text, block_chars=block_chars, max_blocks=0)


# 链接导入把评论拼在正文后时用的分隔标记（main.py 写、本文件读，同一字面量）。
COMMENT_MARKER = "（以下是这条内容下的观众评论，来自观众，不是正文：）"


def split_material_spans(raw_text: str, *, block_chars: int = BLOCK_CHARS) -> List[Dict[str, Any]]:
    """把一份素材的原文切成**可定位的全体块**，正文/评论分层（plans/0923-08 §4.2）。

    返回 `[{"block_no": 1, "span_kind": "body"|"comment", "text": ...}]`，块号从 1
    连续编号 —— 证据包里的 `material_id:23` 就是指这个块号，所以编号必须稳定。

    归属规则（确定性，零模型）：
      * 有 `COMMENT_MARKER` → 标记前 = body、标记后 = comment；
      * 没有标记 → 整篇 body（旧文本/本地文件本来就没有评论段）。
    评论永远不进"剧情事实"的权威区（方案 §4.3）；用户问到的若是评论内容，
    证据块会带着 `span_kind='comment'` 一起进包，模型据此说清"这是观众说的"。
    """
    text = str(raw_text or "")
    body, comment = text, ""
    if COMMENT_MARKER in text:
        body, _, comment = text.partition(COMMENT_MARKER)
    out: List[Dict[str, Any]] = []
    for chunk in split_all_blocks(body, block_chars=block_chars):
        out.append({"block_no": len(out) + 1, "span_kind": "body", "text": chunk})
    for chunk in split_all_blocks(comment, block_chars=block_chars):
        out.append({"block_no": len(out) + 1, "span_kind": "comment", "text": chunk})
    return out


def legacy_blocks_to_spans(old_blocks: List[str], raw_text: str,
                           *, block_chars: int = BLOCK_CHARS) -> List[Dict[str, Any]]:
    """旧素材（plans/0916-21 那批，只有 12 块 JSON 数组）→ 全量块的**回填**产物。

    口径（方案 §4.2）：能由 `raw_text` 重新切出来的走 `split_material_spans`，
    旧数组里那些**在 raw_text 里找不到**的片段标 `legacy_unattributed` —— 它们无法
    可靠归属正文还是评论，因此**不作剧情事实**（`span_kind` 不是 body 就不会被
    证据包当权威原文用）。回填不许把未索引的后段说成"原文没有"。
    """
    spans = split_material_spans(raw_text, block_chars=block_chars)
    if not spans:
        # raw_text 空（旧行可能有 blocks 没 raw_text）：只认确实存在的旧块
        spans = []
    known = " ".join(str(span.get("text") or "") for span in spans)
    for raw_block in list(old_blocks or []):
        text = _clean(raw_block)
        if not text:
            continue
        if text and text[:80] in known:
            continue                       # 已被全量分块覆盖，不重复写
        spans.append({"block_no": len(spans) + 1,
                      "span_kind": "legacy_unattributed", "text": text})
    return spans


def recover_legacy_source_blocks(row: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Recover stored content without fetching media or inventing provenance."""
    analysis = dict(row.get("analysis") or {})
    raw = str(row.get("raw_text") or "")
    old = row.get("blocks") or []
    if isinstance(old, str):
        try:
            import json
            old = json.loads(old)
        except (TypeError, ValueError):
            old = []
    spans = legacy_blocks_to_spans(old if isinstance(old, list) else [], raw)
    from engine.link_fetcher import is_video_url
    coverage = analysis.get("source_coverage") if isinstance(analysis.get("source_coverage"), dict) else {}
    if is_video_url(str(row.get("source_ref") or "")) and not isinstance(coverage.get("body_available"), bool):
        # An old page dump might contain captions, metadata, or both. Existence
        # of text is not proof of a verified video transcript.
        for span in spans:
            if span["span_kind"] == "body":
                span["span_kind"] = "legacy_unattributed"
        analysis["source_coverage"] = {"kind": "video", "body_available": None,
                                       "reason": "legacy_provenance_unknown"}
    return spans, analysis


def needs_summary(raw_text: str) -> bool:
    return len(str(raw_text or "")) > SUMMARY_TRIGGER_CHARS


# ── 相关块选择（2026-09-21）──────────────────────────────────────────────
#
# 注入预算有限（每轮 prompt），而素材可能有十几块。原实现按数组顺序取前几块，
# 于是"用户问第 7 节、模型手里只有第 1 节"——合同又写着"素材里没写的不得编"，
# 模型只能编，或者答"素材里没有"。实测 11,240 字素材 10 块只有 1 块（9.7%）
# 进 prompt；24,000 字以上素材同样只进 1,206 字（8.4%）。
#
# 选块**确定性、零模型调用**：素材盘的问题多是"人名/剧名/第几集/情节词"这类
# 实词，词面命中就够；用模型选块等于把一次调用加到每轮路径上，得不偿失。

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]{2,}")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

# 只列高频功能词：素材盘问句里真正给信号的是实词（人名/剧名/情节词）
_QUERY_STOPWORDS = frozenset((
    "the", "and", "for", "you", "your", "are", "was", "were", "with", "that",
    "this", "these", "those", "what", "which", "who", "whose", "when", "where",
    "why", "how", "did", "does", "doing", "have", "has", "had", "about", "from",
    "they", "them", "their", "his", "her", "she", "him", "its", "but", "not",
    "can", "could", "would", "should", "will", "shall", "may", "might", "tell",
    "talk", "say", "said", "know", "think", "like", "just", "really", "there",
    "here", "then", "than", "some", "any", "all", "one", "two", "get", "got",
    "make", "made", "going", "want", "need", "much", "many", "more", "most",
))
_CN_QUERY_STOPWORDS = frozenset((
    "什么", "怎么", "这个", "那个", "哪个", "哪些", "一下", "我们", "你们",
    "他们", "她们", "知道", "觉得", "聊聊", "说说", "告诉", "可以", "没有",
    "还有", "意思", "是不", "不是", "就是", "然后", "所以", "但是", "因为",
    "如果", "已经", "还是", "或者", "而且", "一样", "为什",
    # 疑问/指代：问句里这些词到处都有，留着会跟正文随机撞上
    "是谁", "哪儿", "哪里", "多少", "几个", "这是", "那是", "的话", "时候",
))


def query_tokens(query: str) -> set:
    """查询串 → 词元集：英文取实词，中文取二元组。

    中文没有词边界，二元组是够用的近似——"亚特兰大"贡献 亚特/特兰/兰大
    三个词元，命中越多说明这一块越相关。
    """
    text = str(query or "").lower()
    tokens = set()
    for word in _ASCII_WORD.findall(text):
        if word not in _QUERY_STOPWORDS:
            tokens.add(word)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.add(run)
            continue
        for index in range(len(run) - 1):
            bigram = run[index:index + 2]
            if bigram not in _CN_QUERY_STOPWORDS:
                tokens.add(bigram)
    return tokens


def score_blocks(blocks: List[str], query: str) -> List[float]:
    """每块与本轮 query 的词面相关度（与 blocks 等长、同序）。

    权重取词元长度：命中"亚特兰大"比命中"第一"更能说明用户问的就是这一块。
    """
    tokens = query_tokens(query)
    if not tokens:
        return [0.0] * len(blocks)
    scores = []
    for block in blocks:
        text = str(block or "").lower()
        scores.append(float(sum(len(token) for token in tokens if token in text)))
    return scores


def select_blocks(blocks: List[str], query: str, limit: int) -> List[int]:
    """挑出与本轮最相关的至多 limit 块，返回**原文顺序**的下标。

    顺序按原文而不是按得分：块被剪成孤岛读起来会断（第 7 块排在第 3 块前面，
    叙事就乱了）。零命中（寒暄、"接着聊"这类没指哪块的话）退回开头的块——
    那时开头的块承担"这份素材在讲什么"的锚点作用。
    """
    total = len(blocks)
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    scores = score_blocks(blocks, query)
    hits = [index for index in range(total) if scores[index] > 0]
    if not hits:
        return list(range(limit))
    hits.sort(key=lambda index: (-scores[index], index))
    picked = set(hits[:limit])
    # 有余量时优先补命中块的邻居：叙事类素材（剧集/长文）上下文连着才读得懂
    for index in hits:
        if len(picked) >= limit:
            break
        for neighbor in (index + 1, index - 1):
            if len(picked) >= limit:
                break
            if 0 <= neighbor < total and neighbor not in picked:
                picked.add(neighbor)
    return sorted(picked)


def _kind_label(kind: str, default: str = "一段素材") -> str:
    return {"diary": "一篇用户写的日记", "link": "一篇用户导入的网页文章",
            "text": "一段用户粘贴的素材"}.get(str(kind or "text"), default)


def _summary_prompt(text: str, kind: str, *, partial: bool = False) -> str:
    """摘要提示词。partial=真 时是"分段读"的一遍，措辞必须禁止它替全文下结论。"""
    if partial:
        head = "下面是%s的**其中一段**。" % _kind_label(kind)
        scope = ("这是分段阅读的一段：只写**这一段里**明确出现的事实，"
                 "不要猜测其他部分、不要写总括性的评价。\n")
    else:
        head = "下面是%s。" % _kind_label(kind)
        scope = ""
    source_note = (
        "④ 方括号里的 speaker/time 是语音转写线索，不是人物身份。视频若混有创作者解说、"
        "作品内对白和评论，分别说明是谁在说；创作者用第一人称代入游戏角色时，"
        "不能把他的评价或操作说成角色自己的台词或想法。无法确定说话来源就保留不确定。\n"
        if "[speaker=" in text else ""
    )
    return (
        head + "请用 3-5 句中文写一段内容提要，要求：\n"
        "① **只写素材里明确出现的事实**（人物、集数、情节、地点、观点），"
        "绝不允许补充素材里没有的细节 —— 这些提要会被用来跟用户聊，编造等于骗人；\n"
        "② 不评价、不建议、不写成学习笔记、不列条目标题；\n"
        "③ 直接输出提要正文。\n" + source_note + scope +
        "\n素材：\n%s" % text[:SUMMARY_INPUT_CHARS]
    )


async def _summary_once(text: str, kind: str, *, partial: bool = False) -> str:
    """一次摘要调用。失败返回空串，绝不抛。"""
    try:
        from llm_client import (DEFAULT_CHAT_MODEL, robust_llm_client,
                                thinking_disabled_extra_body)
        model = os.getenv(ANALYZER_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL
        raw = await robust_llm_client.chat_completion(
            [{"role": "user", "content": _summary_prompt(text, kind, partial=partial)}],
            model=model, temperature=0.2, max_tokens=500, timeout=45,
            extra_body=thinking_disabled_extra_body(model),
        )
        return str(raw or "").strip()
    except Exception:
        return ""


async def _summary_merge(parts: List[str], kind: str) -> str:
    """分段提要 → 一段全文提要（归并也是一次模型调用）。失败退回拼接。"""
    joined = "\n".join("- " + part for part in parts if part)
    try:
        from llm_client import (DEFAULT_CHAT_MODEL, robust_llm_client,
                                thinking_disabled_extra_body)
        model = os.getenv(ANALYZER_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL
        prompt = (
            "下面是同一份%s**按顺序分段**写出的内容提要。请合并成 3-5 句的全文提要，"
            "要求：\n① 覆盖到每一段里的事实，不要只写开头那几段；\n"
            "② 只保留提要里出现过的事实，不得补充新细节；\n"
            "③ 不评价、不列条目标题，直接输出提要正文。\n\n分段提要：\n%s"
            % (_kind_label(kind), joined[:SUMMARY_INPUT_CHARS])
        )
        raw = await robust_llm_client.chat_completion(
            [{"role": "user", "content": prompt}],
            model=model, temperature=0.2, max_tokens=500, timeout=45,
            extra_body=thinking_disabled_extra_body(model),
        )
        merged = str(raw or "").strip()
        if merged:
            return merged
    except Exception:
        pass
    return " ".join(part for part in parts if part).strip()


async def summarize_material(raw_text: str, *, kind: str = "text") -> str:
    """素材摘要。**超长素材分段各读一遍再归并**，失败返回空串，绝不抛。

    2026-09-21：原实现只把**前 12,000 字**送进模型，长素材的后段从来没被读过
    （"知道个大概"）。用户口径："要彻底了解到这个视频的内容，而不是这种囫囵吞枣"，
    且明确接受为此多花时间。分段调用**并行发出**，墙钟时间≈一次调用。
    """
    text = str(raw_text or "").strip()
    if not text:
        return ""
    windows = _windows(text, SUMMARY_INPUT_CHARS)
    if len(windows) == 1:
        return await _summary_once(windows[0], kind)
    parts = [part for part in await asyncio.gather(
        *[_summary_once(window, kind, partial=True) for window in windows]) if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return await _summary_merge(parts, kind)


async def analyze_material(raw_text: str, *, kind: str = "text",
                           title_hint: str = "", allow_model: bool = True) -> Dict:
    """素材分析的**单一入口**：返回 {title, summary, blocks, char_count, analyzer}。

    调用方只需 import 这一个函数；模型名只在本文件出现。
    """
    text = str(raw_text or "")
    summary = ""
    if allow_model and needs_summary(text):
        summary = await summarize_material(text, kind=kind)
    return {
        "title": pick_title(text, title_hint=title_hint),
        "summary": summary,
        "blocks": split_blocks(text),
        "char_count": len(text),
        "analyzer": ANALYZER_NAME,
    }


# ── 深度分析层（plans/0916-24 §三·五 修复 2）──────────────────────────────
#
# 用户原话："AI 要对他这个内容有一些自己的想法……NotebookLM 那个播客的形成，
# 其实就是它对扫描的素材进行了一个分析之后……发现了不一样的观点……这种还是
# 分析的目的。" 所以"读懂"（摘要）不够，还要有**观点**。
#
# ⚠ 借的是"分析出观点"这一层，**不借"做成播客"**（V9 本期不做已否决播客形态：
# 那是听力层、与输出目标不挂钩）。产物必须**服务输出**——`hooks`/`viewpoints`
# 是给 AI 用来引发用户开口的谈资，不是给用户听的一段音频。
#
# 产物是**默认理解框架，不是硬规格**（用户 2026-09-17 口径："具体往哪个方向走
# 看用户的需求"）：缺项、空项都正常（散文没有 timeline），不预设"每份都要出全六项"。
# 与摘要同批、导入时一次性生成，**不进每轮延迟路径**。

DEEP_MODEL_ENV = "MATERIAL_DEEP_MODEL"   # 空则用 MATERIAL_ANALYZER_MODEL / 默认模型
DEEP_THINKING_ENV = "MATERIAL_DEEP_THINKING"  # DeepSeek 深析默认启用 DeepThink
DEEP_THINKING_EFFORT_ENV = "MATERIAL_DEEP_THINKING_EFFORT"
DEEP_THINKING_MAX_TOKENS_ENV = "MATERIAL_DEEP_THINKING_MAX_TOKENS"
DEEP_THINKING_MAX_TOKENS = 16000  # reasoning 与最终 JSON 共用 max_tokens
DEEP_TRIGGER_CHARS = 1200                 # 太短的素材不值得花一次"深读"调用
DEEP_INPUT_CHARS = 12000                  # 与摘要同量级的输入预算
DEEP_FIELDS = ("topics", "people", "timeline", "viewpoints", "hooks", "gaps",
               "voice_sources", "voice_source_spans", "source_map",
               "external_context_queries")
_DEEP_LIST_MAX = 8                        # 每个字段条数上限（注入预算有限）
_VOICE_SOURCE_SPANS_MAX = 48              # 跨窗口保留时间线覆盖，不按 speaker 合并
_VOICE_SOURCE_SPANS_PER_WINDOW = 16

# Public-link background context is fetched once during library import, outside
# the conversation turn. It is stored separately from transcript/comments and
# is only injected when the user's question needs that context.
EXTERNAL_CONTEXT_ENABLED_ENV = "MATERIAL_EXTERNAL_CONTEXT_ENABLED"
EXTERNAL_CONTEXT_SCHEMA_VERSION = 6
EXTERNAL_CONTEXT_QUERY_POLICY = 4
EXTERNAL_CONTEXT_MAX_SOURCES = 3
EXTERNAL_CONTEXT_SNIPPET_CHARS = 420

_EXTERNAL_QUERY_STOP = {
    "official", "background", "setting", "characters", "story", "game", "video",
    "about", "from", "with", "that", "this", "site", "com", "lore", "entity",
    "phenomenon", "phenomena", "supernatural", "paranatural", "officially",
}


def _external_context_terms(title: str, analysis: Dict[str, Any],
                            comments: List[str]) -> List[str]:
    """Extract search candidates, not facts, from a public video analysis."""
    terms: List[str] = []

    def add(value: Any, *, first_clause: bool = False) -> None:
        text = " ".join(str(value or "").split()).strip(" \t\r\n-:：|()[]{}\"'“”")
        if first_clause:
            text = re.split(r"[：:—–|，,。；;]", text, maxsplit=1)[0].strip()
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = " ".join(text.split())[:100]
        if (len(text) < 2 or text.casefold() in {item.casefold() for item in terms}
                or re.fullmatch(r"speaker[_ -]?\d+", text, re.IGNORECASE)):
            return
        terms.append(text)

    add(title)
    source_map = analysis.get("source_map")
    if isinstance(source_map, dict):
        for entity in (source_map.get("entities") or [])[:4]:
            if isinstance(entity, dict):
                add(entity.get("name"))
    for person in (analysis.get("people") or [])[:4]:
        add(person, first_clause=True)
    for topic in (analysis.get("topics") or [])[:2]:
        add(topic, first_clause=True)

    # Comment terms are only search hypotheses. They never enter the source
    # map and cannot establish what happened in the video.
    stop = {"THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WHAT", "WHEN",
            "WHY", "HOW", "GAME", "VIDEO", "PART"}
    comment_text = " ".join(str(value or "")[:500] for value in (comments or [])[:12])
    for candidate in re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", comment_text):
        if candidate.upper() not in stop:
            add(candidate)
    return terms[:8]


def _external_context_query(title: str, analysis: Dict[str, Any],
                            comments: List[str]) -> str:
    candidates = analysis.get("external_context_queries")
    if isinstance(candidates, list):
        for candidate in candidates[:2]:
            query = " ".join(str(candidate.get("query") if isinstance(candidate, dict)
                                  else candidate or "").split()).strip()
            if len(query) >= 8:
                return query[:240]
    terms = _external_context_terms(title, analysis, comments)
    if not terms:
        return ""
    # Stable fallback for legacy analyses without a model-proposed search query.
    # Prefer a compact franchise/entity/topic query over a bag of every name.
    acronyms = [term for term in terms if term.isupper() and len(term) >= 3]
    people = [term for term in terms if term not in acronyms]
    query_terms = []
    for term in (people[:1] + acronyms[:2] + people[1:2]):
        if term not in query_terms:
            query_terms.append(term)
    query = " ".join(query_terms + ["official game setting story"])
    return " ".join(query.split())[:240]


def _external_context_comment_terms(comments: List[str]) -> List[str]:
    """Keep exact named terms from comments as search clues, never as facts.

    Acronyms such as FBC are easy to retain. Chinese transliterations are less
    likely to be emitted as acronyms, so also capture names that directly
    follow an explicit relation verb (for example, ``对抗西斯``). The candidates
    are used only to form a search query; the comment itself remains unverified.
    """
    stop = {
        "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WHAT", "WHEN",
        "WHY", "HOW", "GAME", "VIDEO", "PART", "FBC", "LOL", "DLC", "FPS",
        "世界观", "评论区", "视频", "游戏", "角色", "观众", "内容", "观点", "作者",
        "剧情", "里面", "这里", "那个", "这个", "他们", "我们", "大家", "一个",
        "未知力量", "超自然力量", "异常现象", "共振体", "共振现象", "缺乏认知",
    }
    found: List[str] = []

    def add(value: str) -> None:
        candidate = " ".join(str(value or "").split()).strip(" \t\r\n-:：|()[]{}\"'“”")
        if (len(candidate) < 2 or candidate.upper() in stop
                or candidate.casefold() in {item.casefold() for item in found}):
            return
        found.append(candidate[:48])

    text = "\n".join(str(value or "")[:1800] for value in (comments or [])[:8])
    for token in re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", text):
        add(token)
    for match in re.finditer(
        r"(?:叫作|叫做|也叫|又称|称为|名为)\s*"
        r"([\u4e00-\u9fff]{2,8}|[A-Z][A-Za-z0-9_-]{1,})", text,
    ):
        add(match.group(1))
    # In Chinese comments a two-character proper name is sometimes introduced
    # only by a verb such as "对抗西斯". Accept a short token only when it is
    # immediately followed by punctuation; this avoids treating phrases like
    # "面对未知共振体" or "缺乏认知" as named entities.
    for match in re.finditer(
        r"(?:对抗|抵抗|击退)\s*([\u4e00-\u9fff]{2,4})(?=[，,。；;！？!?：:\s]|$)",
        text,
    ):
        add(match.group(1))
    return found[:4]


def _supplement_comment_search_query(
    queries: List[str], title: str, analysis: Dict[str, Any], comments: List[str],
    protected_queries: Optional[List[str]] = None,
) -> List[str]:
    """Reserve one bounded query for an exact named term raised by commenters."""
    # Body claim verification takes precedence over secondary comment clues.
    if len(queries) >= 2 and any(query in (protected_queries or []) for query in queries):
        return queries[:2]
    targets = _external_context_comment_terms(comments)
    if not targets:
        return queries[:2]

    anchors: List[str] = []

    def add_anchor(value: Any) -> None:
        candidate = " ".join(str(value or "").split()).strip(" \t\r\n-:：|()[]{}\"'“”")
        candidate = re.split(r"[：:—–|，,。；;]", candidate, maxsplit=1)[0].strip()
        if (len(candidate) < 2 or candidate.casefold() in _EXTERNAL_QUERY_STOP
                or candidate.casefold() in {item.casefold() for item in anchors}):
            return
        anchors.append(candidate[:48])

    # Use the material's named people/entities and source identity as anchors;
    # do not inject a product-specific franchise or assume the comment is true.
    for token in re.findall(r"[\w\u4e00-\u9fff]+", str(title or "")):
        add_anchor(token)
    source_map = analysis.get("source_map") if isinstance(analysis, dict) else {}
    if isinstance(source_map, dict):
        for entity in (source_map.get("entities") or [])[:4]:
            if isinstance(entity, dict):
                add_anchor(entity.get("name"))
    for person in (analysis.get("people") or [])[:4]:
        add_anchor(person)
    for topic in (analysis.get("topics") or [])[:4]:
        for token in re.findall(
            r"[A-Z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", str(topic or "")
        ):
            add_anchor(token)

    target_folded = {target.casefold() for target in targets[:2]}
    material_anchor_terms = {
        token.casefold()
        for anchor in anchors
        for token in re.findall(r"[\w\u4e00-\u9fff]+", anchor)
        if len(token) >= 2 and token.casefold() not in _EXTERNAL_QUERY_STOP
        and token.casefold() not in target_folded
    }
    anchor_tokens = {
        token.casefold() for anchor in anchors
        for token in re.findall(r"[\w\u4e00-\u9fff]+", anchor)
    }
    control_context = (
        "control" in anchor_tokens
        or {"dylan", "jesse"}.issubset(anchor_tokens)
    )
    canonical_aliases = (
        ["Hiss"] if control_context
        and any(target.casefold() in {"西斯", "希斯", "吸丝", "嘶啸"}
                for target in targets[:2])
        else []
    )
    # A query mentioning "Hiss" alone is still ambiguous. Keep it only when
    # it is tied to a title/entity/person from this material; otherwise spend
    # the second bounded query on that exact commenter-named term plus context.
    # A query writer may add the canonical spelling of a transliterated name
    # (for example, a Chinese comment's "西斯" alongside the game's "Hiss").
    # Keep that query when it is tied to material anchors and has a distinct
    # proper-name token, even though the translated spelling is not literal in
    # the comment. Otherwise this fallback replaces the useful canonical term
    # with a Chinese-only query that web search cannot resolve.
    target_is_grounded = False
    for query_index, query in enumerate(queries):
        query_tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
        folded_tokens = {token.casefold() for token in query_tokens}
        has_anchor = bool(material_anchor_terms.intersection(folded_tokens))
        has_exact_comment_term = any(
            target.casefold() in folded_tokens for target in targets[:2]
        )
        capitalized_name = any(
            re.fullmatch(r"(?:[A-Z][a-z][A-Za-z0-9_-]*|[A-Z]{2,})", token)
            and token.casefold() not in material_anchor_terms
            and token.casefold() not in _EXTERNAL_QUERY_STOP
            for token in query_tokens
        )
        # The first query is the broad orientation query when two are present;
        # only let a distinct target term in the second query satisfy this test.
        is_target_query = query_index > 0 or len(queries) == 1
        if is_target_query and has_anchor and (has_exact_comment_term or capitalized_name):
            target_is_grounded = True
            break
    if target_is_grounded:
        if canonical_aliases:
            target_indexes = [
                index for index, query in enumerate(queries)
                if any(target.casefold() in query.casefold() for target in targets[:2])
                and material_anchor_terms.intersection(
                    token.casefold() for token in re.findall(
                        r"[\w\u4e00-\u9fff]+", query
                    )
                )
            ]
            if not target_indexes and len(queries) == 1:
                target_indexes = [0]
            for index in target_indexes:
                if not any(alias.casefold() in queries[index].casefold()
                           for alias in canonical_aliases):
                    queries[index] = (queries[index] + " "
                                      + " ".join(canonical_aliases))[:220]
        return queries[:2]

    useful_anchors = [anchor for anchor in anchors
                      if anchor.casefold() not in target_folded]
    target_query = " ".join((targets[:2] + canonical_aliases
                              + useful_anchors[:4] + ["game", "background"]))[:220]
    if not target_query.strip():
        return queries[:2]
    # Keep one broad orientation query, but replace a redundant/model-generic
    # second query so the exact commenter-named concept is not lost.
    if len(queries) >= 2:
        return [queries[0], target_query]
    return (queries + [target_query])[:2]


async def _draft_external_context_queries(title: str, body_text: str,
                                          comments: List[str], query_purposes: Optional[Dict[str, str]] = None) -> List[str]:
    """Propose bounded background searches for older imports lacking query hints.

    This is a one-time import/tape-build operation. The generated strings are
    search hypotheses only; they are never treated as source facts.
    """
    body = " ".join(str(body_text or "").split())[:7000]
    comment_text = "\n".join(
        " ".join(str(value or "").split())[:1800]
        for value in (comments or [])[:4]
    )[:6500]
    comment_named_terms = _external_context_comment_terms(list(comments or []))
    if len(body) < 100:
        return []
    prompt = (
        "You prepare web-search queries to add only the minimum outside context needed "
        "to understand an imported media excerpt and assess a discussable claim. Return JSON only: "
        '{"external_context_queries":[{"query":"...","purpose":"background or claim_verification"}]}. Produce at most two concise queries. '
        "Identify the canonical work/franchise and named people, organization, or "
        "setting the excerpt assumes the audience knows. A distinctive cluster of "
        "character names and organizations may identify a work even when the video "
        "title omits it; include that canonical work title, while retaining at least "
        "one exact name from the supplied material in each query. Prefer official "
        "creator, publisher, or primary-source pages. Produce one broad orientation "
        "query, or a primary-source evidence query for a claim explicitly made in the body. "
        "If a non-fiction body makes a general claim about outcomes or effectiveness, reserve "
        "one query for independent evidence of that claim, not another biography query. "
        "Prioritize actionable causal/effectiveness claims over checking biographical honors or ages. "
        "Use the exact source-provided concept as the anchor; the person's name need not appear in a research query. "
        "For claim_verification include the English scientific equivalent and population/outcome terms "
        "alongside that source anchor, so primary research databases can find the studies. "
        "For non-fiction claims, prioritize independent measurements, original research, "
        "dates and population matching that claim; search for testing it, not confirming it. "
        "Never fabricate a statistic or search premise. A fictional event needs background, not real-world proof. "
        "If a body claim needs evidence, use the second query for that evidence before comment clues. "
        "Otherwise, if the comments contain a named or unusual term, the second query must "
        "be about that exact term, not a generic theme from the video. Preserve the exact "
        "spelling and script of that term; "
        "do not replace a specific named phenomenon with a generic category. Use comments "
        "as clues for what background may be missing, never as proof of a fact. Every "
        "query must contain at least one entity or term that literally appears in the "
        "supplied title, body, or comments; other terms may identify the work from that "
        "distinctive cluster, but must not add a plot claim. Do not introduce a sequel, future release, or plot "
        "event that is not explicitly named in the material. If a comment uses a "
        "non-English transliteration for a named concept, preserve that exact form and "
        "add its likely original-language spelling only when the work context makes it "
        "unambiguous. The second query must pair the exact term with the work title, "
        "its likely canonical spelling when identifiable, and one source-provided anchor; "
        "do not search the transliteration alone. "
        "Do not use a publisher/developer name alone when it is also a common word. If no clear work/entity "
        "can be identified, return an empty array.\n"
        "Title: %s\nBody excerpt:\n%s\nComments (unverified search clues only):\n%s\n"
        "Commenter-named terms parsed from those comments (preserve these in query 2): %s"
    ) % (str(title or "")[:180], body, comment_text,
         ", ".join(comment_named_terms[:3]) or "none")
    result = await _deep_once("", kind="video", title=title, prompt=prompt)
    queries = result.get("external_context_queries") if isinstance(result, dict) else []
    return _normalize_external_query_plan(queries, query_purposes)


def _normalize_external_query_plan(values: Any,
                                   purposes: Optional[Dict[str, str]] = None) -> List[str]:
    """Keep explicit search jobs separate from query text; accept legacy strings."""
    queries = []
    for item in (values if isinstance(values, list) else [])[:2]:
        text = item.get("query") if isinstance(item, dict) else item
        query = " ".join(str(text or "").split())[:220]
        if len(query) < 8 or query in queries:
            continue
        queries.append(query)
        purpose = item.get("purpose") if isinstance(item, dict) else "background"
        if purposes is not None:
            purposes[query] = purpose if purpose in ("background", "claim_verification") else "background"
    return queries


async def _fetch_external_reference(url: str) -> Dict[str, Any]:
    """Read a bounded public HTML page; never enter the paid video/ASR path."""
    import ipaddress
    import socket
    import httpx
    from urllib.parse import urljoin
    from engine.link_fetcher import strip_html

    target = str(url or "")
    original_host = (urlparse(target).hostname or "").lower().removeprefix("www.")
    try:
        async with httpx.AsyncClient(timeout=4, follow_redirects=False,
                                     trust_env=False) as client:
            for _ in range(4):
                parsed = urlparse(target)
                host = parsed.hostname or ""
                if (parsed.scheme not in {"http", "https"} or not host
                        or parsed.username or parsed.password
                        or parsed.port not in {None, 80, 443}
                        or host.lower().removeprefix("www.") != original_host):
                    return {}
                addresses = await asyncio.get_running_loop().getaddrinfo(
                    host, parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM)
                if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global
                                        for row in addresses):
                    return {}
                async with client.stream("GET", target, headers={
                        "User-Agent": "Mozilla/5.0 material-reference-reader"}) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            return {}
                        target = urljoin(target, location)
                        continue
                    if response.status_code != 200 or "html" not in response.headers.get(
                            "content-type", "").lower():
                        return {}
                    chunks = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 500000:
                            return {}
                        chunks.append(chunk)
                    title, text = strip_html(b"".join(chunks).decode(
                        response.encoding or "utf-8", errors="replace"))
                    if len(text.strip()) < 100:
                        return {}
                    return {"url": target, "title": title, "text": text[:24000]}
    except Exception:
        return {}
    return {}


def select_external_reference_passages(context: Dict[str, Any], query: str,
                                       limit: int = 2) -> List[Dict[str, Any]]:
    """Select attributed original passages, without an LLM or network request."""
    if not isinstance(context, dict) or context.get("status") != "ready":
        return []
    # Relational question words describe the requested operation, not the
    # subject. Scoring them as evidence anchors promotes page navigation.
    anchor_query = re.sub(r"\b(?:related|relationship|relationships|relation|relations)\b", " ", query, flags=re.I)
    candidates = []
    for source_index, item in enumerate((context.get("sources") or [])[:3]):
        if not isinstance(item, dict):
            continue
        if item.get("evidence_kind") == "research_abstract" and item.get("body_text"):
            abstract = " ".join(str(item["body_text"]).split())
            # Keep population, findings and closing limitations together. A
            # leading snippet alone often omits the study's actual result.
            quote = abstract if len(abstract) <= 2200 else abstract[:1200] + " […] " + abstract[-995:]
            alias_match = any(term in str(query or "") for term in re.findall(
                r"[\u4e00-\u9fff]{2,20}", str(item.get("retrieval_query") or "")))
            score = score_blocks([abstract + " " + str(item.get("title") or "")], anchor_query)[0]
            if alias_match or re.search(r"研究|证据|\b(?:research|studies|study|evidence|trial)\b", query, re.I):
                score += 1000
            candidates.append((score, source_index, 1, item, quote))
            continue
        snippet = " ".join(str(item.get("snippet") or "").split())[:420]
        passages = [snippet] if snippet else []
        for paragraph in re.split(r"\n+|(?<=[.!?。！？])\s+", str(item.get("body_text") or "")[:24000]):
            paragraph = " ".join(paragraph.split())
            # Dense pipe-separated UI menus are not article sentences.
            if paragraph.count("|") >= 4 and not re.search(r"[。！？.!?]$", paragraph):
                continue
            if len(paragraph) >= 12:
                passages.extend(paragraph[start:start + 520]
                                for start in range(0, len(paragraph), 520))
        scores = score_blocks(passages, anchor_query) if anchor_query.strip() else [0] * len(passages)
        for index in sorted(range(len(passages)), key=lambda i: (-scores[i], i))[:2]:
            candidates.append((scores[index], source_index, index, item, passages[index]))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    if candidates and candidates[0][0] > 0:
        minimum = candidates[0][0] * 0.5
        candidates = [row for row in candidates if row[0] >= minimum]
    selected = []
    seen = set()
    for score, _source, index, item, quote in candidates:
        if quote in seen:
            continue
        seen.add(quote)
        selected.append({"quote": quote, "score": score,
                         "evidence_kind": item.get("evidence_kind") or "web_excerpt",
                         "title": item.get("title") or "",
                         "host": item.get("host") or "",
                         "url": (item.get("body_url") or item.get("url") or "")
                                if index or not snippet else item.get("url") or ""})
        if len(selected) >= limit:
            break
    return selected


async def enrich_external_context(context: Dict[str, Any], *, page_fetcher: Any = None) -> Dict[str, Any]:
    """Upgrade valid cached results in place without repeating web search."""
    if not isinstance(context, dict) or context.get("status") != "ready":
        return context
    result = {**context, "schema_version": EXTERNAL_CONTEXT_SCHEMA_VERSION}
    result["sources"] = [dict(source) for source in context.get("sources", [])[:3]
                         if isinstance(source, dict)]
    if page_fetcher is None:
        page_fetcher = _fetch_external_reference

    async def read_source(source):
        if source.get("content_status") == "fetched" and source.get("body_text"):
            return
        source["content_status"] = "unavailable"
        try:
            page = await asyncio.wait_for(page_fetcher(source["url"]), timeout=6)
        except Exception:
            return
        if not isinstance(page, dict) or not str(page.get("text") or "").strip():
            return
        final_url = str(page.get("url") or source["url"])
        final_host = (urlparse(final_url).hostname or "").lower().removeprefix("www.")
        source_host = (urlparse(source["url"]).hostname or "").lower().removeprefix("www.")
        if final_host != source_host or urlparse(final_url).scheme not in {"http", "https"}:
            return
        source["body_text"] = str(page["text"])[:24000]
        source["body_url"] = final_url[:500]
        source["content_status"] = "fetched"

    await asyncio.gather(*(read_source(source) for source in result["sources"]))
    result["content_checked_at"] = datetime.now(timezone.utc).isoformat()
    return result


async def build_external_context(
    title: str, *, kind: str, source_coverage: Dict[str, Any] | None,
    analysis: Dict[str, Any], comments: List[str] | None = None,
    source_url: str = "", body_text: str = "", search_tool: Any = None,
    enabled: bool | None = None, page_fetcher: Any = None,
    research_tool: Any = None,
) -> Dict[str, Any]:
    """Fetch bounded, attributed background for a public video during import.

    Search results are candidate context only: they never alter the body
    transcript, source summary, or quote-validated narrative index. Returning an
    empty object is a normal failure/unsupported path and never blocks import.
    `search_tool` is injectable so tests can verify provenance without network.
    """
    if enabled is None:
        enabled = os.getenv(EXTERNAL_CONTEXT_ENABLED_ENV, "true").strip().lower() \
            not in ("0", "false", "no", "off")
    coverage = source_coverage if isinstance(source_coverage, dict) else {}
    if not enabled or str(kind or "").strip().lower() != "link":
        return {}
    if (coverage.get("kind") != "video"
            or coverage.get("body_available") is not True):
        return {}

    analysis = analysis if isinstance(analysis, dict) else {}
    query_purposes: Dict[str, str] = {}
    queries = _normalize_external_query_plan(
        analysis.get("external_context_queries"), query_purposes)
    if not queries and body_text:
        queries = await _draft_external_context_queries(
            title, body_text, list(comments or []), query_purposes)
    if not queries:
        fallback_query = _external_context_query(title, analysis, list(comments or []))
        queries = [fallback_query] if fallback_query else []
    if not queries:
        return {}
    queries = _supplement_comment_search_query(
        queries, title, analysis, list(comments or []),
        protected_queries=[query for query, purpose in query_purposes.items()
                           if purpose == "claim_verification"])
    # A model-proposed query is only a search hypothesis. Reject queries whose
    # named terms are absent from the imported title/body/comments; this prevents
    # background search from drifting to adjacent franchises or future entries.
    source_tokens = {
        token.casefold() for token in re.findall(
            r"[\w\u4e00-\u9fff]+",
            " ".join([title, body_text] + list(comments or [])))
        if len(token) >= 3
    }
    filtered_queries = []
    for candidate in queries:
        candidate_tokens = {
            token.casefold() for token in re.findall(r"[\w\u4e00-\u9fff]+", candidate)
            if len(token) >= 3 and token.casefold() not in {
                "official", "background", "setting", "characters", "story", "game",
                "about", "from", "with", "that", "this", "site", "com",
            }
        }
        source_text = " ".join([title, body_text] + list(comments or [])).casefold()
        chinese_anchors = re.findall(r"[\u4e00-\u9fff]{2,20}", candidate)
        if (candidate_tokens.intersection(source_tokens)
                or any(anchor.casefold() in source_text for anchor in chinese_anchors)):
            filtered_queries.append(candidate)
    queries = filtered_queries
    if not queries:
        fallback_query = _external_context_query(title, analysis, list(comments or []))
        queries = [fallback_query] if fallback_query else []
    if not queries:
        return {}
    # Injected search stubs stay offline unless a page reader is also injected.
    if page_fetcher is None and search_tool is None:
        page_fetcher = _fetch_external_reference
    if search_tool is None:
        from agent.web_search_tool import WebSearchTool
        if research_tool is None:
            research_tool = WebSearchTool(provider="crossref", timeout=5,
                                          max_results=8, cache_ttl=86400)
        search_tool = WebSearchTool(
            provider=os.getenv(
                "MATERIAL_EXTERNAL_CONTEXT_PROVIDER",
                os.getenv("CONVERSATION_DIRECTOR_WEB_SEARCH_PROVIDER", "bing_rss")),
            api_key=os.getenv(
                "MATERIAL_EXTERNAL_CONTEXT_API_KEY",
                os.getenv("CONVERSATION_DIRECTOR_WEB_SEARCH_API_KEY", "")),
            timeout=5, max_results=8, cache_ttl=86400,
        )
    async def search_query(query):
        jobs = [search_tool.search(query)]
        if research_tool is not None and query_purposes.get(query) == "claim_verification":
            jobs.append(research_tool.search(query))
        responses = await asyncio.gather(*jobs, return_exceptions=True)
        valid = [response for response in responses
                 if isinstance(response, dict) and response.get("status") == "ok"]
        return {"status": "ok" if valid else "error",
                "results": [item for response in valid for item in response.get("results", [])]}
    try:
        results = await asyncio.wait_for(asyncio.gather(*[
            search_query(query) for query in queries[:2]
        ]), timeout=7.0)
    except Exception:
        return {}

    source_host = (urlparse(str(source_url or "")).hostname or "").lower().removeprefix("www.")
    query_term_sets = [
        {
            token for token in re.findall(r"[\w\u4e00-\u9fff]+", query.casefold())
            if len(token) >= 2 and token not in _EXTERNAL_QUERY_STOP
        }
        for query in queries
    ]
    comment_terms = _external_context_comment_terms(list(comments or []))
    comment_query_indexes = {
        index for index, query in enumerate(queries)
        if any(term.casefold() in query.casefold() for term in comment_terms)
    }
    if not comment_query_indexes and len(query_term_sets) > 1:
        orientation_terms = query_term_sets[0]
        for index in range(1, len(query_term_sets)):
            query = queries[index]
            distinct_names = {
                token.casefold() for token in re.findall(
                    r"\b(?:[A-Z][a-z][A-Za-z0-9_-]*|[A-Z]{2,})\b", query
                )
                if token.casefold() not in _EXTERNAL_QUERY_STOP
            } - orientation_terms
            if distinct_names and query_term_sets[index].intersection(distinct_names):
                comment_query_indexes.add(index)
    orientation_terms = set().union(*(
        term_set for index, term_set in enumerate(query_term_sets)
        if index not in comment_query_indexes
    )) if len(comment_query_indexes) < len(query_term_sets) else set()
    comment_query_terms = {
        index: (query_term_sets[index] - orientation_terms)
        | {term.casefold() for term in comment_terms
           if term.casefold() in query_term_sets[index]}
        for index in comment_query_indexes
    }

    def _has_term(searchable: str, term: str) -> bool:
        if re.fullmatch(r"[a-z0-9_-]+", term, re.IGNORECASE):
            return bool(re.search(
                r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
                searchable, re.IGNORECASE,
            ))
        return term in searchable
    external_sources: List[Dict[str, str]] = []
    seen_urls = set()
    ranked = []
    for query_index, result in enumerate(results):
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        for item in (result.get("results") or []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower().removeprefix("www.")
            if parsed.scheme not in {"http", "https"} or not host:
                continue
            if source_host and (host == source_host or host.endswith("." + source_host)):
                continue
            if url in seen_urls:
                continue
            result_title = " ".join(str(item.get("title") or "").split())[:180]
            snippet = " ".join(str(item.get("snippet") or "").split())[:EXTERNAL_CONTEXT_SNIPPET_CHARS]
            # A query's site: domain is a routing hint, not evidence that the
            # page actually discusses the requested concept. Relevance must be
            # visible in the returned title/snippet, not satisfied by its host.
            searchable = (result_title + " " + snippet + " "
                          + str(item.get("body_text") or "")).casefold()
            # Require two anchors from the *same query*. The old any-token
            # check admitted skincare/dictionary/IT results merely because
            # they contained the ambiguous word "Remedy".
            query_matches = [
                {term for term in term_set if _has_term(searchable, term)}
                for term_set in query_term_sets
            ]
            matched = max(query_matches, key=len, default=set())
            if len(matched) < 2:
                continue
            # Search relevance first; first query breaks ties. This avoids
            # blindly accepting the engine's first result when it is off-topic.
            best_query_index = max(
                range(len(query_matches)),
                key=lambda index: len(query_matches[index]),
                default=query_index,
            )
            # Keep results answering the comment-named concept in view beside
            # general franchise orientation. Otherwise a broad page matching
            # many names can crowd out the very lore term the user asked about.
            answers_comment_term = any(
                index in comment_query_indexes
                and len(query_matches[index]) >= 2
                and any(_has_term(searchable, term)
                        for term in comment_query_terms.get(index, set()))
                for index in range(len(query_matches))
            )
            targeted = (answers_comment_term
                        or (item.get("evidence_kind") == "research_abstract"
                            and query_purposes.get(queries[query_index]) == "claim_verification"))
            ranked.append((int(targeted), len(matched),
                           -best_query_index, item, host, result_title, snippet, url))
            seen_urls.add(url)
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    for _targeted, _score, _query_index, item, host, result_title, snippet, url in ranked:
        external_sources.append({
            "title": result_title,
            "url": url[:500],
            "host": host[:120],
            "snippet": snippet,
            "retrieval_query": queries[_query_index],
            "search_purpose": query_purposes.get(queries[_query_index], "background"),
            **{key: item[key] for key in ("body_text", "content_status", "evidence_kind",
                                         "doi", "published", "journal") if key in item},
        })
        if len(external_sources) >= EXTERNAL_CONTEXT_MAX_SOURCES:
            break
    verification_requested = any(purpose == "claim_verification"
                                 for purpose in query_purposes.values())
    verification_status = (
        "candidates_retrieved" if any(source.get("search_purpose") == "claim_verification"
                                      for source in external_sources)
        else "no_relevant_results" if verification_requested else "not_requested")
    if not external_sources:
        return {
            "schema_version": EXTERNAL_CONTEXT_SCHEMA_VERSION,
            "query_policy": EXTERNAL_CONTEXT_QUERY_POLICY,
            "status": "no_relevant_results",
            "queries": queries[:2],
            "query_purposes": {query: query_purposes.get(query, "background") for query in queries[:2]},
            "verification_status": verification_status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "sources": [],
        }
    context = {
        "schema_version": EXTERNAL_CONTEXT_SCHEMA_VERSION,
            "query_policy": EXTERNAL_CONTEXT_QUERY_POLICY,
        "status": "ready",
        "query": queries[0],
        "queries": queries[:2],
            "query_purposes": {query: query_purposes.get(query, "background") for query in queries[:2]},
            "verification_status": verification_status,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": external_sources,
    }
    return (await enrich_external_context(context, page_fetcher=page_fetcher)
            if page_fetcher is not None else context)


# 确定性标题"像一句话"而不是标题时，值得换成深析顺带产出的标题。
# 实测：英文长文按"前 4 个词"取标题会得到 `The Bear is an`（原文第一句的前四个词），
# 磁带柜里的带名和素材盘开场句都被它污染。判据只认"含限定动词"，所以
# 中文内容标题（3–6 字、无英文动词）与短标题不会被误换。
_TITLE_FINITE_VERB_RE = re.compile(
    r"\b(is|are|was|were|am|be|been|has|have|had|does|do|did|will|would|can|could|"
    r"tells|follows|returns|argues|means|looks|comes|goes)\b", re.IGNORECASE)


def title_needs_model(title: str) -> bool:
    """这个确定性标题是不是"一句话的开头"（⇒ 该用深析产出的标题替换）。"""
    body = str(title or "").strip()
    if len(body.split()) < 3:
        return False
    return bool(_TITLE_FINITE_VERB_RE.search(body))



def needs_deep_analysis(raw_text: str) -> bool:
    return len(str(raw_text or "")) > DEEP_TRIGGER_CHARS


def _deep_prompt(kind: str, title: str, text: str, *, partial: bool = False) -> str:
    kind_label = {"diary": "一篇用户写的日记", "link": "一篇用户导入的网页文章"
                  }.get(str(kind or ""), "一份用户导入的素材")
    scope = ("**这是全文的其中一段**：只写这一段里出现的东西；"
             "某一类在这一段里没有就留空数组，不要为了凑满而推测其他部分。\n"
             if partial else "")
    return (
        "你是素材分析者。下面是%s%s。请**读完之后给出你的理解与观点**，"
        "而不是复述内容：找出素材里真正在讲什么、不同立场或争议点、"
        "以及可以引发讨论的角度。\n"
        "只输出一个 JSON 对象（不要代码块标记、不要解释），字段如下：\n"
        '{"title": "给这份素材起个短标题（≤12 字，像标题不像句子）", '
        '"topics": ["主题 3-5 条"], '
        '"voice_sources": [{"speaker_id":"speaker_0", '
        '"source_type":"creator_narration|in_material_dialogue|interview_or_guest|'
        'quoted_or_recorded_audio|other|uncertain", '
        '"confidence":"high|medium|low", "evidence":"原文短句"}], '
        '"voice_source_spans": [{"start_ms":0,"end_ms":1000,'
        '"source_type":"creator_narration|in_material_dialogue|interview_or_guest|'
        'quoted_or_recorded_audio|other|uncertain",'
        '"confidence":"high|medium|low","evidence":"原文短句"}], '
        '"people": ["人物/主体 + 一句关系"], '
        '"timeline": ["剧集/叙事类才填：本段关键节点"], '
        '"viewpoints": ["素材里的不同立场、观点或争议点"], '
        '"hooks": ["可以引发讨论的角度"], "gaps": ["素材没讲清楚的悬念/未决事项"], '
        '"external_context_queries": [{"query":"最多2条；保留原文概念锚点，研究核验同时加入英文科学术语及对象/效果", '
        '"purpose":"background|claim_verification"}], '
        '"source_map": {"entities": [{"name":"素材中的规范名", "aliases":[{"name":"原文别名", '
        '"evidence":"同时证明别名指向规范名的正文原句"}], '
        '"role":"其在本份素材中的作用", "evidence":"原文中的短句"}], '
        '"beats": [{"order":1, "event":"本段发生的关键变化", "effect":"它推动了什么或留下什么", '
        '"evidence":"原文中的短句"}], '
        '"open_threads": [{"question":"本段仍未回答的问题", "evidence":"原文中的短句"}]} '
        '}\n'
        "硬要求：\n"
        "① 事实必须来自素材——人物、集数、情节不许编；素材没写的宁可不写；\n"
        "② viewpoints/hooks/gaps 是你的**分析**，允许推断，但必须是“这可能/值得注意”"
        "这种口吻，不得把推断写成素材里的事实（素材带里模型会把它们当谈资引用）；\n"
        "③ 某一类在素材里确实没有就留空数组（散文没有 timeline 很正常），不要硬凑；\n"
        "④ 每条控制在 40 字内。\n"
        "⑤ 如果正文带有 [speaker=...; time_ms=...] 转写标记，区分两个不同概念："
        "voice_sources 是对 ASR speaker ID 的粗略汇总；voice_source_spans 才是本次来源归属的依据，"
        "按连续时间片段区分创作者旁白/实况、素材内对白、采访、引用/录音或其他来源。"
        "同一 speaker ID 可能被 ASR 错误合并，也可能在视频剪辑中对应不同来源；不能把整个 ID 固定成一种来源。"
        "source span 的 start_ms/end_ms 必须逐字对应输入中某些转写行的起止时间，evidence 必须是该时间范围内的原句；"
        "在来源变化处拆开，证据不清楚就标 uncertain，绝不按语言、speaker 编号或题材猜。"
        "voice_source_spans 每个窗口最多 %d 条；没有 speaker/time 标记或证据不足时返回空数组。\n"
        "⑥ source_map 是给后续对话检索用的证据索引，不是第二份自由摘要："
        "只覆盖正文，不得把评论区、标题或外部游戏知识混入正文人物/事件；"
        "entities 每个人只选一个全篇一致的规范名；中英文或译名变化时不要把同一人拆成多条，"
        "混合语言时优先使用正文中反复出现的原名，不要把译名另列为新人；"
        "但不确定是不是同一人时也不要强行合并。alias 只有在正文原句同时出现规范名和别名、且明确指向同一人时才写，"
        "每个 alias 都附带这条原句作 evidence。role 必须由对应 evidence 直接支持；引文只提到名字时，不要据此推演身份或经历。"
        "beats 按本素材全文出现顺序写 3–5 个重要转折；长素材分窗时只按各窗口原文先后归并，最终 order 不能重置或倒序。"
        "每项指出发生了什么以及它如何改变目标、认知或局势；event/effect 都不能超出该条 evidence 能支持的范围，"
        "不要把相邻剧情压成一条，也不要补全片段外剧情。"
        "open_threads 只列正文明确留下、且证据句能看出的未决问题。每条 evidence 必须是正文逐字短引文，"
        "不能用评论引文；没有可核对引文就省略该项。不要输出自己估算的时间，时间锚由系统从转写行提取。\n"
        "⑦ external_context_queries 可补充默认读者不熟悉的背景，也可核验正文值得讨论但缺少依据的观点；后者明确对象、时间和待验证问题，优先原始研究或可比统计，不能预设观点正确。虚构剧情不拿现实数据当剧情证据；"
        "从正文识别作品/实体，评论只能提示可能缺少哪段背景。优先官方/创作者一手页面，"
        "不要把搜索词写成事实，不要查询与本段无关的剧情，不确定作品时返回空数组。\n"
        "%s"
        "\n素材（标题：%s）：\n%s"
    ) % (kind_label, ("《%s》" % title) if title else "",
         _VOICE_SOURCE_SPANS_PER_WINDOW, scope,
         title or "（无）", text[:DEEP_INPUT_CHARS])


async def _deep_once(text: str, *, kind: str, title: str,
                     partial: bool = False, prompt: str = "",
                     source_evidence_text: str = "",
                     known_voice_source_spans: List[Dict[str, Any]] = None) -> Dict:
    """一次深析调用。失败返回 {}，绝不抛——一个窗口失败不该毁掉整份分析。"""
    try:
        from llm_client import (DEFAULT_CHAT_MODEL, DEEPSEEK_BASE_URL,
                                robust_llm_client, thinking_disabled_extra_body)
        model = (os.getenv(DEEP_MODEL_ENV, "").strip()
                 or os.getenv(ANALYZER_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL)
        thinking_enabled = os.getenv(DEEP_THINKING_ENV, "true").strip().lower() \
            not in ("0", "false", "no", "off")
        route = str(DEEPSEEK_BASE_URL or "").lower()
        reasoning_effort = ""
        if (thinking_enabled and str(model).lower().startswith("deepseek")
                and ("api.deepseek.com" in route or "opencode.ai" in route)):
            extra_body = {"thinking": {"type": "enabled"}}
            reasoning_effort = os.getenv(DEEP_THINKING_EFFORT_ENV, "low").strip().lower()
            if reasoning_effort not in ("low", "high", "max"):
                reasoning_effort = "low"
            try:
                max_tokens = max(1200, min(
                    16000, int(os.getenv(DEEP_THINKING_MAX_TOKENS_ENV,
                                         str(DEEP_THINKING_MAX_TOKENS)))))
            except (TypeError, ValueError):
                max_tokens = DEEP_THINKING_MAX_TOKENS
        else:
            extra_body = thinking_disabled_extra_body(model)
            max_tokens = 900
        request = {}
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
        raw = await robust_llm_client.chat_completion(
            [{"role": "user", "content": (prompt or _deep_prompt(
                kind, title, text, partial=partial))}],
            model=model, temperature=0.4, max_tokens=max_tokens, timeout=60,
            extra_body=extra_body,
            max_retries=2, gate="background", **request,
        )
        return _parse_deep_json(
            raw, source_text=text,
            source_map_source_text=source_evidence_text or text,
            known_voice_source_spans=known_voice_source_spans or [])
    except Exception:
        return {}


def format_transcript_with_speakers(text: str, segments: List[Dict[str, Any]]) -> str:
    """Render ASR segments as evidence-tagged lines for material analysis.

    The stored transcript remains clean text. This analysis-only view carries
    speaker IDs and timestamps when the ASR provider supplies them.
    """
    rows = []
    for segment in list(segments or []):
        if not isinstance(segment, dict):
            continue
        phrase = " ".join(str(segment.get("text") or "").split())
        if not phrase:
            continue
        speaker_value = segment.get("speaker_id")
        speaker = (str(speaker_value).strip()
                   if speaker_value is not None and str(speaker_value).strip()
                   else "unknown")
        start, end = segment.get("start_ms"), segment.get("end_ms")
        try:
            timestamp = "%s-%s" % (int(start), int(end)) \
                if start is not None and end is not None else "?"
        except (TypeError, ValueError):
            timestamp = "?"
        rows.append("[speaker=%s; time_ms=%s] %s" % (speaker, timestamp, phrase))
    if not rows:
        return str(text or "")

    segment_text = " ".join(str(s.get("text") or "") for s in segments
                             if isinstance(s, dict))
    segment_chars = len("".join(segment_text.split()))
    source_chars = len("".join(str(text or "").split()))
    # If diarization output omits a substantial part of the transcript, do not
    # let a partial structured view masquerade as the entire source.
    if source_chars and segment_chars < source_chars * 0.75:
        return str(text or "")
    return "\n".join(rows)


def material_analysis_body(text: str) -> str:
    """Strip the library's appended viewer-comment section for body-only analysis."""
    return str(text or "").partition(COMMENT_MARKER)[0].strip()


def _windows(text: str, size: int) -> List[str]:
    """把长文切成 ≤size 的窗口，切点尽量落在段落/句子边界上。

    硬切会把"人物 A 做了"和"什么"分到两个窗口，两边都读不出完整事实，
    所以先按段落聚合；单段超长时再按句末标点切；仍超长才硬切。
    """
    body = str(text or "")
    if len(body) <= size:
        return [body]
    windows: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            windows.append(current)
            current = ""

    for para in re.split(r"\n+", body):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            flush()
            # 单段超长：按句末标点攒，攒不下再硬切
            sentence = ""
            for piece in re.split(r"(?<=[.!?。！？；;])\s*", para):
                if not piece:
                    continue
                if len(piece) > size:
                    if sentence:
                        windows.append(sentence)
                        sentence = ""
                    for start in range(0, len(piece), size):
                        windows.append(piece[start:start + size])
                    continue
                if len(sentence) + len(piece) > size:
                    windows.append(sentence)
                    sentence = piece
                else:
                    sentence += piece
            if sentence:
                windows.append(sentence)
            continue
        if current and len(current) + len(para) + 1 > size:
            flush()
        current = (current + "\n" + para) if current else para
    flush()
    return windows or [body]


_VOICE_SOURCE_LABELS = {
    "creator_narration": "创作者旁白/解说",
    "in_material_dialogue": "素材内角色对白",
    "interview_or_guest": "采访者/受访者发言",
    "quoted_or_recorded_audio": "引用或录音内容",
    "other": "其他声音来源",
    "uncertain": "来源未能确认",
}
_VOICE_SOURCE_ALIASES = {
    "creator_narration": "creator_narration", "creator narration": "creator_narration",
    "narration": "creator_narration", "commentary": "creator_narration",
    "up主旁白": "creator_narration", "up主解说": "creator_narration",
    "创作者旁白": "creator_narration", "创作者解说": "creator_narration",
    "玩家解说": "creator_narration", "旁白": "creator_narration",
    "in_material_dialogue": "in_material_dialogue", "in-game dialogue": "in_material_dialogue",
    "game dialogue": "in_material_dialogue", "character dialogue": "in_material_dialogue",
    "角色对白": "in_material_dialogue", "游戏对白": "in_material_dialogue",
    "素材内角色对白": "in_material_dialogue", "游戏内对白": "in_material_dialogue",
    "interview_or_guest": "interview_or_guest", "interview": "interview_or_guest",
    "guest": "interview_or_guest", "采访": "interview_or_guest", "受访者": "interview_or_guest",
    "quoted_or_recorded_audio": "quoted_or_recorded_audio", "recording": "quoted_or_recorded_audio",
    "quoted audio": "quoted_or_recorded_audio", "录音": "quoted_or_recorded_audio",
    "引用": "quoted_or_recorded_audio", "other": "other", "其他": "other",
    "uncertain": "uncertain", "unknown": "uncertain", "unclear": "uncertain",
    "未确认": "uncertain", "未能确认": "uncertain", "不确定": "uncertain",
}
_VOICE_CONFIDENCE_ALIASES = {
    "high": "high", "高": "high", "medium": "medium", "中": "medium",
    "low": "low", "低": "low",
}
_VOICE_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _voice_speaker_id(value: Any) -> str:
    raw = " ".join(str(value or "").lower().split())
    if raw.isdigit():
        return "speaker_%s" % raw
    match = re.search(r"speaker(?:[_ -]?id)?\s*[=_:# -]?\s*(?:speaker[_ -]?)?(\d+)\b", raw)
    if not match:
        match = re.search(r"\bspeaker[_ -]?(\d+)\b", raw)
    return "speaker_%s" % match.group(1) if match else ""


def _voice_source_type(value: Any) -> str:
    raw = " ".join(str(value or "").lower().split())
    if raw in _VOICE_SOURCE_ALIASES:
        return _VOICE_SOURCE_ALIASES[raw]
    # Accept short natural-language labels from legacy analyses while keeping
    # the persisted vocabulary bounded and consistent.
    if any(token in raw for token in ("up主", "创作者", "旁白", "commentary", "narrat")):
        return "creator_narration"
    if any(token in raw for token in ("游戏内", "游戏对白", "角色对白", "角色发言", "in-game", "character dialogue")):
        return "in_material_dialogue"
    if any(token in raw for token in ("采访", "受访者", "采访者", "interview", "guest")):
        return "interview_or_guest"
    if any(token in raw for token in ("录音", "引用", "recording", "quoted")):
        return "quoted_or_recorded_audio"
    if any(token in raw for token in ("未确认", "不确定", "unknown", "unclear", "uncertain")):
        return "uncertain"
    if raw:
        return "other"
    return "uncertain"


def _parse_voice_source(item: Any) -> Dict[str, Any]:
    """Normalize current structured output and older string records alike."""
    if isinstance(item, dict):
        speaker_id = _voice_speaker_id(item.get("speaker_id") or item.get("speaker"))
        source_type = _voice_source_type(item.get("source_type") or item.get("source"))
        confidence_raw = " ".join(str(item.get("confidence") or "").lower().split())
        confidence = _VOICE_CONFIDENCE_ALIASES.get(confidence_raw, "low")
        alternatives = [normalized for normalized in (
            _voice_source_type(value) for value in (item.get("alternatives") or []))
            if normalized != "uncertain"]
        evidence = item.get("evidence") or item.get("quote") or ""
        if isinstance(evidence, list):
            evidence_rows = evidence
        else:
            evidence_rows = [evidence]
    else:
        raw = " ".join(str(item or "").split())
        speaker_id = _voice_speaker_id(raw)
        source_type = _voice_source_type(raw)
        confidence = "low"
        for token, normalized in (("high", "high"), ("medium", "medium"), ("low", "low"),
                                  ("高", "high"), ("中", "medium"), ("低", "low")):
            if re.search(r"(?:confidence|置信度)\s*[:=：]?\s*%s" % re.escape(token), raw, re.I):
                confidence = normalized
                break
        evidence_match = re.search(r"(?:evidence|证据)\s*[:=：]\s*(.*)$", raw, re.I)
        evidence_rows = [evidence_match.group(1)] if evidence_match else []
    evidence = []
    for value in evidence_rows:
        quote = " ".join(str(value or "").split()).strip(" \"'“”‘’")
        if quote and quote not in evidence:
            evidence.append(quote[:160])
        if len(evidence) >= 2:
            break
    if not speaker_id:
        return {}
    result = {"speaker_id": speaker_id, "source_type": source_type,
              "confidence": confidence, "evidence": evidence}
    if isinstance(item, dict) and source_type == "uncertain" and alternatives:
        result["alternatives"] = list(dict.fromkeys(alternatives))
    return result


def format_voice_source(item: Any) -> str:
    """Render a source assessment for prompt context without treating it as fact."""
    parsed = _parse_voice_source(item)
    if not parsed:
        return " ".join(str(item or "").split())[:220]
    label = _VOICE_SOURCE_LABELS.get(parsed["source_type"], "来源未能确认")
    confidence = {"high": "高", "medium": "中", "low": "低"}.get(parsed["confidence"], "低")
    if parsed.get("alternatives"):
        labels = [_VOICE_SOURCE_LABELS.get(value, "来源未能确认")
                  for value in parsed["alternatives"]]
        label = "判断不一致（%s）" % " / ".join(labels)
    evidence = "；证据：%s" % " / ".join(parsed.get("evidence") or []) \
        if parsed.get("evidence") else ""
    return "%s：%s，置信度%s%s" % (parsed["speaker_id"], label, confidence, evidence)


def _timestamped_transcript_rows(text: str) -> List[Dict[str, Any]]:
    rows = []
    for line in str(text or "").splitlines():
        match = re.match(
            r"\[speaker=([^;]+);\s*time_ms=(\d+)-(\d+)\]\s*(.*)$", line.strip())
        if not match:
            continue
        phrase = " ".join(match.group(4).split())
        if not phrase:
            continue
        rows.append({"speaker_id": _voice_speaker_id(match.group(1)),
                     "start_ms": int(match.group(2)), "end_ms": int(match.group(3)),
                     "text": phrase})
    return rows


def _parse_voice_source_span(item: Any, transcript_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep a source label only when its time range and quote match ASR evidence."""
    if not isinstance(item, dict) or not transcript_rows:
        return {}
    try:
        start_ms = int(item.get("start_ms"))
        end_ms = int(item.get("end_ms"))
    except (TypeError, ValueError):
        return {}
    if start_ms < 0 or end_ms <= start_ms:
        return {}
    source_type = _voice_source_type(item.get("source_type") or item.get("source"))
    confidence_raw = " ".join(str(item.get("confidence") or "").lower().split())
    confidence = _VOICE_CONFIDENCE_ALIASES.get(confidence_raw, "low")
    evidence = " ".join(str(item.get("evidence") or item.get("quote") or "").split())
    if not evidence:
        return {}
    first = next((row for row in transcript_rows if row["start_ms"] == start_ms), None)
    last = next((row for row in transcript_rows if row["end_ms"] == end_ms), None)
    if first is None or last is None:
        return {}
    first_index = transcript_rows.index(first)
    last_index = transcript_rows.index(last)
    if first_index > last_index:
        return {}
    covered = transcript_rows[first_index:last_index + 1]
    normalized_evidence = " ".join(evidence.split()).casefold()
    if not any(normalized_evidence in " ".join(row["text"].split()).casefold()
               for row in covered):
        return {}
    result = {"start_ms": start_ms, "end_ms": end_ms,
              "source_type": source_type, "confidence": confidence,
              "evidence": evidence[:160]}
    return result


def format_voice_source_span(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    try:
        start_ms = int(item.get("start_ms"))
        end_ms = int(item.get("end_ms"))
    except (TypeError, ValueError):
        return ""
    evidence = " ".join(str(item.get("evidence") or "").split())
    if start_ms < 0 or end_ms <= start_ms or not evidence:
        return ""
    source_type = _voice_source_type(item.get("source_type"))
    confidence_raw = " ".join(str(item.get("confidence") or "").lower().split())
    confidence_value = _VOICE_CONFIDENCE_ALIASES.get(confidence_raw, "low")
    label = _VOICE_SOURCE_LABELS.get(source_type, "来源未能确认")
    alternatives = item.get("alternatives") or []
    if alternatives:
        labels = [_VOICE_SOURCE_LABELS.get(_voice_source_type(value), "来源未能确认")
                  for value in alternatives]
        label = "判断不一致（%s）" % " / ".join(labels)
    confidence = {"high": "高", "medium": "中", "low": "低"}.get(confidence_value, "低")
    return "[%d-%dms] %s，置信度%s；证据：%s" % (
        start_ms, end_ms, label, confidence, evidence[:160])


def _merge_voice_source_spans(parts: List[Dict]) -> List[Dict[str, Any]]:
    """Merge window-local source ranges by their evidence-backed time interval."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    order: List[tuple] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        for row in (part.get("voice_source_spans") or []):
            if not isinstance(row, dict):
                continue
            try:
                key = (int(row.get("start_ms")), int(row.get("end_ms")))
            except (TypeError, ValueError):
                continue
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(row)
    merged = []
    for key in sorted(order):
        rows = grouped[key]
        types = list(dict.fromkeys(_voice_source_type(r.get("source_type")) for r in rows))
        evidence = list(dict.fromkeys(" ".join(str(r.get("evidence") or "").split())
                                      for r in rows if str(r.get("evidence") or "").strip()))
        confidence = min((_VOICE_CONFIDENCE_ALIASES.get(
            " ".join(str(r.get("confidence") or "").lower().split()), "low") for r in rows),
            key=lambda value: _VOICE_CONFIDENCE_RANK.get(value, 0))
        source_type = types[0] if len(types) == 1 else "uncertain"
        row = {"start_ms": key[0], "end_ms": key[1],
               "source_type": source_type,
               "confidence": confidence if len(types) == 1 else "low",
               "evidence": evidence[0] if evidence else ""}
        if len(types) > 1:
            row["alternatives"] = types
        merged.append(row)
    return merged[:_VOICE_SOURCE_SPANS_MAX]


def _merge_voice_sources(parts: List[Dict]) -> List[Dict[str, Any]]:
    """Merge per-window speaker hypotheses conservatively by ASR speaker ID."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        rows = part.get("voice_sources") or []
        if not isinstance(rows, list):
            rows = [rows]
        for row in rows:
            parsed = _parse_voice_source(row)
            speaker_id = parsed.get("speaker_id")
            if not speaker_id:
                continue
            if speaker_id not in grouped:
                grouped[speaker_id] = []
                order.append(speaker_id)
            grouped[speaker_id].append(parsed)

    merged = []
    for speaker_id in order:
        rows = grouped[speaker_id]
        classified = [row["source_type"] for row in rows
                      if row["source_type"] != "uncertain"]
        distinct = list(dict.fromkeys(classified))
        if len(distinct) == 1 and not any(row["source_type"] == "uncertain" for row in rows):
            source_type = distinct[0]
            alternatives = []
        elif len(distinct) > 1:
            source_type = "uncertain"
            alternatives = distinct
        elif len(distinct) == 1:
            source_type = "uncertain"
            alternatives = distinct
        else:
            source_type = "uncertain"
            alternatives = []
        confidence = min((row["confidence"] for row in rows),
                         key=lambda item: _VOICE_CONFIDENCE_RANK.get(item, 0))
        if alternatives:
            confidence = "low"
        evidence = []
        for row in rows:
            for quote in row.get("evidence") or []:
                if quote not in evidence:
                    evidence.append(quote)
                if len(evidence) >= 2:
                    break
            if len(evidence) >= 2:
                break
        merged_row = {"speaker_id": speaker_id, "source_type": source_type,
                      "confidence": confidence, "evidence": evidence}
        if alternatives:
            merged_row["alternatives"] = alternatives
        merged.append(merged_row)
    return merged[:_DEEP_LIST_MAX]


def _normalized_match_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _normalized_source_offset(quote: Any, source_text: str) -> int:
    quote_key = _normalized_match_text(quote)
    body_key = _normalized_match_text(material_analysis_body(source_text))
    return body_key.find(quote_key) if quote_key else -1


def _source_map_evidence(quote: Any, source_text: str,
                         transcript_rows: List[Dict[str, Any]],
                         voice_source_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep a source-map item only when its short quote is present in body text.

    For timestamped transcripts, the quote must fit inside one saved ASR row;
    its location is derived from that row, never from model-generated times.
    """
    evidence = " ".join(str(quote or "").split()).strip()
    normalized = _normalized_match_text(evidence)
    if len(normalized) < 8 or len(evidence) > 220:
        return {}
    if transcript_rows:
        row = next((item for item in transcript_rows
                    if normalized in _normalized_match_text(item.get("text"))), None)
        if not row:
            return {}
        start_ms, end_ms = int(row["start_ms"]), int(row["end_ms"])
        source_offset = _normalized_source_offset(evidence, source_text)
        matched_types = list(dict.fromkeys(
            str(span.get("source_type") or "uncertain")
            for span in voice_source_spans
            if isinstance(span, dict)
            and int(span.get("start_ms") or -1) <= start_ms
            and int(span.get("end_ms") or -1) >= end_ms
        ))
        source_type = matched_types[0] if len(matched_types) == 1 else "uncertain"
        proof = {"quote": evidence, "start_ms": start_ms, "end_ms": end_ms,
                 "speaker_id": str(row.get("speaker_id") or ""),
                 "source_type": source_type}
        if source_offset >= 0:
            proof["source_offset"] = source_offset
        return proof
    body_key = _normalized_match_text(material_analysis_body(source_text))
    source_offset = body_key.find(normalized)
    if source_offset < 0:
        return {}
    for span in voice_source_spans:
        if not isinstance(span, dict):
            continue
        span_evidence = _normalized_match_text(span.get("evidence"))
        if not span_evidence or normalized not in span_evidence:
            continue
        try:
            start_ms = int(span.get("start_ms"))
            end_ms = int(span.get("end_ms"))
        except (TypeError, ValueError):
            continue
        if start_ms < 0 or end_ms <= start_ms:
            continue
        source_type = str(span.get("source_type") or "uncertain")
        if source_type not in {
            "creator_narration", "in_material_dialogue", "interview_or_guest",
            "quoted_or_recorded_audio", "other", "uncertain",
        }:
            source_type = "uncertain"
        return {"quote": evidence, "start_ms": start_ms, "end_ms": end_ms,
                "source_type": source_type, "source_offset": source_offset}
    return {"quote": evidence, "source_type": "body",
            "source_offset": source_offset}


def _parse_source_map(value: Any, *, source_text: str,
                      voice_source_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and bound the evidence-linked people/story index from DeepThink."""
    if not isinstance(value, dict):
        return {}
    rows = _timestamped_transcript_rows(source_text)
    parsed: Dict[str, Any] = {"validated": True}

    entities = []
    seen_entities = set()
    for item in value.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name") or "").split())[:60]
        role = " ".join(str(item.get("role") or "").split())[:140]
        proof = _source_map_evidence(
            item.get("evidence"), source_text, rows, voice_source_spans)
        name_key = _normalized_match_text(name)
        proof_quote = _normalized_match_text(proof.get("quote"))
        if (not name or not role or not proof or not name_key
                or name_key not in proof_quote):
            continue
        if rows and not any(name_key in _normalized_match_text(row["text"])
                            for row in rows):
            # Some aliases may be spoken only in another language/source range;
            # the canonical name itself must still occur in the body evidence.
            if name_key not in _normalized_match_text(source_text):
                continue
        if name_key in seen_entities:
            continue
        seen_entities.add(name_key)
        aliases = []
        for alias in item.get("aliases") or []:
            if isinstance(alias, dict):
                alias_text = " ".join(str(alias.get("name") or "").split())[:60]
                alias_proof = _source_map_evidence(
                    alias.get("evidence"), source_text, rows, voice_source_spans)
                alias_key = _normalized_match_text(alias_text)
                if (alias_text and alias_proof and alias_key != name_key
                        and alias_key in _normalized_match_text(alias_proof.get("quote"))
                        and name_key in _normalized_match_text(alias_proof.get("quote"))):
                    aliases.append({"name": alias_text, "evidence": alias_proof})
            else:
                # Backward-compatible input is accepted only when the entity's
                # own proof explicitly contains both names.
                alias_text = " ".join(str(alias or "").split())[:60]
                alias_key = _normalized_match_text(alias_text)
                if (alias_text and proof and alias_key != name_key
                        and alias_key in proof_quote and name_key in proof_quote):
                    aliases.append({"name": alias_text, "evidence": proof})
            if len(aliases) >= 4:
                break
        entities.append({"name": name, "aliases": aliases, "role": role,
                         "evidence": proof})
        if len(entities) >= 10:
            break
    if entities:
        parsed["entities"] = entities

    beats = []
    seen_beats = set()
    for item in value.get("beats") or []:
        if not isinstance(item, dict):
            continue
        event = " ".join(str(item.get("event") or "").split())[:180]
        effect = " ".join(str(item.get("effect") or "").split())[:180]
        proof = _source_map_evidence(
            item.get("evidence"), source_text, rows, voice_source_spans)
        key = _normalized_match_text(proof.get("quote") or event)
        if not event or not proof or not key or key in seen_beats:
            continue
        seen_beats.add(key)
        try:
            order = max(1, min(99, int(item.get("order") or len(beats) + 1)))
        except (TypeError, ValueError):
            order = len(beats) + 1
        beats.append({"order": order, "event": event, "effect": effect,
                      "evidence": proof})
        if len(beats) >= 10:
            break
    if beats and all("source_offset" in item["evidence"]
                     for item in beats):
        beats.sort(key=lambda item: item["evidence"]["source_offset"])
    elif beats and all("start_ms" in item["evidence"] for item in beats):
        beats.sort(key=lambda item: (
            item["evidence"].get("start_ms", 10**15), item["order"]))
    else:
        beats.sort(key=lambda item: item["order"])
    for index, item in enumerate(beats, 1):
        item["order"] = index
    if beats:
        parsed["beats"] = beats

    threads = []
    seen_threads = set()
    for item in value.get("open_threads") or []:
        if not isinstance(item, dict):
            continue
        question = " ".join(str(item.get("question") or "").split())[:160]
        proof = _source_map_evidence(
            item.get("evidence"), source_text, rows, voice_source_spans)
        key = _normalized_match_text(question)
        if not question or not proof or key in seen_threads:
            continue
        seen_threads.add(key)
        threads.append({"question": question, "evidence": proof})
        if len(threads) >= 8:
            break
    if threads:
        parsed["open_threads"] = threads
    return parsed if len(parsed) > 1 else {}


def _merge_source_maps(parts: List[Dict]) -> Dict[str, Any]:
    """Combine window maps in source order without flattening evidence anchors."""
    entities: Dict[str, Dict[str, Any]] = {}
    beats: List[Dict[str, Any]] = []
    threads: Dict[str, Dict[str, Any]] = {}
    for part_index, part in enumerate(parts):
        source_map = part.get("source_map") if isinstance(part, dict) else None
        if not isinstance(source_map, dict) or source_map.get("validated") is not True:
            continue
        for item in source_map.get("entities") or []:
            if not isinstance(item, dict):
                continue
            key = _normalized_match_text(item.get("name"))
            if not key:
                continue
            current = entities.get(key)
            if current is None:
                current = dict(item)
                current["aliases"] = list(item.get("aliases") or [])
                entities[key] = current
            else:
                for alias in item.get("aliases") or []:
                    if alias not in current["aliases"]:
                        current["aliases"].append(alias)
                if not current.get("evidence") and item.get("evidence"):
                    current["evidence"] = item["evidence"]
        for item in source_map.get("beats") or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["_window_order"] = part_index
            if not any(_normalized_match_text(
                           (old.get("evidence") or {}).get("quote")
                           if isinstance(old.get("evidence"), dict) else "")
                       == _normalized_match_text(
                           (entry.get("evidence") or {}).get("quote")
                           if isinstance(entry.get("evidence"), dict) else "")
                       for old in beats):
                beats.append(entry)
        for item in source_map.get("open_threads") or []:
            if isinstance(item, dict):
                key = _normalized_match_text(item.get("question"))
                if key and key not in threads:
                    threads[key] = dict(item)
    if not entities and not beats and not threads:
        return {}
    if beats and all(isinstance(item.get("evidence"), dict)
                     and "source_offset" in item["evidence"] for item in beats):
        beats.sort(key=lambda item: item["evidence"]["source_offset"])
    elif beats and all(isinstance(item.get("evidence"), dict)
                       and "start_ms" in item["evidence"] for item in beats):
        beats.sort(key=lambda item: (
            item["evidence"].get("start_ms", 10**15),
            item.get("_window_order", 0), item.get("order", 0)))
    else:
        # Each analysis window is a contiguous slice of source text. Preserve
        # window order first; untimed beats must not be pushed after later events.
        beats.sort(key=lambda item: (
            item.get("_window_order", 0), item.get("order", 0)))
    for item in beats:
        item.pop("_window_order", None)
    for index, item in enumerate(beats, 1):
        item["order"] = index
    result: Dict[str, Any] = {"validated": True}
    if entities:
        result["entities"] = list(entities.values())[:10]
    if beats:
        result["beats"] = beats[:10]
    if threads:
        result["open_threads"] = list(threads.values())[:8]
    return result


def _voice_source_evidence(text: str, *, samples_per_speaker: int = 6) -> str:
    """Build a bounded, source-wide sample so window-specific guesses can be checked."""
    grouped: Dict[str, List[tuple]] = {}
    for line in str(text or "").splitlines():
        match = re.match(
            r"\[speaker=([^;]+);\s*time_ms=([^\]]+)\]\s*(.*)$", line.strip())
        if not match:
            continue
        speaker_id = _voice_speaker_id(match.group(1))
        phrase = " ".join(match.group(3).split())
        if not speaker_id or not phrase:
            continue
        grouped.setdefault(speaker_id, []).append((match.group(2).strip(), phrase))

    rendered = []
    for speaker_id, rows in grouped.items():
        count = min(len(rows), max(1, int(samples_per_speaker)))
        indices = sorted({round(index * (len(rows) - 1) / max(count - 1, 1))
                          for index in range(count)})
        examples = ["[time_ms=%s] %s" % (rows[index][0], rows[index][1][:180])
                    for index in indices]
        rendered.append("%s (%d segments):\n- %s" % (
            speaker_id, len(rows), "\n- ".join(examples)))
    return "\n\n".join(rendered)[:DEEP_INPUT_CHARS]


async def _reconcile_voice_sources(text: str, *, title: str = "") -> List[Dict[str, Any]]:
    """Resolve cross-window disagreements using evenly sampled source-wide evidence.

    This is import-time only and called only when independent long-text windows
    disagree about the same ASR speaker. It does not run in the conversation loop.
    """
    evidence = _voice_source_evidence(material_analysis_body(text))
    if not evidence:
        return []
    prompt = (
        "你在核对一份长素材里按说话人分段的转写。下面每组是同一 ASR speaker ID "
        "在全文不同位置的均匀抽样；总段数仅表示转写片段数量，不代表说话人身份。\n"
        "请综合一个 speaker 的全部例句，判断它更可能属于创作者/播主旁白、素材内人物对白、"
        "采访发言、引用/录音、其他来源，还是证据不足。不同声音可能混在同一素材里；"
        "ASR 也可能把声音分错或合并。同一个 speaker ID 可能包含不同来源的片段；"
        "如果不同例句能分别支持不同来源，就分别输出多个 source_type，不能硬合成一种；"
        "若某些例句无法可靠归属，再单独标 uncertain。不要按语言、speaker 编号或题材单独猜身份。\n"
        "只输出 JSON：{\"voice_sources\":[{\"speaker_id\":\"speaker_0\","
        "\"source_type\":\"creator_narration|in_material_dialogue|interview_or_guest|"
        "quoted_or_recorded_audio|other|uncertain\",\"confidence\":\"high|medium|low\","
        "\"evidence\":\"一到两段输入原句\",\"alternatives\":[\"仅 uncertain 时填写\"]}]}。"
        "每个明确的 speaker_id/source_type 组合输出一条；同一 speaker 可以有多个来源条目。"
        "speaker_id 必须来自输入；evidence 用一到两条输入原句，支撑该条分类。\n"
        "素材标题：《%s》\n各 speaker 全文抽样：\n%s"
    ) % (title or "（无）", evidence)
    result = await _deep_once("", kind="video", title=title, prompt=prompt)
    return list(result.get("voice_sources") or [])


def _merge_deep(parts: List[Dict]) -> Dict:
    """多窗口深析 → 一份产物（**轮转取件**，不是从头截断）。

    轮转（各窗口的第 1 条 → 各窗口的第 2 条 → …）是刻意的：从头截断会让产物
    全部来自素材开头——那正是"只读前 12,000 字"这个毛病换个地方长出来。
    轮转保证每个窗口都进得来，产出的观点覆盖全文。
    """
    merged: Dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict) and str(part.get("title") or "").strip():
            merged["title"] = str(part["title"]).strip()
            break
    for field in DEEP_FIELDS:
        if field == "source_map":
            merged_map = _merge_source_maps(parts)
            if merged_map:
                merged[field] = merged_map
            continue
        if field == "voice_sources":
            merged_voice_sources = _merge_voice_sources(parts)
            if merged_voice_sources:
                merged[field] = merged_voice_sources
            continue
        if field == "voice_source_spans":
            merged_spans = _merge_voice_source_spans(parts)
            if merged_spans:
                merged[field] = merged_spans
            continue
        if field == "external_context_queries":
            plans = []
            seen_queries = set()
            for part in parts:
                purposes = {}
                for query in _normalize_external_query_plan(part.get(field), purposes):
                    if query not in seen_queries:
                        seen_queries.add(query)
                        plans.append({"query": query, "purpose": purposes[query]})
            if plans:
                plans.sort(key=lambda plan: plan["purpose"] != "claim_verification")
                merged[field] = plans[:2]
            continue
        buckets = [list(part.get(field) or []) for part in parts
                   if isinstance(part, dict) and part.get(field)]
        if not buckets:
            continue
        items: List[str] = []
        seen: set = set()
        for rank in range(max(len(bucket) for bucket in buckets)):
            for bucket in buckets:
                if rank >= len(bucket):
                    continue
                entry = str(bucket[rank])
                if not entry or entry in seen:
                    continue
                seen.add(entry)
                items.append(entry)
                item_limit = 2 if field == "external_context_queries" else _DEEP_LIST_MAX
                if len(items) >= item_limit:
                    break
            item_limit = 2 if field == "external_context_queries" else _DEEP_LIST_MAX
            if len(items) >= item_limit:
                break
        if items:
            merged[field] = items
    return merged


def _parse_deep_json(raw: str, *, source_text: str = "",
                     source_map_source_text: str = "",
                     known_voice_source_spans: List[Dict[str, Any]] = None) -> Dict:
    """把模型输出收成六字段 dict。解析失败 → {}（调用方按"没有分析"处理）。"""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        import json
        payload = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Any] = {}
    # title 是标量不是列表：单独取出来（深析顺带给的短标题，见 title_needs_model）
    model_title = " ".join(str(payload.get("title") or "").split())[:24]
    if model_title:
        out["title"] = model_title
    timestamped_rows = _timestamped_transcript_rows(source_text)
    for field in DEEP_FIELDS:
        value = payload.get(field)
        if field == "source_map":
            parsed_map = _parse_source_map(
                value,
                source_text=material_analysis_body(
                    source_map_source_text or source_text),
                voice_source_spans=(out.get("voice_source_spans") or [])
                + list(known_voice_source_spans or []),
            )
            if parsed_map:
                out[field] = parsed_map
            continue
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        if field == "voice_sources":
            items = [parsed for parsed in (_parse_voice_source(item) for item in value)
                     if parsed]
            if items:
                out[field] = items[:_DEEP_LIST_MAX]
            continue
        if field == "voice_source_spans":
            items = [parsed for parsed in (
                _parse_voice_source_span(item, timestamped_rows) for item in value)
                if parsed]
            if items:
                out[field] = items[:_VOICE_SOURCE_SPANS_PER_WINDOW]
            continue
        if field == "external_context_queries":
            purposes: Dict[str, str] = {}
            queries = _normalize_external_query_plan(value, purposes)
            if queries:
                out[field] = [
                    {"query": query, "purpose": purposes[query]}
                    if any(isinstance(item, dict) for item in value) else query
                    for query in queries]
            continue
        items = []
        item_limit = 2 if field == "external_context_queries" else _DEEP_LIST_MAX
        item_chars = 220 if field == "external_context_queries" else 80
        for item in value:
            entry = " ".join(str(item or "").split())
            if entry and entry not in items:
                items.append(entry[:item_chars])
            if len(items) >= item_limit:
                break
        if items:
            out[field] = items
    return out


async def analyze_deep(raw_text: str, *, kind: str = "text", title: str = "",
                       comments: List[str] = None,
                       known_voice_source_spans: List[Dict[str, Any]] = None) -> Dict:
    """素材的深度分析产物（六字段框架）。**失败/超短返回 {}，绝不抛、绝不阻塞导入。**

    评论也进分析输入：用户点名"评论区的观点可以增加 AI 分析的角度、拓宽讨论的
    深度"——评论区正是 viewpoints 的主要来源。

    2026-09-21：**超长素材分段各读一遍再归并**。原实现只读前 12,000 字，
    60,000 字的素材有 80% 从没被读过，产出的"观点"全来自开头。用户口径：
    "要彻底了解到这个视频的内容，而不是这种囫囵吞枣"，且明确接受多花时间。
    分段调用**并行发出**（`asyncio.gather`），墙钟时间≈一次调用。
    """
    text = str(raw_text or "").strip()
    body_text = material_analysis_body(text)
    if not text or not needs_deep_analysis(text + "\n" + "\n".join(comments or [])):
        return {}
    if comments:
        joined = "\n".join(str(c or "").strip() for c in comments if str(c or "").strip())
        if joined:
            text = text + "\n\n" + COMMENT_MARKER + "\n" + joined
    windows = _windows(text, DEEP_INPUT_CHARS)
    source_map_kwargs = {"source_evidence_text": body_text}
    if known_voice_source_spans:
        source_map_kwargs["known_voice_source_spans"] = known_voice_source_spans
    if len(windows) == 1:
        return await _deep_once(
            windows[0], kind=kind, title=title,
            **source_map_kwargs)
    parts = [part for part in await asyncio.gather(
        *[_deep_once(window, kind=kind, title=title, partial=True,
                     **source_map_kwargs)
          for window in windows]) if part]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    merged = _merge_deep(parts)
    voice_sources = list(merged.get("voice_sources") or [])
    if any(isinstance(item, dict) and item.get("alternatives")
           for item in voice_sources):
        reconciled = await _reconcile_voice_sources(text, title=title)
        if reconciled:
            reconciled_speakers = {
                item.get("speaker_id") for item in reconciled
                if isinstance(item, dict) and item.get("speaker_id")
            }
            merged["voice_sources"] = list(reconciled[:_DEEP_LIST_MAX]) + [
                item for item in voice_sources
                if isinstance(item, dict)
                and item.get("speaker_id") not in reconciled_speakers
            ][:max(0, _DEEP_LIST_MAX - len(reconciled))]
    return merged


# ── 图片：要"视觉"，不要"视觉模型工程"（plans/0916-24 §一/§四）─────────────
#
# 结论逐字照办：**调用现成多模态 API**（按张计费、导入时一次性、不进每轮延迟），
# 不自训模型。2026-09-17 探活实测：当时的主端点（方舟 `ep-20260722235131-4m89m`）
# 能读图（合成图 "HELLO 42 BEAR" 答对），**不需要另开通 key**。
# 2026-09-18：主端点切 DeepSeek 官方后**读图能力保持**——用合成图实测
# `deepseek-flash`：32x32 左半红/右半蓝 → LEFT=red, RIGHT=blue；橙底+正中绿块
# → BACKGROUND=orange, SQUARE=green（prompt_tokens 216 ⇒ 图确实进了上下文）。
# ⚠ 但同 key 下的 `deepseek-v4-pro` **不认图**：同请求 prompt_tokens 只有 37
# （图被静默丢弃）、四个答案全错 ⇒ 读图链路别换成 pro。
VISION_MODEL_ENV = "MATERIAL_VISION_MODEL"   # 换视觉模型只改这里（留空用默认端点）
MAX_IMAGE_BYTES = 8 * 1024 * 1024            # 按张计费 + 请求体限制，单张 8MB 封顶

_IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp", "bmp": "image/bmp", "gif": "image/gif",
               "heic": "image/heic"}


def image_mime(filename: str = "", content_type: str = "") -> str:
    """图片 MIME：扩展名优先，MIME 兜底，未知按 png（Ark 认 content-type）。"""
    mime = str(content_type or "").split(";")[0].strip().lower()
    if mime.startswith("image/"):
        return mime
    ext = str(filename or "").strip().lower().rsplit(".", 1)[-1]
    return _IMAGE_MIME.get(ext, "image/png")


def _image_prompt(filename: str) -> str:
    return (
        "这是一张用户导入的图片素材（可能是截图、书页、菜单、剧照、海报或聊天记录）。\n"
        "只输出一个 JSON 对象（不要代码块标记、不要解释）：\n"
        '{"ocr_text": "图里出现的文字原样抄下来（没有就空串）", '
        '"description": "一句话说清这张图是什么", '
        '"entities": ["图里出现的具体人或事物名（没有就空数组）"]}\n'
        "硬要求：\n"
        "① 只写图里**真实可见**的内容——看不出来的东西一律不写，不许猜、不许补全；\n"
        "② ocr_text 保留原有语言（中文就抄中文），不要翻译、不要改写；\n"
        "③ description 用中文，说清“这是什么”（例如“某剧集第 3 季第 2 集的剧照”"
        "只在你确实能从画面判断时才这么写，否则写你看到的画面本身）；\n"
        "④ entities 只放图里明确出现的名字（剧名/人名/品牌/地点）。\n"
        "文件名线索：%s" % (str(filename or "").strip() or "（无）")
    )


# ── 素材词块候选（plans/0916-24 流水线⑤）──────────────────────────────────
#
# "从素材里抽词块候选 → 进 F10 闭环"：用户导入的素材本身就是他自己选的语料，
# 里面的地道表达比泛泛的"好短语"更值得收（他刚读过、有语境、还愿意聊）。
# **与归档归纳共用同一套垃圾判据**（`chunk_evidence.chunk_junk_reason`）——
# 两处各写一套迟早分叉，"词块本里有垃圾"就会从另一条路再回来。

MATERIAL_CHUNK_MODEL_ENV = "MATERIAL_CHUNK_MODEL"   # 空则用深析/摘要的模型
MATERIAL_CHUNK_TRIGGER_CHARS = 600                  # 太短的素材不值得再问一次
MATERIAL_CHUNK_LIMIT = 3                            # 与归档归纳同口径：宁缺毋滥
MATERIAL_CHUNK_INPUT_CHARS = 8000


def needs_material_chunks(raw_text: str) -> bool:
    return len(str(raw_text or "")) > MATERIAL_CHUNK_TRIGGER_CHARS


def _material_chunk_prompt(title: str, text: str, limit: int) -> str:
    return (
        "下面是学习者自己导入的一份英文素材%s。挑出最多 %d 条**值得他整块搬走**的"
        "多词表达（他会拿这些去说自己的话）。\n"
        "kind 只能是 collocation=搭配 / frame=带槽位句型 / marker=话语标记 / "
        "phrasal_verb=短语动词 / fixed=固定表达。\n"
        "每条必须给 example=**你自己写的**另一个情境的例句（不要抄素材里的句子）。\n"
        "优先挑：素材里反复出现的地道说法、对表达有明显帮助的搭配、学习者自己八成"
        "说不出但一看就懂的短语。\n"
        "禁止：单词（单词不进库）、超过 8 个词的整句、疑问句、含省略号/占位符的句型、"
        "专有名词（人名/剧名/品牌）、只看这一份素材才成立的说法。\n"
        '只输出一个 JSON 对象：{"chunks":[{"text":"词块","kind":"collocation",'
        '"gloss":"中文意思","example":"另一个情境的英文例句","slot":"",'
        '"reason":"为什么值得收"}]}，没有就返回空数组。不要输出 JSON 以外的任何文字。\n\n'
        "素材（标题：%s）：\n%s"
        % (("《%s》" % title) if title else "", limit, title or "（无）",
           text[:MATERIAL_CHUNK_INPUT_CHARS])
    )


async def extract_material_chunks(raw_text: str, *, title: str = "",
                                  limit: int = MATERIAL_CHUNK_LIMIT) -> List[Dict]:
    """素材文本 → 词块候选（**代码层判据与归档归纳一致**）。失败返回 []，绝不抛。

    每个候选都过 `chunk_junk_reason`（疑问句/占位符/整句/悬空首尾/词数）——
    这道判据与对话归纳共用，所以真机上那三类垃圾不会从素材这条路进库。
    """
    from engine.chunk_evidence import CHUNK_KINDS, chunk_junk_reason, normalize_chunk
    text = str(raw_text or "").strip()
    if not text or not needs_material_chunks(text):
        return []
    try:
        from llm_client import (DEFAULT_CHAT_MODEL, robust_llm_client,
                                thinking_disabled_extra_body)
        model = (os.getenv(MATERIAL_CHUNK_MODEL_ENV, "").strip()
                 or os.getenv(DEEP_MODEL_ENV, "").strip()
                 or os.getenv(ANALYZER_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL)
        raw = await robust_llm_client.chat_completion(
            [{"role": "user", "content": _material_chunk_prompt(title, text, limit)}],
            model=model, temperature=0.3, max_tokens=700, timeout=45,
            extra_body=thinking_disabled_extra_body(model),
        )
        import json
        body = str(raw or "").strip()
        start, end = body.find("{"), body.rfind("}")
        payload = json.loads(body[start:end + 1]) if 0 <= start < end else {}
        items = payload.get("chunks") if isinstance(payload, dict) else None
    except Exception:
        return []
    out: List[Dict] = []
    seen: set = set()
    for item in (items or []):
        if not isinstance(item, dict) or len(out) >= max(1, int(limit)):
            continue
        candidate_text = " ".join(str(item.get("text") or "").split())
        norm = normalize_chunk(candidate_text)
        if not norm or norm in seen:
            continue
        if chunk_junk_reason(candidate_text):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in CHUNK_KINDS:
            continue
        # 例句必须给且不能是素材里的原句（"换个情境照样能用"的自证，与归纳侧同口径）
        example = " ".join(str(item.get("example") or "").split())
        if not example or normalize_chunk(example) in normalize_chunk(text):
            continue
        seen.add(norm)
        out.append({
            "text": candidate_text[:80], "norm": norm, "kind": kind,
            "gloss": " ".join(str(item.get("gloss") or "").split())[:60],
            "example": example[:160],
            "slot": " ".join(str(item.get("slot") or "").split())[:30],
            "reason": " ".join(str(item.get("reason") or "").split())[:120],
        })
    return out


# ── 素材库专用词块抽取（plans/0923-08 §4.4）──────────────────────────────
#
# 旧 `extract_material_chunks` 只过**形态闸**（chunk_junk_reason），没有
# `0923-05` D2 的"常见 / 跨场景 / 值得专门教"三筛 —— A5 要的"好词块"不能只靠形态。
# 这里**沿用同一段解析与形态代码**，只在同一次模型输出里多要几个字段，再多过一道
# 现成的 `filter_chunk_quality`（零额外模型调用）。
# ⚠ 旧函数签名/提示词/返回与旧口行为**一律不动**：旧客户端吃的是老形状。

def _library_chunk_prompt(title: str, text: str, limit: int) -> str:
    return (
        "下面是学习者自己导入的一份英文素材%s。挑出最多 %d 条**值得他整块搬走**的"
        "多词表达（他会拿这些去说自己的话）。\n"
        "kind 只能是 collocation=搭配 / frame=带槽位句型 / marker=话语标记 / "
        "phrasal_verb=短语动词 / fixed=固定表达。\n"
        # 2026-09-24 用户口径：收录硬标准 = 意思不能靠直译猜出来（与对话归纳同一份措辞，
        # 两处判据不许分叉——用户点名的 "think too much" 就是从素材这条链进来的）。
        "收录硬标准——**这条的意思能不能靠直译猜出来**：学习者认识每个单词、"
        "把词面直译拼起来就得到实际意思的，一律不收"
        "（如 think too much=想太多、work hard=努力工作、come back=回来）；"
        "直译拼不出实际意思、必须整块记的才收"
        "（如 by the way=顺便说一句、according to=根据——字面'与…一致'推不出'根据'）。\n"
        "每条还要给这几项判据（这是入库门槛，宁缺毋滥）：\n"
        "  literal_gloss: 把这条**逐词直译**拼出来的字面意思（中文）——必须与 gloss 明显不同，"
        "两条越像说明这条越不值得收（代码会比对，直译≈实际意思就整条不报）；\n"
        "  frequency_band: common / uncertain / rare —— 只有你确定是**高频常见**才写 common；\n"
        "  transfer_scene: 一句话说清它还能用在**哪个别的场景**（要具体，不能写“很多场合”）；\n"
        "  teaching_worth: true/false —— 是否值得专门教一次；\n"
        "  teaching_reason: 为什么值得教（具体，别写“很有用”）；\n"
        "  nontransparent_or_conventional: true/false —— **只有直译拼不出实际意思时才是 true**；\n"
        "  fixed_pattern: true/false —— 是固定搭配/句型；\n"
        "  memory_hook: 帮学习者记住它的具体线索（可以是空的）。\n"
        "每条必须给 example=**你自己写的**另一个情境的例句（不要抄素材里的句子）。\n"
        "禁止：单词（单词不进库）、超过 8 个词的整句、疑问句、含省略号/占位符的句型、"
        "专有名词（人名/剧名/品牌）、只看这一份素材才成立的说法。\n"
        '只输出一个 JSON 对象：{"chunks":[{"text":"词块","kind":"collocation",'
        '"gloss":"中文意思","literal_gloss":"逐词直译的字面意思","example":"另一个情境的英文例句","slot":"",'
        '"reason":"为什么值得收","frequency_band":"common","transfer_scene":"...",'
        '"teaching_worth":true,"teaching_reason":"...",'
        '"nontransparent_or_conventional":false,"fixed_pattern":true,"memory_hook":""}]}，'
        "没有就返回空数组。不要输出 JSON 以外的任何文字。\n\n"
        "素材（标题：%s）：\n%s"
        % (("《%s》" % title) if title else "", limit, title or "（无）",
           text[:MATERIAL_CHUNK_INPUT_CHARS])
    )


async def extract_library_material_chunks(raw_text: str, *, title: str = "",
                                          limit: int = MATERIAL_CHUNK_LIMIT) -> List[Dict]:
    """素材库导入 → 过 D2 三筛的词块候选（plans/0923-08 §4.4）。失败返回 []，绝不抛。

    与旧 `extract_material_chunks` 的差别只有两处：
      ① 同一次调用多要 D2 字段；
      ② 过 `filter_chunk_quality`（把**这份素材原文**当作"本场对话"传进去，于是
         "transfer_scene 不能是本场原文 / example 不能抄本场原句"这两条就地生效）。
    返回项带齐 D2 字段，供调用方直接落 `chunk_items` 候选。
    """
    from engine.chunk_evidence import (CHUNK_KINDS, chunk_junk_reason,
                                       filter_chunk_quality, normalize_chunk)
    text = str(raw_text or "").strip()
    if not text or not needs_material_chunks(text):
        return []
    try:
        from llm_client import (DEFAULT_CHAT_MODEL, robust_llm_client,
                                thinking_disabled_extra_body)
        model = (os.getenv(MATERIAL_CHUNK_MODEL_ENV, "").strip()
                 or os.getenv(DEEP_MODEL_ENV, "").strip()
                 or os.getenv(ANALYZER_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL)
        raw = await robust_llm_client.chat_completion(
            [{"role": "user", "content": _library_chunk_prompt(title, text, limit)}],
            model=model, temperature=0.3, max_tokens=1100, timeout=45,
            extra_body=thinking_disabled_extra_body(model),
        )
        import json
        body = str(raw or "").strip()
        start, end = body.find("{"), body.rfind("}")
        payload = json.loads(body[start:end + 1]) if 0 <= start < end else {}
        items = payload.get("chunks") if isinstance(payload, dict) else None
    except Exception:
        return []
    norm_text = normalize_chunk(text)
    out: List[Dict] = []
    seen: set = set()
    for item in (items or []):
        if not isinstance(item, dict) or len(out) >= max(1, int(limit)):
            continue
        candidate_text = " ".join(str(item.get("text") or "").split())
        norm = normalize_chunk(candidate_text)
        if not norm or norm in seen:
            continue
        # 形态闸（与旧链同一条判据）→ 再补旧链没有的"整句/超长"死线
        if chunk_junk_reason(candidate_text):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in CHUNK_KINDS:
            continue
        example = " ".join(str(item.get("example") or "").split())
        if not example:
            continue
        cand = {
            "text": candidate_text[:80], "norm": norm, "kind": kind,
            "gloss": " ".join(str(item.get("gloss") or "").split())[:60],
            # 直译性闸的证据（2026-09-24）：没有它就没法复核"是不是能直译猜出来"
            "literal_gloss": " ".join(str(item.get("literal_gloss") or "").split())[:60],
            "example": example[:160],
            "slot": " ".join(str(item.get("slot") or "").split())[:30],
            "reason": " ".join(str(item.get("reason") or "").split())[:120],
            "frequency_band": str(item.get("frequency_band") or "").strip().lower(),
            "transfer_scene": " ".join(str(item.get("transfer_scene") or "").split())[:120],
            "teaching_worth": item.get("teaching_worth"),
            "teaching_reason": " ".join(str(item.get("teaching_reason") or "").split())[:160],
            "nontransparent_or_conventional": item.get("nontransparent_or_conventional"),
            "fixed_pattern": item.get("fixed_pattern"),
            "memory_hook": " ".join(str(item.get("memory_hook") or "").split())[:120],
        }
        # D2 三筛：把这份素材原文当作"本场对话"，于是"不能是本场原文/原句"落地
        if filter_chunk_quality([cand], [{"content": text}]):
            seen.add(norm)
            out.append(cand)
    return out


# ── 成盘归纳：分组提要 · 可聊角度 · 置顶词块（plans/0923-08 §4.2）────────────
#
# 一次轮外调用做完三件事；**任何一项不合格都退到确定性兜底**（逐份短提要 +
# 原句里已有的问题），绝不产生空盘。校验在本地做，不靠模型自证。

MATERIAL_TAPE_MODEL_ENV = "MATERIAL_TAPE_MODEL"      # 空则用素材分析的那一组模型
TAPE_PLAN_BLOCKS_PER_SOURCE = 6                      # 每份素材最多送几块
TAPE_PLAN_BLOCK_CHARS = 900
TAPE_PLAN_MAX_SOURCES = 8                             # 超出上限时拒绝成盘，避免静默截断
TAPE_PLAN_MAX_HOOKS_PER_SOURCE = 2                    # 宁取少量有依据的角度，不灌满清单
TAPE_PLAN_MAX_OUTPUT_TOKENS = 4500


def _tape_plan_output_tokens(source_count: int) -> int:
    """Scale the one-time build output budget with source count (1..8)."""
    count = int(source_count or 0)
    if count < 1 or count > TAPE_PLAN_MAX_SOURCES:
        raise ValueError("一盘最多分析 %d 份素材" % TAPE_PLAN_MAX_SOURCES)
    return min(TAPE_PLAN_MAX_OUTPUT_TOKENS, 900 + 450 * count)


def _tape_block_fields(block: Any, fallback_no: int) -> tuple:
    """Return (block_no, span_kind, text), accepting the old string shape too."""
    if isinstance(block, dict):
        try:
            block_no = int(block.get("block_no") or fallback_no)
        except (TypeError, ValueError):
            block_no = fallback_no
        span_kind = str(block.get("span_kind") or "legacy_unattributed").strip().lower()
        if span_kind not in {"body", "comment", "legacy_unattributed"}:
            span_kind = "legacy_unattributed"
        return block_no, span_kind, str(block.get("text") or "")
    return fallback_no, "body", str(block or "")


def _first_question(blocks: List[str]) -> tuple:
    """从素材块里找第一个"像问题的句子"→ (句子, 块号)。找不到 → ("", 1)。

    取**问号前**的那一段（`他为什么走了？因为钱。` 这种一句里既有问又有答的，
    要的是问号前那半句），所以用 findall 而不是先按句号切分。
    """
    for index, block in enumerate(list(blocks or []), start=1):
        block_no, _span_kind, text = _tape_block_fields(block, index)
        for fragment in re.findall(r"[^\n?？]{6,120}[?？]", text):
            clean = _clean(fragment)
            if clean and 6 <= len(clean) <= 120:
                return clean, block_no
    return "", 1


def _tape_plan_fallback(sources: List[Dict]) -> Dict:
    """确定性兜底：逐份短提要；只有正文原句确实提出问题时才保留讨论点。"""
    groups: List[Dict] = []
    hooks: List[Dict] = []
    for src in list(sources or []):
        try:
            mid = int(src.get("material_id") or 0)
        except (ValueError, TypeError):
            continue
        if not mid:
            continue
        title = _clean(src.get("title")) or "这份素材"
        groups.append({"title": title[:40],
                       "summary": _clean(src.get("summary"))[:220],
                       "material_ids": [mid]})
        source_blocks = list(src.get("blocks") or [])
        body_blocks = [b for i, b in enumerate(source_blocks, start=1)
                       if _tape_block_fields(b, i)[1] == "body"]
        if body_blocks:
            question, block_no = _first_question(body_blocks)
            # A made-up generic question is not an evidence-backed discussion angle.
            # If analysis failed and no source sentence contains a question, leave
            # hooks empty; the opening path can fall back to the real material title.
            if question:
                hooks.append({"text": question,
                              "material_id": mid, "block_no": int(block_no),
                              "evidence_quote": question,
                              "span_kind": "body"})
    summary = "；".join(g["title"] for g in groups)[:400]
    return {"summary": summary, "groups": groups, "hooks": hooks, "top_chunks": []}


def _spread_tape_samples(items: List[tuple], limit: int) -> List[tuple]:
    """按原文位置均匀抽样，避免成盘只看见正文开头。"""
    values = list(items or [])
    count = max(0, int(limit or 0))
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indexes = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[index] for index in indexes]


def _tape_story_navigation(src: Dict, body_blocks: List[tuple]) -> str:
    """Revalidate the saved story index against this source's complete body."""
    saved = src.get("source_map")
    if not isinstance(saved, dict) or saved.get("validated") is not True:
        return ""
    # Stored maps contain proof objects; the parser expects original quotes.
    raw = {}
    for field in ("entities", "beats", "open_threads"):
        raw[field] = []
        for item in saved.get(field) or []:
            if not isinstance(item, dict):
                continue
            proof = item.get("evidence")
            if not isinstance(proof, dict):
                continue
            raw[field].append({**item, "evidence": proof.get("quote") or ""})
    body = "\n".join(item[2] for item in sorted(body_blocks, key=lambda row: row[0]))
    checked = _parse_source_map(raw, source_text=body, voice_source_spans=[])
    if not checked:
        return ""
    lines = []
    for item in checked.get("entities", [])[:5]:
        lines.append("人物：%s；%s；原句：%s" % (
            item["name"], item["role"], item["evidence"]["quote"]))
    beats = _spread_tape_samples(list(enumerate(checked.get("beats", []))), 8)
    for _, item in beats:
        lines.append("推进%d：%s%s；原句：%s" % (
            item["order"], item["event"],
            " → " + item["effect"] if item.get("effect") else "",
            item["evidence"]["quote"]))
    for item in checked.get("open_threads", [])[:2]:
        lines.append("本段未解：%s；原句：%s" % (
            item["question"], item["evidence"]["quote"]))
    return ("\n正文剧情索引（已用本份完整正文重新核对引句；按正文顺序组织，"
            "人物身份只限原句支持的范围；索引是分析导航，不是额外原文；"
            "优先用它纠正旧提要里人物与创作者的混淆，不把操作吐槽混进剧情）：\n- "
            + "\n- ".join(lines))


def _tape_plan_prompt(sources: List[Dict]) -> str:
    parts = []
    for src in list(sources or []):
        all_blocks = list(src.get("blocks") or [])
        normalized = [_tape_block_fields(block, i)
                      for i, block in enumerate(all_blocks, start=1)]
        body_blocks = [item for item in normalized if item[1] == "body"]
        comment_blocks = [item for item in normalized if item[1] == "comment"]
        # 成盘分析要同时看正文与评论，但评论只能作为有限的社区样本。
        # 给评论保留最多 2 块，其余预算给正文；不足时再按原文顺序补齐。
        chosen_comments = _spread_tape_samples(
            comment_blocks, min(2, TAPE_PLAN_BLOCKS_PER_SOURCE))
        body_limit = max(0, TAPE_PLAN_BLOCKS_PER_SOURCE - len(chosen_comments))
        chosen = _spread_tape_samples(body_blocks, body_limit) + chosen_comments
        if len(chosen) < TAPE_PLAN_BLOCKS_PER_SOURCE:
            already = {item[0] for item in chosen}
            chosen.extend(item for item in normalized
                          if item[0] not in already
                          and len(chosen) < TAPE_PLAN_BLOCKS_PER_SOURCE)
        chosen.sort(key=lambda item: item[0])
        span_labels = {"body": "正文", "comment": "评论区样本",
                       "legacy_unattributed": "来源未能区分"}
        body = "\n".join(
            "  [%d·%s] %s" % (block_no, span_labels[span_kind],
                             text[:TAPE_PLAN_BLOCK_CHARS])
            for block_no, span_kind, text in chosen
        )
        coverage = src.get("source_coverage")
        if not isinstance(coverage, dict):
            coverage = {}
        coverage_note = ""
        if (coverage.get("kind") == "video"
                and coverage.get("body_available") is False):
            coverage_note = (
                "\n内容覆盖：没有取得视频正文转写；当前正文块只有页面信息，"
                "其余是评论样本。不得据此归纳UP主观点，摘要必须明确说无法确认；"
                "评论只能作为次要的个体反应。"
            )
        if coverage.get("kind") == "video" and coverage.get("body_available") is None:
            coverage_note += "\n旧文本归属未核验：不能确认是视频正文还是页面介绍，不据此确定视频完整内容或与其他来源的关系。"
        voice_sources = [format_voice_source(value)[:220]
                         for value in (src.get("voice_sources") or [])
                         if value][:4]
        voice_source_spans = [str(value).strip()[:240]
                              for value in (src.get("voice_source_spans") or [])
                              if str(value or "").strip()][:12]
        voice_note = ("\n声源线索（自动分析判断，speaker ID 不是人物身份；"
                      "未确认的来源要保留不确定）：\n- "
                      + "\n- ".join(voice_sources)) if voice_sources else ""
        if voice_source_spans:
            voice_note += ("\n按转写时间标注的来源片段（只适用于所列时间和原句；"
                           "不把 speaker ID 当成来源类型）：\n- "
                           + "\n- ".join(voice_source_spans))
        story_note = _tape_story_navigation(src, body_blocks)
        background_map = src.get("source_map") if isinstance(src.get("source_map"), dict) else {}
        background_query = " ".join([str(src.get("title") or "")] + [
            str(entity.get("name") or "")
            for entity in background_map.get("entities", [])
            if isinstance(entity, dict)])
        background_rows = []
        background_budget = 650
        for item in select_external_reference_passages(src.get("external_context"), background_query):
            row = "- %s（%s）" % (item["quote"][:350], item["url"][:180])
            if len(row) <= background_budget:
                background_rows.append(row)
                background_budget -= len(row)
        background_note = ("\n外部参考背景（只解释身份/术语或辅助分析，不是本素材正文，不能替代本段情节证据）：\n"
                           + "\n".join(background_rows)) if background_rows else ""
        parts.append("素材 id=%s 标题《%s》类型%s\n已有提要：%s%s%s%s%s\n来源块（正文与评论已区分）：\n%s" % (
            src.get("material_id"), _clean(src.get("title")) or "（无）",
            _clean(src.get("kind")) or "text",
            ("已由下方核验剧情索引替代" if story_note
             else _clean(src.get("summary"))[:300] or "（无）"), coverage_note,
            voice_note, story_note, background_note, body))
    header = (
        "下面是学习者素材盘里选中的 %d 份素材（可能互不相关，也可能说同一件事）。\n"
        "请只输出一个 JSON 对象（不要代码块标记、不要解释）：\n"
        '{"summary":"整盘一句话提要（互不相关就说清这是几份不同内容）",'
        '"groups":[{"title":"分组名","summary":"这一组讲什么","material_ids":[1,2]}],'
        '"hooks":[{"text":"一个具体、值得讨论的角度或问题","why_worth_discussing":"为什么这个角度值得讨论，点出具体张力或含义","material_id":1,"block_no":2,'
        '"evidence_quote":"该角度依据的原文短句"}],'
        '"top_chunks":[{"text":"值得记的短语","material_id":1,"block_no":2,"quote":"该块里含这个短语的原句"}]}\n'
        "硬要求：\n"
        "总提要只做内容导航，优先交代素材正文的主题、推进或主要论证；每份来源用一到两句，"
        "不要把剧情细节、创作者评价和评论名单都塞进一句话。评论仅在补出不同且有价值的视角时"
        "另以评论者样本标明，不与正文叙述合成同一个结论。\n"
        "正文若带 speaker/time 标签，它们是 ASR 的说话段线索，不等于角色身份；"
        "要分清作品内对白、创作者旁白/实况操作和观众评论。创作者以‘我’代入角色操作时，"
        "不能直接改写成作品角色自己的判断。来源不确定的细节保持不确定。\n"
        "① groups 必须让每个素材 id **恰好出现一次**；只有**确有共同内容**才合并成一组，"
        "只是类型相同（都是剧/都是论文）而事实无关的**必须各自成组**；\n"
        "② hooks 是建议不是任务，每份素材最多储备 2 个彼此不同、真正值得讨论的角度；"
        "每个角度必须说清具体张力、取舍、因果或未解问题，不能只是换句话概括素材；"
        "why_worth_discussing 用一句话说明它为什么值得讨论，给出角度的分析价值，不能只写‘很有意思’；"
        "每个角度必须给 material_id + block_no + evidence_quote，引用必须逐字来自所标来源块；"
        "正文角度以正文为依据；若角度来自评论，why_worth_discussing 必须说清这是评论者的单个样本，"
        "不能把它扩大成普遍看法；材料没有足够证据支撑有价值的角度时，少给或不给，不要凑数；"
        "轻松剧情优先人际/处境/钱，悬疑优先嫌疑/证据/人物判断，长文优先争点；"
        "拿不准就不给这条；\n"
        "③ 这些 hooks 是后续对话可选的角度储备，不要在一轮里全部倒给用户；"
        "实际对话一次只带出一个低门槛切口。没看过素材的人不需要先认识人物或剧情，"
        "已经带着观点来的人先接住他的观点，再补最多一个新角度。评论是评论者的个别样本，"
        "可以补充、质疑或呼应正文；没有明确证据时不制造二元对立，也不说成观众普遍看法。\n"
        "④ top_chunks 只服务英语学习：仅从正文中挑素材里真实出现的英语多词表达（2–8 个英文词），"
        "quote 必须逐字含有该表达；纯中文正文或没有合适英语表达时返回空数组。"
        "不从中文句子硬造英语，不从评论区抽词块；\n"
        "⑤ 正文和评论必须严格按来源块分别归属；评论里的观点不能写成视频讲者观点，"
        "分析推论不能写成素材事实；\n"
        "⑥ 一律不许编造素材里没有的情节、人物、数字或结论；\n"
        "⑦ summary、groups 和 hooks 要保留每条判断的对象与范围。若材料分别谈到某厂牌、"
        "某地区、某时间段或不同评判标准，不要合并成一个总体排名；明确区分整体结论与局部评价。\n"
        % len(list(sources or []))
    )
    # ⚠ 正文（parts）里可能出现 `%`，所以先把 header 单独格式化完再拼接，
    # 绝不把 `%` 作用在含素材原文的整串上。
    return header + MATERIAL_CLAIM_SCOPE_RULE + "\n" + "\n\n".join(parts)


def _index_blocks(sources: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """{material_id: {"titles": set, "blocks": {no: {text, span_kind}}}} —— 校验索引。"""
    index: Dict[int, Dict[str, Any]] = {}
    for src in list(sources or []):
        try:
            mid = int(src.get("material_id") or 0)
        except (ValueError, TypeError):
            continue
        if not mid:
            continue
        index[mid] = {
            "titles": {_clean(src.get("title"))},
            "blocks": {},
        }
        for fallback_no, block in enumerate(list(src.get("blocks") or []), start=1):
            block_no, span_kind, text = _tape_block_fields(block, fallback_no)
            index[mid]["blocks"][block_no] = {
                "text": text, "span_kind": span_kind,
            }
    return index


def validate_tape_plan(payload: Dict, sources: List[Dict]) -> Dict:
    """对模型产物做**本地**校验（纯函数，零模型）。不合格的项直接丢，不留半成品。

    - groups：必须覆盖每个所选 id 且**恰好一次**，否则整盘分组退兜底；
    - hooks：material_id 必须属于本盘、block_no 必须存在、text 与 why_worth_discussing 非空；
    - top_chunks：除上述外，`quote` 必须**逐字出现在该块里**（原句核验）。
    """
    index = _index_blocks(sources)
    wanted = set(index.keys())
    out = {"summary": _clean((payload or {}).get("summary"))[:400],
           "groups": [], "hooks": [], "top_chunks": []}
    seen_ids: List[int] = []
    for group in ((payload or {}).get("groups") or []):
        if not isinstance(group, dict):
            continue
        members: List[int] = []
        for value in (group.get("material_ids") or []):
            try:
                mid = int(str(value).strip())
            except (ValueError, TypeError):
                continue
            if mid in wanted and mid not in members:
                members.append(mid)
        if not members:
            continue
        titles = [(_clean(index[m]["titles"] and list(index[m]["titles"])[0]) or "素材")
                  for m in members]
        out["groups"].append({
            "title": (_clean(group.get("title")) or "、".join(t[:12] for t in titles))[:40],
            "summary": _clean(group.get("summary"))[:220],
            "material_ids": members,
        })
        seen_ids.extend(members)
    # 每个 id 恰好一次：少一个、多一个都算分组不可用（宁可退兜底，不要漏素材）
    if sorted(seen_ids) != sorted(wanted):
        out["groups"] = []
    hook_counts: Dict[int, int] = {}
    seen_hooks = set()
    for hook in ((payload or {}).get("hooks") or []):
        if not isinstance(hook, dict):
            continue
        try:
            mid = int(str(hook.get("material_id")).strip())
            block_no = int(str(hook.get("block_no")).strip())
        except (ValueError, TypeError):
            continue
        text = _clean(hook.get("text"))
        evidence_quote = _clean(hook.get("evidence_quote"))
        source_block = index.get(mid, {}).get("blocks", {}).get(block_no)
        signature = (mid, block_no, text.casefold())
        why = _clean(hook.get("why_worth_discussing"))
        if (mid in wanted and source_block and text and why and evidence_quote
                and evidence_quote in source_block["text"]
                and signature not in seen_hooks
                and hook_counts.get(mid, 0) < TAPE_PLAN_MAX_HOOKS_PER_SOURCE):
            out["hooks"].append({"text": text[:200], "material_id": mid,
                                 "block_no": block_no,
                                 "evidence_quote": evidence_quote[:240],
                                 "why_worth_discussing": why[:240],
                                 "span_kind": source_block["span_kind"]})
            seen_hooks.add(signature)
            hook_counts[mid] = hook_counts.get(mid, 0) + 1
    for chunk in ((payload or {}).get("top_chunks") or []):
        if not isinstance(chunk, dict):
            continue
        try:
            mid = int(str(chunk.get("material_id")).strip())
            block_no = int(str(chunk.get("block_no")).strip())
        except (ValueError, TypeError):
            continue
        text = _clean(chunk.get("text"))
        quote = _clean(chunk.get("quote"))
        source_block = index.get(mid, {}).get("blocks", {}).get(block_no)
        if mid not in wanted or not source_block:
            continue
        if not text or not quote:
            continue
        if source_block["span_kind"] != "body":
            continue
        if quote not in source_block["text"]:
            continue                       # 原句核验：引用不逐字出现就不要
        words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)
        if not 2 <= len(words) <= 8:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z'’\- ]*", text):
            continue                       # 目标语言是英语；中文/混写不进英语词块卡
        if text.casefold() not in quote.casefold():
            continue                       # 候选短语必须真的出现在所引原句里
        out["top_chunks"].append({"text": text[:60], "material_id": mid,
                                  "block_no": block_no, "quote": quote[:200]})
    return out


async def build_material_tape_plan(sources: List[Dict], *,
                                   allow_model: bool = True) -> Dict:
    """成盘归纳：{summary, groups, hooks, top_chunks}。失败/不合格一律退确定性兜底。

    `sources` = [{"material_id", "title", "kind", "summary", "blocks": [全量块文本]}]。
    本函数**不落库、不建盘**：事务由调用方（l2.finish_material_tape_build）负责。
    """
    source_list = [s for s in (sources or []) if isinstance(s, dict)]
    if not source_list:
        return {"summary": "", "groups": [], "hooks": [], "top_chunks": []}
    if len(source_list) > TAPE_PLAN_MAX_SOURCES:
        raise ValueError("一盘最多分析 %d 份素材" % TAPE_PLAN_MAX_SOURCES)
    fallback = _tape_plan_fallback(source_list)
    if not allow_model:
        return fallback
    try:
        from llm_client import (DEFAULT_CHAT_MODEL, robust_llm_client,
                                thinking_disabled_extra_body)
        model = (os.getenv(MATERIAL_TAPE_MODEL_ENV, "").strip()
                 or os.getenv(MATERIAL_CHUNK_MODEL_ENV, "").strip()
                 or os.getenv(DEEP_MODEL_ENV, "").strip()
                 or os.getenv(ANALYZER_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL)
        raw = await robust_llm_client.chat_completion(
            [{"role": "user", "content": _tape_plan_prompt(source_list)}],
            model=model, temperature=0.3,
            max_tokens=_tape_plan_output_tokens(len(source_list)), timeout=60,
            max_retries=2, gate="background",
            extra_body=thinking_disabled_extra_body(model),
        )
        import json
        body = str(raw or "").strip()
        start, end = body.find("{"), body.rfind("}")
        payload = json.loads(body[start:end + 1]) if 0 <= start < end else {}
    except Exception:
        return fallback
    plan = validate_tape_plan(payload if isinstance(payload, dict) else {}, source_list)
    if not plan["groups"]:
        plan["groups"] = fallback["groups"]
    if not plan["summary"]:
        plan["summary"] = fallback["summary"]
    if not plan["hooks"]:
        plan["hooks"] = fallback["hooks"]
    return plan


def _parse_image_json(raw: str) -> Dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    payload = {}
    if 0 <= start < end:
        try:
            import json
            payload = json.loads(text[start:end + 1])
            if not isinstance(payload, dict):
                payload = {}
        except (ValueError, TypeError):
            payload = {}
    ocr = " ".join(str(payload.get("ocr_text") or "").split())
    desc = " ".join(str(payload.get("description") or "").split())
    entities = [str(e).strip() for e in (payload.get("entities") or [])
                if str(e or "").strip()][:8]
    if not ocr and not desc:
        # 模型没给结构化输出：把原文当描述用（总比"图里什么都没有"对用户有用）
        fallback = " ".join(text.split())
        if not fallback:
            return {}
        desc = fallback[:200]
    return {"ocr_text": ocr, "description": desc, "entities": entities}


async def analyze_image(data: bytes, *, filename: str = "", content_type: str = "") -> Dict:
    """图片 → {ok, text, summary, entities, error}。**一次多模态调用**（方案 §四-2）。

    text = 画面描述 + OCR 文字 —— 这样摘要/分块/[Material] 注入/开场取材全都能
    照常工作，图片和文档在流水线里是同一种东西（方案 §一："云盘只是运输层"同理）。

    失败**不产空带**（§7-1）：返回 ok=False + 明确中文原因给前端显示。
    """
    blob = data or b""
    if not blob:
        return {"ok": False, "text": "", "summary": "", "entities": [],
                "error": "这张图是空的"}
    if len(blob) > MAX_IMAGE_BYTES:
        return {"ok": False, "text": "", "summary": "", "entities": [],
                "error": "图片太大（上限 %dMB），先压缩再导入"
                         % (MAX_IMAGE_BYTES // (1024 * 1024))}
    mime = image_mime(filename, content_type)
    try:
        import base64
        from llm_client import (DEFAULT_CHAT_MODEL, robust_llm_client,
                                thinking_disabled_extra_body)
        model = os.getenv(VISION_MODEL_ENV, "").strip() or DEFAULT_CHAT_MODEL
        payload_url = "data:%s;base64,%s" % (mime, base64.b64encode(blob).decode())
        messages = [{"role": "user", "content": [
            {"type": "text", "text": _image_prompt(filename)},
            {"type": "image_url", "image_url": {"url": payload_url}},
        ]}]
        raw = await robust_llm_client.chat_completion(
            messages, model=model, temperature=0.2, max_tokens=700, timeout=60,
            extra_body=thinking_disabled_extra_body(model),
        )
        parsed = _parse_image_json(raw)
    except Exception:
        return {"ok": False, "text": "", "summary": "", "entities": [],
                "error": "读图失败（视觉模型没响应），稍后再试一次"}
    if not parsed:
        return {"ok": False, "text": "", "summary": "", "entities": [],
                "error": "这张图没能读出内容，换一张更清楚的试试"}
    desc, ocr = parsed["description"], parsed["ocr_text"]
    text = desc
    if ocr:
        text = (desc + "\n\n图中文字：\n" + ocr).strip()
    return {"ok": True, "text": text, "summary": desc,
            "entities": parsed.get("entities") or [], "error": ""}



def merge_external_context(previous, refreshed):
    """Refresh search policy without discarding usable persisted evidence."""
    previous = previous if isinstance(previous, dict) else {}
    refreshed = refreshed if isinstance(refreshed, dict) else {}
    result = dict(refreshed or previous)
    sources, seen = [], set()
    for context in (refreshed, previous):
        for source in context.get("sources", []) or []:
            if not isinstance(source, dict) or not source.get("url"):
                continue
            if source["url"] in seen:
                current = next(item for item in sources if item["url"] == source["url"])
                if not current.get("body_text") and source.get("body_text"):
                    current.update({key: source[key] for key in ("body_text", "body_url", "content_status") if key in source})
                continue
            source = dict(source)
            seen.add(source["url"])
            sources.append(source)
    result["sources"] = sources[:EXTERNAL_CONTEXT_MAX_SOURCES]
    result["query_policy"] = EXTERNAL_CONTEXT_QUERY_POLICY
    result["schema_version"] = EXTERNAL_CONTEXT_SCHEMA_VERSION
    if sources:
        result["status"] = "ready"
    if not refreshed or refreshed.get("status") != "ready":
        result["refresh_status"] = refreshed.get("status", "unavailable")
    return result
