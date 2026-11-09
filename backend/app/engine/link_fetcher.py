"""素材导入：链接抓取与正文提取（plans/0916-21 D4/F3）。

分层原则（用户 2026-09-17 明确："我会用别的大模型做素材分析"）：
    **抓取 / 解析是确定性的；摘要 / 分块 / 引用选择才是模型相关的。**
本文件只做前者，**绝不调用任何模型** —— 模型相关的一律在
`engine/material_analyzer.py` 的单一入口里。将来换大模型只改那一个文件。

底线（方案 §7-1，本项目废过两次的坑）：
    抓不到 / 解析不出 → **返回明确错误**，绝不返回空正文冒充成功。
    "前端拿到 200 但没数据"是这项目最贵的教训之一，宁可失败也不要空素材带。

反爬现实（§7-4）：微信 / 知乎 / 小红书等封闭平台大概率抓不到，**按上面这条明说即可**，
不硬做绕过（绕过既不稳定也踩 ToS）。
"""
from __future__ import annotations

import html as _html
import json
import re
import time
from hashlib import md5
from typing import Dict, List, Tuple
from urllib.parse import urlencode, urlparse

# 视频站点：**音轨就是内容**（真内容在音轨里）。2026-09-21 起走
# `engine/media_transcribe` 把音轨转成文字；转不出来才退回"标题 + 简介"的轻素材。
VIDEO_HOSTS = (
    "youtube.com", "youtu.be", "bilibili.com", "b23.tv", "tiktok.com",
    "douyin.com", "vimeo.com", "ixigua.com", "weibo.com/tv",
)

# 同域下的"图文"路径不是视频：B 站**专栏**（/read/cv…）是文章，正文可抓
# （实测 447 字），按视频站处理只能拿到标题+简介，成一条 35 字的垃圾轻素材。
_NON_VIDEO_PATHS = (r"bilibili\.com/read/",)

# 站点用 HTTP 200 返回的**空占位页**（失效/已删视频）：标题是"视频去哪了呢？"这类，
# 简介是**未填充的模板**（desc 带 `{$0}`/`{$1}` 和 `undefined`）。
# 必须判失败——§7-1：宁可失败，也不产一盘题为"视频去哪了呢？"的垃圾素材带。
_PLACEHOLDER_TITLE_RE = re.compile(
    r"视频去哪了呢|视频不存在|视频已失效|视频已删除|内容不存在|页面不存在|"
    r"该页面无法访问|not\s*found", re.IGNORECASE)
_PLACEHOLDER_TPL_RE = re.compile(r"\{\$\d+\}")

# 一些站点对默认 UA 直接 403；用浏览器 UA 只为拿到公开页面，不做任何伪装绕过
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# 正文提取的最低字数门槛：低于这个数说明抽出来的是导航/版权声明之类的残渣
MIN_ARTICLE_CHARS = 200
# 单次抓取上限：防止一个超大页面把内存和后续 LLM 调用拖垮
MAX_ARTICLE_CHARS = 60000

# ── 视频热评（plans/0916-24 §三·五 修复 1）──────────────────────────────
# 口径（用户 2026-09-17）：评论是**增强项**，让模型大致分得清"这段来自视频本身 /
# 来自评论区"即可，不做硬性标注模板；目的是信息更全面。抓不到就不带，**绝不**因此判失败。
MAX_COMMENTS = 20          # 每次最多带回的评论条数（控注入体积与后续 LLM 成本）
MAX_COMMENT_CHARS = 300    # 单条评论截断长度（评论常是长段子，一条不能吃掉整段预算）
# 评论要额外走 1–2 次网络往返，超时给得比页面抓取短：评论接口慢/被风控时，
# 不该把一次导入拖到页面超时（默认 20s）才返回。
COMMENT_TIMEOUT = 10

_BILIBILI_SUFFIXES = ("bilibili.com", "b23.tv")
_YOUTUBE_SUFFIXES = ("youtube.com", "youtu.be")

# B 站评论接口自 2023 起要 wbi 签名（未签名实测 code=-352/-403）。签名密钥从 nav 现取，
# 不缓存：密钥每天轮换，多一次轻量请求换"永远不会用到过期 key"。
_WBI_MIXIN_TAB = (46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
                  27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
                  37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
                  22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52)
_BILI_NAV = "https://api.bilibili.com/x/web-interface/nav"
_BILI_REPLY_WBI = "https://api.bilibili.com/x/v2/reply/wbi/main"
# 老接口不签名，但 2026-09 实测只回前 3 条热评 —— 作为 wbi 失败时的兜底（有胜于无）
_BILI_REPLY_LEGACY = "https://api.bilibili.com/x/v2/reply"
_BILI_VIEW = "https://api.bilibili.com/x/web-interface/view"
# oid（= aid）优先从视频页 __INITIAL_STATE__ 的 "aid" 取，取不到再按 URL 形态兜
_BILI_AID_RE = re.compile(r'"aid"\s*:\s*(\d+)')
_BILI_AV_RE = re.compile(r"/video/av(\d+)", re.IGNORECASE)
_BILI_BVID_RE = re.compile(r"/video/(BV[0-9A-Za-z]{10})", re.IGNORECASE)

# YouTube 评论藏在 ytInitialData 的 continuation token 后面（要再 POST 一次 innertube）
_YT_API_KEY_RE = re.compile(r'"INNERTUBE_API_KEY"\s*:\s*"([^"]+)"')
_YT_CLIENT_VERSION_RE = re.compile(r'"INNERTUBE_CLIENT_VERSION"\s*:\s*"([^"]+)"')
_YT_INITIAL_DATA_RE = re.compile(r"ytInitialData\s*=\s*")
_YT_COMMENT_SECTION = "comment-item-section"
_YT_LIKE_RE = re.compile(r"^([\d.]+)\s*([KkMm万]?)")
_COMMENT_WS = re.compile(r"\s+")

_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|svg|template|iframe|head)\b.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_BLOCK_BREAK = re.compile(r"</?(p|div|br|li|tr|h[1-6]|section|article|blockquote)\b[^>]*>",
                          re.IGNORECASE)
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_TITLE = re.compile(
    r"""<meta[^>]+(?:property|name)=["'](?:og:title|twitter:title)["'][^>]*>""",
    re.IGNORECASE)
_META_DESC = re.compile(
    r"""<meta[^>]+(?:property|name)=["'](?:og:description|twitter:description|description)["'][^>]*>""",
    re.IGNORECASE)
_META_CONTENT = re.compile(r"""content=["'](.*?)["']""", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"[ \t\u00a0]+")
_BLANK = re.compile(r"\n{3,}")


def is_video_url(url: str) -> bool:
    """是不是视频站链接（V1 走"轻素材"路径，不假装读到了内容）。

    注意先排除**同域的图文路径**：B 站专栏 `/read/cv…` 是文章，走文章路径能拿到正文。
    """
    raw = str(url or "").strip()
    low = raw.lower()
    if any(re.search(pattern, low) for pattern in _NON_VIDEO_PATHS):
        return False
    host = (urlparse(raw).netloc or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == h or host.endswith("." + h) or h in low
               for h in VIDEO_HOSTS)


def is_placeholder_page(title: str, desc: str) -> bool:
    """HTTP 200 的**空占位页**（失效/已删视频）→ True。

    站点对已删内容仍返 200，标题是"视频去哪了呢？"、简介是未填充的模板
    （实测 `{$0}, 视频播放量 undefined、弹幕数 {$1}…`）。这类页面认出来就别收。
    """
    if _PLACEHOLDER_TITLE_RE.search(str(title or "")):
        return True
    return bool(_PLACEHOLDER_TPL_RE.search(str(desc or "")))


def strip_html(html_text: str) -> Tuple[str, str]:
    """HTML → (title, text)。纯确定性：去脚本样式、块级标签转换行、实体解码。

    不做 DOM 解析、不引第三方依赖 —— V1 的正文提取只要求"够用"，
    质量不够时再上 trafilatura 类依赖（那时才进 requirements，§7-2）。
    """
    raw = str(html_text or "")
    title = ""
    match = _TITLE_TAG.search(raw)
    if match:
        title = _clean_inline(match.group(1))
    if not title:
        head_meta = _META_TITLE.search(raw)
        if head_meta:
            content = _META_CONTENT.search(head_meta.group(0))
            if content:
                title = _clean_inline(content.group(1))

    text = _HTML_COMMENT.sub(" ", raw)
    text = _DROP_BLOCKS.sub(" ", text)
    text = _BLOCK_BREAK.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = _html.unescape(text)
    lines = [_clean_inline(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    body = "\n".join(lines)
    body = _BLANK.sub("\n\n", body).strip()
    return title, body


def meta_description(html_text: str) -> str:
    """取 og/description 摘要（视频页的"轻素材"正文就是它）。"""
    match = _META_DESC.search(str(html_text or ""))
    if not match:
        return ""
    content = _META_CONTENT.search(match.group(0))
    return _clean_inline(content.group(1)) if content else ""


def _video_page_description(desc: str) -> str:
    """Remove recommendation/sidebar material appended to video SEO metadata.

    Bilibili's page description can append a "相关视频" list after the actual
    description. That list is not part of the selected video and must not enter
    its fallback summary as if the uploader had discussed those other videos.
    """
    text = _clean_inline(desc)
    lower = text.lower()
    markers = ("相关视频：", "相关视频:", "相关推荐：", "相关推荐:")
    cut_at = min((lower.find(marker.lower()) for marker in markers
                  if lower.find(marker.lower()) >= 0), default=-1)
    return text[:cut_at].rstrip(" ，,；;|-") if cut_at >= 0 else text


def longest_text_block(body: str, min_chars: int = 24) -> str:
    """最长文本块启发式：挑段落最密集的那一段（正文通常连成一大段）。

    比"整页文字"强的地方在于它会丢掉导航/页脚/相关阅读这类零散短句。

    min_chars 默认 24（**中文下限**）：中文信息密度高，一行 35 来字就是一个完整长句
    ——实测阈值 40 会把真正的正文段落当噪音丢掉（"…他前往亚特兰大寻找家人，"39 字），
    结果抽出来的正文比导航还短。英文页面靠"连续段落合并"补回来，不受影响。
    """
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", str(body or ""))]
    paragraphs = [p for p in paragraphs if len(p) >= min_chars]
    if not paragraphs:
        return ""
    # 连续段落合并成块，取字数最大的块（正文段落通常是连续出现的）
    best, best_len, current = [], 0, []
    for para in paragraphs + [""]:
        if para:
            current.append(para)
            continue
        joined = "\n".join(current)
        if len(joined) > best_len:
            best, best_len = current, len(joined)
        current = []
    return "\n".join(best) if best else ""


def _clean_inline(text: str) -> str:
    return _WS.sub(" ", _html.unescape(str(text or ""))).strip()


def _clip(text: str) -> str:
    text = str(text or "")
    if len(text) <= MAX_ARTICLE_CHARS:
        return text
    return text[:MAX_ARTICLE_CHARS]


def comment_site(url: str) -> str:
    """按**重定向后的最终 URL** 判评论站点。

    短链（b23.tv / youtu.be）只有跟到终点才知道是哪家，所以入参必须是 resp.url 而不是原始链接。
    """
    host = (urlparse(str(url or "")).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if any(host == s or host.endswith("." + s) for s in _BILIBILI_SUFFIXES):
        return "bilibili"
    if any(host == s or host.endswith("." + s) for s in _YOUTUBE_SUFFIXES):
        return "youtube"
    return ""


def _clean_comment(value) -> str:
    """评论清洗：解实体 + 折叠空白（评论常带换行）→ 单行文本。"""
    return _COMMENT_WS.sub(" ", _html.unescape(str(value or ""))).strip()


def normalize_comments(raw) -> List[Dict[str, object]]:
    """评论 → 统一 `[{text, likes}]`：截断 → 按点赞降序 → 去重 → 封顶。

    先排序再去重：同一条评论被站点以不同 id 重复返回时，留下点赞数最高那条。
    likes 缺失/非数字一律归 0（不会因为站点少个字段就丢掉整批评论）。
    """
    items: List[Dict[str, object]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        text = _clean_comment(entry.get("text"))[:MAX_COMMENT_CHARS]
        if not text:
            continue
        try:
            likes = int(entry.get("likes") or 0)
        except (TypeError, ValueError):
            likes = 0
        items.append({"text": text, "likes": likes})
    items.sort(key=lambda item: int(item["likes"]), reverse=True)
    seen, out = set(), []
    for item in items:
        text = str(item["text"])
        if text in seen:
            continue
        seen.add(text)
        out.append(item)
    return out[:MAX_COMMENTS]


def _wbi_mixin_key(img_key: str, sub_key: str) -> str:
    raw = str(img_key or "") + str(sub_key or "")
    return "".join(raw[i] for i in _WBI_MIXIN_TAB if i < len(raw))[:32]


def _wbi_sign(params: Dict[str, object], mixin_key: str) -> Dict[str, object]:
    signed = dict(params)
    signed["wts"] = int(time.time())
    query = urlencode(sorted(signed.items()))
    # B 站口径：签名前必须剔掉 !'()* 这几个字符，否则服务端校验不过
    clean = "".join(ch for ch in query if ch not in "!'()*")
    signed["w_rid"] = md5((clean + mixin_key).encode("utf-8")).hexdigest()
    return signed


def parse_bilibili_replies(data) -> List[Dict[str, object]]:
    """从 reply 接口的 data 取置顶 + 热评；新老接口的字段名都兼容。"""
    out: List[Dict[str, object]] = []
    if not isinstance(data, dict):
        return out
    for key in ("top_replies", "top", "replies", "hots"):
        group = data.get(key)
        if not isinstance(group, list):
            continue  # `top` 有时是对象（置顶/UP 主），不是评论列表
        for rep in group:
            if not isinstance(rep, dict):
                continue
            content = rep.get("content")
            message = content.get("message") if isinstance(content, dict) else ""
            out.append({"text": message, "likes": rep.get("like") or 0})
    return out


async def _bilibili_comments(client, final_url: str, page: str) -> List[Dict[str, object]]:
    """B 站热评。评论**不在页面 HTML 里**（实测确认），走 XHR JSON 接口。

    oid（= aid）优先从页面 __INITIAL_STATE__ 的 "aid" 取；拿不到再按 av 号 / view 接口兜。
    wbi 签名失败时退到不签名的老接口（实测只回前 3 条热评，但比没有强）。
    """
    aid = 0
    match = _BILI_AID_RE.search(str(page or ""))
    if match:
        aid = int(match.group(1))
    if not aid:
        match = _BILI_AV_RE.search(str(final_url or ""))
        if match:
            aid = int(match.group(1))
    if not aid:
        match = _BILI_BVID_RE.search(str(final_url or ""))
        if match:
            try:
                view = await client.get(_BILI_VIEW, params={"bvid": match.group(1)})
                payload = view.json()
                if int(payload.get("code") or 0) == 0:
                    aid = int(((payload.get("data") or {}).get("aid")) or 0)
            except Exception:
                aid = 0
    if not aid:
        return []

    mixin = ""
    try:
        nav = await client.get(_BILI_NAV)
        wbi = ((nav.json().get("data") or {}).get("wbi_img")) or {}
        img_url = str(wbi.get("img_url") or "")
        sub_url = str(wbi.get("sub_url") or "")
        if img_url and sub_url:
            mixin = _wbi_mixin_key(img_url.rsplit("/", 1)[-1].split(".")[0],
                                   sub_url.rsplit("/", 1)[-1].split(".")[0])
    except Exception:
        mixin = ""

    if mixin:
        params = _wbi_sign({"oid": aid, "type": 1, "mode": 3, "next": 0,
                            "ps": MAX_COMMENTS, "plat": 1}, mixin)
        payload = (await client.get(_BILI_REPLY_WBI, params=params)).json()
        if int(payload.get("code") or 0) == 0:
            return parse_bilibili_replies(payload.get("data") or {})

    payload = (await client.get(
        _BILI_REPLY_LEGACY,
        params={"type": 1, "oid": aid, "sort": 2, "ps": MAX_COMMENTS, "pn": 1})).json()
    if int(payload.get("code") or 0) == 0:
        return parse_bilibili_replies(payload.get("data") or {})
    return []


def _json_object_at(text: str, start: int) -> str:
    """取 text[start:] 里第一个**括号配对完整**的 JSON 对象（手写配对，不引第三方）。"""
    begin = text.find("{", start)
    if begin < 0:
        return ""
    depth, in_string, escaped = 0, False, False
    for index in range(begin, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index + 1]
    return ""


def bilibili_creator_display_name(page: str) -> str:
    """Read the uploader display name from Bilibili's embedded video metadata.

    This is source attribution metadata, not evidence that the uploader spoke
    any particular line in the video.
    """
    raw = str(page or "")
    match = re.search(r"__INITIAL_STATE__\s*=\s*", raw)
    if not match:
        return ""
    encoded = _json_object_at(raw, match.end())
    if not encoded:
        return ""
    try:
        state = json.loads(encoded)
    except (TypeError, ValueError):
        return ""
    if not isinstance(state, dict):
        return ""
    video = state.get("videoData") or state.get("video_data") or {}
    owner = video.get("owner") if isinstance(video, dict) else {}
    name = owner.get("name") if isinstance(owner, dict) else ""
    if not name:
        up_data = state.get("upData") or state.get("up_data") or {}
        name = up_data.get("name") if isinstance(up_data, dict) else ""
    cleaned = _clean_inline(name)
    return cleaned[:80] if cleaned and not re.search(r"[<>\r\n]", cleaned) else ""


def _yt_runs_text(value) -> str:
    """YouTube 文本字段两种形态：simpleText 或 runs[].text。"""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("simpleText"), str):
        return value["simpleText"]
    runs = value.get("runs")
    if isinstance(runs, list):
        return "".join(str(run.get("text") or "") for run in runs if isinstance(run, dict))
    return ""


def _yt_first_token(node) -> str:
    """子树里第一个 continuation token（端点字段名随版本变，两种都认）。"""
    if isinstance(node, dict):
        endpoint = node.get("continuationEndpoint")
        if isinstance(endpoint, dict):
            command = endpoint.get("continuationCommand")
            if isinstance(command, dict) and command.get("token"):
                return str(command["token"])
        legacy = node.get("nextContinuationData")
        if isinstance(legacy, dict) and legacy.get("continuation"):
            return str(legacy["continuation"])
        for value in node.values():
            token = _yt_first_token(value)
            if token:
                return token
    elif isinstance(node, list):
        for value in node:
            token = _yt_first_token(value)
            if token:
                return token
    return ""


def yt_comment_continuation(data) -> str:
    """评论区 token 在 `sectionIdentifier='comment-item-section'` 的子树里。

    按标记递归找而非写死属性路径：YouTube 改版会换层级，标记相对稳定。
    """
    if isinstance(data, dict):
        if data.get("sectionIdentifier") == _YT_COMMENT_SECTION:
            token = _yt_first_token(data)
            if token:
                return token
        for value in data.values():
            token = yt_comment_continuation(value)
            if token:
                return token
    elif isinstance(data, list):
        for value in data:
            token = yt_comment_continuation(value)
            if token:
                return token
    return ""


def _parse_like_count(value) -> int:
    """点赞数文本 → int："1.2K"/"3.4万"/"1,234" 统一成数字，认不出算 0。"""
    text = _yt_runs_text(value).replace(",", "").strip()
    match = _YT_LIKE_RE.match(text)
    if not match:
        return 0
    try:
        number = float(match.group(1))
    except ValueError:
        return 0
    return int(number * {"k": 1000, "m": 1000000, "万": 10000}.get(match.group(2).lower(), 1))


def parse_youtube_comments(payload) -> List[Dict[str, object]]:
    """从 innertube next 响应里递归取 commentRenderer（改版换位置也不漏）。"""
    out: List[Dict[str, object]] = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            renderer = node.get("commentRenderer")
            if isinstance(renderer, dict):
                out.append({"text": _yt_runs_text(renderer.get("contentText")),
                            "likes": _parse_like_count(renderer.get("voteCount"))})
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return out


async def _youtube_comments(client, page: str) -> List[Dict[str, object]]:
    """YouTube 评论：解析 ytInitialData → 找评论 continuation → POST innertube next。"""
    raw = str(page or "")
    match = _YT_INITIAL_DATA_RE.search(raw)
    if not match:
        return []
    data_text = _json_object_at(raw, match.end())
    if not data_text:
        return []
    try:
        initial = json.loads(data_text)
    except Exception:
        return []
    token = yt_comment_continuation(initial)
    if not token:
        return []
    key_match = _YT_API_KEY_RE.search(raw)
    version_match = _YT_CLIENT_VERSION_RE.search(raw)
    if not key_match or not version_match:
        return []
    payload = {"context": {"client": {"clientName": "WEB",
                                      "clientVersion": version_match.group(1),
                                      "hl": "en"}},
               "continuation": token}
    resp = await client.post("https://www.youtube.com/youtubei/v1/next",
                             params={"key": key_match.group(1), "prettyPrint": "false"},
                             json=payload)
    return parse_youtube_comments(resp.json())


async def _fetch_video_comments(final_url: str, page: str, site: str, *,
                                timeout: float = 20) -> List[Dict[str, object]]:
    """视频页热评（增强项，§三·五 修复 1）：**任何失败一律吞成 []**。

    评论是第二/三次网络往返，不许影响主合同：网络断、站点改版、被风控、超时
    —— 调用方拿到的都是"这次没评论"，而不是"这次抓取失败"。
    """
    if site not in ("bilibili", "youtube"):
        return []
    try:
        import httpx  # 懒导入：与 fetch_url 同一理由（冷路径，缺依赖不拖垮启动）
    except Exception:
        return []
    try:
        headers = {"User-Agent": _BROWSER_UA,
                   "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                   "Referer": "https://www.bilibili.com/" if site == "bilibili"
                              else "https://www.youtube.com/"}
        budget = min(float(timeout or COMMENT_TIMEOUT), float(COMMENT_TIMEOUT))
        async with httpx.AsyncClient(follow_redirects=True, timeout=budget,
                                     headers=headers) as client:
            if site == "bilibili":
                raw = await _bilibili_comments(client, final_url, page)
            else:
                raw = await _youtube_comments(client, page)
    except Exception:
        return []
    return normalize_comments(raw)


async def fetch_url(url: str, *, timeout: int = 20) -> Dict[str, object]:
    """抓一个链接 → {ok, kind, url, title, text, comments, error}。

    ok=False 时 `error` 一定是**可以直接给用户看的中文原因**（§7-1）：
    调用方把它落进 material_items.error 并原样回给前端，不要自己另编一套话术。

    `comments` 是视频页热评（`[{text, likes}]`，非视频页恒为 []）：**增强项**，
    任何失败都只是空列表，绝不改变 ok / error 的判定（§三·五 修复 1）。
    """
    target = str(url or "").strip()
    if not target:
        return {"ok": False, "kind": "link", "url": "", "title": "", "text": "",
                "comments": [], "error": "链接是空的"}
    if not re.match(r"^https?://", target, re.IGNORECASE):
        target = "https://" + target

    video = is_video_url(target)
    try:
        import httpx  # 懒导入：本模块属于冷路径，缺依赖不该拖垮 app 启动
    except Exception:
        return {"ok": False, "kind": "link", "url": target, "title": "", "text": "",
                "comments": [], "error": "服务端缺少抓取依赖（httpx），请先安装后再试"}

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=float(timeout),
            headers={"User-Agent": _BROWSER_UA,
                     "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        ) as client:
            resp = await client.get(target)
    except Exception as exc:
        # 网络层失败：说清楚是什么错，别让用户以为是自己的问题
        return {"ok": False, "kind": "link", "url": target, "title": "", "text": "",
                "comments": [],
                "error": "抓取失败（网络不通或站点拒绝连接）：%s" % type(exc).__name__}

    if resp.status_code >= 400:
        return {"ok": False, "kind": "link", "url": target, "title": "", "text": "",
                "comments": [], "error": "这个链接打不开（HTTP %d）" % resp.status_code}

    page = resp.text or ""
    title, body = strip_html(page)
    desc = meta_description(page)

    if video:
        # 视频页：真内容在音轨里。2026-09-21 起**先试着把音轨转成文字**
        # （plans/0916-21 承诺的"v1.5 转写"）：转写成功 → 视频里讲的内容整段进素材；
        # 失败 → 降级到"轻素材"（标题+简介），并把**失败原因如实说给用户**
        # （用户才知道是"这个视频不行"还是"这条能力没开"），不静默成空带（§1 渠道矩阵 / §7-1）。
        #
        # ⚠ 转写按音频时长实时跑（实测倍率 1.01~1.05×），首版上限 35 分钟
        # ⇒ 这个请求最坏会挂约 6 分钟。上限与并发见 engine/media_transcribe.py。
        if is_placeholder_page(title, desc):
            return {"ok": False, "kind": "link", "url": target, "title": title,
                    "text": "", "comments": [],
                    "error": "这个视频页是空占位页（视频可能已被删除或设为私有），"
                             "换一个链接试试"}
        # 评论按**重定向后的最终 URL** 分站点（b23.tv / youtu.be 短链跟到终点才认得出）
        final_url = str(getattr(resp, "url", "") or target)
        creator_display_name = (
            bilibili_creator_display_name(page)
            if comment_site(final_url) == "bilibili" else ""
        )
        comments = await _fetch_video_comments(final_url, page,
                                               comment_site(final_url),
                                               timeout=timeout)
        from engine import media_transcribe
        spoken = await media_transcribe.transcribe_url(final_url)
        if spoken.get("ok") and str(spoken.get("text") or "").strip():
            return {"ok": True, "kind": "video", "url": target,
                    "title": title or str(spoken.get("title") or ""),
                    "text": _clip(str(spoken.get("text") or "")),
                    "segments": list(spoken.get("segments") or []),
                    "comments": comments, "error": "",
                    "creator_display_name": creator_display_name,
                    "creator_display_name_source": (
                        "bilibili_page_metadata" if creator_display_name else ""),
                    "note": str(spoken.get("note") or "已从视频音轨转写全文")}
        # 转写没成 → 退回轻素材。**原因必须带上**：这句话是用户唯一能看到的解释。
        page_desc = _video_page_description(desc)
        light = "\n".join(x for x in (title, page_desc) if x).strip()
        reason = str(spoken.get("error") or "").strip()
        if len(light) < 10:
            return {"ok": False, "kind": "link", "url": target, "title": title,
                    "text": "", "comments": [],
                    "error": reason or "这个视频页只抓到了标题，连简介都没有；"
                                       "暂时可以复制文字简介粘贴进来"}
        return {"ok": True, "kind": "video_light", "url": target, "title": title,
                "text": _clip(light), "comments": comments, "error": "",
                "creator_display_name": creator_display_name,
                "creator_display_name_source": (
                    "bilibili_page_metadata" if creator_display_name else ""),
                "note": ("只取到标题与简介；视频内容没转出来（%s）" % reason) if reason
                        else "只取到标题与简介；视频内容没转出来"}

    content = longest_text_block(body)
    if len(content) < MIN_ARTICLE_CHARS:
        content = body
    if len(content.strip()) < MIN_ARTICLE_CHARS:
        return {"ok": False, "kind": "link", "url": target, "title": title, "text": "",
                "comments": [],
                "error": "这个链接抓不到正文（可能是需要登录的封闭平台、"
                         "或有反爬），可以复制正文粘贴进来"}
    return {"ok": True, "kind": "article", "url": target, "title": title,
            "text": _clip(content), "comments": [], "error": ""}
