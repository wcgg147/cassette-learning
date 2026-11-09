# -*- coding: utf-8 -*-
"""本地设备文件的**确定性解析**（plans/0916-24 §一/§四 V1.2）。

分工（方案 §一 的结论，别混）：
- 文档（txt/md/pdf/docx/epub）→ **纯文本抽取，不花模型调用**——就是本文件；
- 图片 → 要"视觉"，走 `material_analyzer.analyze_image()` 的多模态入口；
- 视频 → 要离线 ASR，本方案不做（v1.5）。

底线（方案 §7-1，这项目废过两次的坑）：解不出来就返回**明确中文原因**，
绝不产出空素材带。依赖缺失也走这条：给"后端缺解析依赖（xxx）"这种可执行原因，
而不是让 ImportError 冒到 500。

解析库全部**懒导入**：只有用户真的导入这类文件时才需要它们，语音轮次路径
（每轮 1.5s 预算）不该为它们付 import 成本。
"""

from __future__ import annotations

import io
import re
from typing import Dict

# 统一上限：本地文件与将来的云盘拉取走同一条线（方案 §五 也只拉小文件）。
MAX_FILE_BYTES = 20 * 1024 * 1024
# 入库正文上限：一份 500 页 PDF 抽出来几百万字，会把库、分块和注入预算全撑爆。
# 截断处留一句明确说明，用户看得见"后面没读"。
MAX_TEXT_CHARS = 200_000

_TEXT_EXTS = ("txt", "md", "markdown", "text", "log", "csv", "json")
_IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "bmp", "gif", "heic")
_DOC_EXTS = ("pdf", "docx", "epub")

# 解析器的额外 MIME（浏览器/安卓给的 content-type 不可靠，扩展名优先，MIME 兜底）
_MIME_KIND = {
    "application/pdf": "pdf",
    "application/epub+zip": "epub",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "text",
    "text/markdown": "text",
    "text/csv": "text",
    "application/json": "text",
}


def kind_of(filename: str = "", content_type: str = "") -> str:
    """判类型：扩展名优先（用户手里的文件名比 MIME 可信），MIME 兜底。

    返回 'text'|'pdf'|'docx'|'epub'|'image'|''（空 = 不支持，调用方报错）。
    """
    name = str(filename or "").strip().lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext in _TEXT_EXTS:
        return "text"
    if ext in _DOC_EXTS:
        return ext
    if ext in _IMAGE_EXTS:
        return "image"
    mime = str(content_type or "").split(";")[0].strip().lower()
    if mime in _MIME_KIND:
        return _MIME_KIND[mime]
    if mime.startswith("image/"):
        return "image"
    return ""


def _decode(data: bytes) -> str:
    """文本解码：UTF-8（带/不带 BOM）→ GBK → 容错替换。

    中文用户的 txt 里 GBK 很常见；最后一级用 replace 而不是报错——乱码里
    夹着可读正文，比"整份导入失败"对用户有用得多。
    """
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _from_pdf(data: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        # 空密码能解开的就不算"加密"（很多导出工具只是加了权限位）
        try:
            if reader.decrypt("") == 0:
                raise ValueError("encrypted")
        except Exception:
            raise ValueError("encrypted")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")  # 单页崩掉不弃整份（坏页常见于扫描混排的 PDF）
    return "\n\n".join(pages)


def _from_docx(data: bytes) -> str:
    import docx
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if str(p.text or "").strip()]
    # 表格里的内容对"素材"常常才是重点（菜单、字幕表、对照表）
    for table in document.tables:
        for row in table.rows:
            cells = [str(c.text or "").strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _html_to_text(raw: str) -> str:
    text = _SCRIPT_RE.sub(" ", str(raw or ""))
    text = re.sub(r"<(br|/p|/div|/li|/h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    from html import unescape
    text = unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def _from_epub(data: bytes) -> str:
    import ebooklib
    from ebooklib import epub
    book = epub.read_epub(io.BytesIO(data))
    chunks = []
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            chunks.append(_html_to_text(item.get_content().decode("utf-8", errors="replace")))
        except Exception:
            continue
    return "\n\n".join(c for c in chunks if c.strip())


def extract_text(data: bytes, filename: str = "", content_type: str = "") -> Dict:
    """文档 → 正文。返回 {ok, kind, text, note, error}；error 是给用户看的中文原因。

    不返回空 text 的 ok=True——"解析成功但一个字都没有"对用户和模型都没有意义，
    按 §7-1 一律算失败并说清原因（扫描件要引导用户改走图片导入）。
    """
    kind = kind_of(filename, content_type)
    if kind in ("", "image"):
        # image 不是"解析失败"，是"不走这条路"——由调用方分派到多模态入口
        return {"ok": False, "kind": kind, "text": "", "note": "",
                "error": "这种文件暂时读不了（支持 txt/md/pdf/docx/epub 与图片）"}
    size = len(data or b"")
    if size == 0:
        return {"ok": False, "kind": kind, "text": "", "note": "",
                "error": "这个文件是空的"}
    if size > MAX_FILE_BYTES:
        return {"ok": False, "kind": kind, "text": "", "note": "",
                "error": "文件太大（上限 %dMB），先截取一部分再导入"
                         % (MAX_FILE_BYTES // (1024 * 1024))}
    note = ""
    try:
        if kind == "text":
            text = _decode(data)
        elif kind == "pdf":
            text = _from_pdf(data)
        elif kind == "docx":
            text = _from_docx(data)
        else:
            text = _from_epub(data)
    except ValueError as exc:
        if "encrypted" in str(exc):
            return {"ok": False, "kind": kind, "text": "", "note": "",
                    "error": "这份 PDF 有加密保护，先去掉密码再导入"}
        return {"ok": False, "kind": kind, "text": "", "note": "",
                "error": "这份文件读不动：%s" % str(exc)[:80]}
    except ImportError as exc:
        return {"ok": False, "kind": kind, "text": "", "note": "",
                "error": "后端缺解析依赖（%s），装上再试" % type(exc).__name__ }
    except Exception as exc:
        # docx/epub 的"格式不对"异常类型各家不同（PackageNotFoundError/
        # BadZipFile/KeyError），一律收成一句可读原因，绝不 500 也不产空带。
        return {"ok": False, "kind": kind, "text": "", "note": "",
                "error": "这个 .%s 打不开（可能不是真正的 %s 文件，或已损坏）"
                         % (kind, kind.upper())}
    text = _normalize(text)
    if not text:
        hint = {"pdf": "这份 PDF 抽不出文字（扫描件？）——截图后按图片导入能读",
                "docx": "这个 Word 里没有文字内容",
                "epub": "这本电子书里没有正文（可能只有封面/图片）"}.get(kind, "这个文件里没有可读的文字")
        return {"ok": False, "kind": kind, "text": "", "note": "", "error": hint}
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        note = "内容过长，只读了前 %d 字" % MAX_TEXT_CHARS
    return {"ok": True, "kind": kind, "text": text, "note": note, "error": ""}


def _normalize(text: str) -> str:
    """压掉解析残留的空白噪音（PDF 抽出的一堆空行会把分块切得稀碎）。"""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(text or "").splitlines()]
    out = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip()
