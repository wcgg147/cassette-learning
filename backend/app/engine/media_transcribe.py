# -*- coding: utf-8 -*-
"""视频/音频链接 → 文字（2026-09-21；plans/0916-21 里承诺的"v1.5 转写"）。

链路：**yt-dlp 下载音轨 → （切段 + 并发 POST 语音代理）或（整段交批量 ASR）→ 拼接**。

⚠ **为什么现役通道必须切段**：代理那条转写走的是**流式对话端点**
（`wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue`），按**实时速度**跑，
且有 **6 分钟**硬上限（`TRANSCRIBE_MAX_PCM_BYTES`——**我们自己设的常量**，不是服务商限制）。
2026-09-21 实测：整段 6.8 分钟音频被拒；切成 2×205 秒并行转写成功，实时倍率 **1.01~1.05×**；
**6 路并发全部成功**（墙钟 31.5 秒 / 6×30 秒音频）。⇒ 现役上限就是"8 段 × 350 秒 ≈ 45 分钟"。

⚠ **批量通道（2026-09-21 接入）**：火山额度不够后换的**阿里 DashScope 非实时语音识别**
（`engine/asr_filetrans.py`）。**不受实时约束** ⇒ 不切段、不并发，处理倍率实测
0.026×（57 秒音频 1.5 秒出结果）⇒ 上限放到 120 分钟。由 `ASR_TRANSCRIBE_PROVIDER`
开关切（默认仍走现役代理，**先在真实视频上对拍质量再切**，记录见 `bugreports/0921-02`）。

⚠ **拿不到的东西**：只存在于**画面**里的信息（字幕条、图表、配方文字、纯视觉演示）。
音频覆盖的是"视频里说的每一句话"——对有口播的视频即"全部内容"，
但画面文字要另做视觉抽帧（未做）。实测记录见 `bugreports/0921-01`。
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import tempfile
from typing import Dict, List, Optional

# ── 可调旋钮（全部可用环境变量覆盖，换部署形态不用改代码）──────────────
PROXY_ENV = "VOICE_PROXY_BASE"           # 语音代理地址（转写端点在它上面）
DEFAULT_PROXY = "http://127.0.0.1:3211"
YTDLP_ENV = "YTDLP_BIN"                  # 空则用 `python -m yt_dlp`
FFMPEG_ENV = "FFMPEG_BIN"                # 空则用 PATH 上的 ffmpeg

# 代理上限是 6 分钟（360s），留 10 秒余量——贴边容易被它的转码检查拒掉
CHUNK_SECONDS = 350
# 实测：6 路全成功、**10 路也全成功**（10×30 秒音频，墙钟 31.8 秒，实时倍率 1.06×）。
# 取 8 而不是 10：代理是单进程，每路要缓冲 350 秒的 PCM（约 11MB）⇒ 10 路约 110MB
# 额外内存，而它同时还在跑语音对话。8 路留出余量，仍够"一集剧一波并发"。
MAX_CONCURRENCY = 8
# 上限 = 8 段 × 350 秒 ≈ 45 分钟，正好**一波并发**跑完（墙钟≈350 秒≈6 分钟，
# 与视频多长无关——这是切段策略的关键好处）。再长就要排第二波，同步 HTTP 太久
# ⇒ 等换批量 ASR（见文件头）再放开。
MAX_MEDIA_SECONDS = 45 * 60
# 音轨选档：转写够用就行，低码率下载更快。B站 30216=66k / 30232=100k / 30280=183k
AUDIO_FORMAT = "bestaudio[abr<=128]/bestaudio/best"
DOWNLOAD_TIMEOUT = 300                   # yt-dlp 单次调用上限（下载 20MB 实测 <1 秒）
TRANSCRIBE_TIMEOUT = 600                 # 单段转写上限（350 秒音频实测约 350 秒）

# ── 转写通道（2026-09-21）：默认仍是**现役代理流式**，批量挂在开关后面 ────────
# proxy = 现役：代理的流式对话端点，按实时速度跑 ⇒ 必须切段 + 并发（见文件头）。
# dashscope_filetrans = 批量（`engine/asr_filetrans.py`）：不受实时约束 ⇒ 不切段、
#   不并发、上限更宽。**默认不切** —— 先拿真实视频对拍过质量再切（bugreports/0921-02）。
PROVIDER_ENV = "ASR_TRANSCRIBE_PROVIDER"
PROVIDER_PROXY = "proxy"
PROVIDER_BATCH = "dashscope_filetrans"
# 批量通道的时长护栏：**自设**（官方 Audio specifications 的时长上限尚未核，待补），
# 先按"两小时"放行——够覆盖一集剧。
MAX_MEDIA_SECONDS_BATCH = 120 * 60
PROBE_TIMEOUT = 60

# ── 静音门限（**与现役代理同口径**）────────────────────────────────────
# 现役代理对解码后的 16bit PCM 整段算 RMS，低于 40 就明确报"静音"
# （`deployment/doubao-session-server.mjs` 的 `TRANSCRIBE_SILENCE_RMS = 40`，注释写 ≈ -58dBFS）。
# 20*log10(40/32768) ≈ -58.3 dBFS —— 同一个门限在两条通道上必须是同一个数。
# ⚠ 为什么批量通道也必须有：实测算出**近乎静音时模型不返回空、而是编一句**
#   （57 秒底噪 → 返回 `はい。`，见 bugreports/0921-02 §四）⇒ 少了这道闸门，
#   贴一个没说话的链接会得到一盘"AI 编的正文"，而现役会明确报"没有识别到语音"。
SILENCE_DBFS = -58.0


def _proxy_base() -> str:
    return os.getenv(PROXY_ENV, "").strip() or DEFAULT_PROXY


def _provider() -> str:
    """当前转写通道。认不出的值一律退回现役代理（**别让一个错字改掉线上行为**）。"""
    value = os.getenv(PROVIDER_ENV, "").strip().lower()
    return value if value == PROVIDER_BATCH else PROVIDER_PROXY


def _ytdlp_cmd() -> List[str]:
    """yt-dlp 的调用方式：优先环境变量给的二进制，否则用当前解释器的模块。

    用 `python -m yt_dlp` 而不是 `yt-dlp` 命令：服务是常驻进程，**不继承后来才
    装好的 PATH**（`deployment/.env` 里 FFMPEG_BIN 的注释记过同一个坑）。
    """
    pinned = os.getenv(YTDLP_ENV, "").strip()
    if pinned:
        return [pinned]
    return [sys.executable, "-m", "yt_dlp"]


def _ffmpeg() -> Optional[str]:
    pinned = os.getenv(FFMPEG_ENV, "").strip()
    if pinned and os.path.exists(pinned):
        return pinned
    return shutil.which("ffmpeg")


async def _run(cmd: List[str], timeout: float) -> Dict:
    """跑一个外部命令。**用 asyncio 子进程**——导入端点是 async 的，
    同步 subprocess 会把事件循环卡住几秒到几分钟。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        return {"ok": False, "out": "", "err": "命令不存在：%s" % cmd[0]}
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"ok": False, "out": "", "err": "超时（%.0f 秒）" % timeout}
    return {"ok": proc.returncode == 0,
            "out": (out or b"").decode("utf-8", "replace"),
            "err": (err or b"").decode("utf-8", "replace")}


async def probe(url: str) -> Dict:
    """读元数据（不下载）。返回 {ok, title, duration, error}。"""
    got = await _run(
        _ytdlp_cmd() + ["--no-warnings", "--skip-download",
                        "--print", "%(title)s\t%(duration)s", url],
        timeout=PROBE_TIMEOUT)
    if not got["ok"]:
        return {"ok": False, "title": "", "duration": 0,
                "error": _explain(got["err"])}
    line = (got["out"] or "").strip().splitlines()[-1] if got["out"].strip() else ""
    title, _, duration = line.partition("\t")
    try:
        seconds = int(float(duration))
    except (TypeError, ValueError):
        seconds = 0
    return {"ok": True, "title": title.strip(), "duration": seconds, "error": ""}


def _explain(stderr: str) -> str:
    """把 yt-dlp 的英文报错翻成**用户看得懂的中文原因**（不静默、不吞）。"""
    text = str(stderr or "")
    if "not found" in text or "No module named" in text:
        return "服务端缺少下载工具（yt-dlp），请先安装后再试"
    if "timed out" in text or "Timeout" in text:
        return "连接视频站超时；可以稍后重试，或换一个可访问的视频链接"
    if "Video unavailable" in text or "not available" in text:
        return "这个视频取不到（可能已删除、设为私有，或需要登录）"
    if "Unsupported URL" in text:
        return "这个链接不是可下载的视频页"
    return "取视频信息失败：%s" % (text.strip().splitlines()[-1][:160]
                                   if text.strip() else "未知原因")


async def _download_audio(url: str, workdir: str) -> Dict:
    """下音轨到 workdir。返回 {ok, path, error}。"""
    template = os.path.join(workdir, "audio.%(ext)s")
    got = await _run(
        _ytdlp_cmd() + ["--no-warnings", "--no-playlist",
                        "-f", AUDIO_FORMAT, "-o", template, url],
        timeout=DOWNLOAD_TIMEOUT)
    if not got["ok"]:
        return {"ok": False, "path": "", "error": _explain(got["err"])}
    for name in sorted(os.listdir(workdir)):
        if name.startswith("audio."):
            return {"ok": True, "path": os.path.join(workdir, name), "error": ""}
    return {"ok": False, "path": "", "error": "下载完成了但没找到音轨文件"}


async def _split(path: str, workdir: str) -> Dict:
    """按 CHUNK_SECONDS 切成若干段。返回 {ok, chunks, error}。

    `-f segment` 一次调用切完（比我之前逐段 `-ss/-t` 少 N 次进程启动）；
    `-reset_timestamps 1` 让每段从 0 开始，避免部分解码器按绝对时间戳判长度。
    """
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return {"ok": False, "chunks": [],
                "error": "服务端缺少 ffmpeg，无法切分音轨（装好或设 FFMPEG_BIN）"}
    pattern = os.path.join(workdir, "chunk_%03d" + os.path.splitext(path)[1])
    got = await _run(
        [ffmpeg, "-v", "error", "-i", path,
         "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
         "-reset_timestamps", "1", "-c", "copy", pattern],
        timeout=DOWNLOAD_TIMEOUT)
    if not got["ok"]:
        return {"ok": False, "chunks": [],
                "error": "音轨切分失败：%s" % (got["err"].strip()[:160] or "未知原因")}
    chunks = sorted(
        os.path.join(workdir, name) for name in os.listdir(workdir)
        if name.startswith("chunk_"))
    if not chunks:
        return {"ok": False, "chunks": [], "error": "音轨切分后没有产出任何片段"}
    return {"ok": True, "chunks": chunks, "error": ""}


async def _mean_volume_dbfs(path: str) -> Optional[float]:
    """整段平均音量（dBFS）。返回 None = **拿不到**（没 ffmpeg / 解码失败）。

    用 ffmpeg 的 `volumedetect` **流式**扫全段（不把 120 分钟 PCM 读进内存，
    这是它与现役"整段进 Buffer"的差别——批量通道上限 120 分钟，全进内存是 230MB）。
    口径与现役一致：都是整段平均值。

    ⚠ `None` 与"-58dBFS"是两件事，调用方**绝不能把"没测到"当成"静音"**。
    """
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return None
    got = await _run([ffmpeg, "-v", "info", "-i", path,
                      "-af", "volumedetect", "-f", "null", "-"],
                     timeout=DOWNLOAD_TIMEOUT)
    # volumedetect 把报告打到 **stderr**；即使 ffmpeg 退出码非 0 也可能已打出
    # ⇒ 只认报告，不看 got["ok"]。
    text = (got.get("err") or "") + "\n" + (got.get("out") or "")
    found = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", text)
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


async def _transcribe_one(client, path: str, sem: asyncio.Semaphore) -> Dict:
    """转写一段。走代理原生的 /api/transcribe（multipart 字段名必须是 file）。"""
    async with sem:
        try:
            with open(path, "rb") as fh:
                resp = await client.post(
                    _proxy_base().rstrip("/") + "/api/transcribe",
                    files={"file": (os.path.basename(path), fh, "audio/mp4")},
                    timeout=TRANSCRIBE_TIMEOUT)
            data = resp.json()
        except Exception as exc:
            return {"ok": False, "text": "",
                    "error": "转写请求失败（%s）" % type(exc).__name__}
    if not data.get("ok"):
        return {"ok": False, "text": "", "error": str(data.get("error") or "转写失败")}
    return {"ok": True, "text": str(data.get("text") or ""), "error": ""}


async def _transcribe_chunks(paths: List[str]) -> Dict:
    """并发转写所有片段，按原顺序拼接。返回 {ok, text, error, failed}。

    **并行而不是串行**：每段按实时速度跑，串行等于把视频时长原样等一遍
    （一集 40 分钟要等 40 分钟）；并发后墙钟≈单段时长。
    """
    try:
        import httpx  # 懒导入：本模块属冷路径，缺依赖不该拖垮 app 启动
    except Exception:
        return {"ok": False, "text": "", "failed": 0,
                "error": "服务端缺少抓取依赖（httpx），请先安装后再试"}
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    # trust_env=False：本机 env 里挂着的 HTTP(S)_PROXY 会把 localhost 请求也绕走
    # （bugreports/0917-05 记过这个坑，表现为"连上了但拿到的是代理自己的页面"）
    async with httpx.AsyncClient(trust_env=False) as client:
        results = await asyncio.gather(
            *[_transcribe_one(client, path, sem) for path in paths])
    failed = sum(1 for item in results if not item["ok"])
    # 空白文本按"没拿到"处理：`ok` 的含义是"这一段有可用文本"，
    # 否则纯空白的返回会让上层以为成功、产出一盘没有正文的素材带。
    texts = [item["text"].strip() for item in results
             if item["ok"] and item["text"].strip()]
    if not texts:
        first = next((item["error"] for item in results if item["error"]), "")
        return {"ok": False, "text": "", "failed": failed,
                "error": first or "转写结果为空：这段音轨里没有识别到语音"}
    if failed:
        # 部分失败仍给结果（素材不完整好过没有），但把缺口如实说出来
        return {"ok": True, "text": "\n".join(texts), "failed": failed, "error": ""}
    return {"ok": True, "text": "\n".join(texts), "failed": 0, "error": ""}


async def transcribe_url(url: str, *, max_seconds: Optional[int] = None) -> Dict:
    """视频链接 → 转写全文。返回 {ok, text, title, duration, note, error}。

    **绝不抛**：所有失败都收成 `ok=False` + 明确中文原因，让调用方能降级到
    "轻素材"（标题+简介+评论）而不是整条导入失败。

    **两条通道，按 `_provider()` 分流**（默认现役代理，不切行为）：
    · proxy：切段 + 并发（见文件头），上限 `MAX_MEDIA_SECONDS`（45 分钟）；
    · batch：整段一次交给 `asr_filetrans`，不切段不并发，上限
      `MAX_MEDIA_SECONDS_BATCH`（120 分钟）。
    `max_seconds` 不传就按当前通道取对应的上限——**上限跟着通道走**，
    免得切了批量还卡在 45 分钟。
    """
    provider = _provider()
    if max_seconds is None:
        max_seconds = (MAX_MEDIA_SECONDS_BATCH if provider == PROVIDER_BATCH
                       else MAX_MEDIA_SECONDS)
    info = await probe(url)
    if not info["ok"]:
        return {"ok": False, "text": "", "title": "", "duration": 0,
                "note": "", "error": info["error"]}
    duration = int(info.get("duration") or 0)
    if duration and duration > max_seconds:
        return {"ok": False, "text": "", "title": info["title"], "duration": duration,
                "note": "", "error":
                "这个视频太长（%.0f 分钟），当前一次最多处理 %.0f 分钟；"
                "可以先收藏短一点的版本，或等长音频转写通道上线"
                % (duration / 60.0, max_seconds / 60.0)}

    with tempfile.TemporaryDirectory(prefix="zz-media-") as workdir:
        got = await _download_audio(url, workdir)
        if not got["ok"]:
            return {"ok": False, "text": "", "title": info["title"],
                    "duration": duration, "note": "", "error": got["error"]}
        if provider == PROVIDER_BATCH:
            # 上传前先按**现役同一口径**挡掉静音。批量不切段 ⇒ 没有"段"这个粒度，
            # 只能看整段平均——所以门限用的是 dBFS（同一条线，换成整段表达的写法）。
            # ⚠ 拿不到音量时**不拦**：宁可让模型看一眼，也不能把"没测到"当"静音"。
            dbfs = await _mean_volume_dbfs(got["path"])
            if dbfs is not None and dbfs < SILENCE_DBFS:
                return {"ok": False, "text": "", "title": info["title"],
                        "duration": duration, "note": "", "error":
                        "这段音轨里没有可识别的语音（整段平均音量 %.0f dBFS）："
                        "静音或只有背景音，换一个有人讲话的视频试试" % dbfs}
            # 整段传，不切段。懒导入，理由同 `_transcribe_chunks` 的 httpx。
            from engine import asr_filetrans
            done = await asr_filetrans.transcribe_file(got["path"])
            note = "已从视频音轨转写全文（批量）"
        else:
            split = await _split(got["path"], workdir)
            if not split["ok"]:
                return {"ok": False, "text": "", "title": info["title"],
                        "duration": duration, "note": "", "error": split["error"]}
            done = await _transcribe_chunks(split["chunks"])
            note = "已从视频音轨转写全文（%d 段）" % len(split["chunks"])

    text = str(done.get("text") or "").strip()
    if not done["ok"] or not text:
        return {"ok": False, "text": "", "title": info["title"], "duration": duration,
                "note": "", "error": done.get("error") or "转写结果为空"}
    if done.get("failed"):
        note += "；其中 %d 段转写失败，内容有缺口" % done["failed"]
    return {"ok": True, "text": text, "title": info["title"],
            "duration": duration, "note": note, "error": "",
            # Batch filetrans can return timestamped speaker segments. Keep
            # them alongside the backward-compatible flat transcript so the
            # material importer can attribute voices instead of losing them.
            "segments": list(done.get("segments") or [])}
