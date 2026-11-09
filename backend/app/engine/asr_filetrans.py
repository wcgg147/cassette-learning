# -*- coding: utf-8 -*-
"""阿里 DashScope「非实时语音识别」——视频音轨**批量**转写（2026-09-21 实测接入）。

## 为什么要有这条

现役的视频音轨转写走**语音代理的流式对话端点**，按**实时速度**跑（实测倍率
1.01~1.06×，那是它作为实时对话端点的固有属性）⇒ 必须切段（350 秒）+ 8 路并发
才能把 40 分钟压到 ~6 分钟墙钟，且有 45 分钟自设上限。
批量这条路**不受实时约束** ⇒ 不需要切段、不需要并发，上限也能放开。

⚠ **不是默认路径**：由 `media_transcribe.PROVIDER_ENV` 切，先在真实视频上对拍过
质量再切过去（实测记录见 `bugreports/0921-02`）。

## 实测（2026-09-21，真实录音 3 条，全部 SUCCEEDED）

| 音频 | 服务端处理（submit→end） | 倍率 |
|---|---|---|
| 0.7 秒 | 0.4 秒 | 0.60× |
| 5.2 秒 | 0.5 秒 | 0.09× |
| 57 秒 | **1.5 秒** | **0.026×** |

对照现役流式 1.01~1.06× 实时。⇒ 一集 40 分钟音轨，现役要 ~40 分钟/段（8 路并发
后墙钟 ≈6 分钟），批量按此倍率是**秒级~分钟级**。

## 官方契约（已核，两处硬约束）

1. **`file_urls` 必须是公网可访问的 URL**，而我们的音轨是本地临时文件 ⇒ 必须先走
   官方「上传本地文件换临时 URL」：`GET /api/v1/uploads?action=getPolicy&model=…`
   拿 OSS 直传凭证 → POST 到 `upload_host` → 得到 `oss://…` → 提交时带
   `X-DashScope-OssResourceResolve: enable`。
   · 临时存储**仅中国（北京）region**可用，key 必须是该 region 的；
   · 文件 ≤ 1GB；临时 URL **48 小时**过期；官方原文："仅供开发测试，不建议生产
     或高并发"（生产应走自建 OSS）。
2. **结果不在提交响应里**：提交只给 `task_id`，轮询 `GET /api/v1/tasks/{task_id}`，
   成功后从 `output.results[].transcription_url` 再下一个 JSON（那个 URL
   **24 小时**有效）。

## 两个实测踩到的坑（写在这里，别重踩）

⚠ **近乎静音会给幻觉**：一条 57 秒、RMS≈160 的静音录音，模型返回 `はい。` ——
**空音频不返回空，而是编一句**（还串了语言）。上层要么在太短/太静时直接判
"没有识别到语音"，要么在结果侧认出这种短得离谱的输出。**不要把纯底噪送去问模型。**

⚠ **串语言**：同厂实时 `qwen3-asr-flash` 曾被硬否决（中英混说时**翻译**而不是
忠实转写）。本模型实测中文短句是忠实录写的，但默认仍带上 `language_hints`
（本项目用户就是说中文夹英文），少给模型"自己挑语言"的空间。
⇒ 切到这条路之前**必须先在真机上对拍一次**质量。

## 盒子

**换模型/换网关只改本文件**（照 `material_analyzer` 的做法）：抓取、切段、注入
都不得出现模型名。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_ENV = "DASHSCOPE_BASE_URL"
DEFAULT_BASE = "https://dashscope.aliyuncs.com"
KEY_ENV = "DASHSCOPE_API_KEY"
MODEL_ENV = "ASR_FILETRANS_MODEL"
DEFAULT_MODEL = "qwen-audio-3.0-asr-flash-filetrans"
HINTS_ENV = "ASR_FILETRANS_LANG_HINTS"
DIARIZATION_ENV = "ASR_FILETRANS_DIARIZATION"

SUBMIT_PATH = "/api/v1/services/audio/asr/transcription"
TASK_PATH = "/api/v1/tasks/%s"
POLICY_PATH = "/api/v1/uploads"

SUBMIT_TIMEOUT = 60.0
UPLOAD_TIMEOUT = 600.0
RESULT_TIMEOUT = 120.0
POLL_INTERVAL = 2.0
POLL_BUDGET = 900.0        # 单任务轮询总预算（秒）
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024   # 官方 ≤1GB，超了连上传都别试


def _base() -> str:
    return os.getenv(BASE_ENV, "").strip().rstrip("/") or DEFAULT_BASE


def _model() -> str:
    return os.getenv(MODEL_ENV, "").strip() or DEFAULT_MODEL


def _key() -> str:
    return os.getenv(KEY_ENV, "").strip()


def _lang_hints() -> List[str]:
    """默认 `zh,en`（本项目用户说中文夹英文）；设成 `-` 表示不传这个参数。"""
    raw = os.getenv(HINTS_ENV, "").strip()
    if raw == "-":
        return []
    if not raw:
        return ["zh", "en"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _diarization_enabled() -> bool:
    """保留说话人分离结果；用户可通过环境变量显式关闭。"""
    raw = os.getenv(DIARIZATION_ENV, "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _fail(reason: str) -> Dict[str, Any]:
    return {"ok": False, "text": "", "error": reason}


async def _client(timeout: float):
    """外部请求要用 env 里的代理（本机挂着的代理是**通外网**的那条）；
    `trust_env=True` 是默认值，这里显式写出来，免得后人照抄 `_transcribe_one`
    的 `trust_env=False`（那条是打 localhost，口径不同）。"""
    import httpx
    return httpx.AsyncClient(trust_env=True, timeout=timeout)


def _policy_fields(policy: Dict[str, Any], filename: str) -> Tuple[str, str, Dict[str, str]]:
    """getPolicy 的返回 → (上传地址, oss key, 表单字段)。字段名与顺序照官方示例。"""
    key = "%s/%s" % (str(policy.get("upload_dir") or "").rstrip("/"), filename)
    fields = {
        "OSSAccessKeyId": str(policy.get("oss_access_key_id") or ""),
        "Signature": str(policy.get("signature") or ""),
        "policy": str(policy.get("policy") or ""),
        "x-oss-object-acl": str(policy.get("x_oss_object_acl") or ""),
        "x-oss-forbid-overwrite": str(policy.get("x_oss_forbid_overwrite") or ""),
        "key": key,
        "success_action_status": "200",
    }
    return str(policy.get("upload_host") or ""), key, fields


async def _upload(client, path: Path) -> Dict[str, Any]:
    """本地文件 → `oss://…`。返回 {ok, url, error}。"""
    try:
        resp = await client.get(
            _base() + POLICY_PATH,
            params={"action": "getPolicy", "model": _model()},
            headers={"Authorization": "Bearer " + _key()},
            timeout=SUBMIT_TIMEOUT)
        payload = resp.json() or {}
    except Exception as exc:
        return {"ok": False, "url": "", "error": "取上传凭证失败（%s）" % type(exc).__name__}
    policy = payload.get("data") or {}
    if not policy:
        # ⚠ 别只说"凭证为空"：key 失效时服务端是 **401 + `InvalidApiKey`**
        #   （2026-09-21 实踩），旧写法只看 data 是否为空 ⇒ 把"你的 key 无效"
        #   说成"检查 key 与 region"，把真因盖住了。**错误必须如实带出来**。
        detail = " ".join(
            str(payload.get(k) or "").strip() for k in ("code", "message")).strip()
        return {"ok": False, "url": "",
                "error": "取上传凭证被拒（HTTP %s%s）" % (
                    resp.status_code,
                    "：" + detail[:140] if detail else "（检查 key 与 region）")}
    host, key, fields = _policy_fields(policy, path.name)
    if not host or not key:
        return {"ok": False, "url": "", "error": "上传凭证缺 upload_host/upload_dir"}
    try:
        data = path.read_bytes()
    except Exception as exc:
        return {"ok": False, "url": "", "error": "读音频失败（%s）" % type(exc).__name__}
    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False, "url": "",
                "error": "音轨超过 1GB，批量通道一次传不了"}
    try:
        resp = await client.post(host, data=fields,
                                 files={"file": (path.name, data, "application/octet-stream")},
                                 timeout=UPLOAD_TIMEOUT)
    except Exception as exc:
        return {"ok": False, "url": "", "error": "上传音轨失败（%s）" % type(exc).__name__}
    if resp.status_code != 200:
        return {"ok": False, "url": "",
                "error": "上传音轨失败（HTTP %d）" % resp.status_code}
    return {"ok": True, "url": "oss://" + key, "error": ""}


async def _submit(client, oss_url: str) -> Dict[str, Any]:
    """提交任务。返回 {ok, task_id, error}。"""
    body: Dict[str, Any] = {"model": _model(),
                            "input": {"file_urls": [oss_url]},
                            "parameters": {"channel_id": [0]}}
    model = _model().lower()
    # Qwen-Audio 3.x filetrans and Fun-ASR expose speaker_id on sentence
    # results when diarization is enabled. Do not send the provider-specific
    # option to unrelated ASR models.
    if _diarization_enabled() and (
            model.startswith("qwen-audio-3.") or model.startswith("fun-asr")):
        body["parameters"]["diarization_enabled"] = True
    hints = _lang_hints()
    if hints:
        body["parameters"]["language_hints"] = hints
    try:
        resp = await client.post(
            _base() + SUBMIT_PATH, json=body, timeout=SUBMIT_TIMEOUT,
            headers={"Authorization": "Bearer " + _key(),
                     "X-DashScope-Async": "enable",
                     # 用我们自己上传换来的 oss:// 时**必须**带这个头
                     "X-DashScope-OssResourceResolve": "enable"})
        payload = resp.json() or {}
    except Exception as exc:
        return {"ok": False, "task_id": "", "error": "提交转写任务失败（%s）" % type(exc).__name__}
    task_id = str(((payload.get("output") or {}).get("task_id")) or "")
    if resp.status_code != 200 or not task_id:
        return {"ok": False, "task_id": "",
                "error": "提交转写任务被拒（HTTP %s %s）" % (
                    resp.status_code, str(payload.get("message") or payload.get("code") or "")[:80])}
    return {"ok": True, "task_id": task_id, "error": ""}


async def _poll(client, task_id: str) -> Dict[str, Any]:
    """轮询到终态。返回 {ok, output, error}（失败时 error 带官方 code/message）。"""
    import asyncio
    loop = asyncio.get_running_loop()
    deadline = loop.time() + POLL_BUDGET
    last = ""
    while True:
        try:
            resp = await client.get(
                _base() + (TASK_PATH % task_id),
                headers={"Authorization": "Bearer " + _key()},
                timeout=SUBMIT_TIMEOUT)
            output = ((resp.json() or {}).get("output") or {})
        except Exception as exc:
            last = "轮询失败（%s）" % type(exc).__name__
            output = {}
        status = str(output.get("task_status") or "")
        if status == "SUCCEEDED":
            return {"ok": True, "output": output, "error": ""}
        if status in ("FAILED", "UNKNOWN"):
            return {"ok": False, "output": output,
                    "error": "转写任务失败（%s%s）" % (
                        output.get("code") or "", ("：" + str(output.get("message"))[:80])
                        if output.get("message") else "")}
        if loop.time() > deadline:
            return {"ok": False, "output": output, "error": last or "转写任务超时"}
        await asyncio.sleep(POLL_INTERVAL)


async def _fetch_text(client, output: Dict[str, Any]) -> Dict[str, Any]:
    """`transcription_url` → 正文。那个 URL 24 小时有效，**必须当场取**。"""
    results = output.get("results") or []
    url = ""
    if isinstance(results, list) and results and isinstance(results[0], dict):
        url = str(results[0].get("transcription_url") or "")
    if not url:
        return _fail("转写完成但没有结果地址")
    try:
        resp = await client.get(url, timeout=RESULT_TIMEOUT)
        doc = resp.json() or {}
    except Exception as exc:
        return _fail("取转写正文失败（%s）" % type(exc).__name__)
    pieces = []
    segments = []

    def add_sentence(sentence: Dict[str, Any], *, channel_id=None,
                     inherited_speaker=None) -> None:
        sentence_text = str(sentence.get("text") or "").strip()
        if not sentence_text:
            return
        pieces.append(sentence_text)
        start = sentence.get("begin_time", sentence.get("start_time"))
        end = sentence.get("end_time")
        try:
            start_ms = int(start) if start is not None else None
        except (TypeError, ValueError):
            start_ms = None
        try:
            end_ms = int(end) if end is not None else None
        except (TypeError, ValueError):
            end_ms = None
        speaker = sentence.get("speaker_id", inherited_speaker)
        segment = {"text": sentence_text, "start_ms": start_ms,
                   "end_ms": end_ms, "channel_id": channel_id}
        if speaker is not None:
            segment["speaker_id"] = str(speaker)
        segments.append(segment)

    for transcript in (doc.get("transcripts") or []):
        if not isinstance(transcript, dict):
            continue
        channel_id = transcript.get("channel_id")
        direct_sentences = transcript.get("sentences")
        if isinstance(direct_sentences, list) and direct_sentences:
            for sentence in direct_sentences:
                if isinstance(sentence, dict):
                    add_sentence(sentence, channel_id=channel_id)
            continue
        # Some DashScope model versions group sentence rows by speaker.
        speaker_groups = transcript.get("speakers")
        if isinstance(speaker_groups, list) and speaker_groups:
            for group in speaker_groups:
                if not isinstance(group, dict):
                    continue
                for sentence in (group.get("sentences") or []):
                    if isinstance(sentence, dict):
                        add_sentence(sentence, channel_id=channel_id,
                                     inherited_speaker=group.get("speaker_id"))
            continue
        transcript_text = str(transcript.get("text") or "").strip()
        if transcript_text:
            pieces.append(transcript_text)
    text = "\n".join(pieces).strip()
    if not text:
        # 静音/纯底噪会走到这里——**如实报"没有识别到语音"，不要伪造正文**
        return _fail("转写结果为空：这段音轨里没有识别到语音")
    # Sentence rows are the useful source map; sort because a few response
    # formats group sentences under speakers instead of time order.
    if segments and all(s.get("start_ms") is not None for s in segments):
        segments.sort(key=lambda s: (s["start_ms"], s.get("end_ms") or s["start_ms"]))
        # Grouped-by-speaker results append pieces in speaker order. Rebuild
        # the flat transcript chronologically when every piece has a segment.
        if len(segments) == len(pieces):
            text = "\n".join(str(segment.get("text") or "").strip()
                             for segment in segments).strip()
    return {"ok": True, "text": text, "segments": segments, "error": ""}


async def transcribe_file(path) -> Dict[str, Any]:
    """本地音频文件 → 全文。返回 {ok, text, error}；**绝不抛**（与现役转写同款契约）。

    与 `media_transcribe._transcribe_chunks` 的差别：**不切段、不并发**——整段传一次。
    """
    if not _key():
        return _fail("服务端未配置语音识别凭据（DASHSCOPE_API_KEY）")
    src = Path(str(path))
    if not src.exists():
        return _fail("音轨文件不存在")
    try:
        import httpx  # noqa: F401  懒导入：缺依赖不该拖垮 app 启动
    except Exception:
        return _fail("服务端缺少抓取依赖（httpx），请先安装后再试")
    async with await _client(UPLOAD_TIMEOUT) as client:
        up = await _upload(client, src)
        if not up["ok"]:
            return _fail(up["error"])
        sub = await _submit(client, up["url"])
        if not sub["ok"]:
            return _fail(sub["error"])
        polled = await _poll(client, sub["task_id"])
        if not polled["ok"]:
            return _fail(polled["error"])
        return await _fetch_text(client, polled["output"])
