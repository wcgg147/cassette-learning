"""
RobustLLMClient - LongCat 2.0, rate-limited
"""
import os
import logging
import asyncio
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import AsyncOpenAI

logger = logging.getLogger("llm_client")


@asynccontextmanager
async def _null_gate():
    """No-op async context: the caller already holds a permit."""
    yield


logging.getLogger("httpx").setLevel(logging.WARNING)

# A cold-job entrypoint owns its resource default. Child asyncio tasks inherit
# this context; independent foreground tasks keep their own default. Explicit
# per-call gates always take precedence over the inherited resource scope.
_LLM_RESOURCE_SCOPE: ContextVar[Optional[str]] = ContextVar(
    "llm_resource_scope", default=None,
)


@contextmanager
def llm_resource_scope(gate: str):
    """Set a task-local default gate, restoring it on exit/cancellation."""
    name = str(gate).lower()
    if name not in {"realtime", "planning", "background"}:
        raise ValueError("Unknown LLM resource scope")
    token = _LLM_RESOURCE_SCOPE.set(name)
    try:
        yield
    finally:
        _LLM_RESOURCE_SCOPE.reset(token)


def background_llm_job(func):
    """Wrap an async cold-job boundary without changing its calls or result."""
    @wraps(func)
    async def scoped(*args, **kwargs):
        with llm_resource_scope("background"):
            return await func(*args, **kwargs)
    return scoped

DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
MAX_RETRIES = 10

# 单次请求超时的上限。原来硬编码 60s，但后台链路（审计）在第三方网关上
# 实测可能跑到 50~90s，60s 的帽子会把"慢但能成"的调用判成失败，
# 于是审计/记忆整理静默降级。放宽到 180s；实时语音链路自己传的是小值，
# 不受影响。
MAX_REQUEST_TIMEOUT = max(5, int(os.getenv("LLM_MAX_REQUEST_TIMEOUT", "180")))

# ── OpenCode 网关（Zen / Go）─────────────────────────────────────────────
# 2026-09-18：附属链路（审计 / 记忆整理 / 自修复）从方舟切到 OpenCode Go 的
# deepseek 系模型。它的 Chat Completions 与 OpenAI 同形，我们已有的
# AsyncOpenAI 直接能用，**唯一缺的是一个会话头**：
#   缺 `x-opencode-session: ses_<uuid>` → 400 MissingSessionID，全部模型挂。
# 该头只在 URL 命中 opencode.ai 时附加，不影响方舟/DashScope 任何调用。
OPENCODE_HOST_MARKER = "opencode.ai"


def _is_opencode_url(url: Optional[str]) -> bool:
    return OPENCODE_HOST_MARKER in str(url or "").lower()

# 未显式指定 model 时的默认模型（refine / 点词卡片等走这里）。
# 2026-09-18：从方舟接入点 `ep-*` 改为 DeepSeek 官方 `deepseek-flash`。
# 理由：① 方舟接入点持续 400 `ClosedEndpoint`；② 这个默认模型覆盖的是
# 「开场白 / refine / 点词卡片 / 素材摘要 / 素材读图」等散落调用点，它们
# **都不传 base_url**，只要模型名还是 `ep-*` 就一律发往方舟 ⇒ 方舟一挂
# 这些功能全死（症状是"语音能说、AI 不回、开场白也没有"）。
# `deepseek-*` 是模型名；实际网关由 DEEPSEEK_BASE_URL 决定。默认复用 OpenCode，
# 不会自动调用 DeepSeek 官方 API。需要官方端点时必须显式配置其 URL。
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "deepseek-v4.1-flash")

# DeepSeek 官方端点（OpenAI 兼容）。只用于 `deepseek*` 模型名的兜底路由：
# 显式传了 base_url 的调用（审计 / 记忆整理 / 自修复走 OpenCode）不受影响。
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1"),
)


def _resolve_route(model_name: str, base_url: Optional[str], api_key: Optional[str],
                   default_key: Optional[str]) -> tuple:
    """决定该模型名走哪个端点 / 用哪把 key。

    优先级：**显式 base_url > 模型名前缀 > 兜底方舟**。

    ⚠ 这里的"兜底方舟"是历史包袱：模型名没命中任何前缀时会发往方舟。
    所以换供应商时**只改模型名是不够的** —— 名字不在下表里就会被静默发去
    方舟（症状是"改配置没生效"）。要接新供应商，必须在这里加分支，
    或在调用点显式传 `base_url`。

    命中表：
      * `qwen*`            → DashScope（key 回落 `DASHSCOPE_API_KEY`）
      * `doubao*` / `ep-*` → 火山方舟（key 回落 `DOUBAO_API_KEY`）
      * `deepseek*`        → `DEEPSEEK_BASE_URL`（默认 OpenCode；只有显式设置为
                             官方地址时才走 DeepSeek 官方）
    """
    key = api_key or default_key
    if base_url:
        if _is_opencode_url(base_url):
            return base_url, os.getenv("OPENCODE_API_KEY", "") or key
        return base_url, key
    if model_name.startswith("qwen"):
        return ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key or os.getenv("DASHSCOPE_API_KEY", "") or key)
    if model_name.startswith("doubao") or model_name.startswith("ep-"):
        return ("https://ark.cn-beijing.volces.com/api/v3",
                key or os.getenv("DOUBAO_API_KEY", ""))
    if model_name.startswith("deepseek"):
        # OpenCode 端点必须使用 OpenCode key；不要把主链中显式传来的旧
        # DeepSeek/Ark key 携带到另一个网关。
        if _is_opencode_url(DEEPSEEK_BASE_URL):
            return (DEEPSEEK_BASE_URL,
                    os.getenv("OPENCODE_API_KEY", "")
                    or api_key
                    or os.getenv("DEEPSEEK_API_KEY", "")
                    or os.getenv("TUTOR_RESPONSE_API_KEY", "")
                    or default_key)
        return (DEEPSEEK_BASE_URL,
                api_key or os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("TUTOR_RESPONSE_API_KEY", "")
                or default_key)
    return DOUBAO_BASE_URL, key


# Ark 的 doubao/`ep-*` endpoint 可能是"思考型"模型：每次调用先写一大段私有推理
# 再给答案。在配置的外教 endpoint 上实测：一句三个词的提示带思考要 10.4s
# （reasoning_content 4047 字符），关掉思考 0.67s（约 20 倍）；而语音一轮至少付
# 两次这样的调用，代理的 controlled-reply 期限是 12s —— 于是学习者每次都只听到
# 那句 "Sorry, I spaced out" 兜底。导演（conversation_director.py）与记忆整理
# （memory_curator.py）早已在自己的调用点关掉思考，这个开关补齐其余面向学习者的
# 调用。设 TUTOR_DISABLE_THINKING=0 可恢复服务端推理。
_TUTOR_DISABLE_THINKING = os.getenv(
    "TUTOR_DISABLE_THINKING", "1"
).strip().lower() not in ("0", "false", "no", "off")


def thinking_disabled_extra_body(
    model: Optional[str], base_url: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """返回用于关闭服务端推理的 extra_body；不适用时返回 None。

    认这个字段的供应商（2026-09-18 逐个实测，字段名都是 `thinking`）：
      * Ark endpoint：`doubao*` / `ep-*`；
      * DeepSeek 官方（api.deepseek.com）：`deepseek-flash` / `deepseek-v4-pro`
        —— 不关思考时 `reasoning_tokens` 会吃光主链 320 token 的预算，
        `finish_reason=length` 且 `content` 为空（学习者只听到空气泡）；
      * OpenCode 网关（opencode.ai）：`deepseek*` 系。

    `base_url` 给定时按域名再确认一次，避免把 `thinking` 发给不认它的
    DashScope（那边也有 `deepseek-*` 型号）。
    """
    if not _TUTOR_DISABLE_THINKING:
        return None
    name = str(model or "")
    if name.startswith("ep-") or name.startswith("doubao"):
        return {"thinking": {"type": "disabled"}}
    if name.startswith("deepseek"):
        url = str(base_url or "")
        if not url or "api.deepseek.com" in url or _is_opencode_url(url):
            return {"thinking": {"type": "disabled"}}
    return None


def aux_thinking_disabled_extra_body(
    model: Optional[str], base_url: Optional[str]
) -> Optional[Dict[str, Any]]:
    """附属链路（审计 / 自修复 / 记忆整理）关闭服务端推理。

    与上面的 `thinking_disabled_extra_body` 分开，各管各的供应商：

    * 上面那个只管 Ark 的 `doubao*` / `ep-*`；
    * 这个只管 **OpenCode 网关上的 deepseek 系**。

    为什么要单独管：OpenCode 的 deepseek **默认开着推理**，实测同一个真实审计
    prompt（system 3650 字符 + payload）：

        不关思考  deepseek-v4.1-flash  34.7s / 7464 output tokens
        关掉思考  deepseek-v4.1-flash   7.9s / 1456 output tokens   ← 省 80%
        关掉思考  deepseek-flash        6.5s / 1410 output tokens

    ⚠ `deepseek-v4-flash` **不认这个字段**：关不关都跑 167~180s、烧 ~9900 tokens，
    且 1400 max_tokens 会被推理吃光导致 `finish_reason=length`（JSON 截断）。**别选它。**
    """
    if _is_opencode_url(base_url) and str(model or "").startswith("deepseek"):
        return {"thinking": {"type": "disabled"}}
    return None


class LLMCompletionError(RuntimeError):
    """Completion failure carrying only non-sensitive metering facts."""

    def __init__(self, message: str, metering: Dict[str, Any]):
        super().__init__(message)
        self.metering = metering

class RobustLLMClient:
    def __init__(self):
        self.doubao_key = DOUBAO_API_KEY
        self._sem = asyncio.Semaphore(2)
        # ── 后台闸位（2026-09-18 新增，BUG-0917-09 §16.6）────────────────
        # 为什么必须和前台分开：闸位一共只有 2 个，而后台任务单次最长占用
        # `timeout` 秒（记忆整理 15s、审计更久）。共用时会跟**实时语音回复**
        # 抢名额，表现成"用户说完话 AI 迟回"——那才是产品体验的命门。
        # 拆开之后，后台再慢也拿不到前台那 2 个名额，只会在后台队列里自己排队。
        # ⚠ 注意：实时回复走的是 chat_completion（非流式），所以**不能按方法拆**，
        # 由调用方显式 gate 或冷作业入口资源 scope 声明；其余默认走前台闸。
        self._bg_sem = asyncio.Semaphore(max(1, int(os.getenv("BACKGROUND_LLM_CONCURRENCY", "2"))))
        # Deferred conversation planning must neither occupy spoken-reply
        # permits nor queue behind slow memory/audit work. This changes local
        # admission only; provider-wide quotas still apply to every request.
        self._planning_sem = asyncio.Semaphore(2)
        # 每个 (base_url, api_key) 只建一个 AsyncOpenAI。此前在重试循环体内
        # 每次 chat_completion 都 new 一个客户端：AsyncOpenAI 构造会新建
        # httpx.AsyncClient 并同步创建 SSL 上下文（Windows 上加载系统证书库
        # 实测 1–2s 纯 CPU），且阻塞事件循环——语音热路径每轮至少两次 LLM
        # 调用，等于每轮白烧 2–4s（2026-09-15 py-spy 实锤，0914-06 的同族病）。
        self._clients: Dict[tuple, AsyncOpenAI] = {}
        self._client_lock = threading.Lock()
        self._client_initializations: Dict[tuple, asyncio.Task] = {}
        # OpenCode 网关的会话 id，首次用到时惰性生成（见 _get_client）。
        self._opencode_session: str = os.getenv("OPENCODE_SESSION_ID", "")
        # 供应商熔断（2026-10-21）：某个供应商刚超时/503 后，接下来
        # `PROVIDER_COOLDOWN_SECONDS` 秒内的调用**直接走备用**，不再先等一遍
        # 超时。没有这个，一轮里两次调用（导演 + 回复）各等满超时再切换，
        # 实测把单轮拖到 30s+；有了它第二次调用零代价跳过坏掉的供应商。
        self._provider_down_until: Dict[str, float] = {}
        self._provider_cooldown = max(
            1.0, float(os.getenv("PROVIDER_COOLDOWN_SECONDS", "60"))
        )
        # 预热豆包主客户端：把唯一的 1–2s SSL 上下文成本挪到进程启动时，
        # 用户第一轮不再付费。
        if self.doubao_key and DOUBAO_BASE_URL:
            self._clients[(DOUBAO_BASE_URL, self.doubao_key)] = AsyncOpenAI(
                api_key=self.doubao_key, base_url=DOUBAO_BASE_URL,
            )

    def _get_client(self, key: str, url: str, timeout: int) -> AsyncOpenAI:
        """取缓存的客户端；超时按请求传参（create 支持 per-request timeout）。"""
        cache_key = (url, key)
        with self._client_lock:
            client = self._clients.get(cache_key)
            if client is None:
                extra: Dict[str, Any] = {}
                if _is_opencode_url(url):
                    # OpenCode requires a session header; reuse the same ID
                    # while constructing clients for concurrent first calls.
                    if not self._opencode_session:
                        self._opencode_session = os.getenv("OPENCODE_SESSION_ID") or ("ses_" + uuid.uuid4().hex)
                    extra["default_headers"] = {"x-opencode-session": self._opencode_session}
                client = AsyncOpenAI(api_key=key, base_url=url, **extra)
                self._clients[cache_key] = client
        return client

    async def _get_client_async(self, key: str, url: str, timeout: int) -> AsyncOpenAI:
        """Initialize a cold HTTP client off-loop without blocking voice timers."""
        cache_key = (url, key)
        # _get_client holds _client_lock while constructing AsyncOpenAI. On
        # Windows that can include slow synchronous TLS setup, so acquiring
        # this lock here would block the event loop even though construction
        # itself runs in a worker thread. A lock-free dict read is safe here;
        # _get_client rechecks and serializes writes under the lock.
        cached = self._clients.get(cache_key)
        if cached is not None:
            return cached
        pending = self._client_initializations.get(cache_key)
        if pending is None:
            pending = asyncio.create_task(
                asyncio.to_thread(self._get_client, key, url, timeout)
            )
            self._client_initializations[cache_key] = pending

            def finished(task):
                if self._client_initializations.get(cache_key) is task:
                    self._client_initializations.pop(cache_key, None)
                if not task.cancelled():
                    task.exception()  # Consume errors if every waiting turn was cancelled.

            pending.add_done_callback(finished)
        # A timed-out Director must not cancel setup shared with the reply path.
        return await asyncio.shield(pending)

    def _gate(self, kwargs: dict):
        """Explicit per-call gate > task-local job scope > realtime default.

        Unscoped calls keep the original realtime semaphore. An entrypoint
        wrapped as a background job supplies a default for all unannotated
        descendants; explicit realtime/planning/background calls still win.
        """
        # Preserve the original semantics even for an explicitly supplied
        # None/empty gate: that caller has opted into the realtime default.
        gate = str(
            (kwargs.get("gate") or "realtime") if "gate" in kwargs
            else (_LLM_RESOURCE_SCOPE.get() or "realtime")
        ).lower()
        if gate == "planning":
            return self._planning_sem
        if gate == "background":
            return self._bg_sem
        return self._sem

    async def chat_completion(self, messages, model=None, temperature=0.7,
                              max_tokens=None, extra_body=None,
                              base_url=None, api_key=None, return_usage=False, **kwargs):
        model_name = model or DEFAULT_CHAT_MODEL
        # 端点 / key 的选择集中在 _resolve_route()：
        # 显式 base_url > 模型名前缀（qwen / doubao / ep / deepseek）> 兜底方舟。
        url, key = _resolve_route(model_name, base_url, api_key, self.doubao_key)

        # 熔断：该供应商刚失败过就先直接换备用，避免每一轮都先等满超时。
        # 判据是**解析后的端点属于哪个供应商**，与调用方是否显式传 base_url
        # 无关：主链、导演、审计都可能显式传 OpenCode 端点，它挂了这些全都该跳过。
        if not kwargs.get("_failover_done"):
            primary = self._provider_name(url)
            if self._provider_is_down(primary):
                alt = self._failover_route(model_name, url, key)
                if alt:
                    alt_model, alt_url, alt_key = alt
                    logger.warning(
                        "cooldown skip %s(%s) -> %s(%s)",
                        model_name, primary, alt_model, self._provider_name(alt_url),
                    )
                    result = await self.chat_completion(
                        messages, model=alt_model, temperature=temperature,
                        max_tokens=max_tokens, extra_body=extra_body,
                        base_url=alt_url, api_key=alt_key,
                        return_usage=return_usage, _failover_done=True,
                        **{k: v for k, v in kwargs.items()
                           if k not in {"max_retries", "_failover_done"}},
                    )
                    if return_usage and isinstance(result, dict):
                        meta = result.setdefault("metering", {})
                        meta["failover_from"] = model_name
                        meta["failover_reason"] = "provider_cooldown"
                    return result

        # failover 递归调用不再抢一次闸位：外层已经持有，重复获取会在
        # 并发两条请求同时 failover 时把闸位（默认 2 个）耗尽而互锁。
        async with self._gate(kwargs) if not kwargs.get("_failover_done") else _null_gate():
            attempt = 0
            last_failure_code = "completion_failed"
            attempt_limit = max(1, min(MAX_RETRIES, int(kwargs.get("max_retries", MAX_RETRIES))))
            request_timeout = max(2, min(MAX_REQUEST_TIMEOUT, int(kwargs.get("timeout", 60))))
            while attempt < attempt_limit:
                try:
                    # The first attempt must carry zero added delay: the voice
                    # hot path pays this sleep once per LLM call, and there are
                    # at least two LLM calls per turn.  The 0.5s "avoid bursts"
                    # spacing used to sit at the top of this loop body, so every
                    # call slept 500ms before its very first request even when
                    # that request succeeded.  It now only applies to retries,
                    # in addition to the exponential backoff.
                    if attempt > 0:
                        await asyncio.sleep(0.5)  # 重试间隔，避免突发
                        wait = min(2 ** attempt, 10)  # 退避上限降到 10s
                        logger.warning(f"retry {attempt}/{MAX_RETRIES} in {wait}s...")
                        await asyncio.sleep(wait)
                    client = await self._get_client_async(key, url, request_timeout)
                    payload = {"model": model_name, "messages": messages, "temperature": temperature}
                    if max_tokens:
                        payload["max_tokens"] = max_tokens
                    if kwargs.get("reasoning_effort"):
                        payload["reasoning_effort"] = kwargs["reasoning_effort"]
                    if extra_body:
                        payload["extra_body"] = extra_body
                    if kwargs.get("response_format"):
                        payload["response_format"] = kwargs["response_format"]
                    if kwargs.get("tools"):
                        payload["tools"] = kwargs["tools"]
                        payload["tool_choice"] = kwargs.get("tool_choice", "auto")
                    res = await client.chat.completions.create(**payload, timeout=request_timeout)
                    choice = res.choices[0]
                    message = choice.message
                    content = message.content or ""
                    tool_calls = getattr(message, "tool_calls", None) or []
                    if kwargs.get("return_message") and (content.strip() or tool_calls):
                        if hasattr(message, "model_dump"):
                            message_payload = message.model_dump(exclude_none=True)
                        else:
                            message_payload = {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls,
                            }
                        normalized = {
                            "role": "assistant",
                            "content": str(message_payload.get("content") or ""),
                            "tool_calls": message_payload.get("tool_calls") or [],
                            "reasoning_content": message_payload.get("reasoning_content"),
                            "finish_reason": getattr(choice, "finish_reason", None),
                        }
                        logger.info(
                            "OK (%s) finish=%s tool_calls=%s",
                            model_name, normalized["finish_reason"],
                            len(normalized["tool_calls"]),
                        )
                        return normalized
                    if content and content.strip():
                        logger.info(f"OK ({model_name})")
                        if not return_usage:
                            return content.strip()
                        usage = self._normalize_usage(getattr(res, "usage", None))
                        return {
                            "content": content.strip(),
                            "metering": {
                                "provider": self._provider_name(url),
                                "model": model_name,
                                "request_status": "success",
                                "request_count": attempt + 1,
                                "retry_count": attempt,
                                **usage,
                            },
                        }
                    attempt += 1
                    reasoning = str(getattr(message, "reasoning_content", None) or "")
                    logger.warning(
                        "empty %s/%s finish=%s reasoning_chars=%s tool_calls=%s",
                        attempt, attempt_limit, getattr(choice, "finish_reason", None),
                        len(reasoning), len(tool_calls),
                    )
                except Exception as e:
                    attempt += 1
                    last_failure_code = self._classify_failure(e)
                    logger.warning(f"fail {attempt}/{MAX_RETRIES}: {str(e)[:80]}")
                    if last_failure_code in {
                        "insufficient_balance", "authentication_failed",
                    }:
                        # Retrying a depleted account or rejected credential
                        # cannot recover; avoid a long, useless retry storm.
                        break
                    # 上游 503 / 超时 / 连接错误 = 供应商故障，重试同一端点大概率
                    # 还是同一个故障。这里切到备用供应商再试一次，成功就正常返回；
                    # 仍失败才抛 LLMCompletionError（调用方据此向用户报错，不再
                    # 用一句"我没接住"假装成回复）。
                    if (last_failure_code in {
                            "provider_unavailable", "provider_timeout",
                            "completion_failed",
                        }
                            and not kwargs.get("_failover_done")):
                        self._mark_provider_down(self._provider_name(url))
                        alt = self._failover_route(model_name, url, key)
                        if alt:
                            alt_model, alt_url, alt_key = alt
                            logger.warning(
                                "failover %s(%s) -> %s(%s) after %s",
                                model_name, self._provider_name(url),
                                alt_model, self._provider_name(alt_url),
                                last_failure_code,
                            )
                            result = await self.chat_completion(
                                messages, model=alt_model, temperature=temperature,
                                max_tokens=max_tokens, extra_body=extra_body,
                                base_url=alt_url, api_key=alt_key,
                                return_usage=return_usage, _failover_done=True,
                                **{k: v for k, v in kwargs.items()
                                   if k not in {"max_retries", "_failover_done"}},
                            )
                            if return_usage and isinstance(result, dict):
                                meta = result.setdefault("metering", {})
                                meta["failover_from"] = model_name
                                meta["failover_reason"] = last_failure_code
                            return result
            raise LLMCompletionError(
                f"{model_name} failed after {attempt} attempt(s)",
                {
                    "provider": self._provider_name(url),
                    "model": model_name,
                    "request_status": "failed",
                    "request_count": attempt,
                    "retry_count": max(0, attempt - 1),
                    "token_usage_status": "unavailable",
                    "failure_code": last_failure_code,
                },
            )

    async def stream_chat_completion(
        self, messages, model=None, temperature=0.7, max_tokens=None,
        extra_body=None, base_url=None, api_key=None, **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield provider text deltas and one terminal metering event.

        Unlike ``chat_completion`` this method never retries after the first
        visible delta: replaying a partially-heard sentence would be worse than
        surfacing the transport failure to the voice controller.  The caller is
        expected to buffer deltas into complete phrases before playback.
        """
        model_name = model or DEFAULT_CHAT_MODEL
        url, key = _resolve_route(model_name, base_url, api_key, self.doubao_key)

        request_timeout = max(2, min(MAX_REQUEST_TIMEOUT, int(kwargs.get("timeout", 60))))
        emitted = False
        async with self._gate(kwargs):
            try:
                client = await self._get_client_async(key, url, request_timeout)
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if max_tokens:
                    payload["max_tokens"] = max_tokens
                if extra_body:
                    payload["extra_body"] = extra_body
                stream = await client.chat.completions.create(**payload, timeout=request_timeout)
                final_usage = None
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        final_usage = usage
                    for choice in getattr(chunk, "choices", None) or []:
                        delta = getattr(getattr(choice, "delta", None), "content", None)
                        if delta:
                            emitted = True
                            yield {"type": "delta", "text": str(delta)}
                yield {
                    "type": "done",
                    "metering": {
                        "provider": self._provider_name(url),
                        "model": model_name,
                        "request_status": "success",
                        "request_count": 1,
                        "retry_count": 0,
                        **self._normalize_usage(final_usage),
                    },
                }
            except asyncio.CancelledError:
                logger.info("stream cancelled (%s)", model_name)
                raise
            except Exception as exc:
                failure_code = self._classify_failure(exc)
                logger.warning(
                    "stream failed (%s, emitted=%s): %s",
                    model_name, emitted, str(exc)[:80],
                )
                raise LLMCompletionError(
                    f"{model_name} streaming failed",
                    {
                        "provider": self._provider_name(url),
                        "model": model_name,
                        "request_status": "failed",
                        "request_count": 1,
                        "retry_count": 0,
                        "token_usage_status": "unavailable",
                        "failure_code": failure_code,
                        "partial_output": emitted,
                    },
                ) from exc

    @staticmethod
    def _classify_failure(exc: Exception) -> str:
        """Reduce provider errors to a safe, actionable operational code.

        Provider messages can contain request details, so only the category is
        persisted.  In particular, an exhausted balance must not look like a
        random model failure or trigger pointless retries forever.
        """
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        text = " ".join((str(exc or ""), str(body or ""))).lower()
        compact = re.sub(r"\s+", " ", text)
        if any(marker in compact for marker in (
            "insufficient balance", "insufficient_balance", "insufficient quota",
            "quota exceeded", "余额不足", "账户余额", "欠费",
        )):
            return "insufficient_balance"
        if status == 429 or any(marker in compact for marker in (
            "rate limit", "rate_limit", "too many requests", "限流",
        )):
            return "rate_limited"
        if status in {401, 403} or any(marker in compact for marker in (
            "invalid api key", "authentication", "unauthorized", "forbidden",
        )):
            return "authentication_failed"
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or any(
            marker in compact for marker in ("timed out", "timeout")
        ):
            return "provider_timeout"
        if isinstance(status, int) and status >= 500:
            return "provider_unavailable"
        return "completion_failed"

    def _provider_is_down(self, provider: str) -> bool:
        """该供应商是否还在熔断窗口内（刚超时/503 过）。"""
        until = self._provider_down_until.get(provider, 0.0)
        return bool(until) and time.monotonic() < until

    def _mark_provider_down(self, provider: str) -> None:
        """记一次供应商故障，冷却窗口内后续调用直接走备用。"""
        if not provider:
            return
        self._provider_down_until[provider] = time.monotonic() + self._provider_cooldown

    def _failover_route(
        self, model_name: str, url: str, key: str
    ) -> Optional[tuple]:
        """备用供应商路由；没有可用备用时返回 None。

        只在**供应商侧故障**（503 / 超时 / 连接错误）时调用。与 `_resolve_route`
        分开的理由：主链的模型名与端点由配置决定，failover 只换"谁来答这一轮"，
        不改主链配置，也不在正常路径上引入任何额外判断。

        DeepSeek（OpenCode 网关）挂时切 DashScope 的 qwen-plus；反之亦然。
        同名供应商不切（同一个 503 再打一遍没有意义）。
        """
        current = self._provider_name(url)
        if current in {"opencode", "deepseek"}:
            alt_model = os.getenv("FAILOVER_MODEL", "qwen-plus")
            alt_key = os.getenv("DASHSCOPE_API_KEY", "")
            if not alt_key:
                return None
            return (alt_model,
                    os.getenv("FAILOVER_BASE_URL",
                              "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    alt_key)
        if current == "dashscope":
            alt_model = os.getenv("FAILOVER_MODEL_BACK", "deepseek-v4.1-flash")
            alt_url = os.getenv("OPENCODE_BASE_URL", "")
            alt_key = os.getenv("OPENCODE_API_KEY", "")
            if not (alt_url and alt_key):
                return None
            return (alt_model, alt_url, alt_key)
        return None

    @staticmethod
    def _provider_name(base_url: str) -> str:
        url = (base_url or "").lower()
        if "volces.com" in url:
            return "volcengine_ark"
        if "dashscope" in url:
            return "dashscope"
        if "api.deepseek.com" in url:
            return "deepseek_official"
        if "opencode.ai" in url:
            return "opencode"
        if "longcat" in url:
            return "longcat"
        return "openai_compatible"

    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, Any]:
        """Copy provider-reported token counts, otherwise mark unavailable."""
        if usage is None:
            return {
                "input_tokens": None, "output_tokens": None, "total_tokens": None,
                "cached_tokens": None,
                "token_usage_status": "unavailable",
            }
        if hasattr(usage, "model_dump"):
            payload = usage.model_dump()
        elif isinstance(usage, dict):
            payload = usage
        else:
            payload = {
                key: getattr(usage, key, None)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "prompt_tokens_details", "cached_tokens")
            }
        def token(*names):
            for name in names:
                value = payload.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return max(0, int(value))
            return None
        input_tokens = token("prompt_tokens", "input_tokens")
        output_tokens = token("completion_tokens", "output_tokens")
        total_tokens = token("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        # 方舟隐式前缀缓存命中的输入 token 数（OpenAI 兼容形态：
        # usage.prompt_tokens_details.cached_tokens；兼容顶层 cached_tokens）。
        # 只透传不加工——用于 e2e_probe/metering 验证 prompt 重排的命中效果。
        cached_tokens: Optional[int] = None
        details = payload.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
        elif details is not None and not isinstance(details, (str, bytes)):
            cached_tokens = getattr(details, "cached_tokens", None)
        if cached_tokens is None:
            cached_tokens = payload.get("cached_tokens")
        if isinstance(cached_tokens, (int, float)) and not isinstance(cached_tokens, bool):
            cached_tokens = max(0, int(cached_tokens))
        else:
            cached_tokens = None
        reported = any(value is not None for value in (input_tokens, output_tokens, total_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "token_usage_status": "reported" if reported else "unavailable",
        }

robust_llm_client = RobustLLMClient()
