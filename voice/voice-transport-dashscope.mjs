/**
 * voice-transport-dashscope.mjs
 *
 * 把「豆包 Seeduplex 传输层」换成「阿里 DashScope Qwen-Audio Realtime」传输层。
 *
 * ## 设计约束（来自需求原话）
 * "只换掉那个**吃用户音频 + 播 AI 回复**的模型，之前的链路不能失效。"
 * ⇒ 所以本模块**只做一件事**：在 WebSocket 边界做协议翻译，产出一个"长得像豆包
 *    ws"的门面对象。上层 `doubao-session-server.mjs` 里那 3000 行受控回复状态机
 *   （轮次边界、回声抑制、短语队列、兜底、计量）**一行都不用改**，也就不会失效。
 *
 * ## 为什么能这么薄
 * 实测（2026-09-24）两边都是 OpenAI-Realtime 形状：
 *   · ASR 事件名**完全同名**：conversation.item.input_audio_transcription.delta/.completed
 *   · TTS 音频**同格式同字段**：PCM 24k s16le mono，都在 `delta` 字段里
 *   ⇒ 真正要翻的只有 4 处：URL/鉴权、session.create→session.update、
 *     speech_text_buffer.commit→item.create+response.create、以及个别事件改名。
 *
 * ## 实测依据（bugreports/_ali_*.py，全部只读探测）
 *   · 端到端模型 qwen-audio-3.0-realtime-flash：握手 OK、session.update OK；
 *   · ASR：忠实转写，中英混排 3/3 **不翻译**（"我昨天go to the park了"）
 *     —— 注意 `api/engine/asr_filetrans.py` 记过"同厂实时 ASR 中英混说会翻译"，
 *     那条在本模型上**未复现**（见 bugreports 记录）；
 *   · 受控 TTS：`conversation.item.create(input_text)` + `response.create`，
 *     且**必须**带 instructions 要求逐字朗读，否则模型会"回答"而不是"朗读"；
 *   · turn_detection 用 **null（手动）**：`.completed` 有完整带标点转写，
 *     且**绝不自动应答**（server_vad 模式下 .completed 不出、还会多花 token）；
 *   · 专用 qwen3-asr-flash-realtime / qwen3-tts-flash-realtime endpoints have
 *     separate availability and voice IDs; the dedicated TTS path below is
 *     opt-in and only replaces controlled phrase playback. ASR stays on Realtime.
 *
 * ## 开关
 *   VOICE_TRANSPORT=dashscope 时由 doubao-session-server.mjs 选用本模块。
 *   默认（不设）= doubao，行为与改动前**完全一致**。
 *   ALI_TTS_TRANSPORT=dedicated opts controlled replies into a separate TTS WS;
 *   direct E2E sessions never open it.
 */

import { WebSocket } from "ws";
import { randomUUID } from "node:crypto";

// 0924-08：ALI_WS_BASE 可用环境变量覆盖 —— 仅供**离线探针**把垫片指向本地 mock
// 供应商（零额度复现 28s commit → 分段 → 上层关轮全链）；线上缺省不变。
const ALI_WS_BASE = process.env.ALI_WS_BASE || "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=";
const DEFAULT_MODEL = "qwen-audio-3.0-realtime-flash";
const DEFAULT_ASR_MODEL = "qwen3-asr-flash-realtime";
const DEFAULT_VOICE = "longanqian";
const DEDICATED_TTS_MODEL = process.env.ALI_TTS_MODEL || "qwen3-tts-flash-realtime";
const DEDICATED_TTS_VOICE = process.env.ALI_TTS_VOICE || "Cherry";

// 阿里手动模式对**输入音频缓冲**有 30s 硬上限（豆包 Seeduplex 靠服务端 VAD 分段，
// 从不撑爆）。在 28s 处替它提交一次，见 send() 的 input_audio_buffer.append 分支。
// 前端固定 16k s16le mono（cassette-...html:2136/2149）。
const PCM_BYTES_PER_SECOND = 16000 * 2;
const INPUT_AUDIO_MAX_SECONDS = Number(process.env.ALI_INPUT_MAX_SECONDS || 28);

// 朗读指令：实测**必须**带。不带时模型会把文本当"用户说的话"去回答
// （输入"我想点一杯 latte，不用写名字。" → 输出"好的，没问题，这就给你安排…"）。
const READ_VERBATIM_HINT =
  "你是朗读器。逐字朗读用户这句话，一字不改，不要回答、不要添加任何内容。";

// 豆包音色 → 阿里音色。阿里 3.0-flash 可用系统音色有限，统一回落到 longanqian；
// 需要别的嗓子用 ALI_VOICE 覆盖。
const VOICE_MAP = {
  en_female_dacey_uranus_bigtts: "longanqian",
  zh_female_vv_jupiter_bigtts: "longanqian",
};

export function createDashscopeVoiceSocket({ apiKey, log = () => {}, model, voice }) {
  return new DashscopeVoiceSocket({ apiKey, log, model, voice });
}

class DashscopeVoiceSocket {
  constructor({ apiKey, log, model, voice }) {
    this._log = log;
    this._apiKey = apiKey;
    this._model = model || process.env.ALI_REALTIME_MODEL || DEFAULT_MODEL;
    this._asrModel = process.env.ALI_ASR_MODEL || DEFAULT_ASR_MODEL;
    this._voice = voice || process.env.ALI_VOICE || DEFAULT_VOICE;
    // 合成出的"用户消息"会进模型上下文；给个上限避免长会话 token 无界增长。
    this._maxHistoryTurns = Number(process.env.ALI_MAX_HISTORY_TURNS || 8) || 8;
    this._handlers = { open: [], message: [], error: [], close: [] };
    this._sendQueue = [];
    this._sessionUpdateSent = false;
    this._directE2E = false;
    this._dedicatedTtsRequested =
      String(process.env.ALI_TTS_TRANSPORT || "").trim().toLowerCase() === "dedicated";
    this._useDedicatedTts = false;
    this._ttsRaw = null;
    this._ttsReady = false;
    this._ttsQueue = [];
    this._ttsEpoch = 0;
    this._ttsEventCounter = 0;
    this._ttsInResponse = false;
    this._ttsHasAudio = false;
    this._pendingTtsText = "";
    this._mainSessionReady = false;
    this._mainSessionId = "";
    this._sessionCreatedEmitted = false;
    // 本次未提交音频的累计字节数（用于在阿里 30s 上限前主动 commit）。
    this._bufBytes = 0;
    this._maxBufferBytes = INPUT_AUDIO_MAX_SECONDS * PCM_BYTES_PER_SECOND;
    this._log(`[ALI] connecting ${this._model} auth=${apiKey ? "ok" : "MISSING"}`);

    const url = ALI_WS_BASE + encodeURIComponent(this._model);
    // 阿里鉴权走 Authorization: Bearer（豆包走 X-Api-Key）。
    this._raw = new WebSocket(url, { headers: { Authorization: "Bearer " + apiKey } });

    this._raw.on("open", () => {
      this._emit("open");
      const q = this._sendQueue.splice(0);
      for (const item of q) this._rawSend(item);
    });
    this._raw.on("message", (raw) => {
      if (this._dedicatedTtsRequested && this._isProviderEvent(raw, "session.updated")) {
        const event = this._parseProviderEvent(raw);
        this._mainSessionReady = true;
        this._mainSessionId = String(event?.session?.id || "");
        this._maybeEmitSessionCreated();
        return;
      }
      const out = this._translateInbound(raw);
      if (out === null) return; // 被吞掉（如阿里刚连上就发的 session.created）
      this._emit("message", out);
    });
    this._raw.on("error", (e) => this._emit("error", e));
    this._raw.on("close", (code, reason) => this._emit("close", code, reason));
  }

  get readyState() {
    return this._raw ? this._raw.readyState : WebSocket.CONNECTING;
  }

  on(event, cb) {
    if (this._handlers[event]) this._handlers[event].push(cb);
    return this;
  }

  /** 收下"豆包语义"的 JSON 字符串，翻成阿里事件再发。 */
  send(data) {
    let msg = null;
    try {
      msg = typeof data === "string" ? JSON.parse(data) : null;
    } catch (_) {
      msg = null;
    }
    if (!msg || !msg.type) {
      // 非 JSON / 无 type：原样透传（本链路目前不会走到）
      this._queueOrSend(data);
      return;
    }
    this._log(`[ALI] → ${msg.type}`);

    switch (msg.type) {
      case "session.create": {
        const update = this._buildSessionUpdate(msg.session || {});
        this._directE2E = !!update.directE2E;
        this._useDedicatedTts = this._dedicatedTtsRequested && !this._directE2E;
        delete update.directE2E;
        this._queueOrSend(JSON.stringify(update));
        this._sessionUpdateSent = true;
        this._bufBytes = 0;
        if (this._useDedicatedTts) this._connectDedicatedTts();
        return;
      }
      case "speech_text_buffer.commit": {
        // 豆包语义 = "把这段文本念出来"。阿里没有对应事件：
        // 造一条用户消息承载文本 + 明确指令要求逐字朗读（实测不带指令会变成"回答"）。
        const text = String(msg.text || "").trim();
        if (!text) return;
        if (this._useDedicatedTts) {
          this._pendingTtsText = text;
          this._ttsHasAudio = false;
          this._ttsInResponse = true;
          this._sendTtsEvent("input_text_buffer.append", { text });
          this._sendTtsEvent("input_text_buffer.commit");
          return;
        }
        this._sendReadAloud(text);
        return;
      }
      case "response.cancel": {
        // Qwen-TTS Realtime has no documented response.cancel event. Closing
        // its dedicated socket stops an in-flight utterance; reconnect while
        // the learner is speaking so the next reply can use a warm session.
        // Empty cancels are common in controlled mode and must not churn sockets.
        if (this._useDedicatedTts && this._ttsInResponse) {
          this._restartDedicatedTtsAfterCancel();
          return;
        }
        if (this._useDedicatedTts) return;
        this._queueOrSend(JSON.stringify(msg));
        return;
      }
      case "session.close": {
        // 豆包用 session.close；阿里用 session.finish。
        this._queueOrSend(JSON.stringify({ type: "session.finish" }));
        this._closeDedicatedTts();
        return;
      }
      // 用户音频：两边同名，但要顺带记字节数（见上）。
      case "input_audio_buffer.append": {
        // ⛔ 阿里手动模式对输入缓冲有 **30s 硬上限**，超了就每 200ms 回一条
        // `Input audio buffer exceeded maximum duration (30s). Please commit or
        // clear the buffer.`（实测 2026-09-24：一次 59s 连续说话刷出 60 条）。
        // 该文案**不在**上游白名单 `/AudioServerNoAudioInputTooLongError|no session
        // active/` 内 ⇒ 被原样转成 `{type:"error"}` ⇒ 前端 resetReply() +
        // toast('回应没接住 · 再试一次') 风暴，且超过 30s 的那段音频被阿里拒收，
        // 随后一轮的 ASR 会返回空（"不做回复了"）。
        // 为什么豆包没这问题：Seeduplex 用**服务端 VAD**，长句由它自己切成
        // `...transcription.completed` 分段（上层 2419 注释），永不会撑爆缓冲；
        // 换成阿里（turn_detection:null 手动模式）后这层保护就没了。
        // 修法 = 阿里自己给的补救建议 "Please commit or clear the buffer"：
        // 到 28s 主动提交一次，把长话切成段。分段语义与豆包的 provider VAD 一致，
        // 上层 `!clientCommitRequested` 分支本来就会把 provider 段并入同一轮。
        // 正常回合（几秒）永远到不了 28s ⇒ 对既有行为零影响。
        const b64 = typeof msg.audio === "string" ? msg.audio : "";
        if (b64 && !this._directE2E) this._bufBytes += Math.floor((b64.length * 3) / 4);
        this._queueOrSend(JSON.stringify(msg));
        if (!this._directE2E && this._bufBytes >= this._maxBufferBytes) {
          this._log(
            `[ALI] 输入缓冲达 ${Math.round(this._bufBytes / PCM_BYTES_PER_SECOND)}s，` +
            `主动 commit（避免撞 30s 上限）`
          );
          this._queueOrSend(JSON.stringify({ type: "input_audio_buffer.commit" }));
          this._bufBytes = 0;
        }
        return;
      }
      case "input_audio_buffer.commit":
      case "input_audio_buffer.clear": {
        this._bufBytes = 0;
        // Direct realtime sessions use provider server VAD and auto-response.
        // A browser-side commit would create a second, empty/manual response.
        if (this._directE2E && msg.type === "input_audio_buffer.commit") return;
        this._queueOrSend(JSON.stringify(msg));
        return;
      }
      default:
        this._queueOrSend(JSON.stringify(msg));
    }
  }

  close() {
    try { this._raw && this._raw.close(); } catch (_) { /* ignore */ }
    this._closeDedicatedTts();
  }

  // ── Optional dedicated Qwen TTS route ───────────────────

  _sendReadAloud(text) {
    this._queueOrSend(JSON.stringify({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text }],
      },
    }));
    this._queueOrSend(JSON.stringify({
      type: "response.create",
      response: {
        modalities: ["audio", "text"],
        // ⛔ 稿子**必须同时写进 instructions**，不能只靠 item.create！
        // 实测（`_ali_readback_0924.py`，照生产顺序：先推用户音频→ASR 落成一条用户发言，
        // 再走受控朗读）：只给 item 时，模型把**对话里用户自己说的那句**当成
        // "用户这句话"念了出来（症状：用 AI 音色把用户输入重念一遍）；
        // 把稿写进 instructions 后 ⇒ 逐字念稿（✔）。
        instructions: READ_VERBATIM_HINT +
          "\n\n只念下面这一段，不要念对话里其它任何内容：\n" + text,
      },
    }));
  }

  _connectDedicatedTts() {
    const epoch = ++this._ttsEpoch;
    this._ttsReady = false;
    this._ttsQueue = [];
    clearTimeout(this._ttsReadyTimer);
    this._ttsReadyTimer = setTimeout(() => {
      if (epoch === this._ttsEpoch && this._useDedicatedTts && !this._ttsReady) {
        this._fallbackFromDedicatedTts("session setup timeout");
      }
    }, Math.max(1000, Number(process.env.ALI_TTS_SETUP_TIMEOUT_MS) || 8000));
    this._ttsReadyTimer.unref?.();
    const url = ALI_WS_BASE + encodeURIComponent(DEDICATED_TTS_MODEL);
    const socket = new WebSocket(url, { headers: { Authorization: "Bearer " + this._apiKey } });
    this._ttsRaw = socket;
    this._log(
      `[ALI] connecting dedicated TTS model=${DEDICATED_TTS_MODEL} voice=${DEDICATED_TTS_VOICE}`
    );
    socket.on("open", () => {
      if (epoch !== this._ttsEpoch) return;
      this._sendTtsEvent("session.update", {
        session: {
          voice: DEDICATED_TTS_VOICE,
          mode: "server_commit",
          response_format: "opus",
          language_type: "Auto",
        },
      });
    });
    socket.on("message", (raw) => {
      if (epoch !== this._ttsEpoch) return;
      this._handleDedicatedTtsMessage(raw);
    });
    socket.on("error", (error) => {
      if (epoch !== this._ttsEpoch) return;
      this._fallbackFromDedicatedTts(`socket error: ${error?.message || "unknown"}`);
    });
    socket.on("close", () => {
      if (epoch !== this._ttsEpoch || !this._useDedicatedTts) return;
      this._fallbackFromDedicatedTts("socket closed");
    });
  }

  _sendTtsEvent(type, fields = {}) {
    const event = {
      event_id: `voice_${++this._ttsEventCounter}_${randomUUID()}`,
      type,
      ...fields,
    };
    const payload = JSON.stringify(event);
    const socketOpen = this._ttsRaw && this._ttsRaw.readyState === WebSocket.OPEN;
    if (socketOpen && (this._ttsReady || type === "session.update")) {
      try { this._ttsRaw.send(payload); } catch (error) {
        this._fallbackFromDedicatedTts(`send failed: ${error?.message || "unknown"}`);
      }
    } else {
      this._ttsQueue.push(payload);
    }
  }

  _flushTtsQueue() {
    if (!this._ttsRaw || this._ttsRaw.readyState !== WebSocket.OPEN || !this._ttsReady) return;
    const queued = this._ttsQueue.splice(0);
    for (const item of queued) {
      try { this._ttsRaw.send(item); } catch (error) {
        this._fallbackFromDedicatedTts(`queued send failed: ${error?.message || "unknown"}`);
        return;
      }
    }
  }

  _handleDedicatedTtsMessage(raw) {
    const event = this._parseProviderEvent(raw);
    if (!event) return;
    if (event.type === "session.created") return;
    if (event.type === "session.updated") {
      clearTimeout(this._ttsReadyTimer);
      this._ttsReady = true;
      this._flushTtsQueue();
      this._maybeEmitSessionCreated();
      return;
    }
    if (event.type === "error") {
      this._fallbackFromDedicatedTts(
        String(event.error?.message || event.message || "provider error")
      );
      return;
    }
    if (event.type === "response.audio.delta") this._ttsHasAudio = true;
    if (event.type === "response.done") {
      this._ttsInResponse = false;
      this._ttsHasAudio = false;
      this._pendingTtsText = "";
    }
    const translated = this._translateTtsEvent(event);
    if (translated) this._emit("message", translated);
  }

  _translateTtsEvent(event) {
    switch (event.type) {
      case "response.audio.delta":
        return JSON.stringify({ ...event, type: "response.output_audio.delta", delta: event.delta || "" });
      case "response.audio.done":
        return JSON.stringify({ ...event, type: "response.output_audio.done" });
      case "response.audio_transcript.delta":
        return JSON.stringify({ ...event, type: "response.output_text.delta", delta: event.delta || "" });
      case "response.audio_transcript.done":
        return JSON.stringify({ ...event, type: "response.output_text.done" });
      default:
        return JSON.stringify(event);
    }
  }

  _fallbackFromDedicatedTts(reason) {
    if (!this._useDedicatedTts) return;
    const pendingText = this._pendingTtsText;
    const canRetryWholePhrase = !!pendingText && !this._ttsHasAudio;
    this._useDedicatedTts = false;
    clearTimeout(this._ttsReadyTimer);
    this._ttsReady = false;
    this._ttsInResponse = false;
    this._ttsQueue = [];
    this._pendingTtsText = "";
    this._ttsHasAudio = false;
    const socket = this._ttsRaw;
    this._ttsEpoch += 1;
    this._ttsRaw = null;
    try { socket && socket.close(); } catch (_) { /* ignore */ }
    this._log(`[ALI] dedicated TTS disabled; fallback=${canRetryWholePhrase ? "phrase" : "realtime"} reason=${String(reason).slice(0, 100)}`);
    this._maybeEmitSessionCreated();
    if (canRetryWholePhrase) this._sendReadAloud(pendingText);
    else if (pendingText) this._emit("message", JSON.stringify({
      type: "error",
      error: { message: "Dedicated TTS stopped after partial audio; retry the turn." },
    }));
  }

  _restartDedicatedTtsAfterCancel() {
    const socket = this._ttsRaw;
    this._ttsEpoch += 1;
    clearTimeout(this._ttsReadyTimer);
    this._ttsRaw = null;
    this._ttsReady = false;
    this._ttsQueue = [];
    this._ttsInResponse = false;
    this._ttsHasAudio = false;
    this._pendingTtsText = "";
    try { socket && socket.close(); } catch (_) { /* ignore */ }
    this._log("[ALI] dedicated TTS interrupted; reconnecting for next phrase");
    this._connectDedicatedTts();
  }

  _closeDedicatedTts() {
    const socket = this._ttsRaw;
    this._ttsEpoch += 1;
    clearTimeout(this._ttsReadyTimer);
    this._ttsRaw = null;
    this._ttsReady = false;
    this._ttsQueue = [];
    this._useDedicatedTts = false;
    try { socket && socket.close(); } catch (_) { /* ignore */ }
  }

  _maybeEmitSessionCreated() {
    if (this._sessionCreatedEmitted || !this._mainSessionReady) return;
    this._sessionCreatedEmitted = true;
    this._emit("message", JSON.stringify({
      type: "session.created",
      session: { id: this._mainSessionId || undefined },
    }));
  }

  _isProviderEvent(raw, expectedType) {
    return this._parseProviderEvent(raw)?.type === expectedType;
  }

  _parseProviderEvent(raw) {
    try {
      return JSON.parse(typeof raw === "string" ? raw : raw.toString("utf-8"));
    } catch (_) {
      return null;
    }
  }

  // ── 内部 ────────────────────────────────────────────────

  _queueOrSend(str) {
    if (this._raw && this._raw.readyState === WebSocket.OPEN) this._rawSend(str);
    else this._sendQueue.push(str);
  }

  _rawSend(str) {
    try { this._raw.send(str); } catch (e) { this._log(`[ALI] send failed: ${e.message}`); }
  }

  _emit(event, ...args) {
    for (const cb of this._handlers[event] || []) {
      try { cb(...args); } catch (e) { this._log(`[ALI] handler(${event}) threw: ${e.message}`); }
    }
  }

  /** 豆包 session.create 的 extension.* → 阿里 session.update 的扁平字段。 */
  _buildSessionUpdate(src) {
    const ext = src.extension || {};
    const directE2E = ext.direct_e2e === true;
    const doubaoSpeaker = ext.tts && ext.tts.speaker;
    const session = {
      modalities: ["text", "audio"],
      voice: this._voice || VOICE_MAP[doubaoSpeaker] || DEFAULT_VOICE,
      instructions: src.instructions || "",
      input_audio_format: "pcm",     // 16k s16le mono（两边一致）
      // Controlled mode must use opus: the existing downstream path expects Ogg Opus.
      // The opt-in direct E2E chat path uses raw PCM so the browser can start playback
      // on the first frame instead of waiting for an Ogg response to finish.
      // doubao-session-server.mjs
      // 的 AudioFlusher 硬编码 `format:"ogg_opus"`（豆包返回 Ogg Opus），前端
      // 也是 `new Blob([...],{type:'audio/ogg'})` + `new Audio()`/decodeAudioData。
      // 若用 "pcm"，阿里吐的是**裸 PCM**（实测 3.9s 就有 130560B；opus 只需 28750B），
      // 浏览器解不出 ⇒ **完全没声音**。实测 output_audio_format 合法集 = pcm|wav|mp3|opus。
      output_audio_format: (directE2E || ext.controlled_pcm === true) ? "pcm" : "opus",
      // 打开输入转写：ASR 事件名与豆包完全同名，上层无需感知。
      input_audio_transcription: { model: this._asrModel },
      // Controlled mode uses manual commits and cannot auto-generate replies.
      // Direct E2E chat opts into server VAD and provider auto-response above.
      turn_detection: directE2E ? {
        type: "server_vad",
        threshold: 0.5,
        prefix_padding_ms: 300,
        silence_duration_ms: Math.max(
          300,
          Math.min(3000, Number(ext.direct_e2e_vad_ms) || 1000),
        ),
      } : null,
      max_history_turns: this._maxHistoryTurns,
    };
    // 上游用 instruction 做角色设定；这里保留 isFirst 语义不做特殊处理。
    this._log(`[ALI] session.update voice=${session.voice} asr=${this._asrModel} directE2E=${directE2E}`);
    return { type: "session.update", session, directE2E };
  }

  /**
   * 阿里事件 → 豆包形状。返回 JSON 字符串；返回 null 表示吞掉。
   * 只翻"语义同名但字段/名字不同"的那些，其余原样透传。
   */
  _translateInbound(raw) {
    let msg = null;
    try {
      msg = JSON.parse(typeof raw === "string" ? raw : raw.toString("utf-8"));
    } catch (_) {
      return null;
    }
    const t = msg.type || "";

    switch (t) {
      // 阿里一连上就发 session.created（那时我们还没配 session）。
      // 上层把它当作"真正就绪"的信号，所以吞掉它、改在 session.updated 时发，
      // 保证"就绪"= 配置已生效。
      case "session.created":
        return null;
      case "session.updated":
        return JSON.stringify({
          type: "session.created",
          session: { id: msg.session && msg.session.id ? msg.session.id : undefined },
        });
      case "input_audio_buffer.speech_stopped":
        this._bufBytes = 0;
        return JSON.stringify(msg);

      // ASR 增量：阿里是 text(已确认前缀)+stash(待确认后缀)，
      // 豆包上层读的是 transcript/delta ⇒ 补齐成完整累计文本。
      case "conversation.item.input_audio_transcription.delta": {
        const full = (msg.text || "") + (msg.stash || "");
        if (!full) return null;
        return JSON.stringify({ ...msg, type: t, transcript: full, delta: full });
      }

      // 音频：阿里 response.audio.delta(字段 delta, PCM24k) → 豆包 response.output_audio.delta
      case "response.audio.delta":
        return JSON.stringify({ ...msg, type: "response.output_audio.delta", delta: msg.delta || "" });
      case "response.audio.done":
        return JSON.stringify({ ...msg, type: "response.output_audio.done" });

      // 输出文本：阿里的 audio_transcript.* → 豆包的 output_text.*
      // （上层对 *_text.* 一律丢弃，等于"只留转写，不要模型自由回复"）
      case "response.audio_transcript.delta":
        return JSON.stringify({ ...msg, type: "response.output_text.delta", delta: msg.delta || "" });
      case "response.audio_transcript.done":
        return JSON.stringify({ ...msg, type: "response.output_text.done" });

      case "session.finished":
        return JSON.stringify({ type: "session.closed" });

      // ⛔ 阿里这两类 error **必须吞掉**，否则前端每轮弹「回应没接住 · 再试一次」。
      // 上游 doubao-session-server.mjs:2563-2585 会把任何 provider error（除白名单外）
      // 直接转发成 `{type:"error"}`，前端收到就 resetReply + toast("回应没接住")。
      //   ① "Conversation has no active response." —— 我们发 response.cancel 但当时**没有
      //      在飞的响应**。豆包侧对"空取消"不报错；阿里报。而上游**每个用户回合都会发
      //      response.cancel**（豆包有自主回复要取消；我们 turn_detection:null 压根没有）
      //      ⇒ 不吞就是几乎每轮都弹。
      //   ② "Your session was closed because no response was generated for 180 seconds."
      //      —— 服务端空闲回收，与豆包 AudioServerNoAudioInputTooLongError 同类；上游白名单
      //      只认豆包文案。吞掉后由 socket close / session.finished 走既有重连路径。
      //   ③ "Input audio buffer exceeded maximum duration (30s). Please commit or
      //      clear the buffer." —— 输入缓冲撞上限。上面 append 分支已在 28s 主动
      //      commit，这里只是**兜底**（极端 burst 下的竞态），免得一条漏网的错误
      //      又把前端弹成"回应没接住"。
      // 真实错误（鉴权/参数/配额）仍然原样透传，不掩盖问题。
      case "error": {
        const em = String((msg.error && msg.error.message) || msg.message || "");
        if (/no active response|session was closed|no response was generated|exceeded maximum duration|please commit or clear the buffer/i.test(em)) {
          this._log(`[ALI] 吞掉可恢复 error: ${em.slice(0, 80)}`);
          return null;
        }
        return JSON.stringify(msg);
      }

      default:
        return JSON.stringify(msg);
    }
  }
}
