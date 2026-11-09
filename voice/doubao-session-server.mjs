/**
 * doubao-session-server.mjs（小程序适配版）
 *
 * 接入豆包实时语音 3.0（Seeduplex 全双工端到端语音大模型）。
 * 协议: wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue
 * 鉴权: X-Api-Key
 *
 * 改动点:
 *   1. 协议从旧版 Dialog SDK 二进制帧升级为 Seeduplex 3.0 JSON 事件帧
 *   2. 鉴权从 AppID/AccessKey/SecretKey 签名改为 X-Api-Key
 *   3. TTS 输出从 mp3 改为 pcm_s16le @ 24000Hz，代理端包成 WAV 下发给小程序
 *   4. token 透传：从 ws URL query 解析 token，调用后端 /session/start 时带上 Authorization 头
 *   5. 音频拼接（方案 B）：累积豆包下发的音频片段，达到阈值或事件边界时一次性下发
 *   6. 支持通过环境变量配置监听端口、后端 API 地址
 *   7. 增加更详细的日志（带时间戳和连接 ID）
 *
 * 部署: 见同目录 README.md
 *
 * 环境变量:
 *   DEMO_SESSION_PORT     监听端口，默认 3211
 *   DOUBAO_VOICE_API_KEY  豆包实时语音 3.0 API Key（控制台 > API Key 管理）
 *   BACKEND_API_HOST      后端 API 主机，默认 localhost
 *   BACKEND_API_PORT      后端 API 端口，默认 8000
 *   BACKEND_API_PROTOCOL  后端 API 协议，默认 http
 */

import { createServer } from "node:http";
import { request } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { WebSocketServer, WebSocket } from "ws";
import { URL, fileURLToPath } from "node:url";
import { existsSync, readFileSync, mkdirSync, writeFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { createDashscopeVoiceSocket } from "./voice-transport-dashscope.mjs";
import { ProviderAsrPcmBoundary } from "./asr-turn-boundary.mjs";

// 自动尝试加载 .env 文件
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const envCandidates = [
  join(__dirname, ".env"),
  join(__dirname, "..", "产品主文件", "api", ".env"),
  join(__dirname, "..", "api", ".env")
];
for (const envPath of envCandidates) {
  if (existsSync(envPath)) {
    try {
      const content = readFileSync(envPath, "utf-8");
      content.split("\n").forEach(line => {
        const match = line.match(/^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$/);
        if (match && match[1] && !process.env[match[1]]) {
          process.env[match[1]] = match[2].trim();
        }
      });
      break;
    } catch (_) {}
  }
}

// ── 配置 ────────────────────────────────────────────────
const PORT = Number(process.env.DEMO_SESSION_PORT || 3211);
const DOUBBAO_WS = "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue";
const DOUBAO_VOICE_API_KEY = process.env.DOUBAO_VOICE_API_KEY || "";

// 语音传输供应商：doubao（默认，行为与改动前完全一致）| dashscope（阿里 Qwen-Audio Realtime）。
// 只替换"吃用户音频（ASR）+ 播 AI 回复（TTS）"的模型；上层受控回复状态机与后端大脑都不动。
// 协议翻译全部在 voice-transport-dashscope.mjs 的 WS 门面里做。
const VOICE_TRANSPORT = String(process.env.VOICE_TRANSPORT || "doubao").trim().toLowerCase();
const DASHSCOPE_API_KEY = process.env.DASHSCOPE_API_KEY || "";
// Experimental latency path: only plain daily/free chat may bypass the
// per-turn Director → full text generation → separate TTS chain. Keep it off
// unless explicitly enabled; expression/material/scene tapes stay controlled.
const DIRECT_E2E_CHAT_ENABLED = /^(1|true|yes|on)$/i.test(
  String(process.env.VOICE_E2E_DIRECT_CHAT || "").trim(),
);
// Keep the opt-in experiment near the current chat pause window by default;
// set this to 1000 only when explicitly measuring the fastest provider path.
const DIRECT_E2E_VAD_MS = Math.max(
  300,
  Math.min(3000, Number(process.env.VOICE_E2E_DIRECT_CHAT_VAD_MS || 2200) || 2200),
);
const CONTROLLED_PCM_ENABLED = /^(1|true|yes|on)$/i.test(String(process.env.VOICE_CONTROLLED_PCM ?? "true"));
const DIRECT_PCM_FRAME_PREFIX = Buffer.from("|||PCM24|||");

/** 建到语音供应商的连接。dashscope 时返回协议翻译门面（形状与豆包 ws 一致）。 */
function openVoiceSocket() {
  if (VOICE_DISABLED) {
    throw new Error("语音服务未配置 API Key（纯文本模式）；请在 .env 里设置 DASHSCOPE_API_KEY 或 DOUBAO_VOICE_API_KEY 后重启");
  }
  if (VOICE_TRANSPORT === "dashscope") {
    return createDashscopeVoiceSocket({
      apiKey: DASHSCOPE_API_KEY,
      log: (m) => console.log(`[SS ${ts()}] ${m}`),
    });
  }
  return new WebSocket(DOUBBAO_WS, { headers: { "X-Api-Key": DOUBAO_VOICE_API_KEY } });
}

const BACKEND_HOST = process.env.BACKEND_API_HOST || "127.0.0.1";
const BACKEND_PORT = Number(process.env.BACKEND_API_PORT || 8002);
const BACKEND_PROTOCOL = process.env.BACKEND_API_PROTOCOL || "http:";
const VOICE_ARCHITECTURE_REVISION = "phrase-stream-duplex-v2";
const CONTROLLED_REPLY_DEADLINE_MS = Math.max(
  6000,
  Math.min(20000, Number(process.env.CONTROLLED_REPLY_DEADLINE_MS || 12000) || 12000),
);

// plans/0916-19 项 3：用户转写默认**不进日志**（plans/0915-01 隐私红线——
// "只要往日志里打 transcript，隐私设计就前功尽弃"）。默认只打长度等元信息，
// 排障时用 LOG_TRANSCRIPTS=1 临时打开全量。日志文件在服务器上是长期留存的，
// 所以开关默认必须是关。
const LOG_TRANSCRIPTS = /^(1|true|yes|on)$/i.test(String(process.env.LOG_TRANSCRIPTS || "").trim());
// 正文脱敏辅助：关时回一句"已隐去"，开时给前 limit 个字符（排障够用）。
function transcriptForLog(text, limit) {
  const s = String(text || "");
  return LOG_TRANSCRIPTS
    ? s.slice(0, limit)
    : `[正文已隐去 · 长度 ${s.length} · LOG_TRANSCRIPTS=1 可打印]`;
}

// 凭据检查 —— 开源包默认**没有**任何语音供应商 key。
// 原实现在这里直接 process.exit(1)：别人 clone 下来照 README 起服务，
// 代理起不来 ⇒ 前端所有 /api/* 全废（登录/建盘/素材/词块都走这个代理）。
// 改为「无 key → 纯文本模式」：
//   · HTTP /api/* 透传照常工作（四盘的建盘、素材、词块、会话全靠它）
//   · WS 语音连接会收到一条明确的错误，不会静默挂起
// 有 key 时行为与改动前逐字一致。
// 延迟账：这是启动期一次性分支，逐轮 0 次额外模型调用、0ms；有 key 时不进这个分支。
const VOICE_DISABLED = (VOICE_TRANSPORT === "dashscope")
  ? !DASHSCOPE_API_KEY
  : !DOUBAO_VOICE_API_KEY;
if (VOICE_DISABLED) {
  console.warn("[SS] 未配置语音 API Key → 以【纯文本模式】启动。");
  console.warn("[SS]   /api/* 透传、登录、建盘、素材库、词块本、会话均可用；");
  console.warn("[SS]   语音（ASR/TTS/点词发音）不可用。配好 .env 里的 key 后重启即恢复。");
} else if (VOICE_TRANSPORT === "dashscope") {
  console.log("[SS] 语音传输: dashscope（阿里 Qwen-Audio Realtime）；未使用豆包额度");
} else {
  console.log("[SS] 语音传输: doubao（豆包 Seeduplex）");
}


// ── ffmpeg 解析（plans/0916-24 §三·五 修复 3「音频日记」+ 视频转写）────────
// 只有非 WAV 输入才需要它（WAV 由 JS 自己重采样成 16k mono s16le）。
// 顺序：环境变量 FFMPEG_BIN 优先，其次 PATH 上的 ffmpeg。
// 为什么要写死路径：winget 装 ffmpeg 只改 PATH，而**已经在跑的进程不继承新 PATH**
// （服务是常驻进程，重启前的 shell 里没有 ffmpeg），所以 .env 里钉死 FFMPEG_BIN
// 才是可靠路线；PATH 只作为兜底（Linux 服务器上 apt 装完后走这条）。
const FFMPEG_BIN_RAW = String(process.env.FFMPEG_BIN || "").trim();
function probeBinary(bin) {
  if (!bin) return false;
  try {
    const r = spawnSync(bin, ["-version"], { stdio: "ignore", timeout: 8000, windowsHide: true });
    return !r.error && r.status === 0;
  } catch (_) {
    return false;
  }
}
function resolveFfmpegBin() {
  if (FFMPEG_BIN_RAW && probeBinary(FFMPEG_BIN_RAW)) return FFMPEG_BIN_RAW;
  if (FFMPEG_BIN_RAW) {
    console.warn(`[SS ${ts()}] WARN: FFMPEG_BIN 指向的 ffmpeg 不可用（${FFMPEG_BIN_RAW}），改试 PATH`);
  }
  if (probeBinary("ffmpeg")) return "ffmpeg";
  return null;
}
const FFMPEG_BIN = resolveFfmpegBin();
// 找不到 ffmpeg 时的**唯一**用户可见话术：明确说清"缺什么、怎么补"，绝不静默返回空文本。
const FFMPEG_MISSING_MSG =
  "这个格式需要 ffmpeg 转换（未找到 ffmpeg；装好或在 .env 里设 FFMPEG_BIN）";
console.log(
  `[SS ${ts()}] ffmpeg: ${FFMPEG_BIN
    ? FFMPEG_BIN
    : "未找到（只直接收 16k 单声道 s16le WAV，其它格式会明确报错而不是返回空转写）"}`
);

// WAV 音频拼接阈值（字节）。达到此大小、定时器到期、或遇到句子边界事件即下发
// 注：原值 32KB 过大，会导致 TTS 音频在用户开口前一直憋在缓冲区，表现为
//     "AI 文本出来了但音频不响，等用户说话才一次性播放，且断断续续"。
//     现降到 2KB，并配合定时 flush（FLUSH_INTERVAL_MS）实现近实时流式下发，
//     配合 APK 端 AudioTrack 流式播放，用户几乎与豆包同声听到回复。
const AUDIO_FLUSH_THRESHOLD = 2 * 1024;  // 2KB

// 音频流式 flush 间隔（毫秒）。push() 后启动定时器，到期强制下发，
// 避免小片段在缓冲区里堆积造成延迟。
const FLUSH_INTERVAL_MS = 120;

// The browser intentionally rotates the Doubao transport to recover from the
// provider's idle timeout.  That transport lifecycle is shorter than a
// learning conversation, so keep the FastAPI TutorAgent session alive across
// those rotations instead of rebuilding RAG/profile/director state each time.
const BACKEND_SESSION_CACHE_TTL_MS = 30 * 60 * 1000;
const BACKEND_SESSION_CACHE_MAX = 128;
const backendSessionCache = new Map();

function getCachedBackendSession(key) {
  if (!key) return null;
  const cached = backendSessionCache.get(key);
  if (!cached) return null;
  if (Date.now() - cached.updatedAt > BACKEND_SESSION_CACHE_TTL_MS) {
    backendSessionCache.delete(key);
    return null;
  }
  backendSessionCache.delete(key);
  cached.updatedAt = Date.now();
  backendSessionCache.set(key, cached);
  return cached;
}

function cacheBackendSession(key, value) {
  if (!key || !value?.sessionId || !value?.systemPrompt) return;
  backendSessionCache.delete(key);
  backendSessionCache.set(key, { ...value, updatedAt: Date.now() });
  while (backendSessionCache.size > BACKEND_SESSION_CACHE_MAX) {
    const oldestKey = backendSessionCache.keys().next().value;
    backendSessionCache.delete(oldestKey);
  }
}

// ── 工具函数（与原版一致，未改动） ─────────────────────
function createId(prefix) {
  return prefix + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
}

function ts() {
  return new Date().toISOString().split("T")[1].slice(0, 12);
}

function assistantIdentity(message) {
  if (message.role !== "assistant") return {};
  if (message.turn_identity_ambiguous === true) return { turn_identity_ambiguous: true };
  const id = message.turn_id;
  return typeof id === "string" && id.length <= 128 && id === id.trim()
    && !/[\u0000-\u001f]/.test(id) && id ? { turn_id: id } : {};
}

function backendHistoryMessage(item) {
  return { role: item.role, content: item.text, ...assistantIdentity(item) };
}

function cleanDialogContext(history) {
  if (!history || !Array.isArray(history)) return [];
  const clean = [];
  let expectedRole = "user";
  for (const m of history) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    // The review HTML sends `content`, while older native clients send `text`.
    // Dropping `content` here preserved the role labels but erased every
    // utterance, so a reconnect appeared to have history=2 while FastAPI
    // actually received two blank messages and lost the open role-play state.
    const messageText = String(m.text ?? m.content ?? "").trim();
    if (!messageText) continue;
    if (m.role === expectedRole) {
      clean.push({
        role: m.role,
        text: messageText,
        ...assistantIdentity(m),
        timestamp: m.timestamp || Math.floor(Date.now() / 1000)
      });
      expectedRole = expectedRole === "user" ? "assistant" : "user";
    } else {
      if (clean.length > 0) {
        const previous = clean[clean.length - 1];
        if (previous.role === "assistant" && (Object.keys(assistantIdentity(previous)).length
            || Object.keys(assistantIdentity(m)).length)) {
          delete previous.turn_id;
          previous.turn_identity_ambiguous = true;
        }
        previous.text += " " + messageText;
      }
    }
  }
  if (clean.length % 2 !== 0) {
    clean.pop();
  }
  // A reconnect must retain both the scene origin and the latest local state.
  // Keeping only the tail made the first role/goal disappear after ten QA
  // pairs, so a restarted voice socket could silently resume in a stale topic.
  if (clean.length <= 20) return clean;
  return clean.slice(0, 2).concat(clean.slice(-18));
}

function enrichUserMessage(text) {
  if (!text || !text.trim()) return text;

  const hasChinese = /[㐀-鿿]/.test(text);
  const hasEnglish = /[a-zA-Z]/.test(text);
  const lower = text.toLowerCase().trim();

  let tag = null;
  if (/^(嗨|哈喽|hello|hi|hey|哎|我又来了|好久不见|最近怎么样|what'?s up|how are you)/i.test(lower) ||
      /(早上好|下午好|晚上好|good morning|good evening|good night)/i.test(lower)) {
    tag = "寒暄";
  } else if (/[？?]$/.test(text) || /^(为什么|怎么|什么|哪里|谁|多少|几|how|what|why|where|when|who|is|are|do|does|can|could|would|should|will)/i.test(lower)) {
    tag = "提问";
  } else if (/^(练习|学习|复习|教我|告诉我|说说|talk about|practice|teach|learn|review|let'?s)/i.test(lower)) {
    tag = "练习";
  } else if (/^(聊天|闲聊|随便|无聊|chat|bored|random|free talk)/i.test(lower)) {
    tag = "闲聊";
  }

  let driftGuard = null;
  if (/(酒店|推荐.*酒店|约会|旅游|景点|写代码|编程|天气|数学|政治|总统|股票)/i.test(text)) {
    driftGuard = "用户暂时换了话题。先简短回应真实意图；若与学习目标有自然连接，再温和桥接回语言练习，不要生硬拒绝。";
  }

  let langInstr = null;
  if (hasChinese && !hasEnglish) {
    langInstr = "用户说中文。日常交流用中文，但涉及具体单词、语法、纠错时用英文+中文解释。";
  } else if (hasEnglish && !hasChinese) {
    langInstr = "User speaks English. Respond in English, use Chinese only for difficult explanations.";
  } else if (hasChinese && hasEnglish) {
    langInstr = "中英混合。日常用中文，语言点用英文+中文解释。";
  }

  console.log(`[SS ${ts()}] enrichUserMessage: ${transcriptForLog(text, 160)} -> lang:${langInstr ? "yes" : "no"} drift:${driftGuard ? "YES" : "NO"}`);

  const parts = [langInstr, tag ? "意图：" + tag : null, driftGuard].filter(Boolean);
  if (parts.length === 0) return text;
  return parts.join(" ") + "\n\n" + text;
}

function callBackendJson(path, payload, token = "", timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload || {});
    const req = request({
      hostname: BACKEND_HOST,
      port: BACKEND_PORT,
      path,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
      },
    }, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          if (res.statusCode >= 200 && res.statusCode < 300) resolve(parsed);
          else reject(new Error(`${path} returned ${res.statusCode}: ${data.slice(0, 200)}`));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on("error", reject);
    req.setTimeout(Math.max(1000, Number(timeoutMs) || 20000), () => {
      req.destroy(new Error(`${path} timed out after ${timeoutMs}ms`));
    });
    req.write(body);
    req.end();
  });
}

// 点词发音：豆包 Seeduplex 文本转语音（speech_text_buffer.commit），声线与
// AI 回复完全一致（DOUBAO_SPEAKER_EN）。provider 实际返回 Ogg Opus（与主链路
// 相同的"原样透传、客户端嗅探"），按魔数识别 mime；内存缓存（词 -> {audio,mime}）。
// §8.72：记住"问过没有 / 道别没有"，跨连接有效。
// 用户重新连上来 = 他回来了——不该再问一遍"还在吗"（真机实测会无限循环）。
// 唯一清除条件：他真的开口说话（noteUserSpeaking）。
const IDLE_ASK_BY_USER = new Map();  // userId -> { stage: 1|2, at: ts }
const VOCAB_TTS_CACHE = new Map();
const VOCAB_TTS_MAX = 300;
function sniffAudioMime(buf) {
  if (buf.length >= 4 && buf.toString("ascii", 0, 4) === "RIFF") return "audio/wav";
  if (buf.length >= 4 && buf.toString("ascii", 0, 4) === "OggS") return "audio/ogg";
  if (buf.length >= 3 && (buf.toString("ascii", 0, 3) === "ID3" || (buf[0] === 0xFF && (buf[1] & 0xE0) === 0xE0))) return "audio/mpeg";
  return "audio/ogg";
}
// 0916-18 修复 B：开场白复用这条声线（与 AI 回复同一说话人），所以上限从
// "点词 40 字"放宽到"整句 300 字"。句子给自然朗读指令，单词保持旧指令与旧行为
// （点词发音不受影响）。
const VOCAB_TTS_MAX_CHARS = 300;
// provider 的 TTS 会**带一大段尾部静音**返回（实测一句话 2.5s → 18s 音频）。
// 沉默问候只有一句，播 18 秒等于她念完还对空气发呆 15 秒，所以裁首尾静音。
// 失败一律返回原 buffer——宁可长一点，也不能没声音。
function trimTtsSilence(buf) {
  if (!FFMPEG_BIN || !buf || !buf.length) return buf;
  const stamp = `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  const inPath = join(tmpdir(), `alice_tts_${stamp}.ogg`);
  const outPath = join(tmpdir(), `alice_tts_${stamp}_trim.ogg`);
  try {
    writeFileSync(inPath, buf);
    const filter = "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,"
      + "areverse,silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB,areverse";
    const r = spawnSync(FFMPEG_BIN, [
      "-y", "-hide_banner", "-loglevel", "error",
      "-i", inPath, "-af", filter, "-c:a", "libopus", "-b:a", "32k", outPath,
    ], { timeout: 8000 });
    if (r.status !== 0 || !existsSync(outPath)) return buf;
    const out = readFileSync(outPath);
    return out.length ? out : buf;
  } catch (_) {
    return buf;
  } finally {
    try { existsSync(inPath) && unlinkSync(inPath); } catch (_) {}
    try { existsSync(outPath) && unlinkSync(outPath); } catch (_) {}
  }
}

function vocabTtsSynthesize(word) {
  const text = String(word || "").trim();
  if (!text || text.length > VOCAB_TTS_MAX_CHARS) return Promise.reject(new Error("invalid text"));
  const isSentence = text.split(/\s+/).filter(Boolean).length >= 3;
  const role = isSentence ? "Read the sentence aloud naturally." : "Read the given word aloud.";
  const key = text.toLowerCase();
  if (VOCAB_TTS_CACHE.has(key)) return Promise.resolve(VOCAB_TTS_CACHE.get(key));
  if (VOCAB_TTS_CACHE.size >= VOCAB_TTS_MAX) VOCAB_TTS_CACHE.clear();
  return new Promise((resolve, reject) => {
    let ws = null;
    let settled = false;
    const done = (err, value) => {
      if (settled) return;
      settled = true;
      try { ws && ws.close(); } catch (_) {}
      if (err) reject(err); else resolve(value);
    };
    // 整句合成比点词慢（文本更长、返回音频更长），超时上限跟着放宽
    const guard = setTimeout(() => done(new Error("vocab tts timeout")), isSentence ? 15000 : 8000);
    try { ws = openVoiceSocket(); }
    catch (e) { clearTimeout(guard); done(e); return; }
    const pcmChunks = [];
    ws.on("open", () => {
      ws.send(JSON.stringify({
        type: "session.create",
        session: {
          id: "vocab_tts_" + Date.now(),
          model: "1.2.6.1",
          instructions: role,
          extension: {
            asr: { audio_info: { format: "pcm_s16le", sample_rate: 16000, channel: 1 } },
            tts: {
              speaker: process.env.DOUBAO_SPEAKER_EN || "en_female_dacey_uranus_bigtts",
              audio_config: { channel: 1, format: "pcm_s16le", sample_rate: 24000 },
            },
            dialog: { bot_name: "VocabTTS", system_role: role, speaking_style: "neutral", dialog_id: "vocab_tts_" + Date.now() },
          },
        },
      }));
    });
    ws.on("message", (raw) => {
      let msg = null;
      try { msg = JSON.parse(raw.toString("utf-8")); } catch (_) { return; }
      if (msg.type === "session.created") {
        ws.send(JSON.stringify({ type: "speech_text_buffer.commit", text }));
      } else if (msg.type === "response.output_audio.delta") {
        if (msg.delta) pcmChunks.push(Buffer.from(msg.delta, "base64"));
      } else if (msg.type === "response.done" || msg.type === "session.closed") {
        clearTimeout(guard);
        if (!pcmChunks.length) { done(new Error("vocab tts no audio")); return; }
        const raw = Buffer.concat(pcmChunks);
        const entry = { audio: raw.toString("base64"), mime: sniffAudioMime(raw) };
        VOCAB_TTS_CACHE.set(key, entry);
        done(null, entry);
      } else if (msg.type === "error") {
        clearTimeout(guard);
        done(new Error(String(msg.error || "vocab tts error")));
      }
    });
    ws.on("error", (e) => { clearTimeout(guard); done(e); });
  });
}

// ── 音频/视频转写（plans/0916-24 §三·五 修复 3「音频日记」+ 视频转写）──────
// 输入：16k 单声道 s16le WAV（App 录音就是这个格式）或任何 ffmpeg 能解的音视频。
// 输出：{ ok:true, text } / { ok:false, error }（失败一律给中文原因，绝不产空带）。
//
// 消息序列照抄已验证可用的 test_doubao_audio_file.mjs：
//   session.create → 按 20ms/640B 实时推流 → input_audio_buffer.commit → 收转写。
// 与主语音链路的区别（这是"一次性离线转写"不是对话）：
//   1. 不要 AI 回复 —— 见到 response.* 立刻 response.cancel，转写 settle 后直接关 socket，
//      不为一轮被丢弃的对话付钱；
//   2. 转写按 provider VAD 分段，每段 completed 都可能不是全文（长录音里的停顿会切成多段），
//      所以只在"音频已推完 + 最后一次转写事件后静默 TRANSCRIBE_SETTLE_MS"才收尾。
const TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024;        // HTTP 请求体上限 25MB
const TRANSCRIBE_MAX_PCM_BYTES = 6 * 60 * 32000;      // 解码后上限 = 6 分钟（16k mono s16le 32KB/s）
const TRANSCRIBE_MIN_PCM_BYTES = 9600;                // 0.3s，比这短不可能有内容
const TRANSCRIBE_SETTLE_MS = 700;                     // 音频推完后的收尾静默窗口
const TRANSCRIBE_MAX_WAIT_MS = 10 * 60 * 1000;        // 总超时硬上限
const TRANSCRIBE_SILENCE_RMS = 40;                    // 16bit RMS 门限（≈ -58dBFS）
const FFMPEG_TIMEOUT_MS = 180000;                     // 单次转码超时

function looksLikeWav(buf) {
  return buf.length >= 12 &&
    buf.toString("ascii", 0, 4) === "RIFF" &&
    buf.toString("ascii", 8, 12) === "WAVE";
}

// WAV → 16k 单声道 s16le（任意采样率/声道/位深）。移植自 test_doubao_audio_file.mjs
// （已验通：22050Hz 单声道 WAV 也能正确转写），不依赖 ffmpeg。
function wavToPcm16k(buf) {
  if (buf.length < 44) throw new Error("WAV too small");
  const riff = buf.toString("ascii", 0, 4);
  if (riff !== "RIFF" && riff !== "RIFX") throw new Error("not RIFF/WAVE: " + riff);
  let audioFormat = -1, channels = 1, sampleRate = 16000, bits = 16, dataOffset = -1, dataSize = 0;
  let offset = 12;
  while (offset < buf.length - 8) {
    const id = buf.toString("ascii", offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === "fmt ") {
      audioFormat = buf.readUInt16LE(offset + 8);
      channels = buf.readUInt16LE(offset + 10) || 1;
      sampleRate = buf.readUInt32LE(offset + 12) || 16000;
      bits = buf.readUInt16LE(offset + 22) || 16;
    } else if (id === "data") {
      dataOffset = offset + 8;
      dataSize = Math.min(size, buf.length - dataOffset);
    }
    offset += 8 + size;
  }
  if (dataOffset < 0) throw new Error("no data chunk");
  if (audioFormat !== 1 && audioFormat !== 3) {
    // 压缩 WAV（ADPCM 等）解析出来是噪声，宁可让上层退回 ffmpeg
    throw new Error(`compressed WAV (audioFormat=${audioFormat})`);
  }
  const bytesPerSample = bits / 8;
  const frameCount = Math.floor(dataSize / (bytesPerSample * channels));
  const samples = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const p = dataOffset + (i * channels + c) * bytesPerSample;
      let v;
      if (bits === 16) v = buf.readInt16LE(p) / 32768;
      else if (bits === 24) {
        const raw = buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16);
        v = ((raw << 8) >> 8) / 8388608;
      } else if (bits === 32) v = buf.readInt32LE(p) / 2147483648;
      else v = (buf[p] - 128) / 128; // 8-bit
      acc += v;
    }
    samples[i] = acc / channels;
  }
  if (sampleRate === 16000) return floatToS16(samples);
  const ratio = 16000 / sampleRate;
  const outLen = Math.floor(samples.length * ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i / ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, samples.length - 1);
    const frac = pos - i0;
    const v = samples[i0] * (1 - frac) + samples[i1] * frac;
    out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
  }
  return Buffer.from(out.buffer);
}

function floatToS16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    out[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
  }
  return Buffer.from(out.buffer);
}

// 非 WAV → ffmpeg 转 16k 单声道 s16le（视频同一路径，-vn 丢掉视频流）。
//
// ⚠ 输入**必须落临时文件**，不能喂 stdin。2026-09-17 用用户的真机录音（手机录音 App
// 出的 .m4a）实测踩到：`-i pipe:0` 下 ffmpeg **退出码 0 但输出 0 字节**，stderr 只有一句
// `stream 0, offset 0x50: partial file` —— MP4/M4A/MOV 的 moov 在文件尾部，管道不可寻址
// 就读不到索引，于是"看起来成功、其实什么都没解出来"（前端会收到"没有可解码的音轨"这种
// 误导性原因）。同一文件改从**文件路径**输入立刻正常（12.48s → 399360 字节，正好 16k 单声道）。
// 手机录音就是这个容器，等于主用例；所以这里老老实实落临时文件，finally 里删干净。
// （输出侧保持 pipe:1 没问题：s16le 是裸流，不需要寻址。）
function ffmpegToPcm16k(buf) {
  return new Promise((resolve, reject) => {
    if (!FFMPEG_BIN) { reject(new Error(FFMPEG_MISSING_MSG)); return; }
    let tmpPath = "";
    try {
      tmpPath = join(
        tmpdir(),
        `zz-transcribe-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.bin`
      );
      writeFileSync(tmpPath, buf);
    } catch (e) {
      reject(new Error(`无法写入临时文件：${e.message}`));
      return;
    }
    const cleanup = () => {
      if (!tmpPath) return;
      try { unlinkSync(tmpPath); } catch (_) { /* 已被删/占用都无所谓 */ }
      tmpPath = "";
    };
    let proc;
    try {
      proc = spawn(FFMPEG_BIN, [
        "-hide_banner", "-loglevel", "error",
        "-i", tmpPath,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1",
      ], { windowsHide: true });
    } catch (e) { cleanup(); reject(e); return; }

    const out = [];
    const errChunks = [];
    let outBytes = 0;
    let settled = false;
    let guard = null; // 先声明：done() 可能在任何时刻被调用，不能踩 TDZ
    const done = (err, value) => {
      if (settled) return;
      settled = true;
      if (guard) clearTimeout(guard);
      try { proc.kill(); } catch (_) {}
      cleanup();
      if (err) reject(err); else resolve(value);
    };
    guard = setTimeout(
      () => done(new Error(`转码超时（${Math.round(FFMPEG_TIMEOUT_MS / 1000)} 秒）`)),
      FFMPEG_TIMEOUT_MS
    );
    proc.stdout.on("data", (c) => {
      outBytes += c.length;
      if (outBytes > TRANSCRIBE_MAX_PCM_BYTES + 32000) {
        done(new Error("音频/视频太长（超过 6 分钟）"));
        return;
      }
      out.push(c);
    });
    proc.stderr.on("data", (c) => { if (errChunks.length < 40) errChunks.push(c); });
    proc.on("error", (e) => done(new Error(`无法启动 ffmpeg：${e.message}`)));
    proc.on("close", (code) => {
      if (settled) return;
      if (code !== 0) {
        const detail = Buffer.concat(errChunks).toString("utf8").trim().split("\n").pop() || `exit ${code}`;
        done(new Error(`不是可识别的音频/视频文件（ffmpeg: ${detail.slice(0, 160)}）`));
        return;
      }
      if (!outBytes) { done(new Error("文件里没有可解码的音轨（可能是纯视频且无声音，或格式不支持）")); return; }
      done(null, Buffer.concat(out));
    });
    // 输入改成文件路径后 stdin 不再承载音频；关掉它免得父进程往没人读的管道里灌
    // 几十 MB（EPIPE 噪音）。ffmpeg 读的是文件，不需要 stdin。
    try { proc.stdin.end(); } catch (_) { /* 已关闭 */ }
  });
}

// 静音检测：整段 RMS 低于门限 → 明确报"没说话"，而不是让 provider 回一段空转写。
function pcm16Rms(buf) {
  let sum = 0;
  let n = 0;
  for (let i = 0; i + 1 < buf.length; i += 4) { // 每 2 个采样取 1 个，长音频也够快
    const v = buf.readInt16LE(i);
    sum += v * v;
    n++;
  }
  return n ? Math.sqrt(sum / n) : 0;
}

// 分段拼接：中文段之间不加空格，其余加空格（provider 的 VAD 会把长录音切成多段）。
function joinTranscriptSegments(segments) {
  let out = "";
  for (const raw of segments) {
    const s = String(raw || "").trim();
    if (!s) continue;
    // provider 偶尔会吐只有符号/换行的空段（"♪"、"\n"）；这类段没有任何字，丢掉，
    // 否则会拼出一份看着"有内容"其实是垃圾的素材带。
    if (!/[\p{L}\p{N}]/u.test(s)) continue;
    if (!out) { out = s; continue; }
    const prev = out[out.length - 1];
    const isCjk = (ch) => /[\u3400-\u9fff\uf900-\ufaff]/.test(ch);
    out += (isCjk(prev) || isCjk(s[0]) || /\s$/.test(out)) ? s : " " + s;
  }
  return out;
}

// 一次离线转写：推完 PCM → 收转写 → 立刻关 socket。
// 超时按"时长 + 余量"给（实时端点 ⇒ 耗时≈音频时长），并受硬上限约束。
function transcribePcmViaDoubao(pcm) {
  return new Promise((resolve) => {
    const durationMs = Math.round((pcm.length / 32000) * 1000);
    const timeoutMs = Math.min(
      TRANSCRIBE_MAX_WAIT_MS,
      Math.max(60000, Math.round(durationMs * 1.3) + 45000)
    );
    const sessionId = "transcribe_" + Date.now();
    let ws = null;
    let settled = false;
    let segments = [];
    let current = "";
    let sentBytes = 0;
    let sentAll = false;
    let committed = false;
    let streamStartAt = 0;
    let sendTimer = null;
    let settleTimer = null;
    let replyCanceled = false;
    let guard = null; // 先声明：finish() 可能在任何时刻被调用，不能踩 TDZ

    const finish = (err, text) => {
      if (settled) return;
      settled = true;
      if (guard) clearTimeout(guard);
      if (settleTimer) clearTimeout(settleTimer);
      if (sendTimer) clearInterval(sendTimer);
      try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch (_) {}
      if (err) resolve({ ok: false, error: String(err.message || err) });
      else resolve({ ok: true, text });
    };

    // 音频推完 + 最后一次转写事件后静默 SETTLE_MS → 收尾（多段录音不会被第一段 completed 提前截断）
    const scheduleSettle = () => {
      if (settled || !sentAll || !committed) return;
      if (settleTimer) clearTimeout(settleTimer);
      settleTimer = setTimeout(() => {
        const text = joinTranscriptSegments(segments.concat(current ? [current] : []));
        if (!text) finish(new Error("转写结果为空：这段音频里没有识别到语音内容（可能只有环境音或音乐）"));
        else finish(null, text);
      }, TRANSCRIBE_SETTLE_MS);
    };

    guard = setTimeout(
      () => finish(new Error(`转写超时（${Math.round(timeoutMs / 1000)} 秒内没拿到完整结果），请重试或换更短的音频`)),
      timeoutMs
    );

    try {
      ws = openVoiceSocket();
    } catch (e) { finish(e); return; }

    ws.on("open", () => {
      ws.send(JSON.stringify({
        type: "session.create",
        session: {
          id: sessionId,
          model: "1.2.6.1",
          instructions: "你是一位友好的语言学习外教。",
          extension: {
            asr: { audio_info: { format: "pcm_s16le", sample_rate: 16000, channel: 1 } },
            tts: {
              speaker: process.env.DOUBAO_SPEAKER_EN || "en_female_dacey_uranus_bigtts",
              audio_config: { channel: 1, format: "pcm_s16le", sample_rate: 24000 },
            },
            dialog: {
              bot_name: "Transcriber",
              system_role: "只做转写。",
              speaking_style: "neutral",
              dialog_id: sessionId,
            },
          },
        },
      }));
    });

    ws.on("message", (raw) => {
      if (settled) return;
      let msg = null;
      try { msg = JSON.parse(raw.toString("utf-8")); } catch (_) { return; }

      if (msg.type === "session.created") {
        // 按**真实墙钟**推流：每 tick 补上"从开流到现在应该发出去的字节数"。
        // 不照抄 test_doubao_audio_file.mjs 的"每 20ms 固定发 640B"——Windows 上
        // setInterval(20) 实际约 31ms 一跳（系统计时器粒度 15.6ms），固定 640B/跳
        // 会把推流拖成 ~1.55 倍实时，30 秒音频要 47 秒才发完，长录音还会先撞总超时。
        // 单条消息上限 3200B（100ms，与主链路静音保活帧同量级），一次 tick 最多补 8 条，
        // 避免事件循环被卡住后爆出一大串消息。
        streamStartAt = Date.now();
        sendTimer = setInterval(() => {
          if (settled) return;
          if (sentBytes >= pcm.length) {
            clearInterval(sendTimer);
            sendTimer = null;
            sentAll = true;
            try { ws.send(JSON.stringify({ type: "input_audio_buffer.commit" })); } catch (_) {}
            committed = true;
            scheduleSettle();
            return;
          }
          const targetBytes = Math.min(pcm.length, Math.floor((Date.now() - streamStartAt) * 32));
          let budget = 8;
          while (sentBytes < targetBytes && budget > 0) {
            const end = Math.min(sentBytes + 3200, targetBytes);
            const chunk = pcm.subarray(sentBytes, end);
            sentBytes = end;
            budget--;
            try {
              ws.send(JSON.stringify({
                type: "input_audio_buffer.append",
                audio: Buffer.from(chunk).toString("base64"),
              }));
            } catch (_) { /* close 分支会兜住 */ }
          }
        }, 20);
      } else if (msg.type === "conversation.item.input_audio_transcription.delta") {
        // 豆包下发的是"本段累计文本"，不能拼接，取更长的那份（与主链路同一处理）
        const incoming = (typeof msg.transcript === "string" && msg.transcript.length > 0)
          ? msg.transcript
          : (msg.delta || "");
        if (incoming.length > current.length) current = incoming;
        scheduleSettle();
      } else if (msg.type === "conversation.item.input_audio_transcription.completed") {
        const segmentText = String(msg.transcript || current || "").trim();
        current = "";
        if (segmentText) segments.push(segmentText);
        scheduleSettle();
      } else if (msg.type === "response.created" ||
                 msg.type === "response.output_text.delta" ||
                 msg.type === "response.output_audio.delta") {
        // 不要 AI 回复：见到任何回复事件就取消这一轮（转写段已经拿到，回复纯属浪费）
        if (!replyCanceled) {
          replyCanceled = true;
          try { ws.send(JSON.stringify({ type: "response.cancel" })); } catch (_) {}
        }
      } else if (msg.type === "error") {
        const detail = msg.error?.message || msg.error || msg.message || "provider error";
        finish(new Error(`豆包转写失败：${String(detail).slice(0, 160)}`));
      }
    });

    ws.on("error", (e) => finish(new Error(`连接豆包失败：${e.message}`)));
    ws.on("close", () => {
      if (settled) return;
      const text = joinTranscriptSegments(segments.concat(current ? [current] : []));
      if (text && sentAll) {
        // 自己关的/对端正常收尾：已有转写，收下
        finish(null, text);
      } else if (text) {
        finish(new Error("豆包会话提前关闭（音频没传完，转写不完整），请重试"));
      } else {
        finish(new Error("豆包会话提前关闭，没有拿到转写，请重试"));
      }
    });
  });
}

/**
 * 一次性转写：Buffer → { ok, text } / { ok, error }。
 * 绝不返回空文本的成功结果：空/静音/太短/缺 key/ffmpeg 缺失/provider 错/超时都给中文原因。
 */
async function transcribeAudio(buffer) {
  if (!DOUBAO_VOICE_API_KEY) {
    return { ok: false, error: "缺少豆包语音 API Key（DOUBAO_VOICE_API_KEY），无法转写" };
  }
  const buf = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer || []);
  if (!buf.length) return { ok: false, error: "音频是空的（0 字节），没有可转写的内容" };
  if (buf.length > TRANSCRIBE_MAX_BYTES) {
    return { ok: false, error: `文件太大（${(buf.length / 1048576).toFixed(1)}MB，上限 ${TRANSCRIBE_MAX_BYTES / 1048576}MB）` };
  }

  let pcm = null;
  if (looksLikeWav(buf)) {
    try {
      pcm = wavToPcm16k(buf);
    } catch (e) {
      // 压缩/异常 WAV：有 ffmpeg 就交给它，没有就明说
      if (!FFMPEG_BIN) {
        return { ok: false, error: `${FFMPEG_MISSING_MSG}（这个 WAV 也解析不了：${e.message}）` };
      }
      try { pcm = await ffmpegToPcm16k(buf); }
      catch (e2) { return { ok: false, error: `ffmpeg 转码失败：${e2.message}` }; }
    }
  } else {
    if (!FFMPEG_BIN) return { ok: false, error: FFMPEG_MISSING_MSG };
    try { pcm = await ffmpegToPcm16k(buf); }
    catch (e) { return { ok: false, error: `ffmpeg 转码失败：${e.message}` }; }
  }

  if (!pcm || !pcm.length) return { ok: false, error: "音频没有可解码的声音数据，没有可转写的内容" };
  if (pcm.length < TRANSCRIBE_MIN_PCM_BYTES) {
    return { ok: false, error: `音频太短（${(pcm.length / 32000).toFixed(1)} 秒，至少要 0.3 秒），没有可转写的内容` };
  }
  if (pcm.length > TRANSCRIBE_MAX_PCM_BYTES) {
    return { ok: false, error: "音频/视频太长（超过 6 分钟），转写按实时速度进行，请先裁剪后再导入" };
  }
  const rms = pcm16Rms(pcm);
  if (rms < TRANSCRIBE_SILENCE_RMS) {
    return { ok: false, error: `这段音频是静音（音量 RMS ${rms.toFixed(0)}，没有可识别的语音），请换一段有说话的录音` };
  }

  const res = await transcribePcmViaDoubao(pcm);
  if (!res.ok) return res;
  const text = String(res.text || "").trim();
  if (!text) return { ok: false, error: "转写结果为空：这段音频里没有识别到语音内容" };
  return { ok: true, text };
}

// multipart/form-data 解析（只取文件部分）。不引新依赖：只需按 boundary 切块、
// 从每块的头部里读 name/filename。前端用 <input type="file"> 发的就是这个格式。
function parseMultipartFile(body, boundary) {
  const delim = Buffer.from("--" + boundary);
  let idx = body.indexOf(delim);
  if (idx < 0) return null;
  idx += delim.length;
  const parts = [];
  while (idx < body.length) {
    // 结束标记 "--"（boundary 后紧跟的 --）
    if (body[idx] === 0x2d && body[idx + 1] === 0x2d) break;
    if (body[idx] === 0x0d && body[idx + 1] === 0x0a) idx += 2;
    else if (body[idx] === 0x0a) idx += 1;
    let headerEnd = body.indexOf("\r\n\r\n", idx);
    let sepLen = 4;
    if (headerEnd < 0) { headerEnd = body.indexOf("\n\n", idx); sepLen = 2; }
    if (headerEnd < 0) break;
    const headers = body.toString("utf8", idx, headerEnd);
    const bodyStart = headerEnd + sepLen;
    const nextDelim = body.indexOf(delim, bodyStart);
    if (nextDelim < 0) break;
    let dataEnd = nextDelim;
    if (dataEnd >= 2 && body[dataEnd - 2] === 0x0d && body[dataEnd - 1] === 0x0a) dataEnd -= 2;
    else if (dataEnd >= 1 && body[dataEnd - 1] === 0x0a) dataEnd -= 1;
    const nameMatch = /name="([^"]*)"/i.exec(headers);
    const fileMatch = /filename="([^"]*)"/i.exec(headers);
    parts.push({
      name: nameMatch ? nameMatch[1] : "",
      filename: fileMatch ? fileMatch[1] : "",
      data: body.subarray(bodyStart, dataEnd),
    });
    idx = nextDelim + delim.length;
  }
  if (!parts.length) return null;
  // 优先带 filename 的部分（浏览器 <input type="file"> 一定带），其次常见字段名。
  // 都不匹配就返回 null，让上层明确报"没找到文件"——不要退回"把第一个文本字段当音频"，
  // 那样只会得到一条把人带偏的 ffmpeg 报错。
  return parts.find((p) => p.filename) ||
    parts.find((p) => ["file", "audio", "video", "data"].includes(p.name.toLowerCase())) ||
    null;
}

function callBackendNdjson(path, payload, onEvent, token = "", timeoutMs = 20000) {
  let req = null;
  const promise = new Promise((resolve, reject) => {
    const body = JSON.stringify(payload || {});
    req = request({
      hostname: BACKEND_HOST,
      port: BACKEND_PORT,
      path,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
        "Accept": "application/x-ndjson",
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
      },
    }, (res) => {
      let pending = "";
      let errorBody = "";
      const success = res.statusCode >= 200 && res.statusCode < 300;
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        if (!success) {
          errorBody += chunk;
          return;
        }
        pending += chunk;
        let newline = pending.indexOf("\n");
        while (newline >= 0) {
          const line = pending.slice(0, newline).trim();
          pending = pending.slice(newline + 1);
          if (line) {
            try { onEvent(JSON.parse(line)); }
            catch (error) { reject(new Error(`${path} invalid NDJSON: ${error.message}`)); return; }
          }
          newline = pending.indexOf("\n");
        }
      });
      res.on("end", () => {
        if (!success) {
          reject(new Error(`${path} returned ${res.statusCode}: ${errorBody.slice(0, 200)}`));
          return;
        }
        const tail = pending.trim();
        if (tail) {
          try { onEvent(JSON.parse(tail)); }
          catch (error) { reject(new Error(`${path} invalid NDJSON tail: ${error.message}`)); return; }
        }
        resolve();
      });
    });
    req.on("error", reject);
    req.setTimeout(Math.max(1000, Number(timeoutMs) || 20000), () => {
      req.destroy(new Error(`${path} timed out after ${timeoutMs}ms`));
    });
    req.write(body);
    req.end();
  });
  return {
    promise,
    abort(reason = "stream_aborted") {
      if (req && !req.destroyed) req.destroy(new Error(reason));
    },
  };
}

// ── HTTP 健康检查与音频调试转存服务 ───────────────────────
const httpServer = createServer((req, res) => {
  // 允许跨域
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "*");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === "POST" && req.url === "/api/save-debug-audio") {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      try {
        const buf = Buffer.concat(chunks);
        const rootDir = join(__dirname, "..");
        const debugDir = join(rootDir, "debug_audio");
        const recDir = join(debugDir, "recordings");
        if (!existsSync(debugDir)) mkdirSync(debugDir, { recursive: true });
        if (!existsSync(recDir)) mkdirSync(recDir, { recursive: true });

        const lastPath = join(debugDir, "last_user_recording.wav");
        const histPath = join(recDir, `user_${Date.now()}.wav`);
        writeFileSync(lastPath, buf);
        writeFileSync(histPath, buf);
        console.log(`[SS ${ts()}] [DebugAudio] Saved user recording (${buf.length} bytes) to ${lastPath}`);

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, bytes: buf.length, path: lastPath }));
      } catch (err) {
        console.error(`[SS ${ts()}] [DebugAudio] Failed to save user recording:`, err.message);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: err.message }));
      }
    });
    return;
  }

  if (req.method === "POST" && req.url === "/api/vocab-lookup") {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
        const result = await callBackendJson("/vocab/lookup", {
          word: String(body.word || ""),
          sentence: String(body.sentence || ""),
        }, "", 20000);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: String(err?.message || err) }));
      }
    });
    return;
  }

  if (req.method === "POST" && req.url === "/api/vocab-tts") {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
        const entry = await vocabTtsSynthesize(String(body.word || ""));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, audio: entry.audio, mime: entry.mime }));
      } catch (err) {
        console.log(`[SS ${ts()}] [VocabTTS] failed: ${String(err?.message || err)}`);
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: String(err?.message || err) }));
      }
    });
    return;
  }

  // ── POST /api/transcribe（plans/0916-24 §三·五 修复 3「音频日记」+ 视频转写）──
  // 路径选择：必须由代理**原生**处理，且必须放在下面 /api/* 通用透传**之前** return。
  //   1. 后端（FastAPI 8002）没有这条路由，交给透传只会 404 —— 而"200 但没数据 /
  //      静默落到默认响应"这个坑在 opening-line、/api/chunks 上已经踩过两次；
  //   2. /api/transcribe 与后端现有命名（/conversations /chunks /vocab …）不冲突，
  //      将来若后端真要接这条也能加个显式 return 覆盖，不会歧义。
  // body：
  //   首选 multipart/form-data 字段 file（<input type="file"> 默认就是它，CORS 简单类型不需预检）；
  //   也接受裸 body（audio/* | video/* | application/octet-stream）——就是整个请求体当文件。
  // 上限：请求体 25MB（TRANSCRIBE_MAX_BYTES）。
  // 鉴权：与 save-debug-audio / vocab-lookup / vocab-tts 一致 —— 代理本地能力，不校验、
  //   也不转发 Authorization（不发明新鉴权）。
  // 返回：HTTP 200 + { ok:true, text } ；音频层面的失败（空/太短/静音/缺 key/provider 错/超时）
  //   同样 HTTP 200 + { ok:false, error:"中文原因" }；只有传输层面的两种异常用非 200：
  //   413（请求体超 25MB）、400（multipart 里没有文件部分）。
  if (req.method === "POST" && String(req.url || "").split("?")[0] === "/api/transcribe") {
    const ctype = String(req.headers["content-type"] || "");
    const chunks = [];
    let total = 0;
    let aborted = false;
    req.on("data", (c) => {
      if (aborted) return;
      total += c.length;
      if (total > TRANSCRIBE_MAX_BYTES) {
        aborted = true;
        console.log(`[SS ${ts()}] [Transcribe] rejected: body over ${TRANSCRIBE_MAX_BYTES} bytes`);
        res.writeHead(413, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
          ok: false,
          error: `文件太大（超过 ${TRANSCRIBE_MAX_BYTES / 1048576}MB 上限），请先裁剪或压缩后再导入`,
        }));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("error", () => {
      if (aborted) return;
      aborted = true;
      try {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: "上传中断（连接提前断开），请重试" }));
      } catch (_) {}
    });
    req.on("end", async () => {
      if (aborted) return;
      const started = Date.now();
      try {
        const body = Buffer.concat(chunks);
        let fileBuf = body;
        let filename = "";
        const bmatch = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(ctype);
        if (/multipart\/form-data/i.test(ctype)) {
          if (!bmatch) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ ok: false, error: "multipart/form-data 缺少 boundary" }));
            return;
          }
          const parsed = parseMultipartFile(body, (bmatch[1] || bmatch[2] || "").trim());
          if (!parsed) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ ok: false, error: "没有找到上传的文件（multipart 字段名请用 file）" }));
            return;
          }
          fileBuf = parsed.data;
          filename = parsed.filename || "";
        }
        const result = await transcribeAudio(fileBuf);
        console.log(
          `[SS ${ts()}] [Transcribe] ${result.ok ? "ok" : "failed"} ` +
          `bytes=${fileBuf.length} file=${transcriptForLog(filename, 80)} ` +
          `${result.ok
            ? `chars=${result.text.length} text=${transcriptForLog(result.text, 120)}`
            : `error=${result.error}`} ` +
          `cost=${Date.now() - started}ms`
        );
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        console.error(`[SS ${ts()}] [Transcribe] unexpected:`, err?.stack || err);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: `转写失败：${String(err?.message || err)}` }));
      }
    });
    return;
  }

  // ── 会话云同步桥（plans/0915-03）：页面与 3211 同源，浏览器侧不直连 8002，
  // 由代理转发登录/会话接口，避免 CORS。身份用 dev:<uuid> 设备登录，
  // Authorization 头透传给后端做 require_user 校验。
  function forwardBackend(method, path, auth, rawBody, res, timeoutMs = 30000) {
    const headers = { "Authorization": auth || "" };
    if (rawBody != null) {
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = Buffer.byteLength(rawBody);
    }
    const breq = request({
      hostname: BACKEND_HOST, port: BACKEND_PORT, path, method, headers,
    }, (bres) => {
      let data = "";
      bres.on("data", (c) => { data += c; });
      bres.on("end", () => {
        const code = bres.statusCode || 500;
        try {
          const parsed = data ? JSON.parse(data) : {};
          res.writeHead(code, { "Content-Type": "application/json" });
          res.end(JSON.stringify(parsed));
        } catch (error) {
          res.writeHead(502, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: false, error: `${path} invalid JSON: ${error.message}` }));
        }
      });
    });
    breq.on("error", (err) => {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: String(err?.message || err) }));
    });
    breq.setTimeout(timeoutMs, () => breq.destroy(new Error(`${path} timed out`)));
    if (rawBody != null) breq.write(rawBody);
    breq.end();
  }

  // POST /api/auth/dev-login  {code: "dev:<uuid>"} → 后端 /auth/wx_login
  if (req.method === "POST" && req.url === "/api/auth/dev-login") {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
        forwardBackend("POST", "/auth/wx_login", "",
          JSON.stringify({ code: String(body.code || ""), platform: "android" }), res, 15000);
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: String(err?.message || err) }));
      }
    });
    return;
  }

  // ── /api/* 通用透传（一条规则覆盖所有后端接口）──────────────────────────
  //   规则：路径开头 /api 剥掉后原样转给后端；Authorization 头原样透传；
  //         query 原样保留（只有 /api/conversations 的旧写法要改写路径）。
  //   GET  /api/conversations?userId=X            → /conversations/X
  //   GET  /api/conversations/X/hidden            → /conversations/X/hidden
  //   GET  /api/conversations/X/opening-line      → /conversations/X/opening-line
  //   GET  /api/chunks/X                          → /chunks/X
  //   POST /api/chunks/X/<chunkId>/status         → /chunks/X/<chunkId>/status
  //   POST /api/conversations?userId=X            → /conversations/X      （保存）
  //   POST /api/conversations/X/<convId>/hide     → 同路径（进回收站）
  //   POST /api/conversations/X/<convId>/restore  → 同路径（恢复）
  //   POST /api/conversations/X/<convId>/purge    → 同路径（彻底删除）
  //   POST /api/conversations/X/purge_expired     → 同路径
  //
  // 之所以从「两条写死的路由」→「只匹配 /api/conversations」→ 现在改成
  // 「/api 前缀通用」：每加一个后端接口都要在这里补一段，漏了就静默落到末尾的
  // 默认 200 JSON —— 前端拿到 200 但没数据，最难查（opening-line 这么废过一次，
  // /api/chunks 又废了一次，词块本页签因此全空）。现在只要后端加了 /api 路由，
  // 代理无需改动（0916-16 修复项 1）。
  // 注意：vocab-lookup / vocab-tts / save-debug-audio / auth/dev-login 在上面已
  // 显式 return（它们要改写路径或走代理自建逻辑），不会进这里；
  // 鉴权不放代理做——原样透传 Authorization，由后端 401（空头等同未登录）。
  if (req.url && (req.url === "/api" || req.url.startsWith("/api/"))) {
    const u = new URL(req.url, "http://localhost");
    const auth = String(req.headers.authorization || "");
    let backendPath;
    let forwardQuery = "";
    if (u.pathname === "/api/conversations") {
      // 旧写法：userId 走查询串（前端 pull / flush 仍在用）
      const userId = String(u.searchParams.get("userId") || "");
      if (!userId || !auth) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: "userId and Authorization required" }));
        return;
      }
      backendPath = `/conversations/${encodeURIComponent(userId)}`;
      u.searchParams.delete("userId");
      forwardQuery = u.searchParams.toString();
    } else if (u.pathname === "/api" || u.pathname === "/api/") {
      // 裸 /api 没有对应后端路由，直接 404 —— 不要掉进末尾的默认 200 JSON
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "no /api route specified" }));
      return;
    } else {
      // 通用写法：user_id 及子路径都在路径里，query（如 ?scene= / ?conv_id=）原样透传
      backendPath = u.pathname.replace(/^\/api/, "");
      forwardQuery = u.searchParams.toString();
    }
    if (forwardQuery) backendPath += "?" + forwardQuery;
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      // POST 空体也要发 "{}"：后端用 pydantic 模型收 body，真发空体会 422
      const raw = chunks.length
        ? Buffer.concat(chunks).toString("utf8")
        : (req.method === "POST" ? "{}" : null);
      forwardBackend(req.method, backendPath, auth, raw, res, 60000);
    });
    return;
  }

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    status: "ok",
    service: "doubao-session-server (miniprogram-adapted)",
    port: PORT,
    backend: `${BACKEND_PROTOCOL}//${BACKEND_HOST}:${BACKEND_PORT}`,
    voice_architecture_revision: VOICE_ARCHITECTURE_REVISION,
  }));
});

const wsServer = new WebSocketServer({ server: httpServer });

// ── PCM 音频拼接器（方案 B：减少小程序端写文件次数） ───────
/**
 * 每个客户端连接维护一个 AudioFlusher 实例。
 * 收到 response.output_audio.delta 时 push(解码后的 pcm16 buffer)，达到阈值、
 * 定时器到期、或调用 flush() 时，把裸 PCM 用 "|||AUDIO|||"
 * 分隔符协议一次性下发给客户端（由小程序端合并后包成 WAV）。
 *
 * 关键：必须配合定时 flush，否则 TTS 小片段会一直堆积，直到用户开口
 * 才被强制 flush，导致 "AI 音频滞后于文本、用户说话才响"。
 */
class AudioFlusher {
  constructor(clientWs, logTag) {
    this._ws = clientWs;
    this._tag = logTag;
    this._chunks = [];
    this._totalBytes = 0;
    this._seq = 0;
    this._timer = null;       // 定时 flush 句柄
    this._closed = false;     // ws 关闭后禁止再发
  }

  push(audioBuffer) {
    if (this._closed) return;
    this._chunks.push(audioBuffer);
    this._totalBytes += audioBuffer.length;
    if (this._totalBytes >= AUDIO_FLUSH_THRESHOLD) {
      // 达到阈值立即下发，并取消挂起的定时器（避免重复 flush）
      this._clearTimer();
      this.flush("size_threshold");
      return;
    }
    // 未达阈值：启动/续期定时器，到期强制下发，保证流式
    if (!this._timer) {
      this._timer = setTimeout(() => {
        this._timer = null;
        this.flush("interval");
      }, FLUSH_INTERVAL_MS);
    }
  }

  flush(reason) {
    if (this._closed) return;
    this._clearTimer();
    if (this._chunks.length === 0) return;
    const pcm = Buffer.concat(this._chunks);
    // 豆包双工端点实际返回 Ogg Opus 封装（即使请求 pcm_s16le 也一样），
    // 代理原样透传；APK 端嗅探首字节识别格式并流式播放。
    this._seq += 1;
    const hdr = Buffer.from(JSON.stringify({
      type: "audio",
      seq: this._seq,
      bytes: pcm.length,
      format: "ogg_opus",
    }));
    const sep = Buffer.from("|||AUDIO|||");
    this._ws.send(Buffer.concat([hdr, sep, pcm]));
    console.log(`[SS ${ts()}] ${this._tag} flushed pcm: ${pcm.length} bytes (seq=${this._seq}, reason=${reason}, chunks=${this._chunks.length})`);
    this._chunks = [];
    this._totalBytes = 0;
  }

  discard(reason) {
    this._clearTimer();
    const bytes = this._totalBytes;
    this._chunks = [];
    this._totalBytes = 0;
    if (bytes > 0) {
      console.log(`[SS ${ts()}] ${this._tag} discarded wav: ${bytes} bytes (reason=${reason})`);
    }
  }

  _clearTimer() {
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
  }

  /** ws 关闭时调用，阻止后续发送并清理定时器 */
  close() {
    this._closed = true;
    this._clearTimer();
  }
}

// ── WebSocket 连接处理 ─────────────────────────────────
wsServer.on("connection", (clientWs, req) => {
  if (VOICE_DISABLED) {
    // 纯文本模式：明确告诉前端「语音不可用」，别让它静默重连挂起。
    try {
      clientWs.send(JSON.stringify({
        type: "error",
        code: "voice_disabled",
        message: "语音服务未配置 API Key：当前为纯文本模式。四盘的建盘/素材/词块/会话仍可用；配好 .env 后重启可恢复语音。"
      }));
    } catch (_) {}
    try { clientWs.close(1000, "voice_disabled"); } catch (_) {}
    return;
  }

  // 从 ws URL query 解析 token（小程序连接时带 ?token=xxx）
  const parseUrl = new URL(req.url, "http://dummy");
  const token = parseUrl.searchParams.get("token") || "";
  const connectId = createId("conn_");
  const logTag = `[${connectId.slice(-8)}]`;

  // 空闲保活。
  //
  // 豆包在"长时间没有音频输入"后会直接关闭会话——实测约 72 秒，
  // 报 AudioServerNoAudioInputTooLongError。而前端在用户停止说话后就停止录音
  // （stopCapture），空闲期间没有任何音频上行，于是会话被杀；用户再说的话
  // 发给了已关闭的会话：既没被处理、前端也毫无显示（实测就是这个症状）。
  // 对策：空闲时主动喂静音帧，让会话不因"没输入"而被关。
  // 只在没有真实音频上行时补，避免与正常语音叠加。
  const KEEPALIVE_INTERVAL_MS = 20000;
  const KEEPALIVE_SILENCE_B64 = Buffer.alloc(3200).toString("base64"); // 100ms @16k mono s16le
  let lastAudioForwardedAt = 0;
  const keepaliveTimer = setInterval(() => {
    if (!doubaoWs || doubaoWs.readyState !== WebSocket.OPEN) return;
    if (Date.now() - lastAudioForwardedAt < KEEPALIVE_INTERVAL_MS * 0.8) return;
    try {
      doubaoWs.send(JSON.stringify({
        type: "input_audio_buffer.append",
        audio: KEEPALIVE_SILENCE_B64,
      }));
    } catch (_) { /* 保活失败不影响主流程 */ }
  }, KEEPALIVE_INTERVAL_MS);

  console.log(`[SS ${ts()}] ${logTag} Client connected, token=${token ? token.slice(0, 8) + "..." : "(none)"}`);

  let sessionId = createId("sess_");
  let backendSessionId = null;  // backend /session/start session_id; used for all backend API calls
  let backendCacheKey = "";
  let doubaoWs = null, pendingConfig = null, isDoubaoReady = false;
  let currentUserId = "frontend-user";
  let dialogId = "dialog_frontend-user";
  let currentScene = "日常对话";
  let directE2EEnabled = false;
  let directE2ETurnId = '';
  let directResponseId = '';
  let directResponseTurnId = '';
  let directResponseText = '';
  let directResponseActive = false;
  let directResponseDonePending = false;
  let directTranscriptWaitTimer = null;
  let directResponsePlaybackStarted = false;
  let directProviderSpeechStoppedAtMs = 0;
  let currentTargetLang = "en";
  let sessionReadyAtMs = null;
  let firstUserTurnSeen = false;
  let sessionEntryMode = "fallback";
  let currentDialogContext = [];
  // 音色兼容性兜底：若英语音色不被当前模型版本支持，首次报错时自动回退中文精品音色
  let activeSpeaker = null;
  let lastSystemPrompt = null;
  let sessionCreated = false;
  let speakerFallbackDone = false;
  const audioFlusher = new AudioFlusher(clientWs, logTag);

  // 收集对话文本，用于会话结束时存档 + 触发 audit（反馈回流闭环）
  let transcript = [];
  let pendingUserAsr = '';  // current provider segment; the client owns the real turn boundary
  let pendingUserSegments = []; // provider VAD may split one long learner turn into several segments
  let clientCommitRequested = false;
  let clientCommitTimer = null;
  // plans/1011-13: provider late callbacks are filtered by CLIENT PCM position,
  // not by transcript text.  Text equality alone both (a) misses duplicates whose
  // two ASR passes differ by a word, and (b) wrongly drops a learner who really
  // repeats the same short sentence.
  const providerAsrPcmBoundary = new ProviderAsrPcmBoundary();
  // Browser RMS-based silence detection can be held open indefinitely by room
  // noise, echo, or a proxy-distorted audio stream.  Once provider ASR has
  // completed a segment, keep a short merge window for natural multi-part
  // speech, then let the proxy close the learner turn even if the browser never
  // sends stop_speech.
  let providerTurnBoundaryTimer = null;
  let providerFirstSegmentCompletedAt = 0;
  // 0924-08：练习盘（context_mode=expression）的合同是**一轮一提交**（plans/0919-02
  // §4.6）——用户按「说完了」才算说完。但 provider 段（含阿里垫片 28s 主动 commit，
  // `voice-transport-dashscope.mjs:187`）走的是下面的 2.2s 静音自动关轮：用户练表达
  // 一口气讲 28s+，垫片切段后只要喘口气 >2.2s，代理就替他"定稿"并让 AI 开回，
  // 用户还在录 ⇒ 继续说被判成 barge_in。实测 2026-09-24 20:24（`_svc3211f_0924.log`）：
  // `输入缓冲达 28s` → 6s 后 `finalizing held learner turn`（147 字）→ AI 回复。
  // 修法 = **只对 expression 关掉这条自动关轮**：provider 段照样并入同一轮（分段对
  // 用户不可见，⇒ 练习盘实际**没有时长上限**），关轮只认客户端显式提交
  // （「说完了」= stop_speech，或前端 20s 静音兜底）。闲聊盘/素材盘原行为一字不动。
  let sessionCassetteMode = '';
  let sessionIsMaterialTape = false;
  // 闲聊盘 provider 分段只是中间段；与前端 4.2s 闲聊停顿窗对齐，避免
  // ASR 完成一小段后，代理先于前端把较长的连续讲述切成两轮。
  const PROVIDER_TURN_BOUNDARY_GRACE_MS = 4200;
  // P20（2026-09-22）：用户**还在组织语言**时（拖音"嗯/呃"、或英文里夹中文＝缺词求助）
  // 不要按常规 2.2s 就替他关轮。**只放宽这一段这一次的等待**；"谁来关轮"的判定、
  // `end_by_silence` 台账字段都不动 ⇒ 正常说完的那一轮与改动前逐字一致。
  // 数值与前端闲聊盘停顿窗对齐；长段 ASR 已有真实续说 delta 时会重开该窗口，
  // 不会因第一段出现得早而在后续段落刚开始时撞到旧 ceiling。
  const PROVIDER_TURN_BOUNDARY_THINKING_GRACE_MS = 4200;
  const PROVIDER_TURN_BOUNDARY_THINKING_CEILING_MS = 5200;
  let audioBytesSinceProviderFinal = 0;
  let turnPcmBytes = 0;
  // plans/0916-22 修复0：逐轮能量剖面（100ms 一格 RMS）。直播语音是前端裸流
  // 直发（cleanUserSamples 只作用于"录音保存"路径），头尾静音量只能在代理侧
  // 按原始 PCM 估算——用于回答"录音是否完整/是不是在语音中间被切断"。
  let turnEnergy = null;
  function profileTurnAudio(pcm) {
    if (!turnEnergy) turnEnergy = { acc: 0, count: 0, buckets: [] };
    let acc = turnEnergy.acc;
    let count = turnEnergy.count;
    const buckets = turnEnergy.buckets;
    for (let i = 0; i + 1 < pcm.length; i += 2) {
      const sample = pcm.readInt16LE(i);
      acc += sample * sample;
      count += 1;
      if (count >= 1600) { // 100ms @16kHz
        buckets.push(Math.sqrt(acc / count));
        acc = 0;
        count = 0;
        if (buckets.length > 1200) return; // 2 分钟封顶，防长按泄漏
      }
    }
    turnEnergy.acc = acc;
    turnEnergy.count = count;
  }
  function energyProfileMetrics() {
    if (!turnEnergy || !turnEnergy.buckets.length) return {};
    const buckets = turnEnergy.buckets;
    // s16 满幅 32767；RMS>300 视为有人声（只为报表，不影响任何播放链路）
    const loud = (v) => v > 300;
    let head = 0;
    while (head < buckets.length && !loud(buckets[head])) head += 1;
    let tail = buckets.length - 1;
    while (tail >= 0 && !loud(buckets[tail])) tail -= 1;
    return {
      head_trim_ms: head * 100,
      tail_trim_ms: (buckets.length - 1 - tail) * 100,
      end_mid_speech: tail === buckets.length - 1,
    };
  }
  function clearTurnEnergy() { turnEnergy = null; }
  let pendingAssistantText = '';  // 350/550 累积的 AI 回复文本，359 时固化
  let lastCompletedUserText = '';
  // 生成调用带回的"学习者意图规范英文"（权威文本）。非空表示本轮已经
  // 有了与模型同源的文本，独立精修端点就不再抢着覆盖它。
  let turnNormalizedUserText = '';
  // Default path: provider audio is transport-only; only controlled TTS reaches
  // the learner. The separately gated direct E2E chat path intentionally allows
  // provider audio through for latency comparison.
  let suppressCurrentResponse = true;
  let turnTiming = null;
  let controlPending = false;
  let controlledPcmEnabled = false;
  let controlledProviderResponseId = "";
  const controlledSeenResponseIds = new Set();
  let controlledTtsActive = false;
  let controlledReplyReady = null;
  let controlledStreamRequest = null;
  let controlledPhraseQueue = [];
  let controlledCurrentPhrase = null;
  let controlledBackendDone = false;
  let controlledReplyText = '';
  let controlledResponseAnnounced = false;
  let controlledTutorMoveSent = false;
  let controlledTurnFinalized = false;
  let controlledAwaitingPlaybackAck = false;
  let controlledPlaybackAckTimer = null;
  let autoResponseEnded = false;
  let awaitingAutoResponseDone = false;
  let controlledStartTimer = null;
  let activeTurnId = '';
  let lastCommitAt = 0;  // 提交节流：防止客户端提交与服务端 VAD 重复提交同一段音频
  let controlledRequestSeq = 0; // late backend replies must never overwrite a newer turn
  let controlledDeadlineTimer = null;
  let controlledDeadlineExtended = false;
  let lastSuccessfulControlledReply = null;
  let controlledFailureStreak = 0;
  let chatFallbackIndex = 0;
  let lastFinalAsrText = '';
  let lastFinalAsrAt = 0;

  function normalizeEchoText(value) {
    return String(value || '')
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '');
  }

  function playbackMayStillBeAudible() {
    return Boolean(
      controlledTtsActive || controlledAwaitingPlaybackAck ||
      (controlPending && controlledReplyText.trim())
    );
  }

  function isProbablePlaybackEcho(value) {
    const heard = normalizeEchoText(value);
    const spoken = normalizeEchoText(controlledReplyText || pendingAssistantText);
    if (!heard || !spoken) return false;
    // Provider ASR normally returns an accumulated substring of the phrase
    // currently coming out of the speaker.  Only suppress that direct overlap;
    // novel learner words must still pass through and confirm a real barge-in.
    return spoken.includes(heard) || (heard.length >= 8 && heard.includes(spoken));
  }

  function clearClientCommitTimer() {
    if (clientCommitTimer) {
      clearTimeout(clientCommitTimer);
      clientCommitTimer = null;
    }
  }

  function clearProviderTurnBoundaryTimer() {
    if (providerTurnBoundaryTimer) {
      clearTimeout(providerTurnBoundaryTimer);
      providerTurnBoundaryTimer = null;
    }
  }

  // P20（2026-09-22）：这一段像不像"他还在想 / 缺词"？只看**供应商 ASR 已有的段**。
  // 只认强信号（宁漏不误伤）：① 结尾是思考音（嗯/呃/唔/啊/额/唉/哦…）；
  // ② 结尾是逗号/顿号/分号/省略号（话说一半）；③ 英文段里夹了中文（缺词求助的典型形态）。
  // ⚠ **不把"没有句末标点"算进来** —— 中英文 ASR 常常整段不给标点，那样会每轮都多等。
  const _THINKING_TAIL_RE = /(嗯|呃|唔|啊|额|唉|哦)[。.!！?？…\s]*$/;
  const _OPEN_TAIL_RE = /[，,、;；:：…]$/;

  function providerTurnLooksUnfinished() {
    const joined = pendingUserSegments.join(' ').replace(/\s+/g, ' ').trim();
    if (!joined) return false;
    if (_THINKING_TAIL_RE.test(joined)) return true;
    if (_OPEN_TAIL_RE.test(joined)) return true;
    return /[A-Za-z]/.test(joined) && /[\u4e00-\u9fff]/.test(joined);
  }

  function scheduleProviderTurnBoundary(extendOnly = false) {
    if (providerTurnBoundaryTimer && extendOnly) {
      return;
    }
    clearProviderTurnBoundaryTimer();
    if (!providerFirstSegmentCompletedAt) {
      providerFirstSegmentCompletedAt = Date.now();
    }
    const elapsed = Date.now() - providerFirstSegmentCompletedAt;
    const thinking = providerTurnLooksUnfinished();
    const graceMs = thinking
      ? PROVIDER_TURN_BOUNDARY_THINKING_GRACE_MS
      : PROVIDER_TURN_BOUNDARY_GRACE_MS;
    const ceilingMs = PROVIDER_TURN_BOUNDARY_THINKING_CEILING_MS;
    const remainingMs = Math.max(600, Math.min(graceMs, ceilingMs - elapsed));
    providerTurnBoundaryTimer = setTimeout(() => {
      providerTurnBoundaryTimer = null;
      providerFirstSegmentCompletedAt = 0;
      if (!mergedUserTurn() || clientWs.readyState !== WebSocket.OPEN) return;
      clientCommitRequested = true;
      console.log(
        `[SS ${ts()}] ${logTag} provider ASR quiet for ` +
        `${graceMs}ms${thinking ? ' (unfinished/thinking)' : ''}; finalizing held learner turn`
      );
      finalizeClientCommittedTurn();
    }, remainingMs);
  }

  function appendUserSegment(value) {
    const text = String(value || '').trim();
    if (!text) return;
    const last = pendingUserSegments[pendingUserSegments.length - 1] || '';
    if (text === last || last.endsWith(text)) return;
    if (text.startsWith(last) && last) {
      pendingUserSegments[pendingUserSegments.length - 1] = text;
      return;
    }
    pendingUserSegments.push(text);
  }

  function repairPhoneticAsrSlips(text) {
    if (!text) return '';
    return String(text).replace(/\s+/g, ' ').trim();
  }

  function mergedUserTurn(extra = '') {
    appendUserSegment(extra);
    const raw = pendingUserSegments.join(' ').replace(/\s+/g, ' ').trim();
    return repairPhoneticAsrSlips(raw);
  }

  function previewUserTurn(extra = '') {
    const raw = pendingUserSegments.concat(String(extra || '').trim() || [])
      .join(' ').replace(/\s+/g, ' ').trim();
    return repairPhoneticAsrSlips(raw);
  }

  // 按轮次回写用户转写文本。
  //
  // 不能写成"找最后一条 user 行"：精修/规范化都是异步的（精修还带
  // 0/1/3/5 秒退避重试）。若学习者在结果返回前又说了下一句，那条新行就成了
  // "最后一条 user 行"，于是被上一轮的结果覆盖——用户真实问过的内容被抹掉，
  // 系统只能拿上一句再答一遍（实测：问"洗手间有没有纸"被答成"洗手间在哪"）。
  // 有 turnId 就必须按 turnId 命中；没有才退回旧行为。
  function setUserTurnText(turnId, text) {
    const target = String(turnId || '');
    if (target) {
      for (let i = transcript.length - 1; i >= 0; i--) {
        const row = transcript[i];
        if (row.role === 'user' && row.turnId === target) {
          row.content = text;
          return true;
        }
      }
      // 只回退到"本来就没有 turnId"的旧行。若每行都有 turnId 却都不是本轮，
      // 说明这一轮已经不在 transcript 里了，此时宁可不写——写下去就是覆盖别人。
      for (let i = transcript.length - 1; i >= 0; i--) {
        const row = transcript[i];
        if (row.role === 'user' && !row.turnId) {
          row.content = text;
          return true;
        }
      }
      return false;
    }
    for (let i = transcript.length - 1; i >= 0; i--) {
      if (transcript[i].role === 'user') {
        transcript[i].content = text;
        return true;
      }
    }
    return false;
  }

  // 用户转写精修：把 ASR 原文结合当前场景和近期对话语境，由后端 LLM 精确纠偏
  // 重写成地道自然英语，原位替换前端文本气泡，并更新会话记忆与评估文本。
  // 重试直到成功（带退避）；全部重试耗尽才用原文兜底，保证文本气泡永不空白。
  const REFINE_RETRY_DELAYS_MS = [0, 1000, 3000, 5000];
  function refineUserTranscript(turnId, rawText) {
    const text = String(rawText || '').trim();
    if (!text) return;
    const attemptOnce = async () => {
      try {
        const res = await callBackendJson('/session/turn/refine', {
          raw_text: text,
          target_lang: currentTargetLang || 'en',
          scene: currentScene || 'daily_talk',
          recent_dialogue: transcript.slice(-4),
        }, token, 25000);
        const refined = String(res?.refined_text || '').trim();
        if (!refined) throw new Error('empty refine result');
        // plans/0915-23：后端判定精修越界时会把原话回传（fallback=true），
        // 这一样是权威值，只是要在日志里可数。
        const refineRejected = !!res?.fallback;
        // 生成调用已经给出权威文本时不要回写，避免把模型真正读到的
        // 意图覆盖成另一路精修结果，造成三方不一致。
        if (turnNormalizedUserText) return true;
        // 精修文本只改前端气泡，不回写 lastCompletedUserText：
        // 后者会喂给 /session/turn 的纠错记忆（process_turn），若把精修改对
        // 的文本喂回去，学习者的真实错误就被洗白，纠错链路永久空转
        // （真机实录："i watching ... in bilibili" 被洗成 "I'm watching ... on Bilibili"）。
        // Refinement is display text, not authoritative evidence of what was said.
        // Keep the original ASR as the session-fact source.
        callBackendJson('/session/turn/note-fact', {
          session_id: backendSessionId || sessionId,
          text,
        }, token).catch(() => {});
        console.log(
          `[SS ${ts()}] ${logTag} refine turn=${turnId || '(none)'}`
          + (refineRejected ? ` REJECTED(${String(res?.reason || '')})` : '')
          // plans/0916-19 项 3：精修前后的正文同样受开关管（它也是用户原话）
          + (LOG_TRANSCRIPTS
            ? ` raw="${text.slice(0, 90)}" -> "${refined.slice(0, 90)}"`
            : ` raw(${String(text || '').length} chars) -> refined(${refined.length} chars) [正文已隐去 · LOG_TRANSCRIPTS=1 可打印]`)
        );
        setUserTurnText(turnId, refined);
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify({
            type: 'asr_refined',
            turnId: turnId || '',
            text: refined,
            raw: text,
            fallback: refineRejected,
          }));
        }
        return true;
      } catch (error) {
        console.log(`[SS ${ts()}] ${logTag} refine attempt failed: ${String(error?.message || error).slice(0, 120)}`);
        return false;
      }
    };
    (async () => {
      for (const delay of REFINE_RETRY_DELAYS_MS) {
        if (delay) await new Promise((r) => setTimeout(r, delay));
        if (clientWs.readyState !== WebSocket.OPEN) return;
        if (await attemptOnce()) return;
      }
      console.log(`[SS ${ts()}] ${logTag} refine failed after all retries; falling back to raw transcript`);
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(JSON.stringify({
          type: 'asr_refined',
          turnId: turnId || '',
          text,
          raw: text,
          fallback: true,
        }));
      }
    })();
  }

  function finalizeClientCommittedTurn(extra = '') {
    if (!clientCommitRequested) return;
    clearClientCommitTimer();
    clearProviderTurnBoundaryTimer();
    providerFirstSegmentCompletedAt = 0;
    const finalText = mergedUserTurn(extra || pendingUserAsr);
    providerAsrPcmBoundary.noteClientFinal(finalText);
    const mergedSegments = Math.max(1, pendingUserSegments.length);
    pendingUserAsr = '';
    pendingUserSegments = [];
    clientCommitRequested = false;
    audioBytesSinceProviderFinal = 0;
    if (!turnTiming) beginTurnTiming();
    turnTiming.asr_final_at_ms = Date.now();
    turnTiming.asr_input_duration_ms = turnPcmBytes > 0
      ? Math.round(turnPcmBytes / 32) : null;
    turnPcmBytes = 0;
    // plans/0916-22 修复0：录音完整性台账（合并段数 + 头尾静音估算）
    turnTiming.asr_segments_merged = mergedSegments;
    Object.assign(turnTiming, energyProfileMetrics());
    clearTurnEnergy();
    if (!firstUserTurnSeen && sessionReadyAtMs) {
      turnTiming.first_user_speech_latency_ms = Math.max(
        0, turnTiming.asr_final_at_ms - sessionReadyAtMs
      );
      firstUserTurnSeen = true;
    }
    if (!finalText) return;
    const duplicateFinal = finalText === lastFinalAsrText && Date.now() - lastFinalAsrAt < 5000;
    if (duplicateFinal) {
      console.log(`[SS ${ts()}] ${logTag} ignored duplicate client-committed ASR final (${finalText.length} chars)`);
      return;
    }
    lastFinalAsrText = finalText;
    lastFinalAsrAt = Date.now();
    activeTurnId = createId("turn_");
    lastCompletedUserText = finalText;
    turnNormalizedUserText = '';
    controlledRetryCount = 0;
    controlledDeadlineExtended = false;
    transcript.push({ role: 'user', content: finalText, ts: Date.now(), turnId: activeTurnId });
    console.log(`[SS ${ts()}] ${logTag} user turn ${activeTurnId} ASR final (${finalText.length} chars): ${transcriptForLog(finalText, 160)}`);
    clientWs.send(JSON.stringify({ type: "asr_ended", text: finalText, turnId: activeTurnId }));
    // 精修不再抢跑：生成调用会顺带给出"学习者意图的规范英文"，只有它缺失时
    // 才回退到独立精修端点（见 requestControlledReply 的完成/失败分支）。
    requestControlledReply(finalText, false, activeTurnId);
  }

  function finalizeDirectE2EUserTurn(extra = '') {
    if (!directE2EEnabled) return;
    clearClientCommitTimer();
    clearProviderTurnBoundaryTimer();
    providerFirstSegmentCompletedAt = 0;
    const finalText = mergedUserTurn(extra || pendingUserAsr);
    providerAsrPcmBoundary.noteClientFinal(finalText);
    const mergedSegments = Math.max(1, pendingUserSegments.length);
    pendingUserAsr = '';
    pendingUserSegments = [];
    clientCommitRequested = false;
    audioBytesSinceProviderFinal = 0;
    if (!turnTiming || turnTiming.response_end_at_ms) beginTurnTiming();
    turnTiming.asr_final_at_ms = Date.now();
    turnTiming.asr_input_duration_ms = turnPcmBytes > 0
      ? Math.round(turnPcmBytes / 32) : null;
    turnPcmBytes = 0;
    turnTiming.asr_segments_merged = mergedSegments;
    Object.assign(turnTiming, energyProfileMetrics());
    clearTurnEnergy();
    if (!firstUserTurnSeen && sessionReadyAtMs) {
      turnTiming.first_user_speech_latency_ms = Math.max(
        0, turnTiming.asr_final_at_ms - sessionReadyAtMs
      );
      firstUserTurnSeen = true;
    }
    if (!finalText) return;
    const inputTurnId = directE2ETurnId || createId("turn_");
    const duplicateTurnFinal = transcript.some((row) =>
      row.role === 'user' && row.turnId === inputTurnId
    );
    if (duplicateTurnFinal) {
      console.log(`[SS ${ts()}] ${logTag} ignored duplicate direct E2E final for ${inputTurnId}`);
      return;
    }
    lastFinalAsrText = finalText;
    lastFinalAsrAt = Date.now();
    activeTurnId = inputTurnId;
    directE2ETurnId = activeTurnId;
    lastCompletedUserText = finalText;
    turnNormalizedUserText = '';
    transcript.push({ role: 'user', content: finalText, ts: Date.now(), turnId: activeTurnId });
    console.log(`[SS ${ts()}] ${logTag} direct E2E user turn ${activeTurnId} ASR final (${finalText.length} chars): ${transcriptForLog(finalText, 160)}`);
    clientWs.send(JSON.stringify({ type: "asr_ended", text: finalText, turnId: activeTurnId }));
    refineUserTranscript(activeTurnId, finalText);
    if (directResponseDonePending && directResponseTurnId === directE2ETurnId) {
      if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
      directTranscriptWaitTimer = null;
      announceDirectE2EReplyDone();
    }
  }

  function directProviderResponseId(msg) {
    return String(msg?.response_id || msg?.response?.id || '').trim();
  }

  function directProviderEventMatches(msg) {
    const eventResponseId = directProviderResponseId(msg);
    return Boolean(directResponseId && directResponseActive && eventResponseId === directResponseId);
  }

  function abandonDirectE2EReply(reason = 'barge_in', notifyClient = true) {
    const abandonedTurnId = directResponseTurnId;
    if (!abandonedTurnId && !directResponseId && !controlledAwaitingPlaybackAck) return;
    directResponseActive = false;
    directResponseDonePending = false;
    if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
    directTranscriptWaitTimer = null;
    directResponseId = '';
    directResponseTurnId = '';
    directResponseText = '';
    directResponsePlaybackStarted = false;
    pendingAssistantText = '';
    controlledReplyText = '';
    clearControlledPlaybackAck();
    controlledAwaitingPlaybackAck = false;
    controlledTurnFinalized = true;
    const canceledMetrics = finishVoiceMetrics();
    if (abandonedTurnId && sessionId) {
      callBackendJson('/session/metering/status', {
        session_id: backendSessionId || sessionId,
        turn_id: abandonedTurnId,
        status: 'canceled',
        voice_metrics: canceledMetrics,
      }, token).catch(() => {});
    }
    if (notifyClient && clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({
        type: 'response_stopped',
        reason,
        turnId: abandonedTurnId,
        playedMs: 0,
      }));
    }
  }

  function requestDirectE2ETextReply(userText, turnId) {
    const text = String(userText || '').trim();
    const id = String(turnId || '').trim();
    if (!directE2EEnabled || !text || !id || !doubaoWs || doubaoWs.readyState !== WebSocket.OPEN) {
      return false;
    }
    directE2ETurnId = id;
    directProviderSpeechStoppedAtMs = Date.now();
    doubaoWs.send(JSON.stringify({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text }],
      },
    }));
    doubaoWs.send(JSON.stringify({
      type: 'response.create',
      response: { modalities: ['text', 'audio'] },
    }));
    return true;
  }

  function announceDirectE2EReplyDone() {
    if (!directE2EEnabled || !directResponseDonePending || !directResponseId ||
        !directResponseTurnId) return;
    const userTurn = [...transcript].reverse().find((row) =>
      row.role === 'user' && row.turnId === directResponseTurnId
    );
    if (!userTurn) return;
    if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
    directTranscriptWaitTimer = null;
    directResponseDonePending = false;
    directResponseActive = false;
    activeTurnId = directResponseTurnId;
    lastCompletedUserText = String(userTurn.content || '').trim();
    directResponseText = String(directResponseText || pendingAssistantText || '').trim();
    controlledReplyText = directResponseText;
    pendingAssistantText = controlledReplyText;
    controlledTtsActive = false;
    controlledPhraseQueue = [];
    controlledCurrentPhrase = null;
    controlledTurnFinalized = false;
    controlledAwaitingPlaybackAck = true;
    clearControlledPlaybackAck();
    controlledAwaitingPlaybackAck = true;
    const ackTurnId = directResponseTurnId;
    controlledPlaybackAckTimer = setTimeout(() => {
      if (!controlledAwaitingPlaybackAck || controlledTurnFinalized || activeTurnId !== ackTurnId) return;
      console.log(`[SS ${ts()}] ${logTag} direct E2E playback ack timeout for ${ackTurnId}; discarding unheard turn`);
      interruptControlledResponse('playback_ack_timeout');
    }, 45000);
    clientWs.send(JSON.stringify({
      type: 'direct_reply_done',
      turnId: ackTurnId,
      text: controlledReplyText,
    }));
  }

  function failDirectE2EResponse(reason = 'direct_response_failed') {
    const failedTurnId = directResponseTurnId || directE2ETurnId || activeTurnId;
    abandonDirectE2EReply(reason, false);
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ type: 'error', error: reason, turnId: failedTurnId }));
    }
  }

  function directResponseTextFromDone(msg) {
    const output = Array.isArray(msg?.response?.output) ? msg.response.output : [];
    return output.flatMap((item) => Array.isArray(item?.content) ? item.content : [])
      .map((part) => String(part?.transcript || part?.text || '').trim())
      .filter(Boolean)
      .join(' ')
      .trim();
  }

  function clearControlledDeadline() {
    if (controlledDeadlineTimer) {
      clearTimeout(controlledDeadlineTimer);
      controlledDeadlineTimer = null;
    }
  }

  function clearControlledProgress() {
    // Progress now comes from the backend operation that actually started.
    // There is no proxy-side guessed timer to clear.
  }

  function clearControlledPlaybackAck() {
    if (controlledPlaybackAckTimer) {
      clearTimeout(controlledPlaybackAckTimer);
      controlledPlaybackAckTimer = null;
    }
    controlledAwaitingPlaybackAck = false;
  }

  function interruptControlledResponse(reason = 'user_stop', playedMs = 0) {
    directResponseActive = false;
    directResponseDonePending = false;
    if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
    directTranscriptWaitTimer = null;
    directResponseId = '';
    directResponseTurnId = '';
    directResponseText = '';
    directResponsePlaybackStarted = false;
    suppressCurrentResponse = true;
    controlPending = false;
    controlledTtsActive = false;
    controlledReplyReady = null;
    if (controlledStreamRequest) {
      controlledStreamRequest.abort(reason);
      controlledStreamRequest = null;
    }
    controlledPhraseQueue = [];
    controlledCurrentPhrase = null;
    controlledBackendDone = false;
    controlledReplyText = '';
    controlledResponseAnnounced = false;
    controlledTutorMoveSent = false;
    controlledTurnFinalized = false;
    clearControlledPlaybackAck();
    autoResponseEnded = false;
    awaitingAutoResponseDone = false;
    controlledRequestSeq += 1;
    clearControlledDeadline();
    clearControlledProgress();
    pendingAssistantText = '';
    if (controlledStartTimer) {
      clearTimeout(controlledStartTimer);
      controlledStartTimer = null;
    }
    if (turnTiming) turnTiming.stopped_by_user = true;
    const canceledMetrics = finishVoiceMetrics();
    pendingAssistantText = '';
    audioFlusher.discard(reason === 'barge_in' ? 'barge_in' : 'user_stop');
    if (doubaoWs && doubaoWs.readyState === WebSocket.OPEN) {
      doubaoWs.send(JSON.stringify({ type: 'response.cancel' }));
    }
    clientWs.send(JSON.stringify({
      type: 'response_stopped',
      reason,
      turnId: activeTurnId,
      playedMs: Math.max(0, Number(playedMs) || 0),
    }));
    if (activeTurnId && sessionId) {
      callBackendJson('/session/metering/status', {
        session_id: backendSessionId || sessionId,
        turn_id: activeTurnId,
        status: 'canceled',
        voice_metrics: canceledMetrics,
      }, token).catch(() => {});
    }
  }

  // 等待提示：由豆包用同一个声音念出来——复用正式回复的 TTS 通道与队列，
  // 不新开第二路声音、也不让浏览器本地合成（那会出现两个嗓子）。
  // 只在语音空闲且本轮自动回复确实已结束时才播，避免与上一轮尾巴叠音。
  function speakProgressHint(text) {
    const line = String(text || '').trim();
    if (!line) return false;
    if (!controlPending || !autoResponseEnded || controlledTurnFinalized) return false;
    if (controlledTtsActive) return false;
    if (controlledPhraseQueue.some((item) => item.hint)) return false;
    controlledPhraseQueue.push({
      id: `${activeTurnId}:hint`,
      index: -1,
      text: line,
      hint: true,
    });
    maybeStartControlledTts();
    return true;
  }

  function maybeStartControlledTts() {
    if (!controlPending || !autoResponseEnded || controlledTtsActive || controlledTurnFinalized) return;
    if (!controlledPhraseQueue.length) {
      if (controlledBackendDone && controlledReplyText.trim()) announceControlledStreamDone();
      return;
    }
    const phrase = controlledPhraseQueue.shift();
    const reply = String(phrase?.text || '').trim();
    if (!reply) {
      maybeStartControlledTts();
      return;
    }
    if (controlledStartTimer) {
      clearTimeout(controlledStartTimer);
      controlledStartTimer = null;
    }
    audioFlusher.discard("replace_auto_reply");
    controlledCurrentPhrase = phrase;
    controlledProviderResponseId = "";
    if (!phrase.hint) {
      // 提示句只是"占住等待时间"的话，不并入回复正文，避免污染存档与学习证据。
      controlledReplyText = controlledReplyText
        ? `${controlledReplyText} ${reply}` : reply;
      pendingAssistantText = controlledReplyText;
    }
    controlledTtsActive = true;
    suppressCurrentResponse = false;
    if (!controlledResponseAnnounced) {
      controlledResponseAnnounced = true;
      clientWs.send(JSON.stringify({ type: "response_ready", turnId: activeTurnId }));
    }
    clientWs.send(JSON.stringify({
      type: "response_phrase",
      turnId: activeTurnId,
      phraseId: phrase.id,
      ...(controlledPcmEnabled ? { audioFormat: "pcm_s16le", sampleRate: 24000 } : {}),
      index: phrase.index,
      // 显式标记"这句话只是占位提示，不是外教说的话"。前端据此把它只当临时
      // 字幕，不并入 AI 气泡、不进对话历史——否则提示语会跟着存档传给下一轮
      // 的模型（实测：存档里出现 "Let me think about that for a moment.The
      // bathroom is ..." 这种提示语与正文粘在一起的记录）。
      hint: !!phrase.hint,
      text: reply,
    }));
    if (!controlledTutorMoveSent && controlledReplyReady) {
      controlledTutorMoveSent = true;
      clientWs.send(JSON.stringify({
        type: "tutor_move",
        policy: controlledReplyReady.policy || {},
        learningMove: controlledReplyReady.learning_move || {},
        learningState: controlledReplyReady.learning_state || {},
        learnerState: controlledReplyReady.learner_state || {},
        conversationDirector: controlledReplyReady.conversation_director || {},
        generation: controlledReplyReady.generation || {},
      }));
    }
    if (doubaoWs && doubaoWs.readyState === 1) {
      if (turnTiming && !turnTiming.tts_commit_at_ms) {
        turnTiming.tts_commit_at_ms = Date.now();
      }
      doubaoWs.send(JSON.stringify({
        type: "speech_text_buffer.commit",
        text: reply,
      }));
    }
    // 学习证据只在 response.done 后写入一次。那里能同时带上完整回复和
    // 本轮真实语音指标，避免受控回复在 TTS 前后被重复评估。
    console.log(
      `[SS ${ts()}] ${logTag} Controlled phrase ${phrase.index} sent to TTS ` +
      `(${reply.length} chars, queued=${controlledPhraseQueue.length})`
    );
  }

  function queueControlledFallback(userText, path) {
    const reply = buildRecoverableControlledReply(userText, path);
    controlledPhraseQueue = [{ id: `${activeTurnId}:fallback`, index: 0, text: reply }];
    controlledReplyReady = {
      policy: { teaching_act: `${path}_recovery` },
      learning_move: { action: "acknowledge_and_continue", stage: "flow" },
      generation: { path: `proxy_${path}_recovery`, failure_streak: controlledFailureStreak },
    };
    controlledBackendDone = true;
    autoResponseEnded = true;
    awaitingAutoResponseDone = false;
    maybeStartControlledTts();
  }

  // 后端进程重启（部署/崩溃）会让旧的 backendSessionId 失效，表现为
  // respond/stream 返回 "session not found"。此时自动用同一配置重建后端
  // 会话并重试本轮，不让用户看到"回复没有生成完整"的兜底文案。
  let backendRecreatedForTurn = false;
  // 本轮已重试次数。生成失败（超时/后端抖动）时先原地重试，再考虑兜底话术——
  // 实测 LLM 端点抖动时用户会被要求"再说一次"，那是把系统的失败算到用户头上。
  let controlledRetryCount = 0;

  function recreateBackendSession() {
    return new Promise((resolve, reject) => {
      const payload = JSON.stringify({
        user_id: config.userId || "frontend-user",
        target_lang: config.targetLang || "en",
        cefr: config.level || "",
        scene_name: config.scene || "daily_talk",
        scene_role: (config.langName || "English") === "Japanese" ? "Japanese Tutor" : "English Tutor",
        scene_goal: config.sessionGoal
          ? `Practice ${config.langName || "English"}: focus on ${config.sessionGoal}`
          : "Practice " + (config.langName || "English") + " naturally",
        recent_weakness: config.weakness || "grammar",
        conv_id: dialogId,
        context_mode: cassetteMode,
        source_conv_id: config.sourceConvId || "",
        fresh_context: false,
        history: (currentDialogContext || []).map(backendHistoryMessage),
      });
      const req = request({
        hostname: BACKEND_HOST,
        port: BACKEND_PORT,
        path: "/session/start",
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
          ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        },
      }, (res) => {
        let body = "";
        res.on("data", (chunk) => { body += chunk; });
        res.on("end", () => {
          try {
            const data = JSON.parse(body);
            if (data.session_id) {
              backendSessionId = data.session_id;
              if (backendCacheKey) {
                backendSessionCache.delete(backendCacheKey);
                cacheBackendSession(backendCacheKey, {
                  sessionId: backendSessionId,
                  systemPrompt: data.system_prompt || "",
                });
              }
              console.log(`[SS ${ts()}] ${logTag} Backend session recreated after restart: ${backendSessionId}`);
              resolve(backendSessionId);
            } else {
              reject(new Error("session/start returned no session_id"));
            }
          } catch (e) {
            reject(e);
          }
        });
      });
      req.on("error", reject);
    });
  }

  function requestControlledReply(userText, waitForAutoEnd = true, turnId = '') {
    const requestSeq = ++controlledRequestSeq;
    clearControlledPlaybackAck();
    backendRecreatedForTurn = false;
    activeTurnId = turnId || activeTurnId || createId("turn_");
    // 本轮请求属于自己的 turn id：流式回调可能在若干秒后到达，届时
    // activeTurnId 可能已经翻到下一轮，落文本不能跟着 current 走。
    const requestTurnId = activeTurnId;
    controlPending = true;
    controlledTtsActive = false;
    controlledReplyReady = null;
    if (controlledStreamRequest) controlledStreamRequest.abort("replace_turn");
    controlledStreamRequest = null;
    controlledPhraseQueue = [];
    controlledCurrentPhrase = null;
    controlledBackendDone = false;
    controlledReplyText = '';
    controlledResponseAnnounced = false;
    controlledTutorMoveSent = false;
    controlledTurnFinalized = false;
    autoResponseEnded = !waitForAutoEnd;
    awaitingAutoResponseDone = waitForAutoEnd;
    suppressCurrentResponse = true;
    pendingAssistantText = '';
    audioFlusher.discard("await_tutor_policy");
    clientWs.send(JSON.stringify({
      type: "response_pending",
      phase: "heard",
      cancelable: true,
      turnId: activeTurnId,
    }));

    clearControlledDeadline();
    clearControlledProgress();
    if (turnTiming) turnTiming.backend_request_at_ms = Date.now();
    const handleControlledDeadline = () => {
      if (requestSeq !== controlledRequestSeq || !controlPending) return;
      const hasBackendProgress = Boolean(
        turnTiming?.backend_meta_at_ms || turnTiming?.first_progress_at_ms
      );
      if (hasBackendProgress && !controlledDeadlineExtended) {
        controlledDeadlineExtended = true;
        const extensionMs = Math.min(6000, Math.max(0, 20000 - CONTROLLED_REPLY_DEADLINE_MS));
        if (extensionMs > 0) {
          console.log(
            "[SS " + ts() + "] " + logTag + " controlled reply is still progressing; " +
            "extending this turn deadline by " + extensionMs + "ms once"
          );
          controlledDeadlineTimer = setTimeout(handleControlledDeadline, extensionMs);
          return;
        }
      }
      const deadlineStage = turnTiming?.backend_meta_at_ms
        ? 'after_backend_meta'
        : turnTiming?.first_progress_at_ms
          ? 'during_prepare'
          : 'before_first_backend_event';
      const backendWaitMs = turnTiming?.backend_request_at_ms
        ? Math.max(0, Date.now() - turnTiming.backend_request_at_ms) : null;
      console.log(
        `[SS ${ts()}] ${logTag} controlled respond deadline reached; ` +
        `stage=${deadlineStage} backendWaitMs=${backendWaitMs ?? 'n/a'}; reporting failure`
      );
      controlledFailureStreak += 1;
      if (controlledStreamRequest) controlledStreamRequest.abort("controlled_timeout");
      controlledStreamRequest = null;
      // 2026-10-21（用户定调）：超时也是失败，不再编一句兜底话术。
      clientWs.send(JSON.stringify({
        type: 'error',
        error: 'reply generation timed out',
        turnId: activeTurnId,
      }));
      controlPending = false;
      controlledBackendDone = true;
    };
    controlledDeadlineTimer = setTimeout(
      handleControlledDeadline,
      CONTROLLED_REPLY_DEADLINE_MS,
    );

    if (waitForAutoEnd && doubaoWs?.readyState === WebSocket.OPEN) {
      // The provider starts a free-form reply after ASR. Interrupt it; the
      // backend Tutor Policy owns the response that will actually be heard.
      doubaoWs.send(JSON.stringify({ type: "response.cancel" }));
      controlledStartTimer = setTimeout(() => {
        awaitingAutoResponseDone = false;
        autoResponseEnded = true;
        maybeStartControlledTts();
      }, 800);
    }

    // §8.65：标记取完即清——只影响"回来"这一轮，不粘在后面每一轮。
    const returnedFromIdleThisTurn = pendingReturnedFromIdle === true;
    pendingReturnedFromIdle = false;
    controlledStreamRequest = callBackendNdjson('/session/turn/respond/stream', {
      session_id: backendSessionId || sessionId,
      turn_id: activeTurnId,
      user_transcript: userText,
      // 原文另送一路：规范化英文洗掉了"他用了中文/说到一半卡住"这些痕迹，
      // 而它们是语言与流利度信号的唯一来源。多一个字段，不增加任何调用。
      user_raw_transcript: String(lastFinalAsrText || userText || ''),
      // plans/0916-26 修复3：本轮 ASR 合并段数。段数 >1 而字数少 = 这一句被
      // 切碎了，后端据此把这一轮判成低置信、合同改成"先复述确认"再回应。
      asr_segments_merged: Number(turnTiming?.asr_segments_merged) || 0,
      // §8.65：他上一轮是"走开又回来"，后端只在这一轮开头接半句。
      returned_from_idle: returnedFromIdleThisTurn,
    }, (event) => {
      if (requestSeq !== controlledRequestSeq || !controlPending) return;
      if (event?.type === 'meta') {
        if (turnTiming && !turnTiming.backend_meta_at_ms) {
          turnTiming.backend_meta_at_ms = Date.now();
          turnTiming.route_tier = String(event?.turn_execution?.route_tier || 'unknown');
          turnTiming.prepare_latency_ms = Number(event?.timing?.prepare_latency_ms) || null;
        }
        controlledReplyReady = {
          ...(controlledReplyReady || {}),
          policy: event.policy || {},
          learning_move: event.learning_move || {},
          learning_state: event.learning_state || {},
          learner_state: event.learner_state || {},
          conversation_director: event.conversation_director || {},
        };
        return;
      }
      if (event?.type === 'progress') {
        if (turnTiming && !turnTiming.first_progress_at_ms) {
          turnTiming.first_progress_at_ms = Date.now();
        }
        const hintText = String(event.spoken_text || '').trim();
        const hintSpoken = speakProgressHint(hintText);
        clientWs.send(JSON.stringify({
          type: 'response_progress',
          phase: String(event.phase || 'working'),
          text: String(event.text || ''),
          // 提示由豆包用同一个声音念（不是浏览器合成、也不是第二路声音）：
          // 走 controlledPhraseQueue 这条既有 TTS 通道，和正式回复同队列同嗓子。
          spoken: hintSpoken,
          requestedSpoken: Boolean(event.spoken),
          turnId: activeTurnId,
        }));
        return;
      }
      if (event?.type === 'user_intent_normalized') {
        // 生成调用顺带给出的"学习者意图规范英文"：与模型读到的内容同源，
        // 直接用它更新前端气泡、transcript 与后续存档。
        // 注意：同样不回写 lastCompletedUserText——纠错记忆必须看到
        // 学习者的原始错误，规范英文只负责展示。
        const normalized = String(event.text || '').trim();
        if (!normalized) return;
        turnNormalizedUserText = normalized;
        // plans/0915-23：后端判定"规范英文越界/无法验真"时会带着 fallback 退回原话。
        // 这种文本同样要认成权威值（否则又回退精修端点重编一遍），但要记下来，
        // 后面要靠这行日志统计"到底有多少轮被改写了"。
        const rejected = !!event.fallback;
        console.log(
          `[SS ${ts()}] ${logTag} normalized turn=${requestTurnId}`
          + (rejected ? ` REJECTED(${String(event.reason || '')}) raw` : '')
          + ` -> ${transcriptForLog(normalized, 110)}`
        );
        setUserTurnText(requestTurnId, normalized);
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify({
            type: 'asr_refined',
            turnId: requestTurnId || '',
            text: normalized,
            raw: String(event.raw || ''),
            source: 'generation',
            fallback: rejected,
          }));
        }
        return;
      }
      if (event?.type === 'phrase') {
        const phraseText = String(event.text || '').trim();
        if (!phraseText) return;
        if (turnTiming && !turnTiming.first_text_at_ms) turnTiming.first_text_at_ms = Date.now();
        clearControlledDeadline();
        clearControlledProgress();
        controlledFailureStreak = 0;
        controlledPhraseQueue.push({
          id: `${activeTurnId}:${Number(event.index) || 0}`,
          index: Number(event.index) || 0,
          text: phraseText,
        });
        maybeStartControlledTts();
        return;
      }
      if (event?.type === 'done') {
        controlledBackendDone = true;
        controlledReplyReady = { ...(controlledReplyReady || {}), ...event };
        lastSuccessfulControlledReply = controlledReplyReady;
        clearControlledDeadline();
        clearControlledProgress();
        const generation = event.generation || {};
        const backendMs = turnTiming?.asr_final_at_ms
          ? Math.max(0, Date.now() - turnTiming.asr_final_at_ms) : null;
        // Print '-' for any latency the backend did not report instead of
        // throwing: only the phrase-stream path fills all of these in, and a
        // missing field must never break a live turn.
        const msField = (value) => (
          typeof value === 'number' && Number.isFinite(value) ? value : '-'
        );
        console.log(
          `[SS ${ts()}] ${logTag} Phrase stream done: ` +
          `move=${event?.learning_move?.action || 'unknown'}, ` +
          `director=${event?.conversation_director?._meta?.source || 'unknown'}, ` +
          `path=${generation.path || 'unknown'}, phrases=${generation.phrase_count ?? 'n/a'}, ` +
          `prepareMs=${msField(generation.prepare_latency_ms)}, ` +
          `directorMs=${msField(generation.director_latency_ms)}, ` +
          `genMs=${msField(generation.response_generation_latency_ms)}, ` +
          `backendMs=${backendMs ?? generation.backend_total_latency_ms ?? 'n/a'}`
        );
        maybeStartControlledTts();
        return;
      }
      if (event?.type === 'error') {
        throw new Error(String(event.error || 'backend stream error'));
      }
    }, token, CONTROLLED_REPLY_DEADLINE_MS + 1000);
    controlledStreamRequest.promise.then(() => {
      if (requestSeq !== controlledRequestSeq || !controlPending) return;
      controlledStreamRequest = null;
      // 模型没按格式给意图（例如导演直出草稿）时，回退到独立精修端点补齐。
      if (!turnNormalizedUserText) refineUserTranscript(activeTurnId, lastFinalAsrText);
      if (!controlledBackendDone) throw new Error('backend stream ended without done event');
      const cachedBackend = getCachedBackendSession(backendCacheKey);
      if (cachedBackend && backendSessionId) {
        cacheBackendSession(backendCacheKey, {
          sessionId: backendSessionId,
          systemPrompt: cachedBackend.systemPrompt,
        });
      }
    }).catch((error) => {
      if (requestSeq !== controlledRequestSeq || !controlPending) return;
      controlledStreamRequest = null;
      clearControlledProgress();
      const errorText = String((error && error.message) || error || "");
      console.log(`[SS ${ts()}] ${logTag} controlled stream fallback: ${errorText}`);
      // 生成失败（超时/端点抖动）时先原地重试一次再兜底。
      // 用户已经说过了，问题在我们这边——不该让他重说一遍。
      if (controlledRetryCount < 1 && !controlledReplyText
          && !controlledTtsActive && !controlledPhraseQueue.length) {
        controlledRetryCount += 1;
        console.log(`[SS ${ts()}] ${logTag} retrying controlled reply once (${errorText.slice(0, 60)})`);
        setTimeout(() => {
          if (!controlPending || controlledTurnFinalized) return;
          requestControlledReply(userText, false, activeTurnId);
        }, 400);
        return;
      }
      if (/session not found/i.test(errorText) && !backendRecreatedForTurn) {
        // 后端进程重启导致旧会话失效：自动重建后端会话并重试本轮，不打扰用户。
        backendRecreatedForTurn = true;
        recreateBackendSession().then(() => {
          requestControlledReply(userText, false, activeTurnId);
        }).catch((e2) => {
          console.log(`[SS ${ts()}] ${logTag} backend session recreate failed: ${e2.message}`);
          controlledFailureStreak += 1;
          if (!controlledReplyText && !controlledTtsActive && !controlledPhraseQueue.length) {
            queueControlledFallback(userText, "backend_error");
          } else {
            controlledBackendDone = true;
            maybeStartControlledTts();
          }
        });
        return;
      }
      callBackendJson('/session/metering/status', {
        session_id: backendSessionId || sessionId,
        turn_id: activeTurnId,
        status: 'failed',
      }, token).catch(() => {});
      // 失败轮同样保证前端气泡不空白：没有权威文本时才回退精修。
      if (!turnNormalizedUserText) refineUserTranscript(activeTurnId, lastFinalAsrText);
      controlledFailureStreak += 1;
      // 2026-10-21（用户定调）：不再用 queueControlledFallback 编一句"稍等/我没接住"。
      // 生成失败就如实告诉前端这一轮失败了，不能拿系统故障冒充一句回复。
      if (!controlledReplyText && !controlledTtsActive && !controlledPhraseQueue.length) {
        clientWs.send(JSON.stringify({
          type: 'error',
          error: 'reply generation failed',
          turnId: activeTurnId,
        }));
        controlPending = false;
        controlledBackendDone = true;
      } else {
        controlledBackendDone = true;
        maybeStartControlledTts();
      }
    });
  }

  // ── 人走了（plans/1017-02 §8.70 重写）──────────────────────────────
  // 规则（用户定的，别再复杂化）：
  //   ① 这一盘**还没有真正的对话** → 沉默到点就**静默断开**，不说话；
  //   ② 有对话 → 沉默 15s 问一句 → 再 25s 道别 → 6s 后断开；
  //   ③ 用户重新点按钮进来 = 他回来了：之前的状态一律作废，不继承。
  // 台词由模型写（§8.64），用她自己的声线合成（§8.66）。
  const IDLE_CHECKIN_ENABLED = /^(1|true|on|yes)$/i.test(
    String(process.env.IDLE_CHECKIN_ENABLED || ''));
  const IDLE_CHECKIN_MS = Math.max(3000, Number(process.env.IDLE_CHECKIN_MS || 15000));
  const IDLE_FAREWELL_MS = Math.max(3000, Number(process.env.IDLE_FAREWELL_MS || 25000));
  const IDLE_QUIET_CLOSE_MS = Math.max(5000, Number(process.env.IDLE_QUIET_CLOSE_MS || 45000));
  const IDLE_CHECKIN_LINES = [
    "You still there?",
    "Hey — did I lose you?",
    "Did you get pulled away? No rush.",
  ];
  const IDLE_FAREWELL_LINES = [
    "Alright, I'll get out of your way. Come find me later.",
    "Sounds like you got busy — I'll be around.",
    "I'll leave it there. Talk when you're free.",
  ];
  let idleTimer = null;
  let idleStage = 0;        // 0=没进入"人走了"流程 1=已问过 2=已道别/已关
  let pendingReturnedFromIdle = false;  // §8.65：他刚回来，只用一次
  let idleLineSentAt = 0;               // §8.68：防扬声器回声
  let idleLineIdx = 0;
  // 本连接的事实：没聊过、也没听到他说话 ⇒ 沉默时不该开口，直接静默断开。
  let sessionTurnCount = 0;
  let heardLearnerSpeech = false;
  const idleRecentLines = [];

  // 上一轮的原话从 transcript 取（`completedAssistantText` 是 finalizeControlledTutorTurn
  // 内部的局部变量，在这里不可见——2026-10-18 踩过）。
  function lastTranscriptText(role) {
    for (let i = transcript.length - 1; i >= 0; i -= 1) {
      if (transcript[i] && transcript[i].role === role) {
        return String(transcript[i].content || '');
      }
    }
    return '';
  }

  // 让模型写这一句；失败（超时/空/网络）一律回落固定台词——沉默时不该卡住。
  async function composeIdleLine(kind, fallback) {
    try {
      const result = await callBackendJson('/session/idle-line', {
        kind,
        last_user_text: lastTranscriptText('user'),
        last_assistant_text: lastTranscriptText('assistant'),
        recent_lines: idleRecentLines.slice(-4),
      }, token, kind === 'farewell' ? 12000 : 20000);
      const text = String((result && result.text) || '').trim();
      if (text) {
        idleRecentLines.push(text);
        let tts = null;
        try {
          tts = await vocabTtsSynthesize(text);
          if (tts && tts.audio) {
            const trimmed = trimTtsSilence(Buffer.from(tts.audio, 'base64'));
            tts = { audio: trimmed.toString('base64'), mime: tts.mime };
          }
        } catch (err) {
          console.log(`[SS ${ts()}] ${logTag} idle tts failed: ${err && err.message}`);
        }
        return { text, audio: tts ? tts.audio : '', mime: tts ? tts.mime : '' };
      }
      console.log(`[SS ${ts()}] ${logTag} idle-line empty text (token=${(token || '').length})`);
    } catch (err) {
      console.log(`[SS ${ts()}] ${logTag} idle-line failed: ${err && err.message} (token=${(token || '').length})`);
    }
    return { text: fallback, audio: '', mime: '' };
  }

  function clearIdleWatch() {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  }
  function closeIdleSocket(reason) {
    try { clientWs.close(1000, reason); } catch (_) {}
  }

  // 一次"活动结束"（他说完/她说完）后的分岔。
  function onActivityEnded() {
    if (!IDLE_CHECKIN_ENABLED || idleStage >= 2) return;
    clearIdleWatch();
    if (idleStage === 1) {
      idleTimer = setTimeout(() => { idleTimer = null; sendFarewell(); }, IDLE_FAREWELL_MS);
      return;
    }
    if (!sessionTurnCount && !heardLearnerSpeech) {
      idleTimer = setTimeout(() => { idleTimer = null; quietClose(); }, IDLE_QUIET_CLOSE_MS);
      return;
    }
    idleTimer = setTimeout(() => { idleTimer = null; sendCheckin(); }, IDLE_CHECKIN_MS);
  }

  async function sendCheckin() {
    if (!IDLE_CHECKIN_ENABLED || idleStage !== 0) return;
    idleStage = 1;
    const fallback = IDLE_CHECKIN_LINES[idleLineIdx++ % IDLE_CHECKIN_LINES.length];
    const line = await composeIdleLine('checkin', fallback);
    // compose 要几秒：这期间他可能回来了，或者连接已经没了。
    if (idleStage !== 1 || clientWs.readyState !== WebSocket.OPEN) return;
    idleLineSentAt = Date.now();
    clientWs.send(JSON.stringify({
      type: 'idle_checkin', text: line.text, audio: line.audio, mime: line.mime,
    }));
    console.log(`[SS ${ts()}] ${logTag} idle check-in sent`);
    IDLE_ASK_BY_USER.set(currentUserId, { stage: 1, at: Date.now() });
    idleTimer = setTimeout(() => { idleTimer = null; sendFarewell(); }, IDLE_FAREWELL_MS);
  }

  async function sendFarewell() {
    if (!IDLE_CHECKIN_ENABLED || idleStage !== 1) return;
    idleStage = 2;
    const fallback = IDLE_FAREWELL_LINES[idleLineIdx++ % IDLE_FAREWELL_LINES.length];
    const line = await composeIdleLine('farewell', fallback);
    if (clientWs.readyState !== WebSocket.OPEN) return;
    idleLineSentAt = Date.now();
    clientWs.send(JSON.stringify({
      type: 'idle_farewell', text: line.text, audio: line.audio, mime: line.mime,
    }));
    console.log(`[SS ${ts()}] ${logTag} idle farewell sent; closing in 6s`);
    IDLE_ASK_BY_USER.set(currentUserId, { stage: 2, at: Date.now() });
    setTimeout(() => closeIdleSocket('idle_farewell'), 6000);
  }

  function quietClose() {
    if (idleStage !== 0) return;
    idleStage = 2;
    console.log(`[SS ${ts()}] ${logTag} idle: no conversation yet, closing quietly`);
    closeIdleSocket('idle_quiet');
  }

  // 他开口 = 他回来了：作废所有"走了"的状态（包括"刚问过"）。
  function noteUserSpeaking() {
    heardLearnerSpeech = true;
    IDLE_ASK_BY_USER.delete(currentUserId);  // 他回来了，账清掉
    if (Date.now() - idleLineSentAt <= 1500) return;  // 扬声器回声
    if (idleStage === 1) pendingReturnedFromIdle = true;
    idleStage = 0;
    clearIdleWatch();
  }

  function announceControlledStreamDone() {
    if (controlledTurnFinalized || controlledAwaitingPlaybackAck ||
        controlledTtsActive || controlledPhraseQueue.length || !controlledBackendDone) return;
    controlledAwaitingPlaybackAck = true;
    clientWs.send(JSON.stringify({
      type: 'response_stream_done',
      turnId: activeTurnId,
      phraseCount: Number(controlledReplyReady?.generation?.phrase_count) || undefined,
    }));
    // Fail closed: if the browser never confirms playback, never write an
    // unheard assistant reply into dialogue memory or learning evidence.
    controlledPlaybackAckTimer = setTimeout(() => {
      if (!controlledAwaitingPlaybackAck || controlledTurnFinalized) return;
      console.log(`[SS ${ts()}] ${logTag} playback ack timeout for ${activeTurnId}; discarding unheard turn`);
      interruptControlledResponse('playback_ack_timeout');
    }, 45000);
  }

  function finalizeControlledTutorTurn(completedTurnId = '') {
    if (controlledTurnFinalized || !controlledAwaitingPlaybackAck ||
        completedTurnId !== activeTurnId || controlledTtsActive || controlledPhraseQueue.length) return;
    clearControlledPlaybackAck();
    controlledTurnFinalized = true;
    controlPending = false;
    clearControlledDeadline();
    clearControlledProgress();
    completedTurnId = activeTurnId;
    pendingAssistantText = controlledReplyText.trim() || pendingAssistantText.trim();
    const voiceMetrics = finishVoiceMetrics();
    const completedAssistantText = pendingAssistantText.trim();
    if (completedAssistantText) {
      const currentUserRow = [...transcript].reverse().find((row) =>
        row.role === 'user' && row.turnId === completedTurnId
      );
      const historyUserText = String(
        currentUserRow?.content || lastCompletedUserText || ''
      ).trim();
      if (historyUserText) {
        // Keep reconnect/recreated-backend context current. Previously this
        // array was frozen at socket startup, so a later "session not found"
        // rebuild silently lost every turn spoken in the active conversation.
        currentDialogContext = cleanDialogContext([
          ...(currentDialogContext || []),
          { role: 'user', content: historyUserText },
          { role: 'assistant', content: completedAssistantText, turn_id: completedTurnId },
        ]);
      }
      transcript.push({ role: 'assistant', content: completedAssistantText, ts: Date.now() });
      if (lastCompletedUserText && sessionId) {
        const completedUserText = lastCompletedUserText;
        callBackendJson('/session/turn', {
          session_id: backendSessionId || sessionId,
          turn_id: completedTurnId,
          user_transcript: completedUserText,
          assistant_response: completedAssistantText,
          voice_metrics: voiceMetrics,
        }, token).then((turnResult) => {
          clientWs.send(JSON.stringify({ type: 'turn_evaluated', result: turnResult }));
        }).catch((error) => {
          console.log(`[SS ${ts()}] ${logTag} /session/turn failed: ${error.message}`);
        });

        const inputDurationMs = Number(voiceMetrics?.asr_input_duration_ms);
        if (Number.isFinite(inputDurationMs) && inputDurationMs > 0) {
          callBackendJson('/session/speech-assessment', {
            user_id: currentUserId,
            session_id: backendSessionId || sessionId,
            transcript: completedUserText,
            duration_seconds: inputDurationMs / 1000,
            assessment_context: { target_lang: currentTargetLang || 'en' },
          }, token).then((assessment) => {
            clientWs.send(JSON.stringify({ type: 'speech_assessed', assessment }));
          }).catch((error) => {
            console.log(`[SS ${ts()}] ${logTag} /session/speech-assessment failed: ${error.message}`);
          });
        }
      }
    }
    audioFlusher.flush("controlled_turn_done");
    clientWs.send(JSON.stringify({
      type: "response_done",
      turnId: completedTurnId,
      phraseCount: Number(controlledReplyReady?.generation?.phrase_count) || undefined,
    }));
    // §8.70：这一轮说完了 = 一次"活动结束"，由 onActivityEnded 决定下一步。
    sessionTurnCount += 1;
    idleStage = 0;
    onActivityEnded();
    pendingAssistantText = '';
    controlledReplyText = '';
    controlledCurrentPhrase = null;
    controlledBackendDone = false;
    lastCompletedUserText = '';
    turnTiming = null;
    clearTurnEnergy();
    activeTurnId = '';
    if (directE2ETurnId === completedTurnId) directE2ETurnId = '';
    if (directResponseTurnId === completedTurnId) {
      directResponseId = '';
      directResponseTurnId = '';
      directResponseText = '';
      directResponseDonePending = false;
      if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
      directTranscriptWaitTimer = null;
      directResponsePlaybackStarted = false;
    }
    directResponseActive = false;
    suppressCurrentResponse = true;
  }

  function buildRecoverableControlledReply(userText, reason = "backend_error") {
    const compact = String(userText || "").replace(/\s+/g, " ").trim();
    if (sessionIsMaterialTape) {
      if (/\b(?:elaborate|expand on|go into more detail|tell me more|explain more|say more about)\b|具体展开|展开讲|详细讲|多讲一点|再具体说|再解释一下/i.test(compact)) {
        return "I heard you asking for more detail on that point, but the full explanation hasn’t come through yet.";
      }
      if (/\b(can't|cannot|couldn't|unable to) afford\b|\btoo expensive\b|花不起|买不起|负担不起|消费水平/i.test(compact)) {
        return "I get you — both prices feel out of reach. Which one hits harder, the room or the toys?";
      }
      if (/\b(video|film|movie|episode|material|source)\b|视频|素材/i.test(compact) &&
          /\b(other|another|else|point|mention|interesting|discuss|what)\b|还有|其他|提到|内容|有趣/i.test(compact)) {
        return "I can’t confirm another point from the video right now, and I don’t want to guess.";
      }
      return "I’m with you. My reply is taking longer than it should; let me pick up from what you just said.";
    }
    if (sessionCassetteMode !== 'expression' && !sessionIsMaterialTape) {
      // Keep the brief, human-sounding stumble; do not narrate proxy/backend
      // failures as if Alice were a machine. The retry/progress guards below
      // reduce how often any recovery line reaches the user.
      //
      // plans/1011-闲聊盘：措辞三要求（10-11 真机验收反馈）——① 不道歉（低姿态
      // 会把自己放到用户之下）；② 不反问、不要求用户重说（那是把系统失败算
      // 到用户头上）；③ 只做一次短暂的"稍等"，然后自然地继续。
      const chatFallbacks = [
        "Hang on — I want to get this one right.",
        "One moment; let me pick that thread back up.",
        "Give me a beat — I'm still with you.",
        "Hold on, I'm coming back to that.",
      ];
      const reply = chatFallbacks[chatFallbackIndex % chatFallbacks.length];
      chatFallbackIndex += 1;
      return reply;
    }
    if (/\b(?:thank\s+you|thanks)\b|谢谢/i.test(compact)) {
      return "You’re welcome. 不客气。";
    }
    if (/^(?:ok(?:ay)?[,. ]*)?(?:继续|然后呢|and\s+then|what(?:'s|\s+is)\s+next)[?？.!。 ]*$/i.test(compact)) {
      return "Let’s continue from here. 我们从刚才这一步继续。";
    }
    // Never expose director/provider internals as if they were tutor speech.
    // Preserve the backend session and let the conversation continue instead of
    // inventing an answer or replaying the previous scene instruction.
    //
    // 措辞两条要求：① 英文（产品是英语沉浸，中文兜底会破坏它，也实测被用户
    // 直接指出）；② 不把系统的失败说成用户的错——不要说"请再说一次"，
    // 而是自然地接住继续聊，让用户不必重说。
    if (controlledFailureStreak <= 1) {
      return "Hang on — I want to get that right.";
    }
    return "One moment — I'm still with you.";
  }

  function beginTurnTiming() {
    turnTiming = {
      asr_started_at_ms: null,
      asr_final_at_ms: null,
      first_text_at_ms: null,
      first_audio_at_ms: null,
      last_speech_activity_at_ms: null,
      provider_speech_stopped_at_ms: null,
      provider_response_created_at_ms: null,
      response_end_at_ms: null,
      backend_request_at_ms: null,
      backend_meta_at_ms: null,
      first_progress_at_ms: null,
      tts_commit_at_ms: null,
      client_playback_at_ms: null,
      route_tier: 'unknown',
      prepare_latency_ms: null,
      first_user_speech_latency_ms: null,
      stopped_by_user: false,
      asr_input_duration_ms: null,
      head_trim_ms: null,
      tail_trim_ms: null,
      end_mid_speech: null,
      end_by_silence: null,
      asr_segments_merged: null,
    };
  }

  function finishVoiceMetrics() {
    if (!turnTiming) return null;
    turnTiming.response_end_at_ms = Date.now();
    const asrFinal = turnTiming.asr_final_at_ms;
    const metrics = {
      asr_duration_ms: turnTiming.asr_started_at_ms && asrFinal
        ? Math.max(0, asrFinal - turnTiming.asr_started_at_ms) : null,
      first_text_latency_ms: asrFinal && turnTiming.first_text_at_ms
        ? Math.max(0, turnTiming.first_text_at_ms - asrFinal) : null,
      first_audio_latency_ms: asrFinal && turnTiming.first_audio_at_ms
        ? Math.max(0, turnTiming.first_audio_at_ms - asrFinal) : null,
      speech_to_first_audio_ms: turnTiming.last_speech_activity_at_ms && turnTiming.first_audio_at_ms
        ? Math.max(0, turnTiming.first_audio_at_ms - turnTiming.last_speech_activity_at_ms) : null,
      speech_to_playback_ms: turnTiming.last_speech_activity_at_ms && turnTiming.client_playback_at_ms
        ? Math.max(0, turnTiming.client_playback_at_ms - turnTiming.last_speech_activity_at_ms) : null,
      provider_vad_wait_ms: turnTiming.last_speech_activity_at_ms && turnTiming.provider_speech_stopped_at_ms
        ? Math.max(0, turnTiming.provider_speech_stopped_at_ms - turnTiming.last_speech_activity_at_ms) : null,
      vad_to_first_audio_ms: turnTiming.provider_speech_stopped_at_ms && turnTiming.first_audio_at_ms
        ? Math.max(0, turnTiming.first_audio_at_ms - turnTiming.provider_speech_stopped_at_ms) : null,
      vad_to_response_created_ms: turnTiming.provider_speech_stopped_at_ms && turnTiming.provider_response_created_at_ms
        ? Math.max(0, turnTiming.provider_response_created_at_ms - turnTiming.provider_speech_stopped_at_ms) : null,
      backend_meta_latency_ms: asrFinal && turnTiming.backend_meta_at_ms
        ? Math.max(0, turnTiming.backend_meta_at_ms - asrFinal) : null,
      first_progress_latency_ms: asrFinal && turnTiming.first_progress_at_ms
        ? Math.max(0, turnTiming.first_progress_at_ms - asrFinal) : null,
      tts_commit_latency_ms: asrFinal && turnTiming.tts_commit_at_ms
        ? Math.max(0, turnTiming.tts_commit_at_ms - asrFinal) : null,
      client_playback_latency_ms: asrFinal && turnTiming.client_playback_at_ms
        ? Math.max(0, turnTiming.client_playback_at_ms - asrFinal) : null,
      response_duration_ms: asrFinal
        ? Math.max(0, turnTiming.response_end_at_ms - asrFinal) : null,
      first_user_speech_latency_ms: turnTiming.first_user_speech_latency_ms,
      session_entry_mode: sessionEntryMode,
      provider_session_age_ms: sessionReadyAtMs
        ? Math.max(0, turnTiming.response_end_at_ms - sessionReadyAtMs) : null,
      route_tier: turnTiming.route_tier,
      voice_reply_mode: directE2EEnabled ? 'direct_e2e' : 'controlled',
      prepare_latency_ms: turnTiming.prepare_latency_ms,
      stopped_by_user: turnTiming.stopped_by_user,
      // PCM s16le mono @16kHz = 32 bytes/ms. This is an observed
      // duration, not a provider invoice quantity.
      asr_input_duration_ms: turnTiming.asr_input_duration_ms,
      asr_usage_status: turnTiming.asr_input_duration_ms != null ? 'observed' : 'unavailable',
      // plans/0916-22 修复0：录音完整性台账（观测值，随 voice_metrics 下发）
      head_trim_ms: turnTiming.head_trim_ms,
      tail_trim_ms: turnTiming.tail_trim_ms,
      end_mid_speech: turnTiming.end_mid_speech,
      end_by_silence: turnTiming.end_by_silence,
      asr_segments_merged: turnTiming.asr_segments_merged,
      // Audio duration is not decoded here; retain only the observable text unit.
      tts_output_duration_ms: null,
      tts_character_count: pendingAssistantText.trim() ? pendingAssistantText.trim().length : null,
      tts_usage_status: pendingAssistantText.trim() ? 'observed' : 'unavailable',
    };
    console.log(
      `[SS ${ts()}] ${logTag} voice timing mode=${metrics.voice_reply_mode} ` +
      `speech→firstAudio=${metrics.speech_to_first_audio_ms ?? 'n/a'}ms ` +
      `speech→playback=${metrics.speech_to_playback_ms ?? 'n/a'}ms ` +
      `VAD→firstAudio=${metrics.vad_to_first_audio_ms ?? 'n/a'}ms`
    );
    clientWs.send(JSON.stringify({ type: 'voice_metrics', metrics }));
    return metrics;
  }

  doubaoWs = openVoiceSocket();

  doubaoWs.on("open", () => {
    console.log(`[SS ${ts()}] ${logTag} Connected to Doubao Seeduplex`);
    isDoubaoReady = true;
    if (pendingConfig) {
      _startDoubaoSession(pendingConfig, false);
      pendingConfig = null;
    }
  });

  doubaoWs.on("message", (raw) => {
    try {
      const data = typeof raw === "string" ? raw : raw.toString("utf-8");
      const msg = JSON.parse(data);
      console.log(`[SS ${ts()}] ${logTag} Doubao event: ${msg.type}`);

      if (msg.type === "session.created") {
        sessionCreated = true;
        sessionId = msg.session?.id || sessionId;
        sessionReadyAtMs = Date.now();
        console.log(`[SS ${ts()}] ${logTag} Doubao session created: ${sessionId}`);
        clientWs.send(JSON.stringify({ type: "connected", sessionId }));
      } else if (msg.type === "session.closed") {
        sessionCreated = false;
        isDoubaoReady = false;
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify({
            type: "provider_session_closed",
            reason: "idle_or_provider_close",
          }));
        }
      } else if (msg.type === "input_audio_buffer.speech_stopped") {
        if (directE2EEnabled) directProviderSpeechStoppedAtMs = Date.now();
        if (turnTiming && !turnTiming.provider_speech_stopped_at_ms) {
          turnTiming.provider_speech_stopped_at_ms = Date.now();
        }
      } else if (msg.type === "conversation.item.input_audio_transcription.started" || msg.type === "input_audio_buffer.speech_started") {
        // Doubao can emit a new `started` immediately after completing a real
        // segment even when no following words ever arrive. Treating that
        // empty start as learner speech canceled the bounded 2.2s turn timer.
        // Only a real transcription delta below proves continuation.
        pendingUserAsr = "";
        providerAsrPcmBoundary.noteProviderStarted();
        if (directE2EEnabled && msg.type === "input_audio_buffer.speech_started") {
          directProviderSpeechStoppedAtMs = 0;
          const previousReplyInFlight = directResponseActive || directResponseDonePending || controlledAwaitingPlaybackAck;
          // Reserve the new input's identity as soon as its audio begins. The
          // provider can create the matching response before its ASR delta, so
          // waiting until the transcript arrives can bind that response to the
          // previous turn. Keep the old response's separate turn id untouched.
          directE2ETurnId = createId("turn_");
          if (!previousReplyInFlight) {
            if (!turnTiming || turnTiming.response_end_at_ms) beginTurnTiming();
            if (!turnTiming.asr_started_at_ms) turnTiming.asr_started_at_ms = Date.now();
          }
        }
      } else if (msg.type === "input_audio_buffer.committed") {
        // user audio committed, wait for transcription.completed
        // §8.67：他刚说完（哪怕这一轮最后被判成空），就要开始"人还在不在"的计时。
        // 之前只挂在 response_done 上——空 ASR / 没出稿时永远不计时，只能等 provider 超时掐线。
        // §8.70：他说完了——没出稿也算一次"活动结束"（空 ASR 的兜底在这里）。
        if (idleStage === 0) {
          onActivityEnded();
          console.log(`[SS ${ts()}] ${logTag} idle watch armed after commit`);
        }
      } else if (msg.type === "conversation.item.input_audio_transcription.delta") {
        // ASR incremental text
        // 关键修复：豆包下发的 transcript/delta 是"完整累计文本"（每次包含前面所有字），
        // 绝不能拼接。取两者中更长的那份作为最新转写，避免"喂喂豆包喂豆包为什么..."式重复。
        const incoming = (typeof msg.transcript === "string" && msg.transcript.length > 0)
          ? msg.transcript
          : (msg.delta || "");
        if (playbackMayStillBeAudible() && isProbablePlaybackEcho(incoming)) {
          console.log(`[SS ${ts()}] ${logTag} suppressed playback echo ASR delta`);
          pendingUserAsr = '';
          return;
        }
        providerAsrPcmBoundary.noteProviderDelta();
        clearProviderTurnBoundaryTimer();
        // 已有完整 provider 段后，新的真实 ASR delta 证明用户又继续说了。
        // 重新计起安静窗口，避免长讲述被第一段的旧 ceiling 提前收尾。
        if (pendingUserSegments.length > 0) providerFirstSegmentCompletedAt = 0;
        audioFlusher.flush("asr_delta");
        if (!turnTiming || turnTiming.response_end_at_ms) beginTurnTiming();
        if (!turnTiming.asr_started_at_ms) turnTiming.asr_started_at_ms = Date.now();
        if (incoming.length > pendingUserAsr.length) {
          pendingUserAsr = incoming;
        }
        const responseUserTurnAlreadyFinalized = directResponseTurnId && transcript.some((row) =>
          row.role === 'user' && row.turnId === directResponseTurnId
        );
        if (directE2EEnabled && responseUserTurnAlreadyFinalized &&
            (directResponseActive || controlledAwaitingPlaybackAck) &&
            directE2ETurnId === directResponseTurnId) {
          // Echoes were rejected above. Keep the current reply ID stable until
          // the client confirms barge-in, while preparing the next user turn.
          directE2ETurnId = createId("turn_");
        }
        clientWs.send(JSON.stringify({ type: "asr", text: previewUserTurn(pendingUserAsr) }));
      } else if (msg.type === "conversation.item.input_audio_transcription.completed") {
        // Provider VAD completed one ASR segment.  This is not necessarily the
        // learner's turn boundary: the browser has the longer adaptive silence
        // detector and explicitly sends stop_speech when the utterance is done.
        audioFlusher.flush("asr_completed");
        const segmentText = (msg.transcript || pendingUserAsr).trim();
        pendingUserAsr = '';
        if (!segmentText) {
          // The provider may emit ``completed`` after an empty ``started``
          // event (often immediately after TTS).  It is transport noise, not
          // a learner turn: do not cancel playback, start a merge timer or
          // replay text left in an older provider segment.
          audioBytesSinceProviderFinal = 0;
          providerAsrPcmBoundary.noteProviderCompleted();
          console.log(`[SS ${ts()}] ${logTag} ignored empty provider ASR completion`);
          // §8.67：空 ASR 说明"这一轮没接住"，也要开始人还在不在的计时。
          if (idleStage === 0) { onActivityEnded(); }
          return;
        }
        if (playbackMayStillBeAudible() && isProbablePlaybackEcho(segmentText)) {
          // Never let speaker echo cancel controlled TTS or become a learner
          // turn.  This was the server-side half of spontaneous self-barge-in.
          audioBytesSinceProviderFinal = 0;
          audioFlusher.discard("playback_echo_segment");
          providerAsrPcmBoundary.noteProviderCompleted();
          console.log(`[SS ${ts()}] ${logTag} discarded completed playback echo segment`);
          return;
        }
        // plans/1011-13: a provider completion whose item started at or before the
        // client's last final covers audio the client already committed.  Drop it
        // instead of letting it become a SECOND learner turn.  (The old text-only
        // guard below cannot catch this: the two ASR passes often differ by a word.)
        const lateCompletion = providerAsrPcmBoundary.classifyProviderCompletion({
          clientCommitRequested,
          segmentText,
        });
        if (lateCompletion === 'complete_empty_client_commit') {
          clientCommitRequested = true;
        }
        providerAsrPcmBoundary.noteProviderCompleted({
          consumeEmptyClientCommit:
            lateCompletion === 'accept' || lateCompletion === 'complete_empty_client_commit',
        });
        if (lateCompletion !== 'accept' && lateCompletion !== 'complete_empty_client_commit') {
          audioBytesSinceProviderFinal = 0;
          const lateReason = lateCompletion === 'late_duplicate' ? 'duplicate' : 'text mismatch';
          console.log(
            `[SS ${ts()}] ${logTag} ignored provider ASR completion after client final (${lateReason}); `
            + `client=${transcriptForLog(lastFinalAsrText, 90)} provider=${transcriptForLog(segmentText, 90)}`
          );
          return;
        }
        appendUserSegment(segmentText);
        audioBytesSinceProviderFinal = 0;
        if (directE2EEnabled) {
          // Server VAD already owns the turn boundary and has auto-started its
          // reply. Do not cancel it or wait for the controlled-mode commit grace.
          finalizeDirectE2EUserTurn(segmentText);
          return;
        }
        suppressCurrentResponse = true;
        audioFlusher.discard("provider_vad_segment");
        if (doubaoWs?.readyState === WebSocket.OPEN) {
          doubaoWs.send(JSON.stringify({ type: "response.cancel" }));
        }
        if (!clientCommitRequested) {
          const preview = mergedUserTurn();
          if (preview) clientWs.send(JSON.stringify({ type: "asr", text: preview, segmentComplete: true }));
          if (sessionCassetteMode === 'expression' || sessionIsMaterialTape) {
            // 0924-08 练习盘：provider 段（含阿里垫片 28s 主动 commit）只是**切段**，
            // 不是轮次边界。练习盘关轮只认「说完了」或前端 20s 静音兜底；素材盘
            // 关轮依靠浏览器的真实静音窗口/显式 stop_speech，不能把 28s 切段当作说完。
            // 这里绝不能 scheduleProviderTurnBoundary：否则用户练表达一口气讲过 28s，
            // 垫片切段后一喘气就被替他关轮、AI 开回，后续说话全被判成 barge_in。
            console.log(
              `[SS ${ts()}] ${logTag} long-form tape: held provider segment ` +
              `(${segmentText.length} chars), waiting for client turn boundary`
            );
            return;
          }
          scheduleProviderTurnBoundary();
          console.log(
            `[SS ${ts()}] ${logTag} held provider ASR segment for a bounded ` +
            `${PROVIDER_TURN_BOUNDARY_GRACE_MS}ms merge window (${segmentText.length} chars)`
          );
          return;
        }
        finalizeClientCommittedTurn();
      } else if (msg.type === "response.created") {
        if (directE2EEnabled) {
          const responseId = directProviderResponseId(msg);
          if (!responseId) {
            console.log(`[SS ${ts()}] ${logTag} direct E2E response.created missing provider response id; ignoring response`);
            return;
          }
          if (directResponseId && directResponseId !== responseId) {
            // Realtime VAD may begin a new response before the preceding PCM has
            // drained from the browser. Seal the old turn as interrupted before
            // switching response IDs, so its late audio/ACK cannot affect the new one.
            abandonDirectE2EReply('barge_in');
          }
          if (!turnTiming || turnTiming.response_end_at_ms) beginTurnTiming();
          directResponseId = responseId;
          directResponseTurnId = directE2ETurnId || activeTurnId || createId("turn_");
          directResponseText = '';
          if (turnTiming) turnTiming.provider_response_created_at_ms = Date.now();
          directResponseActive = true;
          directResponseDonePending = false;
          directResponsePlaybackStarted = false;
          pendingAssistantText = '';
          directE2ETurnId = directResponseTurnId;
          clientWs.send(JSON.stringify({ type: 'response_pending', turnId: directResponseTurnId, mode: 'direct_e2e' }));
        }
        if (controlledPcmEnabled && !directE2EEnabled && controlledTtsActive && !awaitingAutoResponseDone) {
          const responseId = directProviderResponseId(msg);
          if (controlledProviderResponseId && responseId !== controlledProviderResponseId) return;
          if (controlledSeenResponseIds.has(responseId) && responseId !== controlledProviderResponseId) return;
          controlledProviderResponseId = responseId;
          if (responseId) controlledSeenResponseIds.add(responseId);
          if (!controlledProviderResponseId) {
            interruptControlledResponse('tts_missing_response_id');
            clientWs.send(JSON.stringify({ type: 'error', message: '语音回应未完成' }));
          }
        }
      } else if (msg.type === "response.output_text.delta") {
        if (directE2EEnabled) {
          if (!directProviderEventMatches(msg)) return;
          if (turnTiming && !turnTiming.first_text_at_ms) turnTiming.first_text_at_ms = Date.now();
          const delta = String(msg.delta || "");
          if (delta) {
            pendingAssistantText += delta;
            directResponseText += delta;
          }
          return;
        }
        if (turnTiming && !turnTiming.first_text_at_ms) turnTiming.first_text_at_ms = Date.now();
        // 彻底丢弃豆包模型自主生成的自由文本；受控模式下只允许后端 response_phrase 发给客户端
        return;
      } else if (msg.type === "response.output_text.done") {
        // text generation finished
        if (directE2EEnabled && directProviderEventMatches(msg) && !directResponseText.trim()) {
          directResponseText = String(msg.text || msg.transcript || "").trim();
          pendingAssistantText = directResponseText;
        }
      } else if (msg.type === "response.output_audio.delta") {
        if (directE2EEnabled) {
          if (!directProviderEventMatches(msg)) return;
          if (turnTiming && !turnTiming.first_audio_at_ms) turnTiming.first_audio_at_ms = Date.now();
          const audioBase64 = msg.delta || "";
          if (audioBase64 && clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(Buffer.concat([
              DIRECT_PCM_FRAME_PREFIX,
              Buffer.from(audioBase64, "base64"),
            ]), { binary: true });
          }
          return;
        }
        if (controlledPcmEnabled) {
          if (!controlledTtsActive || !controlledProviderResponseId ||
              directProviderResponseId(msg) !== controlledProviderResponseId) return;
          const audio = Buffer.from(msg.delta || '', 'base64');
          if (audio.length && clientWs.readyState === WebSocket.OPEN) {
            if (turnTiming && !turnTiming.first_audio_at_ms) turnTiming.first_audio_at_ms = Date.now();
            const header = Buffer.from(JSON.stringify({ type: 'audio', format: 'pcm_s16le',
              sampleRate: 24000, turnId: activeTurnId, phraseId: controlledCurrentPhrase?.id }));
            clientWs.send(Buffer.concat([header, Buffer.from('|||AUDIO|||'), audio]), { binary: true });
          }
          return;
        }
        if (turnTiming && !turnTiming.first_audio_at_ms) turnTiming.first_audio_at_ms = Date.now();
        if (!controlledTtsActive) return; // 只有受控TTS激活时才允许下发音频，彻底杜绝豆包自主音频泄漏
        const audioBase64 = msg.delta || "";
        if (audioBase64) {
          audioFlusher.push(Buffer.from(audioBase64, "base64"));
        }
      } else if (msg.type === "response.output_audio.done") {
        // audio generation finished
      } else if (msg.type === "response.done") {
        if (directE2EEnabled) {
          if (!directProviderEventMatches(msg)) return;
          const providerStatus = String(msg.response?.status || '').toLowerCase();
          if (providerStatus && providerStatus !== 'completed') {
            console.log(`[SS ${ts()}] ${logTag} direct E2E provider response ${directResponseId} ended with ${providerStatus}; discarding`);
            failDirectE2EResponse('direct_provider_response_failed');
            return;
          }
          directResponseActive = false;
          directResponseDonePending = true;
          if (turnTiming) turnTiming.provider_response_done_at_ms = Date.now();
          if (!directResponseText.trim()) {
            directResponseText = directResponseTextFromDone(msg);
            pendingAssistantText = directResponseText;
          }
          if (!directResponseText.trim()) {
            failDirectE2EResponse('direct_provider_response_missing_text');
            return;
          }
          const hasUserTurn = transcript.some((row) =>
            row.role === 'user' && row.turnId === directResponseTurnId
          );
          if (!hasUserTurn) {
            if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
            const expectedResponseId = directResponseId;
            directTranscriptWaitTimer = setTimeout(() => {
              if (directResponseId !== expectedResponseId || !directResponseDonePending) return;
              const arrived = transcript.some((row) =>
                row.role === 'user' && row.turnId === directResponseTurnId
              );
              if (!arrived) failDirectE2EResponse('direct_provider_response_missing_user_turn');
            }, 5000);
          }
          announceDirectE2EReplyDone();
          return;
        }
        // response.cancel 也会产生 response.done；它只代表豆包自由回复已停，
        // 不能在这里清空用户转写。保留到受控 TTS 真正结束后再评估。
        if (awaitingAutoResponseDone) {
          awaitingAutoResponseDone = false;
          autoResponseEnded = true;
          audioFlusher.discard("auto_response_canceled");
          maybeStartControlledTts();
          return;
        }
        if (controlledPcmEnabled && controlledTtsActive) {
          if (!controlledProviderResponseId || directProviderResponseId(msg) !== controlledProviderResponseId) return;
          const status = String(msg.response?.status || '');
          if (status !== 'completed') {
            interruptControlledResponse('tts_provider_failed');
            clientWs.send(JSON.stringify({ type: 'error', message: '语音回应未完成' }));
            return;
          }
        }
        if (controlledTtsActive) {
          const completedPhrase = controlledCurrentPhrase;
          // Each speech_text_buffer.commit produces an independent complete
          // Ogg response. Flush it and close the phrase before starting the
          // next one so the browser never tries to decode a partial Ogg page.
          audioFlusher.flush("controlled_phrase_done");
          clientWs.send(JSON.stringify({
            type: "response_phrase_done",
            turnId: activeTurnId,
            phraseId: completedPhrase?.id || '',
            index: completedPhrase?.index ?? 0,
          }));
          controlledTtsActive = false;
          controlledCurrentPhrase = null;
          suppressCurrentResponse = true;
          if (controlledPhraseQueue.length) {
            maybeStartControlledTts();
          } else if (controlledBackendDone) {
            announceControlledStreamDone();
          }
          return;
        }
        if (suppressCurrentResponse && !controlledTtsActive) {
          // Autonomous provider response (usually triggered by its own VAD).
          // It is transport noise for this product, not a completed tutor turn.
          audioFlusher.discard("discard_auto_response_done");
          return;
        }
        // AI response ended: persist assistant text to transcript and evaluate in background
        const voiceMetrics = finishVoiceMetrics();
        const completedAssistantText = pendingAssistantText.trim();
        if (completedAssistantText) {
          transcript.push({ role: 'assistant', content: completedAssistantText, ts: Date.now() });
          if (lastCompletedUserText && sessionId) {
            const completedUserText = lastCompletedUserText;
            callBackendJson('/session/turn', {
              session_id: backendSessionId || sessionId,
              turn_id: activeTurnId,
              user_transcript: completedUserText,
              assistant_response: completedAssistantText,
              voice_metrics: voiceMetrics,
            }, token).then((turnResult) => {
              clientWs.send(JSON.stringify({ type: 'turn_evaluated', result: turnResult }));
            }).catch((error) => {
              console.log(`[SS ${ts()}] ${logTag} /session/turn failed: ${error.message}`);
            });

            const inputDurationMs = Number(voiceMetrics?.asr_input_duration_ms);
            if (Number.isFinite(inputDurationMs) && inputDurationMs > 0) {
              callBackendJson('/session/speech-assessment', {
                user_id: currentUserId,
                session_id: backendSessionId || sessionId,
                transcript: completedUserText,
                duration_seconds: inputDurationMs / 1000,
                assessment_context: { target_lang: currentTargetLang || 'en' },
              }, token).then((assessment) => {
                clientWs.send(JSON.stringify({ type: 'speech_assessed', assessment }));
              }).catch((error) => {
                console.log(`[SS ${ts()}] ${logTag} /session/speech-assessment failed: ${error.message}`);
              });
            }
          }
        }
        pendingAssistantText = '';
        lastCompletedUserText = '';
        turnTiming = null;
        clearTurnEnergy();
        activeTurnId = '';
        controlledTtsActive = false;
        // 受控回复期间不通知客户端"回复结束"（受控回复自己的 response.done 才通知），
        // 否则客户端会提前回到监听态、受控音频到达时时序错乱
        if (!suppressCurrentResponse) {
          // 关键：先把所有待发音频冲刷给客户端，再发 response_done，
          // 保证"音频字节在前、结束信号在后"，否则客户端在空缓冲区时提前结束播放
          audioFlusher.flush("response_done");
          clientWs.send(JSON.stringify({ type: "response_done" }));
          sessionTurnCount += 1;
          idleStage = 0;
          onActivityEnded();
        }
        suppressCurrentResponse = true;
      } else if (msg.type === "error") {
        const errMsg = msg.error?.message || JSON.stringify(msg.error);
        console.log(`[SS ${ts()}] ${logTag} Doubao error: ${errMsg}`);
        // The realtime provider closes an idle audio session after roughly one
        // minute. session.closed immediately follows and the browser reconnects.
        // Do not surface this recoverable lifecycle event as a failed user turn.
        if (/AudioServerNoAudioInputTooLongError|no session active/i.test(errMsg)) {
          return;
        }
        // 若英语音色不被当前模型支持且会话尚未建立，自动用中文精品音色重试一次
        if (!speakerFallbackDone && !sessionCreated && activeSpeaker && lastSystemPrompt
            && /speaker|voice|音色|InvalidSpeaker/i.test(errMsg)) {
          speakerFallbackDone = true;
          activeSpeaker = process.env.DOUBAO_SPEAKER_ZH || "zh_female_vv_jupiter_bigtts";
          console.log(`[SS ${ts()}] ${logTag} Speaker unsupported, retrying with ${activeSpeaker}`);
          _startDoubaoSession(lastSystemPrompt, false);
          return;
        }
        clientWs.send(JSON.stringify({
          type: "error",
          message: `豆包服务错误: ${errMsg}`,
        }));
      }
    } catch(e) {
      console.log(`[SS ${ts()}] ${logTag} Doubao message error: ${e.message}`);
    }
  });

  doubaoWs.on("close", (code, reason) => {
    clearControlledDeadline();
    isDoubaoReady = false;
    sessionCreated = false;
    console.log(`[SS ${ts()}] ${logTag} Doubao closed, code: ${code}, reason: ${reason?.toString()}`);
    audioFlusher.flush("doubao_close");
    audioFlusher.close();
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify({ type: "provider_session_closed", reason: `socket_${code}` }));
    }
  });
  doubaoWs.on("error", (e) => {
    console.log(`[SS ${ts()}] ${logTag} Doubao error: ${e.message}`);
  });

  /**
   * 用给定的 system prompt 启动豆包 Seeduplex 3.0 会话。
   * firstTime=true 时不调后端 /session/start，直接用前端传来的 prompt。
   *
   * 新协议采用标准 Realtime 事件，但 asr/tts/dialog 专有能力通过
   * session.extension 透传，内部结构与原 S2S StartSessionPayload 一致。
   */
  function _startDoubaoSession(systemPrompt, isFirstTime) {
    // 语言感知音色：英语学习场景用美式英语原生音色（不再用中文女声 vv 说英语），
    // 其他语种回退到中文精品音色。可用环境变量覆盖：
    //   DOUBAO_SPEAKER_EN / DOUBAO_SPEAKER_ZH
    const speaker = currentTargetLang === "en"
      ? (process.env.DOUBAO_SPEAKER_EN || "en_female_dacey_uranus_bigtts")
      : (process.env.DOUBAO_SPEAKER_ZH || "zh_female_vv_jupiter_bigtts");
    activeSpeaker = speaker;
    lastSystemPrompt = systemPrompt;
    // 新协议没有 dialog_context 字段，把历史对话拼进 instructions
    let instructions = systemPrompt;
    // 2026-10-14 修正：原句是「就事论事推进业务，用户遇到困难时直接给出具体第一步英文示范」。
    // 前一句是从客服/业务模板抄来的（对闲聊陪伴毫无意义，只会把对话往办业务拽）；
    // 后一句直接命令模型在用户说「没懂/啥意思」时切进教学示范模式 —— 实测泄漏成
    // 逐词解释（"get by on very little 就是…"、"Unemployed 就是…"）。
    // Alice 现在的定位不是外教（见 api/agent/companion_contract.py「不是服务台」），
    // 两处指令方向必须一致，否则同一轮里互相抵消。只改语义方向，保留原有两条有效约束。
    instructions += "\n\n【硬性行为准则】绝不机械道歉，禁止使用sorry/抱歉/对不起；严禁重复盘问用户已回答的问题。用户没听懂时，用一句大白话把刚才那句换个说法重讲一遍，不要切换成教学或示范模式，也不要解释语法或词汇。";
    if (currentDialogContext && currentDialogContext.length > 0) {
      const historyText = currentDialogContext
        .map(item => `${item.role === "user" ? "User" : "Assistant"}: ${item.text}`)
        .join("\n");
      instructions += `\n\n【历史对话上下文，仅作参考，不要重复扮演用户】\n${historyText}`;
    }

    const payload = {
      type: "session.create",
      session: {
        id: dialogId,
        model: "1.2.6.1",
        instructions: instructions,
        extension: {
          ...(VOICE_TRANSPORT === 'dashscope' ? {
            direct_e2e: directE2EEnabled,
            controlled_pcm: controlledPcmEnabled,
            direct_e2e_vad_ms: DIRECT_E2E_VAD_MS,
          } : {}),
          asr: {
            audio_info: { format: "pcm_s16le", sample_rate: 16000, channel: 1 },
            // Provider VAD defaults to about 1.5s. Keep it above the browser's
            // adaptive 5s/8s decision so a thinking pause is not committed
            // by the provider before the client can apply unfinished-speech logic.
            extra: {
              end_smooth_window_ms: 9000,
              enable_asr_twopass: true,
            }
          },
          tts: {
            speaker: speaker,
            audio_config: { channel: 1, format: "pcm_s16le", sample_rate: 24000 }
          },
          dialog: {
            bot_name: "LanguageTutor",
            system_role: systemPrompt,
            // 2026-10-14 修正：原为「自然、友好、有真实轮流感的外教口吻。回复长度随内容变化，先回应用户再教学。」
            // 「外教口吻」「再教学」与 api/agent/companion_contract.py 的「Alice 不是外教」正相反，
            // 同一轮两套指令互相抵消，表现为教学腔 + 人格不成立。改为陪伴型定调，方向与合同一致。
            speaking_style: "自然、友好、有真实轮流感，像熟人聊天。回复长度随内容变化，先回应用户真正说的事。",
            dialog_id: dialogId,
            extra: { enable_user_query_exit: true }
          }
        }
      }
    };
    doubaoWs.send(JSON.stringify(payload));
    console.log(`[SS ${ts()}] ${logTag} Session started (firstTime=${isFirstTime}, promptLen=${systemPrompt.length}, history=${currentDialogContext.length}, speaker=${speaker}, lang=${currentTargetLang})`);
  }

  function tryParseConfig(raw) {
    const text = typeof raw === "string" ? raw : Buffer.isBuffer(raw) ? raw.toString("utf-8") : null;
    if (!text) return null;
    try {
      const cfg = JSON.parse(text);
      if (cfg && cfg.type === "config") return cfg;
      if (cfg && (
        cfg.type === "message" || cfg.type === "stop_response" ||
        cfg.type === "stop_speech" || cfg.type === "resume_turn" ||
        cfg.type === "interrupt_response" || cfg.type === "response_playback_done" ||
        cfg.type === "response_playback_started" || cfg.type === "response_playback_failed" ||
        cfg.type === "speech_activity"
      )) return cfg;
    } catch(e) {}
    return null;
  }

  clientWs.on("message", (data, isBinary) => {
    try {
      const config = tryParseConfig(data);
      if (config) {
        console.log(`[SS ${ts()}] ${logTag} Parsed config, type: ${config.type}, firstTime: ${config.firstTime}`);
        if (config.type === "config") {
          currentUserId = config.userId || "frontend-user";
          // §8.70：新连接 = 他重新点按钮进来 = 他回来了。之前的"走了"状态一律作废。
          idleStage = 0;
          pendingReturnedFromIdle = false;
          sessionTurnCount = 0;
          heardLearnerSpeech = false;
          clearIdleWatch();
          // §8.72：先看"上一轮连接里发生过什么"——这比 lastTurnAgoMs 更能说明连接状态。
          try {
            const _rec = IDLE_ASK_BY_USER.get(currentUserId);
            const _age = _rec ? Date.now() - _rec.at : Infinity;
            const _ago = Number(config.lastTurnAgoMs);
            if (_rec && _age < 180000) {
              // 刚问过 / 刚道别过，他又连回来了 = 他回来了。
              // 不再问"还在吗"；她安静等着，等他说第一句时接半句。
              sessionTurnCount = 0;
              heardLearnerSpeech = false;
              if (_rec.stage === 1) {
                pendingReturnedFromIdle = true;
                console.log(`[SS ${ts()}] ${logTag} idle: asked ${Math.round(_age / 1000)}s ago, he reconnected → wait quietly (no re-ask)`);
              } else {
                console.log(`[SS ${ts()}] ${logTag} idle: said goodbye ${Math.round(_age / 1000)}s ago → quiet path`);
              }
            } else if (Number.isFinite(_ago) && _ago >= 0 && _ago < 300000) {
              sessionTurnCount = 1;
              console.log(`[SS ${ts()}] ${logTag} idle: recent turn ${Math.round(_ago / 1000)}s ago → check-in path`);
            }
          } catch (_) {}
          // 这一盘还没有对话：如果他一直不说话，到点就**静默断开**（不开口）。
          if (IDLE_CHECKIN_ENABLED) onActivityEnded();
          dialogId = config.activeConvId || config.dialogId || ("dialog_" + currentUserId);
          currentScene = config.scene || "日常对话";
          currentTargetLang = config.targetLang || "en";
          sessionEntryMode = config.briefMode || "fallback";

          // plans/0919-02 §4.3：expression（练表达盘）是本批新增的第三种盘型取值。
          // 它与 continuation 同组：**不清历史**（回到这一盘要接着上一轮的任务讲），
          // 只把盘型原样透传给后端 /session/start 的 context_mode。
          const cassetteMode = ["resume", "continuation", "continuation_candidate", "fresh", "expression"].includes(config.contextMode)
            ? config.contextMode
            : (config.freshContext ? "fresh" : "resume");
          sessionCassetteMode = cassetteMode;   // 0924-08：给 provider 段分支判盘型
          sessionIsMaterialTape = String(config.scene || '').trim().toLowerCase() === 'material';
          const normalizedScene = String(config.scene || '').trim().toLowerCase();
          const directChatScene = ['daily', 'daily_talk', 'free_talk', 'daily talk', 'free talk', 'chat', '闲聊', '闲聊盘', '日常对话']
            .includes(normalizedScene);
          directE2EEnabled = VOICE_TRANSPORT === 'dashscope' &&
            DIRECT_E2E_CHAT_ENABLED && config.voiceE2E === true &&
            directChatScene && cassetteMode !== 'expression' &&
            !sessionIsMaterialTape && config.sceneTape !== true;
          controlledPcmEnabled = VOICE_TRANSPORT === 'dashscope' && CONTROLLED_PCM_ENABLED &&
            config.controlledPcm === true && !directE2EEnabled &&
            String(process.env.ALI_TTS_TRANSPORT || '').trim().toLowerCase() !== 'dedicated';
          directE2ETurnId = '';
          directResponseId = '';
          directResponseTurnId = '';
          directResponseText = '';
          directResponseActive = false;
          directResponseDonePending = false;
          if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
          directTranscriptWaitTimer = null;
          directResponsePlaybackStarted = false;
          directProviderSpeechStoppedAtMs = 0;
          clientWs.send(JSON.stringify({
            type: 'voice_reply_mode',
            mode: directE2EEnabled ? 'direct_e2e' : 'controlled',
            vadSilenceMs: directE2EEnabled ? DIRECT_E2E_VAD_MS : undefined,
          }));
          if (directE2EEnabled) {
            console.log(`[SS ${ts()}] ${logTag} opt-in direct E2E chat path enabled (VAD ${DIRECT_E2E_VAD_MS}ms)`);
          }
          const currentTapeContext = cleanDialogContext(config.history);
          const continuationContext = cleanDialogContext(config.continuationHistory);
          let dialogContext = currentTapeContext.length
            ? currentTapeContext
            : (cassetteMode === "continuation" ? continuationContext : currentTapeContext);
          currentDialogContext = dialogContext;
          console.log(`[SS ${ts()}] ${logTag} Loaded ${dialogContext.length} QA messages for dialog_context`);

          backendCacheKey = `${currentUserId}:${dialogId}`;
          const trulyFreshContext = cassetteMode !== "resume" && currentTapeContext.length === 0;
          if (trulyFreshContext) {
            backendSessionCache.delete(backendCacheKey);
          }
          const cachedBackend = trulyFreshContext ? null : getCachedBackendSession(backendCacheKey);

          if (cachedBackend) {
            backendSessionId = cachedBackend.sessionId;
            pendingConfig = cachedBackend.systemPrompt;
            console.log(
              `[SS ${ts()}] ${logTag} Reusing backend session_id: ${backendSessionId} ` +
              `(voice transport rotated; tutor context preserved)`
            );
            if (isDoubaoReady && pendingConfig) {
              _startDoubaoSession(pendingConfig, false);
              pendingConfig = null;
            }
            return;
          }

          // 关键修复：所有用户（包括首次）都走后端 /session/start，让 TutorAgent 注入知识库和教学方法
          // 首次用户的 discovery prompt 由 TutorAgent 内部处理，不再用前端本地 prompt
          {
            // 调后端 /session/start 拿 expert system prompt
            const apiPayload = JSON.stringify({
              user_id: config.userId || "frontend-user",
              target_lang: config.targetLang || "en",
              cefr: config.level || "",
              scene_name: config.scene || "daily_talk",
              scene_role: (config.langName || "English") === "Japanese" ? "Japanese Tutor" : "English Tutor",
              scene_goal: config.sessionGoal
                ? `Practice ${config.langName || "English"}: focus on ${config.sessionGoal}`
                : "Practice " + (config.langName || "English") + " naturally",
              recent_weakness: config.weakness || "grammar",
              conv_id: dialogId,
              context_mode: cassetteMode,
              source_conv_id: config.sourceConvId || "",
              // `freshContext` clears old dialogue only when the tape is truly
              // empty.  Browser reconnects on the same clean tape must retain
              // the role/step state rebuilt from its newly recorded history.
              fresh_context: trulyFreshContext,
              history: dialogContext.map(backendHistoryMessage),
            });

            console.log(`[SS ${ts()}] ${logTag} Calling backend /session/start...`);
            const apiReq = request({
              hostname: BACKEND_HOST,
              port: BACKEND_PORT,
              path: "/session/start",
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "Content-Length": Buffer.byteLength(apiPayload),
                // token 透传：让后端能识别真实用户身份（后端可选择性校验）
                ...(token ? { "Authorization": `Bearer ${token}` } : {}),
              },
            }, (apiRes) => {
              let body = "";
              apiRes.on("data", (chunk) => { body += chunk; });
              apiRes.on("end", () => {
                try {
                  const apiData = JSON.parse(body);
                  pendingConfig = apiData.system_prompt;
                  if (apiData.session_id) {
                    backendSessionId = apiData.session_id;
                    console.log(`[SS ${ts()}] ${logTag} Using backend session_id: ${backendSessionId}`);
                  }
                  cacheBackendSession(backendCacheKey, {
                    sessionId: backendSessionId,
                    systemPrompt: apiData.system_prompt,
                  });
                  console.log(`[SS ${ts()}] ${logTag} Expert prompt: ${apiData.system_prompt.length} chars, ${apiData.knowledge_chunks?.length || 0} knowledge chunks`);
                } catch(e) {
                  pendingConfig = config.systemRole;
                  console.log(`[SS ${ts()}] ${logTag} API fallback (parse error: ${e.message})`);
                }
                if (isDoubaoReady && pendingConfig) {
                  _startDoubaoSession(pendingConfig, false);
                  pendingConfig = null;
                } else {
                  console.log(`[SS ${ts()}] ${logTag} Waiting: isDoubaoReady=${isDoubaoReady}, pendingConfig=${!!pendingConfig}`);
                }
              });
            });
            apiReq.on("error", (e) => {
              pendingConfig = config.systemRole;
              console.log(`[SS ${ts()}] ${logTag} API fallback (network error: ${e.message})`);
              if (isDoubaoReady && pendingConfig) {
                _startDoubaoSession(pendingConfig, false);
                pendingConfig = null;
              }
            });
            apiReq.write(apiPayload);
            apiReq.end();
          }
        } else if (config.type === "resume_turn") {
          const resumedText = String(config.user_transcript || "").trim();
          const resumedTurnId = String(config.turn_id || "").trim();
          if (resumedText && resumedTurnId) {
            console.log(
              `[SS ${ts()}] ${logTag} Resuming in-flight tutor turn ${resumedTurnId} ` +
              `after voice transport rotation`
            );
            if (!turnTiming || turnTiming.response_end_at_ms) beginTurnTiming();
            turnTiming.asr_started_at_ms = Date.now();
            turnTiming.asr_final_at_ms = Date.now();
            activeTurnId = resumedTurnId;
            lastCompletedUserText = resumedText;
            transcript.push({ role: 'user', content: resumedText, ts: Date.now(), turnId: resumedTurnId });
            if (directE2EEnabled) {
              if (!requestDirectE2ETextReply(resumedText, resumedTurnId)) {
                clientWs.send(JSON.stringify({ type: 'error', error: 'direct_resume_unavailable', turnId: resumedTurnId }));
              }
            } else {
              requestControlledReply(resumedText, false, resumedTurnId);
            }
          }
        } else if (config.type === "stop_response") {
          // 本地立即停止并作废迟到回复；供应商只负责取消仍在输出的 TTS。
          interruptControlledResponse('user_stop');
        } else if (config.type === "interrupt_response") {
          // Barge-in is not a session hang-up.  Cancel only the assistant turn,
          // keep the socket alive, then accept the learner's new audio.
          interruptControlledResponse('barge_in', config.played_ms);
        } else if (config.type === "response_playback_done") {
          // Only user-heard content may enter transcript, assessment, and memory.
          const playbackTurnId = String(config.turn_id || '');
          if (directE2EEnabled) {
            if (playbackTurnId !== directResponseTurnId || !directResponsePlaybackStarted) return;
            const userTurn = [...transcript].reverse().find((row) =>
              row.role === 'user' && row.turnId === playbackTurnId
            );
            if (!userTurn) return;
            activeTurnId = playbackTurnId;
            lastCompletedUserText = String(userTurn.content || '').trim();
          }
          finalizeControlledTutorTurn(playbackTurnId);
        } else if (config.type === "response_playback_failed") {
          if (controlledPcmEnabled && String(config.turn_id || '') === activeTurnId &&
              (controlPending || controlledTtsActive || controlledAwaitingPlaybackAck)) {
            interruptControlledResponse('playback_failed', Math.max(0, Number(config.played_ms) || 0));
            return;
          }
          if (!directE2EEnabled ||
              String(config.turn_id || '') !== directResponseTurnId) return;
          interruptControlledResponse('playback_failed');
        } else if (config.type === "response_playback_started") {
          const playbackTurnId = String(config.turn_id || '');
          if (directE2EEnabled && directResponseId &&
              playbackTurnId === directResponseTurnId &&
              (directResponseActive || directResponseDonePending || controlledAwaitingPlaybackAck)) {
            directResponsePlaybackStarted = true;
          }
          const playbackBelongsToActiveTurn = directE2EEnabled
            ? playbackTurnId === directResponseTurnId
            : (!config.turn_id || playbackTurnId === String(activeTurnId));
          if (
            turnTiming && !turnTiming.client_playback_at_ms &&
            playbackBelongsToActiveTurn
          ) {
            turnTiming.client_playback_at_ms = Date.now();
          }
        } else if (config.type === "speech_activity") {
          if (turnTiming) turnTiming.last_speech_activity_at_ms = Date.now();
          // §8.68：这才是"人回来了"。
          noteUserSpeaking();
          // A provider transcription completion is only a segment boundary.
          // While the browser still hears real learner speech, keep all held
          // segments inside the same turn. But if provider already transcribed speech,
          // maintain the bounded boundary timer so background room noise never traps the session.
          // 练习盘和素材盘不维护这条 provider 自动关轮；浏览器静音窗/显式停止
          // 才是这两种长表达盘的轮次边界。
          if (
            pendingUserSegments.length > 0 &&
            sessionCassetteMode !== 'expression' &&
            !sessionIsMaterialTape
          ) {
            scheduleProviderTurnBoundary(true);
          }
        } else if (config.type === "stop_speech") {
          if (directE2EEnabled) {
            // If the learner stops before server VAD has seen enough trailing
            // silence, append timestamped PCM silence so VAD can close the
            // input and start its single automatic response. This synthetic
            // padding is excluded from learner audio duration and assessment.
            if (!directProviderSpeechStoppedAtMs && !directResponseActive &&
                audioBytesSinceProviderFinal > 0 && doubaoWs?.readyState === WebSocket.OPEN) {
              const paddingMs = DIRECT_E2E_VAD_MS;
              const silence = Buffer.alloc(paddingMs * 32);
              doubaoWs.send(JSON.stringify({
                type: 'input_audio_buffer.append',
                audio: silence.toString('base64'),
              }));
              console.log(`[SS ${ts()}] ${logTag} direct E2E appended ${paddingMs}ms silence after stop_speech`);
            }
            if (turnTiming) turnTiming.end_by_silence = String(config.reason || '') !== 'manual';
            return;
          }
          // 1.5 秒内只允许一次 commit：客户端静音提交 + 服务端 VAD 可能重复触发
          // plans/0916-22 修复0：记录收尾方式（silence=2.4s 静音窗自动提交 /
          // manual=用户手动停），供录音完整性台账区分"没说完就停"与"正常收尾"。
          if (turnTiming) {
            turnTiming.end_by_silence = String(config.reason || '') !== 'manual';
          }
          const now = Date.now();
          if (now - lastCommitAt > 1500) {
            lastCommitAt = now;
            clientCommitRequested = true;
            clearClientCommitTimer();
            clearProviderTurnBoundaryTimer();
            const hasUnfinalizedProviderInput =
              audioBytesSinceProviderFinal > 0 || !!pendingUserAsr.trim();
            if (doubaoWs && doubaoWs.readyState === WebSocket.OPEN) {
              console.log(`[SS ${ts()}] ${logTag} committing user audio after client silence/stop`);
              if (audioBytesSinceProviderFinal > 0) {
                doubaoWs.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
              }
            }
            // If ASR already finalized every segment, there is no provider work
            // left to settle. Avoid the fixed fallback wait on this common path;
            // retain it whenever PCM or interim text still needs final ASR.
            if (!hasUnfinalizedProviderInput && pendingUserSegments.length > 0) {
              console.log(`[SS ${ts()}] ${logTag} finalizing already-complete provider ASR without settle wait`);
              finalizeClientCommittedTurn();
            } else {
              clientCommitTimer = setTimeout(() => finalizeClientCommittedTurn(), 450);
            }
          }
        } else if (config.type === "message") {
          const userText = String(config.content || '').trim();
          if (userText) {
            if (directE2EEnabled && (directResponseId || controlledAwaitingPlaybackAck)) {
              interruptControlledResponse('barge_in');
            }
            beginTurnTiming();
            turnTiming.asr_final_at_ms = Date.now();
            activeTurnId = createId("turn_");
            lastCompletedUserText = userText;
            transcript.push({ role: 'user', content: userText, ts: Date.now(), turnId: activeTurnId });
            clientWs.send(JSON.stringify({ type: "asr_ended", text: userText, turnId: activeTurnId }));
            refineUserTranscript(activeTurnId, userText);
            if (directE2EEnabled) {
              if (!requestDirectE2ETextReply(userText, activeTurnId)) {
                clientWs.send(JSON.stringify({ type: 'error', error: 'direct_message_unavailable', turnId: activeTurnId }));
              }
            } else {
              requestControlledReply(userText, false, activeTurnId);
            }
          }
        }
      } else if (isBinary && (data instanceof Buffer || data instanceof ArrayBuffer)) {
        // 上行 PCM 音频
        if (doubaoWs && doubaoWs.readyState === WebSocket.OPEN) {
          const pcm = Buffer.from(data);
          // ⛔ 不要在这里取消计时：真实麦克风**一直在推静音帧**，用"有音频帧"当
          // "人回来了"会让计时每秒被清零（真机表现：安静后直接断线，从没问过）。
          // 真正的开口信号是客户端的 `speech_activity`，见下面那个分支。
          lastAudioForwardedAt = Date.now();
          providerAsrPcmBoundary.noteClientPcm(pcm.length);
          audioBytesSinceProviderFinal += pcm.length;
          turnPcmBytes += pcm.length;
          profileTurnAudio(pcm);
          doubaoWs.send(JSON.stringify({
            type: "input_audio_buffer.append",
            audio: pcm.toString("base64"),
          }));
        } else {
          console.log(`[SS ${ts()}] ${logTag} Doubao not open, dropping audio chunk`);
        }
      } else {
        console.log(`[SS ${ts()}] ${logTag} Ignored unknown non-binary client message`);
      }
    } catch(e) {
      console.log(`[SS ${ts()}] ${logTag} Client message error: ${e.message}`);
    }
  });

  clientWs.on("close", (code, reasonBuffer) => {
    // §8.69：连接没了就别再计时——否则 compose 完成后往死 socket 发，
    // 日志写着"sent"、用户什么也听不到，换连接后还会再问一遍。
    idleStage = 2;
    clearIdleWatch();
    controlledRequestSeq += 1;
    clearInterval(keepaliveTimer);
    clearClientCommitTimer();
    clearProviderTurnBoundaryTimer();
    clearControlledPlaybackAck();
    if (directTranscriptWaitTimer) clearTimeout(directTranscriptWaitTimer);
    directTranscriptWaitTimer = null;
    if (controlledStreamRequest) {
      controlledStreamRequest.abort("client_closed");
      controlledStreamRequest = null;
    }
    clearControlledDeadline();
    clearControlledProgress();
    const closeReason = reasonBuffer ? reasonBuffer.toString() : "";
    const transportRotation = closeReason === "session_reconfigure";
    console.log(`[SS ${ts()}] ${logTag} Client disconnected, reason=${closeReason || '(none)'}`);
    audioFlusher.flush("client_close");
    audioFlusher.close();
    if (doubaoWs && doubaoWs.readyState === WebSocket.OPEN) {
      doubaoWs.send(JSON.stringify({ type: "session.close" }));
      setTimeout(() => doubaoWs.close(), 300);
    } else if (doubaoWs) {
      doubaoWs.close();
    }

    if (transportRotation) {
      console.log(
        `[SS ${ts()}] ${logTag} Voice transport rotation only; preserving backend session ` +
        `${backendSessionId || '(none)'}`
      );
      return;
    }

    if (backendCacheKey) backendSessionCache.delete(backendCacheKey);

    // 会话结束：调后端 /session/end 进行 cefr 评估和画像进化，然后触发 audit
    // 只在有实质对话时触发，避免空会话浪费 token
    // A tape/side switch is an explicit Tutor session boundary.  Finalize even
    // when this particular voice transport has a short transcript: the
    // FastAPI session itself may already contain earlier turns preserved over
    // transport rotations.
    if (transcript.length >= 2 || (closeReason === "conversation_switch" && backendSessionId)) {
      _endSessionAndTriggerAudit(currentUserId, dialogId, backendSessionId || sessionId, transcript, currentScene);
    } else {
      console.log(`[SS ${ts()}] ${logTag} Transcript too short (${transcript.length}), skip end_session/audit`);
    }
  });
  clientWs.on("error", (e) => {
    console.log(`[SS ${ts()}] ${logTag} Client ws error: ${e.message}`);
  });
});

/**
 * 会话结束：调后端 /session/end 进行 cefr 评估和画像进化，然后触发 audit。
 * 注意：对话保存只由小程序端 chat.js 的 _saveConversation 调用一次，
 * 避免这里重复保存导致 error_fingerprints 双倍计数。
 *
 * 异步执行，不阻塞 ws 关闭。失败只记日志，不影响主流程。
 */
function _endSessionAndTriggerAudit(userId, dialogId, sessionId, transcript, scene) {
  const tag = `[SS ${ts()}]`;
  const endSessionId = sessionId;
  console.log(`[SS ${ts()}] [end-session] userId=${userId}, sessionId=${endSessionId}, transcript=${transcript.length} turns`);

  // 1. 调 /session/end：触发 cefr 评估、weakness 进化、画像更新
  // FastAPI 端点接收查询参数 ?session_id=...
  const endPath = `/session/end?session_id=${encodeURIComponent(endSessionId)}`;
  const endReq = request({
    hostname: BACKEND_HOST,
    port: BACKEND_PORT,
    path: endPath,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': 0 },
  }, (res) => {
    let body = '';
    res.on('data', (c) => body += c);
    res.on('end', () => {
      console.log(`${tag} [end-session] /session/end response: ${res.statusCode}`);
      // 2. /session/end 成功后触发 audit
      if (res.statusCode === 200) {
        console.log(`${tag} [audit] queued by backend for session ${sessionId}`);
      }
    });
  });
  endReq.on('error', (e) => {
    console.log(`${tag} [end-session] /session/end error: ${e.message}`);
    // 即使 /session/end 失败，也尝试触发 audit
    _triggerAudit(userId, sessionId);
  });
  endReq.end();
}

/**
 * 触发 audit 审计（异步，不阻塞）。
 * audit 会分析对话，发现问题后调 repair_executor 自动修复。
 * behavior_issue 类问题会调整 style_parameters，下次对话时回流到 prompt。
 */
function _triggerAudit(userId, sessionId = '') {
  const tag = `[SS ${ts()}]`;
  const payload = JSON.stringify({ user_id: userId, force: true, session_id: sessionId || undefined });

  const req = request({
    hostname: BACKEND_HOST,
    port: BACKEND_PORT,
    path: '/audit/trigger',
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
  }, (res) => {
    let body = '';
    res.on('data', (c) => body += c);
    res.on('end', () => {
      console.log(`${tag} [audit] trigger response: ${res.statusCode}`);
      if (res.statusCode === 200) {
        try {
          const result = JSON.parse(body);
          const recs = result.recommendations || [];
          const behaviorRecs = recs.filter(r => r.issue_type === 'behavior_issue');
          const repair = result.repair_result;
          console.log(`${tag} [audit] done: ${recs.length} recommendations, ${behaviorRecs.length} behavior_issue, repair executed=${repair?.executed || 0}`);
        } catch (e) {
          console.log(`${tag} [audit] parse result error: ${e.message}`);
        }
      }
    });
  });
  req.on('error', (e) => console.log(`${tag} [audit] trigger error: ${e.message}`));
  req.write(payload);
  req.end();
}

httpServer.listen(PORT, () => {
  console.log(`[SS] doubao-session-server (Seeduplex 3.0) listening on port ${PORT}`);
  console.log(`[SS] Backend API: ${BACKEND_PROTOCOL}//${BACKEND_HOST}:${BACKEND_PORT}`);
  console.log(`[SS] AI audio: Ogg Opus (raw passthrough, client sniffs format)`);
  console.log(`[SS] audio flush threshold: ${AUDIO_FLUSH_THRESHOLD} bytes, interval: ${FLUSH_INTERVAL_MS}ms`);
});
