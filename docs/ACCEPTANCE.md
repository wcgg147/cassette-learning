# 验收记录 · ACCEPTANCE

> 2026-11-09｜精简后当场重跑，含 14 个既有红测的逐条审计。
> 证据等级按 `AGENT_PROTOCOL.md`：E0 静态 · E1 自动化 · E2 服务真跑 · E3 真机/浏览器。
> **E3 只做到「浏览器真点四盘」；真机语音因为账号欠费未闭环，见 §5。**

---

## 0. 结论

| 层级 | 项 | 结果 |
|---|---|---|
| E0 | 依赖闭包、无悬空 import | ✅ 反查 145 个 .py |
| E0 | 前端 JS 语法 | ✅ 3 个 script 块 `node --check` |
| E0 | 前端构建一致性 | ✅ 8/8 锚点 |
| E0 | 脱敏 | ✅ 真账号 UUID / 明文 key / 作者路径 = 0 命中 |
| E0 | 包内凭据文件 | ✅ 无 `.env` / `config.json` / `*.key`；只有空占位 `.env.example` |
| E1 | `agent` 单测 | ✅ **536 例全绿** |
| E1 | `test_runner` 契约 | ✅ **431 例全绿** |
| E2 | 零 key 起三件套 | ✅ 后端 + 代理（纯文本模式）+ 静态页 |
| E2 | 零 key 走完整前端链路 | ✅ 四盘建盘 + 素材成盘 全 200 |
| **E3** | **浏览器真点四盘** | ✅ 真 Chromium，四盘全部成盘，0 前端报错 |
| E3 | 真机语音（ASR→回复→TTS） | ⚠️ **未闭环**：ASR 通、回复模型因账号欠费超时 |

---

## 1. 14 个既有红测：逐条审计

拆包时这 14 个红在原树同样是红。按要求逐条查了根因，**没有一条是拆分改坏的**；
按「值得修就修、没用就删」处理完，现在归零。

### 1.1 七个开场测试 —— 测试缺前置，修测试

`test_runner/test_expression_opening_topic.py`（5 FAIL + 2 ERROR）

根因：`main.get_opening_line()` 第一行有个总开关 `CHAT_OPENING_ENABLED`，
默认 **false**（2026-10-18 用户拍板「入口不说话」）。测试没把这个开关打开就断言
「必须直出话题卡」，于是全部拿到 `should_open=False` / 无 `topic` 键。
原树 `.env` 也没设这个变量 ⇒ 原树同样红。

**判定：不是代码坏，是测试没声明自己依赖的前提。**
修法：在测试的 `_open()` 辅助里 `patch.object(config, "CHAT_OPENING_ENABLED", True)`，
**不动产品默认值**（那个 false 是产品决策）。修完 13/13 全绿。

### 1.2 六个素材盘测试 —— 测试不隔离环境，修测试

`test_runner/test_material_import.py::VideoPageCommentsTests`（6 FAIL）

根因：这组测试起一个本地 `127.0.0.1:<随机端口>` 桩服务器，再让 `link_fetcher`
去抓。但 **httpx 默认 `trust_env=True`，会读系统代理**（本机 Clash 在
`127.0.0.1:7890`）—— 连回环地址的请求也被送去代理，代理回
`400 The plain HTTP request was sent to HTTPS port`，于是 `ok=False`、
`kind` 退成 `link`。

实测证据：
```
getproxies() -> {'http': 'http://127.0.0.1:7890', ...}
直接 httpx.get(本地桩) -> 400 'plain HTTP request was sent to HTTPS port'
设 NO_PROXY=127.0.0.1 后跑该测试类 -> OK (59 tests)
```

**判定：抓取代码没问题，是测试没隔离「代理环境」这个外部条件。**
修法：`VideoPageCommentsTests.setUpClass` 里临时把 `127.0.0.1,localhost`
加进 `NO_PROXY`，`tearDownClass` 恢复。修完 59/59 全绿。

> 这条对使用者是真收益：任何开着代理的人（国内尤其常见）跑这组测试都会踩同样的坑。

### 1.3 一个场景盘测试 —— 化石测试，删除

`agent/test_learning_orchestrator.py::test_generic_service_greeting_cannot_establish_technology_expo_roleplay`

它断言：建场景那一轮回「Hi! 别担心，有什么我能帮你的吗？」时，
违规列表里应出现含「**双方角色**」的一条。

根因：含「双方角色」的违规文案只有一条，出自
`tutor_agent.py:14712` 的 `scene_announcement` 分支，而那条规则的语义是
**「不要宣告双方角色分工」**——是 2026-09-14 产品决策**反转后的新规则**
（代码注释写明：原规则要求「必须明确告诉用户双方角色」，实测导致
①用户抱怨"进角色还要提醒我" ②宣告时模型切中文，7 个场景全在进场景那轮说中文）。

实测这条回复**产生 0 条违规**，因为「有什么我能帮你的吗」在咖啡店/商店场景里
恰恰是**得体的角色内台词**。要用关键词拦它，第一个误伤的就是正常店员开场。

**判定：这条测试断言的是已被产品决策废弃的旧规则，且用规则近似判断会直接造成误伤。
按「没用就删」处理。** 删除后 agent 套件 536 例全绿。

> 保留的相关保护：`agent/test_expression_contract.py` 里 4 处
> `[Scene Entry]` 断言（prompt 层"第一句必须是角色内台词"）仍在，覆盖同一意图。

---

## 2. E0 · 静态

- 全包扫 `ark-` / `sk-` / 真账号 UUID / 作者绝对路径 / 局域网 IP：**0 命中**
- 包内**没有** `.env` / `config.json` / `*.pem` / `*.key`，只有 `.env.example`（全空占位）
- `config.py` 的 `00000000-…a11c/a11d` 是刻意的全零测试哨兵
- `192.168.1.1` 是 `test_public_page_reader.py` 检测 SSRF 的 RFC1918 测试夹具
- 路由：原树 75 → 副本 68（少 4 条 `/audit/*` + 3 条 `/pipeline/*`），四盘关键路由 0 缺失

---

## 3. E1 · 自动化测试

```bash
cd opensource/backend/app
MEMORY_BACKEND=local CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s agent -p "test_*.py"
# Ran 536 tests — OK

MEMORY_BACKEND=local WX_DEV_MODE=1 CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s test_runner -p "test_*.py"
# Ran 431 tests — OK
```

占位 key 只让导演模块不退化（测试用假 client，**不联网**）。

---

## 4. E2 · 零 key 服务真跑

无任何语音/模型 key 起三件套（后端 8002 / 代理 3211 / 静态页 8080），
代理进入**纯文本模式**：

```
[SS] 未配置语音 API Key → 以【纯文本模式】启动。
[SS]   /api/* 透传、登录、建盘、素材库、词块本、会话均可用；
[SS]   语音（ASR/TTS/点词发音）不可用。配好 .env 里的 key 后重启即恢复。
```

经代理（前端真实路径）逐项实测全 200：

| 调用 | 结果 |
|---|---|
| `POST /api/auth/dev-login` | 200 token + user_id |
| `POST /api/expression/start` | 200 `expr-*_A` |
| `POST /api/scene/start` | 200 `scene-*_A` + `scene_name` |
| `POST /api/materials/{uid}` | 200 `mat-*_A`, status ready |
| `POST /api/material-library/{uid}/text` | 200 |
| `GET /api/chunks/{uid}/notebook` | 200 |
| `GET /api/knowledge/stats` | 200 `{"total_chunks":52,"backend":"local_json"}` |
| `GET /api/settings/language-mode` | 200 |
| `POST /session/start` | 200 session_id + 人格 system_prompt |

---

## 5. E3 · 浏览器真点四盘（已完成）

真 Chromium（Playwright 145，headless），页面自己生成随机 device UUID，
全程零前端报错。

| 盘 | 点击路径 | 结果 |
|---|---|---|
| 闲聊盘 | 带柜 → 新空白带 | 200 opening-line；标题「新会话 02」 |
| 练表达盘 | 带柜 → 练表达 | 200 `expression/start` + `topic-source` + opening-line(scene=expression) + vocab-tts；标题「练表达 · 2026-11-09」 |
| 场景盘 | 带柜 → 场景演绎 → 选「咖啡店」 | 200 `scene/start` + opening-line(scene=咖啡店)；开场白是店员口吻的英文（"Hi there! ... What can I get started for you?"） |
| 素材盘 | 带柜 → 导入素材 → 粘贴文字 → 导入 → 素材库 → 勾选 → 分析并成盘 | 200 `material-library/text` → `material-tapes` → 轮询 `material-tape-builds` → `material-tapes/matb00000001_A`；成盘后自动进盘，标题「眼睛酸」 |

素材盘完整链路（真实请求序列）：
```
POST /api/material-library/{uid}/text            200
GET  /api/material-library/{uid}?section=material 200
POST /api/material-tapes/{uid}                    200
GET  /api/material-tape-builds/{uid}/1            200
GET  /api/material-tapes/{uid}/matb00000001_A     200
```
截图存于本次验收目录（`e3_shots/`）。

### 5.1 E3 顺带发现（未改，留给你判断）

1. **场景盘的盘型标签显示「闲聊盘」**。`tapeKind()` 只认 `material` / `expression` /
   其余一律 `daily`，所以场景带的状态标签与磁带柜分类都归到「闲聊盘」。
   原树同样如此，不是拆分引入的。功能本身正常（开场白、scene 元数据都对）。
2. **素材带成盘后 `hooks` / `top_chunks` / `group.summary` 为空**。
   界面上的两条讨论点是模板话术，不是从这份素材里提炼的。
   本次回复模型不可用（见 §6），分析走的是降级路径；**无法区分"降级导致"还是
   "本来就这样"**，需要有可用模型时复验一次再定性。

---

## 6. 没验到的（如实说明）

- ⛔ **真机语音（ASR→回复→TTS）未闭环**。链路本身跑通了：真实 3211 WS +
  Windows SAPI 合成音频推流后，收到 16 个 `asr` 事件、`asr_ended`、
  `response_pending`、`response_progress` —— 说明 **ASR 转写与轮次判定正常**。
  但回复生成报 `reply generation timed out`：产品 `.env` 里的模型凭据全部失效
  （opencode key 已过期 `api_key_expired`、DeepSeek `Insufficient Balance`、
  DashScope 账户欠费），换本机网关也被限流。
  **这是账号/额度问题，不是代码问题**；有可用模型后重跑本档 §6 的命令即可。
- ⛔ **TTS 合成与音频回放**未验（同上，需要可用的语音 key）。
- ⛔ **真麦克风 / 打断 / 声学延迟**未验（那必须真机）。
- ⛔ **Windows 以外平台**：按你的要求不验。
- ⛔ **Postgres/Redis 形态**：本包故意不含。

---

## 7. 复跑命令

```bash
cd opensource
py -3.12 tools/check_frontend_js.py
cd frontend && py -3.12 build_web.py --check && cd ..
cd backend/app
MEMORY_BACKEND=local CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s agent -p "test_*.py"
MEMORY_BACKEND=local WX_DEV_MODE=1 CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s test_runner -p "test_*.py"
```

有可用模型/语音 key 后补语音验收：
```bash
# 1) 后端（把 TUTOR_RESPONSE_* 指向可用端点）
cd backend/app && MEMORY_BACKEND=local WX_DEV_MODE=1 \
  py -3.12 -m uvicorn main:app --host 127.0.0.1 --port 8002
# 2) 代理
cd voice && node doubao-session-server.mjs
# 3) 推合成音频跑一轮（16k/mono/s16le，100ms 一帧；说完继续推静音帧 + stop_speech）
```
