# 会话磁带 · Cassette Learning

一个拟物风格的会话式语言学习前端 + 后端。把对话装进「磁带」，用「复读机」播放；
四个抽屉各是一种练习方式：

| 抽屉 | 界面叫法 | 干什么 |
|---|---|---|
| 闲聊盘 | `闲聊盘` | 跟 AI 自由聊；她有独立人格与长期记忆 |
| 练习表达盘 | `练表达盘` | 就一个真实话题连续讲一段，按「讲清楚」逐维给提升点 |
| 场景盘 | `场景演绎` | 进一个具体场景（咖啡店 / 餐厅 / 机场）做角色扮演 |
| 素材盘 | `素材盘` | 粘贴文字 / 贴链接 / 传文件，把正在看的东西变成能聊的一盘带 |

> 这个包是从主产品仓库**按「用户会不会碰到」精简出来的副本**，用于开源。
> 只留了四盘 + 复读机/磁带这条使用链路；AI 后台自治行为、另一套部署形态、
> 一次性探针全都没带进来。清单见 [`docs/EXTRACTION.md`](docs/EXTRACTION.md)，
> 验收实测见 [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)。

---

## 快速开始（**不需要任何账号或 key**）

三件套，各开一个终端，都从 `opensource/` 下起：

```bash
# 1) 后端 :8002 —— 纯 SQLite，零外部服务
cd backend/app
py -3.12 -m venv .venv
.venv\Scripts\activate             # Windows；Linux/macOS: . .venv/bin/activate
pip install -r ../../requirements.txt
cp ../../.env.example .env
py -3.12 -m uvicorn main:app --host 127.0.0.1 --port 8002

# 2) 语音代理 :3211 —— 没配 key 也能起（纯文本模式）
cd voice
npm install
cp ../.env.example .env           # 两个语音 key 留空即可
node doubao-session-server.mjs

# 3) 前端 :8080
cd frontend
py -3.12 -m http.server 8080 --bind 127.0.0.1
# 打开 http://127.0.0.1:8080/cassette-learning-interaction-review.html
```

**没配语音 key 时**：代理进入「纯文本模式」，`/api/*` 透传照常工作 ——
登录、四盘建盘、素材导入/成盘、词块本、会话全部可用；只有 ASR/TTS/点词发音
不可用。配好 `.env` 里的 `DASHSCOPE_API_KEY`（或 `DOUBAO_VOICE_API_KEY`）重启
即恢复语音。**前端是连 3211 的，不是直连 8002** —— 所以这一层必须起。

> ⚠️ 本机若裸 `python` 是 3.14 且缺依赖，一律用 `py -3.12`，否则会得到假红。

**配模型 key**：回复链路需要 `TUTOR_RESPONSE_BASE_URL` / `TUTOR_RESPONSE_API_KEY` /
`TUTOR_RESPONSE_MODEL`（任意 OpenAI 兼容端点）。不配也能建盘、导入素材、跑通
整条 HTTP 链路，只是 AI 回复会走兜底并重试约 1 分钟 —— 这是缺 key 的表现，
不是故障。`.env.example` 里列了所有需要填的项，全是空占位。

`WX_DEV_MODE=1` 是**必需项**：开源包没有微信小程序 AppID/Secret，不开这个开关
`/auth/wx_login` 会 500。生产环境请换成真实登录并关掉。

前端 `cassette-learning-interaction-review.html` 是唯一母版；改完要重跑
`py -3.12 build_web.py`（生成 `web-dist/index.html`），`--check` 只校验 8 处锚点。

---

## 目录

```
opensource/
├── README.md  LICENSE  .gitignore  .env.example  requirements.txt
├── frontend/
│   ├── cassette-learning-interaction-review.html   # 母版（唯一真源，零外部资源）
│   ├── build_web.py                                # 母版 → web-dist
│   └── web-dist/index.html                         # 构建产物
├── backend/
│   ├── app/                 # main.py + 逐轮内核 + engine/memory/rag/auth
│   │   ├── agent/           # 四盘共用内核 tutor_agent.py、盘型规则、单测
│   │   ├── engine/          # 表达式五维、素材解析/归纳、词块、话题库、文件解析
│   │   ├── memory/          # L1/L2/L3（只有本地 SQLite + JSON 一种后端）
│   │   ├── rag|skill|loop|auth|test_runner/
│   │   └── ...
│   └── knowledge_base/chunks/  # RAG 播种语料（52 个知识点）
├── voice/                  # ASR/TTS + /api 透传（3 个 .mjs + package.json）
├── tools/                  # 验收脚本（route_dump / check_frontend_js）
└── docs/                   # EXTRACTION.md / ACCEPTANCE.md
```

**没有** `db/` —— SQLite 后端自己建 39 张表（见 `memory/l2_sqlite.py`），不需要
SQL 迁移脚本。也**没有** Postgres / Redis 后端。

---

## 跑测试

后端单测用 **unittest**（不是 pytest）：

```bash
cd backend/app
MEMORY_BACKEND=local CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s agent -p "test_*.py"

MEMORY_BACKEND=local WX_DEV_MODE=1 CONVERSATION_DIRECTOR_API_KEY=sk-offline-test-dummy \
  py -3.12 -m unittest discover -s test_runner -p "test_*.py"
```

占位 key 是让导演模块不退化（测试用假 client，不联网）。
两套测试**目前全绿**（`agent` 536 例、`test_runner` 431 例）。拆包时带过来的
14 个红测已逐条审计：13 个是测试自己缺前置（总开关没开 / 没绕开系统代理），
改测试修好；1 个断言的是已被产品决策废弃的旧规则，删掉。逐条理由见
`docs/ACCEPTANCE.md` §1。
前端：`py -3.12 tools/check_frontend_js.py`（3 个 script 块 `node --check`）。

> 如果你开着 HTTP 代理跑测试：素材盘的抓取用例会在 `setUpClass` 里自己把
> `127.0.0.1` 加进 `NO_PROXY`，不需要你手动处理。

---

## 隐私 / 脱敏

- 母版里原来写死了开发者的真实设备 UUID（= 真账号 user_id，能直接登录读数据）。
  本包改成**首次运行生成随机 UUID 存 localStorage**，功能不变、源码里不再有任何人身份。
- 「回到初始」原来对真账号做硬拦截，本包去掉该保护（每人只删自己的数据，
  后端仍校验 owner）。
- 全包扫过 `ark-` / `sk-` / 明文 key / 真账号 UUID / 作者绝对路径 / 局域网 IP：
  **0 命中**。语音与模型 key 全部走环境变量，`.env` 在 `.gitignore` 里，
  仓库里只有空值占位的 `.env.example`。
- 前端把音频 base64 存在 localStorage（`cassette-state-v2`）—— 产品既有行为，
  既有配额风险也有「语音数据落本地」的隐私含义，继续开发时请留意。

---

## 许可证
> **语料版权**：`backend/knowledge_base/chunks/` 下 52 个 KN 文件为团队原创的外语教学法教学笔记（非抓取第三方原文），与代码采用同一许可证。


MIT（见 [LICENSE](LICENSE)）—— **这是拆分时放的占位选择，正式开源前请自己定**，
尤其是 `voice/` 里各家语音协议的适配层、`knowledge_base/chunks/` 语料的来源，
要不要沿用 MIT 由你拍板。
