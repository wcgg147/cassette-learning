# 拆出清单 · EXTRACTION

> 2026-11-09｜从 `work/1010-练习盘导演修复` 这棵运行树拆出。
> **原树一行未改**。本包是副本，主产品继续用原树跑。
>
> 原则（用户口径）：**只留用户会碰到的、和产品使用/体验有关的文件与代码。**
> 不是"照抄整棵树"，也不是"按语义切四盘"。

---

## 1. 为什么不能按「四盘」切

四个盘共用**一条逐轮流水线**，差异是上下文里的盘型标志分支，不是四套独立模块：

| 盘 | 判据 | 落点 |
|---|---|---|
| 练习表达盘 | `ctx.cassette_mode == "expression"` | `tutor_agent.py:9428` |
| 素材盘 | `ctx.material_context` 非空 / `conversation_meta.scene == "material"` | `tutor_agent.py:3427`、`:20553` |
| 场景盘 | `ctx.scene_context_set`、`agent/scene_world.resolve()` | `agent/scene_world.py:303` |
| 闲聊盘 | 以上都不成立（默认） | `agent/chat_tape_policy.py:22` |

而且 `companion_persona_enabled` 是**按盘算的**（练习/素材关，闲聊/场景开，
`tutor_agent.py:3484`）—— 人格层就是闲聊盘和场景盘的体验本身，必须随包走，
不能当"后台自治"删掉。

⇒ 所以切法不是「抽出四盘」，而是「**整条对话链路留着，把跟用户使用无关的分支砍掉**」。

---

## 2. 留 / 不留

**留**（用户使用四盘 / 复读机会触发）：
- `frontend/` 母版 + 构建脚本 + 产物 —— 复读机/磁带拟物界面，唯一真源，单文件零外部资源
- `backend/app/main.py` 及其 68 条 HTTP 路由、四盘共用的 `agent/tutor_agent.py` 内核
- `engine/` 里四盘要用的：表达式五维、素材解析/归纳、词块、话题库、文件解析、
  链接抓取、视频音轨转写（**保留了 `media_transcribe`/`asr_filetrans`**，因为
  「贴视频链接」是素材盘用户会点的入口，删了会直接 500）
- `memory/` 本地后端（`l1_file` / `l2_sqlite` / `l3_archive` 等，SQLite 自建 39 张表）
- `rag/knowledge_rag_local.py`（本地 JSON 知识点检索）、`skill/`、`loop/`、`auth/`
- `voice/` 三件套（ASR/TTS + `/api/*` 透传）—— 前端所有请求都走它
- `knowledge_base/chunks/` 52 个知识点、`docs/`、`tools/`、四盘相关单测

**不留**（跟用户使用/体验无关）：

| 砍掉 | 理由 |
|---|---|
| `engine/alice_*` 8 个（刷 B站/X/小说、发动态、搜索、朗读） | AI 后台自治生活线，默认关，用户用四盘时完全不触发 |
| `live_model/` + `pipecat_pipeline/` + `/pipeline/*` 3 条路由 | 实时语音管线，跟四盘无关，且原实现 `_build_adapter()` 返回全仓无定义的 `OpenAIChatAdapter`，本来就是坏的 |
| `audit/` + `/audit/*` 4 条路由 | 会话质量自愈 + prompt 回滚；调用点都在 `try/except` 内，缺了只静默跳过 |
| `memory/l1_redis.py`、`l2_postgres.py`、`rag/knowledge_rag.py` | PostgreSQL + Redis 部署形态；开源包只保留纯本地一种 |
| `realtime_text_reply.py` / `realtime_text_history.py` | 实时纯文本语音实验旁路，默认不进 |
| `engine/prompt_optimizer.py` / `trajectory_compressor.py` | 自愈审计 / 提示优化的配套，主链路不用 |
| `backend/db/` 10 个 SQL | 全是 PG 迁移脚本；SQLite 自己建表 |
| 100+ 个 `diag_*` / `eval_*` / `probe_*` 脚本、探索原型 HTML、`cassette-MYTEST.html` | 一次性探针与死原型；其中 `cassette-MYTEST.html` 还带第二个真设备 UUID |
| `android-experience/` | APK 只是 WebView 壳，页面从服务端下发；带它会让母版出现两份真源 |
| `.runtime/`、`.workbuddy/`、`plans/`、`bugreports/`、`reports/`、`_*.txt`、*.log | 运行态 / 内部档案 / 一次性 dump |
| 原树的 `.env` | **有真 key，一个字都没拷** |

**不搬 ≠ 删**：原树里这些都还在，主产品照常跑。

---

## 3. 对副本做的改动（逐条可查）

业务逻辑一行没动，只有边界改动：

1. `main.py`：只保留 `MEMORY_BACKEND=local` 一种后端（非 local 明确报错）；
   移除后台「她自己读东西」sweeper；`/knowledge/stats` 去掉不可达的 PG 分支；
   摘掉 `/pipeline/*`、`/audit/*` 共 7 条路由及其顶层 import。
2. `rag/factory.py`、`memory/provider.py`、`engine/session_manager.py`、
   `memory/__init__.py`、`engine/__init__.py`：收敛成单后端，清掉对已删模块的
   `try/except ImportError` 悬空引用。
3. `rag/methodology_rag.py`：`import asyncpg` 改惰性 —— 纯本地模式不再需要装
   Postgres 驱动。
4. `agent/tutor_agent.py`：移除依赖已删模块的实时文本实验分支、PromptOptimizer
   死块；一处 debug 转储的绝对路径改成可移植（按 `__file__` / 环境变量推导）。
5. `voice/doubao-session-server.mjs`：**无 key 不再 `process.exit(1)`**，改为
   「纯文本模式」启动 —— `/api/*` 透照常工作，WS 语音连接收到明确错误。
   这是启动期一次性分支：逐轮 0 次额外模型调用、0ms；有 key 时行为逐字一致。
6. 前端母版**脱敏 3 处**：写死的真账号 `DEVICE_UUID` → 运行时随机生成存
   localStorage；`RESET_PROTECTED_USER_ID` 置空；注释里的局域网 IP 泛化。
7. `build_web.py` 与 4 个 `test_runner` 契约测试：路径改成按新布局相对定位
   （只改路径常量，8 处替换锚点、断言逻辑一字未动）。
8. 新增：README / LICENSE / .gitignore / .env.example / requirements.txt /
   docs / tools（`route_dump.py`、`check_frontend_js.py`）。

`.env.example` 里所有 key 字段**全是空占位**，`WX_DEV_MODE=1` 是必需项。

---

## 4. 路由对照

```
原树 75 条 → 副本 68 条
缺失：/audit/{history,status,rollback,trigger}、/pipeline/{start,stop,update}
新增：无
四盘关键路由：0 缺失
```

---

## 5. 已知遗留（下一阶段审计，本阶段**故意不修**）

| 项 | 说明 |
|---|---|
| `tutor_agent.py` 约 23k 行 | 四盘与人格/记忆仍在同一主干；按盘分家是大改 |
| 前端约 7.3k 行单文件 | 拆组件是可读性优化，非功能问题 |
| 14 个既有红测 | `agent` 1 个、`test_runner` 13 个，原树完全一样的红；未擅自"修" |
| `?token=dev` | WS URL 里的字面量，代理不校验，生产要换 |
| 音频 base64 存 localStorage | 配额风险 + 语音数据落本地，产品既有行为 |
| config.py 约 150 个 env 开关 | 大部分是可选功能；.env.example 只列必需的 |
| `voice/` 约 4k 行 | 含两家语音协议适配与实验分支，代码级精简未做 |
| 原树 `web-dist` 落后母版一个版本 | 本包已按当前母版重新生成，反而比原树一致 |

---

## 6. 脱敏复核

全包扫描 `ark-` / `sk-` / `Bearer <literal>` / 真账号 UUID / 作者绝对路径 /
`ip:port` / 局域网 IP / 凭据类文件名：

- 真账号 UUID、**0 命中**；明文 API key，**0 命中**
- 包内**没有任何** `.env` / `config.json` / `*.pem` / `*.key`（只有 `.env.example`）
- `.env.example` 全部 key 字段为空
- `config.py` 的 `00000000-…a11c/a11d` 是刻意的全零测试哨兵，非个人数据
- `192.168.1.1` 是 `test_public_page_reader.py` 里检测 SSRF 的 RFC1918 测试夹具，保留
- `voice/doubao-session-server.mjs:90` 的 `apiKey: DASHSCOPE_API_KEY` 是**变量名**

---

## 7. 动了哪些文件

只新建 `opensource/` 一个顶层文件夹 + 一份方案档
`plans/1011-01-开源拆分-四盘与复读机磁带.md` + `.runtime/` 下若干验收中间产物。
**原树任何既有文件都没有增删改** —— 这是本次拆分最重要的一条纪律。
