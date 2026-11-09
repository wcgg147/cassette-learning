import os
import uuid

# 自动加载 .env 文件，使所有 os.getenv 调用都能读到 .env 中的配置（DASHSCOPE_API_KEY 等）
try:
    from dotenv import load_dotenv
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_curr_dir, ".env"))
    load_dotenv(os.path.join(os.path.dirname(_curr_dir), ".env"))
    # Machine-local overrides and secrets. This file is git-ignored and wins
    # over the legacy tracked development .env files.
    load_dotenv(os.path.join(_curr_dir, ".env.local"), override=True)
except ImportError:
    pass


def as_user_uuid(user_id) -> str:
    """把客户端送来的用户标识归一成 uuid 字符串。

    记忆层的 user_id 列都是 `uuid` 类型，而调用方可能送非 UUID 的标识
    （前端固定发 "browser-test"，语音代理默认发 "frontend-user"）。直接
    `$1::uuid` 会让 Postgres 抛 DataError，整条链路（会话启动、审计入库、
    知识用量回写）都在这里断掉。

    合法 UUID 原样返回；其余用 uuid5 确定性映射——同一个标识永远映射到
    同一个 UUID，所以学习记忆跨会话不会丢。
    """
    raw = str(user_id or "").strip()
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "lang-user:" + raw))

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "local").strip().lower()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://llm_user:llm_pass@localhost:5432/llm_os")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")

# -- Doubao / Volcengine Ark voice model --
LIVE_MODEL_PROVIDER = os.getenv("LIVE_MODEL_PROVIDER", "doubao")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_REALTIME_MODEL = os.getenv("DOUBAO_REALTIME_MODEL", "doubao-realtime-voice")
DOUBAO_REALTIME_ENDPOINT = os.getenv("DOUBAO_REALTIME_ENDPOINT",
    "wss://ark.cn-beijing.volces.com/api/v3/realtime/bots")

# Volcengine TTS / voice model credentials
VOLC_ACCESS_KEY = os.getenv("VOLC_ACCESS_TOKEN", "")
VOLC_SECRET_KEY = os.getenv("VOLC_SECRET_KEY", "")
VOLC_LIVE_APP_ID = os.getenv("VOLC_APP_ID", "")
VOLC_LIVE_MODEL = os.getenv("VOLC_LIVE_MODEL", "Doubao_scene_SLM_Doubao_realtime_voice_model2000000857496341314")
# 豆包 Dialog SDK 的 APP_KEY（原硬编码在 doubao-session-server.mjs 中，迁出到 env 以便生产部署）
VOLC_APP_KEY = os.getenv("VOLC_APP_KEY", "")

# ── 微信登录配置 ────────────────────────────────────────
# 小程序：在 mp.weixin.qq.com 注册后获取
# Android 移动应用：在 open.weixin.qq.com 注册后获取
# AppSecret 切勿泄露或提交到代码仓库
WX_APPID = os.getenv("WX_APPID", "")
WX_SECRET = os.getenv("WX_SECRET", "")
# 显式区分两个平台（未配置时回退到 WX_APPID/WX_SECRET 保持兼容）
WX_MINIAPP_APPID = os.getenv("WX_MINIAPP_APPID", "") or WX_APPID
WX_MINIAPP_SECRET = os.getenv("WX_MINIAPP_SECRET", "") or WX_SECRET
WX_ANDROID_APPID = os.getenv("WX_ANDROID_APPID", "")
WX_ANDROID_SECRET = os.getenv("WX_ANDROID_SECRET", "")

L1_MAX_TOKENS = 1000
L2_FACTS_PER_USER = 500
WEIGHT_DECAY_FACTOR = 0.95
WEIGHT_DELETE_THRESHOLD = 0.05
BOOST_SCORE_CLAMP = (-0.5, 0.5)
DREAM_DEBOUNCE_SECONDS = 1800
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "siliconflow")
EMBEDDING_DIM = 1024
# 知识库"这轮有没有覆盖"的判据阈值——低于它才认为库里没有，需要联网查证。
# 标定依据（2026-09-14，siliconflow 1024 维实测）：
#   已覆盖 "什么取餐码是什么意思" → 0.716，命中正确条目 KN-070
#   未覆盖 "I am renting an apartment…lease?" → 0.543，且命中的是"餐厅订位"这类不相关条目
# 换 embedding 供应商后必须重新标定；这只是启发式，不是普适阈值。
KNOWLEDGE_COVERAGE_MIN_SIMILARITY = float(
    os.getenv("KNOWLEDGE_COVERAGE_MIN_SIMILARITY", "0.60")
)
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Secrets must only come from the environment.  Do not add fallback keys here:
# this module is imported by the API, workers, and local test tooling.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")


def _api_key_for_endpoint(base_url: str, *candidates: str) -> str:
    """Use the matching gateway credential for configured OpenCode routes."""
    if "opencode.ai" in str(base_url or "").lower() and OPENCODE_API_KEY:
        return OPENCODE_API_KEY
    return next((str(value) for value in candidates if value), "")

# Policy-controlled tutor response generation.  The realtime voice provider
# remains responsible for ASR/TTS; this model only verbalizes the teaching
# action selected by the backend orchestrator.
TUTOR_RESPONSE_MODEL = os.getenv("TUTOR_RESPONSE_MODEL", "deepseek-v4.1-flash")
TUTOR_RESPONSE_BASE_URL = os.getenv(
    "TUTOR_RESPONSE_BASE_URL", OPENCODE_BASE_URL
)
TUTOR_RESPONSE_API_KEY = _api_key_for_endpoint(
    TUTOR_RESPONSE_BASE_URL,
    os.getenv("TUTOR_RESPONSE_API_KEY", "")
    or os.getenv("DEEPSEEK_API_KEY", "")
    or DOUBAO_API_KEY,
)
TUTOR_RESPONSE_MAX_TOKENS = int(os.getenv("TUTOR_RESPONSE_MAX_TOKENS", "320"))

# ── 回复长度的硬提示（plans/1017-02 §8.30，默认空＝逐字零变化）──────────────
# 真机 400 条回复实测：中位 **107 字**、41% 超 120 字（素材盘 141／练习盘 143）。
# 而长度约束只有 `TutorPolicy.response_length = "1-3个自然短句；解释时可更长"`
# —— **“解释时可更长”是个后门**，模型几乎总认为自己在解释。且该字段全库无人赋值。
# 本变量非空时**顶替**那句提示；为空＝原样。
TUTOR_REPLY_LENGTH_HINT = os.getenv("TUTOR_REPLY_LENGTH_HINT", "").strip()

# ── 改每轮的"默认动作"（plans/1017-02 §8.34，默认空＝逐字零变化）────────────
# 实测：591 条回复里只有 **13%** 在讲她自己，其余都在接话/追问。
# 而合同**明确允许**她"分享自己的独立看法""形成自己的生活叙事"——
# 卡点不在许可，在每轮的 policy 默认值：`conversation_hook` 永远是
# "围绕用户刚说的内容**自然追问**"（全库无任何地方改它）⇒ 每轮都被推去问用户。
# 本变量非空时顶替那两句默认（goal / hook）。为空＝原样。
TUTOR_POLICY_GOAL_HINT = os.getenv("TUTOR_POLICY_GOAL_HINT", "").strip()
TUTOR_POLICY_HOOK_HINT = os.getenv("TUTOR_POLICY_HOOK_HINT", "").strip()

# ── 开场把"她自己"带上（plans/1017-02 §8.31，默认关）────────────────────────
# 实测：主用户 39 条自述（偏好/性格/能力/语言环境/经历）里**没有一条** identity 或
# current_status，而开场只渲染这两类 ⇒ **开场渲染 0 行**，每盘从白纸开始；
# 只有用户问到相关句子时才渲染 6 行。开＝durable 为空时退回 active 自述前几条。
PERSONA_OPENING_RECALL_ENABLED = os.getenv(
    "PERSONA_OPENING_RECALL_ENABLED", "false"
).lower() in ("true", "1", "yes")

# ── 作者种子（plans/1017-02 §8.35，默认关）──────────────────────────────────
# 实测：591 条回复只有 13% 在讲她自己，可核验自述只剩 4 句 ⇒ 对话生不出"她的生活"。
# 种子（`memory/persona_seed.py` 五条）**只进 prompt、不写库**；
# 只要她自己 active 自述覆盖同一 facet，种子自动让位 ⇒ 可被她改写。
PERSONA_SEED_ENABLED = os.getenv(
    "PERSONA_SEED_ENABLED", "false"
).lower() in ("true", "1", "yes")

# ── 关系事件落库（plans/1017-02 §8.36，默认关）──────────────────────────────
# 实测：每轮都在抽关系事件（形状完整：confidence 0.9~0.96 / expires_at / confirmed_by_user），
# 但**从来没写过库**——下一轮又从头算，关系永远停在"刚认识"（117/117 用户）。
# 而"刚认识"的关系契约要求她"主动分享保持轻量" ⇒ 她只有 13% 的回复讲自己、可核验自述剩 4 句。
# 开＝把本轮事件写进 relationship_memory，关系才可能真的推进。
RELATIONSHIP_EVENT_PERSIST_ENABLED = os.getenv(
    "RELATIONSHIP_EVENT_PERSIST_ENABLED", "true"
).lower() in ("true", "1", "yes")

# ── 亲密度改由"她记得你多少"决定（plans/1017-02 §8.37，默认关）──────────────
# 实测：主用户记忆深度 237（实体 134＋有料归档 103），其余 4~20；
# 而她的说话方式几乎没差别（中位 95 vs 104~130、问句率 39% vs 10~31%），
# 因为 familiarity 永远停在 stranger——识别器真实语料 0/466，整条阶梯吊在它上面。
# 开＝用记忆深度给 familiarity 一个下限（≥50 熟悉、≥10 认得），不用字面标签。
RELATIONSHIP_FAMILIARITY_FROM_MEMORY_ENABLED = os.getenv(
    "RELATIONSHIP_FAMILIARITY_FROM_MEMORY_ENABLED", "false"
).lower() in ("true", "1", "yes")

# ── 关系按"内容"计分，且会降（plans/1017-02 §8.38，默认关）──────────────────
# 用户原话：「关系不是单纯聊得多久就能定下来，同事天天说也不见得关系有多好，以及关系是会变化的」。
# ⇒ 次数不计分；只认内容（私事分享 +／关心 +／明确拉远 −），带 30 天半衰期。
# 打分放在**会话结束时**那一次 LLM 调用里，不进语音热路径。
RELATIONSHIP_CLOSENESS_FROM_CONTENT_ENABLED = os.getenv(
    "RELATIONSHIP_CLOSENESS_FROM_CONTENT_ENABLED", "true"
).lower() in ("true", "1", "yes")

# ── 让她自己说过的话沉淀成事实（plans/1017-02 §8.41，默认关）──────────────
# 为什么：她讲自己的事**受材料限制**（底子只有 5 条），不是契约能推出来的。
# 会话末把她自己说过的**长期成立**的偏好/态度抽成 candidate 事实（不是 active），
# 跨会话重复再由既有仲裁升格；三道闸（原话对得上／不涉及用户／不是一次性）在代码里做。
PERSONA_SELF_FACT_ENABLED = os.getenv(
    "PERSONA_SELF_FACT_ENABLED", "true"
).lower() in ("true", "1", "yes")

# 闲聊盘每轮带的"她自己的料"条数（2026-10-22，plans/1022-01 点5）。
# 实测：6 条对零词面重叠的问题固定不变（问"你最近忙什么"和"天气"同一把），
# 因为 rank_facts 靠词面，零重叠就退回同一顺序。给到 10 条让模型自己挑
# （2026-10-22 实测：压掉每行元数据后缀后，10 条真实料 1419 字符，塞得进 1600）；
# 默认关＝仍 6 条、行为不变。只在闲聊盘生效，练习/素材盘不动。
PERSONA_SELF_FACT_SAMPLE_ENABLED = os.getenv(
    "PERSONA_SELF_FACT_SAMPLE_ENABLED", "false"
).lower() in ("true", "1", "yes")
PERSONA_SELF_FACT_SAMPLE_SIZE = int(
    os.getenv("PERSONA_SELF_FACT_SAMPLE_SIZE", "10")
)
# 未闭环的事（plans/1017-02 §8.56）：会话末抽"他提过、还没有下文的事"，
# 下一盘**话题真的碰到**时她才自然接一句"后来怎样"。默认关＝零行为变化。
# 代价：会话末多 1 次模型调用（不在语音路径上）；检索是纯字符串匹配，零调用。
# §8.76：她问过、他没答的问题——存下来，但只在他把话题带回来时提一句。
# 默认 false：关 = 这条链不存在（零读写、零调用）。
# §8.79：她自己的日子（世界线）。一周生成一次、7 天过期、一盘最多说一条。
# §8.90：供给度量（给了她什么、用没用上）。零模型调用；命中在**会话末**判，不在轮次路径上。
COMPANION_SUPPLY_LOG_ENABLED = os.getenv(
    "COMPANION_SUPPLY_LOG_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")

COMPANION_LIFE_LINE_ENABLED = os.getenv(
    "COMPANION_LIFE_LINE_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")

# 她自己读东西（真实外部输入）：默认关，由运营/产品决定何时开。
# 开了之后后台 sweeper 每 30 分钟给最多 3 个活跃用户各读一条（不进对话路径）。
ALICE_READING_ENABLED = os.getenv(
    "ALICE_READING_ENABLED", "false"
).strip().lower() in ("1", "true", "yes")
# 通用爱丽丝外部经历与基础人格账号（固定哨兵 UUID）：
# 外部冲浪/B站观看点赞投币收藏关注评论与阅读沉淀统一归入该共享身份；
# 具体用户与她的对话经历与演化仍归用户各自 user_id，同 facet 下覆盖共享基底。
ALICE_SHARED_USER_ID = os.getenv(
    "ALICE_SHARED_USER_ID", "00000000-0000-0000-0000-00000000a11c"
).strip()
# 与通用测试人格成对绑定的固定持久测试用户 UUID：
# 周期性实测（如每隔 1-2 天累积外部冲浪经历后验收）统一使用该账号对话并保留记忆，
# 严禁测完调用 delete_user_data 清空，以真实累积熟悉度、未闭环话题、印象与用户侧人格分化。
ALICE_TEST_USER_ID = os.getenv(
    "ALICE_TEST_USER_ID", "00000000-0000-0000-0000-00000000a11d"
).strip()
ALICE_EXTERNAL_SHARED_ENABLED = os.getenv(
    "ALICE_EXTERNAL_SHARED_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")
# 白名单（逗号分隔 user_id）：非空时**只**给名单里的人读，不碰真实用户。
# 测试期用：ALICE_READING_ENABLED=true + 这里填测试号。
ALICE_READING_WHITELIST = os.getenv("ALICE_READING_WHITELIST", "").strip()
# 两次阅读之间至少隔多久（小时，可小数）。默认 20 小时；测试期调小，让她白天多逛几次。
try:
    ALICE_READING_MIN_GAP_HOURS = float(os.getenv("ALICE_READING_MIN_GAP_HOURS", "20") or 20)
except ValueError:
    ALICE_READING_MIN_GAP_HOURS = 20.0
# 后台 sweeper 多久扫一次（分钟）。
try:
    ALICE_READING_SWEEP_MINUTES = float(os.getenv("ALICE_READING_SWEEP_MINUTES", "30") or 30)
except ValueError:
    ALICE_READING_SWEEP_MINUTES = 30.0

# ── Alice 海外社交平台 X (Twitter) 渠道与养号保护开关 ──────────────────────
# 默认开启轻量浏览底座（仅当 .runtime/x_cookies.json 存在时生效，无凭据自动跳过）；
# 申诉/养号期默认关闭写互动（点赞/书签/关注）与公开回复，待账号解封稳定后再按需打开。
ALICE_X_ENABLED = os.getenv("ALICE_X_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
try:
    ALICE_X_MIN_GAP_HOURS = float(os.getenv("ALICE_X_MIN_GAP_HOURS", "8") or 8)
except ValueError:
    ALICE_X_MIN_GAP_HOURS = 8.0
ALICE_X_INTERACT_ENABLED = os.getenv("ALICE_X_INTERACT_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
ALICE_X_SPEAK_ENABLED = os.getenv("ALICE_X_SPEAK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
try:
    ALICE_X_INTERACT_DAILY_MAX = int(os.getenv("ALICE_X_INTERACT_DAILY_MAX", "8") or 8)
except ValueError:
    ALICE_X_INTERACT_DAILY_MAX = 8
try:
    ALICE_X_INTERACT_MIN_GAP_MIN = float(os.getenv("ALICE_X_INTERACT_MIN_GAP_MIN", "20") or 20)
except ValueError:
    ALICE_X_INTERACT_MIN_GAP_MIN = 20.0
try:
    ALICE_X_SPEAK_DAILY_MAX = int(os.getenv("ALICE_X_SPEAK_DAILY_MAX", "1") or 1)
except ValueError:
    ALICE_X_SPEAK_DAILY_MAX = 1

# ── Alice 网文阅读（起点中文网 + 番茄小说）登录态书架与追更开关 ──────────────────
# 仅当 .runtime/novel_cookies.json 存在时生效（无凭据自动跳过）；
# 默认开启书架读取、试读与推荐票点赞互动（严禁触发任何需充值点券的消费操作）。
ALICE_NOVEL_ENABLED = os.getenv("ALICE_NOVEL_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
try:
    ALICE_NOVEL_MIN_GAP_HOURS = float(os.getenv("ALICE_NOVEL_MIN_GAP_HOURS", "4") or 4)
except ValueError:
    ALICE_NOVEL_MIN_GAP_HOURS = 4.0
ALICE_NOVEL_INTERACT_ENABLED = os.getenv("ALICE_NOVEL_INTERACT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
try:
    ALICE_NOVEL_DAILY_MAX = int(os.getenv("ALICE_NOVEL_DAILY_MAX", "4") or 4)
except ValueError:
    ALICE_NOVEL_DAILY_MAX = 4
ALICE_NOVEL_COOKIE_FILE = os.getenv("ALICE_NOVEL_COOKIE_FILE", "").strip()

# ── Alice「挂着听」B 站直播：只取声音（ffmpeg -vn）+ 必剪免费 ASR + 实时弹幕，零视觉 token ──
# ALICE_BILI_LIVE_ROOMS：置顶直播间（逗号分隔），开播时优先进候选；另会合并 B 站账号关注列表里在播的房间。
ALICE_BILI_LIVE_ENABLED = os.getenv("ALICE_BILI_LIVE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
ALICE_BILI_LIVE_ROOMS = os.getenv("ALICE_BILI_LIVE_ROOMS", "1883358196").strip()
try:
    ALICE_BILI_LIVE_MIN_GAP_HOURS = float(os.getenv("ALICE_BILI_LIVE_MIN_GAP_HOURS", "6") or 6)
except ValueError:
    ALICE_BILI_LIVE_MIN_GAP_HOURS = 6.0
try:
    ALICE_BILI_LIVE_LISTEN_SECONDS = int(os.getenv("ALICE_BILI_LIVE_LISTEN_SECONDS", "300") or 300)
except ValueError:
    ALICE_BILI_LIVE_LISTEN_SECONDS = 300

# 她此刻在过什么日子（世界线的"状态层"）：按真实时间给"她现在在做什么"，
# 让 Alice 不是"待命状态"。只闲聊盘注入，不新增事实来源。
ALICE_NOW_STATE_ENABLED = os.getenv(
    "ALICE_NOW_STATE_ENABLED", "true"
).strip().lower() in ("1", "true", "yes")

COMPANION_OPEN_QUESTIONS_ENABLED = os.getenv(
    "COMPANION_OPEN_QUESTIONS_ENABLED", "false"
).lower() in ("true", "1", "yes")
OPEN_THREADS_ENABLED = os.getenv(
    "OPEN_THREADS_ENABLED", "true"
).lower() in ("true", "1", "yes")
# 她的留言箱（plans/1017-02 §8.60）：会话末把"明天/后天到期的事"写成一条留言，
# 用户下次打开就能看到（不需要系统推送）。默认关＝零行为变化。
COMPANION_OUTBOX_ENABLED = os.getenv(
    "COMPANION_OUTBOX_ENABLED", "false"
).lower() in ("true", "1", "yes")
# 入口开场白（plans/1017-02 §8.62）：用户 2026-10-18 判定"入口不说话"——
# 她真有事就留消息（outbox），没事就不该在门口喊。默认关。
# 打开它 = 恢复"按间隔分级"的旧行为（<10min off / 10-120min silent / ≥120min voice）。
CHAT_OPENING_ENABLED = os.getenv(
    "CHAT_OPENING_ENABLED", "false"
).lower() in ("true", "1", "yes")
# Voice hot-path budgets are intentionally shorter than batch/chat timeouts.
# A failed turn can fall back, but an apparently frozen voice conversation is
# itself a product failure.  These values bound waiting; device p95 remains the
# acceptance evidence and may justify later tuning.
TUTOR_RESPONSE_TIMEOUT_SECONDS = max(4.0, min(14.0, float(os.getenv(
    "TUTOR_RESPONSE_TIMEOUT_SECONDS", "8"
))))

# AIGC conversation director. A dedicated stronger model can be configured;
# defaults reuse the current OpenAI-compatible tutor endpoint.
CONVERSATION_DIRECTOR_ENABLED = os.getenv(
    "CONVERSATION_DIRECTOR_ENABLED", "true"
).lower() in ("true", "1", "yes")
CONVERSATION_DIRECTOR_MODEL = os.getenv(
    "CONVERSATION_DIRECTOR_MODEL", TUTOR_RESPONSE_MODEL
)
CONVERSATION_DIRECTOR_BASE_URL = os.getenv(
    "CONVERSATION_DIRECTOR_BASE_URL", TUTOR_RESPONSE_BASE_URL
)
CONVERSATION_DIRECTOR_API_KEY = _api_key_for_endpoint(
    CONVERSATION_DIRECTOR_BASE_URL,
    os.getenv("CONVERSATION_DIRECTOR_API_KEY", "") or TUTOR_RESPONSE_API_KEY,
)
CONVERSATION_DIRECTOR_MAX_TOKENS = int(os.getenv(
    "CONVERSATION_DIRECTOR_MAX_TOKENS", "420"
))
CONVERSATION_DIRECTOR_FOREGROUND_TIMEOUT_SECONDS = max(
    0.2, min(3.0, float(os.getenv(
        "CONVERSATION_DIRECTOR_FOREGROUND_TIMEOUT_SECONDS", "1.0"
    )))
)
# 练习盘专用的前台等待预算（2026-10-10）。
#
# 为什么必须单独给一个更大的值：练习盘的 `argument_probe`（五维逻辑探查）
# **只能由导演在同一次 JSON 调用里产出**（§4.2.3：不新增模型调用）。而导演实测
# 单次耗时 **2.9–11.6 秒**（P50≈4.9s；漏斗钩子三批 30+ 次实测，尾部见 7.8s/11.6s）
# ——远超前台的 `CONVERSATION_DIRECTOR_FOREGROUND_TIMEOUT_SECONDS`
# （本机 .env 压到 0.5s）。结果就是练习盘几乎每一轮都超时回退，probe 一条
# 都产不出来，五维链整条失效。
#
# 练习盘不是实时对话轮（用户按"提交"后才要反馈），可以承受更长的思考时间。
# 10.0s：初版 6.0s 按 8011 钩子实测仍掐掉约 20-30% 的轮（真实耗时 7.8s/11.6s
# 的轮 HTTP 面显示 fallback/500），故按 P95 上调；**超时即回退**——退化行为与
# 改动前完全一致，不会把某一轮卡死。其余盘型仍用上面那个 0.2–3.0s 的预算。
CONVERSATION_DIRECTOR_EXPRESSION_TIMEOUT_SECONDS = max(
    0.5, min(30.0, float(os.getenv(
        "CONVERSATION_DIRECTOR_EXPRESSION_TIMEOUT_SECONDS", "10.0"
    )))
)
CONVERSATION_DIRECTOR_SEARCH_TIMEOUT_SECONDS = max(
    4.0, min(20.0, float(os.getenv(
        "CONVERSATION_DIRECTOR_SEARCH_TIMEOUT_SECONDS", "7"
    )))
)
CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED = os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", "false"
).lower() in ("true", "1", "yes")
CONVERSATION_DIRECTOR_WEB_SEARCH_PROVIDER = os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_PROVIDER", "bing_rss"
).strip().lower()
CONVERSATION_DIRECTOR_WEB_SEARCH_MODE = os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_MODE", "adaptive"
).strip().lower()
if CONVERSATION_DIRECTOR_WEB_SEARCH_MODE not in {"adaptive", "always"}:
    CONVERSATION_DIRECTOR_WEB_SEARCH_MODE = "adaptive"
CONVERSATION_DIRECTOR_WEB_SEARCH_API_KEY = os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_API_KEY", ""
)
CONVERSATION_DIRECTOR_WEB_SEARCH_TIMEOUT = max(2, min(15, int(os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_TIMEOUT", "5"
))))
CONVERSATION_DIRECTOR_WEB_SEARCH_MAX_RESULTS = max(1, min(8, int(os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_MAX_RESULTS", "5"
))))
CONVERSATION_DIRECTOR_WEB_SEARCH_CACHE_TTL = max(60, min(86400, int(os.getenv(
    "CONVERSATION_DIRECTOR_WEB_SEARCH_CACHE_TTL", "1800"
))))

# ── 附属链路共用的 OpenCode 网关（2026-09-18 起）────────────────────────────
# 审计 / 记忆整理 / 自修复从方舟切过来。选它是因为它不参与实时语音热路径，
# 且用户已订阅（$10/月 Go 档）。Chat Completions 与 OpenAI 完全同形，
# llm_client 里现成的 AsyncOpenAI 直接可用；唯一的额外要求是请求头
#   `x-opencode-session: ses_<uuid>`
# （缺 → 400 MissingSessionID，全部模型挂）—— llm_client 命中 opencode.ai 时自动附加。
# Asynchronous memory curation. This model never participates in the hot voice
# response path and never writes proficiency/mastery conclusions directly. It
# only proposes evidence-bearing cross-session memory candidates.
MEMORY_CURATOR_ENABLED = os.getenv(
    "MEMORY_CURATOR_ENABLED", "true"
).lower() in ("true", "1", "yes")

# ── 归档摘要写成"内容"而不是"元数据模板"（plans/1017-02 §8.24，默认关）─────
# 实测：237 条 session_archives 里「具体事件：」后**中位 8 字**、**109 条不足 5 字**，
# 其余是 `CEFR / Topics: sports, work, music, food, movies / Errors: …` 这种模板。
# 同一段对话用真摘要跑出来 **180 字**（人名、烤饼干误把盐当糖、没聊完的事）。
# ⇒ 记忆库里没有可被回忆的内容，这才是"Alice 记不住"的根子（检索/rerank/模型都不是）。
# 开：归档摘要走 LLM 内容摘要；失败自动回落模板（关＝逐字原行为）。
ARCHIVE_SUMMARY_LLM_ENABLED = os.getenv(
    "ARCHIVE_SUMMARY_LLM_ENABLED", "false"
).lower() in ("true", "1", "yes")
ARCHIVE_SUMMARY_MAX_MESSAGES = int(os.getenv("ARCHIVE_SUMMARY_MAX_MESSAGES", "40"))
# 1017-02 §8.92: 归档话题索引 + 证据包（默认关；关 = 旧 recent-3 行为逐字不变）
RECALL_EVIDENCE_PACKET_ENABLED = os.getenv(
    "RECALL_EVIDENCE_PACKET_ENABLED", "false"
).lower() in ("true", "1", "yes")
# 10-21 transcript 失败样本：闲聊盘“被质疑先接住 / 让决定就给决定 / 不编生活细节”
CHAT_TAPE_HONESTY_CONTRACT_ENABLED = os.getenv(
    "CHAT_TAPE_HONESTY_CONTRACT_ENABLED", "false"
).lower() in ("true", "1", "yes")

# ⑧ 不编生活细节单独成条（2026-10-22）：它是诚实底线，默认开；
# ⑥⑦⑨⑩（被质疑先接住/给决定/不安慰/先给判断）是语气硬门，保持默认关。
CHAT_TAPE_NO_FABRICATION_ENABLED = os.getenv(
    "CHAT_TAPE_NO_FABRICATION_ENABLED", "true"
).lower() in ("true", "1", "yes")

# 闲聊盘不注入 [Policy]/[Learning Move]/[Learning State]/[Learner State] 四段
# 逐轮 JSON（2026-10-24，用户判据："以爱丽丝独立人格为出发点，搁置所有违背这条
# 目标的指令"）。
# 实测依据：同一句「今天加班到九点，累死了。」跑两次独立实验（各 3 次采样），
#   纯人格 prompt     → 讲自己的事 4/6（"我一般这种时候就瘫沙发刷视频"）
#   人格 + 这 4 段 policy → 讲自己的事 0/6，全部退化成"是活儿多还是临时加的？"
# 且逐条去掉任何一条都救不回来（C 组 6 条全 0/2）⇒ 不是某一条的问题，是这套
# "回复目标/教学动作/继续方式/必须/避免/输出顺序"的结构本身在把她变成服务者：
# 六项全在说"对用户做什么"，没有一项是"你是谁、你怎么想"。
# 置 false 可一键回到旧行为（闲聊盘重新注入这四段）。
# ⚠ 2026-10-24 实测：单独搁置这四段**无效**（讲自己 5/12→4/12，噪声内）。
# 原因不是判据错，而是**只搁置了一部分**——产品链路还有 20 段注入
# （[Conversation Contract] 1386 字符 / [Stance - content] 625 /
# [Disposition] 414 / [Optional Teaching Evidence] 552 …）同样在说
# "你是来教学的"。部分移除不足以改变形态 ⇒ 默认保持 true（旧行为），
# 等全部段落按"独立人格"判据审计后再统一决定。
CHAT_TAPE_TURN_POLICY_ENABLED = os.getenv(
    "CHAT_TAPE_TURN_POLICY_ENABLED", "true"
).lower() in ("true", "1", "yes")

# 闲聊盘「人格先行」（2026-10-24，用户拍板"以爱丽丝独立人格为出发点，搁置违背它的指令"）。
# 开 = 闲聊盘只保留人格、记忆、边界与元叙述/误听等硬红线段，搁置服务型、教学型、
# 控制型段（[Order]/[Pragmatics]/[Frame]/[Teaching]/[Reply Contract]/[Conversation Rhythm]/
# [Turn Rhythm] 的问句预算、[Anti-Parroting] 的三行机器输出、[Policy]/[Learning Move]/
# [Learning State]/[Learner State]/[Teaching Method]/[Assessment SOP]/[Constraints]/
# [Latest Turn Contract]、合同里的 1-3 句与问句配额）。练习盘/素材盘不受影响。
# 默认 true：2026-10-24 真机 A/B 通过——ON 相比 OFF 教学插入归零、超长回复减少、
# 时间戳漏出修复、危中文更少；问句虽多但为自然好奇（从自身人格出发），符合产品目标。
# 置 false 即回到改前（保留服务/教学/控制段注入）。
CHAT_TAPE_PERSONA_FIRST_ENABLED = os.getenv(
    "CHAT_TAPE_PERSONA_FIRST_ENABLED", "true"
).lower() in ("true", "1", "yes")
# 2026-10-21（用户拍板）：闲聊盘不再调用对话导演。
# 普通闲聊轮的前台预算只有 0.5s，而导演计划本身要 1~3s ⇒ 大面积走
# director_pending_fallback（content_need 就是用户原话的填充计划）：付一次调用
# 和 0.5s 等待，却拿不到真计划。保留两条窄路：需要联网的轮次、正在建立/切换场景
# 的轮次。置 true 可一键回到旧行为（闲聊盘每轮都尝试导演）。
# 2026-10-25 B选项：默认不强制闲聊盘回英文（英语环境仍可设 True 回切）。
CHAT_TAPE_ENGLISH_FORCED_ENABLED = os.getenv(
    "CHAT_TAPE_ENGLISH_FORCED_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
CHAT_TAPE_DIRECTOR_ENABLED = os.getenv(
    "CHAT_TAPE_DIRECTOR_ENABLED", "false"
).lower() in ("true", "1", "yes")
# 归档用谁的口吻写（plans/1017-02 §8.28）：
#   archive（默认）＝第三人称档案体「用户烤饼干…」——好检索，但**好念**，
#                     Alice 拿到就会照条交代（实测「我们上次聊了什么」→ 四条全列）。
#   alice        ＝爱丽丝自己的口吻「记得她烤饼干把盐当糖，猫闻一下就走了」——
#                     事实不变，但语气是她自己的，念出来就是回忆，不是汇报。
# 材料决定语气：改材料比加规矩稳（加 prompt 规矩今天已三次无效，§8.19~§8.21）。
ARCHIVE_SUMMARY_VOICE = os.getenv("ARCHIVE_SUMMARY_VOICE", "archive").strip().lower()
MEMORY_CURATOR_MODEL = os.getenv(
    "MEMORY_CURATOR_MODEL", "deepseek-v4.1-flash"
)
MEMORY_CURATOR_BASE_URL = os.getenv(
    "MEMORY_CURATOR_BASE_URL", OPENCODE_BASE_URL
)
# ⚠ 这里**不能**留空：memory_curator.py 对空 key 是整链路 early-return `disabled`。
# 也**不能**回落成方舟 key —— 会拿方舟凭证去打 OpenCode。所以回落顺序是 OC → DashScope。
MEMORY_CURATOR_API_KEY = os.getenv(
    "MEMORY_CURATOR_API_KEY", ""
) or OPENCODE_API_KEY or DASHSCOPE_API_KEY
# ⚠ 上限原先钳死在 900。memory_curator.py 的 _parse_json 虽有 _recover_truncated_object
# 做截断抢救（保住靠前的完整顶层成员，candidates 在最前面所以一般能活），但**尾部的候选、
# episode_summary / contradictions / handoff_patch 会整块丢**，且丢得无声无息。
# 放宽到 2048 只是"允许模型把话说完"，不改模型、不改路由、不碰任何链路的选型。
# 2026-09-18 实测：qwen-turbo 在这条链路上 2.6s、输出约 850~1150 字符，1400 留了余量。
MEMORY_CURATOR_MAX_TOKENS = max(180, min(2048, int(os.getenv(
    "MEMORY_CURATOR_MAX_TOKENS", "1400"
))))
# 2026-09-18 由 10 → 15 → 20：切到 OpenCode deepseek-v4.1-flash 后，同形状 payload
# 实测 4.4s（thinking 关闭），但真实会话的消息条数会更多，留 4× 余量。
# 一超时就降级成 deterministic_candidates() 规则兜底（漏语义线索）。
# ⚠ 这个值不能无脑加大：闸位只有 2 个（llm_client._sem），后台任务会占位。
#    根治办法是让后台走独立闸（已实现：chat_completion(gate="background")）。
MEMORY_CURATOR_TIMEOUT = max(3, min(30, int(os.getenv(
    "MEMORY_CURATOR_TIMEOUT", "20"
))))
MEMORY_CURATOR_TURN_INTERVAL = max(8, min(80, int(os.getenv(
    "MEMORY_CURATOR_TURN_INTERVAL", "20"
))))

# Provider-neutral context engineering. Raw events remain available for the
# session archive; only the model-facing working set is compacted.
CONTEXT_COMPACTION_ENABLED = os.getenv(
    "CONTEXT_COMPACTION_ENABLED", "true"
).lower() in ("true", "1", "yes")
CONTEXT_ACTIVE_HISTORY_TOKENS = max(800, min(12000, int(os.getenv(
    "CONTEXT_ACTIVE_HISTORY_TOKENS", "3200"
))))
CONTEXT_RETAIN_RECENT_MESSAGES = max(4, min(20, int(os.getenv(
    "CONTEXT_RETAIN_RECENT_MESSAGES", "8"
))))
CONTEXT_COMPACTION_MIN_MESSAGES = max(
    CONTEXT_RETAIN_RECENT_MESSAGES + 2,
    min(80, int(os.getenv("CONTEXT_COMPACTION_MIN_MESSAGES", "24"))),
)

# ── 统一学习者画像（plans/0923-04 阶段 1）──────────────────────────────────
# 默认**全关**：新读写路径完全旁路，关闭时逐轮 prompt 与基线逐字一致，
# 因此上线顺序是「先合代码、再按用户/键逐个开」。
# READ_ENABLED 控制**读**（逐轮注入统一画像摘编）；PROJECT_ENABLED 控制**写后投影**
# （把既有五条管道的产物幂等投影进 `learner_profile_facts`）。两者独立，
# 可以先只开投影攒数据、确认无误后再开读。
PROFILE_MEMORY_READ_ENABLED = os.getenv(
    "PROFILE_MEMORY_READ_ENABLED", "true"
).lower() in ("true", "1", "yes")
PROFILE_MEMORY_PROJECT_ENABLED = os.getenv(
    "PROFILE_MEMORY_PROJECT_ENABLED", "true"
).lower() in ("true", "1", "yes")
# ── 练习盘五维「自己的」开关（plans/0919-02 §4.7 ／ 0923-06 §4.2.4）──────────
# 为什么需要单独一个：上面两个开关是**全局**的，同时管着其余盘的统一画像读写。
# 想「只让练习盘用上五维历史、别的一盘都不碰」就跟不了它们 —— 跟了就等于
# 连闲聊盘一起改。本开关只作用于练习盘这一条链：
#   * 读：`_load_expression_argument_weak` → `ctx.expression_profile_weak`
#         → `choose_focus` 的画像排序；
#   * 写：会话末的轮外五维冷评估 → `learner_profile_facts`
#         （`fact_key=error_pattern` + `fact_subkey=argument_<dim>`）。
# 默认 **false** ⇒ 与「这条链不存在」同义（读恒为空列表；冷评估在发模型调用
# **之前**就跳过，不白跑一次背景请求）。打开它**不会**打开上面两个全局开关。
EXPRESSION_ARGUMENT_MEMORY_ENABLED = os.getenv(
    "EXPRESSION_ARGUMENT_MEMORY_ENABLED", "false"
).lower() in ("true", "1", "yes")
# ── 跨盘回忆的「宽语义池」（plans/1017-02 §5 M2）──────────────────────────
# 问题：`HybridHistoryRetriever` 的稠密阶段原本只 embed `sparse[:28] ＋ 最近 10 条`
# 再截到 28 ⇒ **字面不重合、又不是最近 10 条**的改写行**永远进不了语义打分**，
# 无论它语义多像。这就是「换个说法就想不起来」的机制级成因。
# 打开后：语义候选池改成「稀疏前列 ∪ 最近 ∪ 其余字面候选」，上限抬到
# `candidate_limit`（80）。**打分与 RRF 融合逻辑一字未改**——只是让打分器
# 看得见更多行；候选向量按文本 sha256 缓存（`_embedding_cache`），重复回忆的
# 增量成本趋近于 0。
# 默认 **false** ⇒ 与改前同义。⚠ 本开关只作用于**显式回忆**（路由点头才走这条
# 路），不在每轮的语音热路径上。
HISTORY_RECALL_WIDE_SEMANTIC_ENABLED = os.getenv(
    "HISTORY_RECALL_WIDE_SEMANTIC_ENABLED", "false"
).lower() in ("true", "1", "yes")
# ── 被 embed 的文本要不要「去包装」（plans/1017-02 §8.5 / M2b）──────────────
# 入库文本带一套结构性字段标签（`场景=…；角色=…；当前步骤=…` ＋ `用户:`/`AI:`），
# 每行都有、与内容无关 ⇒ 进向量会把所有旧行的向量往一处拉、稀释判别力。
# 台架实测：同一对语义相似度 **0.553 → 0.464**（加包装后），
# 正好掉到稠密独中的 **0.48** 门以下 ⇒ 本该召回的那条被丢掉。
# 打开后**只改"喂给 embedding 的那份文本"**（`embedding_surface()`），
# `searchable` 本身一字不动 ⇒ 稀疏打分、注入 prompt、导出全不受影响。
# 默认 **false** ⇒ 与改前同义。
HISTORY_RECALL_CLEAN_EMBED_TEXT_ENABLED = os.getenv(
    "HISTORY_RECALL_CLEAN_EMBED_TEXT_ENABLED", "false"
).lower() in ("true", "1", "yes")
# ── 回忆路由：把「状态变化问句」当作**预算开关**（plans/1017-02 §5 M0 / §8.9）──
# 现状：`CROSS_SESSION_REFERENCE` 是纯词表，只认「上次／之前／还记得」这类显式标记
# （且这处正则已经因为「没触发」补过两轮词，见 `memory_orchestrator.py` 第 55、67 行注释）；
# 自然追问「我那个房租的事后来怎么样了」**中英各 0/8 触发** ⇒ 检索再好也轮不到出场。
# 台架 `bugreports/eval_definite_reference_gate_1017.py`（16 正例／50 负例）：
#   * 只靠「定指成分」      召回 12/16，误报 **21/50** ⇒ 废
#   * 只靠「状态变化问句」  召回 **15/16**，误报 **1/50** ⇒ 唯一可用
#   * 两者相与             召回 12/16，误报 1/50    ⇒ 加定指反而砍召回（中文常省略）
# ⛔ 它仍然是**词表**（`HANDOFF.md` 硬规则 #2：词表必然漏）⇒ 只当**预算开关**：
#    命中只表示「可以花一次有界检索」，**是否注入由检索自己的分数决定**。
# 2026-10-23 冻结集 A/B（`bugreports/data/ab_off_1023.json` / `ab_on_1023.json`，
# 30 回忆问法 + 20 对照，单跑不并发；ON 复跑一次确认可重复）：
#   route 0.400→0.600（两跑同值）、gold 0.600→0.70/0.77、deny 0.300→0.13/0.17、wrong 0→0/0.03；
#   `specific_vague` 组 gold 0.50→0.90/1.00、deny 0.40→0.00（"我那个房租的事后来怎么样了"）；
#   `changed_fact` 组仍 0.20/0.80（"最后怎么定的"这类不在本词表）。
#   代价：真实 506 条消息新增触发 38 条 = 7.5%，每条多一次 embedding+SQL ≈ **+111ms**，无生成调用；
#   延迟归因：route 未变的 43 轮中位 +599ms（环境噪声），新进回忆的 7 轮中位 +436ms ⇒ 开关本身不构成可测增量。
#   已知误报「你还好吗」+1（与「你还打算辞职吗」词形同构），实测回复未被旧事污染。
# 默认 **true**（2026-10-23 起）；置 false 即回到改前逐字行为（`probe_state_change_recall_1017.py` 可对拍）。
HISTORY_RECALL_STATE_CHANGE_TRIGGER_ENABLED = os.getenv(
    "HISTORY_RECALL_STATE_CHANGE_TRIGGER_ENABLED", "true"
).lower() in ("true", "1", "yes")
# ── 回忆注入闸门：**环境边际 ＋ 词面锚点**（plans/1017-02 §8.11 / M4）──────────
# 背景（§8.10 实测）：绝对余弦判不出"有没有这件事"——带入库包装时正例 peak 0.590／
# 负例 0.524，间隔只有 0.066；`0.48` 那道「稠密独中」闸门下 **80% 的负例照样过门**
# ⇒ 它事实上常开，没有任何阈值能同时拿到召回与不误注入。
# 改判据（§8.11 实测，83 行 ≈ 生产 candidate_limit 80）：
#   `环境边际 = top1 dense − median(池内全部 dense)`，再要求被选中的行**有词面锚点**
#   （`lexical_overlap > 0`）⇒ 留出集 **8/8 召回、2/50 误报（4%）**；
#   不加锚点时是 8/8 但 **12/50**；原始余弦是 7/8 但 **23/50**。
# ⛔ **代价照实记**：词面锚点会系统性砍掉"一个字都不重合"的纯改写
#   （A 组因此丢 2/8：「我腿伤好点了吗」vs 旧行「打篮球把脚踝崴了」）。
#   ⚠ 这也**部分抵消 M2 的价值**——M2 把"字面零重合的行"送进语义阶段，
#   本闸又把它们丢掉。两者同时开时，M2 的增量只剩"有词面重合但排在 28 名之外"的行。
# 默认 **false** ⇒ `search()` 与改前**逐字相同**（`probe_margin_gate_1017.py` 逐字比对）。
HISTORY_RECALL_MARGIN_GATE_ENABLED = os.getenv(
    "HISTORY_RECALL_MARGIN_GATE_ENABLED", "false"
).lower() in ("true", "1", "yes")
# 阈值是**量出来的近似值**，不是普适常数，而且**跟入库文本形状绑定**（83 行夹具实测）：
#   * 带入库包装（现状）：0.10 ⇒ 召回 6/8、误报 2/50（A/B 两组都是）
#   * 开 **M2b 去包装**：0.10 ⇒ 误报升到 5/50；**0.20** ⇒ 召回 8/8、误报 3/50（B 组）
#     ⇒ 两个开关**必须一起调**；只开 M2b 不改阈值会让注入率翻倍。
# 换 embedder / 换音色无关，但换模型或改入库形状后**必须重测**。
HISTORY_RECALL_MARGIN_MIN = float(os.getenv("HISTORY_RECALL_MARGIN_MIN", "0.10"))

# ── 自回声过滤（plans/1017-02 §8.23）─────────────────────────────────────
# 实测：真实用户 11 条回忆轮，检索第一名 **11/11 都是用户这句话本身**
# （`echo_ratio = 1.00`）——因为最近一条记忆就是**这一轮刚写下的摘要**。
# 后果：问"我们之前聊过什么"，系统把"你这句话"当成"旧记忆"还给模型，
# 任何相关性模型（bge / 阿里的 qwen rerank）都会给它满分 ⇒ §8.16 的“分数全饱和”。
# 本开关在候选阶段丢掉「用户这句话的字符二元组 ≥95% 都出现在候选里」的行。
# 判据是**整句重合**，不是话题相近：旧记录里提一句"脚踝"不会触发。
# 默认关；关＝逐字零行为变化。
HISTORY_RECALL_DROP_SELF_ECHO_ENABLED = os.getenv(
    "HISTORY_RECALL_DROP_SELF_ECHO_ENABLED", "false"
).lower() in ("true", "1", "yes")
# ── sparse 绝对证据门槛（2026-10-22，点 1）────────────────────────────────
# 为什么需要：`rank_history_sparse` 的分数按**池内 max_bm25** 归一化，弱池子的
# 第一名也会被放大到 0.62 左右。真实池实测：10 句“回忆形状、但记忆里没这事”的
# 问句，8 句返回非空——命中的是“我上/次说/说的”这类常见二元组，不是内容词。
# 0＝关闭（逐字零变化）。打开后只保留 `sparse_raw`（原始 BM25，已含 IDF）达到
# 门槛的行；整池不过线就返回空。
HISTORY_RECALL_SPARSE_MIN_RAW = float(
    os.getenv("HISTORY_RECALL_SPARSE_MIN_RAW", "0")
)
HISTORY_RECALL_SELF_ECHO_RATIO = float(
    os.getenv("HISTORY_RECALL_SELF_ECHO_RATIO", "0.95")
)

# ── rerank 闸门（2026-10-22，plans/1022-01 点 1 第 3 步）────────────────────
# 词面/稠密都判不出"这条旧记录里到底有没有用户问的那件事"：80 条回忆形状、
# 记忆里确实没有的负例（49 消息块 + 31 归档），rerank top1 最高 0.702，
# 0/80 过 0.74；11 条真实回忆轮 9 条 ≥0.784。闸门只挡"低到不可能是那件事"
# 的候选，不重新排序、不改生成。⛔ 默认关；关＝逐字零行为变化。
# rerank 失败/超时 = 放行（fail-open），不因为外部服务挂掉而吞掉回忆。
HISTORY_RECALL_RERANK_GATE_ENABLED = os.getenv(
    "HISTORY_RECALL_RERANK_GATE_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
HISTORY_RECALL_RERANK_MIN_SCORE = float(
    os.getenv("HISTORY_RECALL_RERANK_MIN_SCORE", "0.74")
)
HISTORY_RECALL_RERANK_TIMEOUT = float(
    os.getenv("HISTORY_RECALL_RERANK_TIMEOUT", "1.0")
)

# ── 最近归档兜底（plans/1017-02 §8.25，默认关）──────────────────────────────
# 实测：真实回忆轮命中 **0/11**。原因之一：像「我们上次聊了什么」「Do you remember
# what we talked about」这种问句**本身没有内容词**，词面检索必然什么都匹配不到。
# 这种问句要的答案是"最近那几场对话"，不是"含某个词的那一段"。
# 开：词面一条都没命中时，直接给最近 N 条 session_archive（按 created_at 倒序）。
# 默认关；关＝逐字零行为变化。
HISTORY_RECALL_RECENT_ARCHIVE_FALLBACK_ENABLED = os.getenv(
    "HISTORY_RECALL_RECENT_ARCHIVE_FALLBACK_ENABLED", "false"
).lower() in ("true", "1", "yes")
HISTORY_RECALL_RECENT_ARCHIVE_COUNT = int(
    os.getenv("HISTORY_RECALL_RECENT_ARCHIVE_COUNT", "2")
)
# ── 回复末尾追加「先回答」一行（plans/1017-02 §8.18）──────────────────────
# 真实 160 对问句轮实测：**48% 的回复第一句没有在回答用户**（开推理法官，§8.17）。
# 而合同第 13 行**早就写着**「用户在问你时先回答他问的点」，但它在合同正文的 **26% 处、
# 后面还有 1268 字**，再加上 28 段记忆 → 埋在前面。
# 本开关只在 system prompt **最末尾**追加一行，**不新增规则、不改任何已有文字**。
# ⚠ 这是**假设**（模型对末尾更敏感），要靠 A/B 验证，不是已证结论。
# 默认 **false** ⇒ prompt 与改前**逐字相同**（`bugreports/probe_answer_first_tail_1017.py`）。
REPLY_ANSWER_FIRST_TAIL_ENABLED = os.getenv(
    "REPLY_ANSWER_FIRST_TAIL_ENABLED", "false"
).lower() in ("true", "1", "yes")
# ── 闲聊盘人格叙事「自己的」开关（plans/1017-01 §12 A1）────────────────────
# 为什么需要单独一个：上面两个 `PROFILE_MEMORY_*` 是**全局**开关，一开连
# 素材/场景/练习四盘一起变；而「用她自己的话记住这个人」只该发生在闲聊盘的
# 伴聊人格上。本开关只作用于那一条链：
#   * 读：`ctx.profile_narratives`（会话启动读一次）→ 逐轮 prompt 的
#         「我记得的这个人」段，**替换**同源的档案行（同一事实只允许一个权威源）；
#   * 写：无。叙事由会话末的 `_profile_narrative_cold_pass` 沉淀，
#         那一步仍归 `PROFILE_MEMORY_PROJECT_ENABLED` 管。
# 默认 **false** ⇒ 与「这条链不存在」同义，四盘 prompt 逐字不变。
# ⚠ 依赖：只开本开关、不开投影 ⇒ 叙事永远为空 ⇒ 自动回落档案行（无害但无效）。
COMPANION_NARRATIVE_MEMORY_ENABLED = os.getenv(
    "COMPANION_NARRATIVE_MEMORY_ENABLED", "true"
).lower() in ("true", "1", "yes")
# 逐轮注入的画像行数上限（0 = 不注入）。有界，避免画像段挤掉既有上下文预算。
PROFILE_MEMORY_MAX_LINES = max(0, min(20, int(os.getenv(
    "PROFILE_MEMORY_MAX_LINES", "8"
))))

# ── 审计与自修复系统配置 ──────────────────────────────────
AUDIT_MIN_SESSIONS = int(os.getenv("AUDIT_MIN_SESSIONS", "5"))       # 最少 pending 会话数才触发审计
AUDIT_TRIGGER_THRESHOLD = int(os.getenv("AUDIT_TRIGGER_THRESHOLD", "5"))  # 轮次触发阈值
# ── 审计与自修复系统配置 ──────────────────────────────────
AUDIT_MIN_SESSIONS = int(os.getenv("AUDIT_MIN_SESSIONS", "5"))       # 最少 pending 会话数才触发审计
AUDIT_TRIGGER_THRESHOLD = int(os.getenv("AUDIT_TRIGGER_THRESHOLD", "5"))  # 轮次触发阈值
# OPENCODE_BASE_URL / OPENCODE_API_KEY 定义在"记忆整理"那一节（本文件靠前处），此处复用。
AUDIT_LLM_MODEL = os.getenv("AUDIT_LLM_MODEL", "deepseek-v4.1-flash")
# 选型依据（2026-09-18 实测，真实审计 prompt：system 3650 字符 + payload 1015 字符，
# 均在 OpenCode Go 档、同一把 key、**显式关思考**）：
#   deepseek-v4.1-flash   7.9s / 1456 output tokens / recs=3 / JSON 完整 / target_id 未编造
#   deepseek-flash        6.5s / 1410 output tokens / recs=3 / 同上
#   deepseek-v4-pro      12.2s / 1071 output tokens / recs=2
#   ⛔ deepseek-v4-flash  167.8s / 9938 tokens —— **不认 thinking 开关**，且 1400 max_tokens
#      会被推理吃光导致 finish_reason=length（JSON 截断）。**不要选它。**
#   ⛔ union-alpha        91.8s，且只认 Anthropic /messages（OpenAI 端点 500）—— 不要用。
# ⚠ 关思考这一步是必须的：不关时 deepseek-v4.1-flash 要 34.7s / 7464 tokens（差 4.4× / 5.1×）。
AUDIT_LLM_BASE_URL = os.getenv("AUDIT_LLM_BASE_URL", OPENCODE_BASE_URL)
AUDIT_LLM_API_KEY = os.getenv("AUDIT_LLM_API_KEY", "") or OPENCODE_API_KEY or DASHSCOPE_API_KEY
# 审计单次请求超时。llm_client 的硬上限已由 60s 放宽到 MAX_REQUEST_TIMEOUT(默认 180s)；
# 这里给 120s 是留足余量（实测 7~12s，但第三方网关会抖动）。
AUDIT_LLM_TIMEOUT = max(5, min(180, int(os.getenv("AUDIT_LLM_TIMEOUT", "120"))))
AUDIT_DUPLICATE_WINDOW_DAYS = int(os.getenv("AUDIT_DUPLICATE_WINDOW_DAYS", "1"))   # 去重窗口改为 1 天
AUDIT_MAX_PAYLOAD_CHARS = int(os.getenv("AUDIT_MAX_PAYLOAD_CHARS", "50000"))  # LLM 输入字符上限
AUDIT_MAX_TURNS_PER_SESSION = int(os.getenv("AUDIT_MAX_TURNS_PER_SESSION", "20"))  # 每会话最多分析轮次

# ── 专用故障修复与灾备模型配置 (来自 llm大模型和.txt) ──────────────────────────
REPAIR_LLM_MODEL = os.getenv("REPAIR_LLM_MODEL", "qwen-max")  # 专用故障与代码报错修复模型 (glm-5.2)
DASHSCOPE_FALLBACK_POOL = ["qwen-plus", "qwen-max", "qwen-turbo"]

# ── 动态自适应与即刻运行控制 ──────────────────────────────────
AUTO_RUN_IMMEDIATELY = os.getenv("AUTO_RUN_IMMEDIATELY", "false").lower() in ("true", "1", "yes")
# Online style changes stay disabled until feedback can be attributed to a
# specific tutor action and validated in a stable user-level experiment.
ENABLE_STYLE_AUTO_OPTIMIZATION = os.getenv(
    "ENABLE_STYLE_AUTO_OPTIMIZATION", "false"
).lower() in ("true", "1", "yes")
INTERVAL_MIN = int(os.getenv("INTERVAL_MIN", "30"))           # 最快动态间隔（秒）
INTERVAL_MAX = int(os.getenv("INTERVAL_MAX", "1800"))         # 最慢动态间隔（秒）
MAX_TURNS_MIN = int(os.getenv("MAX_TURNS_MIN", "5"))           # 会话轮次动态下限
MAX_TURNS_MAX = int(os.getenv("MAX_TURNS_MAX", "30"))          # 会话轮次动态上限
EVOLVE_STAGNATION_THRESHOLD = int(os.getenv("EVOLVE_STAGNATION_THRESHOLD", "3"))  # 停滞计数阈值
EVOLVE_HARDNESS_ADJUST = float(os.getenv("EVOLVE_HARDNESS_ADJUST", "0.2"))          # 困难度升阶幅度

# ── 控制变量测试配置 ──────────────────────────────────────
INCLUDE_EVOLVED_PERSONAS = os.getenv("INCLUDE_EVOLVED_PERSONAS", "true").lower() in ("true", "1", "yes")  # 开启变异高压考题，充分检验极限防守与自修复
TEST_TARGET_PERSONAS = os.getenv("TEST_TARGET_PERSONAS", "")  # 指定测试的 persona_id（逗号分隔），留空则跑默认基线组


def validate_runtime_config() -> None:
    """Fail fast for the supported production route.

    Local development deliberately remains light-weight: SQLite/JSON and mock
    services can be used without cloud credentials.  Production is the WeChat
    mini-program + Doubao proxy + FastAPI route and must never start with
    implicit local storage or missing credentials.
    """
    if MEMORY_BACKEND not in {"local", "redis"}:
        raise RuntimeError("MEMORY_BACKEND must be 'local' or 'redis'.")

    if APP_ENV != "production":
        return

    missing = []
    if MEMORY_BACKEND != "redis":
        missing.append("MEMORY_BACKEND=redis")
    for name, value in {
        "DATABASE_URL": DATABASE_URL,
        "REDIS_URL": REDIS_URL,
        "VOLC_APP_ID": VOLC_LIVE_APP_ID,
        "VOLC_ACCESS_TOKEN": VOLC_ACCESS_KEY,
        "VOLC_APP_KEY": VOLC_APP_KEY,
        "WX_APPID": WX_APPID,
        "WX_SECRET": WX_SECRET,
    }.items():
        if not value:
            missing.append(name)
    if missing:
        raise RuntimeError("Production configuration is incomplete: " + ", ".join(missing))
