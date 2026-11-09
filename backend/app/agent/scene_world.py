"""场景世界锚（plans/0916-03 第九~十三节）。

为什么需要这个模块
------------------
实测（11 组 39 轮，`test_runner/diag_scene_depth.py`）暴露了四件事：

1. **场景没有"世界"**。入戏后 AI 从不贡献任何世界事实：不问杯型/冷热/支付，
   不报价格，不提今日特餐售罄，也不给等待时间。咖啡店 7 组"具体名词/轮"全部 0.0
   —— 所有具体信息都来自用户自己的话。它是镜子，不是对手角色。
2. **没有坐标**。同一个动作（点单时报姓名）在美国是对的、在国内是错的，
   而系统从没定过坐标。用户抱怨的"它让我报名字、完全架空"，
   根因就在 `tutor_agent.py` 的 `[Anti-Parroting]` 把"咖啡店问名字"写成了正确示范
   —— 那是英语国家坐标的行为，被当成了普适示范。
3. **没有进度与后果**。点单点完了又问"您要点什么"；还没付款就说"祝您愉快"。
   每轮从零开始，已确认的信息留不住。
4. **没有能力边界与升级链**。服务员被问"酱里有什么香料"直接编配方，
   被要求退款直接承诺退款；说了"我去问一下"却没有下文。
   代价是幻觉——而根因是**我们没给"我答不了"一个正当出路**。

本模块只做静态数据与纯函数：不调用任何模型，不产生额外延迟。
它回答的不是"要不要进场景"（那是 IntentRouter 的门），
而是"进去之后，这是个什么样的世界"。

字段含义
--------
- place/phases/events：场所面目、现实环节、会发生的事（让世界自己制造处境）
- ladder：服务链——每一层"手里有什么 / 能答什么 / 答不了什么"
- locales：两套坐标（国内 / 英语国家）下的流程差异，其中 name_policy 决定
  是否该索取姓名（国内不得索取）
- relay：转达判据——事实型问题代为转达，权限型/情绪型把角色请来
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 与 tutor_agent._PROMPT_CACHE_ORDER 同一开关（避免循环导入，本地重读同
# 名环境变量）：缓存序下把逐轮变化的行（本轮已确认 / 开场）挪到段尾，
# 让 section 前半逐字稳定（plans/0916-10 续）。=0 逐字回到现状。
_CACHE_ORDER = os.getenv(
    "TUTOR_PROMPT_CACHE_ORDER", "1"
).strip().lower() not in ("0", "false", "no", "off")

LOCALE_DOMESTIC = "domestic"
LOCALE_ENGLISH = "english"
LOCALE_LABEL = {LOCALE_DOMESTIC: "国内（中国）", LOCALE_ENGLISH: "英语国家"}

# 场所标识：只用于"已进入场景后"挑卡片，不承担"要不要进场景"的判断
_CARD_HINTS: List[Tuple[str, str]] = [
    ("cafe", r"咖啡|拿铁|美式|卡布奇诺|摩卡|星巴克|瑞幸|coffee|latte|espresso|"
             "cappuccino|americano|mocha|取餐码|咖啡店|咖啡馆"),
    ("restaurant", r"餐厅|饭店|餐馆|点菜|菜单|点单|服务员|主厨|厨师|领班|订位|忌口|"
                   "restaurant|waiter|waitress|menu|chef|reservation"),
    # 2026-09-22 补（P22）：用户点名的"咖啡店 / 机场"里，机场这张一直缺卡 ⇒
    # 机场戏只能靠模型自带的（英语国家）先验，正是"片面认知"的来源。
    ("airport", r"机场|航站楼|登机口|登机牌|登机|值机|行李托运|托运|安检|航班|机票|"
                "改签|延误|airport|terminal|boarding|check[- ]?in|luggage|"
                "baggage|\bflight\b"),
]

# 坐标线索
_ENGLISH_HINTS = re.compile(
    r"国外|海外|美国|英国|加拿大|澳洲|澳大利亚|新西兰|英语国家|当地|"
    r"America|American|US\b|USA|UK|Britain|British|abroad|overseas|"
    r"小费|\btip\b|starbucks|星巴克",
    re.IGNORECASE,
)
_DOMESTIC_HINTS = re.compile(
    r"国内|中国|大陆|中国特色|瑞幸|luckin|微信|支付宝|美团|淘宝|小程序|扫码|取餐码|桌码",
    re.IGNORECASE,
)


CARDS: Dict[str, Dict[str, Any]] = {
    "cafe": {
        "name": "咖啡店",
        "place": (
            "一家中等价位的连锁咖啡店；早高峰忙、常排队；"
            "有堂食座位与外带窗口；当日常有季节限定饮品；柜台上有限定与会员提示"
        ),
        "phases": [
            "迎接与点单",
            "规格定制（杯型 / 冷热 / 奶 / 甜度 / 冰量）",
            "支付",
            "取餐（叫号或取餐码）",
            "异常处理（做错 / 凉了 / 等太久）",
        ],
        "events": [
            "当日的季节限定饮品刚售罄，需要给替代方案",
            "前面排了四五个人，需要给出真实等待时间",
            "饮品做错了（奶 / 糖 / 冰不符），需要当场重做",
            "扫码机或收银机故障，只能换一种支付方式",
            "顾客的优惠券今天不能用，需要解释规则",
        ],
        "ladder": [
            {
                "role": "店员",
                "holds": "收银台、点单记录、取餐区、当日菜单牌",
                "can": "点单、规格、支付、取餐、简单退换",
                "cannot": "退款、改价、处理投诉、配方与原料细节",
            },
            {
                "role": "店长",
                "holds": "当日营业权限、退款与免单权限、排班",
                "can": "退款、免单、投诉处理、设备故障处置",
                "cannot": "总部政策（跨店优惠券、会员规则）",
            },
            {
                "role": "咖啡师领班（吧台负责人）",
                "holds": "当日豆子与糖浆库存、配方与可替换范围",
                "can": "配方与原料细节、过敏原、可替换项",
                "cannot": "价格与退款",
            },
            {
                "role": "区域经理",
                "holds": "跨店政策与会员规则",
                "can": "政策类问题、跨店投诉",
                "cannot": "当班即时处置（需转回店长）",
            },
        ],
        "relay": "事实型问题（能不能换奶、含不含咖啡因）→ 本轮内去问、带回答案；"
                 "权限型 / 情绪型（退款、投诉、设备故障）→ 把店长请来",
        "locales": {
            LOCALE_DOMESTIC: {
                "order": "小程序或 App 先下单支付，也可在柜台扫码点单",
                "pay": "扫码支付（微信 / 支付宝）为主",
                "pickup": "出取餐码或叫号，店员核对饮品名与号码；也可扫自取码",
                "name_policy": "不索取姓名。国内以取餐码 / 订单号 / 手机号后四位识别，"
                               "索取姓名是错的",
                "tip": "没有小费",
                "extra": "以外带为主，堂食座位有限；杯贴或小票上印有号码与饮品",
            },
            LOCALE_ENGLISH: {
                "order": "在柜台点单，先问杯型、冷热、堂食还是外带",
                "pay": "柜台现场支付，可刷卡或现金",
                "pickup": "常把名字写在杯上、叫名字取餐；也有叫号的店",
                "name_policy": "常问“What name should I put on the cup?”——"
                               "这是当地常态，不是异常",
                "tip": "可能有小费（收银台小费罐或刷卡小费）",
                "extra": "堂食与外带都常见",
            },
        },
    },
    "restaurant": {
        "name": "餐厅（堂食）",
        "place": (
            "一家中等价位的意式小馆，主打手工意面与披萨；"
            "午市忙、晚市常需等位；有当日特餐板；素食与无麸质可做但需提前说"
        ),
        "phases": [
            "迎宾排位（几位、有无预订、等多久）",
            "入座与菜单（纸质或扫码、当日特餐）",
            "点单（主菜 + 配菜 / 熟度 / 辣度 / 饮品）",
            "忌口与替换（过敏 / 素食 / 去配料 / 酱汁单放）",
            "用餐中（催菜 / 上错菜 / 温度 / 加水加饮料）",
            "结账（分账 / 发票 / 打包 / 小费）",
        ],
        "events": [
            "今日特餐刚售罄，需要给替代方案",
            "厨房出菜慢，需要给出真实等待时间",
            "菜上错了，需要核对并当场处理",
            "某道配菜或饮品今天没有了",
            "高峰时段限时就餐，需要提前说明",
        ],
        "ladder": [
            {
                "role": "服务员",
                "holds": "菜单、点单本、当日特餐板、排位表",
                "can": "点单、上菜、排位、结账、常见问题",
                "cannot": "退款、免单、改价、菜品配方与烹饪工艺细节",
            },
            {
                "role": "领班",
                "holds": "当班服务调度、简单的免单权限",
                "can": "投诉处理、催菜、座位调整",
                "cannot": "正式退款与赔偿",
            },
            {
                "role": "店长 / 经理",
                "holds": "退款与免单权限、客诉处置",
                "can": "退款、免单、赔偿、投诉闭环",
                "cannot": "菜品配方与后厨工艺",
            },
            {
                "role": "后厨主厨",
                "holds": "当日食材、备料进度、配方与工艺、忌口可调范围",
                "can": "菜品成分与过敏原、做法、可替换配料",
                "cannot": "账单、退款、排位",
            },
        ],
        "relay": "事实型问题（含不含奶、放不放某种香料）→ 本轮内去问、带回答案；"
                 "权限型 / 情绪型（退款、投诉、要求见厨师）→ 把对应角色请来",
        "locales": {
            LOCALE_DOMESTIC: {
                "order": "扫码点单（桌码）常见，也可叫服务员点单",
                "pay": "扫码或到前台支付；通常不收服务费",
                "pickup": "菜品直接送到桌上，无需取餐码",
                "name_policy": "一般不登记姓名；等位可能留手机号尾号",
                "tip": "没有小费",
                "extra": "餐具、茶水多为自助或主动配齐",
            },
            LOCALE_ENGLISH: {
                "order": "服务员到桌点单；进店先报人数或预订名",
                "pay": "刷卡或现金；朋友聚餐常说 separate checks（分账）",
                "pickup": "菜品上桌，由服务员负责对应座位",
                "name_policy": "等位常登记姓名并叫名字入座——这是当地常态，不是异常",
                "tip": "小费常见为账单的 15%–20%",
                "extra": "餐前面包、冰水常默认提供",
            },
        },
    },
    # ── 2026-09-22 新增（P22：用户点名"咖啡店 / 机场"这一类真实场景模板）──
    # 只新增一个 key：cafe / restaurant 两张卡的内容与渲染逐字未动。
    "airport": {
        "name": "机场（出发）",
        "place": (
            "一座国内一线城市的国际机场出发层；早班机人流大、值机岛与自助机并行；"
            "安检需脱外套、把电子设备与充电宝单独取出；登机口按区分片，广播频繁"
        ),
        "phases": [
            "进航站楼、找航司柜台或自助值机机",
            "值机与选座（含托运行李、超重处理）",
            "过安检（液体、充电宝、电子设备）",
            "找登机口与等候（留意广播与屏显）",
            "登机（凭登机牌或电子登机牌）",
            "异常处理（行李超重 / 延误 / 登机口变更 / 改签）",
        ],
        "events": [
            "值机时提示行李超重，需要当场决定托运付费还是取出随身",
            "安检被告知随身液体超限，需要就地处理",
            "广播通知登机口变更，需要重新找口",
            "航班延误，需要问清等待时间或改签",
            "自助值机机读不出证件，需要转人工柜台",
        ],
        "ladder": [
            {
                "role": "值机员",
                "holds": "值机柜台、座位图、行李标签与托运系统",
                "can": "值机、选座、托运行李、超重收费、改签引导",
                "cannot": "安检规定解释、航班调度与延误原因",
            },
            {
                "role": "安检员",
                "holds": "安检通道与开箱检查权",
                "can": "物品携带规定、开箱复核、要求取出物品",
                "cannot": "值机、改签、行李托运",
            },
            {
                "role": "登机口地服",
                "holds": "登机口、广播、当日座位调整",
                "can": "登机、座位调换、延误与登机顺序说明",
                "cannot": "退改签、行李赔偿",
            },
            {
                "role": "航司柜台 / 客服",
                "holds": "客票与改签权限",
                "can": "改签、退票与退款、延误赔付、特殊旅客安排",
                "cannot": "安检处置与现场放行",
            },
        ],
        "relay": "事实型问题（充电宝能不能带、液体限多少毫升）→ 本轮内去问、带回答案；"
                 "权限型 / 情绪型（改签、赔偿、要求见主管）→ 把航司柜台请来",
        "locales": {
            LOCALE_DOMESTIC: {
                "order": "先在航司 App 或自助值机机选座值机，再到柜台托运；也可直接排人工柜台",
                "pay": "扫码支付（微信 / 支付宝）为主，行李超重费同样扫码",
                "pickup": "凭电子登机牌或纸质登机牌过安检、登机；按登机口排队",
                "name_policy": "核对证件（身份证 / 护照），不靠叫名字登机；"
                               "广播喊名字只在个别提醒场景，索取姓名不是常态",
                "tip": "国内航班没有小费",
                "extra": "电子登机牌为主，身份证即可；充电宝必须随身、不得托运",
            },
            LOCALE_ENGLISH: {
                "order": "在柜台办理值机，先问证件与目的地；行李当场称重",
                "pay": "刷卡或现金；行李超重按件收费",
                "pickup": "按区（zone）分组排队登机，广播常叫名字提醒",
                "name_policy": "地服会在广播里点名提醒——这是当地常态，不是异常",
                "tip": "行李搬运可能给小费",
                "extra": "多数机场安检要求脱鞋、取出笔记本；登机牌纸质与电子并用",
            },
        },
    },
}


def detect_locale(text: str, default: str = LOCALE_DOMESTIC) -> str:
    """判定坐标。国内线索优先（用户身处国内，缺的正是模型不会的那一套）。

    实测依据（plans/0916-03 第十一节）：模型自带的先验是"英语国家连锁咖啡店"，
    所以国内那套流程才是真缺口；用户抱怨"让你报名字、完全架空"，
    正是因为坐标不明时落到了英语国家先验上。
    """
    blob = str(text or "")
    if _DOMESTIC_HINTS.search(blob):
        return LOCALE_DOMESTIC
    if _ENGLISH_HINTS.search(blob):
        return LOCALE_ENGLISH
    return default


def resolve(brief: str) -> Optional[str]:
    """按场景描述挑卡片。**只在已进入场景后调用**（不承担"要不要进"的判断），
    所以"我喜欢喝咖啡"这类提及不会被误伤——那时根本不会走到这里。
    """
    blob = str(brief or "")
    if not blob:
        return None
    best_key, best_hits = None, 0
    for key, pattern in _CARD_HINTS:
        hits = len(re.findall(pattern, blob, re.IGNORECASE))
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key


# ── 场景内"正当话题"词元 ────────────────────────────────────────
# 用途：给"话题漂移"护栏当基线。反例——餐厅戏里 AI 说 "our food is great"，
# 护栏的食物组（noodles|food|面条|食物）在 brief 与用户原话里都找不到匹配，
# 于是把 on-scene 的内容判成"引入了旧话题"并触发重试（白花一次延迟）。
# 世界锚定义了"什么是这个场景里的话题"，所以基线必须含它。
_ON_SCENE_TERMS: Dict[str, str] = {
    "cafe": "食物 food 饮品 drink 咖啡 coffee 甜点 dessert 菜单 menu 吃的 小吃",
    "restaurant": "食物 food 菜品 dish 意面 pasta 面条 noodles 菜单 menu 饮品 drink 甜点",
    "airport": "航班 flight 机票 ticket 登机牌 boarding pass 行李 luggage baggage "
               "登机口 gate 座位 seat 安检 security 护照 passport 证件 ID",
}


def on_scene_terms(card_key: str) -> str:
    """该场景内的正当话题词元；无卡片时返回空串。"""
    return _ON_SCENE_TERMS.get(str(card_key or ""), "")


# ── 已确认信息（确定性抽取，治"问已经答过的问题"） ────────────────
# 实测反例：用户已说 "Medium, oat milk, no sugar."，下一轮店员又问
# "Could you tell me what you'd like to order?"；付款前就说 "Have a great day."
# 这类信息不需要模型判断，代码扫一遍就能拿到。
_DETAIL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(small|medium|large|grande|venti)\b", "杯型"),
    (r"\b(oat|soy|almond|whole|skim|low[- ]fat)\s*milk\b", "奶类"),
    (r"\b(no|less|half|extra)\s*(sugar|ice|sweet\w*)\b", "甜度/冰量"),
    (r"\b(hot|iced|ice)\b", "冷热"),
    (r"\b(for here|to go|takeaway|take away|eat in|dine in)\b", "堂食/外带"),
    (r"\bwithout\s+([a-z]{3,20})\b", "去掉配料"),
    (r"\b(no|without)\s+(onions?|garlic|cheese|nuts?|dairy|meat)\b", "忌口"),
    (r"allergic to\s+([a-z]{3,20})", "过敏"),
    (r"\btable for\s+(\w+)", "人数"),
    (r"\b(reservation|booked?)\b.*?(name|under)\b", "预订"),
    (r"\border number\b|\bpickup (?:number|code)\b|\bnumber is\s+([\w-]+)", "取餐号"),
    (r"\b(one|two|three|\d+)\s+(lattes?|coffees?|cappuccinos?|beers?|pastas?|pizzas?)\b", "数量"),
]

_CJK_ONLY = re.compile(r"^[\u4e00-\u9fff\s，。！？、；：]+$")


def extract_confirmed_details(texts: Iterable[str]) -> List[str]:
    """从最近几轮用户原话里抽出"已经确认过的处境信息"。

    只做确定性匹配，不做语义推断：抓到什么就写什么，宁可少写也不要写错。
    """
    found: List[str] = []
    for text in texts:
        blob = " ".join(str(text or "").split())
        if not blob or _CJK_ONLY.match(blob):
            continue
        for pattern, label in _DETAIL_PATTERNS:
            for match in re.finditer(pattern, blob, re.IGNORECASE):
                raw = match.group(0).strip()
                snippet = f"{label}={raw}"
                if snippet not in found:
                    found.append(snippet)
    return found[-10:]


def render(  # noqa: PLR0913 - 纯渲染，参数直白更易读
    card_key: str,
    locale: str,
    confirmed: Optional[List[str]] = None,
    turn: int = 0,
) -> str:
    """把世界锚渲染成注入 prompt 的紧凑文本。

    设计约束：**不新增任何模型调用**，所以只是把静态数据拼成一段文字；
    文本长度控制在十几行内，避免挤占既有 prompt 预算。
    """
    card = CARDS.get(card_key)
    if not card:
        return ""
    loc = (card.get("locales") or {}).get(locale) or {}
    lines: List[str] = [
        "[Scene World - 本盘已设定的虚构世界，按此具体化；它不是“建议”，是既定事实]",
        f"这是哪儿：{card['place']}",
        f"坐标：{LOCALE_LABEL.get(locale, locale)}"
        + (f" · 点单方式：{loc['order']}" if loc.get("order") else ""),
    ]
    if loc.get("pay"):
        lines.append(f"支付：{loc['pay']}" + (f"；取餐：{loc['pickup']}" if loc.get("pickup") else ""))
    if loc.get("name_policy"):
        lines.append(f"姓名处理（硬规则）：{loc['name_policy']}")
    if loc.get("tip"):
        lines.append(f"小费：{loc['tip']}")
    lines.append("现实环节（用于判断现在走到哪一步）：" + " → ".join(
        f"{i}.{step}" for i, step in enumerate(card["phases"], 1)
    ))
    lines.append(
        "此刻可发生的事（每轮最多引入一件，当成已发生的事实说出来，"
        "不要问用户“要不要”）：" + "；".join(card["events"])
    )
    lines.append("对面角色与能力边界（答不了就必须走下面的服务链，不许自己编）：")
    for item in card["ladder"]:
        lines.append(
            f"  · {item['role']}：手里有{item['holds']}；能答{item['can']}；"
            f"答不了{item['cannot']}"
        )
    lines.append(f"转达规则：{card['relay']}")
    lines.append(
        "【服务链闭环】说“我去问一下”必须在同一轮里给出结果"
        "（例如“我问了下吧台——这次用的是燕麦奶，不含乳制品”），"
        "绝不允许只说去问、却不带答案回来。"
    )
    lines.append(
        "【连续性】已在上文出现过的信息不得重复询问；已经推进过的环节不得倒退回去。"
    )
    if _CACHE_ORDER:
        # 缓存序：confirmed 逐轮增长、开场行第 2 轮消失——都是 mid-string 变化，
        # 会切断整个 section 的公共前缀。统一挪到段尾，前半段逐字稳定。
        if turn <= 0:
            lines.append(
                "【开场】第一轮不要只说“您要点什么”，先给一个具体处境"
                "（如当日限定售罄 / 正在排队 / 需要确认某件事），让用户有具体的话可说。"
            )
        if confirmed:
            lines.append("本轮已确认（禁止再问一遍）：" + " | ".join(confirmed))
        return "\n".join(lines)
    if confirmed:
        lines.append("本轮已确认（禁止再问一遍）：" + " | ".join(confirmed))
    if turn <= 0:
        lines.append(
            "【开场】第一轮不要只说“您要点什么”，先给一个具体处境"
            "（如当日限定售罄 / 正在排队 / 需要确认某件事），让用户有具体的话可说。"
        )
    return "\n".join(lines)


def render_name_rule(locale: str) -> str:
    """坐标相关的姓名规则，供需要单独强调的地方复用（如生成合同）。"""
    if locale == LOCALE_DOMESTIC:
        return (
            "坐标=国内：**不得索取姓名**。国内以取餐码 / 订单号 / 手机号后四位识别，"
            "追问姓名属于架空行为。"
        )
    return (
        "坐标=英语国家：可以问“What name should I put on the cup?”，"
        "这是当地常态。"
    )
