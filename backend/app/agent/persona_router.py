"""人格路由与记忆分区（plans/0916-28）。

用户口径（2026-09-17）："进了角色扮演/场景练习，他就聚焦在这一层——之前那个
'能接用户任何话'的模式就自动切过来了，不需要调用两个模型，可能是同一个模型，
只是做不同的事。"

所以本模块只做**确定性路由**（零模型调用、零额外延迟）：读既有的入戏机制结论
（``ctx.scene_context_set`` / ``language_question`` / 显式出戏说法），决定这一轮
**谁在说话**，并给出该人格的规范块与记忆作用域。

三种人格（都不是新的模型调用，只是同一轮 prompt 里不同的规范块）：
  - ``companion``   模式 A：很了解你的朋友（未进场景；可调用用户全部信息，
                    能教但不像上课，潜移默化嵌一点要学/要巩固的）；
  - ``scene_role``  模式 B：场景里的角色（只说这个世界里会发生的事；教学退到
                    最低，只在用户求助时出手）；
  - ``tutor_sidecar`` 半入戏：用户在场景里问语言本身（词义/语法/怎么说）——
                    换成外教视角答，但**保留当前场景上下文**。

记忆分区（用户拍板：写入侧只标两级 ``real`` / ``scene``，作用域按"盘"）：
  - 标签管性质（是不是戏里的），conv_id 管范围（哪一盘）；
  - 戏内事实不跨盘："咖啡馆的'我点了燕麦拿铁'不许出现在餐厅戏里"；
  - 两级标签以 ``[scene]`` 前缀形式落在既有文本字段里（实体 detail / 归档摘要），
    读取侧剥掉前缀再比对——**不新增数据库列**，避免"多级标签标错=静默污染"。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

SCENE_ORIGIN = "scene"
REAL_ORIGIN = "real"
# 出处前缀：只用于标注与过滤，注入 prompt 前会被剥掉（用户口径："标注出处，
# 不做硬性模板"——所以是一个极轻的前缀，不是一套结构化模板）。
ORIGIN_MARK_RE = re.compile(r"^\s*\[(scene|real)\]\s*")

# 与 tutor_agent._extract_scene_info 同一族的入戏形态（本模块自持一份，
# 避免循环 import；口径必须与那边一致，否则"路由说没入戏、上下文说入戏"）。
_SCENE_SETUP_RE = re.compile(
    r"(你来(?:当|扮演|做|演)|你扮演|你演一下|假装(?:你|我们)|"
    r"(?:我们|咱们)?(?:来)?模拟(?:一下)?|场景(?:练习|是|：)|"
    r"进入(?:一个)?场景|换个角色|roleplay|role\s*play|"
    r"我们(?:在|去)(?:咖啡店|餐厅|机场|酒店|医院|商店|超市|银行|车站))",
    re.IGNORECASE,
)
_ZERO_ARTICLE = re.compile(r"[《》\"'··]")

# 显式出戏 / 语言侧问（半入戏）：判据全部确定性，且都是用户"明说"的形态——
# 用户零操作，只有他自己要出戏才切回外教视角。
_EXIT_CUES = re.compile(
    r"(先不演|不演了|等一下|暂停一下|停一下|先停停|"
    r"退出(?:角色|场景)|出戏|结束(?:角色|场景|这段)|"
    r"换个话题|换(?:个|一)场景|"
    r"(?:这句|这句话|这个词|这个单词|这句话里|这个短语).{0,8}"
    r"(?:怎么说|什么意思|啥意思|英文|用英语|怎么表达)|"
    r"(?:问个|问一下|请?教一下).{0,6}(?:词|语法|表达|句子)|"
    r"(?:语法|时态|发音|读音)(?:问题|怎么|是)|"
    r"stop\s+the\s+roleplay|out\s+of\s+character|pause\s+the\s+scene)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def is_scene_setup_request(text: str) -> bool:
    """这一句是不是"我们开始演某个场景"的请求（复用既有入戏形态）。"""
    raw = _normalize(text)
    if not raw:
        return False
    return bool(_SCENE_SETUP_RE.search(_ZERO_ARTICLE.sub(" ", raw)))


def is_explicit_exit_cue(text: str) -> bool:
    """用户是否明确要出戏/问语言本身（模式 B → 外教视角的唯一开关）。"""
    return bool(_EXIT_CUES.search(_normalize(text)))


def is_language_sidecar(turn_contract: Optional[Dict[str, Any]]) -> bool:
    """本轮是不是语言侧问（词义/语法/怎么说）——复用既有 language_question 结论。"""
    contract = turn_contract or {}
    lang_q = contract.get("language_question") or {}
    if not isinstance(lang_q, dict):
        return False
    if str(lang_q.get("kind") or "").strip():
        return True
    return str(lang_q.get("answer_target") or "").strip() == "previous_reply"


def resolve_persona(
    user_text: str,
    *,
    raw_text: str = "",
    scene_active: bool = False,
    scene_changed: bool = False,
    turn_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """本轮该用哪套人格（确定性，用户零操作）。

    ``scene_changed=True``（本轮刚入戏）直接进 B，不因为同一句里带了"怎么说"
    就判半入戏——入戏那一句本身常常就是"你来当店员，我想练怎么说点单"。
    """
    raw = _normalize(raw_text or user_text)
    if not scene_active:
        return {
            "mode": "companion",
            "reason": "not_in_scene",
            "teaching_posture": "normal",
            "memory_scope": "all",
            "scene_focus": False,
        }
    if scene_changed:
        return {
            "mode": "scene_role",
            "reason": "scene_just_started",
            "teaching_posture": "low",
            "memory_scope": "scene_then_real",
            "scene_focus": True,
        }
    if is_explicit_exit_cue(raw) or is_language_sidecar(turn_contract):
        return {
            "mode": "tutor_sidecar",
            "reason": "explicit_exit_cue" if is_explicit_exit_cue(raw) else "language_question",
            "teaching_posture": "normal",
            "memory_scope": "all",
            "scene_focus": False,
        }
    return {
        "mode": "scene_role",
        "reason": "in_scene",
        "teaching_posture": "low",
        "memory_scope": "scene_then_real",
        "scene_focus": True,
    }


# ---------------------------------------------------------------------------
# 人格规范块（进 constraints["persona"] → prompt 的 [Persona] 段）
# ---------------------------------------------------------------------------

def persona_constraint(
    state: Dict[str, Any],
    *,
    learner_name: str = "",
    tutor_name: str = "",
    scene: str = "",
    tutor_role: str = "",
    learner_role: str = "",
    world_key: str = "",
) -> Dict[str, Any]:
    """把路由结论落成合同/规范块（空字段不写，普通轮零冗余）。"""
    mode = str((state or {}).get("mode") or "companion")
    payload: Dict[str, Any] = {
        "mode": mode,
        "reason": str((state or {}).get("reason") or ""),
        "teaching_posture": str((state or {}).get("teaching_posture") or "normal"),
        "memory_scope": str((state or {}).get("memory_scope") or "all"),
    }
    if learner_name:
        payload["learner_name"] = learner_name
    if tutor_name and mode != "material_guide":
        payload["tutor_name"] = tutor_name
    if mode in {"scene_role", "tutor_sidecar"}:
        if scene:
            payload["scene"] = scene
        if tutor_role:
            payload["tutor_role"] = tutor_role
        if learner_role:
            payload["learner_role"] = learner_role
        if world_key:
            payload["world_key"] = world_key
    payload["rule"] = _PERSONA_RULES[mode]
    return payload


_PERSONA_RULES = {
    "material_guide": (
        "你在素材盘中与用户讨论所选素材：先回答他问的内容，再根据素材证据提出有理由的分析。"
        "你可以有自己的判断，但不以闲聊盘 Alice 的身份、自述、经历或亲密关系发言。"
        "面对用户的推论与联想时，展现专业分析师的探讨素养：严格区分「材料明确直接交代的事实」与「可供探讨的解读/推论视角」。"
        "当用户尝试在材料的不同线索之间建立联系或探讨解读时，不要粗暴断定「你在脑补/毫无关联/You're adding it」；"
        "应先说明材料原文中是否有直接表述，再分析该联想在逻辑上是否说得通、能从哪个角度提供解读价值，保持建设性的讨论空间。"
        "严格区分声源角色：创作者/解说旁白（主观吐槽、感想、操作体验）vs 素材内人物对白/剧情（客观设定与对白）vs 评论区样本（第三方个例反馈）。"
        "绝不能把实况主的吐槽当成剧情角色的想法，也不能把游戏角色的对白当成实况主的立场。"
        "用户聊到素材之外的话题时自然接话，并可自然保留回归通道，不把外部话题伪装成素材事实。"
    ),
    "companion": (
        "你此刻在闲聊盘以陪伴者身份和用户自然交流，不默认你们已经熟悉或亲密；"
        "关系熟悉度与相处边界遵循本轮 Relationship arc contract。可以聊任何话题、答任何问题"
        "（知识/事实/家常都可以），需要时把更自然的说法顺手递给他，"
        "不布置作业、不做课堂式点评、不说'我们来练习'这类元话术。"
    ),
    "scene_role": (
        "你此刻是场景里的角色，不是外教：只说这个虚构世界里会发生的事，"
        "角色能力边界之外的不编（走服务链/给出世界里能拿到的答案）。"
        "教学动作退到最低——只在用户明确求助（问词义/语法/怎么说/卡壳）时才出手，"
        "本轮其余部分保持角色连贯，不点评、不布置练习、不提前送客。"
        "戏内事实优先于戏外事实；用户真实生活的信息只在自然相关时才用。"
    ),
    "tutor_sidecar": (
        "用户这轮问的是语言本身（词义/语法/怎么说），先用外教视角直接答清这一点，"
        "答完再回到当前场景继续——场景上下文与角色都保留，不因为切视角就重启场景。"
    ),
}


def render_persona_section(persona: Dict[str, Any]) -> str:
    """[Persona] 段文本（空/默认时返回空串，普通轮逐字零增量）。"""
    if not isinstance(persona, dict) or not persona:
        return ""
    mode = str(persona.get("mode") or "")
    if not mode:
        return ""
    bits: List[str] = []
    learner = str(persona.get("learner_name") or "").strip()
    tutor = str(persona.get("tutor_name") or "").strip()
    if mode == "companion":
        if learner:
            bits.append(f"用户是 {learner}")
        if tutor:
            bits.append(f"你在他那里的名字是 {tutor}")
    elif mode in {"scene_role", "tutor_sidecar"}:
        role = str(persona.get("tutor_role") or "").strip()
        scene = str(persona.get("scene") or "").strip()
        if role:
            bits.append(f"你的角色是 {role}")
        if scene:
            bits.append(f"场景是 {scene}")
    head = "[Persona] " + ("（" + "；".join(bits) + "）；" if bits else "")
    return head + str(persona.get("rule") or "")


# ---------------------------------------------------------------------------
# 行为倾向（plans/0919-01 §4 第 3 期 3.3）——**底层人格，贯穿所有场景与模式**
# ---------------------------------------------------------------------------

# 用户 2026-09-20 定调（原话）："底层人格这一块可以定大概的一个方向，但是后面有些
# 东西用户适应模块是可以微调的……**这个是贯穿 AI 它任何场景任何模式下的**，它不是
# 说要进入某个模式才会使用它——这个人格，以及底层之上的用户适应微调之后，形成的
# 这种一个蛮具象的一个 AI 人物。"
#
# 所以这一段**不按模式分化**：观点倾向与边界属"人格底层 = 不变 = 不可塑"
# （0919-03 §4.1 三层表原文："不可塑：核心原则、观点倾向、边界"）。模式只决定
# **怎么表现**（那在 `_PERSONA_RULES` 与 §4.4 的情绪式样里），不决定"是不是一个
# 有观点、有边界的人"。
#
# 曾走过的弯路（记下来免得重犯）：第一版把这段做成"三态各一套"，即"进了某个模式
# 才用哪套"——那既违反三层表，也把"表达方式"与"底层人格"混成了一层。用户当场纠正。
#
# 措辞依据：用户口径"否则很容易变成一个只会迎合用户的镜子"；边界里的"只承诺你此刻
# 身份能兑现的事"锚在真机事故上（店员说 "I'll process your refund right away."
# 越权幻觉，改成 "I'll go get the manager right away." 才对）。
_DISPOSITION_STANCE = (
    "你有稳定的善意、好奇心和自己的判断。通常先寻找与用户能相通的部分；"
    "遇到观点分歧时先弄懂他为什么这么想，再决定是否表达不同看法，不急着反驳，也不为了让他高兴而附和。"
    "赞同要出于你确实认同，保留意见要说清理由；新证据或更好的解释出现时，你也可以修正自己。"
    "交流中保持稳定的判断和分寸，随着新证据加深理解，不靠迎合用户来维持对话。"
)
_DISPOSITION_BOUNDARY = (
    "不知道就说不知道，不装懂、不做百科全书；只承诺你此刻身份能兑现的事"
    "（戏里尤其——没权限的走流程：去问、去查、去找谁，不用嘴答应下来）；"
    "不编造外部现实、政策、价格或规则，也不把用户或第三方的经历挪作自己的。"
    "明确场景中的角色只说当前场景允许的事实。"
    "表达情绪和不同意见时保持尊重，不以讨好换取好感，也不把边界变成惩罚或断绝交流。"
)
_DISPOSITION_EXAMPLE = (
    "✅“我懂你为什么这么看。我自己的感觉稍微不一样，因为……；不过你刚才提到的那点确实有道理。” "
    "❌“你说得都对。”／“你错了。”"
)


def render_disposition_section() -> str:
    """``[Disposition]`` 段：跨场景稳定、能从经历中成熟起来的底层人格。

    **不接模式参数、永远渲染**——这是贯穿所有场景所有模式的底，不是"进某个模式
    才启用"的东西（用户 2026-09-20 定调）。模式内的角色限制另见 `_PERSONA_RULES`
    （如 `scene_role` 的"角色能力边界之外的不编"），那是"怎么表现"，不是"是不是
    一个有观点有边界的人"。
    """
    return (
        "[Disposition] 你的行为倾向——**任何场景、任何模式下都成立**，不随切模式而变："
        f"① 有观点——{_DISPOSITION_STANCE}"
        f"② 有边界——{_DISPOSITION_BOUNDARY}"
        "③ 分层不变——这条只讲你**表达自己立场**的分寸，不改变 [Stance - content] 的"
        "「对内容可以有立场」与 [Non-Judgment - language] 的「对语言不评判」——各归各段。"
        f"（例：{_DISPOSITION_EXAMPLE}）"
    )


# ---------------------------------------------------------------------------
# 记忆分区：两级标签 + 按盘作用域
# ---------------------------------------------------------------------------

def tag_origin(text: str, origin: str) -> str:
    """给要落库的文本打出处前缀（幂等；``real`` 是默认值，也显式写，
    这样读取侧不必区分"没标"和"标了 real"）。"""
    body = ORIGIN_MARK_RE.sub("", _normalize(text))
    if not body:
        return ""
    label = SCENE_ORIGIN if str(origin or "") == SCENE_ORIGIN else REAL_ORIGIN
    return f"[{label}] {body}"


def split_origin(text: str) -> tuple:
    """返回 ``(origin, 去前缀正文)``；没有前缀按 ``real`` 处理（兼容旧数据）。"""
    raw = str(text or "")
    match = ORIGIN_MARK_RE.match(raw)
    if not match:
        return REAL_ORIGIN, _normalize(raw)
    return match.group(1), _normalize(ORIGIN_MARK_RE.sub("", raw))


def looks_like_scene_tape(messages: Any) -> bool:
    """整盘是不是场景盘（归档时定标签用）。

    判据：盘内**用户**消息里出现过入戏请求。用整盘而不是最近若干条——
    归档发生在会话进行中/结束时，逐轮状态（ctx.scene_context_set）已经拿不到，
    而"这盘进过戏"这件事只由用户自己那句请求决定。
    """
    if isinstance(messages, dict):
        items = messages.get("messages") or []
    else:
        items = messages or []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") != "user":
            continue
        if is_scene_setup_request(str(item.get("content") or "")):
            return True
    return False


def entity_origin(entity: Any) -> str:
    """一条实体行的出处（detail 前缀）。"""
    detail = entity.get("detail") if isinstance(entity, dict) else None
    origin, _ = split_origin(str(detail or ""))
    return origin


def filter_entities_for_persona(
    entities: Any, *, persona_mode: str, conv_id: str = ""
) -> List[Dict[str, Any]]:
    """按当前人格过滤实体（戏内事实按盘可见——**任何人格都不跨盘**）。

    - **戏内事实（scene）只在同一盘可见**，戏内戏外都一样：咖啡馆的"我点了
      燕麦拿铁"不许出现在餐厅戏里，也不该被当成用户的真实生活（用户拍板：
      标签管性质、conv_id 管范围，两件事分开做）；
    - 戏外事实（real）跨盘共享——弱点/偏好/真实经历本来就该越用越懂；
    - 返回的行已剥掉出处前缀（模型看不到 ``[scene]`` 标记）。

    ``persona_mode`` 目前只用于将来的形态差异（如戏内只带必要项），当前不改变
    可见性规则——**范围由 conv_id 决定，不由人格决定**。
    """
    out: List[Dict[str, Any]] = []
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        origin, body = split_origin(str(entity.get("detail") or ""))
        same_tape = bool(conv_id) and str(entity.get("conv_id") or "") == str(conv_id)
        if origin == SCENE_ORIGIN and not same_tape:
            continue
        item = dict(entity)
        item["detail"] = body
        item["origin"] = origin
        out.append(item)
    return out
