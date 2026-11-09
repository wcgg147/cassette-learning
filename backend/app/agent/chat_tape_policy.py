# -*- coding: utf-8 -*-
"""闲聊盘（Alice）自己的规则，从 `tutor_agent.py` 里拆出来。

为什么要单独一个模块（2026-10-17）：
`tutor_agent.py` 单文件 1.8 万行，三棵工作树同时在改它；把闲聊盘的规则塞进去，
每次改动都会和别人的 hunk 缠在一起——**机械按 hunk 切分会切出半截代码**（已实测编译失败）。
把这些规则集中在这里，`tutor_agent.py` 只剩几行调用，冲突面小得多。

这里的规则都来自产品判断，不是教学策略：
- **D011**：闲聊盘＝英语环境（用户说中文也用英文回，只有问词义才用中文）。
- **§8.51**：词汇预算（`adaptive_language_support`）是**教学盘**契约，闲聊盘不挂——
  否则"用户说中文、她回英文"会让每句英文都算"超预算新词"，每轮被重写。
- **§8.52**：长度不判违规。原来"用户≤25字且回复>180字⇒违规"会把自然的话整条重写，
  这正是公式感的来源；改成提示层的"语力跟随"。
"""

from __future__ import annotations

from typing import Any


def is_companion_chat(ctx: Any) -> bool:
    """这一轮是不是**闲聊盘**（Alice 日常对话）。

    判据：没有场景扮演、没有素材上下文、不是练表达盘、人格模式是 companion。
    练习盘（expression）、素材盘、角色扮演都不算——那些地方教学闸是应该生效的。
    """
    if getattr(ctx, "scene_context_set", False):
        return False
    if getattr(ctx, "material_context", None):
        return False
    if str(getattr(ctx, "cassette_mode", "") or "") == "expression":
        return False
    persona_state = getattr(ctx, "persona_state", None) or {}
    try:
        mode = str(persona_state.get("mode") or "companion")
    except AttributeError:
        mode = "companion"
    return mode == "companion"


def is_chat_tape_prepared(prepared: dict | None) -> bool:
    """`prepared` 里带的盘型标记（闸门判定用它，避免重复推断）。"""
    return bool((prepared or {}).get("chat_tape"))


def prompt_tail() -> str:
    """闲聊盘 prompt 末尾的固定段(必须放结尾;近因比措辞重要)。"""
    tail = ""
    if _english_forced():
        tail += (
            "\n\n[这一盘是英语环境] 对方用中文说，你也用英语回;"
            "只有对方问某个词或某句是什么意思时,才用中文解释那一点;"
            "给建议、出主意也整段用英语,别把建议部分切成中文。"
        )
    tail += (
        "[语力跟随] 对方说得短,你也别接成长篇大论;接住他这句话,"
        "真想多知道一句就自然问出来(不是每轮都问),不想问就接个情绪或给个看法"
        "(「听起来挺省事的」「换我也懒得做」);不用每句都解释你为什么这么答。"
        "[别重复自己] 你自己的事(在读什么、喝什么、天气这类)别每轮都拎出来当话头,"
        "对方追问才展开。"
        "[一个点别反复讲] 你已经讲过的道理、或他重复提到同一件事,"
        "别整句重说一遍,也别再给方案——换个角度,或只接一句情绪,"
        "或问一句你真正想知道的。"
        "[不许编经历] 你的过去只有给你的种子和这几天的材料;问到具体的经历"
        "(学过什么乐器、去过哪里、以前做什么),材料里没有就直接说没有,不要造细节。"
        "[别列菜单] 他问推荐、问怎么办,给**一个**主张,最多带一句备选;"
        "不要一口气报三四个选项让他自己挑。"
        "[别用口头禅] 不要每轮先用一句「我确认一下你的意思」之类的垫场话;"
        "确认听懂就直接接内容,不用每次都先声明。"
    )
    _manner = _manner_list()
    if _manner:
        tail += "\n[你自己的说话方式] " + "；".join(_manner) + "。"
    return tail


def language_free(ctx) -> bool:
    """闲聊盘 + 强制英文开关关 = 语言自由（不推英文、不带教学口吻）。"""
    if not is_companion_chat(ctx):
        return False
    return not _english_forced()


# 运行时覆盖（None=跟随配置；True=英语环境；False=原生自由）。
# 由 /settings/language-mode 接口设置，内存生效、不用重启；进程重启后回到配置值。
_runtime_english_forced = None


def set_english_forced(value):
    """运行时切换闲聊盘语言模式。True=英语环境；False=原生自由；None=跟随配置。"""
    global _runtime_english_forced
    _runtime_english_forced = None if value is None else bool(value)


def english_forced_state() -> dict:
    """当前语言模式：给前端按钮显示用。"""
    try:
        from config import CHAT_TAPE_ENGLISH_FORCED_ENABLED as _cfg
    except Exception:
        _cfg = False
    return {
        "forced": _english_forced(),
        "runtime": _runtime_english_forced,
        "config": bool(_cfg),
        "mode": "english_env" if _english_forced() else "native",
    }


def _english_forced() -> bool:
    """闲聊盘是否强制英语环境(2026-10-25 B开关,默认关)。"""
    if _runtime_english_forced is not None:
        return bool(_runtime_english_forced)
    try:
        from config import CHAT_TAPE_ENGLISH_FORCED_ENABLED as _on
    except Exception:
        _on = False
    return bool(_on)



def _manner_list() -> list:
    """说话方式种子(role=manner);开关关/读失败返回空。"""
    try:
        from config import PERSONA_SEED_ENABLED as _seed_on
        if not _seed_on:
            return []
        from memory.persona_seed import PERSONA_SEED_FACTS as _facts
        out = []
        for f in _facts or []:
            v = str(f.get("value_en") or "").strip().rstrip(".")
            if v and str(f.get("role")) == "manner":
                out.append(v)
        return out
    except Exception:
        return []



def allows_vocabulary_budget(ctx: Any) -> bool:
    """要不要挂教学盘的词汇预算（`adaptive_language_support`）。

    闲聊盘不挂：用户说中文、她回英文时，"新增英文内容词"必然超预算，
    结果是每轮判违规、每轮重写（实测 12 轮命中 11 轮）。
    """
    return not is_companion_chat(ctx)


# ── 人格先行（2026-10-24）──────────────────────────────────────────────
# 用户判据："以爱丽丝独立人格为出发点，搁置所有违背这条目标的指令"。
# 闲聊盘的 system prompt 里，服务型 / 教学型 / 控制型段与人格段并存，模型只能折中成
# 客服腔（实测：去掉单段无效，因为剩余段仍在说"你是来服务/教学的"）。
# 这里给出**唯一的**搁置清单：按段落起始标签匹配，只搁置列表里的段，其余原样保留。
# 开关 `CHAT_TAPE_PERSONA_FIRST_ENABLED` 关闭时，所有函数都返回与改前逐字相同的结果。

# 服务 / 教学 / 控制型段的起始标签（匹配 section 文本的前缀）。
# 不含：[Role]（只输出说给用户的话，是输出边界）、[Language]（用户问词义才用中文，
#      是语言卫生）、[Semantic Fidelity]（防误听，是事实约束）、[Reality]（防编造）、
#      [Continuity]、[Roleplay Language]、[No Meta]（防元叙述，红线）、[Precision]。
_PERSONA_FIRST_SHELVED_PREFIXES: tuple = (
    "[Language]",         # 中文输入→英文教学转译 + 【硬规则】给英文例句（初版教学遗留）
    "[Order]",            # 服从 Turn Controller 的服务编排
    "[Frame]",            # 服从 interaction_frame 的服务切换
    "[Pragmatics]",       # 服从 speaker_position/response_obligations（coach 与人格冲突）
    "[Teaching]",         # 一轮最多一个教学动作
    "[Reply Contract]",   # reply_contract 是服务边界
    "[Conversation Rhythm]",  # 问号节奏（配额式）
    "[Turn Rhythm]",      # 问句预算 + 碎片接住（替用户补全句子 = 服务式）
)

# 逐轮控制 JSON 段（[Policy] / [Constraints] 等），闲聊盘开关开时搁置。
_PERSONA_FIRST_SHELVED_DYNAMIC_PREFIXES: tuple = (
    "[Policy]",
    "[Learning Move]",
    "[Learning State]",
    "[Learner State]",
    "[Teaching Method",
    "[Assessment SOP]",
    "[Constraints]",
    "[Latest Turn Contract",
)

# 闲聊盘合同里的**配额式**子句（数字约束 / 问号配额）。按子串精确匹配后删除。
_PERSONA_FIRST_SHELVED_CONTRACT_CLAUSES: tuple = (
    # ⑤ 是数字配额（"1到2个短句"），按用户产品判断搁置。
    "⑤ 不啰嗦：1到2个短句说完，不铺垫、不总结、不复述上一轮说过的。",
    # ③ 曾随 ⑤ 一起被当"配额式"搁置——但它不是配额，是**姿态**
    # （不把话递回去、给看法然后停住）。真机实测：搁置后 13 轮里 8 轮以问句收尾，
    # 变成访谈主持人。2026-10-26 起不再搁置。
)


def persona_first_enabled() -> bool:
    """开关读取（延迟导入，避免模块级循环依赖）。"""
    try:
        from config import CHAT_TAPE_PERSONA_FIRST_ENABLED as _on
    except Exception:
        return False
    return bool(_on)


def shelved_section(section: str) -> bool:
    """这一段是否属于被搁置的服务/教学/控制段。"""
    text = str(section or "").lstrip()
    return text.startswith(_PERSONA_FIRST_SHELVED_PREFIXES)


def filter_stable_sections(sections: list, ctx: Any) -> list:
    """闲聊盘且开关开：去掉服务/教学/控制段；其余段（含空串）原样返回。"""
    if not (persona_first_enabled() and is_companion_chat(ctx)):
        return sections
    return [s for s in sections if not shelved_section(s)]


def filter_dynamic_tail(tail: list, ctx: Any) -> list:
    """闲聊盘且开关开：去掉逐轮控制 JSON 段；其余段原样返回。"""
    if not (persona_first_enabled() and is_companion_chat(ctx)):
        return tail
    out = []
    for item in tail:
        text = str(item or "").lstrip()
        if text.startswith(_PERSONA_FIRST_SHELVED_DYNAMIC_PREFIXES):
            continue
        out.append(item)
    return out


def trim_contract(contract: str, ctx: Any) -> str:
    """闲聊盘合同：开关开时去掉配额式子句；关时原样。"""
    if not (persona_first_enabled() and is_companion_chat(ctx)):
        return contract
    out = contract
    for clause in _PERSONA_FIRST_SHELVED_CONTRACT_CLAUSES:
        out = out.replace(clause, "")
    return out


def persona_first_keep(ctx: Any) -> bool:
    """这一轮是否要应用"人格先行"搁置（闲聊盘且开关开）。

    返回 False 表示**不搁置**（开关关 / 非闲聊盘）。调用方据此决定是否过滤段落，
    保证练习盘/素材盘/场景盘与开关关时的 prompt 逐字不变。
    """
    return bool(persona_first_enabled() and is_companion_chat(ctx))


# 合同文本里**相邻字面量**会拼成同一个列表元素（例如 [Turn Rhythm] 的结尾紧接
# [Stance - content]），所以按"列表元素"搁置会误伤人格段。这里按**段头区间**删：
# 从每个被搁置的段头开始，到下一个任意段头为止。段头集合 = 下面的服务/教学段 +
# 人格段（保留段），保证区间不会跨过人格段。
_PERSONA_FIRST_CUT_HEADS: tuple = (
    "[Order]", "[Frame]", "[Pragmatics]", "[Language]", "[Semantic Fidelity]",
    "[Precision]", "[Reality]", "[Continuity]", "[Roleplay Language]", "[Teaching] ",
    "[Reply Contract]", "[Conversation Rhythm]", "[Turn Rhythm]", "[Stance - content]",
    "[Non-Judgment - language]", "[Becoming]", "[Disposition]", "[No Meta]",
    "[Expression Teaching]", "[Anti-Parroting]",
)
_PERSONA_FIRST_CUT_SHELVED: tuple = (
    "[Order]", "[Frame]", "[Pragmatics]", "[Language]", "[Teaching] ",
    "[Reply Contract]", "[Conversation Rhythm]", "[Turn Rhythm]",
)


def cut_shelved_spans(text: str, ctx: Any) -> str:
    """闲聊盘且开关开：按段头区间删除服务/教学段，保留人格段。

    区间 = 被搁置段头 .. 下一个任意段头（或文末）。开关关或非闲聊盘时原样返回。
    """
    if not (persona_first_enabled() and is_companion_chat(ctx)):
        return text
    import re
    heads = sorted(
        ((text.find(h), h) for h in _PERSONA_FIRST_CUT_HEADS if text.find(h) >= 0),
        key=lambda x: x[0],
    )
    if not heads:
        return text
    spans = []
    for i, (pos, head) in enumerate(heads):
        if head not in _PERSONA_FIRST_CUT_SHELVED:
            continue
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        spans.append((pos, end))
    if not spans:
        return text
    out, cursor = [], 0
    for start, end in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
