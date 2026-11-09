"""Inspectable per-turn policy for a natural language tutor.

The large model is responsible for wording.  This module decides the smaller,
more important question first: what should the tutor do *this turn*?
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .intent_router import TutorIntent
except ImportError:
    from agent.intent_router import TutorIntent


@dataclass
class TutorPolicy:
    mode: str = "rapport"
    user_intent: str = "chitchat"
    emotion: str = "low"
    response_goal: str = "先回应内容，让用户愿意继续表达"
    teaching_act: str = "natural_extension"
    correction_budget: int = 0
    relationship_signal: str = ""
    conversation_hook: str = "围绕用户刚说的内容自然追问"
    response_length: str = "1-3个自然短句；解释时可更长"
    must_do: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)
    reason: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def _goal_hint(self) -> str:
        """回复目标：配置给了就用配置（plans/1017-02 §8.34），否则原样。"""
        try:
            from config import TUTOR_POLICY_GOAL_HINT
        except Exception:
            TUTOR_POLICY_GOAL_HINT = ""
        return TUTOR_POLICY_GOAL_HINT or self.response_goal

    def _hook_hint(self) -> str:
        """继续方式：默认永远是"围绕用户刚说的内容自然追问"（全库无人赋值），
        实测导致 87% 的回复在讲用户。配置非空时顶替。"""
        try:
            from config import TUTOR_POLICY_HOOK_HINT
        except Exception:
            TUTOR_POLICY_HOOK_HINT = ""
        return TUTOR_POLICY_HOOK_HINT or self.conversation_hook

    def _length_hint(self) -> str:
        """长度提示：配置给了就用配置，没给就退回原来的 `response_length`。

        plans/1017-02 §8.30：原值里带“解释时可更长”这个后门，而该字段全库无人赋值。
        """
        try:
            from config import TUTOR_REPLY_LENGTH_HINT
        except Exception:
            TUTOR_REPLY_LENGTH_HINT = ""
        return TUTOR_REPLY_LENGTH_HINT or self.response_length

    def to_prompt(self) -> str:
        lines = [
            "[Current Turn Policy - 仅指导当前一轮]",
            f"- 模式：{self.mode}",
            f"- 用户意图/情绪：{self.user_intent} / {self.emotion}",
            f"- 回复目标：{self._goal_hint()}",
            f"- 教学动作：{self.teaching_act}（本轮最多纠正 {self.correction_budget} 个关键错误）",
            f"- 继续方式：{self._hook_hint()}",
            f"- 长度：{self._length_hint()}",
        ]
        if self.relationship_signal:
            lines.append(f"- 关系记忆：{self.relationship_signal}")
        if self.must_do:
            lines.append("- 必须：" + "；".join(self.must_do))
        if self.must_avoid:
            lines.append("- 避免：" + "；".join(self.must_avoid))
        if self.constraints:
            lines.append("- 策略约束：" + str(self.constraints))
        lines.append("- 输出顺序：先回应用户真正表达的内容；需要时只做一个教学动作；最后留下自然的继续空间。")
        return "\n".join(lines)


class TutorPolicyPlanner:
    """Deterministic, low-latency next-best-tutor-move selector."""

    def plan(
        self,
        *,
        intent_result,
        errors: List[Dict[str, Any]],
        cefr: str,
        correction_style: str = "recast",
        relationship: Optional[Dict[str, Any]] = None,
        active_probe: Optional[Dict[str, Any]] = None,
        turn_count: int = 0,
    ) -> TutorPolicy:
        intent = intent_result.intent
        emotion = getattr(intent_result, "emotion_intensity", "low")
        relationship = relationship or {}
        policy = TutorPolicy(
            user_intent=intent.value,
            emotion=emotion,
            relationship_signal=self._relationship_signal(relationship),
            must_avoid=["复述整段用户原话", "连续抛出多个问题", "无依据地声称用户已掌握"],
        )

        if emotion == "high" or intent in (TutorIntent.COMPLAINT, TutorIntent.META_QUESTION):
            policy.mode = "listening"
            policy.response_goal = "先直接回应用户的不满或问题，恢复信任"
            policy.teaching_act = "affective_support"
            policy.correction_budget = 0
            policy.conversation_hook = "给出一个具体调整，并允许用户决定是否继续"
            policy.must_do = ["首句直接回应原因或感受", "承认具体问题并立即调整"]
            policy.must_avoid += ["立即拉回语言练习", "纠错", "空泛道歉"]
            policy.reason = "高情绪或元问题时，关系修复优先于教学推进"
            return policy

        if intent == TutorIntent.CONFUSION:
            policy.mode = "repair"
            policy.response_goal = "用更简单的方式让用户马上听懂"
            policy.teaching_act = "scaffolding"
            policy.correction_budget = 0
            policy.conversation_hook = "给一个极小的尝试或二选一提示"
            policy.must_do = ["换简单说法", "给一个具体例子"]
            policy.must_avoid += ["直接重复原解释", "堆语法术语"]
            policy.reason = "用户已明确困惑，应先降低认知负荷"
            return policy

        if intent == TutorIntent.GREETING:
            policy.mode = "rapport"
            policy.response_goal = "像认识用户的老师一样自然开场"
            policy.teaching_act = "memory_bridge" if policy.relationship_signal else "interest_discovery"
            policy.correction_budget = 0
            policy.conversation_hook = "引用上次开放话题，或给一个低压力的具体切入点"
            policy.must_do = ["简短、温暖、有变化"]
            policy.must_avoid += ["重复自我介绍", "宣读课程规则"]
            policy.reason = "寒暄首先建立连续关系"
            return policy

        if intent == TutorIntent.FAREWELL:
            policy.mode = "reflect"
            policy.response_goal = "自然收尾并留下下次可继续的线索"
            policy.teaching_act = "session_close"
            policy.correction_budget = 0
            policy.conversation_hook = "一句具体肯定加一个下次开放环"
            policy.must_avoid += ["开启新练习", "长篇总结"]
            policy.reason = "用户已经表达结束意图"
            return policy

        if active_probe:
            policy.mode = "practice"
            policy.response_goal = "收集一次可验证的学习证据"
            policy.teaching_act = "pushed_output" if active_probe.get("stage") == "transfer" else "concept_check"
            policy.correction_budget = 0
            policy.conversation_hook = active_probe.get("prompt", "请用户尝试一次")
            policy.must_do = ["等待用户作答后再反馈"]
            policy.must_avoid += ["提前给答案"]
            policy.reason = "当前存在已安排的学习检查"
            return policy

        if errors:
            repeated = any(int(e.get("count", 0) or 0) >= 2 for e in errors)
            beginner = (cefr or "A1").upper() in ("A1", "A2", "N5", "N4")
            policy.mode = "practice"
            policy.response_goal = "保持对话流，同时处理最影响表达的一个错误"
            policy.correction_budget = 1
            if correction_style == "elicitation" and not beginner and not repeated:
                policy.teaching_act = "elicitation"
                policy.conversation_hook = "给一个很短的自我修正提示"
            elif beginner or repeated:
                policy.teaching_act = "explicit_correction"
                policy.conversation_hook = "请用户只重说关键片段，随后回到原话题"
            else:
                policy.teaching_act = "recast"
                policy.conversation_hook = "沿着用户内容继续问一个真问题"
            policy.must_do = ["先接住语义，再纠正", "只选最关键的一个错误"]
            policy.must_avoid += ["错误清单", "把整轮变成语法讲解"]
            policy.reason = "检测到错误，但流利表达仍需优先保护"
            return policy

        policy.mode = "rapport" if intent == TutorIntent.CHITCHAT else "practice"
        policy.response_goal = "认真回应内容，并让用户多说一点"
        policy.teaching_act = "natural_extension"
        policy.correction_budget = 0
        policy.conversation_hook = "围绕内容问一个真正值得回答的问题；不必每轮都提问"
        policy.must_do = ["内容回应优先于教学展示"]
        policy.must_avoid += ["强行引入无关语言点", "机械给2-3个选项"]
        policy.reason = f"第{turn_count + 1}轮没有必须纠正的问题，保持自然交流"
        return policy

    @staticmethod
    def _relationship_signal(relationship: Dict[str, Any]) -> str:
        # 只对“明确未来任务”的 open_loop（存储时已要求含 明天/下周/面试/考试/
        # 准备/打算/要去 等标记）发出桥接提示，且只取任务子句。
        # 不再引用 last_topic 或整句复述旧对话——那会把上一会话的具体内容
        # （如“耳机换货”）带进全新会话。
        open_loop = (relationship.get("open_loop") or "").strip()
        if not open_loop:
            return ""
        if not re.search(
            r"(明天|下周|下个月|准备|打算|面试|考试|要去|"
            r"tomorrow|next week|plan to|going to|interview|exam)",
            open_loop, re.IGNORECASE,
        ):
            return ""
        return f"上次提到：{open_loop[:60]}"
