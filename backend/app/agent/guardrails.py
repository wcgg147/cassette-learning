from dataclasses import dataclass, field
from typing import List, Optional
import re


FORBIDDEN_IDENTITY_WORDS = [
    "豆包", "字节跳动", "抖音", "ByteDance", "Doubao",
    "Hermes", "Nous Research", "system prompt", "framework",
    "架构", "底层", "框架",
]

UNREALISTIC_EXPRESSIONS = [
    r"welcome\s*back",
    r"欢迎回来",
    r"又来了",
    r"好久不见",
    r"上次见面",
]

PRAGMATIC_UNREALISTIC_PATTERNS = [
    (
        r"(?:how\s+many\s+[\w\s]+\s+do\s+you\s+have\s+(?:here|in\s+stock|in\s+the\s+store)|你们这里有多少(?:瓶|个|支|份|件)[\u4e00-\u9fa5]+)",
        "pragmatic_unrealistic: asking staff for total store inventory is unnatural for a customer. Express purchase need (I'd like...) or ask availability (Do you have...) instead.",
    ),
]

TECHNICAL_LEAK_PATTERNS = [
    r"(hermes|nous\s+research)",
    r"system\s*prompt",
    r"framework",
    r"架构",
    r"底层",
    r"框架",
]

NEGATIVE_EMOTION_WORDS = [
    "根本不听", "自说自话", "没感觉", "变态", "好烦", "听不懂", "卡住了",
    "烦透了", "受不了", "气死了", "滚", "闭嘴", "讨厌死了",
    "不想学", "没意思", "无聊", "挫败", "沮丧", "头疼", "快疯了",
]

ACKNOWLEDGMENT_WORDS = ["理解", "说得对", "我明白", "嗯", "懂", "了解", "确实", "有道理"]

QUESTION_PATTERNS = [
    r"[？?]$",
    r"^(为什么|怎么|什么|哪里|谁|多少|几|how|what|why|where|when|who|is|are|do|does|can|could|would|should|will)\b",
]


@dataclass
class GuardrailResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    feedback: Optional[str] = None


class ResponseGuardrail:
    """Response guardrail: validate model output, reject with feedback if violations found."""

    def __init__(self, max_length: int = 240):
        self.max_length = max_length

    def check(
        self,
        response: str,
        intent: str,
        user_message: str,
        memory: dict = None,
        scene_context: dict = None,
    ) -> GuardrailResult:
        violations = []

        if len(response) > self.max_length:
            violations.append(
                f"too_long: {len(response)} chars. Max is {self.max_length}."
            )

        if self._is_repeating_user(response, user_message):
            violations.append("repeating_user: don't echo user's words.")

        if intent in ("greeting", "farewell") and len(response) > 180:
            violations.append(f"too_long_for_{intent}: keep the social turn concise.")

        if self._is_generic_question(response):
            violations.append("generic_question: suggest a specific topic, not 'what do you want?'")

        if self._is_apologetic(response):
            violations.append("too_apologetic: equal communication, not groveling.")

        emotional_violation = self.check_emotional_response(response, user_message)
        if emotional_violation:
            violations.append(emotional_violation)

        question_violation = self.check_direct_answer(response, user_message)
        if question_violation:
            violations.append(question_violation)

        identity_violation = self.check_identity_consistency(response)
        if identity_violation:
            violations.append(identity_violation)

        scenario_violation = self.check_scenario_realism(response, scene_context)
        if scenario_violation:
            violations.append(scenario_violation)

        pragmatic_violation = self.check_pragmatic_realism(response)
        if pragmatic_violation:
            violations.append(pragmatic_violation)

        if violations:
            feedback = "Rewrite:\n" + "\n".join(f"- {v}" for v in violations)
            return GuardrailResult(passed=False, violations=violations, feedback=feedback)

        return GuardrailResult(passed=True)

    def check_pragmatic_realism(self, response: str) -> Optional[str]:
        """Check if response expressions violate basic real-world common sense."""
        for pattern, warning in PRAGMATIC_UNREALISTIC_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return warning
        return None

    def check_emotional_response(self, response: str, user_message: str) -> Optional[str]:
        has_negative_emotion = any(word in user_message for word in NEGATIVE_EMOTION_WORDS)
        if not has_negative_emotion:
            return None
        has_acknowledgment = any(word in response for word in ACKNOWLEDGMENT_WORDS)
        if not has_acknowledgment:
            return "missing_acknowledgment: user expressed negative emotion, response must contain acknowledgment (理解/说得对/我明白/嗯)."
        return None

    def check_direct_answer(self, response: str, user_message: str) -> Optional[str]:
        is_question = any(re.search(p, user_message, re.IGNORECASE) for p in QUESTION_PATTERNS)
        if not is_question:
            return None
        if self._is_generic_question(response):
            return "avoid_generic_question: user asked a question, provide a direct answer instead of generic response."
        return None

    def check_identity_consistency(self, response: str) -> Optional[str]:
        for word in FORBIDDEN_IDENTITY_WORDS:
            if word in response:
                return f"identity_violation: response contains forbidden word '{word}'. Identity must always be '语言外教'."
        for pattern in TECHNICAL_LEAK_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return (
                    f"identity_violation: response contains technical leak '{match.group()}'. "
                    "Internal framework/system terms must never appear in user-facing output."
                )
        return None

    def _is_repeating_user(self, response: str, user_message: str) -> bool:
        user_words = set(re.findall(r'[\w\u4e00-\u9fff]+', user_message.lower()))
        response_words = set(re.findall(r'[\w\u4e00-\u9fff]+', response.lower()))
        # Exclude very common greetings from the check
        common_words = {"hi", "hello", "hey", "你好", "嗨", "哈喽"}
        user_words -= common_words
        response_words -= common_words
        if not user_words:
            return False
        overlap = len(user_words & response_words) / len(user_words)
        return overlap > 0.6

    def _is_generic_question(self, response: str) -> bool:
        patterns = [
            r"(你想聊什么|你想练什么|今天想聊|今天练什么|what\s*do\s*you\s*want|how\s*about\s*your)",
            r"(今天过得怎么样|你今天怎么样|how'?s\s*your\s*day|what'?s\s*on\s*your\s*mind)",
        ]
        return any(re.search(p, response, re.IGNORECASE) for p in patterns)

    def _is_apologetic(self, response: str) -> bool:
        patterns = [
            r"(都是我的错|我真的非常非常抱歉|万分抱歉)",
            r"(i(?:'m| am)\s+so\s+so\s+sorry|it'?s\s+all\s+my\s+fault)",
        ]
        return any(re.search(p, response, re.IGNORECASE) for p in patterns)

    def check_scenario_realism(self, response: str, scene_context: dict = None) -> Optional[str]:
        """Task 8: Check if response expressions match the established scene context."""
        if not scene_context or not scene_context.get("scene_context_set"):
            return None
        for pattern in UNREALISTIC_EXPRESSIONS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                active_scene = scene_context.get("active_scene", "该场景")
                return (
                    f"scenario_unrealistic: response contains '{match.group()}' "
                    f"which implies prior familiarity inconsistent with '{active_scene}' scene. "
                    "Use a neutral greeting consistent with the scene."
                )
        return None
