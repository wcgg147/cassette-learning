"""Estimate learner state from the current utterance *and* conversation context.

Intent labels are useful signals, not commands.  A question such as
``what does blended mean?`` can be lexical curiosity, a sign of overload, or
part of leaving the current task.  This module makes that uncertainty explicit
before the teaching policy chooses a move.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

try:
    from .intent_router import IntentResult, TutorIntent
except ImportError:
    from agent.intent_router import IntentResult, TutorIntent


@dataclass
class LearnerState:
    task_stage: str = "orient"
    communicative_intent: str = "open_expression"
    intent_confidence: float = 0.5
    cognitive_state: str = "ready"       # ready | local_confusion | overloaded
    affective_state: str = "steady"      # steady | frustrated | disengaged
    engagement: str = "engaged"          # engaged | uncertain | leaving
    autonomy_need: str = "medium"        # low | medium | high
    language_mode: str = "target_language"
    exit_likelihood: float = 0.0
    engagement_score: float = 0.6
    interaction_rhythm: str = "protect_flow"  # protect_flow | invite_depth | lower_load | restore_autonomy | close_gracefully
    consecutive_short_replies: int = 0
    recent_question_pressure: int = 0
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearnerStateEstimator:
    """Fast, explainable state estimator; no LLM call on the realtime path."""

    _EXIT_MARKERS = (
        "不聊了", "换个话题", "先这样", "到这里", "不想继续", "算了",
        "bye", "goodbye", "that's all", "stop", "switch topic",
    )
    _AUTONOMY_MARKERS = (
        "为什么", "不要", "别", "我想", "我不想", "你又", "你怎么",
        "rather", "instead", "don't", "stop asking",
    )
    _LEXICAL_PATTERNS = (
        "what does", "what is the meaning", "什么意思", "这个词", "这个单词",
        "怎么理解", "词义", "meaning of", "怎么说", "英文怎么", "英文是",
    )
    _EXPRESSION_EVALUATION_PATTERNS = (
        r"(?:我)?(?:这样|这么).{0,8}(?:说|问|表达).{0,8}(?:自然|地道|贴切|正确|对吗|可以吗|可不可以|有没有问题)",
        r"(?:这个|这句|我的).{0,12}(?:说法|表达|英语|英文).{0,8}(?:自然|地道|贴切|正确|对吗|可以吗|有没有问题)",
        r"(?:自然|地道|贴切|常用|native).{0,4}(?:吗|么|不)",
        r"\b(?:is|does)\s+.{1,80}\s+(?:sound\s+)?(?:natural|correct|right)\b",
    )
    # 2026-10-26：删掉 "这样是不是" —— 它是"我这样是不是有点没劲"这种自我怀疑句的
    # 子串，会把普通闲聊判成"用户在调整教学方式"，然后命中一句硬编码的规则复述
    # （"明白，之后按这个顺序：先在当前情境里给一个新词…"），用户看到的就是
    # 一段 prompt 原文。教学偏好要用户明确在说"教/引导/流程"，不是随口一问。
    _TEACHING_PREFERENCE_MARKERS = (
        "你应该", "你不应该", "你难道不应该", "你的引导", "这个引导",
        "这个问题问", "问题太宽泛", "不够精准", "后面你可能需要",
        "这种形式", "这个流程", "循序渐进",
    )
    _QUESTION_REJECTION_PATTERNS = (
        r"(?:这样|这么|这个|这种).{0,10}(?:问|追问).{0,10}(?:没意思|无聊|没必要|没用|不自然)",
        r"(?:别|不要).{0,10}(?:再|一直|老是|每次)?(?:问|追问)",
        r"(?:我)?(?:不是|不都|已经).{0,8}(?:说过|说了|回答过|讲过).{0,8}(?:了|吗|吧)",
        r"(?:没有|没).{0,8}(?:我)?(?:不是|不都|已经)?(?:说过|说了|回答过).{0,8}(?:吗|吧|了)",
        r"\b(?:stop|don['’]?t|do\s+not)\s+(?:keep\s+)?asking\b",
    )
    _MEMORY_COMMITMENT_PATTERNS = (
        r"(?:明确)?记一下", r"(?:帮我|给我)?记住", r"(?:请|要)记得",
        r"(?:以后|下次|下一次|下回).{0,30}(?:多练|继续练|重点练|练习|学习|改进|纠正|关注|聚焦)",
        r"(?:还|更)?想多练", r"想重点练", r"希望(?:以后|下次)?.{0,16}练",
        r"(?:remember|note).{0,24}(?:practice|learn|focus)",
    )
    _SESSION_RECAP_PATTERNS = (
        r"(?:我|我们).{0,5}(?:刚才|刚刚|前面|之前|后面).{0,12}(?:聊|说|提|问).{0,10}(?:什么|啥|哪些|内容)",
        r"(?:刚才|刚刚|前面|之前|后面).{0,12}(?:聊|说|提|问)(?:了)?(?:什么|啥|哪些)",
        r"(?:没有|没)(?:有)?(?:别的|其他)(?:的)?(?:内容)?(?:了)?[，,。.!！ ]*(?:是吧|吗|对吧)",
        r"(?:总结|回顾).{0,12}(?:刚才|这盘|本次|这次|前面)",
    )
    _SCENARIO_NEGOTIATION_MARKERS = (
        "角色扮演", "情景训练", "情景上面的训练", "场景训练", "场景对话",
        "把我当成", "你就当成", "你当服务员", "你来当", "我来当",
        "扮演一个角色", "角色上面", "实际的流程", "真实流程",
    )
    _OVERLOAD_MARKERS = (
        "完全听不懂", "根本听不懂", "怎么都听不懂", "太难", "跟不上", "太快", "一头雾水", "overwhelmed",
    )
    _LEARNING_CONTENT_MARKERS = (
        "电视剧", "英剧", "美剧", "电影", "视频", "播客", "字幕", "台词",
        "tv show", "series", "movie", "video", "podcast", "subtitle",
    )
    _FRUSTRATION_MARKERS = (
        "好蠢", "好烦", "根本不", "自说自话", "没用", "太机械", "annoying",
        "答非所问", "偏题", "跑题", "脱离话题", "脱离了", "没有半毛钱关系",
        "没什么关系", "又回到", "不是已经", "突然用全英文", "突然用纯英文",
    )

    _SKIP_ITEM_PATTERNS = (
        r"不想(?:说|读|跟读|重复|练)(?:这|这个|这句)?",
        r"(?:这句|这个|ok[,， ]*no problem).{0,10}(?:太简单|没必要|不用练|不用跟读)",
        r"(?:没必要|不需要|不用).{0,8}(?:跟读|重复|再说|练这句)",
        r"(?:跳过|先跳过)(?:这句|这个|这一题)?",
        r"我就不说了",
    )
    _REAL_WORLD_CONTEXT_PATTERNS = (
        r"(?:国外|当地).{0,18}(?:维修渠道|维修店|售后|授权维修|招牌)",
        r"(?:维修渠道|售后|官方维修|授权维修|二手维修店|维修店的招牌)",
        r"(?:哪里|怎么|如何).{0,12}(?:找|找到|搜索).{0,10}(?:维修店|维修点|售后)",
        r"(?:维修店|维修渠道).{0,12}(?:有没有|常见|实际|现实|靠谱么|靠谱吗)",
    )
    _COUNTERPART_TURN_PATTERNS = (
        r"(?:然后|那).{0,10}(?:对方|人家|他|工作人员).{0,10}(?:怎么|会怎么)(?:说|回复|回答)",
        r"(?:对方|人家|他|工作人员).{0,12}(?:同意|不同意|拒绝).{0,12}(?:怎么说|说什么|回复)",
        r"(?:先给我|应该先给我).{0,12}(?:对方|人家).{0,12}(?:说的话|台词|回复)",
    )

    def estimate(
        self,
        *,
        user_text: str,
        intent_result: IntentResult,
        learning_state: Any = None,
        history: List[Dict[str, Any]] | None = None,
        target_lang: str = "en",
        cefr: str = "",
    ) -> LearnerState:
        text = (user_text or "").strip()
        lower = text.lower()
        history = history or []
        signals: List[str] = []
        task_stage = getattr(learning_state, "stage", "orient") or "orient"

        recent_user = [
            str(item.get("content") or "").strip()
            for item in history[-8:]
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ][-3:]
        recent_assistant = [
            str(item.get("content") or "").strip()
            for item in history[-8:]
            if item.get("role") == "assistant" and str(item.get("content") or "").strip()
        ][-3:]
        consecutive_short = self._trailing_short_replies(recent_user + ([text] if text else []))
        # A bilingual reply often puts the question before its Chinese gloss,
        # so checking only the final character misses the actual interaction
        # pressure ("What did you do? 中文解释。").  Count turns containing a
        # question, not only turns ending in one.
        question_pressure = sum(1 for item in recent_assistant if re.search(r"[?？]", item))

        exit_likelihood = 0.0
        # “我没叫你换话题 / 我只是不想跟读这句”是在修正当前教学动作，
        # 不是退出当前主题。旧逻辑只看到“换个话题”四个字就清空任务。
        negated_topic_switch = bool(re.search(
            r"(?:没|没有|不是|并不是|别).{0,8}(?:叫|让|要你)?(?:我|你)?换(?:个)?话题|"
            r"换(?:个)?话题.{0,12}(?:不是|并不是|我只是)",
            lower,
        ))
        if intent_result.intent == TutorIntent.FAREWELL:
            exit_likelihood = 0.95
            signals.append("explicit_farewell")
        elif any(marker in lower for marker in self._EXIT_MARKERS) and not negated_topic_switch:
            exit_likelihood = 0.8
            signals.append("exit_or_switch_marker")

        skip_current_item = any(re.search(pattern, lower) for pattern in self._SKIP_ITEM_PATTERNS)
        if skip_current_item:
            signals.append("skip_current_item")

        real_world_context = any(
            re.search(pattern, lower) for pattern in self._REAL_WORLD_CONTEXT_PATTERNS
        )
        if real_world_context:
            signals.append("real_world_context_request")

        counterpart_turn_request = any(
            re.search(pattern, lower) for pattern in self._COUNTERPART_TURN_PATTERNS
        )
        if counterpart_turn_request:
            signals.append("counterpart_turn_request")

        lexical = any(marker in lower for marker in self._LEXICAL_PATTERNS)
        if lexical:
            signals.append("lexical_question")

        # This is not an ordinary correction opportunity.  The learner has
        # explicitly asked for a judgement, so it must override a stored
        # preference such as ``no_correction`` for this turn only.
        expression_evaluation = bool(
            re.search(r"[A-Za-z]", text)
            and any(re.search(pattern, lower, re.IGNORECASE)
                    for pattern in self._EXPRESSION_EVALUATION_PATTERNS)
        )
        if expression_evaluation:
            signals.append("explicit_expression_evaluation")

        teaching_preference = any(
            marker in lower for marker in self._TEACHING_PREFERENCE_MARKERS
        )
        if teaching_preference:
            signals.append("teaching_preference")

        question_rejection = any(
            re.search(pattern, lower, re.IGNORECASE)
            for pattern in self._QUESTION_REJECTION_PATTERNS
        )
        if question_rejection:
            signals.append("question_style_rejected")
            # The learner's explicit rejection is stronger evidence than a
            # short history window.  Treat it as sustained pressure so the
            # next reply cannot immediately ask another generic question.
            question_pressure = max(question_pressure, 2)

        memory_commitment = any(
            re.search(pattern, lower, re.IGNORECASE)
            for pattern in self._MEMORY_COMMITMENT_PATTERNS
        )
        if memory_commitment:
            signals.append("explicit_memory_commitment")

        session_recap = any(
            re.search(pattern, lower, re.IGNORECASE)
            for pattern in self._SESSION_RECAP_PATTERNS
        )
        if session_recap:
            signals.append("session_recap_request")

        restaurant_flow_terms = sum(
            1 for marker in ("餐厅", "服务员", "迎宾", "几位", "菜单", "带位", "座位", "扫码点单")
            if marker in lower
        )
        scenario_negotiation = (
            intent_result.intent == TutorIntent.SCENARIO_ROLEPLAY
            or any(marker in lower for marker in self._SCENARIO_NEGOTIATION_MARKERS)
            or restaurant_flow_terms >= 3
            or (
                getattr(learning_state, "roleplay_active", False)
                and intent_result.intent == TutorIntent.PRACTICE
                and any(marker in lower for marker in ("怎么练", "怎么对话", "怎么开始"))
            )
        )
        if scenario_negotiation:
            signals.append("scenario_negotiation")

        role_line_help = bool(
            getattr(learning_state, "scenario", "")
            and (
                re.search(r"(我|客人|顾客|这个角色).{0,18}(该|应该|要|可以)?怎么说", text)
                or re.search(r"(作为|扮演).{0,12}(客人|顾客).{0,12}怎么说", text)
            )
        )
        if role_line_help:
            signals.append("role_line_help")

        learning_content_difficulty = (
            any(marker in lower for marker in self._LEARNING_CONTENT_MARKERS)
            and any(marker in lower for marker in ("听不懂", "看不懂", "没听懂", "不理解", "can't understand", "cannot understand"))
        )
        if learning_content_difficulty:
            signals.append("learning_content_difficulty")

        overloaded = any(marker in lower for marker in self._OVERLOAD_MARKERS) or (
            consecutive_short >= 2 and task_stage in {"attempt", "scaffold", "repair", "transfer", "flow"}
        )
        frustrated = (
            intent_result.emotion_intensity == "high"
            or any(marker in lower for marker in self._FRUSTRATION_MARKERS)
        )
        if overloaded:
            signals.append("overload_signal")
        if consecutive_short >= 2:
            signals.append("consecutive_short_replies")
        if question_pressure >= 2:
            signals.append("recent_question_pressure")
        if frustrated:
            signals.append("frustration_signal")

        autonomy = "high" if (
            intent_result.intent in {TutorIntent.COMPLAINT, TutorIntent.META_QUESTION}
            or any(marker in lower for marker in self._AUTONOMY_MARKERS)
            or question_rejection
        ) else "medium"
        if autonomy == "high":
            signals.append("autonomy_request")
        preference_request = any(marker in lower for marker in (
            "先别纠错", "先听我", "别连续问", "换一个话题", "不要直接替我",
            "don't correct", "just listen", "switch topic", "don't answer for me",
        ))
        if preference_request:
            signals.append("explicit_interaction_preference")

        if exit_likelihood >= 0.75:
            communicative_intent = "exit_or_switch"
            engagement = "leaving"
        elif skip_current_item:
            communicative_intent = "skip_current_item"
            engagement = "engaged"
        elif memory_commitment:
            communicative_intent = "memory_commitment"
            engagement = "engaged"
        elif session_recap:
            communicative_intent = "session_recap"
            engagement = "engaged"
        elif real_world_context:
            communicative_intent = "real_world_context"
            engagement = "engaged"
        elif counterpart_turn_request:
            communicative_intent = "counterpart_turn_request"
            engagement = "engaged"
        elif scenario_negotiation:
            communicative_intent = "scenario_negotiation"
            engagement = "engaged"
        elif role_line_help:
            communicative_intent = "role_expression_support"
            engagement = "engaged"
        elif expression_evaluation:
            communicative_intent = "expression_evaluation"
            engagement = "engaged"
        elif teaching_preference:
            communicative_intent = "teaching_preference"
            engagement = "engaged"
        elif lexical and task_stage not in {"reflect"}:
            communicative_intent = "lexical_clarification"
            engagement = "engaged"
        elif (
            intent_result.intent in {TutorIntent.COMPLAINT, TutorIntent.META_QUESTION}
            or frustrated or preference_request or question_rejection
        ):
            communicative_intent = "method_negotiation"
            engagement = "uncertain"
        elif learning_content_difficulty and not overloaded:
            communicative_intent = "learning_difficulty"
            engagement = "engaged"
        elif intent_result.intent == TutorIntent.CONFUSION:
            communicative_intent = "clarification_request"
            engagement = "uncertain"
        elif intent_result.intent == TutorIntent.GREETING:
            communicative_intent = "session_entry"
            engagement = "engaged"
        else:
            communicative_intent = "open_expression"
            engagement = "engaged"

        cognitive = "overloaded" if overloaded else "local_confusion" if (
            (intent_result.intent == TutorIntent.CONFUSION and not learning_content_difficulty) or lexical
        ) else "ready"
        affective = "frustrated" if frustrated else "disengaged" if engagement == "leaving" else "steady"

        engagement_score = 0.7
        if exit_likelihood >= 0.75:
            engagement_score = 0.1
        elif frustrated:
            engagement_score = 0.25
        elif overloaded:
            engagement_score = 0.35
        elif len(text) >= 36 or len(recent_user) >= 2:
            engagement_score = 0.82
        if question_pressure >= 2:
            engagement_score = min(engagement_score, 0.45)

        if exit_likelihood >= 0.75:
            rhythm = "close_gracefully"
        elif frustrated or autonomy == "high":
            rhythm = "restore_autonomy"
        elif overloaded or question_pressure >= 2:
            rhythm = "lower_load"
        elif engagement_score >= 0.8 and task_stage in {"attempt", "flow", "transfer"}:
            rhythm = "invite_depth"
        else:
            rhythm = "protect_flow"

        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        has_target = self._has_target_language(text, target_lang)
        recent_target_only = [
            self._has_target_language(item, target_lang)
            and not any("\u4e00" <= ch <= "\u9fff" for ch in item)
            for item in recent_user
        ]
        sustained_target_language = len(recent_target_only) >= 2 and all(recent_target_only[-2:])
        recent_chinese_support = any(
            any("\u4e00" <= ch <= "\u9fff" for ch in item) for item in recent_user
        )
        if has_chinese and has_target:
            language_mode = "mixed_support"
        elif has_chinese:
            language_mode = "chinese_support"
        elif has_target and recent_chinese_support and not sustained_target_language:
            # One successful English sentence is evidence of an attempt, not
            # consent to remove the Chinese support used moments earlier.
            language_mode = "mixed_support"
            signals.append("recent_chinese_support_retained")
        else:
            language_mode = "target_language"

        # Two English turns show willingness to speak; they do not prove that
        # an A1/A2 learner understands unrestricted tutor English. Keep full
        # bilingual support until proficiency evidence, rather than the latest
        # input language alone, justifies removing it.
        normalized_cefr = str(cefr or "").upper()
        if normalized_cefr in {"A1", "A2", "UNKNOWN"} and language_mode == "target_language":
            language_mode = "mixed_support"
            signals.append("cefr_support_retained")

        # A lexical question immediately after an unfinished task normally
        # means "help me continue", not "abandon the task".  Conversely, an
        # exit marker always wins over its literal wording.
        if lexical and exit_likelihood >= 0.75:
            signals.append("lexical_question_deprioritized_by_exit")

        return LearnerState(
            task_stage=task_stage,
            communicative_intent=communicative_intent,
            intent_confidence=float(intent_result.confidence),
            cognitive_state=cognitive,
            affective_state=affective,
            engagement=engagement,
            autonomy_need=autonomy,
            language_mode=language_mode,
            exit_likelihood=exit_likelihood,
            engagement_score=engagement_score,
            interaction_rhythm=rhythm,
            consecutive_short_replies=consecutive_short,
            recent_question_pressure=question_pressure,
            signals=signals,
        )

    @staticmethod
    def _has_target_language(text: str, target_lang: str) -> bool:
        if target_lang == "ja":
            return any("\u3040" <= ch <= "\u30ff" for ch in text)
        return sum(ch.isascii() and ch.isalpha() for ch in text) >= 2

    @staticmethod
    def _trailing_short_replies(items: List[str]) -> int:
        count = 0
        for item in reversed(items):
            compact = "".join(item.split())
            # A short answer is only a rhythm signal after it repeats; it is
            # not a proficiency judgement and must never be stored as one.
            if len(compact) <= 12:
                count += 1
            else:
                break
        return count
