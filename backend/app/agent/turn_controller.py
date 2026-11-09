"""Authoritative per-turn routing for the language tutor.

The controller does not write learner-facing prose.  It converts the latest
learner turn into one durable decision contract that every downstream planner,
fallback and reply validator must follow.  A turn may contain more than one
speech act; ``primary_frame`` only selects who speaks first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List


@dataclass(frozen=True)
class TurnDecision:
    version: str = "turn-controller-v3"
    acts: List[str] = field(default_factory=list)
    primary_frame: str = "general"
    response_sequence: List[str] = field(default_factory=list)
    speaker_position: str = "coach"
    advance_scene: bool = False
    resume_frame: str = ""
    replan_required: bool = False
    strong_reasoning_required: bool = False
    planning_route: str = "fast_path"
    routing_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
    response_obligations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TurnController:
    """Single source of truth for the current turn's interaction route."""

    LANGUAGE_QUERY = re.compile(
        r"(什么意思|啥意思|是.{0,12}意思吗|单词|短语|语法|时态|疑问句|"
        r"(?:是|是不是|算不算).{0,16}(?:过去式|过去分词|现在时|将来时)|"
        r"(?:过去式|过去分词|现在时|将来时).{0,10}(?:对吗|对吧|是吧|吗|么|怎么用|怎么说)|"
        r"发音|读音|重音|怎么读|读作|"
        r"英文怎么说|怎么用英文|怎么(?:问|说|表达)|如何(?:问|说|表达)|"
        r"(?:这里|这个场景|这种情况).{0,12}(?:用|说).{0,40}(?:还是|或者|和|哪个|哪种|更自然|对吗|可以吗)|"
        r"(?:这样|这个说法).{0,8}(?:说|问)?.{0,8}(?:自然|地道|对吗|可以吗|可不可以)|"
        r"\bwhat\s+does\b.{0,40}\bmean\b|\bhow\s+(?:do|can|should)\s+i\s+say\b|"
        r"\bwhat\s+can\s+i\s+say\b)",
        re.IGNORECASE,
    )
    PROCESS_FEEDBACK = re.compile(
        r"(你(?:刚才|现在|又).{0,20}(?:回复|回答|理解|重复|偏题|角色)|"
        r"你(?:好像|是不是).{0,18}(?:错乱|没听懂|没理解|理解错|搞错|偏题|重复)|"
        r"我(?:刚才|已经|都)?(?:跟你说了|跟你说过|说过).{0,36}|"
        r"为什么.{0,18}(?:回复|回答|重复|切换|打断)|"
        r"上下文|导演层|提示词|系统故障|语音故障|前端|后端|架构|"
        r"不要让我(?:跟读|重复)|先不要(?:跟读|练习)|教学方式|模板|套路|老是.{0,12}问|一直.{0,12}问)",
        re.IGNORECASE,
    )
    REPETITION_FEEDBACK = re.compile(
        r"\b(?:why\s+(?:do|are)\s+you\s+(?:keep\s+)?repeating|"
        r"you\s+keep\s+repeating(?:\s+yourself)?|"
        r"stop\s+repeating(?:\s+yourself)?|"
        r"don't\s+repeat(?:\s+that)?(?:\s+again)?|"
        r"you\s+already\s+(?:said|answered)\s+that)\b",
        re.IGNORECASE,
    )
    SCENE_CONTROL = re.compile(
        r"(你来当|你扮演|我来当|直接开始|重新开始|继续刚才|回到刚才|"
        r"换个场景|换一个场景|进入.{0,8}(?:场景|角色))",
        re.IGNORECASE,
    )
    INCOMPLETE = re.compile(
        r"(?:^|[，,。 ]])(?:呃|嗯|让我想想|等一下|我还没说完)(?:$|[，,。 ]])|"
        r"(?:然后|and|but|because)\s*$",
        re.IGNORECASE,
    )
    LATIN = re.compile(r"[A-Za-z]{2,}")
    DIRECT_ROLE_LINE = re.compile(
        r"(?:^|[.!?。！？]\s*)(?:hi|hello|excuse\s+me|please|"
        r"can\s+i|could\s+i|may\s+i|do\s+you|could\s+you|can\s+you|"
        r"i\s+(?:want|need|have|would|will|'ll)|my\s+)",
        re.IGNORECASE,
    )
    SIDECAR_FOLLOWUP = re.compile(
        r"(是吧|对吧|对吗|才对|哪个|哪一个|前面那个|刚才那个|"
        r"怎么读|发音|读音|重音|是.{0,10}还是|你的意思是)",
        re.IGNORECASE,
    )
    MEMORY_COMMITMENT = re.compile(
        r"(?:明确)?记一下|(?:帮我|给我)?记住|(?:请|要)记得|"
        r"(?:以后|下次|下一次|下回).{0,30}(?:多练|继续练|重点练|练习|学习|改进|纠正|关注|聚焦)|"
        r"(?:还|更)?想多练|想重点练",
        re.IGNORECASE,
    )
    SESSION_RECAP = re.compile(
        r"(?:我|我们).{0,5}(?:刚才|刚刚|前面|之前|后面).{0,12}(?:聊|说|提|问).{0,10}(?:什么|啥|哪些|内容)|"
        r"(?:刚才|刚刚|前面|之前|后面).{0,12}(?:聊|说|提|问)(?:了)?(?:什么|啥|哪些)|"
        r"(?:没有|没)(?:有)?(?:别的|其他)(?:的)?(?:内容)?(?:了)?[，,。.!！ ]*(?:是吧|吗|对吧)",
        re.IGNORECASE,
    )
    CONTEXTUAL_REFERENCE = re.compile(
        r"(我(?:问|说)的是|我的意思是|不是.{0,24}(?:而是|是)|"
        r"刚才(?:那个|那句|那里|说的|问的)|"
        r"前面(?:那个|那句|说的|问的)|"
        r"那个(?:词|短语|发音|意思|问题)|"
        r"在那里|在刚才.{0,12}(?:场景|对话))",
        re.IGNORECASE,
    )
    ASR_SELF_CORRECTION = re.compile(
        r"(?:我说的是|我是说|我刚才说的是).{0,45}(?:不是|而不是)|"
        r"(?:你听成|你识别成|刚才识别成).{0,35}(?:了|对吧|是吧)",
        re.IGNORECASE,
    )
    RESUME_AFTER_SIDECAR = re.compile(
        r"(回答完.{0,12}继续|说完.{0,12}继续|解释完.{0,12}继续|"
        r"然后.{0,8}(?:继续|回到|接着)|再.{0,8}(?:继续|回到)|"
        r"继续(?:刚才|之前|原来|上面)?.{0,12}(?:流程|场景|角色|对话))",
        re.IGNORECASE,
    )
    TOPIC_RELEASE = re.compile(
        r"(?:算了|先不说这个|这个不重要|不重要了|不聊这个了|换个话题吧|"
        r"先到这里|不用继续了)(?:[，,。.!！ ]|$)|"
        # Keep the English form anchored to the start of the utterance. This
        # recognizes a speaker releasing the current topic while avoiding
        # examples, reported speech, or questions about the phrase itself.
        # _clean_transcript removes standalone fillers but can leave their
        # sentence punctuation ("Mm. ..." -> ". ...").
        r"^\s*[.!?…]*\s*(?:(?:mm|hmm|well|okay|yeah)[,，.!！… ]+)*"
        r"(?:i(?:['’]m| am)\s+(?:(?:a bit|a little|kind of|pretty|really)\s+)?"
        r"tired of\s+(?:talking about|discussing)\s+(?:it|this|that)"
        r"(?:\s+(?:now|right now|anymore|for now))?|"
        r"i\s+don['’]t\s+want\s+to\s+talk\s+about\s+(?:it|this|that)"
        r"(?:\s+(?:anymore|now|right now|for now))?)\s*[.!?…]*\s*$",
        re.IGNORECASE,
    )
    FREE_TOPIC_SHIFT = re.compile(
        r"(?:想|我们|可以|来|随便).{0,10}(?:聊聊|聊一聊|聊|谈谈|谈一谈|说说).{1,80}|"
        r"(?:我最近|我刚|我之前|这两天).{0,10}(?:看|听|玩|读|关注|喜欢).{1,80}|"
        r"\b(?:let['’]?s|can\s+we|i\s+(?:want|would\s+like)\s+to)\s+"
        r"(?:talk|chat|discuss)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _has_spoken_semantic_uncertainty(text: str) -> bool:
        """Detect only strong ASR/disfluency evidence, never mere bad grammar.

        A learner is allowed to make grammatical mistakes.  This gate exists
        for turns where repeated fragments make the intended proposition
        unsafe to infer; those turns deserve a fresh semantic plan rather than
        a confident recast of a guessed meaning.
        """
        tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text or "")
        if len(tokens) < 6:
            return False
        repeated_fragment = bool(re.search(
            r"\b([A-Za-z]{2,})\b(?:[\s,，.。]+\1\b){1,}",
            text or "",
            re.IGNORECASE,
        ))
        restart_markers = len(re.findall(
            r"(?:\b(?:uh|um|er|you\s+know)\b|\.{2,}|…)",
            text or "",
            re.IGNORECASE,
        ))
        return repeated_fragment or restart_markers >= 2

    @staticmethod
    def _has_implicit_expression_gap(text: str) -> bool:
        """Detect a learner filling an unfinished English attempt with Chinese.

        This is not a generic "mixed language means correction" rule.  It only
        activates when the turn contains a substantial English attempt plus
        evidence that Chinese is standing in for a missing expression: an
        explicit phrasing cue, several spoken restarts, or an unfinished
        English fragment immediately followed by a meaningful Chinese clause.
        """
        value = " ".join(str(text or "").split()).strip()
        english_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", value)
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", value)
        if len(english_tokens) < 4 or not chinese_chunks:
            return False
        # 2026-09-22（plans/0922-07 §12.5）：门槛从「中文总字数 ≥4」降到「≥2」。
        # 原值把「医院」(2 字) /「充电器」(3 字) 这类**最常见的卡壳词**全挡在门外
        # ⇒ P20-② 的整套合同（acts=implicit_expression_support /
        #   sequence=[supply_natural_expression, respond_to_intended_meaning]）
        #   从未被激活，实测 A16 十遍 0/10。chinese_chunks 本身就要求 {2,}，
        # 所以 <2 等价于「一个中文词都没有」——与上面的 `not chinese_chunks` 同义，
        # 这行从此只是兜底，不再误杀两字词。
        if sum(len(chunk) for chunk in chinese_chunks) < 2:
            return False
        phrasing_cue = bool(re.search(
            r"(怎么说|用英文|英文(?:该|要|是|怎么)|说成|表达成|"
            r"(?:就是|也就是|我是想|我想)(?:说|表达)|"
            r"说(?:它|这|那|成)?是.{0,18}(?:英文|怎么)|"
            r"\bhow\s+(?:do|can|should)\s+i\s+say\b)",
            value,
            re.IGNORECASE,
        ))
        restart_count = len(re.findall(
            r"\b(?:uh|um|er)\b|\b([A-Za-z]{2,})\b(?:[\s,，.。]+\1\b)",
            value,
            re.IGNORECASE,
        ))
        unfinished_then_chinese = bool(re.search(
            r"(?:\b(?:the|a|an|is|are|was|were|which|how|more|most|best|"
            r"because|but|and|to|for)\b[\s,，]*){1,4}"
            # 2026-09-22（plans/0922-07 §12.5）：中文部分从 {4,} 降到 {2,}。
            # 「the 医院」这种「功能词 + 两字名词」正是要抓的形态；{4,} 只认
            # 「…because 我不太喜欢结局」那种长从句，反而漏掉最短最常见的缺口。
            r"[\u4e00-\u9fff]{2,}",
            value,
            re.IGNORECASE,
        ))
        return phrasing_cue or restart_count >= 2 or unfinished_then_chinese

    @classmethod
    def contextual_asr_resolution(
        cls, user_text: str, recent_dialogue: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Describe a short probable proper-noun ASR near miss using dialogue.

        The backend currently receives transcript text, not acoustic N-best
        hypotheses.  This packet therefore asks the reasoning layer to use a
        strong semantic constraint (for example, an immediately preceding
        question asking for a map or title) without rewriting stored evidence.
        """
        value = " ".join(str(user_text or "").split()).strip()
        recent_dialogue = list(recent_dialogue or [])[-6:]
        latin_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", value)
        if not latin_tokens or len(latin_tokens) > 4 or len(value) > 80:
            return {}
        if cls.DIRECT_ROLE_LINE.search(value) or cls.LANGUAGE_QUERY.search(value):
            return {}
        normalized = " ".join(token.lower() for token in latin_tokens)
        if normalized in {
            "ok", "okay", "yes", "no", "thank you", "thanks", "hello", "hi",
            "i know", "i see", "got it", "all right", "alright",
        }:
            return {}
        last_assistant = next((
            str(item.get("content") or "")
            for item in reversed(recent_dialogue)
            if str(item.get("role") or "") == "assistant"
        ), "")
        selection_prompt = bool(re.search(
            r"(哪(?:张|个|一张|一部|一个|支|队|位).{0,12}"
            r"(?:地图|图|剧|电影|角色|人物|队伍|选手|名字|型号)|"
            r"(?:地图|图|剧|电影|角色|人物|队伍|选手|名字|型号).{0,12}哪|"
            r"\b(?:which|what)\s+(?:map|show|movie|character|player|team|name|model)\b)",
            last_assistant,
            re.IGNORECASE,
        ))
        learner_challenge = bool(re.search(
            r"(这你都听不懂|你都不知道|我说的是|没听懂|听成)", value
        ))
        if not selection_prompt and not learner_challenge:
            return {}
        recent_text = " ".join(
            str(item.get("content") or "") for item in recent_dialogue
        )
        topic_hints = re.findall(
            r"\b(?:[A-Za-z]{2,}\d+[A-Za-z0-9]*|[A-Z]{2,6}|"
            r"[A-Za-z]{2,}\s+(?:II|IV|VI|\d+))\b",
            recent_text,
        )
        topic_hints = [
            item for item in topic_hints
            if item.lower() not in {"ai", "ok", "pm", "am"}
        ][-4:]
        return {
            "raw_transcript": value[:80],
            "recent_question": last_assistant[:220],
            "topic_hints": list(dict.fromkeys(topic_hints)),
            "resolution_rule": (
                "把它视为当前类别中的专有名词候选，不先按普通语法片段解释；"
                "若语境只有一个强候选，简短确认后继续；仍有多个候选时只问最短澄清问题。"
            ),
        }

    @staticmethod
    def _has_multiple_response_needs(text: str) -> bool:
        """Flag a real compound turn without treating every long monologue as a list.

        Spoken learners often join several requests with ``还有/另外/and``.
        The director is useful only when at least two actionable clauses are
        visible; length alone is not enough to justify an extra model call.
        """
        value = " ".join(str(text or "").split())
        if len(value) < 45:
            return False
        question_signals = len(re.findall(
            r"[?？]|(?:怎么|为什么|是不是|能不能|可不可以|什么意思|如何)|"
            r"\b(?:what|why|how|can\s+i|could\s+i|should\s+i|do\s+you)\b",
            value,
            re.IGNORECASE,
        ))
        addition_signals = len(re.findall(
            r"(?:还有|另外|以及|同时|第[一二三四1-4]个|然后|再一个)|"
            r"\b(?:also|another|and\s+then|besides|one\s+more)\b",
            value,
            re.IGNORECASE,
        ))
        explicit_requests = len(re.findall(
            r"(?:我想|我还想|我希望|需要|帮我|告诉我|解释|纠正|继续)|"
            r"\b(?:i\s+(?:want|need|hope)|please|tell\s+me|help\s+me)\b",
            value,
            re.IGNORECASE,
        ))
        return question_signals >= 2 or (
            addition_signals >= 1 and question_signals + explicit_requests >= 2
        )

    @classmethod
    def decide(
        cls,
        *,
        user_text: str,
        communicative_intent: str,
        scene_active: bool,
        scene_changed: bool,
        existing_plan: Dict[str, Any] | None,
        previous_decision: Dict[str, Any] | None = None,
        previous_frame: str = "",
        contextual_asr: Dict[str, Any] | None = None,
        material_tape: bool = False,
    ) -> TurnDecision:
        text = " ".join(str(user_text or "").split()).strip()
        previous_decision = previous_decision or {}
        acts: List[str] = []
        evidence: List[str] = []

        asr_self_correction = bool(cls.ASR_SELF_CORRECTION.search(text))
        # On a material tape, model-estimated lexical intent must not override
        # a substantive question about the source. Explicit language requests
        # remain routed by surface cues in LANGUAGE_QUERY; this flag only
        # suppresses the weaker inferred intent when the active job is content.
        explicit_language_query = (
            bool(cls.LANGUAGE_QUERY.search(text)) or asr_self_correction
            or (not material_tape and communicative_intent in {
                "lexical_clarification", "expression_evaluation", "role_expression_support",
            })
        )
        implicit_expression_gap = cls._has_implicit_expression_gap(text)
        # Multi-word meaning questions need a fresh semantic check even while a
        # durable role-play scene is being reused.  A cached scene blueprint can
        # preserve roles and steps, but it cannot decide whether a newly heard
        # phrase is idiomatic or an ASR near-miss.
        lexical_phrase_query = bool(re.search(
            r"\bwhat\s+does\s+[\"“']?[A-Za-z]+(?:\s+[A-Za-z]+){1,4}[\"”']?\s+mean\b|"
            r"[\"“']?[A-Za-z]+(?:\s+[A-Za-z]+){1,4}[\"”']?\s*(?:是|的)?\s*(?:什么意思|啥意思)",
            text,
            re.IGNORECASE,
        ))
        vague_language_request = bool(re.fullmatch(
            r"(?:what\s+can\s+i\s+say|我(?:现在)?(?:该|可以|能)怎么说)[?？.!。 ]*",
            text, re.IGNORECASE,
        ))
        memory_commitment = bool(cls.MEMORY_COMMITMENT.search(text)) or (
            communicative_intent == "memory_commitment"
        )
        session_recap = bool(cls.SESSION_RECAP.search(text)) or (
            communicative_intent == "session_recap"
        )
        repetition_feedback = bool(cls.REPETITION_FEEDBACK.search(text))
        process_feedback = bool(cls.PROCESS_FEEDBACK.search(text)) or repetition_feedback or communicative_intent in {
            "method_negotiation", "teaching_preference", "skip_current_item",
        }
        language_query = bool(
            explicit_language_query or (implicit_expression_gap and not process_feedback)
        )
        contextual_asr = dict(contextual_asr or {})
        if process_feedback or language_query:
            contextual_asr = {}
        scene_control = bool(cls.SCENE_CONTROL.search(text)) or communicative_intent in {
            "scenario_negotiation", "counterpart_turn_request",
        }
        target_language_present = bool(cls.LATIN.search(text))
        direct_role_line = bool(cls.DIRECT_ROLE_LINE.search(text))
        incomplete = bool(cls.INCOMPLETE.search(text)) and len(text) < 90
        contextual_reference = bool(cls.CONTEXTUAL_REFERENCE.search(text))
        resume_after_sidecar = bool(cls.RESUME_AFTER_SIDECAR.search(text))
        free_topic_shift = bool(cls.FREE_TOPIC_SHIFT.search(text))
        # When the same turn also opens a new subject, route that user-led
        # subject instead of swallowing the turn as a release-only response.
        topic_release = bool(cls.TOPIC_RELEASE.search(text)) and not free_topic_shift
        multiple_response_needs = cls._has_multiple_response_needs(text)
        semantic_uncertainty = cls._has_spoken_semantic_uncertainty(text)
        prior_frame = str(previous_decision.get("primary_frame") or previous_frame or "")
        resumable_role_frame = bool(
            prior_frame == "in_role"
            or str(previous_decision.get("resume_frame") or "") == "in_role"
        )
        sidecar_followup = bool(
            prior_frame == "language_sidecar"
            and len(text) <= 70
            and cls.SIDECAR_FOLLOWUP.search(text)
            and not process_feedback
            and not scene_control
        )

        if memory_commitment:
            acts.append("memory_commitment")
            evidence.append("explicit_memory_or_future_learning_commitment")
        elif session_recap:
            acts.append("session_recap")
            evidence.append("explicit_same_session_recap_request")
        elif process_feedback:
            acts.append("process_feedback")
            evidence.append("process_or_system_feedback")
        elif topic_release:
            acts.append("topic_release")
            evidence.append("learner_released_current_topic")
        elif free_topic_shift:
            acts.append("free_topic_shift")
            evidence.append("learner_explicitly_started_a_user_led_topic")
        if language_query:
            if implicit_expression_gap and not explicit_language_query:
                acts.append("implicit_expression_support")
                evidence.append("mixed_language_fills_unfinished_target_expression")
            else:
                acts.append("language_question")
                evidence.append("explicit_language_question")
            if asr_self_correction:
                acts.append("asr_self_correction")
                evidence.append("learner_corrected_previous_asr_interpretation")
        elif sidecar_followup:
            acts.append("language_followup")
            evidence.append("short_followup_to_previous_language_sidecar")
        if scene_control:
            acts.append("scene_control")
            evidence.append("explicit_scene_or_role_control")
        if contextual_asr:
            acts.append("contextual_asr_resolution")
            evidence.append("short_entity_like_transcript_after_selection_prompt")

        role_utterance = bool(
            scene_active
            and not memory_commitment
            and not session_recap
            and not topic_release
            and not free_topic_shift
            and (
                (
                    target_language_present
                    and (resumable_role_frame or scene_control)
                    and (not language_query or direct_role_line)
                )
                or communicative_intent in {
                    "counterpart_turn_request", "real_world_context",
                }
                or (
                    communicative_intent == "open_expression"
                    and resumable_role_frame
                    and not language_query and not sidecar_followup
                )
                or (
                    previous_frame in {"language_sidecar", "process_sidecar"}
                    and resumable_role_frame
                    and text and not language_query and not sidecar_followup
                    and not process_feedback and not memory_commitment and not session_recap
                )
            )
        )
        if role_utterance:
            acts.append("role_utterance")
            evidence.append("active_scene_with_role-directed_content")
        if (
            incomplete and not language_query and not process_feedback
            and not memory_commitment and not session_recap
        ):
            acts.append("still_speaking")
            evidence.append("incomplete_turn_signal")
        if not acts:
            acts.append("general_conversation")

        # Process feedback is answered outside the fictional role. A language
        # question is also a sidecar, but a role utterance in the same turn is
        # retained as the next obligation rather than discarded.
        if memory_commitment:
            primary_frame = "process_sidecar"
            speaker_position = "coach"
            sequence = ["confirm_memory_commitment"]
            obligations = [
                "只确认用户本轮明确要求记住的场景事实、后续学习偏好和语言重点；"
                "不得以场景角色继续索取小票、姓名、房号等业务信息"
            ]
        elif session_recap:
            primary_frame = "process_sidecar"
            speaker_position = "coach"
            sequence = ["answer_same_session_recap"]
            obligations = [
                "只依据same_session_context按真实先后回答本盘回顾；"
                "既包括场景事实，也包括语言侧问和用户明确留下的后续学习偏好，不能只复述场景主线"
            ]
        elif process_feedback:
            primary_frame = "process_sidecar"
            speaker_position = "coach"
            sequence = ["address_process_feedback"]
            if repetition_feedback:
                obligations = [
                    "自然回应用户指出的重复；不要复述已经回答过的事实，也不要重复整段道歉。"
                    "只有有新信息或用户明确要求展开时才补充，并保持用户当前话题。"
                ]
            else:
                obligations = ["直接回应用户指出的对话或教学过程问题，不机械道歉、不强行拉回练习"]
        elif topic_release:
            primary_frame = "general"
            speaker_position = "listener"
            sequence = ["acknowledge_topic_release"]
            obligations = [
                "简短接受用户不再继续当前点；不纠错、不追问、不强塞英文话口，也不替用户选择新话题"
            ]
        elif free_topic_shift:
            primary_frame = "general"
            speaker_position = "listener"
            sequence = ["follow_user_led_topic"]
            obligations = [
                "以用户刚主动开启的话题为当前主线；旧角色场景只保留为可恢复检查点，"
                "本轮不得继续冒充旧场景对方"
            ]
        elif language_query:
            primary_frame = "language_sidecar"
            speaker_position = "coach"
            if implicit_expression_gap and not explicit_language_query:
                sequence = ["supply_natural_expression", "respond_to_intended_meaning"]
                obligations = [
                    "识别学习者在中英混合里用中文补上的表达缺口；完整保留原意，"
                    "给一条符合当前话题的自然完整英文，不得只复述事实或改问无关问题",
                    "给完表达后用一句自然内容回应承接原话；不要求跟读，不把整段英文再逐句翻译成中文",
                ]
            else:
                sequence = ["answer_language_question"]
                obligations = [
                    "先精确回答本轮语音实际所指的单词、短语、语法或自然度问题；"
                    "前端转写词若与紧邻对话明显冲突，结合最近刚出现的语言项和语音近似纠偏，"
                    "不能机械相信错误转写；仍有多个合理候选时只确认最短歧义",
                    "解释短语时先保留词或短语的基本含义，再说明它在当前原句里的自然中文转述；"
                    "不能因为中文译法更灵活就声称原词已经变成另一个概念",
                ]
            if asr_self_correction:
                obligations.append(
                    "用户正在把上一轮误听的Y纠正为X；结合立即上一条用户原话恢复尚未回答的原问题，"
                    "直接回答原问题，不能只复述‘你说的是X，不是Y’；若原问题仍无法确定，只问一个简短澄清问题"
                )
            if lexical_phrase_query:
                obligations.append(
                    "先核验该短语在当前现实语境是否成立；若疑似ASR误听或并非常见搭配，"
                    "先确认可能的原短语，禁止为误听文本编造固定含义"
                )
            if vague_language_request:
                obligations.append(
                    "这句表达请求没有明确指向；先结合最近一轮确定唯一对象。若最近有多个候选，"
                    "只问一个极短澄清问题，不能复播上一轮答案"
                )
        elif sidecar_followup:
            primary_frame = "language_sidecar"
            speaker_position = "coach"
            sequence = ["answer_language_followup"]
            obligations = ["承接上一轮尚未结束的语言问题，只回答本轮确认点，不恢复角色、不另起练习"]
        elif incomplete:
            primary_frame = "listening"
            speaker_position = "listener"
            sequence = ["brief_listening_acknowledgement"]
            obligations = ["只作极短在听回应，不教学、不推进场景"]
        elif scene_active and (role_utterance or scene_control):
            primary_frame = "in_role"
            speaker_position = "counterpart"
            sequence = ["respond_as_counterpart"]
            obligations = ["以当前场景中的对方角色直接回应学习者最新内容"]
        else:
            primary_frame = "general"
            speaker_position = "coach"
            sequence = ["respond_to_latest_input"]
            obligations = ["直接回应最新输入，不复用上一轮答案"]
            if contextual_asr:
                sequence = ["resolve_contextual_entity", "continue_current_topic"]
                obligations = [
                    "当前短转写疑似专有名词近音；依据recent_dialogue、recent_question和topic_hints消歧，"
                    "不要把它当普通语法片段。只有一个强候选时简短确认并继续；否则只问一个最短澄清问题"
                ]

        resume_frame = ""
        if (
            primary_frame in {"language_sidecar", "process_sidecar"}
            and (resumable_role_frame or role_utterance or scene_control)
        ):
            resume_frame = "in_role"
        if (
            role_utterance
            and primary_frame in {"language_sidecar", "process_sidecar"}
            and not memory_commitment
            and not session_recap
        ):
            sequence.append("respond_as_counterpart")
            obligations.append("完成侧问后承接同一轮中的角色诉求，不能丢掉其中一个需求")
        elif (
            scene_active
            and resume_after_sidecar
            and primary_frame in {"language_sidecar", "process_sidecar"}
            and "respond_as_counterpart" not in sequence
        ):
            sequence.append("respond_as_counterpart")
            obligations.append("用户明确要求回答侧问后继续原流程；先完成侧问，再以原对方角色说下一句")

        multi_act = len(set(acts)) > 1
        routing_reasons: List[str] = []
        if scene_changed:
            routing_reasons.append("scene_changed")
        if not existing_plan:
            routing_reasons.append("missing_plan")
        if scene_control:
            routing_reasons.append("scene_control")
        if process_feedback:
            routing_reasons.append("process_feedback")
        if communicative_intent == "real_world_context":
            routing_reasons.append("real_world_context")
        if lexical_phrase_query:
            routing_reasons.append("lexical_phrase_verification")
        if memory_commitment:
            routing_reasons.append("memory_commitment")
        if session_recap:
            routing_reasons.append("session_recap")
        if multi_act:
            routing_reasons.append("multiple_speech_acts")
        if contextual_reference and (language_query or process_feedback or session_recap):
            routing_reasons.append("contextual_reference")
        if asr_self_correction:
            routing_reasons.append("asr_self_correction")
        if semantic_uncertainty:
            routing_reasons.append("spoken_semantic_uncertainty")
        if vague_language_request:
            routing_reasons.append("vague_language_request")
        if topic_release:
            routing_reasons.append("topic_release")
        if free_topic_shift:
            routing_reasons.append("free_topic_shift")
        if implicit_expression_gap and not explicit_language_query:
            routing_reasons.append("implicit_expression_gap")
        if contextual_asr:
            routing_reasons.append("contextual_asr_resolution")
        if multiple_response_needs:
            acts.append("multiple_response_needs")
            routing_reasons.append("multiple_response_needs")
            obligations.append(
                "先拆出用户本轮所有明确问题、请求和约束，按原出现顺序逐项覆盖；"
                "最多保留四项，不得只回答最后一句或最容易的一项"
            )

        general_target_language_turn = bool(
            primary_frame == "general"
            and target_language_present
            and len(cls.LATIN.findall(text)) >= 3
            and not topic_release
        )
        if general_target_language_turn:
            routing_reasons.append("general_target_language_turn")

        replan_required = bool(
            scene_changed
            or not existing_plan
            or scene_control
            or process_feedback
            or communicative_intent == "real_world_context"
            or lexical_phrase_query
            or memory_commitment
            or session_recap
            or multi_act
            or "contextual_reference" in routing_reasons
            or asr_self_correction
            or semantic_uncertainty
            or vague_language_request
            or topic_release
            or free_topic_shift
            or multiple_response_needs
            or (implicit_expression_gap and not explicit_language_query)
            or bool(contextual_asr)
        )
        confidence = 0.9
        if language_query and process_feedback:
            confidence = 0.72
        elif incomplete:
            confidence = 0.7
        elif semantic_uncertainty:
            confidence = 0.62
        elif contextual_asr:
            confidence = 0.68
        elif implicit_expression_gap and not explicit_language_query:
            confidence = 0.74
        if confidence < 0.8:
            routing_reasons.append("low_route_confidence")
        strong_reasoning_required = bool(replan_required or confidence < 0.8)
        planning_route = "strong_model" if strong_reasoning_required else "fast_path"

        return TurnDecision(
            acts=list(dict.fromkeys(acts)),
            primary_frame=primary_frame,
            response_sequence=sequence,
            speaker_position=speaker_position,
            advance_scene=primary_frame == "in_role",
            resume_frame=resume_frame,
            replan_required=replan_required,
            strong_reasoning_required=strong_reasoning_required,
            planning_route=planning_route,
            routing_reasons=list(dict.fromkeys(routing_reasons)),
            confidence=confidence,
            evidence=evidence,
            response_obligations=obligations,
        )

    @staticmethod
    def apply_to_plan(plan: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """Make the controller authoritative without replacing scene content."""
        if not plan or not decision:
            return
        frame = str(decision.get("primary_frame") or "general")
        reused_scene_plan = bool((plan.get("_meta") or {}).get("reused_scene_plan"))
        if reused_scene_plan:
            # These fields describe one learner turn, not the durable scene.
            # Keeping them while reusing a scene blueprint is what made old
            # corrections and old questions reappear on later turns.
            for key in (
                "content_need", "reply_draft", "target_expression",
                "learner_excerpt", "corrected_expression", "correction_note_zh",
            ):
                plan[key] = ""
            plan.update({
                "correction_needed": False,
                "correction_type": "none",
                "correction_priority": 0,
                "uncertain_spans": [],
                "unresolved_slots": [],
            })
            (plan.get("_meta") or {}).pop("deferred_correction", None)
        general_need = (
            "general_chat"
            if set(decision.get("acts") or []) & {"general_conversation", "free_topic_shift"}
            else str(plan.get("response_need") or "real_world_answer")
        )
        response_need = {
            "in_role": "roleplay_turn",
            "language_sidecar": "expression_support",
            "process_sidecar": "teaching_feedback",
            "listening": "listen_and_continue",
            "general": general_need,
        }.get(frame, str(plan.get("response_need") or "real_world_answer"))
        plan.update({
            "interaction_frame": frame,
            "response_need": response_need,
            "speaker_position": decision.get("speaker_position") or "coach",
            "advance_scene": bool(decision.get("advance_scene")),
            "resume_frame": str(decision.get("resume_frame") or ""),
            "turn_acts": list(decision.get("acts") or []),
            "response_sequence": list(decision.get("response_sequence") or []),
            "turn_controller": dict(decision),
        })
        obligations = list(decision.get("response_obligations") or [])
        existing = [] if reused_scene_plan else list(plan.get("response_obligations") or [])
        plan["response_obligations"] = list(dict.fromkeys(obligations + existing))[:8]
        if frame == "in_role":
            plan["user_should_speak_now"] = True
            plan["language_move"] = "roleplay_output"
        elif frame != "general":
            plan["user_should_speak_now"] = False
        # A draft produced for a different route is unsafe. Downstream
        # generation can still use a fresh director draft after this decision
        # has been included in the director request.
        draft_frame = str((plan.get("_meta") or {}).get("decision_frame") or "")
        if draft_frame and draft_frame != frame:
            plan["reply_draft"] = ""
