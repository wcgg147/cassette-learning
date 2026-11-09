from .style_parameters import StyleParameters


class StyleAwarePromptBuilder:
    """根据风格参数动态生成 system prompt"""

    def build(self, params: StyleParameters, context=None) -> str:
        parts = []

        # 1. 身份
        parts.append("[System] 你是用户的对话伙伴。")

        # 2. 语气描述
        tone_desc = self._describe_tone(params)
        if tone_desc:
            parts.append(f"[Tone] {tone_desc}")

        # 3. 教学策略
        teaching_desc = self._describe_teaching(params)
        if teaching_desc:
            parts.append(f"[Teaching] {teaching_desc}")

        # 4. 呈现规则
        presentation_desc = self._describe_presentation(params)
        if presentation_desc:
            parts.append(f"[Presentation] {presentation_desc}")

        # 5. 记忆（如果有）
        if context and hasattr(context, "recent_sessions") and context.recent_sessions:
            parts.append("[Memory] 最近会话：")
            for s in context.recent_sessions[:2]:
                summary = s.get("summary", "") if isinstance(s, dict) else ""
                if summary:
                    parts.append(f"  · {summary}")

        # 6. 当前策略（如果有）
        if context and hasattr(context, "latest_intent") and context.latest_intent:
            parts.append(f"[Strategy] 意图：{context.latest_intent}")

        return "\n".join(parts)

    def _describe_tone(self, params: StyleParameters) -> str:
        """根据参数生成语气描述"""
        descriptors = []

        if params.warmth > 0.7:
            descriptors.append("热情友好")
        elif params.warmth < 0.3:
            descriptors.append("冷静专业")

        if params.humor > 0.6:
            descriptors.append("偶尔幽默")

        if params.directness > 0.7:
            descriptors.append("直接了当")
        elif params.directness < 0.3:
            descriptors.append("委婉含蓄")

        if params.formality > 0.7:
            descriptors.append("正式礼貌")
        elif params.formality < 0.3:
            descriptors.append("随意自然")

        return "，".join(descriptors) if descriptors else "自然平衡"

    def _describe_teaching(self, params: StyleParameters) -> str:
        """根据参数生成教学策略描述"""
        parts = []

        correction_map = {
            "recast": "隐性重述正确形式",
            "explicit": "直接指出错误并解释",
            "elicitation": "引导用户自己发现错误",
        }
        parts.append(f"纠错方式：{correction_map.get(params.correction_style, '隐性重述')}")

        if params.challenge_level > 0.7:
            parts.append("问题较难，鼓励挑战")
        elif params.challenge_level < 0.3:
            parts.append("问题简单，建立信心")

        if params.error_tolerance > 0.7:
            parts.append("不急于纠错，先鼓励表达")

        return "；".join(parts)

    def _describe_presentation(self, params: StyleParameters) -> str:
        """根据参数生成呈现规则"""
        parts = []

        # preferred_length is a learned center, not a hard truncation limit.
        if params.preferred_length <= 25:
            parts.append("偏好简短回复：普通交流1-2句；需要解释时允许展开")
        elif params.preferred_length <= 50:
            parts.append("偏好适中回复：普通交流1-3句，随内容调整")
        else:
            parts.append("用户接受较充分解释，但口语交流仍保持轮流感")

        if params.language_mix > 0.7:
            parts.append("主要使用目标语言，难点用母语解释")
        elif params.language_mix < 0.3:
            parts.append("主要使用母语，穿插目标语言词汇")
        else:
            parts.append("母语和目标语言自然混合")

        if params.example_frequency > 0.6:
            parts.append("多给例子帮助理解")

        return "；".join(parts)
