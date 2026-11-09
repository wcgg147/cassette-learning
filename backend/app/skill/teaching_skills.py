"""
Skill System - Methodologies as injectable tools.

参考架构：
- education-agent-skills: 包含165个基于证据的教学技能，覆盖20个领域
- agentskills.io: 开放技能标准，每个技能是一个包含 SKILL.md 的文件夹

架构设计：
┌──────────────────────────────────────────────────────────┐
│  多维度技能匹配引擎                                       │
│  ├── 等级匹配（统一 proficiency 标准）                    │
│  ├── 场景匹配（scene_tags）                              │
│  ├── 错误类型匹配                                        │
│  ├── 学习目标匹配（learning_objectives）                  │
│  └── 证据基础（evidence_sources）                        │
├──────────────────────────────────────────────────────────┤
│  统一 Proficiency 标准                                   │
│  ├── CEFR（欧洲标准）：A1, A2, B1, B2, C1, C2            │
│  ├── JLPT（日本语能力测试）：N5, N4, N3, N2, N1          │
│  └── 映射关系：N5→A1, N4→A2, N3→B1, N2→B2, N1→C1        │
└──────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
from abc import ABC, abstractmethod


class ProficiencyLevel(Enum):
    """
    统一的语言能力等级标准。

    使用 CEFR（欧洲语言共同参考框架）作为统一标准，
    JLPT（日本语能力测试）通过映射转换为 CEFR 等级。

    映射关系（参考行业通用对照）：
    - N5 → A1（入门级）
    - N4 → A2（初级）
    - N3 → B1（中级）
    - N2 → B2（中高级）
    - N1 → C1（高级）
    """

    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"

    @classmethod
    def from_jlpt(cls, jlpt: str) -> "ProficiencyLevel":
        """
        将 JLPT 等级转换为 CEFR 等级。

        Args:
            jlpt: JLPT 等级字符串（"N5", "N4", "N3", "N2", "N1"）

        Returns:
            对应的 CEFR 等级
        """
        mapping = {
            "N5": cls.A1,
            "N4": cls.A2,
            "N3": cls.B1,
            "N2": cls.B2,
            "N1": cls.C1,
        }
        return mapping.get(jlpt, cls.A1)

    @classmethod
    def parse(cls, level: str) -> "ProficiencyLevel":
        """
        解析等级字符串，支持 CEFR 和 JLPT 格式。

        Args:
            level: 等级字符串（如 "A1", "N5", "B2"）

        Returns:
            对应的 ProficiencyLevel 枚举值
        """
        try:
            return cls(level)
        except ValueError:
            return cls.from_jlpt(level)

    def is_compatible(self, other: "ProficiencyLevel") -> bool:
        """
        检查两个等级是否兼容（当前等级 >= 目标等级）。

        Args:
            other: 目标等级

        Returns:
            是否兼容
        """
        order = [self.A1, self.A2, self.B1, self.B2, self.C1, self.C2]
        return order.index(self) >= order.index(other)


class LearningObjective(Enum):
    """
    学习目标枚举。

    参考 education-agent-skills 的分类体系：
    - 语法掌握（Grammar Mastery）
    - 词汇扩展（Vocabulary Expansion）
    - 口语流利度（Speaking Fluency）
    - 听力理解（Listening Comprehension）
    - 阅读理解（Reading Comprehension）
    - 写作能力（Writing Ability）
    - 发音纠正（Pronunciation Correction）
    - 会话策略（Conversation Strategies）
    """

    GRAMMAR_MASTERY = "grammar_mastery"
    VOCABULARY_EXPANSION = "vocabulary_expansion"
    SPEAKING_FLUENCY = "speaking_fluency"
    LISTENING_COMPREHENSION = "listening_comprehension"
    READING_COMPREHENSION = "reading_comprehension"
    WRITING_ABILITY = "writing_ability"
    PRONUNCIATION_CORRECTION = "pronunciation_correction"
    CONVERSATION_STRATEGIES = "conversation_strategies"
    SELF_CORRECTION = "self_correction"


class EvidenceSource(str, Enum):
    """
    证据来源枚举。

    每个教学技能都标注了教育学/语言学的研究依据，
    参考 education-agent-skills 的证据基础体系。
    """

    COMMUNICATIVE_LANGUAGE_TEACHING = "communicative_language_teaching"
    TASK_BASED_LEARNING = "task_based_learning"
    NATURAL_APPROACH = "natural_approach"
    COGNITIVE_CODE_LEARNING = "cognitive_code_learning"
    SCAFFOLDING = "scaffolding"
    COMPREHENSIBLE_INPUT = "comprehensible_input"
    RECAST_FEEDBACK = "recast_feedback"
    SELF_CORRECTION = "self_correction"
    SHADOWING = "shadowing"
    CONTRASTIVE_ANALYSIS = "contrastive_analysis"


@dataclass
class TeachingSkill:
    """
    教学技能类 - 遵循 agentskills.io 开放技能标准。

    标准格式：每个技能是一个包含 SKILL.md 的文件夹，
    内有 name/description frontmatter。

    多维度匹配属性：
    - proficiency: 统一的能力等级标准
    - scene_tags: 教学场景标签
    - learning_objectives: 学习目标
    - evidence_sources: 证据基础
    - error_types: 针对的错误类型
    """

    name: str
    description: str
    instruction: str
    example_dialogue: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    proficiency: ProficiencyLevel = ProficiencyLevel.A1
    scene_tags: List[str] = field(default_factory=list)
    target_lang: str = "any"
    boost_score: float = 0.0
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    evidence_sources: List[EvidenceSource] = field(default_factory=list)
    error_types: List[str] = field(default_factory=list)

    def to_system_prompt_block(self) -> str:
        block = f"[Skill: {self.name}] {self.instruction}"
        if self.example_dialogue:
            block += f"\n示例: {self.example_dialogue}"
        return block

    def to_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": f"use_skill_{self.name}",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trigger": {
                            "type": "string",
                            "description": "The specific situation that prompted using this skill",
                            "enum": ["error_detected", "user_confused", "review_needed", "natural_flow"],
                        },
                        "context": {
                            "type": "string",
                            "description": "Brief context about what the user said or did",
                        },
                    },
                    "required": ["trigger"],
                },
            },
        }

    def to_tool_result(self, trigger: str, context: str) -> str:
        return json.dumps({
            "skill": self.name,
            "instruction": self.instruction,
            "example": self.example_dialogue,
            "trigger": trigger,
            "context": context,
        }, ensure_ascii=False)

    def to_skill_md(self) -> str:
        """
        生成符合 agentskills.io 标准的 SKILL.md 格式内容。

        Returns:
            SKILL.md 格式的字符串
        """
        return f"""---
name: {self.name}
description: {self.description}
proficiency: {self.proficiency.value}
target_lang: {self.target_lang}
learning_objectives: {', '.join([lo.value for lo in self.learning_objectives])}
evidence_sources: {', '.join([es.value for es in self.evidence_sources])}
scene_tags: {', '.join(self.scene_tags)}
---

## Instruction

{self.instruction}

## Example

{self.example_dialogue}
"""


class SkillMatcher(ABC):
    """
    技能匹配器抽象基类。

    参考 education-agent-skills 的设计，支持多种匹配策略。
    """

    @abstractmethod
    def match(self, skill: TeachingSkill, context: Dict[str, Any]) -> float:
        """
        计算技能与上下文的匹配度。

        Args:
            skill: 教学技能
            context: 匹配上下文

        Returns:
            匹配度分数（0.0-1.0）
        """
        pass


class ProficiencyMatcher(SkillMatcher):
    """
    等级匹配器。

    根据用户的语言能力等级匹配技能。
    """

    def match(self, skill: TeachingSkill, context: Dict[str, Any]) -> float:
        user_proficiency = context.get("proficiency")
        if not user_proficiency:
            return 0.5

        if isinstance(user_proficiency, str):
            user_proficiency = ProficiencyLevel.parse(user_proficiency)

        if skill.proficiency == user_proficiency:
            return 1.0
        if user_proficiency.is_compatible(skill.proficiency):
            return 0.8
        return 0.2


class SceneMatcher(SkillMatcher):
    """
    场景匹配器。

    根据当前教学场景匹配技能。
    """

    def match(self, skill: TeachingSkill, context: Dict[str, Any]) -> float:
        scene = context.get("scene")
        if not scene or not skill.scene_tags:
            return 0.5

        if scene in skill.scene_tags:
            return 1.0
        for tag in skill.scene_tags:
            if tag in scene or scene in tag:
                return 0.6
        return 0.1


class ErrorTypeMatcher(SkillMatcher):
    """
    错误类型匹配器。

    根据检测到的错误类型匹配技能。
    """

    def match(self, skill: TeachingSkill, context: Dict[str, Any]) -> float:
        error_types = context.get("error_types", [])
        if not error_types or not skill.error_types:
            return 0.5

        matching_errors = set(error_types) & set(skill.error_types)
        if matching_errors:
            return 1.0

        for error_type in error_types:
            error_base = error_type.lower().replace("ja_", "").replace("en_", "")
            if any(error_base in et.lower() for et in skill.error_types):
                return 0.7
            if error_base in skill.name.lower():
                return 0.6
        return 0.1


class LearningObjectiveMatcher(SkillMatcher):
    """
    学习目标匹配器。

    根据会话目标匹配技能。
    """

    def match(self, skill: TeachingSkill, context: Dict[str, Any]) -> float:
        session_goals = context.get("session_goals", [])
        if not session_goals or not skill.learning_objectives:
            return 0.5

        matching_goals = set(session_goals) & {lo.value for lo in skill.learning_objectives}
        if matching_goals:
            return 1.0
        return 0.3


class MultiDimensionSkillMatcher:
    """
    多维度技能匹配引擎。

    综合使用多种匹配器，按照权重计算最终匹配度。

    匹配维度：
    1. 等级匹配（proficiency）- 权重: 0.3
    2. 场景匹配（scene）- 权重: 0.25
    3. 错误类型匹配（error_types）- 权重: 0.25
    4. 学习目标匹配（session_goals）- 权重: 0.2

    参考 education-agent-skills 的多维度匹配设计。
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "proficiency": 0.3,
            "scene": 0.25,
            "error_types": 0.25,
            "session_goals": 0.2,
        }
        self.matchers: Dict[str, SkillMatcher] = {
            "proficiency": ProficiencyMatcher(),
            "scene": SceneMatcher(),
            "error_types": ErrorTypeMatcher(),
            "session_goals": LearningObjectiveMatcher(),
        }

    def calculate_score(self, skill: TeachingSkill, context: Dict[str, Any]) -> float:
        """
        计算技能的综合匹配分数。

        Args:
            skill: 教学技能
            context: 匹配上下文

        Returns:
            综合匹配分数（0.0-1.0）
        """
        total_score = 0.0
        total_weight = 0.0

        for matcher_name, matcher in self.matchers.items():
            weight = self.weights.get(matcher_name, 0.0)
            if weight > 0:
                score = matcher.match(skill, context)
                total_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0
        return total_score + skill.boost_score

    def select_skills(
        self,
        skills: List[TeachingSkill],
        context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[TeachingSkill]:
        """
        选择最匹配的技能。

        Args:
            skills: 技能列表
            context: 匹配上下文
            top_k: 返回的技能数量

        Returns:
            按匹配度排序的技能列表
        """
        scored_skills = [
            (skill, self.calculate_score(skill, context))
            for skill in skills
        ]

        scored_skills.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, score in scored_skills[:top_k]]


PRESET_SKILLS: Dict[str, TeachingSkill] = {}


def _register(skill: TeachingSkill):
    PRESET_SKILLS[skill.name] = skill
    return skill


_register(TeachingSkill(
    name="recast_feedback",
    description="Naturally reformulate user's incorrect sentence correctly without pointing out the mistake. Best for fluency-focused conversation.",
    instruction="When the user makes a grammar error, naturally reformulate their sentence correctly in your response. Do NOT explicitly say 'you made an error'. Just model the correct form.",
    example_dialogue='User: "I goed to the store yesterday." -> Tutor: "Oh, you went to the store? What did you buy?"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "free_talk", "daily_life"],
    target_lang="en",
    learning_objectives=[LearningObjective.SPEAKING_FLUENCY, LearningObjective.GRAMMAR_MASTERY],
    evidence_sources=[EvidenceSource.RECAST_FEEDBACK, EvidenceSource.NATURAL_APPROACH],
    error_types=["en_grammar", "en_verb_conjugation"],
))

_register(TeachingSkill(
    name="elicitation_en",
    description="Prompt user to self-correct by pausing or asking a clarifying question. Best for B1+ learners who know the rule but slip.",
    instruction="When the user makes a minor error, prompt them to self-correct. Pause or ask 'Sorry, ...?' to make them notice the gap.",
    example_dialogue='User: "He go to school." -> Tutor: "Sorry, he...?" (waiting for user to try "he goes")',
    proficiency=ProficiencyLevel.B1,
    scene_tags=["en", "grammar_focus"],
    target_lang="en",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY, LearningObjective.SELF_CORRECTION],
    evidence_sources=[EvidenceSource.SELF_CORRECTION, EvidenceSource.SCAFFOLDING],
    error_types=["en_grammar", "en_subject_verb_agreement"],
))

_register(TeachingSkill(
    name="cn_ja_contrast",
    description="Use Chinese-Japanese contrast to highlight kanji meaning differences. Essential for Chinese learners who see familiar characters.",
    instruction="When encountering a Japanese word written in kanji that means something different from Chinese, compare and contrast explicitly. List the Chinese meaning vs Japanese meaning.",
    example_dialogue='AI: "注意：中文的「勉強」是勉强的意思，但日语的「勉強（べんきょう）」是学习的意思。写法一样，意思完全不同。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["ja", "vocabulary", "kanji"],
    target_lang="ja",
    learning_objectives=[LearningObjective.VOCABULARY_EXPANSION],
    evidence_sources=[EvidenceSource.CONTRASTIVE_ANALYSIS],
    error_types=["ja_vocabulary", "ja_kanji"],
))

_register(TeachingSkill(
    name="particle_drill",
    description="Correct Japanese particle errors (は/が/に/で/を) and explain the rule in Chinese. Best for N5-N4 learners.",
    instruction="When the user makes a particle error, model the correct usage naturally in your response. Then briefly explain the rule in Chinese. Particles to watch: は/が (topic vs subject), に/で (destination vs action place), を (object).",
    example_dialogue='User: "学校に勉強する" -> Tutor: "学校で勉強するんですね。记住：动作发生的场所用「で」，移动的目的地用「に」。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["ja", "grammar_focus", "particles"],
    target_lang="ja",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY],
    evidence_sources=[EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING, EvidenceSource.COGNITIVE_CODE_LEARNING],
    error_types=["ja_particle", "ja_ha_ga", "ja_ni_de"],
))

_register(TeachingSkill(
    name="te_form_pattern",
    description="Teach Japanese te-form (て形) through common daily patterns. Focus on godan verb sound changes.",
    instruction="When encountering te-form errors, teach the correct conjugation pattern. Group verbs by ending: う→って, む→んで, く→いて, ぐ→いで, etc. Then drill with common patterns: てください, ています, てもいいですか.",
    example_dialogue='User: "買いて" -> Tutor: "買う的て形是買って哦。う结尾的五段动词变成って。再来一次：これを買ってください。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["ja", "grammar_focus", "te_form"],
    target_lang="ja",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY],
    evidence_sources=[EvidenceSource.COGNITIVE_CODE_LEARNING, EvidenceSource.SCAFFOLDING],
    error_types=["ja_verb_conjugation", "ja_te_form"],
))

_register(TeachingSkill(
    name="keigo_scaffolding",
    description="Introduce Japanese honorific forms (敬語/謙譲語) gradually. Start with です/ます, then introduce special verbs in context.",
    instruction="Introduce honorific (敬語) and humble (謙譲語) forms gradually. Start with です/ます体. Only introduce special verbs (いらっしゃる, おっしゃる, 申す, 致す) when the context naturally calls for it.",
    example_dialogue='Tutor: "对朋友可以说「先生がそう言った」，但对老师本人应该说「先生がそうおっしゃいました」。おっしゃる是言う的尊敬语。"',
    proficiency=ProficiencyLevel.A2,
    scene_tags=["ja", "keigo", "roleplay"],
    target_lang="ja",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY, LearningObjective.CONVERSATION_STRATEGIES],
    evidence_sources=[EvidenceSource.SCAFFOLDING, EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING],
    error_types=["ja_keigo", "ja_honorific"],
))

_register(TeachingSkill(
    name="transitivity_pairs",
    description="Teach Japanese transitive/intransitive verb pairs (他動詞/自動詞) when user confuses them.",
    instruction="When the user confuses transitive and intransitive verbs, explain the pair clearly. Key pairs: 開ける(他)/開く(自), 消す(他)/消える(自), 入れる(他)/入る(自). Rule: を + transitive, が + intransitive.",
    example_dialogue='Tutor: "ドアが開いた（自动，门自己开了）。ドアを開けた（他动，有人开了门）。が提示主体，を提示宾语。"',
    proficiency=ProficiencyLevel.A2,
    scene_tags=["ja", "grammar_focus", "transitivity"],
    target_lang="ja",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY],
    evidence_sources=[EvidenceSource.COGNITIVE_CODE_LEARNING, EvidenceSource.CONTRASTIVE_ANALYSIS],
    error_types=["ja_transitivity", "ja_verb_transitive"],
))

_register(TeachingSkill(
    name="shadow_repeat",
    description="Model correct Japanese pronunciation with natural rhythm. Have user shadow-repeat for muscle memory.",
    instruction="When demonstrating a sentence, break it down by rhythm groups. Have the user shadow (repeat) at natural speed. Focus on pitch accent and natural rhythm, not just individual sounds.",
    example_dialogue='Tutor: "跟我重复三遍：昨日、友達と映画を見に行きました。きのう、ともだちと、えいがを、みにいきました。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["ja", "pronunciation", "free_talk"],
    target_lang="ja",
    learning_objectives=[LearningObjective.PRONUNCIATION_CORRECTION, LearningObjective.SPEAKING_FLUENCY],
    evidence_sources=[EvidenceSource.SHADOWING, EvidenceSource.COMPREHENSIBLE_INPUT],
    error_types=["ja_pronunciation", "ja_pitch_accent"],
))

_register(TeachingSkill(
    name="conversation_gambits",
    description="Teach natural Japanese conversation fillers (相槌) to make speech sound more fluent.",
    instruction="Introduce natural Japanese fillers naturally in conversation: そうですね、なるほど、ええと、あのう、そうなんですか。Model when and how to use each one.",
    example_dialogue='Tutor: "日本人说话时经常用「あのう」来停顿思考，像中文的\'那个\'。ええと是卡壳时说。なるほど表示\'原来如此\'。自然地用起来吧！"',
    proficiency=ProficiencyLevel.A2,
    scene_tags=["ja", "conversation", "free_talk"],
    target_lang="ja",
    learning_objectives=[LearningObjective.CONVERSATION_STRATEGIES, LearningObjective.SPEAKING_FLUENCY],
    evidence_sources=[EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING, EvidenceSource.TASK_BASED_LEARNING],
    error_types=["ja_conversation", "ja_pause"],
))

# ── 英语教学技能补充（基于二语习得文献） ──
# 参考：Long(Interaction Hypothesis), Swain(Output Hypothesis),
#       Schmidt(Noticing Hypothesis), Wray(Formulaic Sequences),
#       Wood/Bruner(Scaffolding), Krashen(Affective Filter)

_register(TeachingSkill(
    name="task_based_roleplay",
    description="Run any learner-described real-world roleplay as one coherent communicative task, without a scene whitelist or scripted dialogue.",
    instruction=(
        "Treat the learner's own scene description as an open-world real interaction, not a sentence exercise or a "
        "preset script. Stay in the inferred counterpart role and use general world knowledge to produce only the next "
        "natural in-role response. If the learner gets stuck, briefly give level-appropriate support, then resume the "
        "same interaction. Never restart, change topic, or ask something that the real counterpart would not ask."
    ),
    example_dialogue="No reusable dialogue: infer the next utterance from the learner's scene, roles, and conversation history.",
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "roleplay", "daily_life"],
    target_lang="en",
    learning_objectives=[LearningObjective.SPEAKING_FLUENCY, LearningObjective.CONVERSATION_STRATEGIES],
    evidence_sources=[EvidenceSource.TASK_BASED_LEARNING, EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING],
    error_types=["en_roleplay_drift", "en_scene_break", "en_stuck"],
))

_register(TeachingSkill(
    name="clarification_request",
    description="When the user seems confused or gives an irrelevant response, ask for clarification using simpler language. Based on Long's Interaction Hypothesis - negotiation of meaning drives acquisition.",
    instruction="When the user doesn't understand your question or gives an irrelevant answer, do NOT repeat the same sentence. Instead, rephrase using simpler vocabulary, slower structure, or offer choices. Confirm understanding with 'Do you mean...?' or 'Are you saying...?'",
    example_dialogue='User looks confused after "What\'s your favorite cuisine?" -> Tutor: "我的意思是，你最喜欢吃什么？Do you like Chinese food, or Italian food?"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "free_talk", "daily_life", "roleplay"],
    target_lang="en",
    learning_objectives=[LearningObjective.CONVERSATION_STRATEGIES, LearningObjective.LISTENING_COMPREHENSION],
    evidence_sources=[EvidenceSource.COMPREHENSIBLE_INPUT, EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING],
    error_types=["en_confusion", "en_misunderstanding", "en_silence"],
))

_register(TeachingSkill(
    name="scaffolding_hint",
    description="When the user is stuck or can only produce single words, provide scaffolding (first letters, fill-in-the-blank, or binary choices) to help them produce a complete sentence. Based on Wood/Bruner's Scaffolding theory.",
    instruction="When the user is stuck, silent, or only says single words, do NOT give the full answer. Instead, provide a scaffold: (1) give the sentence starter (e.g. 'I would like to...'), (2) offer two choices (e.g. 'coffee or tea?'), (3) give first letter hint. Wait for them to complete it. Gradually remove support as they improve. EXCEPTION (starter, not guessing game): when the learner already got their meaning across and only swapped in a Chinese word for the one English word they don't know, first finish answering what they asked, then give that one word directly - no first-letter hints, no fill-in-the-blank, no asking them to try again.",
    example_dialogue='User says only "coffee" -> Tutor: "好，试试说完整句：I would like ___ coffee. 填一个词，a 还是 two？" User: "a" -> Tutor: "对！I would like a coffee. 你来说一遍。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "free_talk", "daily_life", "roleplay", "grammar_focus"],
    target_lang="en",
    learning_objectives=[LearningObjective.SPEAKING_FLUENCY, LearningObjective.GRAMMAR_MASTERY],
    evidence_sources=[EvidenceSource.SCAFFOLDING, EvidenceSource.COMPREHENSIBLE_INPUT],
    error_types=["en_silence", "en_fragment", "en_stuck"],
))

_register(TeachingSkill(
    name="formulaic_sequence",
    description="Teach high-frequency formulaic sequences (chunks) as unanalyzed wholes. Based on Wray's research - fluent speakers use大量 prefabricated chunks, not word-by-word construction.",
    instruction="When teaching new expressions, teach them as complete chunks first, not individual words. Common chunks: 'I would like to...', 'Could you...?', 'How about...?', 'I'm looking forward to...', 'What do you mean by...?'. Drill the whole chunk until automatic, then analyze components later.",
    example_dialogue='User says "I want coffee" -> Tutor: "更地道的说法是 I would like a coffee（我想要一杯咖啡）。这是一整句话，跟我念一遍：I would like a coffee。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "free_talk", "daily_life", "roleplay", "vocabulary"],
    target_lang="en",
    learning_objectives=[LearningObjective.VOCABULARY_EXPANSION, LearningObjective.SPEAKING_FLUENCY],
    evidence_sources=[EvidenceSource.COMPREHENSIBLE_INPUT, EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING],
    error_types=["en_word_by_word", "en_unnatural_expression"],
))

_register(TeachingSkill(
    name="collocation_focus",
    description="Highlight and teach natural collocations (words that commonly go together). Chinese learners often translate word-by-word, producing unnatural collocations.",
    instruction="When the user produces an unnatural word combination, identify the collocation error and teach the correct pairing. Common problem areas for Chinese learners: do/homework (not make homework), take/a photo (not do photo), heavy/rain (not big rain), catch/a cold (not get cold). Present the correct collocation as a chunk.",
    example_dialogue='User says "I will make homework" -> Tutor: "做作业应该用 do homework，不是 make。记住了：do homework, do exercise, do business。"',
    proficiency=ProficiencyLevel.A2,
    scene_tags=["en", "vocabulary", "free_talk", "daily_life"],
    target_lang="en",
    learning_objectives=[LearningObjective.VOCABULARY_EXPANSION, LearningObjective.GRAMMAR_MASTERY],
    evidence_sources=[EvidenceSource.CONTRASTIVE_ANALYSIS, EvidenceSource.COMPREHENSIBLE_INPUT],
    error_types=["en_collocation", "en_word_choice", "en_chinglish"],
))

_register(TeachingSkill(
    name="concept_check",
    description="Use simple questions to verify the user actually understood a new word or grammar point, rather than asking 'Do you understand?'. Based on CLT methodology.",
    instruction="After teaching a new word or structure, do NOT ask 'Do you understand?' (users always say yes). Instead, ask a concept checking question that requires demonstrating understanding. Examples: after teaching 'I used to smoke', ask 'Do I smoke now? Yes or no?'",
    example_dialogue='Tutor teaches "I have been to Paris" -> Tutor: "考考你，I have been to Paris 意思是：A 我现在在巴黎，B 我以前去过巴黎但现在不在，C 我没去过巴黎。选哪个？"',
    proficiency=ProficiencyLevel.A2,
    scene_tags=["en", "vocabulary", "grammar_focus", "free_talk"],
    target_lang="en",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY, LearningObjective.VOCABULARY_EXPANSION],
    evidence_sources=[EvidenceSource.COMMUNICATIVE_LANGUAGE_TEACHING, EvidenceSource.COGNITIVE_CODE_LEARNING],
    error_types=["en_misunderstanding", "en_confusion"],
))

_register(TeachingSkill(
    name="contrastive_cn_en",
    description="Identify and explain errors caused by Chinese-to-English language transfer (母语迁移). Based on Contrastive Analysis theory - understanding L1 interference helps predict and correct L2 errors.",
    instruction="When the user makes an error that stems from Chinese language patterns, explain the structural difference explicitly. Key contrast areas: (1) Chinese has no articles -> English requires a/an/the; (2) Chinese has no subject-verb agreement -> English requires 3rd person singular -s; (3) Chinese has no tense morphology -> English requires past tense markers; (4) Chinese verbs don't take prepositions -> English: look AT, listen TO; (5) Chinese time adverbs come before verb -> English time adverbs at sentence end.",
    example_dialogue='User says "I yesterday go to school" -> Tutor: "中文说我昨天去学校，但英语时间不能放中间。应该是 I went to school yesterday。记住：英语时间放句尾，而且要用过去式 went。"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "grammar_focus", "free_talk", "daily_life"],
    target_lang="en",
    learning_objectives=[LearningObjective.GRAMMAR_MASTERY, LearningObjective.SPEAKING_FLUENCY],
    evidence_sources=[EvidenceSource.CONTRASTIVE_ANALYSIS, EvidenceSource.COGNITIVE_CODE_LEARNING],
    error_types=["en_article_missing", "en_subject_verb_agreement", "en_tense_error", "en_preposition", "en_word_order", "en_chinglish"],
))

_register(TeachingSkill(
    name="pushed_output_task",
    description="When the user has been performing well, push them to produce more complex output using newly learned structures. Based on Swain's Output Hypothesis - producing language forces learners to notice gaps in their knowledge.",
    instruction="When the user has successfully completed 2-3 exchanges correctly, push them to use the new structure in a different context. Give a specific task: 'Now make a sentence using [structure] about [topic].' Do NOT let them stay in their comfort zone.",
    example_dialogue='User successfully used "I would like" in ordering food -> Tutor: "不错！现在用 I would like to 造一个新句子，说说你想去哪个国家旅行？"',
    proficiency=ProficiencyLevel.A2,
    scene_tags=["en", "free_talk", "grammar_focus", "roleplay"],
    target_lang="en",
    learning_objectives=[LearningObjective.SPEAKING_FLUENCY, LearningObjective.SELF_CORRECTION],
    evidence_sources=[EvidenceSource.TASK_BASED_LEARNING, EvidenceSource.SELF_CORRECTION],
    error_types=["en_no_progress", "en_repetitive_simple"],
))

_register(TeachingSkill(
    name="affective_support",
    description="When the user shows signs of frustration, anxiety, or repeated failure, lower the difficulty and provide emotional support. Based on Krashen's Affective Filter Hypothesis - high anxiety blocks language acquisition.",
    instruction="When the user seems frustrated (long silences, 'I can't', repeated errors on same point, or explicit complaint), STOP correcting. Instead: (1) acknowledge difficulty, (2) switch to an easier topic, (3) praise what they did right, (4) give a very simple task they can succeed at. Do NOT push through frustration.",
    example_dialogue='User says "I can\'t speak English" after multiple errors -> Tutor: "你已经说得很好了！学语言就是慢慢来的过程。我们聊点简单的——你喜欢猫还是狗？Cats or dogs？"',
    proficiency=ProficiencyLevel.A1,
    scene_tags=["en", "free_talk", "daily_life"],
    target_lang="en",
    learning_objectives=[LearningObjective.SPEAKING_FLUENCY, LearningObjective.CONVERSATION_STRATEGIES],
    evidence_sources=[EvidenceSource.COMPREHENSIBLE_INPUT, EvidenceSource.NATURAL_APPROACH],
    error_types=["en_frustration", "en_anxiety", "en_giving_up"],
))
