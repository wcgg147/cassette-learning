"""Deterministic learning-state controller for spoken-language sessions.

The language model may choose wording, but it must not choose whether the
learner is orienting, attempting, repairing, or transferring an expression.
This module owns that progression and keeps it inspectable in tests/logs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

try:
    from .intent_router import TutorIntent
except ImportError:
    from agent.intent_router import TutorIntent

try:
    from .learner_state import LearnerState as EstimatedLearnerState
except ImportError:
    from agent.learner_state import LearnerState as EstimatedLearnerState


@dataclass
class LearningState:
    stage: str = "orient"
    mission: str = "把一个真实想法用英语说清楚"
    communicative_goal: str = "表达一个完整、可理解的意思"
    target_expression: str = ""
    support_level: int = 0
    attempts: int = 0
    successful_attempts: int = 0
    repair_attempts: int = 0
    transfer_attempts: int = 0
    deepening_turns: int = 0
    turns_since_correction: int = 99
    support_recoveries: int = 0
    autonomy_repairs: int = 0
    completion_evidence: List[str] = None
    last_move: str = "mission_opening"
    scenario: str = ""
    learner_role: str = ""
    tutor_role: str = ""
    scenario_step: str = ""
    scenario_brief: str = ""
    roleplay_turn: int = 0
    roleplay_active: bool = False
    pending_recast: str = ""
    recast_uptake: bool = False
    transfer_branches_completed: int = 0
    active_language_item: str = ""
    active_language_item_stage: str = ""
    active_language_item_source: str = ""
    turns_since_language_item: int = 0
    language_item_reuses: int = 0

    def __post_init__(self) -> None:
        if self.completion_evidence is None:
            self.completion_evidence = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningMove:
    stage: str
    action: str
    goal: str
    user_must_speak: bool = True
    support_level: int = 0
    feedback_mode: str = "meaning_first"
    prompt_hint: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearningOrchestrator:
    """Advance one small spoken-output mission without using an LLM."""

    META_INTENTS = {TutorIntent.COMPLAINT, TutorIntent.META_QUESTION}

    def start(self, scene: Dict[str, Any], profile: Dict[str, Any]) -> LearningState:
        raw_goal = str((scene or {}).get("goal") or "").strip()
        generic = (
            not raw_goal
            or raw_goal.lower().startswith("practice english naturally")
            or raw_goal in {"在自然对话中练习目标语言", "Practice English naturally"}
        )
        mission = "把一个真实想法用英语说清楚" if generic else raw_goal[:180]
        weakness = str((profile or {}).get("recent_weakness") or "").strip()
        goal = "表达一个完整、可理解的意思"
        if weakness:
            goal += f"；必要时自然练习 {weakness[:80]}"
        return LearningState(mission=mission, communicative_goal=goal)

    def plan(
        self,
        state: LearningState,
        *,
        user_text: str,
        intent: TutorIntent,
        errors: List[Dict[str, Any]],
        target_lang: str,
        learner_state: EstimatedLearnerState | None = None,
        conversation_plan: Dict[str, Any] | None = None,
        chat_tape: bool = False,
    ) -> LearningMove:
        """Plan one tutor move.

        ``chat_tape`` 只**搁置**两条教学分支，不删除它们：闲聊盘里用户用中文
        聊天、没有义务说目标语，``prompt_output``（"用户尚未尝试目标语言"）和
        ``reduce_cognitive_load``（"认知负荷过高"）会把纯闲聊判成教学状态，
        每轮换一套教学指令 —— 实测 4 轮闲聊触发了 4 种不同 action，回复里
        出现"先引用"式复述和插入中文教学。练习盘/素材盘传默认 False，
        行为逐字不变。
        """
        text = (user_text or "").strip()
        attempted = self._attempted_target_language(text, target_lang)

        # Intent is only a fallback.  The estimated state combines intent with
        # task stage, affect, engagement and explicit exit/autonomy signals.
        communicative_intent = getattr(learner_state, "communicative_intent", "")
        cognitive_state = getattr(learner_state, "cognitive_state", "")
        affective_state = getattr(learner_state, "affective_state", "")
        exit_likelihood = float(getattr(learner_state, "exit_likelihood", 0.0) or 0.0)
        rhythm = getattr(learner_state, "interaction_rhythm", "protect_flow")
        self._track_active_language_item(
            state,
            text=text,
            communicative_intent=communicative_intent,
            errors=errors,
            conversation_plan=conversation_plan or {},
        )
        language_item_due = state.active_language_item_stage == "ready_for_reuse"
        language_item_hint = self._language_item_reuse_hint(state) if language_item_due else ""

        if communicative_intent == "exit_or_switch" or exit_likelihood >= 0.75:
            state.stage = "reflect"
            state.last_move = "session_reflection"
            return LearningMove(
                stage=state.stage,
                action="session_reflection",
                goal="尊重用户结束或切换的意图，用具体证据自然收尾",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="evidence_summary",
                reason="上下文显示用户正在退出或切换任务",
            )

        if communicative_intent == "skip_current_item":
            state.autonomy_repairs += 1
            state.last_move = "skip_current_item"
            return LearningMove(
                stage=state.stage,
                action="skip_current_item",
                goal="停止当前过于简单或用户明确拒绝的练习，并在原话题原进度继续",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="restore_autonomy",
                prompt_hint=(
                    "用户拒绝的是当前这一句/跟读动作，不是整个话题。只确认一次并立即跳过；"
                    "禁止再次要求他说同一句，也禁止跳到面条、时尚、猫狗、电影等新话题。"
                    "从当前真实任务中尚未解决的下一步继续，提供更有信息量的内容。"
                ),
                reason="用户明确拒绝当前练习项；应恢复选择权但保留原任务连续性",
            )

        if communicative_intent == "real_world_context":
            state.stage = "flow"
            state.last_move = "ground_real_world_context"
            return LearningMove(
                stage=state.stage,
                action="ground_real_world_context",
                goal="先解决现实中渠道、对象和行动顺序，再进入语言练习",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="real_world_grounding",
                prompt_hint=(
                    "用户先问现实任务怎样发生，不是在要一个单词。先用中文给出可靠且有条件边界的"
                    "行动顺序；不要臆造随机路人、咖啡师一定知道，也不要立刻造句或跟读。"
                    "现实路径说清后，只补一个当下真正用得上的自然英文表达；信息随国家/品牌变化时明说。"
                ),
                reason="语言任务依赖现实渠道；先建立真实行动模型，才能选择对话对象和台词",
            )

        if communicative_intent == "counterpart_turn_request":
            state.stage = "flow"
            state.last_move = "supply_counterpart_turn"
            return LearningMove(
                stage=state.stage,
                action="supply_counterpart_turn",
                goal="先给出当前场景中对方自然会说的话，再把轮次交还用户",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="turn_taking_support",
                prompt_hint=(
                    "用户问的是对方下一句。先以对方角色给出一句现实、自然、完整的台词和中文义；"
                    "若用户明确提出同意/拒绝两种情况，则各给一种。不要重复用户刚才的话，"
                    "不要先要求用户说 Thank you，也不要反过来问用户对方该说什么。"
                ),
                reason="角色对话必须遵守轮流关系；此刻轮到AI提供对方台词",
            )

        if communicative_intent == "scenario_negotiation" or intent == TutorIntent.SCENARIO_ROLEPLAY:
            restart_requested = any(marker in text for marker in ("重新开始", "从头开始", "重来一遍"))
            if state.roleplay_active and state.scenario and not restart_requested:
                # Product/role feedback inside an active scene changes the
                # contract, not the user's progress.  Resetting to the host's
                # first question here caused the repeated "How many people?"
                # loop seen in real voice transcripts.
                state.stage = "flow"
                state.last_move = "repair_roleplay_contract"
                return LearningMove(
                    stage=state.stage,
                    action="repair_roleplay_contract",
                    goal="应用用户对角色和流程的修正，但保留已经完成的场景步骤",
                    user_must_speak=False,
                    support_level=state.support_level,
                    feedback_mode="roleplay_contract_repair",
                    prompt_hint=(
                        f"场景仍是“{state.scenario_brief or state.scenario}”，角色不变。"
                        "具体承认刚才哪句话脱离角色，然后直接以当前AI角色自然接下一句；"
                        "不得重开场景、改写双方身份或切成教学解说。"
                    ),
                    reason="用户正在修正场景规则；修正规则不能抹掉已完成的角色进度",
                )
            state.stage = "flow"
            state.scenario = state.scenario or "open_roleplay"
            state.scenario_brief = state.scenario_brief or text[:240]
            state.learner_role = state.learner_role or "场景中的用户本人"
            state.tutor_role = state.tutor_role or "场景中与用户交谈的真实角色"
            state.scenario_step = "active_exchange"
            state.roleplay_active = True
            state.last_move = "establish_roleplay_scene"
            return LearningMove(
                stage=state.stage,
                action="establish_roleplay_scene",
                goal="固定真实场景、双方角色和当前步骤，然后立即开始演绎",
                user_must_speak=True,
                support_level=state.support_level,
                feedback_mode="roleplay_contract",
                prompt_hint=(
                    f"场景原意={state.scenario_brief}；用户角色={state.learner_role}；"
                    f"AI角色={state.tutor_role}。不要套固定剧本或枚举步骤；根据现实常识自行判断"
                    "该角色此刻最自然的一句话。不要讨论产品架构，不要抽离成造句题；"
                    "本轮只说AI角色的一次真实回应。"
                ),
                reason="用户正在定义真实场景、角色和流程，必须先建立可持续的角色状态",
            )

        if communicative_intent == "role_expression_support" and state.scenario:
            state.last_move = "support_role_line"
            return LearningMove(
                stage=state.stage,
                action="support_role_line",
                goal="只帮助用户当前角色完成当前一步该说的一句话",
                user_must_speak=True,
                support_level=max(1, state.support_level),
                feedback_mode="role_line_scaffold",
                prompt_hint=(
                    f"场景仍是“{state.scenario_brief or state.scenario}”；"
                    f"AI当前角色是“{state.tutor_role}”，用户角色是“{state.learner_role}”。"
                    "先明确回答是否已经进入该角色，并承接AI上一句在该场景里的意思；"
                    "再只帮助用户当前角色完成当前一步的一句自然英文，紧邻中文义。"
                    "不得换题、不得重新开场，也不要展开产品架构解释。"
                ),
                reason="用户问的是自己当前角色此刻如何表达，需要单角色语言支架",
            )

        if (
            state.roleplay_active
            and state.scenario
            and (
                communicative_intent in {"method_negotiation", "teaching_preference"}
                or intent in self.META_INTENTS
            )
        ):
            state.stage = "flow"
            state.autonomy_repairs += 1
            state.last_move = "repair_roleplay_contract"
            return LearningMove(
                stage=state.stage,
                action="repair_roleplay_contract",
                goal="回应用户指出的角色或衔接错误，并从当前场景原地继续",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="roleplay_contract_repair",
                prompt_hint=(
                    f"不要重开场景；场景仍是“{state.scenario_brief or state.scenario}”。"
                    "用一句中文指出刚才角色或衔接哪里错了，随后直接以当前AI角色自然接话。"
                ),
                reason="活跃角色场景中的反馈应原地修复，不能触发通用方法修复或旧练习",
            )

        # A clean first turn such as "我不知道怎么和服务员沟通" contains a
        # confusion word, but it is not a request to repair a previous tutor
        # mistake.  Establish the relationship and current situation before
        # the generic confusion branch.  Explicit method/teaching/content
        # questions above and below still retain their specialised handling.
        if state.stage == "orient" and communicative_intent in {
            "clarification_request", "open_expression", "session_entry", ""
        }:
            substantive = self._has_substantive_content(text)
            state.stage = "flow"
            if attempted:
                state.attempts += 1
            action = "connect_first_turn" if substantive else "connection_opening"
            if state.scenario:
                # Persist only that an open interaction is active.  The model,
                # not this state machine, decides what a real counterpart says.
                state.scenario_step = state.scenario_step or "active_exchange"
                state.roleplay_active = True
            state.last_move = action
            return LearningMove(
                stage=state.stage,
                action=action,
                goal=(
                    "先接住用户刚才表达的具体意思与情绪，再轻轻延续对话"
                    if substantive else
                    "自然回应招呼，并给用户一个无需准备、可以直接开口的入口"
                ),
                user_must_speak=False,
                support_level=0,
                feedback_mode="meaning_first",
                prompt_hint=(
                    "先引用或回应用户话里的一个具体信息，给出真正有内容的自然反应；"
                    "不要纠错、评分、布置练习或泛泛夸奖。顺着他的内容给一个有目的的自然追问——"
                    "问他话里一个真值得回答的具体细节，只问一个；需要时再配一个轻量表达支撑。"
                    if substantive else
                    "简短回应招呼，不自我介绍、不宣布课程任务；只给一个低压力、"
                    "用户可以从真实小事开始说的自然邀请。"
                ),
                reason="首轮先建立被听见和被理解的关系感，再决定是否进入教学动作",
            )

        if communicative_intent != "teaching_preference" and (
            communicative_intent == "method_negotiation"
            or affective_state == "frustrated"
            or intent in self.META_INTENTS
        ):
            state.autonomy_repairs += 1
            state.last_move = "method_repair"
            return LearningMove(
                stage=state.stage,
                action="method_repair",
                goal="根据用户本轮是在提出具体过程问题还是仍在表达想法，直接回应或短暂倾听",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="no_correction",
                prompt_hint=(
                    "用户提出了具体的教学或对话过程问题时，直接简短回答并落实；"
                    "用户仍在描述、整理想法且没有要求动作时，只给一句自然的在听回应并等待。"
                    "不要复述故障、不要自我审计、不要强行回到旧练习。"
                ),
                reason="用户正在评价产品或教学方法，必须先恢复控制感和信任",
            )

        if communicative_intent == "teaching_preference":
            state.autonomy_repairs += 1
            state.last_move = "adapt_teaching_preference"
            return LearningMove(
                stage=state.stage,
                action="adapt_teaching_preference",
                goal="把用户刚说明的学习方式立即变成下一步可感知的教学动作",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="preference_applied",
                prompt_hint=(
                    "最多用一句话确认你理解的偏好，然后立刻按它示范：新词先给中文语义，"
                    "再给一个明确的中文表达目标，让用户自行组织英文；需要时只留一个半句句架。"
                    "不要只说你说得对，也不要马上抛出宽泛问题。"
                ),
                reason="用户正在定义更适合自己的支架方式，应即时应用而不是继续原策略",
            )

        if communicative_intent == "learning_difficulty":
            state.stage = "flow"
            state.last_move = "address_learning_difficulty"
            return LearningMove(
                stage=state.stage,
                action="address_learning_difficulty",
                goal="先回应用户遇到的真实理解困难，再定位具体材料和困难类型",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="meaning_first",
                prompt_hint=(
                    "明确承接用户正在接触的材料；可以自然询问剧名，或区分语速、连读和生词，"
                    "但本轮不要把它误当成对AI回复的投诉，不要让用户造句、填空或解释为什么学英语。"
                ),
                reason="用户描述的是外部学习材料的理解困难，不是没听懂AI解释或在投诉教学方法",
            )

        if communicative_intent == "expression_evaluation":
            self._activate_language_item(
                state,
                str((conversation_plan or {}).get("corrected_expression") or ""),
                "correction",
            )
            state.last_move = "evaluate_expression_then_resume"
            return LearningMove(
                stage=state.stage,
                action="evaluate_expression_then_resume",
                goal="判断用户刚说的英文是否自然，并给一个可直接使用的更自然版本",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="one_requested_correction",
                prompt_hint=(
                    "这是用户主动请求的表达评估，优先级高于普通纠错偏好。先说明原句是自然、"
                    "能听懂但不自然，还是意思有误；若需要调整，只给一个最自然、符合当前现实角色的版本，"
                    "再用一句中文指出关键差异。不要泛泛夸奖，不要求跟读，并保留原场景供下一轮继续。"
                ),
                reason="用户明确询问表达是否自然，必须提供真实、可执行的语言反馈",
            )

        if communicative_intent == "lexical_clarification":
            state.last_move = "clarify_lexical_then_resume"
            return LearningMove(
                stage=state.stage,
                action="clarify_lexical_then_resume",
                goal="精确解决当前词义、短语或语法侧问，同时保留原场景和学习目标",
                support_level=state.support_level,
                feedback_mode="brief_lexical_clarification",
                prompt_hint=(
                    "先直接回答用户所问的中英映射、含义或关键语法差异。只有用户明确追问用法或表示要继续练，"
                    "才补一个贴近原场景的例句或半句支架；本轮不要自动布置造句、跟读，也不要推进角色流程。"
                    "若同一句还有第二个明确问题，按出现顺序一并回答。"
                ),
                reason="用户在进行中的任务中提出局部词义问题，仍有继续参与的信号",
            )

        if cognitive_state in {"local_confusion", "overloaded"} or intent == TutorIntent.CONFUSION:
            if chat_tape and cognitive_state == "overloaded":
                # 闲聊盘搁置这条：用户说中文闲聊时 cognitive_state 常被判成
                # overloaded，会把"聊得轻松"当成"听不懂"而切进支架教学。
                # 只跳过 overloaded 这一支；用户真的问"什么意思/没懂"仍走
                # local_confusion ⇒ clarify_then_check（下方原逻辑）。
                pass
            else:
                state.stage = "scaffold"
                state.support_level = min(3, state.support_level + 1)
                state.support_recoveries += 1
                action = "reduce_cognitive_load" if cognitive_state == "overloaded" else "clarify_then_check"
                state.last_move = action
                return LearningMove(
                    stage=state.stage,
                    action=action,
                    goal="让用户立即听懂，并保留一次极小的自主表达",
                    support_level=state.support_level,
                    prompt_hint=self._scaffold_hint(state.support_level),
                    reason="状态估计显示局部困惑或认知负荷过高",
                )

        if intent == TutorIntent.FAREWELL:
            state.stage = "reflect"
            state.last_move = "session_reflection"
            return LearningMove(
                stage=state.stage,
                action="session_reflection",
                goal="用具体证据收尾并留下下次任务",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="evidence_summary",
                reason="用户希望结束会话",
            )

        if state.stage == "orient":
            # The first reply establishes the relationship contract. Even when
            # the learner already produced target-language output, do not turn
            # the opening into an immediate correction, score or transfer drill.
            substantive = self._has_substantive_content(text)
            state.stage = "flow"
            if attempted:
                state.attempts += 1
            action = "connect_first_turn" if substantive else "connection_opening"
            state.last_move = action
            return LearningMove(
                stage=state.stage,
                action=action,
                goal=(
                    "先接住用户刚才表达的具体意思与情绪，再轻轻延续对话"
                    if substantive else
                    "自然回应招呼，并给用户一个无需准备、可以直接开口的入口"
                ),
                user_must_speak=False,
                support_level=0,
                feedback_mode="meaning_first",
                prompt_hint=(
                    "先引用或回应用户话里的一个具体信息，给出真正有内容的自然反应；"
                    "不要纠错、评分、布置练习或泛泛夸奖。顺着他的内容给一个有目的的自然追问——"
                    "问他话里一个真值得回答的具体细节，只问一个；需要时再配一个轻量表达支撑。"
                    if substantive else
                    "简短回应招呼，不自我介绍、不宣布课程任务；只给一个低压力、"
                    "用户可以从真实小事开始说的自然邀请。"
                ),
                reason="首轮先建立被听见和被理解的关系感，再决定是否进入教学动作",
            )

        transition = str((conversation_plan or {}).get("topic_transition") or "continue")
        if (
            state.roleplay_active and state.scenario
            and state.recast_uptake and transition == "same_goal_new_phase"
        ):
            state.stage = "transfer"
            state.transfer_attempts += 1
            state.transfer_branches_completed += 1
            state.recast_uptake = False
            state.pending_recast = ""
            state.last_move = "continue_roleplay_transfer_branch"
            return LearningMove(
                stage=state.stage,
                action="continue_roleplay_transfer_branch",
                goal="在同一现实目标的另一处理分支中验证刚学会的表达能力",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode="in_role_transfer",
                prompt_hint=(
                    f"保持场景“{state.scenario_brief or state.scenario}”和双方身份，"
                    "直接从导演指定的新处理阶段给出当前角色下一句。不得重播问候、姓名、房间号、"
                    "订单号等已完成开场，也不要宣布现在开始迁移练习。"
                ),
                reason="学习者已在角色中正确使用过升级表达；现在用同一目标的真实分支验证迁移",
            )

        # 用户主导话题：任何用户主动聊起的个人话题（兴趣、经历、见闻、生活琐事等）
        # 都可以成为英语练习载体——不按话题内容分类（不片面），只看用户是否在主导
        # 一个具体话题。用户可能先用中文代替不熟的英文，正好借机听懂内容、教他英文、
        # 引导他大胆说。场景状态原样保留，用户随时可以回到场景。
        interaction_frame = str((conversation_plan or {}).get("interaction_frame") or "")
        if (
            state.roleplay_active and state.scenario
            and (
                interaction_frame == "general"
                or self._is_user_driven_topic(text)
            )
        ):
            uncertain_spans = list((conversation_plan or {}).get("uncertain_spans") or [])
            unresolved_slots = list((conversation_plan or {}).get("unresolved_slots") or [])
            if uncertain_spans or unresolved_slots or str(
                (conversation_plan or {}).get("pragmatic_check") or ""
            ) == "clarify":
                state.stage = "flow"
                state.last_move = "clarify_uncertain_expression"
                return LearningMove(
                    stage=state.stage,
                    action="clarify_uncertain_expression",
                    goal="先确认没有听稳的语义，再继续同一话题并提供语言反馈",
                    user_must_speak=False,
                    support_level=state.support_level,
                    feedback_mode="meaning_clarification",
                    prompt_hint=(
                        "保留已经听清的内容，只引用最短的歧义片段，用一个简短中文问题确认用户原意。"
                        "不要替用户补成完整故事，不要在意思未确认前纠错、改写或推进场景。"
                    ),
                    reason="本轮存在导演标记的语义不确定项；先确认含义比纠正猜出来的句子更可靠",
                )
            state.stage = "flow"
            state.last_move = "follow_interest_topic"
            topic = " ".join(str(text or "").split())[:60]
            state.turns_since_correction += 1
            allow_recast = bool(
                            errors
                            and self._correction_is_due(errors, state.turns_since_correction)
                            and rhythm not in {"lower_load", "restore_autonomy"}
                            and not chat_tape
                        )
            if allow_recast:
                state.turns_since_correction = 0
                self._activate_language_item(
                    state, str(errors[0].get("correction") or ""), "correction"
                )
            return LearningMove(
                stage=state.stage,
                action="follow_interest_topic",
                goal="顺着用户主导的话题交流，并把用户此刻想表达的内容引导成贴合其水平的目标语",
                user_must_speak=False,
                support_level=state.support_level,
                feedback_mode=(
                    "interest_topic_recast" if allow_recast
                    else "interest_topic_meaning_first"
                ),
                prompt_hint=(
                    f"用户正在聊自己的话题：“{topic}”。先自然接住内容、必要时用中文确认你听懂了；"
                    "把目标语自然织进回应：至少有一个贴合其水平的完整英文句子，必要时紧邻一句中文义。"
                    "顺着他的内容给一个有目的的自然追问——问他话里一个真值得回答的具体细节，只问一个；"
                    "不得把用户每句话都改造成练习任务。"
                    + (
                        "本轮有一处高价值语言问题，可以做一次轻量改写；根据上下文在直接自然复述、"
                        "一句短对比或先回应内容后补充表达之间选择，不能沿用最近一轮相同的纠错开头，"
                        "尤其不要固定使用‘顺便说……更自然’。"
                        if allow_recast else
                        "本轮处于纠错冷却期或没有可靠错误证据，不追加语言点评；只让内容和英语话口自然继续。"
                    )
                    + "不要回到“酒店退房/账单多扣/网购到货”等存储的练习场景菜单，"
                    "不要替用户重选场景，也不要纯中文闲聊而不引导英文。"
                    "原场景保留为可随时返回，但本轮以用户的话题为主。"
                    + language_item_hint
                ),
                reason="用户主导的任何话题都能作为英语练习载体；先听懂中文、再教英文表达，"
                       "让用户敢于用英文聊自己感兴趣的东西",
            )

        # 普通对话通道：没有场景演绎、没有语言侧问/方法协商等专门意图，用户只是在
        # 自然聊天（兴趣、见闻、观点、生活、天气、趣事……）。对话本身就该有趣、
        # 丰富、像人，并贴合水平地轻量引导英语——而不是一律变成练习支架或场景菜单。
        # 只在空闲态（stage=flow、无场景）触发，避免打断正在进行的练习流程；
        # 带错误的目标语产出仍走修复/接续流程，不属于普通闲聊。
        if (
            state.stage == "flow"
            and not state.roleplay_active
            and not state.scenario
            and self._is_general_chat(text, intent, errors)
        ):
            uncertain_spans = list((conversation_plan or {}).get("uncertain_spans") or [])
            unresolved_slots = list((conversation_plan or {}).get("unresolved_slots") or [])
            if uncertain_spans or unresolved_slots or str(
                (conversation_plan or {}).get("pragmatic_check") or ""
            ) == "clarify":
                state.stage = "flow"
                state.last_move = "clarify_uncertain_expression"
                return LearningMove(
                    stage=state.stage,
                    action="clarify_uncertain_expression",
                    goal="先确认没有听稳的语义，再继续自然聊天和语言反馈",
                    user_must_speak=False,
                    support_level=0,
                    feedback_mode="meaning_clarification",
                    prompt_hint=(
                        "只保留已经确定的意思，并引用最短的歧义片段问一个简短中文确认问题。"
                        "不要猜用户喜欢谁、看过哪一季或想表达什么，也不要纠正一个尚未确认的意思。"
                    ),
                    reason="导演检测到本轮语义仍有不确定项；先听准，再纠正",
                )
            state.stage = "flow"
            state.last_move = "engage_general_topic"
            state.turns_since_correction += 1
            allow_recast = bool(
                            errors
                            and self._correction_is_due(errors, state.turns_since_correction)
                            and rhythm not in {"lower_load", "restore_autonomy"}
                            and not chat_tape
                        )
            if allow_recast:
                state.turns_since_correction = 0
                self._activate_language_item(
                    state, str(errors[0].get("correction") or ""), "correction"
                )
            return LearningMove(
                stage=state.stage,
                action="engage_general_topic",
                goal="像真实朋友一样自然、有趣地回应，并在话题内贴合水平地轻量引导英语表达",
                user_must_speak=False,
                support_level=0,
                feedback_mode=(
                    "natural_conversation_recast" if allow_recast
                    else "natural_conversation_meaning_first"
                ),
                prompt_hint=(
                    "用户只是在自然聊天。像真实朋友一样接住话题：可以说说相关的趣事、"
                    "分享一点见闻或文化常识、给出观点，偶尔才反问用户，让对话丰富有趣；"
                    "先准确回答用户当前内容，再把至少一个符合用户水平的完整目标语句子自然融进同一话题。"
                    "可以紧邻一句中文义；本轮也可以不提问，让用户自行接话。不得要求跟读、造句或跳去无关练习。"
                    + (
                        "本轮有一处经过判断的高价值语言问题，可以轻量纠正一次。纠正仍要做，"
                        "但呈现方式要贴合当前对话：可直接用正确表达回应内容、可用一句短对比，"
                        "也可先答内容再补一句自然说法；不要重复最近使用过的纠错句式，尤其不要"
                        "每轮固定说‘顺便说……更自然’，只处理一处且不展开语法课。"
                        if allow_recast else
                        "本轮没有可靠错误证据或仍在纠错冷却期：不追加‘更自然/可以说’式点评，"
                        "只准确回应内容，并自然保留英文交流氛围。"
                    )
                    + "不要变成练习支架、不要抛单词/语法/场景菜单，也不要追问学习目标。"
                    "贴合用户水平，重点是对话像人、有趣。"
                    + language_item_hint
                ),
                reason="对话的趣味与丰富度本身就是语言学习的燃料；普通闲聊不该被强行"
                       "变成练习支架或拽回场景菜单",
            )

        if (
            state.roleplay_active and state.scenario and attempted
            and interaction_frame != "general"
        ):
            if re.fullmatch(r"\s*just\s+my[\s,!.?]*", text, re.IGNORECASE):
                state.turns_since_correction = 0
                state.last_move = "repair_roleplay_line"
                return LearningMove(
                    stage=state.stage,
                    action="repair_roleplay_line",
                    goal="只修正当前角色台词中的一个关键错误，然后停在原步骤等待重说",
                    user_must_speak=True,
                    support_level=max(1, state.support_level),
                    feedback_mode="one_role_line_correction",
                    prompt_hint="这里应说 Just me（就我一位）；只修正这一处，等待用户重说，不推进场景。",
                    reason="Just my 不能表达就我一位；当前只处理这个可验证的语言错误",
                )
            state.turns_since_correction += 1
            if (
                errors
                and self._correction_is_due(errors, state.turns_since_correction)
                and rhythm not in {"lower_load", "restore_autonomy"}
            ):
                issue = errors[0]
                excerpt = str(issue.get("excerpt") or "").strip()
                correction = str(issue.get("correction") or "").strip()
                explanation = str(issue.get("explanation") or "").strip()
                state.roleplay_turn += 1
                state.scenario_step = "active_exchange"
                state.stage = "flow"
                state.attempts += 1
                state.turns_since_correction = 0
                state.last_move = "continue_roleplay_with_recast"
                correction_hint = (
                    f"本轮只处理这一处：{excerpt} → {correction}。"
                    if excerpt and correction else
                    "本轮只自然改写一个最影响准确性或自然度的问题。"
                )
                if explanation:
                    correction_hint += f"原因只需极短说明：{explanation}。"
                state.pending_recast = correction
                state.recast_uptake = False
                self._activate_language_item(state, correction, "correction")
                return LearningMove(
                    stage=state.stage,
                    action="continue_roleplay_with_recast",
                    goal="短促呈现一个更准确或更自然的表达，随即以当前角色继续真实流程",
                    user_must_speak=True,
                    support_level=state.support_level,
                    feedback_mode="one_brief_upgrade_in_role",
                    prompt_hint=(
                        correction_hint
                        + f"继续扮演 {state.tutor_role}，承接用户原意并推进“"
                        f"{state.scenario_brief or state.scenario}”当前一步。"
                        "不要退出角色讲课，不要默认要求跟读，也不要一次纠正第二处。"
                    ),
                    reason="用户已完成真实表达；当前有一个高价值语言问题，适合短促改写后继续场景",
                )
            if state.pending_recast and self._matches_recast(text, state.pending_recast):
                state.recast_uptake = True
                evidence = f"在角色场景中正确使用升级表达：{state.pending_recast}"
                if evidence not in state.completion_evidence:
                    state.completion_evidence.append(evidence)
            state.roleplay_turn += 1
            state.scenario_step = "active_exchange"
            state.stage = "flow"
            state.attempts += 1
            state.last_move = "continue_roleplay_scene"
            return LearningMove(
                stage=state.stage,
                action="continue_roleplay_scene",
                goal="保持双方角色和场景原意，让大模型根据现实常识自然接下一轮",
                user_must_speak=True,
                support_level=state.support_level,
                feedback_mode="in_role_meaning_first",
                prompt_hint=(
                    f"这是场景内第 {state.roleplay_turn} 次用户回应。继续扮演 {state.tutor_role}，"
                    f"严格依据场景原意“{state.scenario_brief or state.scenario}”自然接话；"
                    "由你用常识判断真实对方此刻会说什么，不要退出角色评价、"
                    "不要填空造句、不要讨论开发问题，也不要跳到无关话题。"
                    + language_item_hint
                ),
                reason="用户已在角色内输出目标语言，场景应继续而不是切回通用教学流程",
            )

        state.attempts += 1
        state.turns_since_correction += 1
        if not attempted and not chat_tape:
            # 闲聊盘搁置这条：闲聊里用户用中文说话是**正常行为**，不是"没试过
            # 说目标语"。这条一旦生效，每轮都会把纯聊天判成需要布置输出的教学轮
            # （实测闲聊第 4 轮 action=prompt_output，回复尾部出现
            # "Try saying it in..."），与 [Conversation Contract] 的
            # "不必每轮推进任务"直接冲突。练习盘/素材盘不受影响。
            state.stage = "scaffold"
            state.support_level = min(3, max(1, state.support_level + 1))
            state.last_move = "prompt_output"
            return LearningMove(
                stage=state.stage,
                action="prompt_output",
                goal="让用户亲自完成一个最小表达，而不是由 AI 替他说",
                support_level=state.support_level,
                prompt_hint=self._scaffold_hint(state.support_level),
                reason="用户尚未尝试使用目标语言表达",
            )

        # Feedback is useful only when the learner still has capacity to use
        # it.  Protect at least two learner turns after a correction; during
        # that cooldown, keep meaning moving instead of teaching again.
        if errors and self._correction_is_due(errors, state.turns_since_correction) \
                and rhythm not in {"lower_load", "restore_autonomy"}:
            state.stage = "repair"
            state.repair_attempts += 1
            state.turns_since_correction = 0
            state.last_move = "repair_one_issue"
            self._activate_language_item(
                state, str(errors[0].get("correction") or ""), "correction"
            )
            return LearningMove(
                stage=state.stage,
                action="repair_one_issue",
                goal="保留原意，只修复最影响理解的一个片段并让用户重说",
                support_level=state.support_level,
                feedback_mode="one_correction_then_retry",
                prompt_hint="先确认你听懂了用户的意思；只给一个修正；请用户只重说关键片段。",
                reason="用户已经输出，但存在可操作的关键问题",
            )

        if errors:
            state.stage = "flow"
            state.last_move = "acknowledge_and_continue"
            return LearningMove(
                stage=state.stage,
                action="acknowledge_and_continue",
                goal="优先回应用户表达的内容，不在此刻追加纠错",
                support_level=state.support_level,
                feedback_mode="meaning_only",
                prompt_hint="接住具体内容；最多一个自然追问，不解释语言规则。",
                reason="纠错冷却期或用户当前负荷不适合再教学",
            )

        state.successful_attempts += 1
        state.support_level = max(0, state.support_level - 1)
        if state.stage in {"repair", "scaffold", "attempt"}:
            state.stage = "transfer"
            state.last_move = "transfer_prompt"
            return LearningMove(
                stage=state.stage,
                action="transfer_prompt",
                goal="在一个略有变化的新情境里再次表达同一类意思",
                support_level=state.support_level,
                feedback_mode="brief_success_then_transfer",
                prompt_hint="简短确认表达成功，然后改变人物、地点或原因，让用户再说一句。",
                reason="首次表达成功，需要验证能否迁移而不是直接声称掌握",
            )

        if state.stage == "transfer":
            state.transfer_attempts += 1
            state.completion_evidence.append("在变化情境中再次完成表达")
        state.stage = "flow"
        state.deepening_turns += 1
        state.last_move = "listen_and_deepen"
        return LearningMove(
            stage=state.stage,
            action="listen_and_deepen",
            goal="先回应用户刚才的真实内容，再用一个有意义的下一步让对话自然深入",
            user_must_speak=True,
            support_level=state.support_level,
            feedback_mode="meaning_first",
            prompt_hint=("引用用户刚说的一个具体信息并作出真实回应；最多问一个能带来新信息、"
                         "选择或后果的问题。不要总结学习，不要转入纠错。"),
            reason="用户已能继续表达；应保持对话流而非过早宣布教学结束",
        )

    @staticmethod
    def _matches_recast(user_text: str, corrected_expression: str) -> bool:
        """Require observable reuse before treating a recast as uptake."""
        stop = {
            "a", "an", "the", "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "is", "am", "are", "was", "were", "to", "of", "for",
        }
        user_tokens = {
            token.lower() for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", user_text or "")
            if token.lower() not in stop
        }
        correction_tokens = [
            token.lower() for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", corrected_expression or "")
            if token.lower() not in stop
        ]
        if not correction_tokens:
            return False
        required = 1 if len(correction_tokens) == 1 else min(2, len(correction_tokens))
        return sum(token in user_tokens for token in set(correction_tokens)) >= required

    @staticmethod
    def _correction_is_due(errors: List[Dict[str, Any]], turns_since: int) -> bool:
        """Let a current, profile-prioritised weakness surface one turn sooner."""
        if not errors:
            return False
        priority = int(errors[0].get("priority") or 1)
        recurring = bool(errors[0].get("is_recurring") or priority >= 3)
        return turns_since >= (1 if recurring else 2)

    @classmethod
    def _track_active_language_item(
        cls,
        state: LearningState,
        *,
        text: str,
        communicative_intent: str,
        errors: List[Dict[str, Any]],
        conversation_plan: Dict[str, Any],
    ) -> None:
        """Maintain one bounded preview/use/reuse thread inside a conversation.

        This is intentionally not a vocabulary queue.  Only the latest explicit
        lexical question or the one correction selected for this turn becomes
        active; everything else stays in the durable error/profile system.
        """
        candidate = ""
        source = ""
        queried_item = cls._extract_queried_language_item(text)
        if communicative_intent == "lexical_clarification" or queried_item:
            candidate = queried_item
            source = "lexical_question"

        if candidate:
            if candidate.lower() != state.active_language_item.lower():
                state.active_language_item = candidate
                state.active_language_item_stage = "introduced"
                state.active_language_item_source = source
                state.turns_since_language_item = 0
                state.language_item_reuses = 0
            else:
                # Asking what an already-seen item means is evidence of
                # noticing it, not evidence that the learner can use it.
                state.active_language_item_stage = "introduced"
                state.active_language_item_source = source
                state.turns_since_language_item = 0
            return

        if not state.active_language_item:
            return
        state.turns_since_language_item += 1
        if cls._matches_recast(text, state.active_language_item):
            state.language_item_reuses += 1
            state.active_language_item_stage = "used"
            state.turns_since_language_item = 0
            evidence = f"在后续表达中主动使用：{state.active_language_item}"
            if evidence not in state.completion_evidence:
                state.completion_evidence.append(evidence)
            return
        if (
            state.active_language_item_stage in {"introduced", "used"}
            and state.turns_since_language_item >= 3
        ):
            state.active_language_item_stage = "ready_for_reuse"

    @staticmethod
    def _activate_language_item(state: LearningState, item: str, source: str) -> None:
        candidate = " ".join(str(item or "").split())[:180]
        if not candidate or candidate.lower() == state.active_language_item.lower():
            return
        state.active_language_item = candidate
        state.active_language_item_stage = "introduced"
        state.active_language_item_source = source
        state.turns_since_language_item = 0
        state.language_item_reuses = 0

    @staticmethod
    def _extract_queried_language_item(text: str) -> str:
        value = " ".join(str(text or "").split())
        patterns = (
            r"\bwhat\s+does\s+(.{1,60}?)\s+mean\b",
            r"[\"“']?([A-Za-z][A-Za-z' -]{1,58}?)[\"”']?\s*(?:是|的)?\s*(?:什么意思|啥意思)",
            r"[\"“']?([A-Za-z][A-Za-z' -]{1,58}?)[\"”']?.{0,10}(?:怎么读|发音|读音)",
        )
        for pattern in patterns:
            match = re.search(pattern, value, re.IGNORECASE)
            if not match:
                continue
            candidate = " ".join(match.group(1).strip(" \"'“”.,?!，。？！").split())
            words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", candidate)
            if 1 <= len(words) <= 5:
                return " ".join(words)[:80]
        return ""

    @staticmethod
    def _language_item_reuse_hint(state: LearningState) -> str:
        item = state.active_language_item
        if not item:
            return ""
        return (
            f"如果“{item}”能自然表达用户当前正在发展的新信息，在回应里自然复现一次并紧邻中文义；"
            "优先让它迁移到当前话题的新细节，不得复述旧问题、倒退到旧话题、考定义或宣布复习；"
            "不要宣布复习；不能自然承接就继续暂缓，绝不为了塞入它而打断兴趣流。"
        )

    @staticmethod
    def _extract_glossed_language_item(text: str) -> str:
        """Find one explicitly taught item in an assistant reply.

        This deliberately ignores unexplained English.  The response model may
        use ordinary known English freely; only a compact item that it actually
        marked with a Chinese gloss enters the bounded exposure/reuse loop.
        """
        value = " ".join(str(text or "").split())
        patterns = (
            r"\bmeans\s+([A-Za-z][A-Za-z' -]{1,40}?)\s*[（(][\u4e00-\u9fff][^）)]{0,28}[）)]",
            r"[\"“']?([A-Za-z][A-Za-z' -]{1,58}?)[\"”']?\s*(?:的意思是|意思是|表示|指的是)",
            r"[\"“']?([A-Za-z][A-Za-z' -]{1,58}?)[\"”']?\s*[（(][\u4e00-\u9fff][^）)]{0,28}[）)]",
            r"[\"“']?([A-Za-z][A-Za-z' -]{1,58}?)[\"”']?\s+means\s+[\"“']?[\u4e00-\u9fff]",
        )
        for pattern in patterns:
            match = re.search(pattern, value, re.IGNORECASE)
            if not match:
                continue
            candidate = " ".join(match.group(1).strip(" \"'“”.,?!，。？！").split())
            words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", candidate)
            if 1 <= len(words) <= 5:
                return " ".join(words)[:80]
        return ""

    @classmethod
    def observe_assistant_reply(cls, state: LearningState, assistant_text: str) -> None:
        """Track receptive reuse and one newly glossed item after generation."""
        if (
            state.active_language_item_stage == "ready_for_reuse"
            and cls._matches_recast(assistant_text, state.active_language_item)
        ):
            state.active_language_item_stage = "recycled"
            evidence = f"在同一会话中延迟复现：{state.active_language_item}"
            if evidence not in state.completion_evidence:
                state.completion_evidence.append(evidence)

        introduced = cls._extract_glossed_language_item(assistant_text)
        if introduced and introduced.lower() != state.active_language_item.lower():
            cls._activate_language_item(state, introduced, "assistant_gloss")

    @staticmethod
    def build_constraints(move: LearningMove, learner_state: EstimatedLearnerState | None) -> Dict[str, Any]:
        """Return boundaries for the model, not a pre-written reply."""
        state = learner_state.to_dict() if learner_state else {}
        action = move.action
        question_pressure = int(state.get("recent_question_pressure") or 0)
        if action in {"engage_general_topic", "follow_interest_topic"}:
            conversation_shape = (
                "content_reply_no_question"
                if question_pressure >= 2
                else "content_first_optional_question"
            )
        else:
            conversation_shape = "task_driven"
        max_explanation = 1
        if action in {"clarify_lexical_then_resume", "clarify_then_check"}:
            max_explanation = 2
        elif action == "reduce_cognitive_load":
            max_explanation = 1
        first_connection = action in {"connect_first_turn", "connection_opening"}
        forbidden = [
            "无目标寒暄", "重新抛出口语/单词/语法菜单", "一次纠正多个问题",
            "把用户的表达任务直接替他说完",
        ]
        if first_connection:
            forbidden += [
                "首轮立即纠错、评分或要求重说",
                "用说得很好、太棒了等泛泛表扬代替回应具体内容",
                "同时给表达支撑又连续追问",
            ]
        if action == "method_repair":
            forbidden += [
                "填空、造句、跟读或要求用户立刻练习",
                "复用旧句架或引入与投诉前话题无关的新例子",
                "把选择和修复责任重新推给用户",
                "仅因用户上一句用了目标语言就切换成纯目标语言",
            ]
        if action == "address_learning_difficulty":
            forbidden += [
                "使用你指出得很具体等投诉修复口吻",
                "回到为什么想学英语或其他旧训练目标",
                "在尚不知道具体材料时要求填空、造句或跟读",
            ]
        if action == "clarify_lexical_then_resume":
            forbidden += [
                "只给中文提示却不提供用户所问的英文表达",
                "没有被要求时自动布置造句、跟读或推进角色流程",
                "提供未释义的额外英文选项",
            ]
        if action == "adapt_teaching_preference":
            forbidden += [
                "只表示赞同而不立即应用",
                "继续使用用户刚指出有问题的宽泛提问",
                "引入与当前话题无关的新领域",
                "提供未释义的新英文词",
            ]
        if action == "establish_roleplay_scene":
            forbidden += [
                "把场景说明降级成孤立单词或填空造句",
                "讨论产品架构而不立即应用角色",
                "替用户完成双方台词",
                "套用后端预制场景、固定台词或枚举式流程",
                "引入用户当前场景原意中不存在的旧练习",
            ]
        if action == "support_role_line":
            forbidden += [
                "回答产品架构或复述用户的开发分析",
                "代替双方完成整段角色演绎",
                "离开当前角色和当前步骤",
                "只给英文而没有紧邻中文义",
            ]
        if action == "follow_interest_topic":
            forbidden += [
                "回到存储的练习场景菜单或替用户重选场景",
                "把话题降级成孤立单词或填空造句",
                "纯中文闲聊而不引导英文表达",
                "替用户把整句英文说完而不留话口",
                "把话题生硬转回酒店退房/账单/网购等模板场景",
            ]
        if action == "clarify_uncertain_expression":
            forbidden += [
                "把没听清的片段擅自补成完整意思",
                "在语义尚未确认时纠错或给更自然版本",
                "一次提出多个澄清问题",
                "推进场景或切换到无关话题",
            ]
        if action == "engage_general_topic":
            forbidden += [
                "变成练习支架、抛单词/语法/场景菜单或追问学习目标",
                "把对话拽回存储的练习场景",
                "纯中文闲聊而不自然地融进目标语",
                "每轮都长篇讲规则或教学点评",
            ]
        if action in {
            "continue_roleplay_scene", "continue_roleplay_with_recast",
            "continue_roleplay_transfer_branch", "repair_roleplay_contract", "repair_roleplay_line",
        }:
            forbidden += [
                "退出角色进行泛泛表扬或教学点评",
                "要求填空、造句、跟读或重复",
                "跳回 because、movies、music 或其他旧练习",
                "擅自重设场景、角色或把角色演绎改成预制练习步骤",
            ]
        if action == "repair_roleplay_contract":
            forbidden += ["重新从场景开头开始", "抹掉用户已经完成的角色台词"]
        if action == "repair_roleplay_line":
            forbidden += ["在用户重说正确之前推进场景", "引入新的词汇练习"]
        if action == "continue_roleplay_transfer_branch":
            forbidden += ["重播场景开场", "重复索取已提供身份信息", "把迁移说成教学任务"]
        return {
            "primary_goal": move.goal,
            "allowed_action": action,
            "user_must_speak": move.user_must_speak,
            "next_user_action": move.prompt_hint,
            "correction_budget": 1 if move.feedback_mode in {
                "one_correction_then_retry", "one_brief_upgrade_in_role",
                "one_role_line_correction", "natural_conversation_recast",
                "interest_topic_recast",
            } else 0,
            "first_turn_relation_priority": first_connection,
            "support_or_question_budget": 1 if first_connection else None,
            "max_explanation_units": max_explanation,
            "interaction_rhythm": state.get("interaction_rhythm", "protect_flow"),
            "conversation_shape": conversation_shape,
            "recent_question_pressure": question_pressure,
            "language_mode": state.get("language_mode", "target_language"),
            "generation_freedom": "模型自行选择自然语气、例子和措辞；不得改变当前目标或替用户完成表达",
            "forbidden": forbidden,
            "state_basis": state,
        }

    @staticmethod
    def _is_user_driven_topic(text: str) -> bool:
        """用户主导话题判定：不按话题内容分类（任何话题都能作为英语练习载体），
        只看用户是否在主导一个具体话题而不是继续当前场景的角色台词。

        - 中文为主且有实质内容 → 用户在用自己的话聊一个具体内容（很可能就是
          他的兴趣），需要先听懂再教英文表达；
        - 或明确提到场景外的具体话题/实体 → 同样视为用户主导话题。
        """
        t = " ".join(str(text or "").strip().split())
        if not t or not LearningOrchestrator._has_substantive_content(t):
            return False
        # 纯确认/语气词不算用户主导话题（允许“好的知道了”这类确认词拼接）。
        if re.fullmatch(
            r"(?:嗯+|好+|ok+|好的?|知道[了啦]?|明白[了啦]?|继续|然后呢?|对对+|是是+)+"
            r"[。！!？?，,\.]*$",
            t, re.IGNORECASE,
        ):
            return False
        chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in t)
        if chinese >= 3:
            return True
        if re.search(
            r"(老友记|friends|黑袍|the\s+boys|祖国人|homelander|美剧|英剧|韩剧|日剧|"
            r"电视剧|电影|综艺|动漫|动画|漫画|小说|音乐|乐队|歌手|专辑|游戏|球赛|"
            r"演员|导演|角色|剧情|书|歌|新闻|美食|旅游|宠物|运动|爱好|"
            r"tv\s*show|tv\s*series|movie|film|actor|director|album|song|band|"
            r"game|match|book|novel|character|plot|news|food|travel|pet|hobby)",
            t, re.IGNORECASE,
        ):
            return True
        return False

    @staticmethod
    def _is_general_chat(
        text: str, intent: TutorIntent, errors: List[Dict[str, Any]]
    ) -> bool:
        """普通对话判定：用户只是在自然聊天（中文闲聊、一般问题、英语闲聊）。
        带错误的目标语产出也属于自然聊天——由 engage_general_topic 用一处轻量
        重述处理（correction_budget=1），而不是把用户拖进正式修复操练。
        表达求助（“怎么说/什么意思”）仍交给语言支撑分支。
        """
        t = " ".join(str(text or "").strip().split())
        if not t or not LearningOrchestrator._has_substantive_content(t):
            return False
        if re.fullmatch(
            r"(?:嗯+|好+|ok+|好的?|知道[了啦]?|明白[了啦]?|继续|然后呢?|对对+|是是+)+"
            r"[。！!？?，,\.]*$",
            t, re.IGNORECASE,
        ):
            return False
        # 表达求助/词义类（"怎么说/什么意思"等）交给语言支撑分支。
        if re.search(
            r"(怎么说|怎么表达|怎么讲|什么意思|啥意思|怎么读|怎么用|不会说|说不来|"
            r"how\s+to\s+say|what\s+does\b|means?\b)",
            t, re.IGNORECASE,
        ):
            return False
        chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in t)
        if chinese >= 3:
            return True
        # 一般问题 / 闲聊意图（英语问句、问候、英语陈述等）也算普通对话。
        return intent in {TutorIntent.CHITCHAT, TutorIntent.QUESTION, TutorIntent.GREETING}

    @staticmethod
    def _attempted_target_language(text: str, target_lang: str) -> bool:
        if target_lang == "ja":
            return any("\u3040" <= ch <= "\u30ff" for ch in text)
        latin = sum(ch.isascii() and ch.isalpha() for ch in text)
        return latin >= 2

    @staticmethod
    def _has_substantive_content(text: str) -> bool:
        """Distinguish a real opening from a bare greeting or filler."""
        compact = " ".join((text or "").strip().lower().split())
        if not compact:
            return False
        bare_openings = {
            "hi", "hello", "hey", "hi!", "hello!", "hey!",
            "你好", "你好呀", "嗨", "哈喽", "在吗",
        }
        if compact in bare_openings:
            return False
        chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in compact)
        words = [word for word in compact.replace("，", " ").replace("。", " ").split() if word]
        return chinese >= 3 or len(words) >= 3 or len(compact) >= 12

    @staticmethod
    def _scaffold_hint(level: int) -> str:
        return {
            1: "只给 1 个贴近当前话题的英文关键词并紧跟中文义，再用中文说明这一句要表达的具体意思。",
            2: "先用中文说明具体表达目标，再给一个可填空的半句句架，保留关键信息让用户自己完成。",
            3: "给一个很短的中英对照示范，再让用户只替换其中一个明确的信息重新说。",
        }.get(level, "给最小提示并让用户继续表达。")
