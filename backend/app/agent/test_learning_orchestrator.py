import asyncio
import re
import unittest
from unittest.mock import patch

import config
from agent.conversation_director import ConversationDirector
from agent.intent_router import IntentRouter, TutorIntent
from agent.learner_state import LearnerState, LearnerStateEstimator
from agent.learning_orchestrator import LearningMove, LearningOrchestrator
from agent.tutor_agent import SessionContext, TutorAgent
from agent.tutor_policy import TutorPolicy, TutorPolicyPlanner
from agent.turn_controller import TurnController
from skill.teaching_skills import PRESET_SKILLS


class IntentRouterFarewellTests(unittest.TestCase):
    def test_temporal_later_in_material_question_is_not_a_farewell(self):
        text = (
            "He brings up self-media pulling talent away, and later suggests "
            "Moodyz may be leaning on established performers. Are those two "
            "different explanations?"
        )
        intent = IntentRouter().classify(text)
        state = LearnerStateEstimator().estimate(
            user_text=text, intent_result=intent, target_lang="en", cefr="A1"
        )

        self.assertEqual(intent.intent, TutorIntent.QUESTION)
        self.assertEqual(state.exit_likelihood, 0.0)
        self.assertNotIn("explicit_farewell", state.signals)

    def test_explicit_see_you_later_remains_a_farewell(self):
        intent = IntentRouter().classify("Talk to you later!")
        self.assertEqual(intent.intent, TutorIntent.FAREWELL)


class LearningOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.engine = LearningOrchestrator()
        self.state = self.engine.start(
            {"name": "daily_talk", "goal": "Practice English naturally"},
            {"cefr": "A2"},
        )

    def move(self, text, intent=TutorIntent.CHITCHAT, errors=None):
        return self.engine.plan(
            self.state,
            user_text=text,
            intent=intent,
            errors=errors or [],
            target_lang="en",
        )

    def test_bare_greeting_opens_connection_without_assigning_a_task(self):
        move = self.move("Hi", TutorIntent.GREETING)
        self.assertEqual(move.action, "connection_opening")
        self.assertEqual(move.stage, "flow")
        self.assertFalse(move.user_must_speak)
        self.assertIn("不宣布课程任务", move.prompt_hint)

    def test_first_substantive_turn_connects_before_correction_or_transfer(self):
        move = self.move(
            "I feel lonely after moving here",
            errors=[{"type": "article", "excerpt": "moving here"}],
        )
        constraints = self.engine.build_constraints(move, None)
        self.assertEqual(move.action, "connect_first_turn")
        self.assertEqual(move.feedback_mode, "meaning_first")
        self.assertFalse(move.user_must_speak)
        self.assertEqual(constraints["correction_budget"], 0)
        self.assertEqual(constraints["support_or_question_budget"], 1)
        self.assertTrue(constraints["first_turn_relation_priority"])
        # P11（2026-09-22）：追问从"二选一/也可以都不加"改为**默认要做**（只问一个）。
        self.assertIn("只问一个", move.prompt_hint)

    def test_no_target_language_attempt_raises_scaffold(self):
        self.move("Hi", TutorIntent.GREETING)
        move = self.move("我不知道怎么说", TutorIntent.CHITCHAT)
        self.assertEqual(move.action, "prompt_output")
        self.assertEqual(move.support_level, 1)
        self.assertIn("关键词", move.prompt_hint)

    def test_idle_error_turn_gets_light_recast_in_natural_chat(self):
        # 空闲自然对话中，带错误的目标语产出走轻量重述（engage_general_topic，
        # correction_budget=1），不把用户拖进正式修复操练。
        self.move("Hi", TutorIntent.GREETING)
        move = self.move(
            "I didn't went yesterday",
            errors=[{"type": "en_past_tense", "excerpt": "didn't went"}],
        )
        self.assertEqual(move.action, "engage_general_topic")
        self.assertEqual(move.feedback_mode, "natural_conversation_recast")
        constraints = self.engine.build_constraints(move, None)
        self.assertEqual(constraints["correction_budget"], 1)
        # 话题继续，仍是自然聊天；下一轮进入纠错冷却，不机械重复点评。
        flow = self.move(
            "I didn't go to work last Monday",
            errors=[{"type": "en_past_tense", "excerpt": "didn't went"}],
        )
        self.assertEqual(flow.action, "engage_general_topic")
        self.assertEqual(flow.feedback_mode, "natural_conversation_meaning_first")
        self.assertEqual(self.engine.build_constraints(flow, None)["correction_budget"], 0)

    def test_uncertain_general_expression_is_clarified_before_correction(self):
        self.move("Hi", TutorIntent.GREETING)
        move = self.engine.plan(
            self.state,
            user_text="I like the mesh... fresh wine",
            intent=TutorIntent.CHITCHAT,
            errors=[{"type": "director_naturalness", "excerpt": "fresh wine"}],
            target_lang="en",
            conversation_plan={
                "uncertain_spans": ["the mesh... fresh wine"],
                "unresolved_slots": ["用户想比较的两部剧"],
                "pragmatic_check": "clarify",
            },
        )
        self.assertEqual(move.action, "clarify_uncertain_expression")
        self.assertEqual(move.feedback_mode, "meaning_clarification")
        self.assertEqual(self.engine.build_constraints(move, None)["correction_budget"], 0)

    def test_roleplay_error_keeps_in_role_recast(self):
        # 角色场景中的错误走 in-role 单点重述，不退出角色。
        self.move("Hi", TutorIntent.GREETING)
        self.state.roleplay_active = True
        self.state.scenario = "phone store"
        self.state.tutor_role = "店员"
        move = self.move(
            "I didn't went yesterday",
            errors=[{"type": "en_past_tense", "excerpt": "didn't went"}],
        )
        self.assertEqual(move.action, "continue_roleplay_with_recast")
        self.assertEqual(move.feedback_mode, "one_brief_upgrade_in_role")

    def test_controller_general_frame_suspends_stale_roleplay_for_english_followup(self):
        self.move("Hi", TutorIntent.GREETING)
        self.state.roleplay_active = True
        self.state.scenario = "hotel front desk"
        self.state.tutor_role = "前台"
        move = self.engine.plan(
            self.state,
            user_text="My favorite team is Team Spirit.",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
            conversation_plan={"interaction_frame": "general", "response_need": "general_chat"},
        )
        self.assertEqual(move.action, "follow_interest_topic")
        self.assertNotEqual(move.action, "continue_roleplay_scene")

    def test_method_complaint_does_not_force_practice(self):
        self.move("Hi", TutorIntent.GREETING)
        move = self.move("为什么你又问我今天怎么样", TutorIntent.META_QUESTION)
        self.assertEqual(move.action, "method_repair")
        self.assertFalse(move.user_must_speak)

    def test_lexical_clarification_is_constrained_not_a_scene_reset(self):
        estimator = LearnerStateEstimator()
        self.state.stage = "attempt"
        text = "What does blended mean?"
        user_state = estimator.estimate(
            user_text=text,
            intent_result=IntentRouter().classify(text),
            learning_state=self.state,
            history=[],
            target_lang="en",
        )
        move = self.engine.plan(
            self.state,
            user_text=text,
            intent=TutorIntent.QUESTION,
            errors=[],
            target_lang="en",
            learner_state=user_state,
        )
        constraints = self.engine.build_constraints(move, user_state)
        self.assertEqual(move.action, "clarify_lexical_then_resume")
        self.assertEqual(constraints["max_explanation_units"], 2)
        self.assertTrue(constraints["user_must_speak"])

    def test_lexical_item_moves_from_preview_to_delayed_natural_reuse(self):
        estimator = LearnerStateEstimator()
        self.state.stage = "flow"
        text = "What does go through mean?"
        user_state = estimator.estimate(
            user_text=text,
            intent_result=IntentRouter().classify(text),
            learning_state=self.state,
            history=[],
            target_lang="en",
        )
        self.engine.plan(
            self.state,
            user_text=text,
            intent=TutorIntent.QUESTION,
            errors=[],
            target_lang="en",
            learner_state=user_state,
        )
        self.assertEqual(self.state.active_language_item, "go through")
        self.assertEqual(self.state.active_language_item_stage, "introduced")

        self.move("I will try another card.")
        self.move("The machine still says no.")
        due_move = self.move("I may ask the cashier for help.")
        self.assertEqual(self.state.active_language_item_stage, "ready_for_reuse")
        self.assertIn("go through", due_move.prompt_hint)
        self.assertIn("不要宣布复习", due_move.prompt_hint)
        self.assertIn("不得复述旧问题", due_move.prompt_hint)

        self.engine.observe_assistant_reply(
            self.state, "Your payment still didn't go through. 付款还是没有成功。"
        )
        self.assertEqual(self.state.active_language_item_stage, "recycled")
        self.assertTrue(any("延迟复现" in item for item in self.state.completion_evidence))

    def test_assistant_gloss_starts_one_bounded_language_item_loop(self):
        self.engine.observe_assistant_reply(
            self.state,
            "Playing T often means taking risks（冒险）and making quick moves.",
        )
        self.assertEqual(self.state.active_language_item, "taking risks")
        self.assertEqual(self.state.active_language_item_stage, "introduced")
        self.assertEqual(self.state.active_language_item_source, "assistant_gloss")

        # Asking about the item shows noticing, not productive command of it.
        estimator = LearnerStateEstimator()
        text = "Taking risks 是啥意思？"
        user_state = estimator.estimate(
            user_text=text,
            intent_result=IntentRouter().classify(text),
            learning_state=self.state,
            history=[],
            target_lang="en",
        )
        self.engine.plan(
            self.state,
            user_text=text,
            intent=TutorIntent.QUESTION,
            errors=[],
            target_lang="en",
            learner_state=user_state,
        )
        self.assertEqual(self.state.active_language_item_stage, "introduced")
        self.assertEqual(self.state.language_item_reuses, 0)

    def test_recurring_current_error_can_surface_after_one_cooldown_turn(self):
        self.move("Hi", TutorIntent.GREETING)
        first = self.move(
            "I didn't went yesterday",
            errors=[{
                "type": "en_past_tense", "excerpt": "didn't went",
                "correction": "I didn't go yesterday.",
            }],
        )
        self.assertEqual(first.feedback_mode, "natural_conversation_recast")
        second = self.move(
            "I didn't watched it",
            errors=[{
                "type": "en_past_tense", "excerpt": "didn't watched",
                "correction": "I didn't watch it.", "priority": 3,
                "is_recurring": True,
            }],
        )
        self.assertEqual(second.feedback_mode, "natural_conversation_recast")

    def test_explicit_expression_evaluation_overrides_ordinary_flow(self):
        estimator = LearnerStateEstimator()
        self.state.stage = "flow"
        text = "Hi, my card can't pay. 我这样说自然吗？"
        user_state = estimator.estimate(
            user_text=text,
            intent_result=IntentRouter().classify(text),
            learning_state=self.state,
            history=[],
            target_lang="en",
            cefr="A2",
        )
        self.assertEqual(user_state.communicative_intent, "expression_evaluation")
        move = self.engine.plan(
            self.state,
            user_text=text,
            intent=TutorIntent.QUESTION,
            errors=[],
            target_lang="en",
            learner_state=user_state,
        )
        self.assertEqual(move.action, "evaluate_expression_then_resume")
        self.assertEqual(move.feedback_mode, "one_requested_correction")
        self.assertFalse(move.user_must_speak)


class ReplyContractCompositionTests(unittest.TestCase):
    def test_language_sidecar_keeps_direct_answer_and_policy_boundaries(self):
        policy = TutorPolicy(
            response_goal="解释用户刚问的短语",
            must_do=["先回答go through在付款语境中的意思"],
            must_avoid=["恢复收银员角色"],
        )
        move = LearningMove(
            stage="clarify",
            action="clarify_lexical_then_resume",
            goal="解释当前短语",
            user_must_speak=False,
            feedback_mode="meaning_first",
        )
        contract = TutorAgent._build_reply_contract(
            policy=policy,
            move=move,
            constraints={
                "correction_budget": 0,
                "max_explanation_units": 2,
                "conversation_shape": "task_driven",
                "language_mode": "mixed_support",
                "forbidden": ["没有被要求时自动布置跟读"],
            },
            turn_decision={
                "primary_frame": "language_sidecar",
                "speaker_position": "coach",
                "response_sequence": ["answer_language_question"],
                "response_obligations": ["按原句语境解释"],
                "confidence": 0.9,
            },
            director={},
        )
        self.assertEqual(contract["required_blocks"], ["answer_language_question"])
        self.assertIn("direct_language_answer", contract["permitted_blocks"])
        self.assertIn("one_contextual_example", contract["permitted_blocks"])
        self.assertIn("先回答go through在付款语境中的意思", contract["response_obligations"])
        self.assertIn("恢复收银员角色", contract["prohibited_behaviors"])
        self.assertNotIn("本轮主动纠正用户没有询问的语言问题", contract["prohibited_behaviors"])

    def test_general_no_question_shape_does_not_reintroduce_template_question(self):
        policy = TutorPolicy(response_goal="承接用户刚说的剧集内容")
        move = LearningMove(
            stage="flow",
            action="follow_interest_topic",
            goal="自然聊用户选择的话题",
            user_must_speak=False,
            feedback_mode="meaning_first",
        )
        contract = TutorAgent._build_reply_contract(
            policy=policy,
            move=move,
            constraints={
                "correction_budget": 0,
                "conversation_shape": "content_reply_no_question",
                "language_mode": "mixed_support",
            },
            turn_decision={
                "primary_frame": "general",
                "response_sequence": ["respond_to_latest_input"],
            },
            director={},
        )
        self.assertNotIn(
            "one_optional_question_if_it_genuinely_advances_the_topic",
            contract["permitted_blocks"],
        )
        self.assertIn("用例行追问或问号收尾", contract["prohibited_behaviors"])
        self.assertIn("本轮主动纠正用户没有询问的语言问题", contract["prohibited_behaviors"])

    def test_in_role_contract_allows_one_recast_without_losing_counterpart_turn(self):
        policy = TutorPolicy(response_goal="作为店员回应付款失败")
        move = LearningMove(
            stage="practice",
            action="continue_roleplay_with_recast",
            goal="继续收银台交涉",
            user_must_speak=True,
            feedback_mode="one_brief_upgrade_in_role",
        )
        contract = TutorAgent._build_reply_contract(
            policy=policy,
            move=move,
            constraints={
                "correction_budget": 1,
                "conversation_shape": "task_driven",
                "language_mode": "mixed_support",
            },
            turn_decision={
                "primary_frame": "in_role",
                "speaker_position": "counterpart",
                "response_sequence": ["respond_as_counterpart"],
            },
            director={},
        )
        self.assertEqual(contract["teaching"]["correction_budget"], 1)
        self.assertIn("counterpart_turn", contract["permitted_blocks"])
        self.assertIn("one_brief_recast", contract["permitted_blocks"])

    def test_authoritative_director_prompt_still_contains_final_reply_contract(self):
        agent = TutorAgent.__new__(TutorAgent)
        director = {
            "_meta": {"source": "director_model"},
            "content_need": "回答用户最新语言问题",
        }
        contract = {
            "version": "reply-contract-v1",
            "required_blocks": ["answer_language_question"],
            "response_obligations": ["先回答词义"],
            "permitted_blocks": ["direct_language_answer"],
            "prohibited_behaviors": ["恢复旧角色"],
        }
        ctx = SessionContext(
            session_id="s-contract",
            user_id="u-contract",
            target_lang="en",
            conversation_plan=director,
            scene_context_set=True,
        )
        prompt = agent._build_control_prompt(ctx, {
            "conversation_director": director,
            "constraints": {"turn_contract": {"reply_contract": contract}},
        })
        self.assertIn("reply-contract-v1", prompt)
        self.assertIn("prohibited_behaviors", prompt)
        self.assertIn("本轮最终回复边界", prompt)


class ControlledReplyContractTests(unittest.TestCase):
    def test_explicit_naturalness_check_rejects_generic_praise(self):
        user = "Hi, my card can't pay. Can you help me? 我这样说自然吗？"
        prepared = {
            "learning_move": {"action": "evaluate_expression_then_resume"},
            "constraints": {
                "language_mode": "mixed_support",
                "turn_contract": {
                    "explicit_expression_evaluation": True,
                    "current_focus_terms": [],
                },
            },
        }
        weak = "你说得很自然！Hi, my card can't pay. Can you help me?"
        violations = TutorAgent._controlled_reply_violations(weak, prepared, user)
        self.assertTrue(any("具体判断" in item or "无依据" in item for item in violations))
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        repaired = TutorAgent._enforce_controlled_reply(
            weak, prepared, ctx, user_transcript=user
        )
        self.assertIn("My card isn’t working", repaired)
        self.assertIn("不自然", repaired)

    def test_short_followup_focus_blocks_replaying_previous_answer(self):
        focus = TutorAgent._short_turn_focus_terms("Doesn't have qr code?")
        self.assertIn("qr", focus)
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "turn_contract": {"current_focus_terms": focus},
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "你可以用现金或另一张卡。You can use cash or another card.",
            prepared,
            "Doesn't have qr code?",
        )
        self.assertTrue(any("新增焦点" in item for item in violations))

    def test_payment_fragment_is_interpreted_as_merchant_acceptance_question(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.active_scene = "open_roleplay"
        ctx.scenario_brief = "在国外超市自助结账，银行卡刷不过，向工作人员询问付款方式"
        ctx.scenario_role = "超市工作人员"
        ctx.messages = [
            {"role": "user", "content": "My card won't go through."},
            {"role": "assistant", "content": "You can use another payment method."},
        ]
        contract = TutorAgent._payment_method_turn_contract(
            ctx, "I doesn't have qr code Doesn't have qr code"
        )
        self.assertEqual(contract["kind"], "payment_method_availability")
        self.assertEqual(contract["method"], "QR code")
        self.assertEqual(contract["mode"], "in_role")
        self.assertEqual(contract["natural_question"], "Can I pay by QR code?")

    def test_question_wording_request_stays_a_one_turn_language_sidecar(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.scenario_brief = "在超市询问付款方式"
        contract = TutorAgent._payment_method_turn_contract(
            ctx, "when doesn't have qr code 这个疑问句"
        )
        self.assertEqual(contract["mode"], "language_sidecar")
        self.assertEqual(contract["natural_question"], "Can I pay by QR code?")

    def test_payment_naturalness_check_is_not_overridden_as_worker_turn(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.scenario_brief = "在超市询问付款方式"
        contract = TutorAgent._payment_method_turn_contract(
            ctx, "Can I use cash 这样说自然吗？"
        )
        self.assertEqual(contract["mode"], "language_sidecar")

    def test_payment_statement_is_not_mistaken_for_availability_question(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.scenario_brief = "在超市询问付款方式"
        contract = TutorAgent._payment_method_turn_contract(
            ctx, "Alright, I'll use another card."
        )
        self.assertEqual(contract, {})

    def test_material_summary_card_is_not_mistaken_for_payment_card(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.active_scene = "material"
        ctx.scenario_brief = "讨论导入视频的正文和评论区"
        contract = TutorAgent._payment_method_turn_contract(
            ctx,
            "The card says the comments add different views. Did anyone mention something the video itself doesn't?",
        )
        self.assertEqual(contract, {})

    def test_explicit_card_acceptance_question_still_routes_as_payment(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        contract = TutorAgent._payment_method_turn_contract(ctx, "Do you accept card?")
        self.assertEqual(contract["kind"], "payment_method_availability")
        self.assertEqual(contract["method"], "card")

    def test_term_meaning_is_anchored_to_its_original_payment_utterance(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [
            {"role": "user", "content": "My card won't go through."},
            {"role": "assistant", "content": "Can you use another card?"},
        ]
        decision = {"primary_frame": "language_sidecar"}
        contract = TutorAgent._language_question_turn_contract(
            ctx, "What does go through mean?", decision
        )
        self.assertEqual(contract["term"].lower(), "go through")
        self.assertIn("won't go through", contract["source_utterance"])
        self.assertEqual(contract["semantic_domain"], "payment")

    def test_payment_meaning_rejects_an_unrelated_dictionary_sense(self):
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "term_meaning",
                        "term": "go through",
                        "source_utterance": "My card won't go through.",
                        "semantic_domain": "payment",
                        "exclusive_current_question": True,
                    },
                },
                "conversation_director": {"interaction_frame": "language_sidecar"},
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "Go through 在这里是经历、办理的意思。",
            prepared,
            "What does go through mean?",
        )
        self.assertTrue(any("付款语境" in item for item in violations))

    def test_language_answer_cannot_replace_the_current_term_with_stale_history(self):
        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "term_meaning",
                        "term": "go through",
                        "source_utterance": "My card won't go through.",
                        "semantic_domain": "payment",
                        "exclusive_current_question": True,
                    },
                },
            },
        }
        stale = "cash 的发音是 /kæʃ/。Can I pay in cash?"
        violations = TutorAgent._controlled_reply_violations(
            stale, prepared, "Through, what does go through mean? 喂喂喂。"
        )
        self.assertTrue(any("本轮明确询问" in item for item in violations))

    def test_noisy_asr_term_can_resolve_to_strong_immediate_context_candidate(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [
            {
                "role": "assistant",
                "content": '\"Synonyms\" are words that have similar meanings.',
            },
        ]
        contract = TutorAgent._language_question_turn_contract(
            ctx,
            "Cinema 是啥意思？",
            {"primary_frame": "language_sidecar"},
        )
        self.assertTrue(contract["asr_term_unverified"])
        self.assertIn("Synonyms", contract["contextual_term_candidates"])

        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": contract,
                },
            },
        }
        resolved = TutorAgent._controlled_reply_violations(
            "Synonyms 的意思是同义词。",
            prepared,
            "Cinema 是啥意思？",
        )
        self.assertFalse(any("可能是语音误识别" in item for item in resolved))

        unrelated = TutorAgent._controlled_reply_violations(
            "Formal 的意思是正式的。",
            prepared,
            "Cinema 是啥意思？",
        )
        self.assertTrue(any("可能是语音误识别" in item for item in unrelated))

    def test_payment_context_answer_cannot_mix_in_unrelated_dictionary_senses(self):
        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "term_meaning",
                        "term": "go through",
                        "source_utterance": "My card won't go through.",
                        "semantic_domain": "payment",
                        "exclusive_current_question": True,
                    },
                },
            },
        }
        mixed_senses = (
            "go through 是经历或仔细检查；Your payment didn't go through "
            "就是付款没有成功。"
        )
        violations = TutorAgent._controlled_reply_violations(
            mixed_senses, prepared, "What does go through mean?"
        )
        self.assertTrue(any("无关字典义" in item for item in violations))

    def test_exclusive_language_question_rejects_stale_presence_and_choice(self):
        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "term_meaning",
                        "term": "go through",
                        "source_utterance": "My card won't go through.",
                        "semantic_domain": "payment",
                        "exclusive_current_question": True,
                    },
                },
            },
        }
        stale = "我还在。go through 和 cash 你想继续聊哪个？"
        violations = TutorAgent._controlled_reply_violations(
            stale, prepared, "What does go through mean?"
        )
        self.assertTrue(any("在不在" in item for item in violations))
        self.assertTrue(any("继续哪个词" in item for item in violations))

    def test_pronunciation_answer_must_name_the_current_term(self):
        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "pronunciation",
                        "term": "mesh",
                        "exclusive_current_question": True,
                    },
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "cash 的发音是 /kæʃ/。", prepared, "Mesh 的发音是 mesh，对吧？"
        )
        self.assertTrue(any("本轮明确询问" in item for item in violations))

    def test_short_confirmation_is_compacted_without_replaying_the_definition(self):
        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "brief_confirmation",
                        "response_style": "one_short_sentence",
                    },
                },
            },
        }
        reply = TutorAgent._compact_language_sidecar_reply(
            '“Go through” 在这里的意思是支付成功。', prepared, "对吧？"
        )
        self.assertEqual(reply, "对。")

    def test_pronunciation_answer_drops_the_stale_scene_example(self):
        prepared = {
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {
                        "kind": "pronunciation",
                        "term": "cash",
                        "response_style": "one_short_sentence",
                    },
                },
            },
        }
        reply = TutorAgent._compact_language_sidecar_reply(
            "对，cash 发音就是 /kæʃ/。想问能不能用现金，就说 Can I pay in cash?",
            prepared,
            "cash 的发音是 cash，对吧？",
        )
        self.assertEqual(reply, "对，cash 发音就是 /kæʃ/。")

    def test_language_contract_marks_pronunciation_as_a_single_point_answer(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        contract = TutorAgent._language_question_turn_contract(
            ctx,
            "cash 的发音是这样，对吧？",
            {"primary_frame": "language_sidecar"},
        )
        self.assertEqual(contract["kind"], "pronunciation")
        self.assertEqual(contract["term"].lower(), "cash")
        self.assertEqual(contract["response_style"], "one_short_sentence")

    def test_pronunciation_question_cannot_be_stolen_by_payment_keywords(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.scenario_brief = "在超市询问付款方式"
        contract = TutorAgent._payment_method_turn_contract(
            ctx, "Cash 的发音是 cash，对吧？"
        )
        self.assertEqual(contract, {})

    def test_noisy_pronunciation_term_uses_recent_language_context(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [
            {"role": "user", "content": "Can I use cash?"},
            {"role": "assistant", "content": "Can I pay in cash?"},
        ]
        contract = TutorAgent._language_question_turn_contract(
            ctx,
            "Has 的发音是 cash，对吧？",
            {"primary_frame": "language_sidecar"},
        )
        self.assertEqual(contract["kind"], "pronunciation")
        self.assertEqual(contract["term"].lower(), "cash")
        self.assertTrue(contract["asr_context_repaired"])

    def test_direct_role_line_restores_counterpart_after_language_sidecar(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.active_scene = "open_roleplay"
        ctx.scenario_brief = "在国外超市自助结账，向工作人员询问付款方式"
        ctx.scenario_role = "超市工作人员"
        learner = type("Learner", (), {"communicative_intent": "open_expression"})()
        plan = {
            "response_need": "expression_support",
            "interaction_frame": "language_sidecar",
            "speaker_position": "coach",
            "reply_draft": "这个说法很自然。",
        }
        TutorAgent._stabilize_turn_plan(
            ctx,
            "Can I use cash?",
            learner,
            plan,
            previous_interaction_frame="language_sidecar",
        )
        self.assertEqual(plan["interaction_frame"], "in_role")
        self.assertEqual(plan["speaker_position"], "counterpart")
        self.assertEqual(plan["response_need"], "roleplay_turn")
        self.assertEqual(plan["reply_draft"], "")

    def test_payment_role_turn_rejects_coach_commentary_and_repairs_as_worker(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_context_set = True
        ctx.scenario_brief = "在超市询问付款方式"
        contract = TutorAgent._payment_method_turn_contract(ctx, "Can I use cash?")
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "turn_contract": {"pragmatic_intent": contract},
                "conversation_director": {
                    "_meta": {"source": "director_model"},
                    "interaction_frame": "in_role",
                    "speaker_position": "counterpart",
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                },
            },
        }
        coaching = "当然可以，这个说法很自然。Can I use cash? 就是我能用现金吗？"
        violations = TutorAgent._controlled_reply_violations(
            coaching, prepared, "Can I use cash?"
        )
        self.assertTrue(any("工作人员身份" in item for item in violations))
        repaired = TutorAgent._enforce_controlled_reply(
            coaching, prepared, ctx, "Can I use cash?"
        )
        self.assertIn("pay in cash", repaired)
        self.assertNotIn("这个说法", repaired)

    def test_unknown_terms_respect_a2_budget_without_forced_glosses(self):
        prepared = {
            "learning_move": {"action": "listen_and_deepen"},
            "constraints": {
                "language_mode": "target_language",
                "adaptive_language_support": {
                    "known_terms": ["you", "remember", "any", "from", "it"],
                    "new_term_budget": 1,
                },
            },
        }
        reply = "Do you remember any funny or exciting scene from it?"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "i can't remember the character"
        )
        self.assertTrue(any("超过预算" in item for item in violations))
        self.assertFalse(any("没有中文语义预告" in item for item in violations))

    def test_a2_mixed_support_keeps_comprehensible_english_floor(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.profile = {"cefr": "A2"}
        learner = type("Learner", (), {"language_mode": "mixed_support"})()
        contract = TutorAgent._build_adaptive_language_contract(
            ctx, "我想继续聊 CS2。", learner
        )
        self.assertIn("目标语优先", contract["presentation_goal"])
        self.assertIn("不等于他不要英文输入", contract["presentation_goal"])

    def test_spoken_response_budget_is_shorter_for_language_sidecar(self):
        sidecar = {
            "constraints": {
                "turn_contract": {"decision": {"primary_frame": "language_sidecar"}}
            }
        }
        general = {
            "constraints": {"turn_contract": {"decision": {"primary_frame": "content"}}}
        }
        self.assertEqual(TutorAgent._spoken_response_token_budget(sidecar, 320), 144)
        self.assertEqual(TutorAgent._spoken_response_token_budget(general, 320), 220)
        self.assertEqual(TutorAgent._spoken_response_token_budget(general, 120), 120)

    def test_chinese_semantic_preview_can_support_one_new_term(self):
        prepared = {
            "learning_move": {"action": "listen_and_deepen"},
            "constraints": {
                "language_mode": "mixed_support",
                "adaptive_language_support": {
                    "known_terms": ["do", "you", "like", "this"],
                    "new_term_budget": 1,
                },
            },
        }
        reply = "你喜欢这部剧里的主角吗？Do you like the main character?"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "我记不住角色名字"
        )
        self.assertFalse(any("没有中文语义预告" in item for item in violations))

    def test_topic_anchor_blocks_unrequested_lyrics_detour(self):
        prepared = {
            "learning_move": {"action": "adapt_teaching_preference"},
            "constraints": {
                "language_mode": "chinese_support",
                "conversation_anchor": {"topic": "Stranger Things"},
            },
        }
        reply = "可以试试简单英文歌曲的 lyrics，lyrics 是歌词。"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "不开字幕时遇到不认识的词应该怎么办"
        )
        self.assertTrue(any("禁止跳到歌曲" in item for item in violations))

    def test_user_requested_topic_change_is_not_unrequested_drift(self):
        """用户自己要求换话题时，回复顺着换不能被判成模型自漂移。

        真机会话：用户说 "Let's not talk about coffee."，护栏却回
        "那就接着聊 An Americano，不换题。"，把用户发起的换题当成模型漂移，
        并让下一轮用户被迫质问。原判据只看回复里有没有出现
        电影/歌曲/食物/时尚/猫狗，完全不知道用户刚要求换题。
        """
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_anchor": {"topic": "An Americano"},
            },
        }
        # 回复里出现了新话题（食物），但话题是用户要求换的 → 不算漂移。
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        reply = "Sure, let's switch. Do you like noodles or rice better?"
        user = "Can we change the topic? Let's not talk about coffee."
        self.assertEqual(TutorAgent._unrequested_topic_drift(reply, user), "")
        violations = TutorAgent._controlled_reply_violations(reply, prepared, user)
        self.assertFalse(any("用户没有换题" in item for item in violations))
        enforced = TutorAgent._enforce_controlled_reply(
            reply, prepared, ctx, user_transcript=user
        )
        self.assertEqual(enforced, reply)
        self.assertNotIn("不换题", enforced)

    def test_topic_change_request_detector_excludes_negated_corrections(self):
        """“我没叫你换话题”是纠正教学动作，不是换题请求。"""
        D = TutorAgent._user_requests_topic_change
        for asked in (
            "So can we change the topic? Let's not talk about coffee.",
            "So can we talk about the other typical?换换一话题。不聊咖啡。",
            "换个话题吧",
            "我们聊点别的",
            "别说这个了",
            "can we switch to another topic",
        ):
            self.assertTrue(D(asked), asked)
        for not_asked in (
            "我没叫你换个话题，我只是觉得 OK, no problem 太简单了，不想跟读。",
            "别换话题，我们先把这句说完",
            "don't change the topic",
            "I didn't ask you to change the topic",
            "An Americano, please.",
            "",
        ):
            self.assertFalse(D(not_asked), not_asked)

    def test_learning_difficulty_does_not_translate_a_concept_user_already_used(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "address_learning_difficulty"},
            "constraints": {"user_must_speak": False, "language_mode": "chinese_support"},
        }
        transcript = "我最近在看一部英语电视剧，但是没有字幕，有很多地方听不懂。"
        low_value = "看没有 subtitle 的英语剧确实不容易，subtitle 就是字幕的意思哦。"
        violations = TutorAgent._controlled_reply_violations(
            low_value, prepared, transcript
        )
        self.assertTrue(any("用户已经明确说出字幕" in item for item in violations))
        enforced = TutorAgent._enforce_controlled_reply(
            low_value, prepared, ctx, user_transcript=transcript
        )
        self.assertIn("你最近看的是哪一部", enforced)
        self.assertNotIn("subtitle", enforced.lower())

    def test_mixed_support_allows_short_comprehensible_english_without_translation(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "connect_first_turn"},
            "constraints": {"language_mode": "mixed_support", "user_must_speak": False},
        }
        transcript = (
            "我先说一下为什么喜欢这部剧。"
            "i watch the strange things yesterday. i like it because the story is interesting."
        )
        pure_english = "Stranger Things is a hit! What part do you like most?"
        violations = TutorAgent._controlled_reply_violations(
            pure_english, prepared, transcript
        )
        self.assertFalse(any("mixed_support" in item for item in violations))
        enforced = TutorAgent._enforce_controlled_reply(
            pure_english, prepared, ctx, user_transcript=transcript
        )
        self.assertEqual(enforced, pure_english)
        self.assertNotIn("我先用中文接住", enforced)

    def test_ordinary_reply_rejects_automatic_sentence_translation(self):
        prepared = {
            "learning_move": {"action": "listen_and_deepen"},
            "constraints": {"language_mode": "mixed_support"},
        }
        reply = "Playing CT requires patience. 这句话的意思是玩 CT 需要耐心。"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "那你更喜欢 CT 还是 T？"
        )
        self.assertTrue(any("没有请求翻译" in item for item in violations))

        asked = TutorAgent._controlled_reply_violations(
            reply, prepared, "Playing CT requires patience 是什么意思？"
        )
        self.assertFalse(any("没有请求翻译" in item for item in asked))

    def test_first_connection_rejects_generic_praise_and_immediate_practice(self):
        prepared = {
            "learning_move": {"action": "connect_first_turn"},
            "constraints": {"correction_budget": 0},
        }
        violations = TutorAgent._controlled_reply_violations(
            "Great job! Let's practice that again.", prepared
        )
        self.assertTrue(any("泛泛表扬" in item for item in violations))
        self.assertTrue(any("立即布置练习" in item for item in violations))

    def test_generic_opening_is_replaced_by_concrete_mission(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.learning_state = LearningOrchestrator().start({}, {})
        reply = TutorAgent._enforce_controlled_reply(
            "Hi! How are you today?",
            {"learning_move": {"action": "mission_opening"}},
            ctx,
        )
        self.assertNotIn("How are you", reply)
        self.assertIn("I think", reply)

    def test_generic_course_menu_is_replaced_during_method_repair(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        reply = TutorAgent._enforce_controlled_reply(
            "抱歉。你想练习英语的哪个方面？练口语、记单词还是语法？",
            {"learning_move": {"action": "method_repair"}},
            ctx,
        )
        # 0924-05：用户这轮没说中文（user_transcript 为空）⇒ 兜底必须说目标语，
        # 不能再吐中文元叙述。
        self.assertRegex(reply, r"keep going")
        self.assertFalse(re.search(r"[\u4e00-\u9fff]", reply), reply)
        self.assertNotIn("你指出得很具体", reply)
        self.assertNotIn("记单词", reply)

    def test_method_repair_fallback_lines_are_never_meta_narration(self):
        """0924-05 守护：兜底三档（中文/英文/日文 × 有/无承接句）都不许是元叙述。

        历史缺陷：`_method_repair_fallback` 曾把被 `_DRAFT_META_RE` 拦下的那句中文
        `继续你刚才说的"…"，我直接顺着这件事往下回应。` 原样当台词播出去。
        """
        from agent.tutor_agent import _DRAFT_META_RE

        prior = "i like stranger things because the story is interesting."
        for lang, lines in TutorAgent._METHOD_REPAIR_LINES.items():
            for key, template in lines.items():
                body = template.format(prior=prior)
                self.assertNotIn("\n", body)
                self.assertLessEqual(len(body), 400)
                self.assertIsNone(_DRAFT_META_RE.search(body),
                                  f"{lang}/{key} 命中元叙述词表: {body}")

    def test_method_repair_fallback_speaks_the_learners_language(self):
        """0924-05：用户用英文提问题时，修复回复不许整句中文。"""
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            messages=[
                {"role": "user", "content": "A small room in Singapore costs 800 a night."},
                {"role": "assistant", "content": "Which one surprises you more?"},
            ],
        )
        reply = TutorAgent._method_repair_fallback(ctx, "Don't you remember?")
        self.assertRegex(reply, r"[A-Za-z]")
        self.assertNotRegex(reply, r"[\u4e00-\u9fff]")
        self.assertIn("Singapore", reply)

    def test_stale_drill_is_blocked_during_method_repair(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            messages=[
                {"role": "user", "content": "I watched Stranger Things yesterday. I like it because the story is interesting."},
                {"role": "assistant", "content": "Great! Now tell me why you like another show."},
            ],
        )
        prepared = {
            "learning_move": {"action": "method_repair"},
            "constraints": {"user_must_speak": False},
        }
        bad_reply = "I want to practice English because ____ 你可以换个原因说说看哦？"
        violations = TutorAgent._controlled_reply_violations(
            bad_reply, prepared, "你现在的回复跟我的问题没有半毛钱关系。"
        )
        self.assertTrue(any("禁止填空" in item for item in violations))
        repaired = TutorAgent._enforce_controlled_reply(
            bad_reply, prepared, ctx,
            user_transcript="你为什么突然用全英文了？而且回答和我没什么关系。",
        )
        self.assertIn("Stranger Things", repaired)
        self.assertIn("中文支撑", repaired)
        self.assertNotIn("____", repaired)
        self.assertNotIn("你可以换", repaired)

    def test_stale_practice_goal_is_replaced_with_last_real_topic(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            messages=[
                {"role": "user", "content": "我先说为什么喜欢这部剧。i watch the strange things yesterday. i like it because the story is interesting."},
                {"role": "assistant", "content": "你指出得很具体：我刚才没有根据你的目标推进。"},
            ],
        )
        prepared = {
            "learning_move": {"action": "method_repair"},
            "constraints": {"user_must_speak": False},
        }
        bad_reply = "刚才没贴合你的需求。我调整一下，回到练习英语的目标，聊聊你为什么想练英语。"
        violations = TutorAgent._controlled_reply_violations(
            bad_reply, prepared, "你怎么像第三方评估角色，记忆又错乱了吗？"
        )
        self.assertTrue(any("禁止填空" in item for item in violations))
        repaired = TutorAgent._enforce_controlled_reply(
            bad_reply, prepared, ctx,
            user_transcript="你怎么像第三方评估角色，记忆又错乱了吗？",
        )
        self.assertIn("strange things", repaired.lower())
        self.assertNotIn("为什么想练英语", repaired)

    def test_method_repair_can_resume_content_without_self_audit(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            messages=[
                {"role": "user", "content": "我在说为什么喜欢这部剧。i like stranger things because the story is interesting."},
                {"role": "assistant", "content": "你指出得很具体，我停止旧练习。"},
            ],
        )
        prepared = {
            "learning_move": {"action": "method_repair"},
            "constraints": {"user_must_speak": False, "language_mode": "chinese_support"},
        }
        content_only = "你昨天看《Stranger Things》的时候很多地方没听懂，不过故事确实很有意思。"
        complaint = "你怎么像第三方评估角色，你的记忆又错乱了吗？"
        violations = TutorAgent._controlled_reply_violations(
            content_only, prepared, complaint
        )
        self.assertFalse(any("通用自我审计" in item for item in violations))
        repaired = TutorAgent._enforce_controlled_reply(
            content_only, prepared, ctx, user_transcript=complaint
        )
        self.assertNotIn("你指出得很具体", repaired)
        self.assertIn("stranger things", repaired.lower())

    def test_method_repair_blocks_polite_push_back_to_user(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            messages=[
                {"role": "user", "content": "我喜欢 Stranger Things，因为故事很有意思。"},
            ],
        )
        prepared = {
            "learning_move": {"action": "method_repair"},
            "constraints": {"user_must_speak": False, "language_mode": "chinese_support"},
        }
        pushed = "刚才我误解了你的反馈。现在你可以继续说说喜欢《怪奇物语》的其他理由。"
        complaint = "你怎么像第三方评估角色？"
        violations = TutorAgent._controlled_reply_violations(
            pushed, prepared, complaint
        )
        self.assertTrue(any("禁止填空" in item for item in violations))
        enforced = TutorAgent._enforce_controlled_reply(
            pushed, prepared, ctx, user_transcript=complaint
        )
        self.assertNotIn("你可以继续说", enforced)
        self.assertIn("Stranger Things", enforced)
        self.assertNotIn("你指出得很具体", enforced)

    def test_travel_opening_cannot_stop_at_empathy(self):
        prepared = {
            "learning_move": {"action": "connect_first_turn"},
            "constraints": {"language_mode": "chinese_support"},
        }
        user = "我想去国外旅游，但是英语不太好，有点担心在餐厅点餐。"
        empathy_only = "哦，我特别理解这种担心！国外餐厅点餐确实让人紧张。"
        violations = TutorAgent._controlled_reply_violations(
            empathy_only, prepared, user
        )
        self.assertTrue(any("不能停在共情" in item for item in violations))

        learning_menu = "你想先练 I want 这类表达，还是先记餐厅常见的食物词呀？"
        violations = TutorAgent._controlled_reply_violations(
            learning_menu, prepared, user
        )
        self.assertTrue(any("课程菜单选择" in item for item in violations))

    def test_i_want_to_order_gets_evidence_and_natural_upgrade(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "transfer_prompt"},
            "constraints": {"language_mode": "mixed_support"},
        }
        weak = "对，I want to order 说得很棒！现在换个场景。"
        violations = TutorAgent._controlled_reply_violations(
            weak, prepared, "I want to order，是吧？"
        )
        self.assertTrue(any("I'd like to order" in item for item in violations))
        repaired = TutorAgent._enforce_controlled_reply(
            weak, prepared, ctx, user_transcript="I want to order，是吧？"
        )
        self.assertIn("能听懂", repaired)
        self.assertIn("I'd like to order", repaired)

    def test_mixed_support_does_not_splice_single_word_into_chinese(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "listen_and_deepen"},
            "constraints": {"language_mode": "mixed_support"},
        }
        awkward = "说得很好！你平时喜欢热茶还是冰 tea？"
        violations = TutorAgent._controlled_reply_violations(
            awkward, prepared, "I want to order a cup of tea."
        )
        self.assertTrue(any("生硬塞进中文语法" in item for item in violations))
        repaired = TutorAgent._enforce_controlled_reply(
            awkward, prepared, ctx,
            user_transcript="I want to order a cup of tea.",
        )
        self.assertNotIn("冰 tea", repaired)
        self.assertIn("I'd like a cup of tea", repaired)

    def test_respond_turn_uses_backend_generation_after_policy(self):
        class FakeClient:
            async def chat_completion(self, **kwargs):
                self.messages = kwargs["messages"]
                return "Let’s make one clear point. Start with: I think..."

        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(
                session_id="s", user_id="u", target_lang="en",
                messages=[{"role": "assistant", "content": "previous reply"}],
            )
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}
            agent._build_system_prompt = lambda _ctx: "controlled tutor"

            async def prepare(_session_id, _text):
                return {
                    "session_id": "s",
                    "policy_prompt": "mission policy",
                    "learning_move": {"action": "mission_opening"},
                    "learning_state": ctx.learning_state.to_dict(),
                }

            agent.prepare_turn = prepare
            fake = FakeClient()
            with patch("llm_client.robust_llm_client", fake):
                result = await agent.respond_turn("s", "Hi")
                self.assertEqual(result["generation_mode"], "policy_controlled")
                self.assertIn("director_source", result["generation"])
                self.assertEqual(result["generation"]["response_model"], config.TUTOR_RESPONSE_MODEL)
            self.assertIn("I think", result["assistant_response"])
            self.assertEqual(fake.messages[-1]["content"], "Hi")
            self.assertTrue(any(m["content"] == "previous reply" for m in fake.messages))

        asyncio.run(run())

    def test_respond_turn_strips_internal_intent_lines_from_reply_and_history(self):
        class FakeClient:
            async def chat_completion(self, **_kwargs):
                return (
                    "USER_INTENT: Do you remember what I said about my wedding plans?\n"
                    "USER_INTENT_SOURCE: 我之前说过婚礼安排吗\n\n"
                    "I remember the date, but not the venue."
                )

        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}

            async def prepare(_session_id, _text):
                return {
                    "session_id": "s",
                    "policy_prompt": "bounded reply policy",
                    "learning_move": {"action": "listen_and_deepen"},
                    "learning_state": ctx.learning_state.to_dict(),
                    "conversation_director": {
                        "_meta": {"source": "fallback", "failure_code": "timeout"},
                    },
                    "constraints": {"language_mode": "mixed_support"},
                }

            async def record(*_args, **_kwargs):
                return None

            agent.prepare_turn = prepare
            agent._record_metering_event = record
            with patch("llm_client.robust_llm_client", FakeClient()):
                result = await agent.respond_turn(
                    "s", "Do you remember what I said about my wedding plans?"
                )

            visible = result["assistant_response"]
            self.assertEqual(visible, "I remember the date, but not the venue.")
            self.assertNotIn("USER_INTENT", visible)
            self.assertNotIn("USER_INTENT_SOURCE", visible)
            assistant_history = [
                message["content"] for message in ctx.messages
                if message.get("role") == "assistant"
            ]
            self.assertEqual(assistant_history[-1], visible)
            self.assertNotIn("USER_INTENT", assistant_history[-1])

        asyncio.run(run())

    def test_valid_director_draft_skips_second_model_round_trip(self):
        class FailIfCalledClient:
            calls = 0

            async def chat_completion(self, **kwargs):
                self.calls += 1
                raise AssertionError("valid director draft must not call response model")

        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}

            director = {
                "reply_draft": "Please show me your baggage tag. 请把行李牌给我。",
                "response_need": "roleplay_turn",
                "language_presentation": "target_then_chinese",
                "known_facts": ["用户在机场处理未到达行李"],
                "_meta": {"source": "director_model", "latency_ms": 820},
            }

            async def prepare(_session_id, _text):
                return {
                    "session_id": "s",
                    "policy_prompt": "roleplay policy",
                    "learning_move": {"action": "continue_roleplay_scene"},
                    "learning_state": ctx.learning_state.to_dict(),
                    "learner_state": {"language_mode": "mixed_support"},
                    "conversation_director": director,
                    "constraints": {
                        "language_mode": "mixed_support",
                        "conversation_director": director,
                    },
                }

            async def record(*_args, **_kwargs):
                return None

            agent.prepare_turn = prepare
            agent._record_metering_event = record
            fake = FailIfCalledClient()
            with patch("llm_client.robust_llm_client", fake):
                result = await agent.respond_turn("s", "行李牌要给你看吗？")
            self.assertEqual(fake.calls, 0)
            self.assertEqual(result["generation"]["path"], "director_reply_draft")
            self.assertIn("baggage tag", result["assistant_response"])

        asyncio.run(run())

    def test_invalid_generated_reply_is_repaired_without_second_remote_call(self):
        class CountingClient:
            def __init__(self):
                self.calls = 0

            async def chat_completion(self, **_kwargs):
                self.calls += 1
                return "你说得对，我明白了。"

        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}

            async def prepare(_session_id, _text):
                return {
                    "session_id": "s",
                    "policy_prompt": "bounded reply policy",
                    "learning_move": {"action": "listen_and_deepen"},
                    "learning_state": ctx.learning_state.to_dict(),
                    "conversation_director": {
                        "_meta": {"source": "fallback", "failure_code": "timeout"},
                    },
                    "constraints": {"language_mode": "mixed_support"},
                }

            async def record(*_args, **_kwargs):
                return None

            agent.prepare_turn = prepare
            agent._record_metering_event = record
            fake = CountingClient()
            # First validation rejects the generated text; the second confirms
            # that deterministic enforcement has completed the repair.
            with (
                patch("llm_client.robust_llm_client", fake),
                patch.object(
                    TutorAgent,
                    "_controlled_reply_violations",
                    side_effect=[["禁止先附和用户"], []],
                ),
            ):
                result = await agent.respond_turn("s", "继续说吧")
            self.assertEqual(fake.calls, 1)
            self.assertEqual(result["generation"]["path"], "generated_local_repair")

        asyncio.run(run())

    def test_streamed_voice_reply_also_avoids_remote_second_pass(self):
        class CountingClient:
            def __init__(self):
                self.calls = 0

            async def chat_completion(self, **_kwargs):
                self.calls += 1
                return "你说得对，我明白了。"

        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}

            async def prepare(_session_id, _text, progress_hook=None):
                return {
                    "session_id": "s",
                    "policy_prompt": "bounded reply policy",
                    "learning_move": {"action": "listen_and_deepen"},
                    "learning_state": ctx.learning_state.to_dict(),
                    "conversation_director": {
                        "_meta": {"source": "fallback", "failure_code": "timeout"},
                    },
                    "constraints": {"language_mode": "mixed_support"},
                    "turn_execution": {"route_tier": "ordinary"},
                }

            async def record(*_args, **_kwargs):
                return None

            agent.prepare_turn = prepare
            agent._record_metering_event = record
            fake = CountingClient()
            events = []
            with (
                patch("llm_client.robust_llm_client", fake),
                patch.object(
                    TutorAgent,
                    "_controlled_reply_violations",
                    side_effect=[["禁止先附和用户"], []],
                ),
            ):
                async for event in agent.respond_turn_stream("s", "继续说吧"):
                    events.append(event)
            done = next(event for event in events if event.get("type") == "done")
            self.assertEqual(fake.calls, 1)
            self.assertEqual(
                done["generation"]["path"], "provider_buffered_local_repair"
            )
            self.assertTrue(any(event.get("type") == "phrase" for event in events))

        asyncio.run(run())

    def test_late_background_director_cannot_overwrite_a_newer_turn(self):
        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.turn_count = 3
            ctx.director_plan_revision = 7
            ctx.conversation_plan = {"marker": "current-turn"}
            agent._sessions = {"s": ctx}
            agent._director_tasks = {}

            async def old_plan():
                return {"marker": "stale-turn", "interaction_frame": "language_sidecar"}

            task = asyncio.create_task(old_plan())
            agent._director_tasks["s"] = task
            await agent._finish_background_director_plan(
                session_id="s",
                revision=7,
                task=task,
                user_transcript="What does it mean?",
                learner_state=object(),
                previous_interaction_frame="in_role",
                planned_turn_count=2,
                turn_decision={"primary_frame": "language_sidecar"},
            )
            self.assertEqual(ctx.conversation_plan["marker"], "current-turn")

        asyncio.run(run())

    def test_prepare_turn_routes_first_content_to_relationship_connection(self):
        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}
            agent.intent_router = IntentRouter()
            agent.learner_state_estimator = LearnerStateEstimator()
            agent.learning_orchestrator = LearningOrchestrator()
            agent.policy_planner = TutorPolicyPlanner()
            agent._diagnose_without_writing = lambda _text, _lang: [
                {"type": "en_past_tense", "excerpt": "move"}
            ]

            result = await agent.prepare_turn(
                "s", "I feel lonely after I move here last month"
            )

            self.assertEqual(result["learning_move"]["action"], "connect_first_turn")
            self.assertEqual(result["constraints"]["correction_budget"], 0)
            self.assertFalse(result["constraints"]["user_must_speak"])
            self.assertEqual(result["policy"]["teaching_act"], "connect_first_turn")

        asyncio.run(run())

    def test_prepare_turn_preserves_topic_release_and_user_led_topic_actions(self):
        async def run_case(user_text):
            class FakeDirector:
                async def plan(self, **kwargs):
                    return ConversationDirector.fallback_plan(
                        user_text=kwargs["user_text"],
                        learner_intent=kwargs["learner_intent"],
                        language_mode=kwargs["language_mode"],
                        scene=kwargs["scene"],
                    )

            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            ctx.learning_state = LearningOrchestrator().start({}, {})
            agent._sessions = {"s": ctx}
            agent._director_tasks = {}
            agent._conversation_plans = {}
            agent._conversation_ledgers = {}
            agent._conversation_emotions = {}
            agent._conversation_states = {}
            agent.intent_router = IntentRouter()
            agent.learner_state_estimator = LearnerStateEstimator()
            agent.learning_orchestrator = LearningOrchestrator()
            agent.policy_planner = TutorPolicyPlanner()
            agent.conversation_director = FakeDirector()
            agent._diagnose_without_writing = lambda *_args: []
            agent._ingest_web_evidence_to_kb = lambda *_args: asyncio.sleep(0)
            return await agent.prepare_turn("s", user_text)

        async def run():
            released = await run_case(
                "Mm. I am a bit tired of talking about it now."
            )
            release_acts = released["constraints"]["turn_contract"]["decision"][
                "acts"
            ]
            self.assertIn("topic_release", release_acts)
            self.assertEqual(
                released["learning_move"]["action"], "acknowledge_topic_release"
            )
            self.assertFalse(released["constraints"]["user_must_speak"])

            shifted = await run_case(
                "I'm tired of talking about it now. Let's talk about music."
            )
            shifted_acts = shifted["constraints"]["turn_contract"]["decision"][
                "acts"
            ]
            self.assertIn("free_topic_shift", shifted_acts)
            self.assertNotIn("topic_release", shifted_acts)
            self.assertNotEqual(
                shifted["learning_move"]["action"], "acknowledge_topic_release"
            )

        asyncio.run(run())


class LearnerStateEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.estimator = LearnerStateEstimator()
        self.learning_state = LearningOrchestrator().start({}, {})
        self.learning_state.stage = "attempt"

    def estimate(self, text):
        return self.estimator.estimate(
            user_text=text,
            intent_result=self.router.classify(text),
            learning_state=self.learning_state,
            history=[],
            target_lang="en",
        )

    def test_lexical_question_preserves_active_task(self):
        state = self.estimate("What does blended mean?")
        self.assertEqual(state.communicative_intent, "lexical_clarification")
        self.assertEqual(state.cognitive_state, "local_confusion")
        self.assertEqual(state.engagement, "engaged")

    def test_exit_signal_wins_over_same_sentence_lexical_question(self):
        state = self.estimate("I want to stop now. What does blended mean?")
        self.assertEqual(state.communicative_intent, "exit_or_switch")
        self.assertGreaterEqual(state.exit_likelihood, 0.75)
        self.assertIn("lexical_question_deprioritized_by_exit", state.signals)

    def test_frustration_is_state_not_a_generic_question(self):
        state = self.estimate("你又在自说自话，好蠢")
        self.assertEqual(state.communicative_intent, "method_negotiation")
        self.assertEqual(state.affective_state, "frustrated")
        self.assertEqual(state.autonomy_need, "high")

    def test_natural_topic_drift_feedback_routes_to_method_repair(self):
        samples = [
            "你为什么突然用全英文了？而且你回答的好像和我都没什么关系。",
            "你现在的回复又跟我的问题没有半毛钱关系了。",
            "你现在已经完完全全脱离了我们聊的话题。",
        ]
        for text in samples:
            with self.subTest(text=text):
                result = self.router.classify(text)
                self.assertEqual(result.intent, TutorIntent.COMPLAINT)
                state = self.estimate(text)
                self.assertEqual(state.communicative_intent, "method_negotiation")

    def test_content_question_with_why_is_not_misclassified_as_complaint(self):
        text = (
            "我先跟你说一下我为什么喜欢这部剧吧。还有它的名字是啥？"
            "i watch the strange things yesterday. i like it because the story is interesting."
        )
        self.assertNotEqual(self.router.classify(text).intent, TutorIntent.COMPLAINT)

    def test_third_party_evaluator_feedback_is_a_complaint(self):
        text = (
            "你还是没有针对我的输入来回复，怎么你好像当起了第三方的评估角色，"
            "你的记忆又错乱了吗？"
        )
        self.assertEqual(self.router.classify(text).intent, TutorIntent.COMPLAINT)
        self.assertEqual(self.estimate(text).communicative_intent, "method_negotiation")

    def test_one_english_attempt_does_not_remove_recent_chinese_support(self):
        state = self.estimator.estimate(
            user_text="I watched Stranger Things yesterday.",
            intent_result=self.router.classify("I watched Stranger Things yesterday."),
            learning_state=self.learning_state,
            history=[
                {"role": "user", "content": "我最近在看一部英语电视剧，但很多地方听不懂。"},
                {"role": "assistant", "content": "你在看什么剧呀？"},
            ],
            target_lang="en",
        )
        self.assertEqual(state.language_mode, "mixed_support")
        self.assertIn("recent_chinese_support_retained", state.signals)

    def test_a2_english_turns_do_not_remove_profile_based_support(self):
        state = self.estimator.estimate(
            user_text="What time should I check out?",
            intent_result=self.router.classify("What time should I check out?"),
            learning_state=self.learning_state,
            history=[
                {"role": "user", "content": "I have a reservation."},
                {"role": "assistant", "content": "May I have your name?"},
                {"role": "user", "content": "My name is Zhang."},
            ],
            target_lang="en",
            cefr="A2",
        )
        self.assertEqual(state.language_mode, "mixed_support")
        self.assertIn("cefr_support_retained", state.signals)

    def test_tv_listening_difficulty_is_not_high_emotion_or_method_complaint(self):
        text = "我最近在看一部英语电视剧，但是没有字幕，有很多地方听不懂。"
        intent = self.router.classify(text)
        self.assertNotEqual(intent.emotion_intensity, "high")
        state = self.estimate(text)
        self.assertEqual(state.communicative_intent, "learning_difficulty")
        self.assertEqual(state.affective_state, "steady")
        self.assertEqual(state.cognitive_state, "ready")
        self.assertIn("learning_content_difficulty", state.signals)

    def test_tv_listening_difficulty_has_its_own_move_even_after_first_turn(self):
        text = "我最近在看一部英语电视剧，但是没有字幕，有很多地方听不懂。"
        intent = self.router.classify(text)
        state = self.estimate(text)
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        move = LearningOrchestrator().plan(
            learning,
            user_text=text,
            intent=intent.intent,
            errors=[],
            target_lang="en",
            learner_state=state,
        )
        self.assertEqual(move.action, "address_learning_difficulty")
        self.assertFalse(move.user_must_speak)

    def test_how_to_say_a_concept_routes_to_lexical_scaffold(self):
        text = "我喜欢里面的一个男主角吧，不知道是怎么说。"
        state = self.estimate(text)
        self.assertEqual(state.communicative_intent, "lexical_clarification")

    def test_teaching_process_feedback_is_applied_not_treated_as_practice(self):
        text = (
            "你应该先解释新词的意思，再用中文告诉我具体要表达什么，"
            "然后让我自己组织英文，这个流程是不是更自然？"
        )
        intent = self.router.classify(text)
        state = self.estimate(text)
        self.assertEqual(state.communicative_intent, "teaching_preference")
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        move = LearningOrchestrator().plan(
            learning,
            user_text=text,
            intent=intent.intent,
            errors=[],
            target_lang="en",
            learner_state=state,
        )
        self.assertEqual(move.action, "adapt_teaching_preference")
        self.assertFalse(move.user_must_speak)
        self.assertIn("中文语义", move.prompt_hint)

    def test_question_inside_bilingual_reply_counts_as_question_pressure(self):
        text = "我不想继续回答这种泛泛的问题。"
        state = self.estimator.estimate(
            user_text=text,
            intent_result=self.router.classify(text),
            learning_state=self.learning_state,
            history=[
                {
                    "role": "assistant",
                    "content": "What did you do after lunch? 中文是午饭后做了什么。",
                },
                {"role": "user", "content": "就随便走了走。"},
                {
                    "role": "assistant",
                    "content": "Did you buy anything? 我是问有没有买东西。",
                },
            ],
            target_lang="en",
        )
        self.assertEqual(state.recent_question_pressure, 2)
        self.assertIn("recent_question_pressure", state.signals)

    def test_rejecting_a_question_style_routes_to_method_repair(self):
        text = "都行啊，你这样问没意思。"
        state = self.estimate(text)
        self.assertEqual(state.communicative_intent, "method_negotiation")
        self.assertEqual(state.autonomy_need, "high")
        self.assertGreaterEqual(state.recent_question_pressure, 2)
        self.assertIn("question_style_rejected", state.signals)

    def test_explicit_remember_and_future_practice_is_a_memory_commitment(self):
        text = (
            "我明确记一下：是蓝牙耳机，左边没有声音，我想换货。"
            "以后我也还想多练售后沟通和过去式。"
        )
        state = self.estimate(text)
        self.assertEqual(state.communicative_intent, "memory_commitment")
        self.assertIn("explicit_memory_commitment", state.signals)

    def test_same_session_recap_has_a_distinct_intent(self):
        state = self.estimate("我们后面还聊了啥吗？")
        self.assertEqual(state.communicative_intent, "session_recap")
        self.assertIn("session_recap_request", state.signals)

    def test_repeated_short_replies_lower_load_instead_of_more_questions(self):
        history = [
            {"role": "assistant", "content": "What happened after that?"},
            {"role": "user", "content": "I don't know"},
            {"role": "assistant", "content": "What did you want to say?"},
            {"role": "user", "content": "Not sure"},
        ]
        state = self.estimator.estimate(
            user_text="Hmm",
            intent_result=self.router.classify("Hmm"),
            learning_state=self.learning_state,
            history=history,
            target_lang="en",
        )
        self.assertEqual(state.cognitive_state, "overloaded")
        self.assertEqual(state.interaction_rhythm, "lower_load")
        self.assertGreaterEqual(state.consecutive_short_replies, 2)

    def test_explicit_listen_preference_changes_strategy_not_just_wording(self):
        state = self.estimate("先别纠错，也别连续问我问题，先听我说完。")
        self.assertEqual(state.communicative_intent, "method_negotiation")
        self.assertIn("explicit_interaction_preference", state.signals)

    def test_scene_feedback_with_practice_word_routes_to_roleplay(self):
        text = (
            "如果是在餐厅情景训练，你就把我当成客人，你当服务员，"
            "按照迎宾问几位、带位、给菜单、点餐的真实流程来练习。"
        )
        self.assertEqual(self.router.classify(text).intent, TutorIntent.SCENARIO_ROLEPLAY)
        state = self.estimate(text)
        self.assertEqual(state.communicative_intent, "scenario_negotiation")

    def test_active_scene_role_question_routes_to_single_role_support(self):
        self.learning_state.scenario = "open_roleplay"
        self.learning_state.scenario_brief = "我想练习和宠物领养中心的工作人员沟通"
        self.learning_state.learner_role = "场景中的用户本人"
        self.learning_state.tutor_role = "宠物领养中心的工作人员"
        self.learning_state.scenario_step = "active_exchange"
        self.learning_state.roleplay_active = True
        state = self.estimate("我是客人，现在应该怎么说？")
        self.assertEqual(state.communicative_intent, "role_expression_support")
        move = LearningOrchestrator().plan(
            self.learning_state,
            user_text="我是客人，现在应该怎么说？",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
            learner_state=state,
        )
        self.assertEqual(move.action, "support_role_line")
        self.assertTrue(move.user_must_speak)

    def test_scene_negotiation_establishes_executable_roles(self):
        text = "在陶艺工作室里把我当学员，你当陶艺老师，按真实情况演绎。"
        state = self.estimate(text)
        learning = LearningOrchestrator().start({}, {})
        move = LearningOrchestrator().plan(
            learning,
            user_text=text,
            intent=self.router.classify(text).intent,
            errors=[],
            target_lang="en",
            learner_state=state,
        )
        self.assertEqual(move.action, "establish_roleplay_scene")
        self.assertEqual(learning.scenario, "open_roleplay")
        self.assertIn("陶艺工作室", learning.scenario_brief)
        self.assertEqual(learning.scenario_step, "active_exchange")

    def test_target_language_attempt_advances_open_roleplay_turn(self):
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "我在宠物领养中心和工作人员沟通"
        learning.learner_role = "场景中的用户本人"
        learning.tutor_role = "工作人员"
        learning.scenario_step = "active_exchange"
        learning.roleplay_active = True
        move = LearningOrchestrator().plan(
            learning,
            user_text="Two people, please.",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
        )
        self.assertEqual(move.action, "continue_roleplay_scene")
        self.assertEqual(learning.scenario_step, "active_exchange")
        self.assertEqual(learning.roleplay_turn, 1)

    def test_open_scene_guard_blocks_stale_topic_and_internal_state_speech(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_roleplay": {
                    "scene": "open_roleplay",
                    "brief": "我在宠物领养中心和工作人员沟通",
                },
            },
        }
        internal = TutorAgent._controlled_reply_violations(
            "角色固定，我们从头开始。", prepared, "I like this dog."
        )
        self.assertTrue(any("内部场景状态" in item for item in internal))
        stale = TutorAgent._controlled_reply_violations(
            "I like movies because ____. 你来填空。", prepared, "I like this dog."
        )
        self.assertTrue(any("旧话题" in item for item in stale))

    def test_invalid_scene_opening_is_flagged_without_script_replacement(self):
        prepared = {
            "learning_move": {"action": "establish_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_roleplay": {
                    "scene": "open_roleplay",
                    "brief": "我在陶艺工作室和老师沟通",
                },
            },
        }
        reply = "我们回到练习 because。"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, user_transcript="按真实陶艺工作室场景演绎。"
        )
        self.assertTrue(any("立即开始真实对话" in item for item in violations))
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        self.assertEqual(
            TutorAgent._enforce_controlled_reply(reply, prepared, ctx, "按真实场景演绎。"),
            reply,
        )


    def test_repeated_controlled_reply_is_detected_and_collapsed(self):
        repeated = (
            '你说的 "you can show" 少了宾语。那你想展示什么？'
            '你说的 "you can show" 少了宾语。那你想展示什么？'
            '你说的 "you can show" 少了宾语。'
        )
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {"language_mode": "mixed_support"},
        }
        violations = TutorAgent._controlled_reply_violations(
            repeated, prepared, "you can show"
        )
        self.assertTrue(any("循环重复" in item for item in violations))
        collapsed = TutorAgent._collapse_repeated_reply(repeated)
        self.assertLess(len(collapsed), len(repeated))
        self.assertEqual(collapsed.count("你说的"), 1)

    def test_product_feature_question_uses_feature_not_this_is_a_function(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {"language_mode": "chinese_support"},
        }
        transcript = "功能的单词怎么说？我都不知道怎么说。"
        reply = TutorAgent._enforce_controlled_reply(
            '功能的英文是 function。This is a function.', prepared, ctx, transcript
        )
        self.assertIn("feature", reply)
        self.assertIn("What features does it have?", reply)
        self.assertNotIn("This is a function", reply)

    def test_fast_comparative_question_gets_faster_explanation(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {"language_mode": "chinese_support"},
        }
        reply = TutorAgent._enforce_controlled_reply(
            "这里用 fast 就好。", prepared, ctx, "为什么没有用 fast 的比较级？"
        )
        self.assertIn("faster", reply)
        self.assertIn("helps you do things faster", reply)

    def test_current_turn_speech_claim_fails_closed_without_audio_evidence(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        prepared = {
            "learning_move": {"action": "method_repair"},
            "constraints": {"language_mode": "chinese_support"},
            "assessment": {
                "speech": {
                    "pronunciation": {"decision": "insufficient_evidence"},
                    "prosody": {"decision": "insufficient_evidence"},
                    "fluency": {"decision": "insufficient_evidence"},
                }
            },
        }
        reply = TutorAgent._enforce_controlled_reply(
            "Can I show you this function?", prepared, ctx,
            "你是用我的原音，还是单纯用文本判断发音、语调和语速？",
        )
        self.assertIn("声学证据", reply)
        self.assertIn("ASR 文本", reply)
        self.assertNotIn("Can I show", reply)

    def test_booth_worker_owns_demo_and_corrects_yes_you_can_show_naturally(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [
            {"role": "assistant", "content": "Can I show you this feature?"},
        ]
        prepared = {
            "learning_move": {"action": "repair_roleplay_contract"},
            "constraints": {
                "language_mode": "chinese_support",
                "scene_roleplay": {
                    "brief": "国外科技展展位交流",
                    "tutor_role": "展位工作人员",
                    "learner_role": "访客",
                },
            },
        }
        reply = TutorAgent._enforce_controlled_reply(
            "You can show me. 你想展示什么？", prepared, ctx,
            "Yes, you can show. 这个用法有问题吗？",
        )
        self.assertIn("Yes, please.", reply)
        self.assertIn("由我继续介绍产品", reply)
        self.assertNotIn("你想展示什么", reply)

    def test_new_session_entry_rejects_invented_previous_mistake(self):
        prepared = {
            "learning_move": {"action": "connect_first_turn"},
            "constraints": {"language_mode": "mixed_support", "fresh_context": True},
        }
        reply = "刚才我跳过了你的顾虑，这是我的疏忽。Menu, please."
        transcript = "我的英语不太好，担心去国外餐厅不知道怎么和服务员沟通。"
        violations = TutorAgent._controlled_reply_violations(reply, prepared, transcript)
        self.assertTrue(any("虚构上一轮错误" in item for item in violations))

    def test_fresh_context_opening_forbids_old_open_loops(self):
        """plans/0916-09 §4.5：措辞由"一概不得承接"收窄为"提议不接管"，
        但保护意图不变——旧话题内容不得被注入本块，且必须写明用户没接就放下。
        """
        agent = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(
            session_id="s",
            user_id="u",
            target_lang="en",
            fresh_context=True,
            recent_sessions=[{"summary": "上次在投诉语音截断"}],
            relationship_memory={"open_loop": "继续处理上次的截断问题"},
        )
        opening = agent._build_opening_context(ctx)
        self.assertIn("空白会话", opening)
        self.assertNotIn("截断问题", opening)
        # 提议只许一次 + 没接就放下（防幻影话题，plans/0915-02 事故的护栏）
        self.assertIn("最多在开场白里被提议过一次", opening)
        self.assertIn("当轮放下", opening)
        self.assertIn("不得再承接", opening)

    def test_arbitrary_unlisted_scene_is_extracted_without_template(self):
        agent = TutorAgent.__new__(TutorAgent)
        text = "我想去宠物领养中心，但不知道怎么和工作人员沟通。"
        scene = agent._extract_scene_info(text)
        self.assertEqual(scene["scene"], "open_roleplay")
        self.assertEqual(scene["role"], "工作人员")
        self.assertEqual(scene["step"], "active_exchange")
        self.assertEqual(scene["brief"], text)
        self.assertNotIn("steps", scene)

    def test_technology_expo_first_turn_never_routes_to_old_complaint_repair(self):
        text = (
            "我以后可能会去国外参加科技展，但是我的英语不太好，有点担心，"
            "不知道怎么和展位上的工作人员交流。每个展位和产品展示也不一样，对吧？"
        )
        router = IntentRouter()
        estimator = LearnerStateEstimator()
        agent = TutorAgent.__new__(TutorAgent)
        scene = agent._extract_scene_info(text)
        learning = LearningOrchestrator().start({}, {})
        learning.scenario = scene["scene"]
        learning.scenario_brief = scene["brief"]
        learning.learner_role = scene["learner_role"]
        learning.tutor_role = scene["role"]
        learning.scenario_step = scene["step"]
        routed = router.classify(text)
        learner = estimator.estimate(
            user_text=text, intent_result=routed, learning_state=learning,
            history=[], target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning, user_text=text, intent=routed.intent, errors=[],
            target_lang="en", learner_state=learner,
        )
        self.assertEqual(routed.intent, TutorIntent.SCENARIO_ROLEPLAY)
        self.assertEqual(learner.communicative_intent, "scenario_negotiation")
        self.assertEqual(scene["role"], "展位上的工作人员")
        self.assertEqual(move.action, "establish_roleplay_scene")
        self.assertNotEqual(move.action, "method_repair")

    def test_first_communication_worry_establishes_open_roleplay(self):
        text = "我的英语不太好，有点担心去国外餐厅，不知道怎么和服务员沟通。"
        router = IntentRouter()
        estimator = LearnerStateEstimator()
        learning = LearningOrchestrator().start({}, {})
        scene = TutorAgent.__new__(TutorAgent)._extract_scene_info(text)
        learning.scenario = scene["scene"]
        learning.scenario_brief = scene["brief"]
        learning.learner_role = scene["learner_role"]
        learning.tutor_role = scene["role"]
        learning.scenario_step = scene["step"]
        learner = estimator.estimate(
            user_text=text,
            intent_result=router.classify(text),
            learning_state=learning,
            history=[],
            target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning,
            user_text=text,
            intent=router.classify(text).intent,
            errors=[],
            target_lang="en",
            learner_state=learner,
        )
        self.assertEqual(move.action, "establish_roleplay_scene")
        self.assertTrue(learning.roleplay_active)
        self.assertEqual(learning.scenario, "open_roleplay")
        self.assertEqual(learning.scenario_step, "active_exchange")
        self.assertIn("不要套固定剧本", move.prompt_hint)

    def test_active_scene_feedback_preserves_completed_step(self):
        text = "你已经问过几位了，我也回答了 Just me，为什么又重新开始？"
        router = IntentRouter()
        estimator = LearnerStateEstimator()
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "我在餐厅和服务员沟通"
        learning.learner_role = "场景中的用户本人"
        learning.tutor_role = "服务员"
        learning.scenario_step = "active_exchange"
        learning.roleplay_active = True
        learner = estimator.estimate(
            user_text=text,
            intent_result=router.classify(text),
            learning_state=learning,
            history=[],
            target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning,
            user_text=text,
            intent=router.classify(text).intent,
            errors=[],
            target_lang="en",
            learner_state=learner,
        )
        self.assertEqual(move.action, "repair_roleplay_contract")
        self.assertEqual(learning.scenario_step, "active_exchange")

    def test_just_my_is_corrected_without_advancing_scene(self):
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "我在餐厅和服务员沟通"
        learning.learner_role = "场景中的用户本人"
        learning.tutor_role = "服务员"
        learning.scenario_step = "active_exchange"
        learning.roleplay_active = True
        move = LearningOrchestrator().plan(
            learning,
            user_text="Just my.",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
        )
        self.assertEqual(move.action, "repair_roleplay_line")
        self.assertEqual(learning.scenario_step, "active_exchange")

    def test_corrected_just_me_continues_open_scene(self):
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "我在餐厅和服务员沟通"
        learning.learner_role = "场景中的用户本人"
        learning.tutor_role = "服务员"
        learning.scenario_step = "active_exchange"
        learning.roleplay_active = True
        move = LearningOrchestrator().plan(
            learning,
            user_text="Just me.",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
        )
        self.assertEqual(move.action, "continue_roleplay_scene")
        self.assertEqual(learning.scenario_step, "active_exchange")
        self.assertEqual(learning.roleplay_turn, 1)

    def test_roleplay_uses_one_brief_upgrade_then_continues_in_role(self):
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "在租车柜台取车"
        learning.learner_role = "租车客户"
        learning.tutor_role = "租车工作人员"
        learning.scenario_step = "确认每日价格"
        learning.roleplay_active = True
        learning.turns_since_correction = 99
        orchestrator = LearningOrchestrator()
        move = orchestrator.plan(
            learning,
            user_text="So how much I should pay for one day?",
            intent=TutorIntent.CHITCHAT,
            errors=[{
                "type": "director_word_order",
                "excerpt": "how much I should pay for one day",
                "correction": "How much should I pay per day?",
                "explanation": "疑问句中 should 放在主语前",
            }],
            target_lang="en",
        )
        self.assertEqual(move.action, "continue_roleplay_with_recast")
        self.assertEqual(move.feedback_mode, "one_brief_upgrade_in_role")
        self.assertIn("How much should I pay per day?", move.prompt_hint)
        self.assertEqual(learning.roleplay_turn, 1)
        self.assertEqual(learning.turns_since_correction, 0)
        constraints = orchestrator.build_constraints(move, None)
        self.assertEqual(constraints["correction_budget"], 1)

        cooldown_move = orchestrator.plan(
            learning,
            user_text="The deposit I should pay how much?",
            intent=TutorIntent.CHITCHAT,
            errors=[{"type": "director_word_order", "excerpt": "deposit"}],
            target_lang="en",
        )
        self.assertEqual(cooldown_move.action, "continue_roleplay_scene")

    def test_roleplay_recast_uptake_transfers_to_new_branch_without_reopening(self):
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "致电酒店核对错误的 minibar 费用"
        learning.learner_role = "住客"
        learning.tutor_role = "酒店前台"
        learning.scenario_step = "核对争议费用"
        learning.roleplay_active = True
        learning.turns_since_correction = 99
        orchestrator = LearningOrchestrator()

        recast = orchestrator.plan(
            learning,
            user_text="I doesn't use the minibar.",
            intent=TutorIntent.CHITCHAT,
            errors=[{
                "type": "subject_verb_agreement",
                "excerpt": "I doesn't use",
                "correction": "I didn't use the minibar.",
                "explanation": "昨天的事情用 didn't",
            }],
            target_lang="en",
        )
        self.assertEqual(recast.action, "continue_roleplay_with_recast")
        self.assertEqual(learning.pending_recast, "I didn't use the minibar.")

        uptake = orchestrator.plan(
            learning,
            user_text="I'm sure I didn't use the minibar.",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
        )
        self.assertEqual(uptake.action, "continue_roleplay_scene")
        self.assertTrue(learning.recast_uptake)

        transfer = orchestrator.plan(
            learning,
            user_text="那如果酒店需要进一步核实呢？",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
            conversation_plan={
                "topic_transition": "same_goal_new_phase",
                "current_phase": "客房部核查分支",
            },
        )
        self.assertEqual(transfer.action, "continue_roleplay_transfer_branch")
        self.assertFalse(transfer.user_must_speak)
        self.assertIn("不得重播问候", transfer.prompt_hint)
        self.assertEqual(learning.transfer_branches_completed, 1)
        constraints = orchestrator.build_constraints(transfer, None)
        self.assertIn("重播场景开场", constraints["forbidden"])

    def test_open_scene_contract_blocks_unrelated_old_chitchat(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_roleplay": {
                    "scene": "open_roleplay",
                    "brief": "我在宠物领养中心和工作人员沟通",
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "Let's talk about movies because they are fun.（聊电影吧。）", prepared, "I like this dog."
        )
        self.assertTrue(any("旧话题" in item for item in violations))

    def test_reconnect_rehydrates_open_scene_from_history(self):
        agent = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(
            session_id="s",
            user_id="u",
            target_lang="en",
            learning_state=LearningOrchestrator().start({}, {}),
        )
        history = [
            {"role": "user", "content": "我想去宠物领养中心，但不知道怎么和工作人员沟通。"},
            {"role": "assistant", "content": "What kind of pet are you looking for?"},
            {"role": "user", "content": "I like this dog."},
        ]
        agent._rehydrate_scene_state(ctx, history)
        self.assertTrue(ctx.scene_context_set)
        self.assertEqual(ctx.active_scene, "open_roleplay")
        self.assertEqual(ctx.scenario_role, "工作人员")
        self.assertIn("宠物领养中心", ctx.scenario_brief)
        self.assertTrue(ctx.learning_state.roleplay_active)
        self.assertEqual(ctx.learning_state.scenario_step, "active_exchange")

    def test_reconnect_rehydrates_language_sidecar_for_short_followup(self):
        agent = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        history = [
            {"role": "user", "content": "What does go through mean?"},
            {
                "role": "assistant",
                "content": "It means a payment was accepted successfully.",
            },
        ]
        agent._rehydrate_turn_state(ctx, history)
        self.assertEqual(ctx.turn_decision["primary_frame"], "language_sidecar")
        followup = TurnController.decide(
            user_text="对吧？",
            communicative_intent="open_expression",
            scene_active=True,
            scene_changed=False,
            existing_plan={},
            previous_decision=ctx.turn_decision,
        ).to_dict()
        self.assertEqual(followup["primary_frame"], "language_sidecar")
        self.assertEqual(followup["response_sequence"], ["answer_language_followup"])

    def test_expo_reconnect_routes_role_question_to_current_role_support(self):
        agent = TutorAgent.__new__(TutorAgent)
        learning = LearningOrchestrator().start({}, {})
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            learning_state=learning,
        )
        history = [
            {
                "role": "user",
                "content": "以后可能会去国外参加科技展，但是我的英语不太好，有点担心，不知道怎么和展位上的工作人员交流。",
            },
            {
                "role": "assistant",
                "content": "Hi! 别担心，有什么我能帮你的吗？What can I do for you?",
            },
        ]
        agent._rehydrate_scene_state(ctx, history)
        text = "我现在不知道怎么说啊？你现在已经进入到这个展位工作人员的角色了吗？还是怎么样？"
        router = IntentRouter()
        estimator = LearnerStateEstimator()
        routed = router.classify(text)
        learner = estimator.estimate(
            user_text=text, intent_result=routed, learning_state=learning,
            history=history, target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning, user_text=text, intent=routed.intent, errors=[],
            target_lang="en", learner_state=learner,
        )
        self.assertTrue(learning.roleplay_active)
        self.assertEqual(learning.tutor_role, "展位上的工作人员")
        self.assertEqual(learner.communicative_intent, "role_expression_support")
        self.assertEqual(move.action, "support_role_line")
        self.assertIn("AI当前角色", move.prompt_hint)

    def test_scene_repair_fallback_does_not_inject_a_scripted_line(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.active_scene = "open_roleplay"
        ctx.scenario_brief = "我在宠物领养中心和工作人员沟通"
        reply = TutorAgent._method_repair_fallback(ctx, "你为什么又重新开始？")
        self.assertNotIn("How many people", reply)
        self.assertNotIn("Right this way", reply)

    def test_hotel_roleplay_also_uses_open_scene_contract(self):
        agent = TutorAgent.__new__(TutorAgent)
        scene = agent._extract_scene_info(
            "我担心去国外酒店办理入住，不知道怎么和前台沟通。"
        )
        self.assertEqual(scene["scene"], "open_roleplay")
        self.assertEqual(scene["learner_role"], "场景中的用户本人")
        self.assertEqual(scene["role"], "前台")
        self.assertNotIn("steps", scene)

        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = scene["scene"]
        learning.scenario_brief = scene["brief"]
        learning.learner_role = scene["learner_role"]
        learning.tutor_role = scene["role"]
        learning.scenario_step = scene["step"]
        learning.roleplay_active = True
        move = LearningOrchestrator().plan(
            learning,
            user_text="Yes, I have a reservation.",
            intent=TutorIntent.CHITCHAT,
            errors=[],
            target_lang="en",
        )
        self.assertEqual(move.action, "continue_roleplay_scene")
        self.assertEqual(learning.scenario_step, "active_exchange")
        self.assertEqual(learning.roleplay_turn, 1)

    def test_turn_method_is_selected_and_injected_into_real_control_prompt(self):
        agent = TutorAgent.__new__(TutorAgent)
        learning = LearningOrchestrator().start({}, {})
        learning.stage = "flow"
        learning.scenario = "open_roleplay"
        learning.scenario_brief = "我在酒店和前台沟通办理入住"
        learning.learner_role = "场景中的用户本人"
        learning.tutor_role = "前台"
        learning.scenario_step = "active_exchange"
        learning.roleplay_active = True
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            profile={"cefr": "A2"}, learning_state=learning,
            active_scene="open_roleplay", scenario_brief=learning.scenario_brief,
            learner_role=learning.learner_role, scenario_role="前台",
            scenario_step="active_exchange",
            scene_context_set=True,
        )
        learner = LearnerStateEstimator().estimate(
            user_text="Yes, I have a reservation.",
            intent_result=IntentRouter().classify("Yes, I have a reservation."),
            learning_state=learning,
            history=[],
            target_lang="en",
        )
        move = LearningMove(
            stage="flow", action="continue_roleplay_scene",
            goal="推进酒店入住一步", prompt_hint="保持前台角色",
        )
        method = agent._select_turn_method(
            ctx=ctx, errors=[], learner_state=learner, learning_move=move
        )
        self.assertEqual(method["name"], "task_based_roleplay")
        self.assertIn("task_based_learning", method["evidence_sources"])
        ctx.active_method = method
        prompt = agent._build_control_prompt(
            ctx,
            {
                "teaching_method": method,
                "learning_move": move.to_dict(),
                "learner_state": learner.to_dict(),
                "constraints": {"scene_roleplay": {
                    "scene": "open_roleplay", "brief": learning.scenario_brief,
                }},
                "policy": {},
            },
        )
        self.assertIn("Teaching Method - execute this turn", prompt)
        self.assertIn("task_based_roleplay", prompt)
        self.assertIn("我在酒店和前台沟通办理入住", prompt)
        self.assertIn("不得寻找预制场景", prompt)

    def test_selected_teaching_skills_are_exposed_as_tools(self):
        agent = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.active_skills = [PRESET_SKILLS["task_based_roleplay"]]
        tools = agent._build_tools(ctx)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "use_skill_task_based_roleplay")

    def test_current_turn_rag_retrieval_uses_scene_method_and_user_input(self):
        class FakeKnowledgeRAG:
            def __init__(self):
                self.query = ""

            async def retrieve_by_text(self, **kwargs):
                self.query = kwargs["query_text"]
                return [{
                    "chunk_id": "KN-test", "title": "Hotel check-in",
                    "content": "Use reservation and passport in the real check-in sequence.",
                    "source": "test",
                }]

        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            agent.knowledge_rag = FakeKnowledgeRAG()
            ctx = SessionContext(
                session_id="s", user_id="u", target_lang="en",
                profile={"cefr": "A2"}, active_scene="open_roleplay",
                scenario_brief="我想练习在酒店办理入住，和前台沟通。",
            )
            ctx.active_method = {"name": "task_based_roleplay"}
            move = LearningMove(
                stage="flow", action="establish_roleplay_scene",
                goal="开始酒店入住",
            )
            results = await agent._retrieve_turn_knowledge(
                ctx=ctx,
                user_transcript="我想练习在酒店办理入住。",
                errors=[],
                learning_move=move,
            )
            self.assertEqual(results[0]["chunk_id"], "KN-test")
            self.assertNotIn("hotel_checkin_flow", agent.knowledge_rag.query)
            self.assertIn("task_based_roleplay", agent.knowledge_rag.query)
            self.assertIn("酒店办理入住", agent.knowledge_rag.query)

        asyncio.run(run())

    def test_phone_repair_channel_is_grounded_before_language_drill(self):
        text = "我主要不懂国外的维修渠道，有没有官方售后或者独立维修店？"
        routed = IntentRouter().classify(text)
        learning = LearningOrchestrator().start({}, {})
        learner = LearnerStateEstimator().estimate(
            user_text=text, intent_result=routed, learning_state=learning,
            history=[], target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning, user_text=text, intent=routed.intent, errors=[],
            target_lang="en", learner_state=learner,
        )
        self.assertEqual(learner.communicative_intent, "real_world_context")
        self.assertEqual(move.action, "ground_real_world_context")
        self.assertFalse(move.user_must_speak)

        prepared = {
            "learning_move": move.to_dict(),
            "constraints": {"language_mode": "chinese_support"},
        }
        violations = TutorAgent._controlled_reply_violations(
            "repair 是维修，你试着说 Where is the repair shop?", prepared, text
        )
        self.assertTrue(any("现实渠道" in item for item in violations))

    def test_explain_situation_establishes_open_scene(self):
        agent = TutorAgent.__new__(TutorAgent)
        scene = agent._extract_scene_info(
            "手机充不进电，不知道怎么跟维修店的工作人员说明情况。"
        )
        self.assertIsNotNone(scene)
        self.assertEqual(scene["scene"], "open_roleplay")
        self.assertIn("维修店", scene["role"])

    def test_refusing_one_drill_does_not_switch_topic(self):
        text = "我没叫你换个话题，我只是觉得 OK, no problem 太简单了，不想跟读。"
        routed = IntentRouter().classify(text)
        learning = LearningOrchestrator().start({}, {})
        learner = LearnerStateEstimator().estimate(
            user_text=text, intent_result=routed, learning_state=learning,
            history=[], target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning, user_text=text, intent=routed.intent, errors=[],
            target_lang="en", learner_state=learner,
        )
        self.assertEqual(learner.exit_likelihood, 0.0)
        self.assertEqual(learner.communicative_intent, "skip_current_item")
        self.assertEqual(move.action, "skip_current_item")

        prepared = {
            "learning_move": move.to_dict(),
            "constraints": {"language_mode": "mixed_support"},
        }
        violations = TutorAgent._controlled_reply_violations(
            "那我们聊 fashion 吧，你试着说 I like fashion.", prepared, text
        )
        self.assertTrue(any("不等于换话题" in item for item in violations))
        self.assertTrue(any("禁止再次" in item for item in violations))

    def test_counterpart_reply_is_supplied_before_learner_response(self):
        text = "如果对方不同意借手机，他会怎么回复我？"
        routed = IntentRouter().classify(text)
        learning = LearningOrchestrator().start({}, {})
        learner = LearnerStateEstimator().estimate(
            user_text=text, intent_result=routed, learning_state=learning,
            history=[], target_lang="en",
        )
        move = LearningOrchestrator().plan(
            learning, user_text=text, intent=routed.intent, errors=[],
            target_lang="en", learner_state=learner,
        )
        self.assertEqual(move.action, "supply_counterpart_turn")
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        fixed = TutorAgent._enforce_controlled_reply(
            "你先说 Thank you very much.",
            {"learning_move": move.to_dict(), "constraints": {"language_mode": "mixed_support"}},
            ctx,
            text,
        )
        self.assertIn("I can’t lend you my phone", fixed)
        self.assertIn("No worries", fixed)

    def test_polite_phone_refusal_uses_natural_complete_sentence(self):
        text = (
            "对方可能不喜欢陌生人用他的手机，还会加 I'm sorry，"
            "这种略带歉意的话应该怎么说更自然？"
        )
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        fixed = TutorAgent._enforce_controlled_reply(
            "你可以说 I don't like ___ my phone，选个词试试。",
            {"learning_move": {"action": "method_repair"}, "constraints": {}},
            ctx,
            text,
        )
        self.assertIn("not comfortable lending my phone", fixed)
        self.assertNotIn("___", fixed)

    def test_general_chat_requires_complete_english_not_a_question(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "language_mode": "mixed_support",
                "recent_question_pressure": 0,
                "turn_contract": {"decision": {"primary_frame": "general"}},
            },
        }
        no_english = TutorAgent._controlled_reply_violations(
            "她这个角色确实很有意思。", prepared, "我挺喜欢 Eleven。"
        )
        self.assertTrue(any("完整" in item and "英文" in item for item in no_english))
        natural_no_question = TutorAgent._controlled_reply_violations(
            "我也喜欢她。She is strong, but she still feels human.",
            prepared,
            "我挺喜欢 Eleven。",
        )
        self.assertFalse(any("完整" in item and "英文" in item for item in natural_no_question))

    def test_question_pressure_selects_no_question_shape_and_rejects_question(self):
        learner = LearnerState(recent_question_pressure=2, language_mode="mixed_support")
        move = LearningMove(
            stage="flow", action="engage_general_topic", goal="自然聊天",
            user_must_speak=False, support_level=0,
        )
        constraints = LearningOrchestrator.build_constraints(move, learner)
        self.assertEqual(constraints["conversation_shape"], "content_reply_no_question")
        prepared = {
            "learning_move": move.to_dict(),
            "constraints": {
                **constraints,
                "turn_contract": {"decision": {"primary_frame": "general"}},
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "She is a fascinating character. 你最喜欢她哪一点？",
            prepared,
            "I like Eleven.",
        )
        self.assertTrue(any("连续追问" in item for item in violations))

    def test_episode_is_not_silently_changed_to_season(self):
        prepared = {"learning_move": {"action": "engage_general_topic"}, "constraints": {}}
        violations = TutorAgent._controlled_reply_violations(
            "Oh, you watched season 5!", prepared, "我看过怪奇物语第5集。"
        )
        self.assertTrue(any("第N集" in item for item in violations))

    def test_cefr_level_needs_assessment_evidence(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {},
            "assessment": {"observed_item_count": 3},
        }
        violations = TutorAgent._controlled_reply_violations(
            "我觉得你大概是 A2 到 B1。", prepared, "你觉得我现在的语言等级是什么？"
        )
        self.assertTrue(any("证据化CEFR" in item for item in violations))


if __name__ == "__main__":
    unittest.main()
