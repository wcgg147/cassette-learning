import unittest

try:
    from .turn_controller import TurnController
except ImportError:
    from agent.turn_controller import TurnController


class TurnControllerTests(unittest.TestCase):
    def decide(self, text, intent="open_expression", **kwargs):
        return TurnController.decide(
            user_text=text,
            communicative_intent=intent,
            scene_active=kwargs.pop("scene_active", True),
            scene_changed=kwargs.pop("scene_changed", False),
            existing_plan=kwargs.pop("existing_plan", {"scene_brief": "超市结账"}),
            previous_decision=kwargs.pop("previous_decision", {}),
            previous_frame=kwargs.pop("previous_frame", "in_role"),
            contextual_asr=kwargs.pop("contextual_asr", {}),
            material_tape=kwargs.pop("material_tape", False),
        ).to_dict()

    def test_ordinary_role_turn_reuses_scene_without_replanning(self):
        decision = self.decide("Can I use cash?")
        self.assertEqual(decision["primary_frame"], "in_role")
        self.assertFalse(decision["replan_required"])
        self.assertFalse(decision["strong_reasoning_required"])
        self.assertEqual(decision["planning_route"], "fast_path")
        self.assertEqual(decision["response_sequence"], ["respond_as_counterpart"])

    def test_inferred_lexical_intent_does_not_override_material_question(self):
        decision = self.decide(
            "They both describe someone at a low point. What does each story show about getting through it?",
            communicative_intent="lexical_clarification",
            material_tape=True,
            scene_active=False,
            existing_plan={},
            previous_frame="",
        )
        self.assertNotIn("language_question", decision["acts"])
        self.assertIn("respond_to_latest_input", decision["response_sequence"])

    def test_explicit_word_meaning_stays_language_sidecar_on_material_tape(self):
        decision = self.decide(
            'What does "discouraged" mean in this story?',
            communicative_intent="lexical_clarification",
            material_tape=True,
        )
        self.assertIn("language_question", decision["acts"])
        self.assertEqual(decision["primary_frame"], "language_sidecar")

    def test_language_question_is_temporary_sidecar(self):
        decision = self.decide('What does "go through" mean?')
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertEqual(decision["resume_frame"], "in_role")
        self.assertEqual(decision["response_sequence"], ["answer_language_question"])
        self.assertTrue(decision["replan_required"])
        self.assertTrue(any("ASR" in item for item in decision["response_obligations"]))

    def test_language_question_keeps_current_term_and_phrase_meaning_grounded(self):
        decision = self.decide(
            "kind of place 这里的 place 还是地点的意思吗？",
            communicative_intent="lexical_clarification",
        )
        obligations = " ".join(decision["response_obligations"])
        self.assertIn("语音实际所指", obligations)
        self.assertIn("不能机械相信错误转写", obligations)
        self.assertIn("基本含义", obligations)
        self.assertIn("自然中文转述", obligations)

    def test_suspect_payment_phrase_requests_fresh_semantic_verification(self):
        decision = self.decide("payment decision 是什么意思？")
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertTrue(decision["replan_required"])
        self.assertTrue(any("编造固定含义" in item for item in decision["response_obligations"]))

    def test_first_language_question_without_a_plan_still_requests_planning(self):
        decision = self.decide(
            'What does "go through" mean?', existing_plan={}
        )
        self.assertTrue(decision["replan_required"])

    def test_how_to_phrase_fragment_is_not_mistaken_for_a_role_turn(self):
        decision = self.decide("when doesn't have qr code 这个疑问句")
        self.assertEqual(decision["acts"], ["language_question"])
        self.assertEqual(decision["response_sequence"], ["answer_language_question"])

    def test_short_confirmation_keeps_the_language_sidecar_open(self):
        decision = self.decide(
            "开始才对，是吧？",
            previous_frame="language_sidecar",
            previous_decision={"primary_frame": "language_sidecar"},
        )
        self.assertEqual(decision["acts"], ["language_followup"])
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertEqual(decision["response_sequence"], ["answer_language_followup"])

    def test_one_turn_can_keep_language_and_role_acts(self):
        decision = self.decide("这个说法自然吗？Can I use cash?")
        self.assertIn("language_question", decision["acts"])
        self.assertIn("role_utterance", decision["acts"])
        self.assertEqual(
            decision["response_sequence"],
            ["answer_language_question", "respond_as_counterpart"],
        )
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertIn("multiple_speech_acts", decision["routing_reasons"])

    def test_past_tense_question_then_resume_keeps_both_obligations(self):
        decision = self.decide(
            "Bot 在这里是过去式对吧？回答完继续刚才的流程。"
        )
        self.assertIn("language_question", decision["acts"])
        self.assertIn("scene_control", decision["acts"])
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertEqual(
            decision["response_sequence"],
            ["answer_language_question", "respond_as_counterpart"],
        )
        self.assertTrue(decision["strong_reasoning_required"])

    def test_contextual_language_reference_uses_strong_reasoning(self):
        decision = self.decide("我问的是刚才那个短语是什么意思？")
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertTrue(decision["replan_required"])
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertEqual(decision["planning_route"], "strong_model")
        self.assertIn("contextual_reference", decision["routing_reasons"])

    def test_asr_self_correction_recovers_unanswered_language_question(self):
        decision = self.decide("我说的是 borrow，不是 blue，对吧？")
        self.assertIn("language_question", decision["acts"])
        self.assertIn("asr_self_correction", decision["acts"])
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertEqual(decision["response_sequence"], ["answer_language_question"])
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertTrue(any("原问题" in item for item in decision["response_obligations"]))

    def test_mixed_role_line_and_here_use_question_keeps_both_acts(self):
        decision = self.decide("I want to borrow a car. 这里用 borrow 还是 rent 更自然？")
        self.assertIn("language_question", decision["acts"])
        self.assertIn("role_utterance", decision["acts"])
        self.assertEqual(
            decision["response_sequence"],
            ["answer_language_question", "respond_as_counterpart"],
        )

    def test_process_feedback_does_not_force_roleplay(self):
        decision = self.decide("为什么你又重复上一轮回答？", intent="method_negotiation")
        self.assertEqual(decision["primary_frame"], "process_sidecar")
        self.assertFalse(decision["advance_scene"])

    def test_english_repeat_complaint_stays_on_feedback_and_does_not_restate_answer(self):
        decision = self.decide("Why do you keep repeating the answer?", scene_active=False)
        self.assertEqual(decision["primary_frame"], "process_sidecar")
        self.assertEqual(decision["response_sequence"], ["address_process_feedback"])
        self.assertTrue(any(
            "不要复述已经回答过的事实" in item
            for item in decision["response_obligations"]
        ))

    def test_explicit_memory_commitment_exits_role_without_advancing_scene(self):
        decision = self.decide(
            "我明确记一下：是蓝牙耳机左边没声音，我想换货。以后我也还想多练售后沟通和过去式。",
            intent="memory_commitment",
        )
        self.assertEqual(decision["acts"], ["memory_commitment"])
        self.assertEqual(decision["primary_frame"], "process_sidecar")
        self.assertEqual(decision["response_sequence"], ["confirm_memory_commitment"])
        self.assertFalse(decision["advance_scene"])
        self.assertTrue(decision["replan_required"])

    def test_same_session_recap_is_not_treated_as_roleplay(self):
        decision = self.decide("我们后面还聊了啥吗？", intent="session_recap")
        self.assertEqual(decision["acts"], ["session_recap"])
        self.assertEqual(decision["response_sequence"], ["answer_same_session_recap"])
        self.assertFalse(decision["advance_scene"])

    def test_no_other_followup_stays_in_session_recap(self):
        decision = self.decide("没有其他的了，是吧？")
        self.assertEqual(decision["acts"], ["session_recap"])
        self.assertEqual(decision["primary_frame"], "process_sidecar")

    def test_direct_line_after_sidecar_restores_counterpart(self):
        decision = self.decide(
            "Alright, I'll use another card.",
            previous_frame="language_sidecar",
            previous_decision={"primary_frame": "language_sidecar", "resume_frame": "in_role"},
        )
        self.assertEqual(decision["primary_frame"], "in_role")
        self.assertFalse(decision["replan_required"])

    def test_real_world_question_inside_scene_stays_addressed_to_counterpart(self):
        decision = self.decide("那家便利店在哪里？", intent="real_world_context")
        self.assertEqual(decision["primary_frame"], "in_role")
        self.assertTrue(decision["replan_required"])

    def test_apply_to_plan_makes_route_authoritative(self):
        plan = {
            "interaction_frame": "language_sidecar",
            "response_obligations": ["保留现实事实"],
        }
        decision = self.decide("Can I use cash?")
        TurnController.apply_to_plan(plan, decision)
        self.assertEqual(plan["interaction_frame"], "in_role")
        self.assertEqual(plan["speaker_position"], "counterpart")
        self.assertTrue(plan["user_should_speak_now"])
        self.assertIn("保留现实事实", plan["response_obligations"])

    def test_ordinary_general_english_turn_uses_fast_path(self):
        decision = self.decide(
            "I haven't watched it, but I saw some clips.",
            scene_active=False,
            previous_frame="general",
        )
        self.assertEqual(decision["primary_frame"], "general")
        self.assertFalse(decision["strong_reasoning_required"])
        self.assertIn("general_target_language_turn", decision["routing_reasons"])

    def test_explicit_user_topic_suspends_stale_role_scene(self):
        decision = self.decide(
            "我们先聊聊我最近看的 CS2 比赛吧。",
            scene_active=True,
            previous_frame="in_role",
        )
        self.assertEqual(decision["primary_frame"], "general")
        self.assertIn("free_topic_shift", decision["acts"])
        self.assertFalse(decision["advance_scene"])
        self.assertTrue(decision["strong_reasoning_required"])

    def test_language_sidecar_from_general_topic_does_not_resume_stale_scene(self):
        decision = self.decide(
            "What does atmosphere mean?",
            scene_active=True,
            previous_frame="general",
            previous_decision={"primary_frame": "general"},
        )
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertEqual(decision["resume_frame"], "")

    def test_followup_english_opinion_stays_general_after_topic_shift(self):
        decision = self.decide(
            "My favorite team is Team Spirit.",
            scene_active=True,
            previous_frame="general",
            previous_decision={"primary_frame": "general"},
        )
        self.assertEqual(decision["primary_frame"], "general")
        self.assertNotIn("role_utterance", decision["acts"])
        self.assertFalse(decision["strong_reasoning_required"])
        plan = {
            "response_need": "roleplay_turn",
            "_meta": {"reused_scene_plan": True},
        }
        TurnController.apply_to_plan(plan, decision)
        self.assertEqual(plan["response_need"], "general_chat")

    def test_long_compound_turn_gets_ordered_obligation_planning(self):
        decision = self.decide(
            "我还想知道 atmosphere 怎么说，另外刚才那句时态对不对，以及回答完以后继续聊比赛。",
            scene_active=False,
            previous_frame="general",
        )
        self.assertIn("multiple_response_needs", decision["acts"])
        self.assertIn("multiple_response_needs", decision["routing_reasons"])
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertTrue(any("逐项覆盖" in item for item in decision["response_obligations"]))

    def test_mixed_language_gap_is_routed_as_implicit_expression_support(self):
        decision = self.decide(
            "yeah this map is the uh best uh which one 说是玩家最多的一张图",
            scene_active=False,
            previous_frame="general",
        )
        self.assertIn("implicit_expression_support", decision["acts"])
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertEqual(
            decision["response_sequence"],
            ["supply_natural_expression", "respond_to_intended_meaning"],
        )
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertTrue(any("完整英文" in item for item in decision["response_obligations"]))

    def test_ordinary_bilingual_content_is_not_forced_into_a_sidecar(self):
        decision = self.decide(
            "I watched it yesterday, 但是我不太喜欢结局。",
            scene_active=False,
            previous_frame="general",
        )
        self.assertEqual(decision["primary_frame"], "general")
        self.assertNotIn("implicit_expression_support", decision["acts"])

    def test_short_entity_near_miss_uses_recent_selection_question(self):
        recent = [
            {"role": "user", "content": "我们聊聊 CS2。"},
            {"role": "assistant", "content": "你特别喜欢哪张地图？"},
        ]
        packet = TurnController.contextual_asr_resolution("Does to 吧。", recent)
        self.assertTrue(packet)
        self.assertIn("CS2", packet["topic_hints"])
        decision = self.decide(
            "Does to 吧。",
            scene_active=False,
            previous_frame="general",
            contextual_asr=packet,
        )
        self.assertIn("contextual_asr_resolution", decision["acts"])
        self.assertEqual(decision["primary_frame"], "general")
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertEqual(
            decision["response_sequence"],
            ["resolve_contextual_entity", "continue_current_topic"],
        )

    def test_natural_process_correction_is_not_misrouted_as_chat(self):
        for text in (
            "我跟你说了我最近没玩游戏吗？",
            "你好像出现错乱了。",
        ):
            with self.subTest(text=text):
                decision = self.decide(
                    text, scene_active=False, previous_frame="general"
                )
                self.assertEqual(decision["primary_frame"], "process_sidecar")
                self.assertIn("process_feedback", decision["acts"])

    def test_repeated_spoken_fragments_lower_route_confidence(self):
        decision = self.decide(
            "Because I think the, the conversation is, you know, very wild.",
            scene_active=False,
            previous_frame="general",
        )
        self.assertIn("spoken_semantic_uncertainty", decision["routing_reasons"])
        self.assertLess(decision["confidence"], 0.8)

    def test_topic_release_does_not_force_an_english_hook(self):
        decision = self.decide(
            "我觉得你没有完全理解，算了，不重要了。",
            scene_active=False,
            previous_frame="general",
        )
        self.assertIn("topic_release", decision["acts"])
        self.assertEqual(decision["response_sequence"], ["acknowledge_topic_release"])
        self.assertFalse(decision["advance_scene"])

    def test_explicit_english_topic_release_does_not_force_a_followup(self):
        decision = self.decide(
            "Mm. I am a bit tired of talking about it now.",
            scene_active=False,
            previous_frame="general",
        )
        self.assertIn("topic_release", decision["acts"])
        self.assertEqual(decision["response_sequence"], ["acknowledge_topic_release"])
        self.assertFalse(decision["advance_scene"])

    def test_cleaned_topic_release_and_common_paraphrases_are_recognized(self):
        try:
            from .tutor_agent import TutorAgent
        except ImportError:
            from agent.tutor_agent import TutorAgent

        cases = (
            "Mm. I am a bit tired of talking about it now.",
            "I’m tired of talking about this right now.",
            "I don't want to talk about it.",
        )
        for text in cases:
            with self.subTest(text=text):
                cleaned = TutorAgent._clean_transcript(text)
                decision = self.decide(
                    cleaned, scene_active=False, previous_frame="general"
                )
                self.assertIn("topic_release", decision["acts"])
                self.assertEqual(
                    decision["response_sequence"], ["acknowledge_topic_release"]
                )

    def test_quoted_topic_release_phrase_is_not_a_release(self):
        for text in (
            'Does “I’m tired of talking about it now” sound natural?',
            'Alice said, "I\'m tired of talking about it now."',
        ):
            with self.subTest(text=text):
                decision = self.decide(
                    text, scene_active=False, previous_frame="general"
                )
                self.assertNotIn("topic_release", decision["acts"])

    def test_topic_release_followed_by_new_subject_routes_to_that_subject(self):
        decision = self.decide(
            "I'm tired of talking about it now. Let's talk about music.",
            scene_active=False,
            previous_frame="general",
        )
        self.assertIn("free_topic_shift", decision["acts"])
        self.assertNotIn("topic_release", decision["acts"])
        self.assertEqual(decision["response_sequence"], ["follow_user_led_topic"])

    def test_vague_what_can_i_say_gets_fresh_language_planning(self):
        decision = self.decide(
            "What can I say?", scene_active=False, previous_frame="general"
        )
        self.assertEqual(decision["primary_frame"], "language_sidecar")
        self.assertTrue(decision["strong_reasoning_required"])
        self.assertIn("vague_language_request", decision["routing_reasons"])

    def test_cached_scene_drops_old_turn_correction(self):
        plan = {
            "interaction_frame": "language_sidecar",
            "reply_draft": "上一轮答案",
            "correction_needed": True,
            "corrected_expression": "Old correction",
            "response_obligations": ["重复上一轮问题"],
            "_meta": {"reused_scene_plan": True},
        }
        decision = self.decide("Can I use cash?")
        TurnController.apply_to_plan(plan, decision)
        self.assertEqual(plan["reply_draft"], "")
        self.assertFalse(plan["correction_needed"])
        self.assertNotIn("重复上一轮问题", plan["response_obligations"])


if __name__ == "__main__":
    unittest.main()
