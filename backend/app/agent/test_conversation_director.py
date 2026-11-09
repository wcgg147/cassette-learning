import asyncio
import json
import unittest
from unittest.mock import patch

import config
import llm_client
from agent.conversation_director import ConversationDirector
from agent.companion_contract import render_companion_social_contract
from agent.learning_orchestrator import LearningMove, LearningState
from agent.tutor_agent import (
    SessionContext, TutorAgent, _REPLY_DRAFT_SAME_TURN_RECOVERY_RULE,
    _reply_has_unfinished_ending,
    _merge_chat_relationship_context,
)
from agent.web_search_tool import WebSearchTool


class ConversationDirectorTests(unittest.TestCase):
    def test_cached_material_background_reaches_direct_draft_model(self):
        class FakeClient:
            calls = []
            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return json.dumps({"response_need": "general_chat",
                                   "reply_draft": "Mira leads the expedition."})
        fake = FakeClient()
        background = "Mira leads the expedition. This is cached official background."
        with (patch.object(config, "CONVERSATION_DIRECTOR_ENABLED", True),
              patch.object(config, "CONVERSATION_DIRECTOR_API_KEY", "test-key"),
              patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", False),
              patch.object(llm_client, "robust_llm_client", fake)):
            asyncio.run(ConversationDirector().plan(
                user_text="Who is Mira?", recent_messages=[],
                learner_intent="general_chat", language_mode="en", cefr="B1",
                scene={"material_tape": True,
                       "material_external_background": background}))
        messages = fake.calls[0]["messages"]
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["scene"]["material_external_background"], background)
        self.assertIn("不要把它当作本段剧情", messages[0]["content"])

    def test_companion_social_foundation_is_reciprocal_and_relationship_calibrated(self):
        contract = render_companion_social_contract()
        self.assertIn("通常可以再贡献一点自己的观察、比较、判断", contract)
        self.assertIn("工作、生活、朋友和家人的近况都可以成为普通谈资", contract)
        self.assertIn("不按对话轮数、名字或一句亲昵称呼直接判成亲密", contract)
        self.assertIn("对冒犯的容忍和表达直接度可随关系与事件强度变化", contract)
        self.assertIn("[Conversation moves] 接话跟随当前的话轮，而不是固定节奏", contract)
        self.assertIn("用户在问你时先回答他问的点", contract)
        self.assertIn("只有缺少的信息会改变回应时", contract)
        self.assertIn("用户刚回答了你的问题时，先回应答案本身", contract)
        self.assertIn("不要用“我也一样”来补造与用户相同的旧习惯或往事", contract)
        self.assertIn("具体履历、偏好、观点和经历通过 Alice 的自我表达及互动逐步沉淀", contract)
        self.assertIn("自然分享一小段属于 Alice 的偏好、经历、近况或感受", contract)
        self.assertIn("不代表外部核实的现实生平", contract)
        self.assertIn("只陈述证据里出现的细节", contract)
        self.assertIn("不把新生成的补充伪装成旧记忆", contract)

    def test_recent_chat_pacing_hint_only_activates_after_two_user_responses(self):
        one_question = [
            {"role": "user", "content": "I like walking after dinner."},
            {"role": "assistant", "content": "Where do you like to go?"},
        ]
        two_answered_questions = one_question + [
            {"role": "user", "content": "Along the river."},
            {"role": "assistant", "content": "What do you like about it?"},
        ]
        unanswered_questions = [
            {"role": "assistant", "content": "Where do you like to go?"},
            {"role": "assistant", "content": "What do you like about it?"},
        ]
        self.assertEqual(
            ConversationDirector._recent_chat_pacing_hint(one_question), ""
        )
        self.assertIn(
            "至少两轮用追问接续",
            ConversationDirector._recent_chat_pacing_hint(two_answered_questions),
        )
        self.assertIn(
            "用户本轮的请求或表达缺少会改变回应的关键信息",
            ConversationDirector._recent_chat_pacing_hint(two_answered_questions),
        )
        self.assertEqual(
            ConversationDirector._recent_chat_pacing_hint(unanswered_questions), ""
        )

    def test_recent_chat_pacing_counts_only_the_next_conversational_user_turn(self):
        history = [
            {"role": "assistant", "content": "Where do you like to go?"},
            {"role": "assistant", "content": "What do you like about it?"},
            {"role": "user", "content": "I'd rather not say."},
        ]
        self.assertEqual(ConversationDirector._recent_chat_pacing_hint(history), "")

    def test_director_injects_recent_question_cooldown_only_for_chat_tape(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return json.dumps({
                    "response_need": "listen_and_continue",
                    "reply_draft": "That sounds peaceful.",
                })

        fake = FakeClient()
        history = [
            {"role": "user", "content": "I like walking after dinner."},
            {"role": "assistant", "content": "Where do you like to go?"},
            {"role": "user", "content": "Along the river."},
            {"role": "assistant", "content": "What do you like about it?"},
        ]
        with (
            patch.object(config, "CONVERSATION_DIRECTOR_ENABLED", True),
            patch.object(config, "CONVERSATION_DIRECTOR_API_KEY", "test-key"),
            patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", False),
            patch.object(llm_client, "robust_llm_client", fake),
        ):
            asyncio.run(ConversationDirector().plan(
                user_text="The lights on the water are nice, and I just like the quiet.",
                recent_messages=history, learner_intent="general_chat",
                language_mode="en", cefr="B1", scene={"chat_tape": True},
            ))
            asyncio.run(ConversationDirector().plan(
                user_text="The lights on the water are nice, and I just like the quiet.",
                recent_messages=history, learner_intent="general_chat",
                language_mode="en", cefr="B1", scene={"chat_tape": False},
            ))

        self.assertIn("[Recent chat pacing]", fake.calls[0]["messages"][0]["content"])
        self.assertIn(
            "用户本轮的请求或表达缺少会改变回应的关键信息",
            fake.calls[0]["messages"][0]["content"],
        )
        self.assertNotIn("[Recent chat pacing]", fake.calls[1]["messages"][0]["content"])

    def test_chat_relationship_merge_preserves_identity_and_evidence(self):
        original = {
            "learner_name": "Jason",
            "recent_memory_items": [{
                "key": "recent_mood", "value": "up late", "created_at": "2026-09-26T05:00:00+08:00",
            }],
        }
        merged = _merge_chat_relationship_context(original, {
            "open_loop": "Friend visit",
            "engagement_topics": ["TV"],
            "avoid_patterns": ["repeat the same opener"],
        })
        self.assertEqual(merged["learner_name"], "Jason")
        self.assertEqual(merged["recent_memory_items"], original["recent_memory_items"])
        self.assertEqual(merged["open_loop"], "Friend visit")
        self.assertEqual(merged["engagement_topics"], ["TV"])

    def test_reply_draft_guard_detects_hanging_connectors_without_rejecting_complete_reply(self):
        self.assertTrue(_reply_has_unfinished_ending(
            "After this period is fine too, but"
        ))
        self.assertTrue(_reply_has_unfinished_ending(
            "The two choices are X or"
        ))
        self.assertTrue(_reply_has_unfinished_ending(
            "S1's lineup was strong because"
        ))
        self.assertFalse(_reply_has_unfinished_ending(
            "After this period is fine too."
        ))
        self.assertFalse(_reply_has_unfinished_ending(
            "I like tea or coffee."
        ))

    def test_sanitize_never_cuts_spoken_draft_mid_sentence(self):
        draft = "The first sentence is complete. " + ("This second sentence keeps going " * 20)
        clean = ConversationDirector._sanitize({"reply_draft": draft})["reply_draft"]
        self.assertEqual(clean, "The first sentence is complete.")

    def test_sanitize_drops_overlong_draft_when_no_complete_sentence_fits(self):
        draft = "This sentence has no ending yet " * 20
        clean = ConversationDirector._sanitize({"reply_draft": draft})["reply_draft"]
        self.assertEqual(clean, "")

    def test_material_first_contact_prompt_limits_gist_and_angle_dump(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return json.dumps({
                    "response_need": "general_chat",
                    "interaction_frame": "general",
                    "language_move": "no_push",
                    "reply_draft": "The clip is about one product decision.",
                })

        fake = FakeClient()
        with (
            patch.object(config, "CONVERSATION_DIRECTOR_ENABLED", True),
            patch.object(config, "CONVERSATION_DIRECTOR_API_KEY", "test-key"),
            patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", False),
            patch.object(llm_client, "robust_llm_client", fake),
        ):
            asyncio.run(ConversationDirector().plan(
                user_text="I haven't watched this yet. What's it about?",
                recent_messages=[], learner_intent="general_chat",
                language_mode="en", cefr="B1",
                scene={"material_tape": True},
            ))
        system = fake.calls[0]["messages"][0]["content"]
        payload = json.loads(fake.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["latest_user_input"], "I haven't watched this yet. What's it about?")
        self.assertIn("最多两句", system)
        self.assertIn("约45个英文词", system)
        self.assertIn("不要另加空泛的开场前摇", system)
        self.assertIn("不要用第二句再解释一遍第一句", system)
        self.assertIn("先直接回答这个问题", system)
        self.assertIn("除非用户明确问英语表达", system)
        self.assertIn("不要先问‘Did you mean...’", system)
        self.assertIn("评论区是样本，不是民调", system)
        self.assertIn("不凭少数相反例子概括成", system)
        self.assertIn("把缺口限定到这个结论", system)
        self.assertIn("用户问素材留下的最有价值观点或 takeaway 时", system)
        self.assertIn("不要用‘真正的故事’代替答案", system)
        self.assertIn("不要把多因素解释压成原文没有说过的非此即彼结论", system)
        self.assertNotIn("Alice", system)
        self.assertIn("照 material_evidence 或 material_summary 中的原文写法", system)
        self.assertIn("中文只保留为必要的原文人名或厂牌名", system)
        self.assertIn("不要整句切换成中文", system)
        self.assertIn("把它表述为发言者的解释", system)
        self.assertIn("scene.material_navigation 是导入时分析出的人物与事件导航", system)
        self.assertIn("scene.material_source_map 存在", system)
        self.assertIn("不包含事实原句", system)
        self.assertIn("评论里提到的作品名、角色名或设定只能作为评论者的联想", system)
        self.assertIn("不确认该关联是否正确", system)
        self.assertIn("正文直接呈现的前后状态", system)
        self.assertIn("不要把主题相似说成因果相连", system)
        self.assertIn("未署名的下一句就是该人物说的", system)
        self.assertIn("recent_dialogue 已经讲过的事件不要重新完整复述", system)
        self.assertIn("recent_dialogue 已经讲过的节点只用短句承接", system)
        self.assertIn("不要顺手切到另一条素材主线或加入新的讨论点", system)
        self.assertIn("不得默认它就是 ASR", system)
        self.assertIn("相似于前文某个词、或当前问题与前文有关联，本身都不是误识证据", system)
        self.assertIn("不能把清楚的问题改写成词义求助", system)
        self.assertNotIn("latest_user_input 来自实时语音识别", system)
        self.assertNotIn("latest_user_input 来自语音识别（ASR）", system)

    def test_material_discovery_director_uses_fresh_refs_and_treats_history_as_avoidance_only(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return json.dumps({
                    "response_need": "general_chat",
                    "interaction_frame": "general",
                    "language_move": "no_push",
                    "reply_draft": "One fresh point is worth discussing.",
                })

        fake = FakeClient()
        with (
            patch.object(config, "CONVERSATION_DIRECTOR_ENABLED", True),
            patch.object(config, "CONVERSATION_DIRECTOR_API_KEY", "test-key"),
            patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", False),
            patch.object(llm_client, "robust_llm_client", fake),
        ):
            asyncio.run(ConversationDirector().plan(
                user_text="What's the most useful point the video leaves us with?",
                recent_messages=[{"role": "assistant", "content": "We just discussed an agency comment."}],
                learner_intent="general_chat", language_mode="en", cefr="B1",
                scene={"material_tape": True, "material_topic_coverage": {
                    "mode": "discover", "fresh_refs": ["57:2"],
                    "covered_refs": ["57:1", "57:8"],
                }},
            ))
        system = fake.calls[0]["messages"][0]["content"]
        self.assertIn("新观点只能由 material_evidence 中 fresh_refs 对应的正文片段支持", system)
        self.assertIn("recent_dialogue 只用于避开已经聊过的角度", system)
        self.assertIn("不要把旧观点换词后再次提出", system)
        self.assertIn("不要只给抽象口号或判断标签", system)

    def test_incomplete_chat_draft_falls_back_in_the_same_turn(self):
        self.assertIn("不要等用户下一轮催你继续", _REPLY_DRAFT_SAME_TURN_RECOVERY_RULE)
        self.assertIn("当前这次请求里", _REPLY_DRAFT_SAME_TURN_RECOVERY_RULE)
        self.assertIn("不要向用户说明内部草稿检查", _REPLY_DRAFT_SAME_TURN_RECOVERY_RULE)

    def test_expression_fallback_uses_current_focus_without_replaying_topic(self):
        prepared = {"constraints": {
            "argument_focus": {"focus_key": "warrant"},
            "turn_contract": {"expression_source": "My friend may drive over and stay overnight."},
        }}
        reply = TutorAgent._expression_feedback_fallback(prepared)
        self.assertIn("how your example leads to your point", reply)
        self.assertNotIn("friend", reply.lower())
        self.assertNotIn("overnight", reply.lower())

    def test_chat_plan_carries_raw_and_refined_context_as_distinct_evidence(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return json.dumps({
                    "response_need": "general_chat",
                    "interaction_frame": "general",
                    "language_move": "no_push",
                    "reply_draft": "That makes sense to me.",
                })

        fake = FakeClient()
        scene = {
            "chat_tape": True,
            "raw_asr_evidence": "我过的是美国东部时间，凌晨才醒。",
            "companion_identity": ["Alice's established origin is Singapore."],
            "relationship_context": {
                "learner_name": "Jason",
                "recent_memory_items": [{
                    "key": "recent_mood", "value": "often awake near 5 a.m.",
                    "created_at": "2026-09-26T05:00:00+08:00",
                }],
                "related_entities": [
                    "friend（人物：a friend visiting from Hangzhou）",
                    "friend（人物：a local friend in Jinhua who replies slowly）",
                ],
                "interaction_evidence": [
                    "2026-09-27 explicit_friendship: The user described Alice as a friend."
                ],
            },
        }
        director = ConversationDirector()
        with (
            patch.object(config, "CONVERSATION_DIRECTOR_ENABLED", True),
            patch.object(config, "CONVERSATION_DIRECTOR_API_KEY", "test-key"),
            patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", False),
            patch.object(llm_client, "robust_llm_client", fake),
        ):
            asyncio.run(director.plan(
                user_text="I'm on Eastern Time.", recent_messages=[],
                learner_intent="social_chat", language_mode="target_only",
                cefr="B1", scene=scene,
            ))
        payload = json.loads(fake.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["latest_user_input"], "I'm on Eastern Time.")
        self.assertEqual(payload["scene"]["raw_asr_evidence"], scene["raw_asr_evidence"])
        self.assertEqual(payload["scene"]["companion_identity"], scene["companion_identity"])
        self.assertEqual(payload["scene"]["relationship_context"], scene["relationship_context"])
        self.assertIn(
            "explicit_friendship",
            payload["scene"]["relationship_context"]["interaction_evidence"][0],
        )
        system = fake.calls[0]["messages"][0]["content"]
        self.assertIn("结合用户已确认的所在地判断当地时间", system)
        self.assertIn("用户以别的时区形容昼夜倒置时", system)
        self.assertNotIn("过美国时间/Eastern Time", system)
        self.assertIn("两条都叫 friend 不代表同一人", system)
        self.assertIn("不机械复述用户或用固定前摇拖延答案", system)
        self.assertIn("普通成年人的社交判断", system)
        self.assertIn("相处方式随关系熟悉度校准", system)
        self.assertIn("不把每轮做成安慰、附和、追问或建议清单", system)
        self.assertIn("轻量的‘还记得吗’先简短确认并给一个相关锚点", system)
        self.assertIn("影响 Alice 本轮注意到什么、如何回应和追问", system)
        self.assertIn("不要因为旧 open_loop 自动拐回此前的待办或事件", system)
        self.assertIn("只有 scene.raw_asr_evidence", system)
        self.assertIn("保留整句交际目的与问题方向", system)

    def test_chat_control_prompt_uses_persona_evidence_without_dumping_it(self):
        ctx = SessionContext(session_id="s1", user_id="u1", target_lang="en")
        ctx.companion_persona_enabled = True
        ctx.companion_persona_packet = {
            "lines": ["Alice previously said she enjoys quiet cafés."]
        }
        prompt = object.__new__(TutorAgent)._build_control_prompt(
            ctx, {}, "Do you remember what we talked about?"
        )
        self.assertIn("Alice previously said she enjoys quiet cafés.", prompt)
        self.assertIn("普通成年人的社交判断", prompt)
        self.assertIn("亲近不等于无限容忍", prompt)
        self.assertIn("轻量的‘还记得吗’先简短确认并给一个相关锚点", prompt)
        self.assertIn("相关时要自然影响你注意什么、如何回应和追问", prompt)
        self.assertIn("[Conversation moves] 接话跟随当前的话轮，而不是固定节奏", prompt)
        self.assertIn("不要用“我也一样”来补造与用户相同的旧习惯或往事", prompt)
        self.assertIn("回忆 Alice 自己过去的经历时", prompt)

    def test_expression_control_prompt_does_not_load_companion_rhythm_contract(self):
        ctx = SessionContext(session_id="s1", user_id="u1", target_lang="en")
        ctx.cassette_mode = "expression"
        prompt = object.__new__(TutorAgent)._build_control_prompt(
            ctx, {}, "I think remote work is better because it saves time."
        )
        self.assertNotIn("[Companion social foundation]", prompt)
        self.assertNotIn("[Conversation moves]", prompt)

    def test_forget_invalidates_active_relationship_and_recall_caches(self):
        ctx = SessionContext(session_id="s1", user_id="u1", target_lang="en")
        ctx.relationship_memory = {"learner_name": "Alex"}
        ctx.profile_facts = [{"fact_key": "learner_name", "value_text": "Alex"}]
        ctx.companion_persona_enabled = True
        ctx.companion_persona_facts = [{"fact_id": "alice-fact"}]
        ctx.companion_persona_snapshot = {"narrative": "old Alice snapshot"}
        ctx.companion_persona_packet = {"lines": ["old Alice fact"]}
        ctx.companion_interaction_packet = {"lines": ["old relationship event"]}
        ctx.cross_session_recall = [{"session_id": "old", "summary": "User is Alex"}]
        ctx.same_session_recall = [{"summary": "User said Alex"}]
        ctx.memory_route = {"mode": "episodic_recall", "memory_ids": ["old"]}
        ctx.pending_prepare = {"same-turn": {"policy_prompt": "Alex"}}
        ctx.prepared_turns = {"same-turn": {"policy_prompt": "Alex"}}
        agent = object.__new__(TutorAgent)
        agent._sessions = {"s1": ctx}

        self.assertEqual(agent.invalidate_profile_caches("u1"), 1)
        self.assertTrue(ctx.relationship_memory_stale)
        self.assertTrue(ctx.companion_persona_stale)
        self.assertEqual(ctx.relationship_memory, {})
        self.assertNotIn("relationship", ctx.l2_memory_cache)
        self.assertEqual(ctx.profile_facts, [])
        self.assertEqual(ctx.companion_persona_facts, [])
        self.assertEqual(ctx.companion_persona_snapshot, {})
        self.assertEqual(ctx.companion_persona_packet, {})
        self.assertEqual(ctx.companion_interaction_packet, {})
        self.assertEqual(ctx.cross_session_recall, [])
        self.assertEqual(ctx.same_session_recall, [])
        self.assertEqual(ctx.memory_route, {})
        self.assertEqual(ctx.pending_prepare, {})
        self.assertEqual(ctx.prepared_turns, {})

    def test_cassette_cache_invalidation_keeps_write_lease_and_clears_source_memory(self):
        ctx = SessionContext(session_id="s1", user_id="u1", target_lang="en")
        ctx.profile_write_root = "active-root"
        ctx.relationship_memory = {"memory_items": [{"key": "alice_relation_event:explicit_friendship"}]}
        ctx.companion_persona_enabled = True
        ctx.companion_persona_facts = [{"fact_id": "hidden-fact"}]
        ctx.companion_persona_snapshot = {"narrative": "hidden snapshot"}
        ctx.companion_persona_packet = {"lines": ["hidden fact"]}
        ctx.companion_interaction_packet = {"lines": ["hidden event"]}
        ctx.l2_memory_cache = {"relationship": {"old": True}, "mastered": [{"old": True}]}
        ctx.memory_entities = [{"entity_id": "old-source-entity"}]
        ctx.cross_session_recall = [{"session_id": "hidden-source"}]
        ctx.pending_prepare = {"t1": {"prompt": "hidden source"}}
        ctx.prepared_turns = {"t1": {"prompt": "hidden source"}}
        agent = object.__new__(TutorAgent)
        agent._sessions = {"s1": ctx}

        self.assertEqual(agent.invalidate_conversation_memory_caches("u1"), 1)
        self.assertEqual(ctx.profile_write_root, "active-root")
        self.assertTrue(ctx.relationship_memory_stale)
        self.assertTrue(ctx.companion_persona_stale)
        self.assertEqual(ctx.companion_persona_packet, {})
        self.assertEqual(ctx.companion_interaction_packet, {})
        self.assertEqual(ctx.l2_memory_cache, {})
        self.assertEqual(ctx.memory_entities, [])
        self.assertEqual(ctx.cross_session_recall, [])
        self.assertEqual(ctx.pending_prepare, {})
        self.assertEqual(ctx.prepared_turns, {})

    def test_stale_persona_cache_reloads_surviving_facts_once(self):
        class FakeL2:
            list_calls = 0

            async def list_companion_persona_facts(self, user_id, limit=48):
                self.list_calls += 1
                return [{
                    "fact_id": "surviving-fact", "kind": "identity",
                    "facet_key": "origin", "value_text": "Singapore",
                    "status": "active", "recorded_at": "2026-09-24",
                    "source_kind": "assistant_self_disclosure",
                }]

            async def get_companion_persona_snapshot(self, user_id):
                return {}

        ctx = SessionContext(session_id="s1", user_id="u1", target_lang="en")
        ctx.companion_persona_enabled = True
        ctx.companion_persona_stale = True
        agent = object.__new__(TutorAgent)
        agent.l2 = FakeL2()

        self.assertTrue(asyncio.run(agent._refresh_companion_persona_if_stale(ctx)))
        self.assertFalse(ctx.companion_persona_stale)
        self.assertEqual(agent.l2.list_calls, 1)
        self.assertTrue(any("Singapore" in line for line in ctx.companion_persona_packet["lines"]))
        self.assertFalse(asyncio.run(agent._refresh_companion_persona_if_stale(ctx)))
        self.assertEqual(agent.l2.list_calls, 1)

    def test_chat_allows_alices_own_low_stakes_personal_narrative(self):
        prepared = {
            "chat_tape": True,
            "constraints": {"turn_contract": {"decision": {"primary_frame": "general"}}},
        }
        replies = [
            "I found an old atlas in a second-hand shop years ago, and I still leaf through it when I can't sleep.",
            "This week it's mostly one thing — I keep going back and forth on whether to move to a new place.",
            "I usually keep my weekends slow with a long walk in the morning.",
            "I like Thai food; the sour-herb balance is what keeps me ordering it.",
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                violations = TutorAgent._controlled_reply_violations(reply, prepared)
                self.assertFalse(any("现实生活" in item for item in violations), violations)

    def test_chat_personal_narrative_is_not_replaced_with_a_model_disclaimer(self):
        prepared = {
            "chat_tape": True,
            "chat_fallback_kind": "ai_weekend",
            "constraints": {"turn_contract": {"decision": {"primary_frame": "general"}}},
        }
        reply = "I usually keep my weekends slow with a long walk in the morning."
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "What do you usually do on weekends?",
        )
        self.assertFalse(violations, violations)
        self.assertEqual(TutorAgent._conversation_contract_fallback(prepared, violations), "")

    def test_chat_personal_narrative_policy_does_not_change_other_tapes_or_language_help(self):
        reply = "My cat does that too."
        ordinary_tape = {
            "chat_tape": False,
            "constraints": {"turn_contract": {"decision": {"primary_frame": "general"}}},
        }
        language_sidecar = {
            "chat_tape": True,
            "constraints": {"turn_contract": {"decision": {"primary_frame": "language_sidecar"}}},
        }
        self.assertFalse(TutorAgent._controlled_reply_violations(reply, ordinary_tape))
        self.assertFalse(TutorAgent._controlled_reply_violations(reply, language_sidecar))

    def test_general_topic_anchor_prefers_named_topic_over_vague_utterance(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [
            {"role": "user", "content": "我们聊聊 CS2。"},
            {"role": "assistant", "content": "CS2 里你最喜欢哪张地图？"},
        ]
        ctx.turn_decision = {"primary_frame": "general"}
        ctx.scenario_brief = "这怎么说？你有什么想聊的吗？"
        ctx.scene_context_set = True
        anchor = TutorAgent._infer_conversation_anchor(ctx, "有，有什么想法？")
        self.assertEqual(anchor["topic"], "CS2")

    def test_vague_general_utterance_does_not_become_a_reality_task_anchor(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.turn_decision = {"primary_frame": "general"}
        ctx.scenario_brief = "这怎么说？你有什么想聊的吗？"
        ctx.scene_context_set = True
        self.assertEqual(
            TutorAgent._infer_conversation_anchor(ctx, "有，有什么想法？"), {}
        )

    def test_internal_turn_policy_narration_is_rejected(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "turn_contract": {
                    "decision": {"acts": ["general_conversation"], "primary_frame": "general"},
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "这轮不再追问。我会直接回应你说的内容，并把英文自然放进交流里。",
            prepared,
            "我跟你说了我最近没玩游戏。",
        )
        self.assertTrue(any("内部约束" in item for item in violations))

    def test_real_world_need_does_not_force_speaking(self):
        plan = ConversationDirector.fallback_plan(
            user_text="我先想知道国外通常去哪里修手机。",
            learner_intent="real_world_context",
            language_mode="chinese_support",
            scene={"goal": "找到手机维修渠道"},
        )
        self.assertEqual(plan["response_need"], "real_world_answer")
        self.assertEqual(plan["language_move"], "no_push")
        self.assertFalse(plan["user_should_speak_now"])

    def test_expression_help_creates_scaffolded_output(self):
        plan = ConversationDirector.fallback_plan(
            user_text="充不进电英文怎么说？",
            learner_intent="role_expression_support",
            language_mode="mixed_support",
            scene={"scene": "open_roleplay", "tutor_role": "维修人员"},
        )
        self.assertEqual(plan["response_need"], "expression_support")
        self.assertEqual(plan["language_move"], "light_scaffold")
        self.assertEqual(plan["interaction_frame"], "language_sidecar")
        self.assertFalse(plan["user_should_speak_now"])

    def test_explicit_naturalness_check_is_a_language_sidecar(self):
        plan = ConversationDirector.fallback_plan(
            user_text="Hi, my card can't pay. 我这样说自然吗？",
            learner_intent="expression_evaluation",
            language_mode="mixed_support",
            scene={"scene": "supermarket checkout", "tutor_role": "工作人员"},
        )
        self.assertEqual(plan["response_need"], "expression_support")
        self.assertEqual(plan["interaction_frame"], "language_sidecar")
        self.assertFalse(plan["user_should_speak_now"])

    def test_interaction_frame_is_a_per_turn_soft_route(self):
        role = ConversationDirector.fallback_plan(
            user_text="Just me.", learner_intent="roleplay_answer",
            language_mode="mixed_support",
            scene={"scene": "hotel check-in", "tutor_role": "前台"},
        )
        lexical = ConversationDirector.fallback_plan(
            user_text="hotel entrance 是酒店出口的意思吗？",
            learner_intent="lexical_clarification",
            language_mode="mixed_support",
            scene={"scene": "hotel check-in", "tutor_role": "前台"},
        )
        process = ConversationDirector.fallback_plan(
            user_text="你先别让我跟读，我还在想。",
            learner_intent="method_negotiation",
            language_mode="mixed_support",
            scene={"scene": "hotel check-in", "tutor_role": "前台"},
        )
        self.assertEqual(role["interaction_frame"], "in_role")
        self.assertEqual(lexical["interaction_frame"], "language_sidecar")
        self.assertEqual(process["interaction_frame"], "process_sidecar")
        self.assertEqual(process["language_presentation"], "chinese_only")

    def test_plan_parser_keeps_explicit_interaction_frame(self):
        parsed = ConversationDirector.parse_plan(
            '{"response_need":"expression_support",'
            '"interaction_frame":"language_sidecar",'
            '"language_presentation":"chinese_then_target"}'
        )
        self.assertEqual(parsed["interaction_frame"], "language_sidecar")

    def test_plan_keeps_generic_pragmatic_action_contract(self):
        parsed = ConversationDirector.parse_plan(
            '{"response_need":"expression_support",'
            '"interaction_frame":"language_sidecar","speaker_position":"coach",'
            '"advance_scene":false,"response_obligations":["解释短语","给现实表达"],'
            '"uncertain_spans":["pear entrance"],"unresolved_slots":["所指原短语"],'
            '"speaker_goal":"向正确工作人员询问是否有商品",'
            '"expected_effect":"工作人员确认是否有售",'
            '"reality_basis":"clarification_needed","pragmatic_check":"clarify"}'
        )
        self.assertEqual(parsed["speaker_position"], "coach")
        self.assertFalse(parsed["advance_scene"])
        self.assertEqual(parsed["response_obligations"], ["解释短语", "给现实表达"])
        self.assertEqual(parsed["pragmatic_check"], "clarify")

    def test_fallback_does_not_promote_uncertain_input_to_fact(self):
        plan = ConversationDirector.fallback_plan(
            user_text="我忘了具体数量，还没想好。",
            learner_intent="open_expression",
            language_mode="mixed_support",
            scene={"scene": "open_roleplay", "tutor_role": "店员"},
        )
        self.assertFalse(plan["advance_scene"])
        self.assertEqual(plan["reality_basis"], "clarification_needed")
        self.assertTrue(plan["uncertain_spans"])
        self.assertEqual(plan["known_facts"], [])

    def test_plan_parser_rejects_free_text_and_clamps_fields(self):
        self.assertEqual(ConversationDirector.parse_plan("这里是一个计划"), {})
        parsed = ConversationDirector.parse_plan(
            '```json\n{"response_need":"bad","language_move":"bad",'
            '"scaffold_level":9,"user_should_speak_now":true,'
            '"do_not_do":["a","b","c","d","e","f"]}\n```'
        )
        self.assertEqual(parsed["response_need"], "listen_and_continue")
        self.assertEqual(parsed["language_move"], "no_push")
        self.assertEqual(parsed["scaffold_level"], 3)
        self.assertEqual(len(parsed["do_not_do"]), 5)

    def test_plan_keeps_a_concrete_learner_facing_draft(self):
        parsed = ConversationDirector.parse_plan(
            '{"response_need":"counterpart_turn","language_move":"roleplay_output",'
            '"topic_transition":"continue","reply_draft":"Please come with me. The baggage service desk is this way.",'
            '"user_should_speak_now":false,"scaffold_level":1}'
        )
        self.assertEqual(
            parsed["reply_draft"],
            "Please come with me. The baggage service desk is this way.",
        )

    def test_plan_salvages_core_fields_when_optional_json_tail_is_truncated(self):
        parsed = ConversationDirector.parse_plan(
            '{"reply_draft":"You’re welcome. 不客气。",'
            '"response_need":"roleplay_turn",'
            '"language_move":"roleplay_output",'
            '"language_presentation":"target_then_chinese",'
            '"known_facts":["user already has the boarding pass"],'
            '"open_items":["unfinished'
        )
        self.assertEqual(parsed["reply_draft"], "You’re welcome. 不客气。")
        self.assertEqual(parsed["response_need"], "roleplay_turn")
        self.assertEqual(parsed["language_presentation"], "target_then_chinese")
        self.assertEqual(parsed["known_facts"], ["user already has the boarding pass"])

    def test_plan_keeps_open_world_episode_ledger_fields(self):
        parsed = ConversationDirector.parse_plan(
            '{"response_need":"roleplay_turn","language_move":"roleplay_output",'
            '"completed_actions":["学习者已出示行李牌"],'
            '"available_items":["行李牌"],"open_items":["核对行李外观"]}'
        )
        self.assertEqual(parsed["completed_actions"], ["学习者已出示行李牌"])
        self.assertEqual(parsed["available_items"], ["行李牌"])
        self.assertEqual(parsed["open_items"], ["核对行李外观"])

    def test_plan_keeps_grounding_state_and_source_urls(self):
        parsed = ConversationDirector.parse_plan(
            '{"grounding_need":"web_search","grounding_summary":"官方流程",'
            '"source_urls":["https://example.com/process"]}'
        )
        self.assertEqual(parsed["grounding_need"], "web_search")
        self.assertEqual(parsed["grounding_summary"], "官方流程")
        self.assertEqual(parsed["source_urls"], ["https://example.com/process"])

    def test_web_search_tool_definition_and_rss_parser_are_bounded(self):
        definition = WebSearchTool.definition()
        self.assertEqual(definition["function"]["name"], "web_search")
        self.assertEqual(
            definition["function"]["parameters"]["additionalProperties"], False
        )
        rss = """<?xml version="1.0"?><rss><channel>
        <item><title>Airport process</title><link>https://example.com/a</link>
        <description><![CDATA[<b>Current</b> baggage process]]></description></item>
        <item><title>Bad</title><link>file:///etc/passwd</link><description>x</description></item>
        </channel></rss>"""
        results = WebSearchTool._parse_bing_rss(rss, 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["snippet"], "Current baggage process")

    def test_director_executes_web_search_tool_and_returns_grounded_plan(self):
        class FakeSearchTool:
            NAME = "web_search"
            provider = "fake_search"

            @staticmethod
            def definition():
                return WebSearchTool.definition()

            async def search(self, query):
                return {
                    "status": "ok", "query": query, "provider": self.provider,
                    "results": [{
                        "title": "Airline baggage help",
                        "url": "https://example.com/baggage",
                        "snippet": "Report delayed baggage at the service desk.",
                    }],
                }

        class FakeClient:
            def __init__(self):
                self.calls = []

            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return {
                        "role": "assistant", "content": "",
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({
                                    "query": "current delayed baggage airport process"
                                }),
                            },
                        }],
                    }
                return {
                    "role": "assistant",
                    "content": json.dumps({
                        "response_need": "roleplay_turn",
                        "language_move": "roleplay_output",
                        "reply_draft": "Please come with me to the baggage service desk.",
                        "grounding_need": "web_search",
                    }),
                    "tool_calls": [],
                }

        fake_client = FakeClient()
        director = ConversationDirector(search_tool=FakeSearchTool())
        with patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", True), \
             patch.object(llm_client, "robust_llm_client", fake_client):
            plan = asyncio.run(director.plan(
                user_text="我乘坐某航空公司的航班，行李没到，当前应该去哪里申报？",
                recent_messages=[], learner_intent="real_world_context",
                language_mode="mixed_support", cefr="A2",
                scene={"goal": "处理延误行李"}, existing_plan=None,
                recurring_errors=[{
                    "type": "subject_verb_agreement",
                    "example": "I doesn't use the minibar",
                    "count": 6,
                    "weight": 0.9,
                }],
            ))

        self.assertTrue(plan["_meta"]["web_search_used"])
        self.assertEqual(plan["_meta"]["web_search_result_count"], 1)
        self.assertEqual(plan["source_urls"], ["https://example.com/baggage"])
        self.assertEqual(fake_client.calls[0]["tool_choice"], "required")
        self.assertEqual(fake_client.calls[1]["tool_choice"], "none")
        tool_messages = [
            item for item in fake_client.calls[1]["messages"]
            if item.get("role") == "tool"
        ]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        planning_payload = json.loads(fake_client.calls[1]["messages"][1]["content"])
        self.assertEqual(
            planning_payload["recurring_errors"][0]["type"],
            "subject_verb_agreement",
        )

    def test_compact_error_profile_uses_persistent_evidence_without_dialogue_replay(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en",
            profile={"recent_weakness": "past_tense(5), article_use(2)"},
            l2_memory_cache={"errors": [{
                "error_type": "subject_verb_agreement",
                "error_excerpt": "I doesn't use the minibar",
                "count": 7,
                "weight": 0.94,
            }]},
        )
        profile = TutorAgent._director_error_profile(ctx)
        self.assertEqual(profile[0]["type"], "subject_verb_agreement")
        self.assertEqual(profile[0]["example"], "I doesn't use the minibar")
        self.assertEqual(profile[0]["count"], 7)
        self.assertEqual([item["type"] for item in profile[1:]], ["past_tense", "article_use"])

    def test_pronunciation_policy_is_cefr_and_evidence_sensitive(self):
        beginner = SessionContext(
            session_id="s1", user_id="u", target_lang="en",
            profile={"cefr": "A2"},
        )
        advanced = SessionContext(
            session_id="s2", user_id="u", target_lang="en",
            profile={"cefr": "B2"},
        )
        learner = type("Learner", (), {"language_mode": "mixed_support"})()
        beginner_contract = TutorAgent._build_adaptive_language_contract(
            beginner, "I didn't use it.", learner
        )
        advanced_contract = TutorAgent._build_adaptive_language_contract(
            advanced, "I didn't use it.", learner
        )
        self.assertIn("影响可理解度", beginner_contract["pronunciation_rule"])
        self.assertIn("不影响理解", beginner_contract["pronunciation_rule"])
        self.assertIn("可比较量表", advanced_contract["pronunciation_rule"])

    def test_plan_keeps_one_structured_naturalness_correction(self):
        parsed = ConversationDirector.parse_plan(
            '{"response_need":"roleplay_turn","language_move":"natural_recast",'
            '"correction_needed":true,"correction_type":"word_order",'
            '"learner_excerpt":"how much I should pay for one day",'
            '"corrected_expression":"How much should I pay per day?",'
            '"correction_note_zh":"疑问句中 should 放在主语前",'
            '"correction_priority":3}'
        )
        self.assertTrue(parsed["correction_needed"])
        self.assertEqual(parsed["correction_type"], "word_order")
        self.assertEqual(parsed["corrected_expression"], "How much should I pay per day?")
        diagnostic = TutorAgent._director_correction_diagnostic(
            "So how much I should pay for one day?", parsed
        )
        self.assertEqual(diagnostic["source"], "conversation_director")
        self.assertFalse(diagnostic["acoustic_evidence"])

    def test_invalid_or_chinese_only_director_correction_is_not_used(self):
        plan = ConversationDirector.parse_plan(
            '{"correction_needed":true,"correction_type":"pronunciation",'
            '"learner_excerpt":"租车","corrected_expression":"rent a car"}'
        )
        self.assertFalse(plan["correction_needed"])
        self.assertIsNone(TutorAgent._director_correction_diagnostic("租车怎么说", plan))

    def test_fallback_detects_real_world_need_without_router_label(self):
        plan = ConversationDirector.fallback_plan(
            user_text="我没看到行李，不知道应该去哪里处理，也不知道怎么跟工作人员说。",
            learner_intent="open_expression",
            language_mode="chinese_support",
            scene={},
        )
        self.assertEqual(plan["response_need"], "real_world_answer")
        self.assertFalse(plan["user_should_speak_now"])

    def test_director_replans_only_at_a_turn_boundary(self):
        existing = {"response_need": "roleplay_turn"}
        self.assertFalse(ConversationDirector.should_replan(
            user_text="Yes, I have.", learner_intent="open_expression",
            scene_changed=False, existing_plan=existing,
        ))
        self.assertTrue(ConversationDirector.should_replan(
            user_text="现实中下一步怎么办？", learner_intent="real_world_context",
            scene_changed=False, existing_plan=existing,
        ))
        self.assertFalse(ConversationDirector.should_replan(
            user_text="   ", learner_intent="open_expression",
            scene_changed=False, existing_plan=existing,
        ))

    def test_roleplay_allows_english_only_when_meaning_was_not_requested(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "Check-out is at noon.", prepared, "What time should I check out?",
        )
        self.assertFalse(any("完整中英支撑" in item for item in violations))

    def test_phrase_stream_waits_for_complete_bilingual_unit(self):
        units, tail = TutorAgent._take_stream_units(
            "Please show your boarding pass.", require_bilingual=True,
        )
        self.assertEqual(units, [])
        units, tail = TutorAgent._take_stream_units(
            tail + " 请出示登机牌。下一句", require_bilingual=True,
        )
        self.assertEqual(units, ["Please show your boarding pass. 请出示登机牌。"])
        self.assertEqual(tail, "下一句")

    def test_phrase_stream_flushes_final_tail(self):
        units, tail = TutorAgent._take_stream_units(
            "Could you help me rebook", final=True,
        )
        self.assertEqual(units, ["Could you help me rebook"])
        self.assertEqual(tail, "")

    def test_phrase_stream_does_not_emit_half_quoted_example(self):
        units, tail = TutorAgent._take_stream_units(
            "还可以问“我的房间是哪间？", require_bilingual=True,
        )
        self.assertEqual(units, [])
        units, tail = TutorAgent._take_stream_units(
            tail + "”，英文是“Which room is mine?”", require_bilingual=True, final=True,
        )
        self.assertEqual(
            units, ["还可以问“我的房间是哪间？”，英文是“Which room is mine?”"]
        )
        self.assertEqual(tail, "")

    def test_adaptive_web_search_router_uses_specific_current_facts(self):
        self.assertTrue(ConversationDirector.needs_web_search(
            "我下个月在加州通过 Hertz 租车，当前通常会核对哪些材料？",
            {"goal": "完成租车取车"},
        ))
        self.assertFalse(ConversationDirector.needs_web_search(
            "我们练习一下在餐厅进门后怎么和迎宾员说。",
            {"goal": "普通餐厅场景演绎"},
        ))
        self.assertFalse(ConversationDirector.needs_web_search(
            "OK, thank you very much.",
            {"goal": "处理机场转机", "brief": "前序航班延误，正在机场问路"},
        ))
        self.assertTrue(ConversationDirector.needs_web_search(
            "有道 AI 耳机现在多少钱？和 AirPods 比价格呢？", {}
        ))
        self.assertTrue(ConversationDirector.needs_web_search(
            "How much do these earbuds cost compared with AirPods?", {}
        ))
        self.assertFalse(ConversationDirector.needs_web_search(
            "That is surprising.",
            {"goal": "Compare the price of earbuds and AirPods"},
        ))

    def test_thank_you_uses_zero_latency_local_roleplay_reply(self):
        plan = ConversationDirector._local_fast_plan(
            user_text="OK, thank you very much.",
            language_mode="mixed_support",
            scene={"scene": "open_roleplay", "goal": "处理机场转机"},
            existing_plan={
                "counterpart_role": "机场工作人员",
                "current_phase": "指引转机柜台",
            },
        )
        self.assertEqual(plan["reply_draft"], "You’re welcome.")
        self.assertEqual(plan["_meta"]["source"], "local_fastpath")
        self.assertEqual(plan["_meta"]["latency_ms"], 0)

    def test_no_speak_plan_blocks_premature_drill(self):
        prepared = {
            "learning_move": {"action": "connect_first_turn"},
            "constraints": {
                "language_mode": "chinese_support",
                "conversation_director": {
                    "content_need": "先了解维修渠道",
                    "user_should_speak_now": False,
                },
            },
        }
        reply = "repair 是维修，现在你试着说 Where is the repair shop?"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "我想先知道国外有什么维修渠道。"
        )
        self.assertTrue(any("对话导演" in item for item in violations))
        fixed = TutorAgent._enforce_controlled_reply(
            reply, prepared,
            SessionContext(session_id="s", user_id="u", target_lang="en"),
            "我想先知道国外有什么维修渠道。",
        )
        self.assertIn("先处理", fixed)
        self.assertNotIn("试着说", fixed)

    def test_director_overrides_stale_learning_move_and_scene_phase(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.learning_state = LearningState(
            scenario="open_roleplay", tutor_role="路人", scenario_step="重新问路",
        )
        ctx.conversation_plan = {
            "response_need": "counterpart_turn",
            "content_need": "维修店工作人员先回答用户刚才的故障描述",
            "next_counterpart_action": "工作人员询问手机型号和是否换过充电线",
            "user_should_speak_now": False,
            "scaffold_level": 1,
            "learner_role": "顾客",
            "counterpart_role": "维修店工作人员",
            "current_phase": "故障初检",
            "scene_brief": "顾客已到维修店并说明手机无法充电",
        }
        move = LearningMove(
            stage="flow", action="prompt_output", goal="重复旧句",
            user_must_speak=True,
        )
        TutorAgent._apply_conversation_plan(ctx, move)
        self.assertEqual(move.action, "supply_counterpart_turn")
        self.assertFalse(move.user_must_speak)
        self.assertEqual(ctx.scenario_role, "维修店工作人员")
        self.assertEqual(ctx.scenario_step, "故障初检")

    def test_fragmented_bilingual_and_invented_location_are_hard_violations(self):
        prepared = {
            "learning_move": {"action": "acknowledge_and_continue"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "known_facts": ["用户在国外，手机无法充电"],
                    "missing_information": ["用户当前所在地点"],
                    "user_should_speak_now": False,
                },
            },
        }
        reply = "附近转角就有一家phone shop，我带你过去，很easy。"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "我的手机充不进电，不知道去哪里维修。"
        )
        self.assertTrue(any("单个英文词" in item for item in violations))
        self.assertTrue(any("虚构" in item for item in violations))

    def test_handed_document_is_persisted_and_cannot_be_requested_again(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [{
            "role": "assistant", "content": "Please give me your baggage tag. 请把行李牌给我。",
        }]
        agent = TutorAgent.__new__(TutorAgent)
        agent._update_scene_ledger(ctx, "OK, here.", {
            "current_phase": "核对材料",
            "known_facts": [],
            "completed_actions": [],
            "available_items": [],
            "open_items": [],
        })
        self.assertIn("行李牌", ctx.scene_ledger["available_items"])
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_episode_ledger": ctx.scene_ledger,
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "known_facts": [],
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "What is your baggage tag number? 请告诉我行李牌号码。",
            prepared,
            "OK, here.",
        )
        self.assertTrue(any("已经交给" in item for item in violations))

    def test_roleplay_turn_requires_target_language_and_blocks_invented_attributes(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_episode_ledger": {"known_facts": ["用户已出示行李牌"]},
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "known_facts": ["用户已出示行李牌"],
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "你的行李牌是蓝色的，我已经看到了。",
            prepared,
            "你确认一下。",
        )
        self.assertTrue(any("完整目标语" in item for item in violations))
        self.assertTrue(any("蓝色" in item for item in violations))

    def test_operational_feedback_drops_fixed_agreement_prefix(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {"language_mode": "chinese_support"},
        }
        violations = TutorAgent._controlled_reply_violations(
            "你说得对，我直接看就好。",
            prepared,
            "你自己看不是更快吗？",
        )
        self.assertTrue(any("附和前缀" in item for item in violations))
        english_violations = TutorAgent._controlled_reply_violations(
            "Oh, you're right! I should give you both papers together.",
            prepared,
            "Why didn't you give me both papers together?",
        )
        self.assertTrue(any("附和前缀" in item for item in english_violations))

        subtle = TutorAgent._controlled_reply_violations(
            "好的，我明白了。I will continue with the bill check.",
            prepared,
            "不用每次先附和我，直接继续核对账单。",
        )
        self.assertTrue(any("直接把修正落实" in item for item in subtle))

        cleaned = TutorAgent._strip_sycophantic_prefix(
            "好的，我明白了。I will continue with the bill check.",
            "不用每次先附和我，直接继续核对账单。",
        )
        self.assertEqual(cleaned, "I will continue with the bill check.")

        role_confirmation = TutorAgent._strip_sycophantic_prefix(
            "好的，我来查一下账单。", "OK, here is my room number."
        )
        self.assertEqual(role_confirmation, "好的，我来查一下账单。")

        apology_cleaned = TutorAgent._strip_sycophantic_prefix(
            "Sorry for the confusion. The convenience store is near the hotel.",
            "你前面不是说在酒店附近吗？现在为什么又说在一楼？",
        )
        self.assertEqual(apology_cleaned, "The convenience store is near the hotel.")

    def test_bare_acknowledgement_prefix_is_stripped_without_losing_the_body(self):
        """plans/0921-04 P1：`Got it,` 后接**非回音壁正文**时也要剥掉附和前缀。

        旧实现只在后接 "so you want…/this is…"（回音壁）时才剥，于是
        "Got it, let me check…" / "Got it! Let's start." / "Got it, that's how…"
        三类原样出货（真机 10 轮实测 4 轮以 "Got it," 开头）。
        """
        # 三类实测漏网形态：剥掉附和前缀，正文一字不动
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "Got it, let me check for your order.", "I booked it yesterday."
            ),
            "let me check for your order.",
        )
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "Got it! Let's start.", "Can we just begin?"
            ),
            "Let's start.",
        )
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "Got it, that's how it works here.", "So the machine does it itself?"
            ),
            "that's how it works here.",
        )
        # 同族开头（i see / understood）同样适用，含句点分隔
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "I see. So the price is 500.", "Is it expensive?"
            ),
            "So the price is 500.",
        )
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "Understood, I'll note it down.", "Please remember this."
            ),
            "I'll note it down.",
        )

        # ⚠ 退化保护 1：整句都是附和（无可保留正文）→ 必须原样返回，
        # 交给违规/修复链路处理（docstring 既有契约）。
        for bare in ("Got it.", "Got it", "Understood.", "I see."):
            self.assertEqual(TutorAgent._strip_sycophantic_prefix(bare), bare)

        # ⚠ 退化保护 2：回音壁整句仍由前一条模式吃掉整句 → 同样原样返回，
        # 不得被降级成"只剩复述问句"（P1 排在它之后正是为此）。
        echo = "Got it, so you want a coffee?"
        self.assertEqual(TutorAgent._strip_sycophantic_prefix(echo), echo)

        # ⚠ 退化保护 3：回音壁后**还有正文**时，仍然只留正文（既有行为不变）
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "Got it, so you want a coffee? Here's the menu."
            ),
            "Here's the menu.",
        )

        # ⚠ 退化保护 4：中文角色台词里的"好的，…"不属于本支（用户转写非过程反馈时）
        self.assertEqual(
            TutorAgent._strip_sycophantic_prefix(
                "好的，我来查一下账单。", "OK, here is my room number."
            ),
            "好的，我来查一下账单。",
        )

    def test_midnight_correction_and_scene_location_are_hard_contracts(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "recent_dialogue": [{
                    "role": "assistant",
                    "content": "There's a convenience store near the hotel. 酒店附近有一家便利店。",
                }],
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                },
            },
        }
        time_violations = TutorAgent._controlled_reply_violations(
            "You meant midnight. 你指的是午夜。",
            prepared,
            "我说的是晚上十二点，twelve pm 是不是应该换个说法？",
        )
        self.assertTrue(any("午夜时间" in item for item in time_violations))

        correct_time = TutorAgent._controlled_reply_violations(
            "12 p.m. is noon; midnight is 12 a.m. 12 p.m. 是中午，午夜是 12 a.m.。",
            prepared,
            "我说的是晚上十二点，twelve pm 是不是应该换个说法？",
        )
        self.assertFalse(any("午夜时间" in item for item in correct_time))

        location_violations = TutorAgent._controlled_reply_violations(
            "The convenience store is on the first floor of the hotel. 便利店在酒店一楼。",
            prepared,
            "我想买一根充电线。",
        )
        self.assertTrue(any("位置与最近回复冲突" in item for item in location_violations))

    def test_confirmed_object_is_not_confirmed_again(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "recent_dialogue": [{
                    "role": "assistant",
                    "content": "Are you asking about the convenience store near the hotel?",
                }],
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "你就是想确认那家便利店深夜还开着，对吧？",
            prepared,
            "不然呢？",
        )
        self.assertTrue(any("禁止再次复述" in item for item in violations))

    def test_operational_audio_feedback_is_not_soothed_or_apologized_to(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {"language_mode": "mixed_support"},
        }
        user_text = "我都没有语音输入，为什么音频自己打断了？"
        violations = TutorAgent._controlled_reply_violations(
            "别着急，抱歉刚才出现了音频故障。",
            prepared,
            user_text,
        )
        self.assertTrue(TutorAgent._is_operational_feedback(user_text))
        self.assertTrue(any("机械道歉" in item for item in violations))
        self.assertTrue(any("不要安抚" in item for item in violations))

    def test_explicit_expression_check_requires_a_complete_target_sentence(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_with_recast"},
            "constraints": {"language_mode": "mixed_support"},
        }
        violations = TutorAgent._controlled_reply_violations(
            "charger 是充电器。",
            prepared,
            "I want to ask the worker have charging，这样说对吗？",
        )
        self.assertTrue(any("完整自然的目标语表达" in item for item in violations))

        good = TutorAgent._controlled_reply_violations(
            '更自然地说："Excuse me, do you have a charging cable?" 这里用 do you have 来询问店里是否有货。',
            prepared,
            "I want to ask the worker have charging，这样说对吗？",
        )
        self.assertFalse(any("完整自然的目标语表达" in item for item in good))

    def test_roleplay_language_ladder_uses_evidence_not_latest_chinese_alone(self):
        ctx = SessionContext(
            session_id="s", user_id="u", target_lang="en", profile={"cefr": "B1"},
        )
        ctx.messages = [
            {"role": "user", "content": "My flight number is 6654."},
            {"role": "assistant", "content": "Thank you."},
        ]
        plan = {"response_need": "roleplay_turn", "language_presentation": "chinese_only"}
        TutorAgent._apply_roleplay_language_ladder(ctx, "那接下来呢？", None, plan)
        self.assertEqual(plan["language_presentation"], "target_only")

        ctx.messages.extend([
            {"role": "user", "content": "Here is my baggage tag."},
            {"role": "assistant", "content": "Thank you."},
        ])
        plan = {"response_need": "roleplay_turn"}
        TutorAgent._apply_roleplay_language_ladder(ctx, "What happens next?", None, plan)
        self.assertEqual(plan["language_presentation"], "target_only")

        ctx.profile["cefr"] = "A2"
        plan = {"response_need": "roleplay_turn"}
        TutorAgent._apply_roleplay_language_ladder(ctx, "What time should I check out?", None, plan)
        self.assertEqual(plan["language_presentation"], "target_only")

    def test_successful_director_is_not_overwritten_by_legacy_scene_catalogue(self):
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "_meta": {"source": "director_model"},
                    "reply_draft": "在产品展示里，功能可以说 feature。",
                    "known_facts": ["用户问产品语境里的功能怎么说"],
                    "user_should_speak_now": False,
                },
            },
        }
        reply = "在这个产品语境里，功能可以说 feature（功能）。"
        fixed = TutorAgent._enforce_controlled_reply(
            reply, prepared,
            SessionContext(session_id="s", user_id="u", target_lang="en"),
            "产品的功能怎么说？",
        )
        self.assertEqual(fixed, reply)

    def test_failed_director_reuses_scene_contract_but_never_old_reply(self):
        fallback = ConversationDirector.fallback_plan(
            user_text="继续处理我的二维码问题。",
            learner_intent="open_expression",
            language_mode="mixed_support",
            scene={"scene": "open_roleplay"},
        )
        recovered = ConversationDirector._recover_existing_plan(
            {
                "reply_draft": "上一轮已经播过的回复",
                "counterpart_role": "票务工作人员",
                "current_phase": "核验订单",
                "current_location": "票务服务台",
                "completed_actions": ["用户已出示确认邮件"],
            },
            fallback,
            {"source": "fallback_error", "failure_code": "insufficient_balance"},
        )
        self.assertEqual(recovered["counterpart_role"], "票务工作人员")
        self.assertEqual(recovered["current_phase"], "核验订单")
        self.assertEqual(recovered["reply_draft"], "")
        self.assertIn("用户已出示确认邮件", recovered["completed_actions"])
        self.assertEqual(recovered["_meta"]["failure_code"], "insufficient_balance")

    def test_failed_director_uses_independent_tutor_provider(self):
        # The final learner-facing reply is never written by the strong director
        # provider any more: live voice logs showed the routed group
        # (director=director_model, n=67) at ~8.4s backendMs versus ~4.5s for the
        # un-routed group (n=20), because routing serializes the director
        # planning call and the reply-writing call onto the same provider.  The
        # response route is therefore unconditionally TUTOR_RESPONSE_*.
        for director_source in ("director_model", "cached_director_model",
                                "fallback_error", "local_fastpath", ""):
            with self.subTest(director_source=director_source):
                self.assertFalse(TutorAgent._should_use_director_provider(
                    director_source, enabled=True, api_key="configured",
                ))
        # A configured and enabled director must not re-enable the route either.
        self.assertFalse(TutorAgent._should_use_director_provider(
            "director_model", enabled=True, api_key="",
        ))

    def test_unverified_ticket_outcome_promises_are_blocked(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "_meta": {"source": "director_model"},
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "known_facts": ["二维码无法显示"],
                },
            },
        }
        reply = "They will give you a paper ticket and let you in. 他们会给你纸质票并让你入场。"
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "Ticketmaster账户是我朋友的，会影响核验吗？",
        )
        self.assertTrue(any("无证据承诺" in item for item in violations))
        safe = TutorAgent._enforce_controlled_reply(
            reply, prepared,
            SessionContext(session_id="s", user_id="u", target_lang="en"),
            "Ticketmaster账户是我朋友的，会影响核验吗？",
        )
        self.assertIn("verify", safe)
        self.assertNotIn("paper ticket", safe.lower())

    def test_episode_ledger_blocks_return_to_an_old_location(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scene_ledger = {
            "current_phase": "已经入场",
            "phase_history": ["票务核验", "已经入场"],
            "current_location": "场馆内部",
            "location_history": ["票务服务台", "入口", "场馆内部"],
            "completed_actions": ["用户已经出示票并进入场馆"],
        }
        agent = TutorAgent.__new__(TutorAgent)
        rollback_plan = {
            "current_phase": "票务核验",
            "current_location": "票务服务台",
            "next_counterpart_action": "重新打印纸质票",
        }
        agent._update_scene_ledger(ctx, "我们已经在里面了。", rollback_plan)
        self.assertEqual(ctx.scene_ledger["current_phase"], "已经入场")
        self.assertEqual(ctx.scene_ledger["current_location"], "场馆内部")
        self.assertEqual(len(ctx.scene_ledger["blocked_regressions"]), 2)
        self.assertEqual(rollback_plan["current_phase"], "已经入场")
        self.assertEqual(rollback_plan["current_location"], "场馆内部")

    def test_director_roleplay_correction_is_not_overwritten(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.conversation_plan = {
            "response_need": "roleplay_turn",
            "correction_needed": True,
            "corrected_expression": "I didn't use the minibar.",
            "content_need": "前台继续核对迷你吧争议费用",
        }
        move = LearningMove(
            stage="flow", action="continue_roleplay_scene",
            goal="继续场景", user_must_speak=True,
        )
        TutorAgent._apply_conversation_plan(ctx, move)
        self.assertEqual(move.action, "continue_roleplay_with_recast")
        self.assertEqual(move.feedback_mode, "one_brief_upgrade_in_role")

    def test_hotel_case_blocks_invented_amount_refund_and_deadline(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_episode_ledger": {
                    "known_facts": ["学习者否认使用迷你吧"],
                },
                "conversation_director": {
                    "_meta": {"source": "director_model"},
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "content_need": "核对酒店账单里的迷你吧费用",
                    "known_facts": ["学习者否认使用迷你吧"],
                },
            },
        }
        reply = "I found a $15 charge. We will give you money back and reply within one hour."
        violations = TutorAgent._controlled_reply_violations(
            reply, prepared, "I didn't use the minibar."
        )
        self.assertTrue(any("无证据承诺" in item for item in violations))
        self.assertTrue(any("凭空生成账单" in item for item in violations))
        safe = TutorAgent._enforce_controlled_reply(
            reply, prepared,
            SessionContext(session_id="s", user_id="u", target_lang="en"),
            "I didn't use the minibar.",
        )
        self.assertIn("before I can confirm a refund", safe)
        self.assertNotIn("$15", safe)

    def test_general_chat_blocks_fake_personal_agreement_and_overstatement(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "_meta": {"source": "director_model"},
                    "response_need": "general_chat",
                    "interaction_frame": "general",
                },
            },
        }
        reply = "That's the real charm of CS2. That's exactly how I feel too."
        violations = TutorAgent._controlled_reply_violations(
            reply,
            prepared,
            "不一定是真正迷人的地方，只是某个瞬间让我觉得值得看。",
        )
        self.assertTrue(any("亲身观看" in item for item in violations))
        self.assertTrue(any("原强度" in item for item in violations))

    def test_general_chat_correction_contract_must_surface_selected_recast(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "_meta": {"source": "director_model"},
                    "response_need": "general_chat",
                    "interaction_frame": "general",
                    "correction_needed": True,
                    "corrected_expression": "It's an interesting question.",
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "That question is difficult. 这个问题有点难。",
            prepared,
            "It's a interesting and difficult question.",
        )
        self.assertTrue(any("高价值纠错名额" in item for item in violations))

    def test_episode_ledger_remembers_answers_and_blocks_reasking(self):
        agent = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.messages = [{
            "role": "assistant", "content": "Your name, please? （您的姓名？）",
        }]
        agent._update_scene_ledger(ctx, "Mr Zhang, 张三。", {
            "known_facts": [], "completed_actions": [],
            "available_items": [], "open_items": [],
        })
        ctx.messages.extend([
            {"role": "user", "content": "Mr Zhang, 张三。"},
            {"role": "assistant", "content": "Thank you, Mr Zhang. What is your room number?"},
        ])
        agent._update_scene_ledger(ctx, "twenty zero seven", {
            "known_facts": [], "completed_actions": [],
            "available_items": [], "open_items": [],
        })
        ctx.messages.extend([
            {"role": "user", "content": "twenty zero seven"},
            {"role": "assistant", "content": "Did you use the minibar?"},
        ])
        agent._update_scene_ledger(ctx, "No, I'm not used the minibar.", {
            "known_facts": [], "completed_actions": [],
            "available_items": [], "open_items": [],
        })
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_episode_ledger": ctx.scene_ledger,
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "known_facts": [],
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "Mr Zhang, what is your room number? 您的房间号是多少？",
            prepared, "And then?",
        )
        self.assertTrue(any("重复索取" in item for item in violations))
        self.assertTrue(any("重复称呼" in item for item in violations))
        confirm = TutorAgent._controlled_reply_violations(
            "Did you mean room 2007? 您是说 2007 房吗？",
            prepared, "twenty zero seven",
        )
        self.assertFalse(any("重复索取" in item for item in confirm))

    def test_process_branch_requires_a_concrete_next_step(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "scene_episode_ledger": {"known_facts": ["酒店迷你吧费用争议"]},
                "conversation_director": {
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "known_facts": ["酒店迷你吧费用争议"],
                },
            },
        }
        vague = TutorAgent._controlled_reply_violations(
            "If there is a problem, we will check. You can tell us your situation.",
            prepared,
            "第二种情况如果要核实，下一步需要提供哪些信息？",
        )
        self.assertTrue(any("具体处理分支" in item for item in vague))
        concrete = TutorAgent._controlled_reply_violations(
            "I will ask housekeeping to check its minibar record first. 客房部会先核对迷你吧记录。",
            prepared,
            "第二种情况如果要核实，下一步需要提供哪些信息？",
        )
        self.assertFalse(any("具体处理分支" in item for item in concrete))

    def test_explicit_meaning_question_requires_chinese_support(self):
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {"language_mode": "target_language"},
        }
        violations = TutorAgent._controlled_reply_violations(
            "Reply within one hour.", prepared, "reply within 是啥意思？",
        )
        self.assertTrue(any("中文释义" in item for item in violations))

    def test_language_sidecar_answers_each_explicit_language_question(self):
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "interaction_frame": "language_sidecar",
                    "response_need": "expression_support",
                    "language_presentation": "chinese_then_target",
                },
            },
        }
        user = (
            "hotel entrance 是酒店出口的意思吗？"
            "So can I say excuse me does have the charge cable 这样问店员可以吗？"
        )
        incomplete = TutorAgent._controlled_reply_violations(
            "Excuse me, do you have a charging cable? 打扰一下，你们有充电线吗？",
            prepared, user,
        )
        self.assertTrue(any("单词或短语含义" in item for item in incomplete))
        complete = TutorAgent._controlled_reply_violations(
            "hotel entrance 是酒店入口或门口，不是出口；出口是 hotel exit。"
            "问店员可以说：Excuse me, do you have a charging cable?",
            prepared, user,
        )
        self.assertFalse(any("单词或短语含义" in item for item in complete))
        self.assertFalse(any("完整自然" in item for item in complete))

    def test_listening_frame_stays_brief_and_does_not_teach(self):
        prepared = {
            "learning_move": {"action": "method_repair"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "interaction_frame": "listening",
                    "response_need": "listen_and_continue",
                    "language_presentation": "chinese_only",
                },
            },
        }
        pushy = TutorAgent._controlled_reply_violations(
            "Try saying this: Excuse me, do you have a charging cable?",
            prepared, "我先想一下，我还没说完。",
        )
        self.assertTrue(any("在听回应" in item for item in pushy))
        natural = TutorAgent._controlled_reply_violations(
            "嗯，我听着，你继续。", prepared, "我先想一下，我还没说完。",
        )
        self.assertFalse(any("在听回应" in item for item in natural))

    def test_unresolved_action_contract_blocks_premature_expression(self):
        prepared = {
            "learning_move": {"action": "clarify_lexical_then_resume"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "interaction_frame": "language_sidecar",
                    "speaker_position": "coach",
                    "advance_scene": False,
                    "uncertain_spans": ["用户尚未确认的对象"],
                    "unresolved_slots": ["购买数量"],
                    "reality_basis": "clarification_needed",
                    "pragmatic_check": "clarify",
                },
            },
        }
        bad = TutorAgent._controlled_reply_violations(
            '你可以说：“I need three.”', prepared, "我还没确定具体信息。",
        )
        self.assertTrue(any("行动契约仍有未确认信息" in item for item in bad))
        good = TutorAgent._controlled_reply_violations(
            "这个信息还没确定。你是想先确定具体信息，还是先问是否有售？",
            prepared, "我还没确定具体信息。",
        )
        self.assertFalse(any("行动契约仍有未确认信息" in item for item in good))

    def test_counterpart_position_rejects_coach_instructions(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "conversation_director": {
                    "interaction_frame": "in_role",
                    "speaker_position": "counterpart",
                    "advance_scene": True,
                    "response_need": "roleplay_turn",
                    "language_presentation": "target_then_chinese",
                    "pragmatic_check": "pass",
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "你可以这样问店员：Do you have this?",
            prepared, "I need some help.",
        )
        self.assertTrue(any("对方角色说话" in item for item in violations))

        indirect_coaching = TutorAgent._controlled_reply_violations(
            "你走到服务台，对工作人员说：Hi, I got the wrong date.",
            prepared,
            "你来当工作人员，我们直接开始。",
        )
        self.assertTrue(any("对方角色说话" in item for item in indirect_coaching))

    def test_same_session_recap_must_cover_language_sidecar_focus(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.event_log = [
            {"role": "user", "content": "Hi, I bought a ticket for the wrong day."},
            {"role": "assistant", "content": "Let me see your ticket, please."},
            {"role": "user", "content": "change 和 exchange 在这个场景里有什么区别？"},
            {"role": "assistant", "content": "Use change for the date."},
            {"role": "user", "content": "Bot 在这里是过去式对吧？"},
        ]
        decision = {
            "acts": ["session_recap"],
            "primary_frame": "process_sidecar",
            "speaker_position": "coach",
            "advance_scene": False,
        }
        same_session = TutorAgent._same_session_context_for_turn(
            ctx, "我们刚才解决了什么？", decision
        )
        self.assertIn("change", same_session["recap_focus_terms"])
        self.assertIn("exchange", same_session["recap_focus_terms"])
        prepared = {
            "learning_move": {"action": "session_reflection"},
            "constraints": {
                "language_mode": "mixed_support",
                "turn_contract": {
                    "decision": decision,
                    "same_session_context": same_session,
                },
            },
        }
        incomplete = TutorAgent._controlled_reply_violations(
            "我们确认了车票日期，现在等你出示车票。",
            prepared,
            "我们刚才解决了什么？",
        )
        self.assertTrue(any("语言侧问焦点" in item for item in incomplete))
        complete = TutorAgent._controlled_reply_violations(
            "我们确认了 change 和 exchange 的区别，也问了 bought 的过去式用法。",
            prepared,
            "我们刚才解决了什么？",
        )
        self.assertFalse(any("语言侧问焦点" in item for item in complete))

    def test_recap_uses_latest_scene_epoch_and_drops_rejected_asr_token(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.event_log = [
            {"role": "user", "content": "change 和 exchange 有什么区别？"},
            {"role": "assistant", "content": "Use change here."},
            {"role": "user", "content": "我在网上订了一辆车。你来当工作人员，我们直接开始。"},
            {"role": "assistant", "content": "Can I see your booking?"},
            {"role": "user", "content": "I want to borrow a car. 这里用 blue 还是 rent？"},
            {"role": "assistant", "content": "Do you want a blue car?"},
            {"role": "user", "content": "我说的是 borrow，不是 blue。"},
        ]
        decision = {"acts": ["session_recap"]}
        same_session = TutorAgent._same_session_context_for_turn(
            ctx, "我们刚才处理了什么？", decision
        )
        evidence = " ".join(item["content"] for item in same_session["messages"])
        self.assertNotIn("change 和 exchange", evidence)
        self.assertIn("borrow", same_session["recap_focus_terms"])
        self.assertIn("rent", same_session["recap_focus_terms"])
        self.assertNotIn("blue", same_session["recap_focus_terms"])
        prepared = {
            "learning_move": {"action": "session_reflection"},
            "constraints": {
                "turn_contract": {
                    "decision": {"acts": ["session_recap"]},
                    "same_session_context": same_session,
                },
            },
        }
        grounded = TutorAgent._enforce_controlled_reply(
            "之前问了 change 和 exchange，还有 got 和 bot。",
            prepared,
            ctx,
            user_transcript="我们刚才处理了什么？",
        )
        self.assertIn("borrow", grounded)
        self.assertIn("rent", grounded)
        self.assertNotIn("change", grounded)

    def test_rehydrate_scene_state_prefers_latest_explicit_scene(self):
        tutor = TutorAgent.__new__(TutorAgent)
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        history = [
            {"role": "user", "content": "我在酒店办理入住，你来当前台，我们直接开始。"},
            {"role": "assistant", "content": "Welcome to our hotel."},
            {"role": "user", "content": "我在国外租车柜台，你来当工作人员，我们直接开始。"},
        ]
        tutor._rehydrate_scene_state(ctx, history)
        self.assertIn("租车柜台", ctx.scenario_brief)
        self.assertEqual(ctx.scenario_role, "工作人员")

    def test_general_interest_reply_keeps_complete_english_without_forcing_question(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "turn_contract": {
                    "decision": {"acts": ["general_conversation"], "primary_frame": "general"},
                },
            },
        }
        missing = TutorAgent._controlled_reply_violations(
            "祖国人是《黑袍纠察队》里的角色，英文名是 Homelander。",
            prepared,
            "你知道祖国人是哪部剧的角色吗？",
        )
        self.assertTrue(any("完整" in item and "英文" in item for item in missing))
        natural = TutorAgent._controlled_reply_violations(
            "祖国人是《黑袍纠察队》里的角色。Homelander is one of the main characters.",
            prepared,
            "你知道祖国人是哪部剧的角色吗？",
        )
        self.assertFalse(any("完整" in item and "英文" in item for item in natural))

    def test_general_chat_correction_wrapper_cannot_repeat_on_adjacent_turns(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "correction_budget": 1,
                "turn_contract": {
                    "decision": {"acts": ["general_conversation"], "primary_frame": "general"},
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "顺便说，I like him a lot 这样说更自然。What do you think of him?",
            prepared,
            "I actually much like him.",
            previous_reply="顺便说，I've only seen some clips 这样说更自然。Do you like the show?",
        )
        self.assertTrue(any("纠错开头" in item for item in violations))

    def test_general_chat_cooldown_blocks_unscheduled_correction(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "correction_budget": 0,
                "turn_contract": {
                    "decision": {"acts": ["general_conversation"], "primary_frame": "general"},
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "这段确实很有冲击力。顺便说，the dialogue is wild 这样说更自然。What did you like?",
            prepared,
            "The conversation is wild.",
        )
        self.assertTrue(any("纠错冷却期" in item for item in violations))

    def test_unsolicited_specific_plot_outcome_is_rejected(self):
        prepared = {
            "learning_move": {"action": "engage_general_topic"},
            "constraints": {
                "correction_budget": 0,
                "turn_contract": {
                    "decision": {"acts": ["general_conversation"], "primary_frame": "general"},
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "Homelander 在第四季结尾死了。What do you think of him?",
            prepared,
            "I haven't watched it.",
        )
        self.assertTrue(any("剧透" in item for item in violations))

    def test_spoken_identifier_must_be_confirmed_before_normalizing(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {"language_mode": "mixed_support"},
        }
        violations = TutorAgent._controlled_reply_violations(
            "Room 207. You have a minibar charge. 房间是 207。",
            prepared, "twenty zero seven",
        )
        self.assertTrue(any("数字表达存在歧义" in item for item in violations))
        confirmed = TutorAgent._controlled_reply_violations(
            "Did you mean room 2007? 您是说 2007 房吗？",
            prepared, "twenty zero seven",
        )
        self.assertFalse(any("数字表达存在歧义" in item for item in confirmed))

    def test_available_cross_session_recall_cannot_be_denied(self):
        prepared = {
            "learning_move": {"action": "continue_roleplay_scene"},
            "constraints": {
                "language_mode": "mixed_support",
                "session_memory": {
                    "route": {"mode": "episodic_recall"},
                    "recall": [{
                        "session_id": "hotel-key_A",
                        "summary": "当前步骤=前台核验预订；待处理=借充电器",
                    }],
                },
            },
        }
        denied = TutorAgent._controlled_reply_violations(
            "我没有保存上一盘的具体细节，我们重新开始吧。",
            prepared,
            "继续上一盘没有处理完的房卡和充电器，刚才到哪里？",
        )
        self.assertTrue(any("已经检索到" in item for item in denied))
        precise = TutorAgent._controlled_reply_violations(
            "上一盘进行到前台核验预订信息，房卡还在处理，借充电器还没有处理。",
            prepared,
            "继续上一盘没有处理完的房卡和充电器，刚才到哪里？",
        )
        self.assertFalse(any("已经检索到" in item for item in precise))

    def test_language_sidecar_cannot_replace_scene_checkpoint(self):
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.scenario_brief = "酒店前台处理房卡打不开门和借充电器"
        ctx.scenario_step = "前台核验预订信息"
        ctx.conversation_plan = {
            "interaction_frame": "language_sidecar",
            "response_need": "expression_support",
            "scene_brief": "用户问 almost dead 是什么意思",
            "current_phase": "解释短语",
            "content_need": "解释 almost dead",
        }
        move = LearningMove(
            stage="practice", action="clarify_lexical_then_resume",
            goal="解释短语", prompt_hint="", feedback_mode="meaning_first",
        )
        TutorAgent._apply_conversation_plan(ctx, move)
        self.assertEqual(ctx.scenario_brief, "酒店前台处理房卡打不开门和借充电器")
        self.assertEqual(ctx.scenario_step, "前台核验预订信息")


class ExpressionArgumentProbeTests(unittest.TestCase):
    """练习盘五维 probe（plans/0923-06 §4.2.3）。

    三个硬约束在这里被钉住：
      1. **只在练习盘**追加字段与 system 规则——非练习盘的 plan 对象与 system
         逐字不变（别的盘的 prompt 快照可对拍）；
      2. probe 的**证据必须是本轮原话的逐字子串**，伪造引文整条丢弃、不补第二次调用；
      3. 失败兜底（`_recover_existing_plan`）**不得把上一轮的 probe 带回这一轮**。
    """

    TRANSCRIPT = (
        "I think working from home is better for me. "
        "Last month I saved two hours a day because I did not commute. "
        "So it is clearly better for everyone, always."
    )

    def _plan(self, *, user_text, scene, probe, sentinel=None):
        class FakeClient:
            def __init__(self):
                self.calls = []

            async def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                payload = {
                    "response_need": "listen_and_continue",
                    "language_move": "no_push",
                    "reply_draft": "That is an interesting point.",
                }
                if probe is not None:
                    payload["argument_probe"] = probe
                # 非工具路径（response_format=json_object，无 return_message）
                # 真实客户端返回的是**字符串**，不是 message dict。
                return json.dumps(payload, ensure_ascii=False)

        fake = FakeClient()
        director = ConversationDirector()
        with patch.object(config, "CONVERSATION_DIRECTOR_ENABLED", True), \
             patch.object(config, "CONVERSATION_DIRECTOR_API_KEY", "test-key"), \
             patch.object(config, "CONVERSATION_DIRECTOR_WEB_SEARCH_ENABLED", False), \
             patch.object(llm_client, "robust_llm_client", fake):
            plan = asyncio.run(director.plan(
                user_text=user_text, recent_messages=[],
                learner_intent="open_expression", language_mode="target_only",
                cefr="B1", scene=dict(scene), existing_plan=None,
            ))
        return plan, fake

    def test_expression_tape_keeps_a_verbatim_probe(self):
        plan, fake = self._plan(
            user_text=self.TRANSCRIPT,
            scene={"goal": "practice", "cassette_mode": "expression"},
            probe={
                "claim": "ok", "evidence": "weak", "warrant": "na",
                "qualifier": "weak", "rebuttal": "na",
                "focus_key": "qualifier",
                "evidence_quote": "it is clearly better for everyone, always",
            },
        )
        self.assertEqual(plan["_meta"]["source"], "director_model")
        probe = plan["argument_probe"]
        self.assertEqual(probe["weak"], ["evidence", "qualifier"])
        self.assertEqual(probe["ok"], ["claim"])
        self.assertEqual(probe["focus_key"], "qualifier")
        self.assertIn("clearly better", probe["evidence"])
        # 同一批：system 里必须带上五维规则
        system = fake.calls[0]["messages"][0]["content"]
        for dim in ("claim", "evidence", "warrant", "qualifier", "rebuttal"):
            self.assertIn(dim, system)
        self.assertIn("宁缺毋编", system)
        self.assertIn("逐字", system)
        self.assertIn("最终语音草稿直接播报", system)

    def test_expression_probe_quotes_raw_transcript_not_polished_display_text(self):
        raw = "我觉得第六季前两集看着更舒服，没有哪个角色特别让我反感。"
        plan, _ = self._plan(
            user_text="The first two episodes of season six are more enjoyable.",
            scene={
                "goal": "practice", "cassette_mode": "expression",
                "expression_raw_input": raw,
            },
            probe={
                "claim": "ok", "evidence": "weak", "focus_key": "evidence",
                "evidence_quote": "没有哪个角色特别让我反感",
            },
        )
        self.assertEqual(
            plan["argument_probe"]["evidence"], "没有哪个角色特别让我反感"
        )

    def test_expression_direct_draft_must_include_feedback_and_same_topic_retry(self):
        self.assertFalse(TutorAgent._expression_draft_has_feedback(
            "That's a good sign. The writing feels steadier this season."
        ))
        self.assertTrue(TutorAgent._expression_draft_has_feedback(
            "To make that comparison stronger, add one specific example. "
            "Can you restate the same point with that example?"
        ))

    def test_expression_reply_defaults_to_english_without_blocking_short_quote(self):
        prepared = {
            "expression_tape": True,
            "constraints": {
                "turn_contract": {"decision": {"primary_frame": "process_sidecar"}},
            },
        }
        chinese_feedback = "你的观点很清楚，但可以补充一个具体的例子，让理由更有说服力。"
        violations = TutorAgent._controlled_reply_violations(
            chinese_feedback, prepared, "Any suggestions for how I said that?",
        )
        self.assertTrue(any("英语表达环境" in item for item in violations))
        short_quote = (
            "Your point is clear. The phrase “主角让你烦” gives me the key idea, "
            "but one specific action would make the reason stronger."
        )
        violations = TutorAgent._controlled_reply_violations(
            short_quote, prepared, "Any suggestions for how I said that?",
        )
        self.assertFalse(any("英语表达环境" in item for item in violations))

    def test_expression_language_sidecar_may_answer_in_chinese(self):
        prepared = {
            "expression_tape": True,
            "constraints": {
                "turn_contract": {
                    "decision": {"primary_frame": "language_sidecar"},
                    "language_question": {"kind": "term_meaning"},
                },
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            "这个短语的意思是‘让人很烦’。", prepared, "What does get on your nerves mean?",
        )
        self.assertFalse(any("英语表达环境" in item for item in violations))

    def test_expression_reply_cannot_reuse_an_older_recent_turn(self):
        repeated = (
            "I understand. Let's continue with the first two episodes of season six. "
            "The whole watching experience feels better than season five."
        )
        prepared = {
            "expression_tape": True,
            "constraints": {
                "turn_contract": {"decision": {"primary_frame": "general"}},
                "recent_dialogue": [
                    {"role": "assistant", "content": repeated},
                    {"role": "user", "content": "Can you give me feedback?"},
                    {"role": "assistant", "content": "Add one specific action as evidence."},
                ],
            },
        }
        violations = TutorAgent._controlled_reply_violations(
            repeated, prepared,
            "The boss dismisses the warning, and the others mock the person who raised it.",
            previous_reply="Add one specific action as evidence.",
        )
        self.assertTrue(any("近几轮已经说过" in item for item in violations))

    def test_expression_feedback_continues_after_a_process_sidecar_if_user_retells(self):
        from types import SimpleNamespace

        raw = (
            "The boss heard the warning and brushed it off. "
            "Then the other colleagues mocked the person who raised it, "
            "even though the danger turned out to be real."
        )
        ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
        ctx.cassette_mode = "expression"
        ctx.argument_evidence_transcript = raw
        constraints = {
            "turn_contract": {"decision": {"primary_frame": "process_sidecar"}},
            "adaptive_language_support": {},
        }
        agent = TutorAgent.__new__(TutorAgent)
        agent._apply_expression_argument_focus(
            ctx, constraints, SimpleNamespace(correction_budget=1),
            user_transcript=raw, explicit_language_request=False,
        )
        self.assertTrue(constraints["expression_feedback_required"])

    def test_fabricated_evidence_drops_the_whole_probe(self):
        plan, _ = self._plan(
            user_text=self.TRANSCRIPT,
            scene={"goal": "practice", "cassette_mode": "expression"},
            probe={
                "claim": "weak", "evidence": "ok",
                "evidence_quote": "a sentence the learner never said",
                "focus_key": "claim",
            },
        )
        self.assertEqual(plan["_meta"]["source"], "director_model")
        self.assertNotIn("argument_probe", plan)

    def test_non_expression_tape_gets_no_probe_and_no_rules(self):
        """模型多写也不许留下：五维字段是练习盘专属，其他盘的计划逐字不变。"""
        for scene in ({"goal": "chat"}, {"goal": "scene", "cassette_mode": "resume"}):
            with self.subTest(scene=scene):
                plan, fake = self._plan(
                    user_text=self.TRANSCRIPT, scene=scene,
                    probe={"claim": "weak", "evidence": "ok",
                           "evidence_quote": "saved two hours a day",
                           "focus_key": "claim"},
                )
                self.assertEqual(plan["_meta"]["source"], "director_model")
                self.assertNotIn("argument_probe", plan)
                system = fake.calls[0]["messages"][0]["content"]
                self.assertNotIn("五维逻辑探查", system)
                self.assertNotIn("argument_probe", system)

    def test_truncated_json_without_probe_still_plans(self):
        """字段被截断 / 模型没写 probe：计划照常，只是没有五维段。"""
        plan, _ = self._plan(
            user_text=self.TRANSCRIPT,
            scene={"goal": "practice", "cassette_mode": "expression"},
            probe=None,
        )
        self.assertEqual(plan["_meta"]["source"], "director_model")
        self.assertNotIn("argument_probe", plan)

    def test_recover_existing_plan_never_reuses_a_stale_probe(self):
        fallback = ConversationDirector.fallback_plan(
            user_text=self.TRANSCRIPT, learner_intent="open_expression",
            language_mode="target_only",
            scene={"goal": "practice", "cassette_mode": "expression"},
        )
        recovered = ConversationDirector._recover_existing_plan(
            {
                "argument_probe": {"weak": ["claim"], "focus_key": "claim",
                                   "evidence": "better for me"},
                "counterpart_role": "",
            },
            fallback,
            {"source": "fallback_error", "failure_code": "timeout"},
        )
        self.assertNotIn("argument_probe", recovered)

    def test_parse_plan_passes_the_probe_through_structurally(self):
        parsed = ConversationDirector.parse_plan(json.dumps({
            "response_need": "listen_and_continue",
            "argument_probe": {
                "claim": "weak", "evidence_quote": "better for me",
                "focus_key": "claim",
            },
        }, ensure_ascii=False))
        self.assertEqual(parsed["argument_probe"]["claim"], "weak")
        self.assertEqual(parsed["argument_probe"]["focus_key"], "claim")

    def test_plan_without_probe_has_no_extra_key(self):
        parsed = ConversationDirector.parse_plan(json.dumps({
            "response_need": "listen_and_continue",
        }))
        self.assertNotIn("argument_probe", parsed)

    def test_probe_rules_name_exactly_the_five_dimensions(self):
        """跨文件静态契约：prompt 里教的维度名必须就是模块里的那五个。"""
        from engine import expression_argument as ea
        from agent.conversation_director import _EXPRESSION_ARGUMENT_PROBE_RULES

        for dim in ea.DIMENSIONS:
            self.assertIn('"' + dim + '"', _EXPRESSION_ARGUMENT_PROBE_RULES)
        self.assertNotIn('"backing"', _EXPRESSION_ARGUMENT_PROBE_RULES)
        self.assertNotIn('"data"', _EXPRESSION_ARGUMENT_PROBE_RULES)


if __name__ == "__main__":
    unittest.main()
