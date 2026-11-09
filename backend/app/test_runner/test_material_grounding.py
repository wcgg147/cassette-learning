# -*- coding: utf-8 -*-
"""素材盘的"防编"链路：按轮选相关块 + 注入预算（2026-09-21）。

背景（真机口径，用户原话）："素材盘就是为了让 ai 的回复不编，起码在聊素材内的
内容的时候是"。但改动前注入按 blocks 数组**顺序**取前几块，实测 11,240 字素材
10 块只有 1 块（9.7%）进 prompt，24,000 字以上同样只进 1,206 字（8.4%）——
用户问第 7 节时模型手里只有第 1 节，而合同写着"素材里没写的不得编"，
模型只能编或答"素材里没有"。

本文件锁住四件事：
1. **按轮选块**：问哪一段就给哪一段（词面命中，确定性、零模型调用）；
2. **预算**：总长不超上限，且**正文永不为空**（谈资不得把原文挤光）；
3. **节选告知**：只给部分块时点明"没挑中的不代表素材里没有"；
4. **[Constraints] 剔重**：已单独成段的键不再进 JSON（否则同一份内容进两遍）。

全部离线：不打模型、不连库。
"""
import os
import json
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from agent.tutor_agent import (SessionContext, TutorAgent,
                               _constraints_for_prompt,
                               _drop_overgeneralized_comment_sentences,
                               _drop_unasked_comment_sentences,
                               _material_user_asks_body_content,
                               _drop_unasked_material_language_sentences,
                               _material_has_unasked_language_teaching,
                               _material_user_asks_language_help,
                               _clean_material_reply)
from agent.conversation_director import ConversationDirector
from engine import material_analyzer


class MaterialLanguagePresentationTests(unittest.TestCase):
    def test_material_source_question_does_not_trigger_live_web_search(self):
        question = (
            "Does the game show Pope giving the order, or is it only stated by the narrator?"
        )
        self.assertFalse(ConversationDirector.needs_material_web_search(question))
        self.assertTrue(ConversationDirector.needs_material_web_search(
            "Can you search the web for an outside source on this claim?"
        ))

    def test_external_context_is_only_added_when_user_needs_it(self):
        context = {"status": "ready", "sources": [{
            "title": "Control – The Hiss", "url": "https://controlgame.com/hiss",
            "host": "controlgame.com", "snippet": "The Hiss is an otherworldly force.",
        }]}
        self.assertTrue(TutorAgent._external_context_needed(
            "Is Hiss the same as Patterning?", context))
        self.assertTrue(TutorAgent._external_context_needed(
            "I have never played this game, what is going on?", context))
        self.assertTrue(TutorAgent._external_context_needed(
            "I'm new to Control—who are Dylan and Jesse?", context))
        self.assertTrue(TutorAgent._external_context_needed(
            "What does Patterning mean here?", context))
        self.assertFalse(TutorAgent._external_context_needed(
            "What did the squad do after setting up the relay?", context))

    def test_external_context_renderer_labels_background_and_citation(self):
        context = {"status": "ready", "sources": [{
            "title": "Control – The Hiss", "url": "https://controlgame.com/hiss",
            "host": "controlgame.com", "snippet": "A named hostile force, not all paranormal phenomena.",
        }]}
        rendered = TutorAgent._material_external_context_text(
            context, source_title="CONTROL P5", budget=700,
            include_source_details=True)
        self.assertIn("CONTROL P5", rendered)
        self.assertIn("not all paranormal phenomena", rendered)
        self.assertIn("https://controlgame.com/hiss", rendered)
        compact = TutorAgent._material_external_context_text(
            context, source_title="CONTROL P5", budget=700)
        self.assertIn("not all paranormal phenomena", compact)
        self.assertNotIn("controlgame.com", compact)
        self.assertNotIn("Control – The Hiss", compact)
        self.assertTrue(TutorAgent._material_user_asks_external_source(
            "Can you give me the source link?"))
        self.assertFalse(TutorAgent._material_user_asks_external_source(
            "What is the Hiss?"))
        self.assertTrue(TutorAgent._external_context_needed(
            "Can you give me the source link?", context))

    def test_external_background_is_synthesized_without_prompted_source_narration(self):
        external = {"status": "ready", "sources": [{
            "title": "Control – The Hiss", "url": "https://controlgame.com/hiss",
            "host": "controlgame.com",
            "snippet": "The Hiss is a hostile force in the Oldest House.",
        }]}
        ctx = SessionContext(session_id="external-background", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "title": "CONTROL P5", "kind": "video",
            "blocks": ["Grist says time is being patterned across overlapping timelines."],
            "analysis": {"external_context": external},
        }
        single = TutorAgent._material_text(ctx, "What does Patterning mean here?")
        self.assertIn("不要报告搜索过程、网页来源或资料清单", single)
        self.assertIn("结合本段附近情节解释它在故事里的含义或作用", single)
        self.assertIn("先用背景资料支持的一句话给出定义", single)
        self.assertIn("系列背景只用来解释专名", single)
        self.assertIn("外部背景不能识别本段未点名的敌人或画面", single)
        self.assertIn("先回答本轮明确问的点", single)
        self.assertNotIn("引用外部事实时标明来自外部页面", single)
        self.assertNotIn("引用外部事实时自然标明来自外部页面", single)
        self.assertNotIn("https://controlgame.com/hiss", single)
        source_question = TutorAgent._material_text(
            ctx, "Can you give me the source link for the Hiss background?")
        self.assertIn("https://controlgame.com/hiss", source_question)

        multi = TutorAgent._material_text_multi(
            {"multi": True},
            {"sources": [{"material_id": 61, "title": "CONTROL P5",
                          "external_context": external}]},
            query="Does Patterning mean the Hiss?",
        )
        self.assertIn("不要报告搜索过程、网页来源或资料清单", multi)
        self.assertIn("不能因为名称或描述相似就推断它们相同", multi)
        self.assertIn("先点名并解释第一个术语", single)
        self.assertNotIn("引用外部事实时自然标明来自外部页面", multi)

    def test_material_definition_must_name_the_requested_concept(self):
        prepared = {"constraints": {"material_tape": True}}
        missing_hiss = TutorAgent._controlled_reply_violations(
            "That's the FBC's big building in Control, the one the research team is used to working inside.",
            prepared,
            "I haven't played Control. What are people talking about when they mention the Hiss?",
        )
        self.assertTrue(any("没有回答到所问概念" in item and "Hiss" in item
                            for item in missing_hiss))

        missing_patterning = TutorAgent._controlled_reply_violations(
            "I don't have a complete answer to that yet.", prepared,
            "And what does Grist mean by Patterning in this part? Is that the Hiss?",
        )
        self.assertTrue(any("没有回答到所问概念" in item for item in missing_patterning))

        grounded = TutorAgent._controlled_reply_violations(
            "Patterning is Grist's word for overlapping timelines, not the Hiss.",
            prepared,
            "And what does Grist mean by Patterning in this part? Is that the Hiss?",
        )
        self.assertFalse(any("没有回答到所问概念" in item for item in grounded))

    def test_material_relation_answer_cannot_link_then_deny_two_concepts(self):
        query = "What does Grist mean by Patterning here? Is it the Hiss?"
        muddled = (
            "It just means the Hiss took them or reached them. "
            "And no, Patterning isn't the Hiss itself; Grist says time is stacked and repeated."
        )
        clean = (
            "Patterning is Grist's term for overlapping, repeated timelines. "
            "No, it isn't the Hiss."
        )
        unrelated = (
            "It means time is stacked and repeated. Patterning isn't the Hiss."
        )
        self.assertTrue(TutorAgent._material_relation_answer_is_confusing(query, muddled))
        self.assertFalse(TutorAgent._material_relation_answer_is_confusing(query, clean))
        self.assertFalse(TutorAgent._material_relation_answer_is_confusing(query, unrelated))
        violations = TutorAgent._controlled_reply_violations(
            muddled, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("概念关系解释前后矛盾" in item for item in violations))

    def test_material_relation_guard_does_not_change_other_tapes(self):
        query = "What does Grist mean by Patterning here? Is it the Hiss?"
        muddled = (
            "It just means the Hiss took them or reached them. "
            "And no, Patterning isn't the Hiss itself."
        )
        violations = TutorAgent._controlled_reply_violations(
            muddled, {"constraints": {"material_tape": False}}, query
        )
        self.assertFalse(any("概念关系解释前后矛盾" in item for item in violations))

    def test_material_answer_does_not_repeat_the_same_conclusion_in_adjacent_sentences(self):
        query = "By the end of this part, does Dylan know how to stop it?"
        repeated = (
            "They still don't know how to stop it. "
            "Dylan says it plainly: they still don't know how to stop it."
        )
        useful_followup = (
            "They still don't know how to stop it. The part ends with the team still searching for a way."
        )

        self.assertTrue(TutorAgent._material_reply_repeats_adjacent_core(query, repeated))
        self.assertFalse(TutorAgent._material_reply_repeats_adjacent_core(query, useful_followup))
        self.assertEqual(
            TutorAgent._material_trim_repeated_followup_sentence(query, repeated),
            "They still don't know how to stop it.",
        )
        self.assertEqual(
            TutorAgent._material_trim_repeated_followup_sentence(query, useful_followup),
            useful_followup,
        )
        violations = TutorAgent._controlled_reply_violations(
            repeated, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("同一回复内相邻句复述同一核心事实" in item
                            for item in violations))

    def test_material_answer_repeat_guard_respects_quote_requests_and_other_tapes(self):
        repeated = (
            "They still don't know how to stop it. "
            "Dylan says it plainly: they still don't know how to stop it."
        )
        self.assertFalse(TutorAgent._material_reply_repeats_adjacent_core(
            "What were Dylan's exact words?", repeated
        ))
        violations = TutorAgent._controlled_reply_violations(
            repeated, {"constraints": {"material_tape": False}},
            "By the end of this part, does Dylan know how to stop it?",
        )
        self.assertFalse(any("同一回复内相邻句复述同一核心事实" in item
                             for item in violations))

    def test_material_reply_flags_reused_internal_sentence_frame_across_turns(self):
        previous = (
            "Grist gives Dylan a device to track the fluctuations. "
            "What I find interesting is that she keeps testing the pattern herself."
        )
        current = (
            "The relays are meant to collect measurements under the Oldest House. "
            "What I find interesting is that she keeps going even though each timeline ends badly."
        )
        self.assertTrue(TutorAgent._material_reuses_internal_sentence_frame(
            current, previous, "Why does she keep working on it?"
        ))
        violations = TutorAgent._controlled_reply_violations(
            current, {"constraints": {"material_tape": True}},
            "Why does she keep working on it?", previous,
        )
        self.assertTrue(any("回复中段复用了上一轮的句子引导框架" in item
                            for item in violations))

    def test_material_sentence_frame_guard_ignores_common_starters_quotes_and_other_tapes(self):
        previous = "Grist gives Dylan a device. What I find interesting is how she uses it."
        different_frame = "The relay needs a signal. What surprises me is the timing."
        generic_starter = "The relay needs a signal. And the team starts setting it up."
        reused_frame = "The relay needs a signal. What I find interesting is the timing."

        self.assertFalse(TutorAgent._material_reuses_internal_sentence_frame(
            different_frame, previous
        ))
        self.assertFalse(TutorAgent._material_reuses_internal_sentence_frame(
            generic_starter, "The team has a plan. And the signal is still weak."
        ))
        self.assertFalse(TutorAgent._material_reuses_internal_sentence_frame(
            reused_frame, previous, "Can you quote the exact words?"
        ))
        self.assertFalse(TutorAgent._material_reuses_internal_sentence_frame(
            'The line matters here. “What I find interesting is how she uses it.”',
            previous,
        ))
        violations = TutorAgent._controlled_reply_violations(
            reused_frame, {"constraints": {"material_tape": False}},
            "Why is that interesting?", previous,
        )
        self.assertFalse(any("回复中段复用了上一轮的句子引导框架" in item
                             for item in violations))

    def test_material_answer_does_not_quote_the_same_yes_no_conclusion_again(self):
        query = "So by the end of this part, does she know how to stop it yet?"
        repeated_quote = (
            'No — not by the end of this part. Dylan says it plainly: '
            '"we still don\'t know how to stop it, but maybe now we have a chance at understanding it."'
        )
        self.assertTrue(TutorAgent._material_reply_repeats_adjacent_core(
            query, repeated_quote
        ))
        self.assertEqual(
            TutorAgent._material_trim_repeated_followup_sentence(query, repeated_quote),
            "No — not by the end of this part.",
        )

        repeated_paraphrase = (
            "No — not yet. Near the end, Dylan says they still don't know how to stop it, "
            "but they may finally understand it."
        )
        self.assertTrue(TutorAgent._material_reply_repeats_adjacent_core(
            query, repeated_paraphrase
        ))
        self.assertEqual(
            TutorAgent._material_trim_repeated_followup_sentence(query, repeated_paraphrase),
            "No — not yet.",
        )

        attributed_repeat = (
            "Right — by the end they still don't know how to stop it. "
            "Dylan says they don't know how to stop the force yet, but maybe now they can understand it."
        )
        self.assertTrue(TutorAgent._material_reply_repeats_adjacent_core(
            "And by the end they still haven't figured out how to stop it?",
            attributed_repeat,
        ))
        self.assertEqual(
            TutorAgent._material_trim_repeated_followup_sentence(
                "And by the end they still haven't figured out how to stop it?",
                attributed_repeat,
            ),
            "Right — by the end they still don't know how to stop it.",
        )

    def _plan(self, *, raw="", shown="", material=True):
        ctx = SessionContext(session_id="lang", user_id="u", target_lang="en")
        ctx.material_context = {"title": "Singapore school language"} if material else None
        ctx.last_raw_transcript = raw
        plan = {"language_presentation": "chinese_only"}
        TutorAgent._apply_material_language_presentation(ctx, shown, plan)
        return plan["language_presentation"]

    def test_chinese_content_opinion_keeps_english_response(self):
        self.assertEqual(
            self._plan(raw="我觉得中文进入学校是因为华人多。",
                       shown="I think Chinese is taught because many Chinese people live there."),
            "target_only",
        )

    def test_chinese_meaning_question_allows_chinese_explanation(self):
        self.assertEqual(
            self._plan(raw="你刚才说的什么意思？", shown="What do you mean?"),
            "chinese_only",
        )

    def test_english_meaning_question_uses_simple_english(self):
        self.assertEqual(
            self._plan(raw="What do you mean by that?", shown="What do you mean by that?"),
            "target_only",
        )

    def test_explicit_english_request_for_chinese_is_respected(self):
        self.assertEqual(
            self._plan(raw="Can you explain that in Chinese?",
                       shown="Can you explain that in Chinese?"),
            "chinese_only",
        )

    def test_unrequested_chinese_material_reply_is_flagged_for_rewrite(self):
        prepared = {"constraints": {
            "material_tape": True,
            "conversation_director": {"language_presentation": "target_only"},
        }}
        violations = TutorAgent._controlled_reply_violations(
            "新加坡的学校让英语成为主要教学语言，中文按族群作为母语课程。",
            prepared,
            user_transcript="I think their school system is different.",
        )
        self.assertTrue(any("英语输入环境" in item for item in violations))

    def test_requested_chinese_material_explanation_is_not_flagged(self):
        prepared = {"constraints": {
            "material_tape": True,
            "conversation_director": {"language_presentation": "chinese_only"},
        }}
        violations = TutorAgent._controlled_reply_violations(
            "英语是学校的主要教学语言。",
            prepared,
            user_transcript="什么意思？",
        )
        self.assertFalse(any("英语输入环境" in item for item in violations))

    def test_other_tapes_are_unchanged(self):
        self.assertEqual(
            self._plan(raw="What does that mean?", shown="What does that mean?",
                       material=False),
            "chinese_only",
        )

    def test_material_comment_sample_cannot_be_spoken_as_an_audience_majority(self):
        prepared = {"constraints": {"material_tape": True}}
        violations = TutorAgent._controlled_reply_violations(
            "The comments mostly push back on the ranking; one says X and another says Y.",
            prepared,
            user_transcript="What do the comments add?",
        )
        self.assertTrue(any("评论区是个体样本" in item for item in violations))

    def test_material_body_takeaway_query_does_not_get_a_comment_only_answer(self):
        query = "So if you set the names aside, what's the most useful point the video leaves us with?"
        reply = "Those are individual comments, so they don't tell us what viewers overall think."
        self.assertTrue(_material_user_asks_body_content(query))
        violations = TutorAgent._controlled_reply_violations(
            reply, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("回复却转向未询问的评论区" in item for item in violations))
        self.assertEqual(_drop_unasked_comment_sentences(reply, query), "")

    def test_explicit_comment_question_keeps_comment_answer(self):
        query = "What do the comments add to the video?"
        reply = "One commenter adds an agency angle."
        self.assertFalse(_material_user_asks_body_content(query))
        self.assertEqual(_drop_unasked_comment_sentences(reply, query), reply)

    def test_material_reply_drops_unsolicited_language_lesson_but_keeps_answer(self):
        query = "I only saw the title. What's this video mainly about?"
        reply = (
            'Quick note first — "only" means just that one thing and nothing more. '
            "As for the video, it's a rundown of this year's new faces in the industry."
        )
        self.assertEqual(
            _drop_unasked_material_language_sentences(reply, query),
            "As for the video, it's a rundown of this year's new faces in the industry.",
        )
        violations = TutorAgent._controlled_reply_violations(
            reply, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("不要插入词义、语法或表达纠正" in item
                            for item in violations))

    def test_material_reply_drops_inline_definition_of_a_user_word(self):
        query = "I only saw the title. What's this video mainly about?"
        reply = (
            "It's a roundup of the first-half newcomers. "
            'So "mainly" here just means the main thing the video is about.'
        )
        self.assertEqual(
            _drop_unasked_material_language_sentences(reply, query),
            "It's a roundup of the first-half newcomers.",
        )

    def test_material_reply_drops_unsolicited_term_correction_and_keeps_answer(self):
        query = "So if you set the names aside, what's the most useful point?"
        reply = (
            "I think you mean 'label's' — like the label's point, the studio's point. "
            "Anyway, the most useful point is that the creator sees a thinner pool of newcomers."
        )
        self.assertEqual(
            _clean_material_reply(reply, query),
            "The most useful point is that the creator sees a thinner pool of newcomers.",
        )
        violations = TutorAgent._controlled_reply_violations(
            reply, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("不要插入词义、语法或表达纠正" in item
                            for item in violations))

    def test_material_reply_drops_spontaneous_quick_check_corrections(self):
        query = "Then what do they mean by Patterning? Is it the Hiss?"
        reply = (
            "Quick check first — did you mean 'than,' the word I used a moment ago? "
            "If so, 'than' is just a comparing word: 'more than,' 'bigger than.' "
            "No — the Hiss is one specific hostile force, not the name for every strange thing."
        )
        self.assertEqual(
            _clean_material_reply(reply, query),
            "No — the Hiss is one specific hostile force, not the name for every strange thing.",
        )

        follow_up = (
            "Quick check — did you mean 'moment' there, not 'meant'? "
            "Anyway, on the story: Grist wants the readings from each timeline so she can compare the pattern."
        )
        self.assertEqual(
            _clean_material_reply(follow_up, "Why does Grist need data from every timeline?"),
            "Grist wants the readings from each timeline so she can compare the pattern.",
        )
        unasked_word_note = (
            "I think you mean 'against' — the word I used earlier. "
            "Patterning is Grist's name for the overlapping timelines she observes."
        )
        self.assertEqual(
            _clean_material_reply(
                unasked_word_note,
                "What does Grist mean by Patterning? Is that the Hiss?",
            ),
            "Patterning is Grist's name for the overlapping timelines she observes.",
        )

    def test_material_reply_drops_unsolicited_chinese_phrase_praise(self):
        query = "So in this clip, that order is only stated by the narrator, right?"
        reply = (
            "Yes, the order is only stated by the narrator. "
            "关键差别在 ‘only stated by’ 这个说法很准，不用改。"
        )
        self.assertEqual(
            _drop_unasked_material_language_sentences(reply, query),
            "Yes, the order is only stated by the narrator.",
        )

    def test_material_reply_drops_unsolicited_chinese_grammar_praise(self):
        query = "So in this clip, the order is only stated by the narrator, right?"
        reply = (
            "The order is reported by the narrator, not shown as a scene. "
            "你这句语法没问题，只是 ‘stated’ 稍书面一点，口语里也常说 ‘mentioned’。"
        )
        self.assertEqual(
            _drop_unasked_material_language_sentences(reply, query),
            "The order is reported by the narrator, not shown as a scene.",
        )

    def test_material_reply_drops_unsolicited_english_idiom_praise_but_keeps_answer(self):
        query = "So the Jesse part is still left hanging for the next episode, right?"
        reply = (
            'That sentence is natural — "left hanging" is exactly what people say '
            "when a story opens a question and doesn't close it. And yes, that's "
            "the shift — Dylan's real thread is Jesse, and he won't explain what he saw."
        )

        self.assertEqual(
            _clean_material_reply(reply, query),
            "And yes, that's the shift — Dylan's real thread is Jesse, and he won't explain what he saw.",
        )

    def test_material_reply_keeps_idiom_explanation_when_user_asks(self):
        query = "Is 'left hanging' a natural phrase here?"
        reply = 'Yes, "left hanging" is a natural phrase for an unresolved story.'

        self.assertEqual(_clean_material_reply(reply, query), reply)

    def test_material_reply_drops_question_echo_but_preserves_direct_answer(self):
        query = "What does he think is behind that?"
        reply = (
            "So you're asking what he thinks the reason is. "
            "He doesn't really explain the gap; he just says he sees it as a difference in taste."
        )
        self.assertEqual(
            _clean_material_reply(reply, query),
            "He doesn't really explain the gap; he just says he sees it as a difference in taste.",
        )

    def test_material_explicit_language_question_keeps_requested_explanation(self):
        query = "What does the word 'only' mean here?"
        reply = '"Only" means that nothing else is included.'
        self.assertEqual(
            _drop_unasked_material_language_sentences(reply, query), reply
        )
        self.assertFalse(any(
            "不要插入词义、语法或表达纠正" in item
            for item in TutorAgent._controlled_reply_violations(
                reply, {"constraints": {"material_tape": True}}, query
            )
        ))

    def test_control_prompt_scopes_missing_support_to_the_claim_asked_about(self):
        ctx = SessionContext(session_id="material-claim-scope", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "title": "First-half newcomer review",
            "kind": "video",
            "creator_display_name": "纯黑",
            "summary": "The speaker makes a five-year quality comparison and mentions rankings elsewhere.",
            "blocks": ["The speaker says this is the weakest first half in five years."],
        }
        agent = TutorAgent.__new__(TutorAgent)
        prompt = agent._build_control_prompt(
            ctx, {}, user_transcript="Does the video show evidence for the five-year claim?")
        self.assertIn("把缺口限定到这个结论", prompt)
        self.assertIn("中文人名和厂牌名必须照素材摘要或原文中的字形逐字引用", prompt)
        self.assertIn("中文只保留为必要的原文人名或厂牌名", prompt)
        self.assertIn("不要整句切换成中文", prompt)
        self.assertIn("先说作者提供的解释", prompt)
        self.assertIn("不要先问‘Did you mean...’", prompt)
        self.assertIn("用户只问素材大意时，用1到2句说明素材主题和它覆盖的内容", prompt)
        self.assertIn("不需要点名时省略专名", prompt)
        self.assertIn("需要指认这段旁白时直接用账号名作归属", prompt)
        self.assertIn("不要先称“the narrator/commentator”再补账号名", prompt)
        self.assertIn("只有评论原话明确表达同一立场时才说它支持作者", prompt)
        self.assertIn("先准确说出评论实际提出的关系或例子", prompt)
        self.assertIn("说明它与正文观点的连接点和证明边界", prompt)
        self.assertIn("评论与正文观点若提到同一机构或人物", prompt)
        self.assertIn("评论里相邻的评价与名单不能自动拼成因果", prompt)
        self.assertIn("不能只贴‘支持/不支持/补充背景/不同角度’标签", prompt)
        self.assertIn("把 recent_dialogue 里已经说过的开场白当作双方共同知道的信息", prompt)
        self.assertIn("不要换词复述开场的判断、例子或悬念", prompt)
        self.assertIn("以 So/Then 开头且没有提出新问题焦点的确认问句", prompt)
        self.assertIn("数量词和范围必须能在正文证据中逐项对应", prompt)
        self.assertIn("把不同解释逐项说到并保留其确定程度", prompt)
        self.assertIn("不要只给‘有用/有意思/很有背景’这类标签", prompt)
        self.assertIn("只举一条就说‘一位评论者’", prompt)
        self.assertIn("上一轮聊过评论区不代表本轮还在问评论", prompt)
        self.assertIn("评论里的数字只能表述为评论者声称或提到的数字", prompt)
        self.assertIn("专名的实体类别和关系要照来源保留", prompt)
        self.assertIn("不能因缺少纵向量化数据就说素材完全没有比较", prompt)
        self.assertIn("不要把它升级成已证实的行业规律", prompt)
        self.assertIn("不要为了重申主题再讲一遍开场结论", prompt)
        self.assertIn("不能略掉或改称无关判断", prompt)
        self.assertIn("超出正文的机制只能明确标成推测", prompt)
        self.assertIn("用户问素材留下的最有价值观点或 takeaway 时", prompt)
        self.assertIn("不要用‘真正的故事’代替答案", prompt)
        self.assertIn("不要把多因素解释压成原文没有说过的非此即彼结论", prompt)
        self.assertIn("不要拿相邻的表现、数字或例子替代因果解释", prompt)

    def test_material_confirmation_and_creator_name_are_per_turn_context(self):
        ctx = SessionContext(session_id="material-creator-turn", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "multi": True,
            "sources": [{
                "material_id": 61,
                "title": "Control gameplay",
                "creator_display_name": "纯黑",
                "creator_display_name_source": "user_confirmed",
                "voice_source_spans": [{
                    "source_type": "creator_narration",
                    "confidence": "high",
                }],
            }],
        }
        agent = TutorAgent.__new__(TutorAgent)
        prompt = agent._build_control_prompt(ctx, {
            "constraints": {
                "material_tape": True,
                "material_confirmation_followup": True,
                "material": "[Material] The order is only stated in the source.",
            },
        }, user_transcript="So it is only stated in the voice-over, right?")
        self.assertIn("[Material Confirmation - this turn]", prompt)
        self.assertIn("不是英语表达求助", prompt)
        self.assertIn("[Material Creator Naming - this source]", prompt)
        self.assertIn("纯黑's voice-over/commentary", prompt)
        self.assertIn("同一事实只解释一次", prompt)
        renamed = TutorAgent._prefer_creator_handle_for_narration(
            "The narrator says Pope sent the research team below. "
            "That order is not shown on screen.",
            ctx,
        )
        self.assertEqual(
            renamed,
            "纯黑 says Pope sent the research team below. "
            "That order is not shown on screen.",
        )

    def test_material_reply_cannot_speak_for_comments_as_a_group(self):
        prepared = {"constraints": {"material_tape": True}}
        violations = TutorAgent._controlled_reply_violations(
            "The comments push back on the video's main point. One says X, another questions Y.",
            prepared,
            user_transcript="What do the comments add?",
        )
        self.assertTrue(any("评论区是个体样本" in item for item in violations))

    def test_material_reply_can_say_comment_examples_do_not_establish_a_majority(self):
        prepared = {"constraints": {"material_tape": True}}
        violations = TutorAgent._controlled_reply_violations(
            "The examples include different individual reactions, but they don't tell us what most viewers think.",
            prepared,
            user_transcript="Can we tell what most viewers think?",
        )
        self.assertFalse(any("评论区是个体样本" in item for item in violations))

    def test_material_reply_does_not_generalize_about_comments_via_they(self):
        query = "Do the comments add a useful perspective, or are they mostly individual preferences?"
        reply = (
            "They don't really build on his argument; they just swap in their own favorites. "
            "One comment adds an agency perspective by naming S-Flirt."
        )
        violations = TutorAgent._controlled_reply_violations(
            reply, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("评论区是个体样本" in item for item in violations))
        self.assertEqual(
            _drop_overgeneralized_comment_sentences(reply, query),
            "One comment adds an agency perspective by naming S-Flirt.",
        )

    def test_comment_guard_prunes_only_bad_sentence_and_keeps_the_gist(self):
        reply = (
            "The video reviews the first-half newcomer lineup and the speaker's ranking. "
            "The comments push back on the video's main point."
        )
        self.assertEqual(
            _drop_overgeneralized_comment_sentences(reply),
            "The video reviews the first-half newcomer lineup and the speaker's ranking.",
        )

    def test_comment_guard_preserves_individual_examples(self):
        reply = (
            "The comments push back on the ranking. "
            "One commenter prefers Hakata Ayaha, while another criticizes S1's choice."
        )
        self.assertEqual(
            _drop_overgeneralized_comment_sentences(reply),
            "One commenter prefers Hakata Ayaha, while another criticizes S1's choice.",
        )

    def test_body_evidence_answer_drops_unrequested_comment_details(self):
        query = "When he says this, is it backed by sales or audience data, or his own judgment?"
        reply = (
            "It's his judgment; the video gives no sales or audience figures. "
            "One commenter lists several names, and another names an MVP."
        )
        self.assertEqual(
            _drop_unasked_comment_sentences(reply, query),
            "It's his judgment; the video gives no sales or audience figures.",
        )
        violations = TutorAgent._controlled_reply_violations(
            reply, {"constraints": {"material_tape": True}}, query
        )
        self.assertTrue(any("正文证据问题夹入评论区内容" in item for item in violations))

    def test_body_takeaway_answer_drops_unrequested_comment_details(self):
        query = "What's the most useful point the video leaves us with?"
        reply = (
            "The video suggests a thinner newcomer pool may be changing how labels recruit. "
            "One comment names a different favorite."
        )
        self.assertEqual(
            _drop_unasked_comment_sentences(reply, query),
            "The video suggests a thinner newcomer pool may be changing how labels recruit.",
        )

    def test_comment_followup_is_not_mistaken_for_body_evidence_query(self):
        query = "What does the comment section add to that point?"
        reply = "One commenter agrees, while another disagrees."
        self.assertEqual(_drop_unasked_comment_sentences(reply, query), reply)

    def test_failed_material_takeaway_repair_falls_back_on_topic(self):
        prepared = {"constraints": {"material_tape": True}}
        query = "What's the most useful point the video leaves us with?"
        for violation in (
            "回复不能为空",
            "评论区是个体样本",
            "用户当前问视频或素材正文，回复却转向未询问的评论区",
        ):
            with self.subTest(violation=violation):
                reply = TutorAgent._conversation_contract_fallback(
                    prepared, [violation], query
                )
                self.assertEqual(reply, "I don't want to guess at the video's main point.")

    def test_unfinished_reply_is_rejected_for_material_and_other_tapes(self):
        reply = "The speaker ranks the studios by the newcomers they signed, but"
        self.assertTrue(any("回复句尾停在未完成" in item for item in
                            TutorAgent._controlled_reply_violations(
                                reply, {"constraints": {"material_tape": True}})))
        self.assertTrue(any("回复句尾停在未完成" in item for item in
                            TutorAgent._controlled_reply_violations(
                                reply, {"constraints": {}})))

    def test_unfinished_embedded_clause_is_rejected_and_keeps_complete_prefix(self):
        reply = (
            "The narrator says Pope sent them to bring back the data. "
            "That purpose is not something a character"
        )
        self.assertTrue(any("回复句尾停在未完成" in item for item in
                            TutorAgent._controlled_reply_violations(
                                reply, {"constraints": {"material_tape": True}})))
        self.assertEqual(
            TutorAgent._complete_reply_prefix(reply),
            "The narrator says Pope sent them to bring back the data.",
        )

    def test_long_unpunctuated_reply_is_rejected_but_short_fragment_is_valid(self):
        reply = (
            "The narrator reports that Pope sent the team down to find the other researchers "
            "and bring their data back. The game dialogue does not confirm that order; "
            "the characters never confirm"
        )
        self.assertTrue(any("回复句尾停在未完成" in item for item in
                            TutorAgent._controlled_reply_violations(
                                reply, {"constraints": {"material_tape": True}})))
        self.assertEqual(
            TutorAgent._complete_reply_prefix(reply),
            "The narrator reports that Pope sent the team down to find the other researchers and bring their data back.",
        )
        self.assertFalse(any("回复句尾停在未完成" in item for item in
                             TutorAgent._controlled_reply_violations(
                                 "Yes", {"constraints": {"material_tape": True}})))

    def test_incomplete_reply_uses_a_valid_same_turn_director_draft(self):
        prepared = {"constraints": {"material_tape": True}}
        reply, used_draft = TutorAgent._incomplete_reply_fallback(
            "The video suggests that one label",
            "The speaker says S1 took most of the standout newcomers this half.",
            prepared,
            user_transcript="What's the main point of the video?",
        )
        self.assertTrue(used_draft)
        self.assertEqual(
            reply,
            "The speaker says S1 took most of the standout newcomers this half.",
        )

    def test_incomplete_reply_rejects_an_unfinished_director_draft(self):
        prepared = {"constraints": {"material_tape": True}}
        reply, used_draft = TutorAgent._incomplete_reply_fallback(
            "The video suggests that one label",
            "The speaker says S1 took most of the standout newcomers, but",
            prepared,
            user_transcript="What's the main point of the video?",
        )
        self.assertFalse(used_draft)
        self.assertEqual(reply, "I don't have a complete answer to that yet.")


class MaterialIndirectFollowupTests(unittest.TestCase):
    def test_broad_overview_wording_uses_the_story_recap_path(self):
        ctx = SessionContext(session_id="broad-overview", user_id="u", target_lang="en")
        ctx.material_context = {
            "sources": [{"material_id": 61, "title": "Episode five",
                         "analysis_source_map": {"validated": True, "beats": [{
                             "order": 1, "event": "The search begins.",
                             "evidence": {"quote": "The search begins below the station.",
                                          "source_offset": 0},
                         }]}}],
            "evidence_blocks": [
                {"material_id": 61, "block_no": no, "span_kind": "body",
                 "text": f"Story stage {no}. " + "detail " * 120}
                for no in range(1, 9)
            ],
            "hooks": [{"text": "Is this courage or recklessness?",
                       "material_id": 61, "block_no": 1,
                       "span_kind": "body", "evidence_quote": "Story stage 1."}],
        }
        query = "I only saw the title of this part. What's going on in it?"
        self.assertTrue(TutorAgent._is_material_story_recap_query(query))
        packet = TutorAgent._material_evidence_packet(ctx, query)
        self.assertEqual(packet["refs"], ["61:1", "61:3", "61:6", "61:8"])
        self.assertTrue(TutorAgent._is_material_story_recap_query(
            "What's this part about?"))
        self.assertTrue(TutorAgent._is_material_story_recap_query(
            "这段视频主要讲什么？"))

    def test_missed_episode_recap_spans_the_story_instead_of_only_keyword_matches(self):
        ctx = SessionContext(session_id="missed-episode", user_id="u", target_lang="en")
        ctx.material_context = {
            "sources": [{"material_id": 61, "title": "Episode five",
                         "analysis_source_map": {"validated": True, "beats": [{
                             "order": 1, "event": "The search begins.",
                             "evidence": {"quote": "The search begins below the station.",
                                          "source_offset": 0},
                         }]} }],
            "evidence_blocks": [
                {"material_id": 61, "block_no": no, "span_kind": "body",
                 "text": f"Story stage {no}. " + "detail " * 120}
                for no in range(1, 9)
            ],
            "hooks": [],
        }
        query = "I watched the earlier parts but missed this one. Catch me up."
        packet = TutorAgent._material_evidence_packet(ctx, query)
        self.assertEqual(packet["refs"], ["61:1", "61:3", "61:6", "61:8"])
        self.assertLessEqual(sum(map(len, packet["lines"])),
                             TutorAgent.MATERIAL_EVIDENCE_MAX_CHARS)
        rendered = TutorAgent._material_text_multi(ctx.material_context, packet, query)
        self.assertIn("The search begins", rendered)
        self.assertNotIn("素材库逐份摘要", rendered)

    def test_validated_story_map_survives_runtime_cleaning_twice(self):
        raw = {"validated": True, "beats": [{
            "order": 1, "event": "The team enters the underground space.",
            "evidence": {"quote": "The team enters the underground space.",
                         "source_offset": 120, "source_type": "in_material_dialogue"},
        }]}
        once = TutorAgent._material_source_map({"source_map": raw})
        twice = TutorAgent._material_source_map({"source_map": once})
        self.assertEqual(twice, once)
        self.assertEqual(twice["beats"][0]["event"],
                         "The team enters the underground space.")

    def test_video_hasnt_talked_about_other_things_selects_an_uncovered_hook(self):
        ctx = SessionContext(session_id="material-followup", user_id="u", target_lang="en")
        ctx.material_context = {
            "summary": "视频正文未转写。仅从页面标题《2026 first-half newcomer review》"
                       "可以判断主题范围，无法确认UP主的具体论点；评论区仅是观众个体反应。",
            "groups": [{
                "title": "正文缺失+评论样本",
                "summary": "评论区仅是观众个体反应。",
                "material_ids": [9],
            }],
            "evidence_blocks": [
                {"material_id": 7, "block_no": 1, "span_kind": "body",
                 "text": "The creator describes hotel prices in Singapore."},
                {"material_id": 7, "block_no": 8, "span_kind": "body",
                 "text": "The creator compares local food and public transport costs."},
            ],
            "hooks": [{"text": "What else did the creator find surprising?",
                       "material_id": 7, "block_no": 8}],
            "sources": [{"material_id": 7, "title": "Singapore video"}],
        }
        query = "In his video he hasn't talked about other things yet, it's quite interesting."
        packet = TutorAgent._material_evidence_packet(ctx, query)
        self.assertIn("7:8", packet["refs"])

    def test_relevant_video_block_survives_budget_and_gets_exact_timestamp(self):
        unrelated = ("Unrelated background passage. " * 50)[:1200]
        request_quote = (
            "but to get the data for that, i need to cross this chasm and set up relays "
            "directly underneath the oldest house."
        )
        relevant = (
            "Dr. Grist asks Dylan to cross the chasm and set up relays so she can collect "
            "data on the Patterning. all my research into the patterning has come up "
            "empty, but i missed one very important factor. " + request_quote
            + (" Additional transcript context." * 35)
        )
        context = {
            "multi": True,
            "sources": [{
                "material_id": 61,
                "title": "Control gameplay P5",
                "kind": "video",
                "source_coverage": {"kind": "video", "body_available": True},
                "voice_source_spans": [{
                    "start_ms": 1792260,
                    "end_ms": 1801000,
                    "source_type": "in_material_dialogue",
                    "confidence": "high",
                    "evidence": "all my research into the patterning has come up empty, but i missed one very important factor.",
                }, {
                    "start_ms": 1815600,
                    "end_ms": 1827560,
                    "source_type": "in_material_dialogue",
                    "confidence": "high",
                    "evidence": request_quote,
                }],
            }],
            "evidence_blocks": [
                {"material_id": 61, "block_no": 1, "span_kind": "body",
                 "text": unrelated},
                {"material_id": 61, "block_no": 10, "span_kind": "body",
                 "text": relevant},
                {"material_id": 61, "block_no": 11, "span_kind": "body",
                 "text": "Dylan says he saw something about Jesse."},
            ],
            "hooks": [],
        }
        ctx = SessionContext(session_id="material-priority", user_id="u",
                             target_lang="en")
        ctx.material_context = context

        packet = TutorAgent._material_evidence_packet(
            ctx, "What exactly did she need Dylan to do with those relays, and why?")

        self.assertIn("61:10", packet["refs"])
        self.assertNotIn("61:1", packet["refs"])
        self.assertTrue(any("set up relays" in line for line in packet["lines"]))
        self.assertTrue(any("30:16" in line for line in packet["lines"]))
        self.assertFalse(any("29:52" in line for line in packet["lines"]))
        self.assertFalse(any("Additional transcript context" in line
                             for line in packet["lines"]), packet["lines"])
        self.assertTrue(all("素材61" not in line and "第10段" not in line
                            for line in packet["lines"]))

        rendered = TutorAgent._material_text_multi(context, packet)
        self.assertIn("视频仅在原文证据行附有精确时间锚时", rendered)
        self.assertIn("绝不向用户输出素材 ID、段号", rendered)

    def test_internal_material_refs_are_removed_before_reply_is_delivered(self):
        reply = (
            "In [素材61·第11段] Dylan says he saw something about Jesse. "
            "That detail is repeated in [Material 25, section 4]."
        )

        cleaned = _clean_material_reply(reply, "What did Dylan see?")

        self.assertNotIn("素材61", cleaned)
        self.assertNotIn("第11段", cleaned)
        self.assertNotIn("Material 25", cleaned)
        self.assertIn("Dylan says he saw something about Jesse", cleaned)

    def test_comment_blocks_are_secondary_unless_user_asks_about_comments(self):
        ctx = SessionContext(session_id="material-comments", user_id="u", target_lang="en")
        ctx.material_context = {
            "evidence_blocks": [
                {"material_id": 9, "block_no": 1, "span_kind": "body",
                 "text": "Title: 2026 newcomer review. Views: 430004. Saves: 11613. "
                         "The creator reviews the studios' 2026 newcomer lineups."},
                {"material_id": 9, "block_no": 2, "span_kind": "comment",
                 "text": "A viewer calls X the half-year MVP."},
            ],
            "hooks": [
                {"text": "The creator compares newcomer lineups.",
                 "material_id": 9, "block_no": 1, "span_kind": "body"},
                {"text": "A viewer calls X the MVP.",
                 "material_id": 9, "block_no": 2, "span_kind": "comment"},
            ],
            "sources": [{"material_id": 9, "title": "2026 first-half newcomer review",
                         "source_coverage": {"kind": "video",
                                             "body_available": False}}],
        }
        default_packet = TutorAgent._material_evidence_packet(
            ctx, "What is the video about?")
        self.assertEqual(default_packet["lines"], [],
                         "page metadata must not masquerade as a transcript")
        self.assertEqual(len(default_packet["hooks"]), 0,
                         "page-only sources cannot support creator-argument hooks")
        self.assertEqual(default_packet["total_blocks"], 2)
        self.assertIn("2026 first-half newcomer review", default_packet["summary"])
        self.assertNotIn("评论区", default_packet["summary"])
        self.assertEqual(default_packet["groups"], [])
        rendered = TutorAgent._material_text_multi(ctx.material_context, default_packet)
        self.assertIn("2026 first-half newcomer review", rendered)
        self.assertIn("不要主动报播放、点赞、收藏等数字", rendered)
        self.assertNotIn("430004", rendered)
        self.assertNotIn("11613", rendered)
        self.assertNotIn("个体反应", rendered)

        comment_packet = TutorAgent._material_evidence_packet(
            ctx, "What are people saying in the comments?")
        self.assertTrue(any("评论" in line for line in comment_packet["lines"]))
        self.assertTrue(all("430004" not in line and "11613" not in line
                            for line in comment_packet["lines"]))
        self.assertEqual(len(comment_packet["hooks"]), 0)

    def test_explicit_comment_query_retrieves_matching_individual_sample(self):
        ctx = SessionContext(session_id="material-comment-samples", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "evidence_blocks": [{
                "material_id": 57,
                "block_no": 8,
                "span_kind": "comment",
                "text": (
                    "The first viewer prefers a different newcomer.\n"
                    "This viewer calls S-Flirt the agency MVP and lists names at IP and S1.\n"
                    "Another viewer only likes the video's production quality."
                ),
            }],
            "sources": [{"material_id": 57, "title": "Newcomer review"}],
        }

        packet = TutorAgent._material_evidence_packet(
            ctx, "Does the S-Flirt comment support the claim about S1?")

        self.assertEqual(packet["refs"], ["57:8"])
        self.assertEqual(len(packet["lines"]), 1)
        self.assertIn("S-Flirt", packet["lines"][0])
        self.assertIn("IP and S1", packet["lines"][0])
        self.assertNotIn("prefers a different newcomer", packet["lines"][0])
        self.assertNotIn("production quality", packet["lines"][0])
        self.assertIn("[评论区样本]", packet["lines"][0])

    def test_material_context_keeps_full_source_summary_separate_from_tape_summary(self):
        info = {
            "multi": True,
            "summary": "The tape overview says the source compares several labels.",
            "sources": [{
                "material_id": 57,
                "title": "2026 first-half newcomer review",
                "summary": "The library summary names the labels and distinguishes "
                           "one label's strongest newcomer from the overall standout.",
            }],
            "evidence_blocks": [{
                "material_id": 57, "block_no": 1, "span_kind": "body",
                "text": "The creator reviews newcomer lineups across labels.",
            }],
            "hooks": [],
        }
        ctx = SessionContext(session_id="source-summary", user_id="u", target_lang="en")
        ctx.material_context = info
        packet = TutorAgent._material_evidence_packet(ctx, "What is this video about?")
        rendered = TutorAgent._material_text_multi(info, packet)
        self.assertIn("整盘提要", rendered)
        self.assertIn("素材库逐份摘要", rendered)
        self.assertIn("one label's strongest newcomer", rendered)
        self.assertIn("具体事实仍核对下方原文", rendered)
        self.assertIn("某厂牌内部的最佳", rendered)
        self.assertIn("整期总体最佳", rendered)
        self.assertIn("明确问“哪些/分别有什么”", rendered)
        self.assertIn("绝不自行音译、翻译或改写拼写", rendered)
        self.assertIn("不能擅自升级为唯一原因", rendered)

    def test_non_comment_question_uses_body_summary_when_tape_overview_mixes_comments(self):
        info = {
            "multi": True,
            "summary": "The speaker blames weak recruitment. Comments mostly discuss personal favorites.",
            "sources": [{
                "material_id": 57,
                "title": "First-half newcomer review",
                "summary": "The speaker blames self-media and studios preferring proven performers.",
            }],
            "groups": [{
                "title": "Comment discussion",
                "summary": "Viewers mostly discuss personal favorites.",
                "material_ids": [57],
            }],
            "evidence_blocks": [{
                "material_id": 57, "block_no": 1, "span_kind": "body",
                "text": "The creator says social media and studio strategy hurt newcomer recruitment.",
            }, {
                "material_id": 57, "block_no": 2, "span_kind": "comment",
                "text": "A commenter prefers a different performer.",
            }],
            "hooks": [],
        }
        ctx = SessionContext(session_id="body-summary", user_id="u", target_lang="en")
        ctx.material_context = info
        body_packet = TutorAgent._material_evidence_packet(
            ctx, "Why does the speaker call this a weak year?")
        body_rendered = TutorAgent._material_text_multi(info, body_packet)
        self.assertIn("self-media and studios preferring proven performers", body_rendered)
        self.assertNotIn("personal favorites", body_rendered)
        self.assertNotIn("Comment discussion", body_rendered)

        comment_packet = TutorAgent._material_evidence_packet(
            ctx, "What do the comments say?")
        comment_rendered = TutorAgent._material_text_multi(info, comment_packet)
        self.assertIn("personal favorites", comment_rendered)

    def test_page_metrics_are_available_only_when_explicitly_requested(self):
        ctx = SessionContext(session_id="material-page-metadata", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "evidence_blocks": [{
                "material_id": 12, "block_no": 1, "span_kind": "body",
                "text": "Title: 2026 newcomer review. Views: 430004. Saves: 11613.",
            }],
            "sources": [{"material_id": 12, "title": "2026 newcomer review",
                         "source_coverage": {"kind": "video",
                                             "body_available": False}}],
        }
        packet = TutorAgent._material_evidence_packet(
            ctx, "How many views and saves does the video have?")
        self.assertEqual(len(packet["lines"]), 1)
        self.assertIn("430004", packet["lines"][0])
        self.assertIn("11613", packet["lines"][0])

    def test_repeat_challenge_is_detected_in_english_and_chinese(self):
        self.assertTrue(TutorAgent._is_material_repeat_challenge(
            "We talked about this thing, right? Are you sure?"))
        self.assertTrue(TutorAgent._is_material_repeat_challenge("我们刚才聊过这个吗？你确定吗？"))
        self.assertFalse(TutorAgent._is_material_repeat_challenge(
            "Did the guy in the video talk about anything else interesting?"))

    def test_paraphrased_confirmation_of_last_material_answer_is_detected(self):
        history = [{
            "role": "assistant",
            "content": ("From what I can see, it's only the narrator who says it. "
                        "The line about Pope sending the research team in comes from "
                        "the creator's commentary, not character dialogue. The clips "
                        "don't show Pope giving that order on screen."),
        }]
        self.assertTrue(TutorAgent._is_material_confirmation_followup(
            "So in this clip, the order is only stated by the narrator, right?", history))
        self.assertFalse(TutorAgent._is_material_confirmation_followup(
            "What does Pope do later in the game?", history))
        self.assertFalse(TutorAgent._is_material_confirmation_followup(
            "Is the narrator's interpretation correct?", history))
        story_history = [{
            "role": "assistant",
            "content": ("By the end, Dylan tells Zoe that what he saw concerns Jesse, "
                        "and that lead is left hanging."),
        }]
        self.assertTrue(TutorAgent._is_material_confirmation_followup(
            "So the Jesse clue is left open for the next part?", story_history))
        self.assertFalse(TutorAgent._is_material_confirmation_followup(
            "So who is Jesse?", story_history))
        self.assertFalse(TutorAgent._is_material_confirmation_followup(
            "So why does Grist think the Oldest House can help?", story_history))

    def test_confirmation_followup_flags_repeated_source_explanation(self):
        previous = (
            "From what I can see, it's only the narrator who says it. In the first "
            "material block, the line about Pope sending the research team in comes "
            "from the creator's commentary, not character dialogue. The clips don't "
            "show Pope giving that order on screen."
        )
        repeated = (
            "The line about Pope sending the research team comes from the creator's "
            "narration in [Material 61, section 1], not from any character dialogue. "
            "The clips I have don't show Pope giving that order on screen."
        )
        violations = TutorAgent._controlled_reply_violations(
            repeated,
            {"constraints": {"material_tape": True,
                             "recent_dialogue": [{"role": "assistant",
                                                  "content": previous}]}},
            "So in this clip, the order is only stated by the narrator, right?",
        )
        self.assertTrue(any("确认素材结论" in item for item in violations))

    def test_verbose_affirmative_confirmation_is_compacted(self):
        previous = (
            "From what I can see, it's only the narrator who says it. In the first "
            "material block, the line about Pope sending the research team comes "
            "from 纯黑's commentary, not character dialogue. The clip doesn't show "
            "Pope giving that order on screen."
        )
        question = "So in this clip, the order is only stated by the narrator, right?"
        repeated = (
            "One small thing: ‘only stated’ can sound like you're doubting it, so if "
            "you just want to confirm, ‘the order comes from the narrator’ sounds a "
            "bit softer. And yes — in [素材61·第1段] the line about Pope sending the "
            "research team comes from 纯黑's commentary, not from an in-game scene "
            "where Pope speaks, so the order itself isn't shown on screen in that clip."
        )
        prepared = {"constraints": {
            "material_tape": True,
            "recent_dialogue": [{"role": "assistant", "content": previous}],
        }}
        cleaned = _clean_material_reply(repeated, question)
        self.assertTrue(_material_has_unasked_language_teaching(repeated, question))
        self.assertNotIn("One small thing", cleaned)
        compact = TutorAgent._compact_material_confirmation(
            cleaned, question, prepared["constraints"]["recent_dialogue"]
        )
        self.assertEqual(compact, "Exactly.")
        compact_without_history = TutorAgent._compact_material_confirmation(
            cleaned, question, [], force_confirmation=True
        )
        self.assertEqual(compact_without_history, "Exactly.")
        repaired = TutorAgent._enforce_controlled_reply(
            repeated, prepared, SimpleNamespace(messages=[]), question
        )
        self.assertEqual(repaired, "Exactly.")
        self.assertTrue(TutorAgent._is_material_confirmation_shape(
            "So in this clip, the order is only stated by the narrator, right?"
        ))
        self.assertFalse(_material_user_asks_language_help(
            "So in this clip, the order is only stated by the narrator, right?"
        ))

    def test_material_confirmation_and_creator_name_are_per_turn_context(self):
        ctx = SessionContext(session_id="material-creator-turn", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "multi": True,
            "sources": [{
                "material_id": 61,
                "title": "Control gameplay",
                "creator_display_name": "纯黑",
                "creator_display_name_source": "user_confirmed",
                "voice_source_spans": [{
                    "source_type": "creator_narration",
                    "confidence": "high",
                }],
            }],
        }
        agent = TutorAgent.__new__(TutorAgent)
        prompt = agent._build_control_prompt(ctx, {
            "constraints": {
                "material_tape": True,
                "material_confirmation_followup": True,
                "material": "[Material] The order is only stated in the source.",
            },
        }, user_transcript="So it is only stated in the voice-over, right?")
        self.assertIn("[Material Confirmation - this turn]", prompt)
        self.assertIn("不是英语表达求助", prompt)
        self.assertIn("[Material Creator Naming - this source]", prompt)
        self.assertIn("纯黑's voice-over/commentary", prompt)
        self.assertIn("同一事实只解释一次", prompt)

    def test_material_context_carries_verified_creator_display_name(self):
        ctx = SessionContext(session_id="material-creator", user_id="u",
                             target_lang="en")
        ctx.material_context = {
            "title": "Control gameplay",
            "kind": "link",
            "creator_display_name": "纯黑",
            "blocks": ["Pope sends the research team below the building."],
        }
        rendered = TutorAgent._material_text(ctx, "Who says Pope sent them down?")
        self.assertIn("发布者账号名：纯黑", rendered)
        self.assertIn("不单独证明声源身份", rendered)

    def test_elaboration_request_is_detected_without_treating_it_as_a_plain_repeat(self):
        self.assertTrue(TutorAgent._is_material_elaboration_request(
            "Can you elaborate a bit more on this point?"))
        self.assertTrue(TutorAgent._is_material_elaboration_request("能具体展开讲一讲这个点吗？"))
        self.assertFalse(TutorAgent._is_material_elaboration_request(
            "Can you say that exact sentence again?"))


class MaterialTopicCoverageTests(unittest.TestCase):
    def _context(self):
        ctx = SessionContext(session_id="material-topics", user_id="u",
                             target_lang="en", conversation_id="conv-1")
        ctx.material_context = {
            "multi": True,
            "summary": "A creator reviews newcomer recruitment and the market.",
            "sources": [{"material_id": 57, "title": "Newcomer review",
                         "summary": "The creator reviews several angles."}],
            "evidence_blocks": [
                {"material_id": 57, "block_no": 1, "span_kind": "body",
                 "text": "The creator says S1 took many standout newcomers."},
                {"material_id": 57, "block_no": 2, "span_kind": "body",
                 "text": "The creator says agencies shape which newcomers reach studios."},
                {"material_id": 57, "block_no": 3, "span_kind": "body",
                 "text": "The creator says a thinner talent pool may affect future recruitment."},
            ],
            "hooks": [
                {"text": "The creator's view of S1's newcomer lineup.",
                 "material_id": 57, "block_no": 1, "span_kind": "body"},
                {"text": "How agencies shape the flow of new talent.",
                 "material_id": 57, "block_no": 2, "span_kind": "body"},
                {"text": "What a thinner talent pool could mean for future recruitment.",
                 "material_id": 57, "block_no": 3, "span_kind": "body"},
            ],
        }
        return ctx

    def test_opening_and_completed_discovery_turn_advance_to_uncovered_angles(self):
        ctx = self._context()
        agent = TutorAgent.__new__(TutorAgent)
        agent._sessions = {ctx.session_id: ctx}
        agent.note_material_opening("u", "conv-1", "The opener about S1.",
                                    topic_refs=["57:1"])

        query = "What's the most useful point the video leaves us with?"
        first = TutorAgent._material_evidence_packet(ctx, query)
        self.assertEqual(first["topic_coverage"]["mode"], "discover")
        self.assertEqual(first["topic_coverage"]["fresh_refs"], ["57:2"])
        self.assertIn("57:2", first["refs"])
        self.assertNotIn("57:1", first["refs"],
                         "broad discovery should exclude the already introduced opening block")
        self.assertEqual(ctx.material_topic_coverage["introduced_refs"], ["57:1"])
        self.assertEqual(first["topic_turn"]["introduced_refs"], ["57:2"])
        self.assertTrue(first["topic_coverage"]["covered_topics"])
        self.assertIn("本轮新观点只能从证据包里新补充的正文片段", TutorAgent._material_text_multi(
            ctx.material_context, first))

        TutorAgent._apply_material_topic_turn(
            ctx, first["topic_turn"], "Agencies shape which newcomers reach studios.")
        second = TutorAgent._material_evidence_packet(ctx, query)
        self.assertEqual(second["topic_coverage"]["fresh_refs"], ["57:3"])
        self.assertIn("57:3", second["refs"])
        self.assertNotIn("57:1", second["refs"])
        self.assertNotIn("57:2", second["refs"])

    def test_broad_discovery_uses_fresh_body_without_reinjecting_whole_tape_summaries(self):
        ctx = self._context()
        ctx.material_context["groups"] = [{
            "title": "Overall analysis", "material_ids": [57],
            "summary": "WHOLE_GROUP_SUMMARY_SHOULD_NOT_STEER_DISCOVERY",
        }]
        TutorAgent._remember_material_topic_refs(
            ctx, introduced=["57:1", "57:2"])

        packet = TutorAgent._material_evidence_packet(
            ctx, "What's the most useful point the video leaves us with?")
        rendered = TutorAgent._material_text_multi(ctx.material_context, packet)

        self.assertEqual(packet["summary"], "")
        self.assertEqual(packet["topic_coverage"]["fresh_refs"], ["57:3"])
        self.assertEqual(packet["refs"], ["57:3"],
                         "broad discovery should inject one selected fresh angle")
        self.assertIn("[正文]", rendered)
        self.assertNotIn("素材57·第3段", rendered)
        self.assertIn("本轮新观点只能从证据包里新补充的正文片段", rendered)
        self.assertIn("不要只给抽象口号或判断标签", rendered)
        self.assertNotIn("A creator reviews newcomer recruitment and the market.", rendered)
        self.assertNotIn("The creator reviews several angles.", rendered)
        self.assertNotIn("WHOLE_GROUP_SUMMARY_SHOULD_NOT_STEER_DISCOVERY", rendered)

    def test_broad_discovery_uses_only_the_selected_hook_quote_from_a_long_block(self):
        ctx = self._context()
        ctx.material_context["evidence_blocks"][2]["text"] = (
            "The talent pool is thinner. S1 already takes most standout newcomers."
        )
        ctx.material_context["hooks"][2]["evidence_quote"] = (
            "The talent pool is thinner."
        )
        ctx.material_context["evidence_blocks"].append({
            "material_id": 57, "block_no": 4, "span_kind": "body",
            "text": "A separate later point in the source.",
        })
        TutorAgent._remember_material_topic_refs(
            ctx, introduced=["57:1", "57:2"])

        packet = TutorAgent._material_evidence_packet(
            ctx, "So what's the most useful point the video leaves us with?")
        rendered = TutorAgent._material_text_multi(ctx.material_context, packet)

        self.assertEqual(packet["topic_coverage"]["fresh_refs"], ["57:3"])
        self.assertEqual(packet["refs"], ["57:3"])
        self.assertIn("The talent pool is thinner.", rendered)
        self.assertNotIn("S1 already takes most standout newcomers", rendered)
        self.assertNotIn("A separate later point in the source", rendered)

    def test_specific_followup_can_retrieve_a_covered_angle(self):
        ctx = self._context()
        TutorAgent._remember_material_topic_refs(ctx, introduced=["57:1", "57:2"])

        packet = TutorAgent._material_evidence_packet(
            ctx, "What does the creator say about S1's newcomers?")

        self.assertEqual(packet["topic_coverage"]["mode"], "follow_user")
        self.assertIn("57:1", packet["refs"])
        self.assertIn("57:1", packet["topic_turn"]["discussed_refs"])

    def test_specific_entity_question_marks_a_related_suggested_angle(self):
        ctx = self._context()
        ctx.material_context["hooks"][2]["text"] = (
            "How S1's agency relationships shape the flow of new talent.")
        TutorAgent._remember_material_topic_refs(ctx, introduced=["57:1"])

        packet = TutorAgent._material_evidence_packet(ctx, "What about S1?")

        self.assertIn("57:3", packet["topic_turn"]["discussed_refs"])
        TutorAgent._apply_material_topic_turn(
            ctx, packet["topic_turn"], "The question is about S1's agency relationships.")
        broad = TutorAgent._material_evidence_packet(
            ctx, "What's the most useful point the video leaves us with?")
        self.assertEqual(broad["topic_coverage"]["fresh_refs"], ["57:2"])
        self.assertNotIn("57:3", broad["refs"])
        self.assertTrue(broad["topic_coverage"]["covered_topics"])

    def test_sparse_hooks_fall_back_to_uncovered_body_evidence(self):
        ctx = self._context()
        ctx.material_context["hooks"] = ctx.material_context["hooks"][:1]
        TutorAgent._remember_material_topic_refs(ctx, introduced=["57:1"])

        specific = TutorAgent._material_evidence_packet(
            ctx, "What does the source say about agencies?")
        self.assertIn("57:2", specific["topic_turn"]["discussed_refs"])
        TutorAgent._apply_material_topic_turn(
            ctx, specific["topic_turn"], "Agencies influence which talent reaches studios.")

        broad = TutorAgent._material_evidence_packet(
            ctx, "What's the most useful point the video leaves us with?")
        self.assertEqual(broad["topic_coverage"]["fresh_refs"], ["57:3"])
        self.assertIn("57:3", broad["refs"])

    def test_topic_reference_updates_are_bounded_and_idempotent(self):
        ctx = self._context()
        TutorAgent._remember_material_topic_refs(
            ctx, introduced=["57:1", "57:1", "bad"], discussed=["57:2"])
        TutorAgent._remember_material_topic_refs(ctx, introduced=["57:1"])
        self.assertEqual(ctx.material_topic_coverage["introduced_refs"], ["57:1"])
        self.assertEqual(ctx.material_topic_coverage["discussed_refs"], ["57:2"])
        TutorAgent._apply_material_topic_turn(ctx, {"introduced_refs": ["57:3"]}, "")
        self.assertNotIn("57:3", ctx.material_topic_coverage["introduced_refs"])

    def test_broad_discovery_query_detection_covers_takeaway_wording(self):
        self.assertTrue(TutorAgent._is_material_discovery_query(
            "So what's the main takeaway?") )
        self.assertTrue(TutorAgent._is_material_discovery_query("还有什么值得讨论？"))
        self.assertFalse(TutorAgent._is_material_discovery_query(
            "Can you explain the agency point in more detail?"))


class MaterialTapePlanSourceAndLanguageTests(unittest.TestCase):
    def test_tape_plan_marks_missing_video_transcript_and_demotes_comments(self):
        prompt = material_analyzer._tape_plan_prompt([{
            "material_id": 8,
            "title": "2026 first-half newcomer review",
            "kind": "video",
            "summary": "视频正文未转写。",
            "source_coverage": {
                "kind": "video",
                "body_available": False,
                "scope": "page_and_comments_only",
            },
            "blocks": [
                {"block_no": 1, "span_kind": "body",
                 "text": "页面标题与视频简介。"},
                {"block_no": 2, "span_kind": "comment",
                 "text": "一位观众认为X是MVP。"},
            ],
        }])
        self.assertIn("没有取得视频正文转写", prompt)
        self.assertIn("不得据此归纳UP主观点", prompt)
        self.assertIn("评论只能作为次要的个体反应", prompt)

    def test_tape_plan_prompt_keeps_body_and_comment_sources_distinct(self):
        blocks = ([{"block_no": i, "span_kind": "body", "text": "正文块%d" % i}
                   for i in range(1, 8)]
                  + [{"block_no": 8, "span_kind": "comment", "text": "评论样本一"},
                     {"block_no": 9, "span_kind": "comment", "text": "评论样本二"}])
        prompt = material_analyzer._tape_plan_prompt([{
            "material_id": 3, "title": "中文视频", "kind": "video",
            "summary": "中文提要", "blocks": blocks,
        }])
        self.assertIn("[1·正文]", prompt)
        self.assertIn("[8·评论区样本] 评论样本一", prompt)
        self.assertIn("[9·评论区样本] 评论样本二", prompt)
        self.assertIn("不说成观众普遍看法", prompt)
        self.assertIn("纯中文正文或没有合适英语表达时返回空数组", prompt)
        self.assertIn("[7·正文] 正文块7", prompt,
                      "有限块预算应覆盖正文尾部，避免只读开头")
        self.assertNotIn("[2·正文]", prompt,
                         "正文块应按全文位置抽样，而非只取前几块")
        self.assertIn("每份素材最多储备 2 个", prompt)
        self.assertIn("why_worth_discussing", prompt)
        self.assertIn("具体张力、取舍、因果或未解问题", prompt)

    def test_tape_plan_source_and_output_budgets_scale_to_eight_sources(self):
        self.assertEqual(material_analyzer._tape_plan_output_tokens(1), 1350)
        self.assertGreater(material_analyzer._tape_plan_output_tokens(2),
                           material_analyzer._tape_plan_output_tokens(1))
        self.assertEqual(material_analyzer._tape_plan_output_tokens(8), 4500)
        with self.assertRaises(ValueError):
            material_analyzer._tape_plan_output_tokens(9)
        prompt = material_analyzer._tape_plan_prompt([
            {"material_id": mid, "title": "Source %d" % mid,
             "summary": "Summary %d" % mid,
             "blocks": [{"block_no": 1, "span_kind": "body",
                         "text": "Evidence from source %d." % mid}]}
            for mid in range(1, 9)
        ])
        for mid in range(1, 9):
            self.assertIn("素材 id=%d" % mid, prompt)

    def test_validate_tape_plan_rejects_non_english_and_comment_chunks(self):
        sources = [{
            "material_id": 4, "title": "Mixed video", "blocks": [
                {"block_no": 1, "span_kind": "body",
                 "text": "The host says this feels like home after a long day."},
                {"block_no": 2, "span_kind": "body",
                 "text": "中文句子：这个菜很下饭。"},
                {"block_no": 3, "span_kind": "comment",
                 "text": "I really like the camera work."},
            ],
        }]
        result = material_analyzer.validate_tape_plan({
            "summary": "A mixed-language source.",
            "groups": [{"title": "Mixed video", "material_ids": [4]}],
            "hooks": [{"text": "A viewer likes the camera work.",
                       "why_worth_discussing": "This is one commenter's view, not a broader finding.",
                       "material_id": 4, "block_no": 3,
                       "evidence_quote": "I really like the camera work."}],
            "top_chunks": [
                {"text": "feels like home", "material_id": 4, "block_no": 1,
                 "quote": "The host says this feels like home after a long day."},
                {"text": "很下饭", "material_id": 4, "block_no": 2,
                 "quote": "中文句子：这个菜很下饭。"},
                {"text": "like the camera work", "material_id": 4, "block_no": 3,
                 "quote": "I really like the camera work."},
            ],
        }, sources)
        self.assertEqual(result["hooks"][0]["span_kind"], "comment")
        self.assertEqual([item["text"] for item in result["top_chunks"]],
                         ["feels like home"])

    def test_validate_tape_plan_keeps_multiple_supported_angles(self):
        sources = [{
            "material_id": 7, "title": "Interview", "blocks": [
                {"block_no": 1, "span_kind": "body",
                 "text": "The team focused on one product goal."},
                {"block_no": 2, "span_kind": "comment",
                 "text": "Some viewers questioned the source."},
            ],
        }]
        result = material_analyzer.validate_tape_plan({
            "groups": [{"title": "Interview", "material_ids": [7]}],
            "hooks": [
                {"text": "Was focus the reason for progress?",
                 "why_worth_discussing": "It asks whether the stated goal explains the result.",
                 "material_id": 7,
                 "block_no": 1, "evidence_quote": "focused on one product goal"},
                {"text": "What made viewers question the source?",
                 "why_worth_discussing": "This is a source-authenticity question raised in a comment.",
                 "material_id": 7,
                 "block_no": 2, "evidence_quote": "Some viewers questioned the source."},
                {"text": "Unsupported angle",
                 "why_worth_discussing": "This angle has no evidence in its cited block.",
                 "material_id": 7,
                 "block_no": 1, "evidence_quote": "unrelated words"},
                {"text": "A third supported angle",
                 "why_worth_discussing": "This would exceed the per-source angle budget.",
                 "material_id": 7, "block_no": 1,
                 "evidence_quote": "The team focused on one product goal."},
                {"text": "Angle with no analysis field", "material_id": 7,
                 "block_no": 1,
                 "evidence_quote": "The team focused on one product goal."},
            ],
        }, sources)
        self.assertEqual(len(result["hooks"]), 2)
        self.assertEqual([hook["span_kind"] for hook in result["hooks"]],
                         ["body", "comment"])
        self.assertTrue(all(hook["why_worth_discussing"] for hook in result["hooks"]))

    def test_fallback_opening_hook_uses_body_before_comment(self):
        result = material_analyzer._tape_plan_fallback([{
            "material_id": 9, "title": "Two spans", "summary": "Summary",
            "blocks": [
                {"block_no": 1, "span_kind": "comment",
                 "text": "Why is the host wrong about this?"},
                {"block_no": 2, "span_kind": "body",
                 "text": "What does the speaker think changed?"},
            ],
        }])
        self.assertEqual(result["hooks"][0]["block_no"], 2)

    def test_fallback_does_not_invent_a_discussion_hook_without_a_source_question(self):
        result = material_analyzer._tape_plan_fallback([{
            "material_id": 10, "title": "A factual report", "summary": "Summary",
            "blocks": [{"block_no": 4, "span_kind": "body",
                        "text": "The report compares the two projects by cost and duration."}],
        }])
        self.assertEqual(result["groups"][0]["material_ids"], [10])
        self.assertEqual(result["hooks"], [])

    def test_material_prompt_only_includes_hooks_with_evidence_in_this_packet(self):
        info = {
            "multi": True,
            "summary": "Two different sources.",
            "sources": [
                {"material_id": 11, "title": "Source A"},
                {"material_id": 12, "title": "Source B"},
            ],
            "hooks": [
                {"text": "Does the claimed tradeoff follow from the example?",
                 "why_worth_discussing": "The example may not establish the broader claim.",
                 "material_id": 11, "block_no": 3,
                 "evidence_quote": "The speaker describes one case.",
                 "span_kind": "body"},
                {"text": "Would the second source change that conclusion?",
                 "why_worth_discussing": "This source is unrelated and was not selected for this turn.",
                 "material_id": 12, "block_no": 8,
                 "evidence_quote": "Another report describes a different case.",
                 "span_kind": "body"},
            ],
        }
        packet = {
            "sources": info["sources"], "summary": info["summary"],
            "groups": [], "hooks": info["hooks"],
            "refs": ["11:3"],
            "lines": ["[素材11·第3段] The speaker describes one case."],
            "covered": 1, "total_blocks": 2,
        }
        rendered = TutorAgent._material_text_multi(info, packet)
        self.assertIn("Does the claimed tradeoff follow from the example?", rendered)
        self.assertIn("The example may not establish the broader claim.", rendered)
        self.assertNotIn("Would the second source change that conclusion?", rendered)
        self.assertNotIn("This source is unrelated", rendered)

        packet["refs"] = ["11:99"]
        rendered_without_matching_evidence = TutorAgent._material_text_multi(info, packet)
        self.assertNotIn("可以聊的角度", rendered_without_matching_evidence)

    def test_validate_tape_plan_preserves_source_ownership_at_eight_materials(self):
        sources = [
            {"material_id": mid, "title": "Source %d" % mid,
             "blocks": [{"block_no": 1, "span_kind": "body",
                         "text": "Source %d presents a distinct claim." % mid},
                        {"block_no": 2, "span_kind": "body",
                         "text": "Source %d gives a separate example." % mid}]}
            for mid in range(1, 9)
        ]
        payload = {
            "groups": [{"title": "Source %d" % mid, "material_ids": [mid]}
                       for mid in range(1, 9)],
            "hooks": [
                {"text": "What does source %d's claim establish?" % mid,
                 "why_worth_discussing": "It separates the claim from the evidence offered.",
                 "material_id": mid, "block_no": 1,
                 "evidence_quote": "Source %d presents a distinct claim." % mid}
                for mid in range(1, 9)
            ] + [
                {"text": "What does source %d's example add?" % mid,
                 "why_worth_discussing": "The example may narrow the claim's scope.",
                 "material_id": mid, "block_no": 2,
                 "evidence_quote": "Source %d gives a separate example." % mid}
                for mid in range(1, 9)
            ],
        }
        result = material_analyzer.validate_tape_plan(payload, sources)
        self.assertEqual({mid for group in result["groups"]
                          for mid in group["material_ids"]}, set(range(1, 9)))
        self.assertEqual(len(result["hooks"]), 16)
        for hook in result["hooks"]:
            source_text = next(source for source in sources
                               if source["material_id"] == hook["material_id"])
            block = next(block for block in source_text["blocks"]
                         if block["block_no"] == hook["block_no"])
            self.assertIn(hook["evidence_quote"], block["text"])

    def test_material_repeat_challenge_rejects_repeated_full_explanation(self):
        previous = (
            "Yes — there is one more thing, and it's about money. In [素材25·第4段] "
            "he says the food there is the most expensive he's hit in Asia, even more than Hong Kong, "
            "and a plain curry dish cost him around sixty or seventy. He also mentions a McDonald's "
            "cone going for almost a hundred, which honestly surprised me — I'd have guessed fast food "
            "would be the cheap option."
        )
        repeated = (
            "Yes, I'm sure — that part is in [素材25·第4段], where he says the food is the most "
            "expensive he's hit in Asia, even more than Hong Kong. The curry for around sixty or "
            "seventy and the McDonald's cone near a hundred are both from that same segment, so I'm "
            "not making it up."
        )
        prepared = {"constraints": {
            "material_tape": True,
            "recent_dialogue": [{"role": "assistant", "content": previous}],
            "conversation_director": {"language_presentation": "target_only"},
        }}
        violations = TutorAgent._controlled_reply_violations(
            repeated, prepared,
            user_transcript="We talked about this thing, right? Are you sure?",
        )
        self.assertTrue(violations, violations)

    def test_material_elaboration_rejects_replaying_the_previous_answer(self):
        previous = (
            "In [素材25·第4段], he compares Singapore with Hong Kong and says the food was the "
            "priciest he had found in Asia. A curry cost around sixty or seventy, and a McDonald's "
            "cone was close to a hundred. The cone surprised me because fast food is usually expected "
            "to be the cheaper choice."
        )
        prepared = {"constraints": {
            "material_tape": True,
            "recent_dialogue": [{"role": "assistant", "content": previous}],
            "conversation_director": {"language_presentation": "target_only"},
        }}
        violations = TutorAgent._controlled_reply_violations(
            "Sure. " + previous,
            prepared,
            user_transcript="Can you elaborate a bit more on this point?",
            previous_reply=previous,
        )
        self.assertTrue(violations, violations)

    def test_duplicate_question_is_rejected_inside_one_material_reply(self):
        question = "Do you think fast food is the only cheaper option in a place like that?"
        self.assertTrue(TutorAgent._has_repeated_reply_loop(question + " " + question))


# 12 个各带独立人名的小节 → 长到能切出多块，且"问谁"能唯一命中一块
_SECTIONS = (
    ("瑞克", "他在亚特兰大郊外醒来，发现医院空无一人，于是骑车回家找妻儿。"),
    ("肖恩", "他和瑞克的妻子洛莉一起带着卡尔逃到城外营地，负责维持秩序。"),
    ("洛莉", "她在营地里照顾卡尔，同时为丈夫的生死不明而煎熬。"),
    ("卡尔", "这个男孩在营地边缘被丧尸咬伤，所幸只是擦伤，从此学会用枪。"),
    ("达里尔", "他靠弩和狩猎为生，起初不合群，后来成为最可靠的猎手。"),
    ("格伦", "他熟悉城市街道，负责进城搜刮物资，还救过一个被困的陌生人。"),
    ("玛姬", "她住在农场，父亲赫谢尔经营兽医诊所，后来收留了整个团队。"),
    ("米琼恩", "她带着一把武士刀和两个被链子牵着的丧尸独自行走，最终加入团队。"),
    ("尼根", "他领导救世军，用铁丝缠绕的球棒露西尔威慑其他社区，要求进贡。"),
    ("总督", "他统治伍德伯里小镇，把丧尸头养在鱼缸里，靠谎言维持统治。"),
    ("赫谢尔", "这位老兽医失去了一条腿，却仍在农场里为伤员做手术。"),
    ("萨莎", "她失去了哥哥和弟弟，从狙击手成长为团队里最冷静的观察者。"),
)
_FILLER = (
    "营地的日常包括找水、修车、轮流守夜和清点弹药，每个人都要轮班。"
    "他们用废弃的汽车堵住公路入口，在围栏上挂铃铛当作警报。"
    "夜里能听见远处丧尸的呻吟，孩子们被要求不许单独离开围栏。"
    "食物按人头分配，药品要登记，谁受伤谁就排在后面等。"
    "白天有人在瞭望塔上放哨，用镜子反光给外出的人报平安。"
)
LONG_BODY = "\n".join(
    "第%d节 %s。%s %s" % (index, who, what, _FILLER * 3)
    for index, (who, what) in enumerate(_SECTIONS, 1)
)

# 深析六字段按提示词允许的最大量（每条 ≤40 字、每字段 ≤8 条）
MAX_DEEP = {
    field: ["第%d条分析内容大概四十个字左右的一段话" % i for i in range(1, 9)]
    for field in ("topics", "people", "timeline", "viewpoints", "hooks", "gaps")
}


def _blocks(raw_text: str = LONG_BODY) -> list:
    return material_analyzer.split_blocks(raw_text)


def _material_ctx(blocks: list, *, summary: str = "提要句。", analysis=None):
    ctx = SessionContext(session_id="m", user_id="u", target_lang="en")
    ctx.material_context = {
        "title": "行尸走肉", "kind": "link", "summary": summary,
        "blocks": blocks, "analysis": analysis or {},
    }
    return ctx


def _injected_block_numbers(text: str) -> list:
    """注入文本里的正文块号（原文序号）。"""
    numbers = []
    for line in text.split("\n"):
        if line.startswith("正文片段"):
            numbers.append(int(line.split("：", 1)[0].replace("正文片段", "")))
    return numbers


class QueryTokenTests(unittest.TestCase):
    def test_english_content_words_kept_function_words_dropped(self):
        tokens = material_analyzer.query_tokens("What did Rick do about the farm?")
        self.assertIn("rick", tokens)
        self.assertIn("farm", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("the", tokens)

    def test_chinese_bigrams(self):
        tokens = material_analyzer.query_tokens("尼根是谁？")
        self.assertIn("尼根", tokens)
        self.assertNotIn("是谁", tokens, "疑问词不进词元集")


class ScoreBlockTests(unittest.TestCase):
    def test_score_is_zero_without_query(self):
        self.assertEqual(material_analyzer.score_blocks(["abc"], ""), [0.0])

    def test_hit_block_scores_higher(self):
        blocks = ["尼根用露西尔威慑社区", "玛姬住在农场"]
        scores = material_analyzer.score_blocks(blocks, "尼根是谁")
        self.assertGreater(scores[0], scores[1])


class SelectBlockTests(unittest.TestCase):
    def setUp(self):
        self.blocks = _blocks()
        self.assertGreater(len(self.blocks), 3, "夹具要能切出多块才测得出选择")

    def _block_of(self, who: str) -> int:
        """该人名所在的块**下标（0 基）**——select_blocks 返回的也是 0 基。"""
        for index, block in enumerate(self.blocks):
            if who in block:
                return index
        self.fail("夹具里没有 %s" % who)

    def test_asked_about_term_gets_its_block(self):
        for who in ("尼根", "萨莎", "达里尔", "玛姬", "总督"):
            picked = material_analyzer.select_blocks(self.blocks, who + "是谁", 3)
            self.assertIn(self._block_of(who), picked,
                          "问 %s 时必须给出它所在的块" % who)

    def test_result_is_in_document_order(self):
        picked = material_analyzer.select_blocks(self.blocks, "萨莎和瑞克", 3)
        self.assertEqual(picked, sorted(picked), "呈现顺序按原文，不按得分")

    def test_uninformative_query_falls_back_to_leading_blocks(self):
        for query in ("嗯", "接着聊吧", ""):
            picked = material_analyzer.select_blocks(self.blocks, query, 3)
            self.assertEqual(picked, [0, 1, 2], "无从判断时退回开头块")

    def test_limit_zero_returns_empty(self):
        self.assertEqual(material_analyzer.select_blocks(self.blocks, "尼根", 0), [])

    def test_short_material_returns_all(self):
        short = ["只有一块"]
        self.assertEqual(material_analyzer.select_blocks(short, "任何", 3), [0])


class MaterialInjectionTests(unittest.TestCase):
    def test_non_material_tape_is_empty(self):
        ctx = SessionContext(session_id="m", user_id="u", target_lang="en")
        self.assertEqual(TutorAgent._material_text(ctx, "随便聊"), "")

    def test_query_selects_relevant_block(self):
        blocks = _blocks()
        target = next(index for index, block in enumerate(blocks, 1)
                      if "尼根" in block)
        text = TutorAgent._material_text(_material_ctx(blocks), "尼根是谁？")
        self.assertIn(target, _injected_block_numbers(text),
                      "改动前这里永远只有第 1 块")

    def test_within_budget(self):
        text = TutorAgent._material_text(
            _material_ctx(_blocks(), summary="提要。" * 200, analysis=MAX_DEEP),
            "尼根是谁")
        self.assertLessEqual(len(text), TutorAgent.MATERIAL_INJECT_MAX_CHARS)

    def test_body_survives_max_deep_analysis(self):
        """谈资不得把正文挤光——改动前满额深析会让正文块数变成 0，
        模型手里只剩摘要和自己的推断（"对着自己的分析聊"）。"""
        text = TutorAgent._material_text(
            _material_ctx(_blocks(), summary="提要。" * 200, analysis=MAX_DEEP),
            "尼根是谁")
        self.assertTrue(_injected_block_numbers(text),
                        "任何情况下都必须有原文块进 prompt")

    def test_deep_and_summary_are_capped(self):
        text = TutorAgent._material_text(
            _material_ctx(_blocks(), summary="提要。" * 200, analysis=MAX_DEEP),
            "尼根")
        deep = TutorAgent._material_analysis_text(MAX_DEEP)
        self.assertGreater(len(deep), TutorAgent.MATERIAL_DEEP_MAX_CHARS)
        self.assertLessEqual(len(deep[:TutorAgent.MATERIAL_DEEP_MAX_CHARS]),
                             TutorAgent.MATERIAL_DEEP_MAX_CHARS)
        self.assertIn("[Material·分析]", text)

    def test_subset_is_announced(self):
        text = TutorAgent._material_text(_material_ctx(_blocks()), "尼根是谁")
        self.assertIn("没被挑中的段落不代表素材里没有", text)

    def test_no_notice_when_every_block_fits(self):
        text = TutorAgent._material_text(_material_ctx(["只有一块正文"]), "任何")
        self.assertNotIn("没被挑中的段落不代表素材里没有", text)

    def test_summary_is_marked_as_paraphrase(self):
        text = TutorAgent._material_text(_material_ctx(_blocks()), "尼根")
        self.assertIn("概括，不是原文", text,
                      "提要是模型写的，不点明会被当成素材原话引用")

    def test_block_numbers_use_source_order(self):
        """块号必须是原文序号：模型据此知道自己在读长文的第几段，
        才说得出"我看到的是第 3、7 段"，而不是笼统断言"素材里没写"。"""
        blocks = _blocks()
        text = TutorAgent._material_text(_material_ctx(blocks), "萨莎是谁")
        numbers = _injected_block_numbers(text)
        self.assertEqual(numbers, sorted(numbers))
        self.assertTrue(all(1 <= number <= len(blocks) for number in numbers))

    def test_own_view_instruction_is_present(self):
        """需求 b：对素材要有自己的看法，并且讲出来、可不同意用户。"""
        text = TutorAgent._material_text(
            _material_ctx(_blocks(), analysis={"viewpoints": ["尼根是个暴君"]}),
            "尼根")
        self.assertIn("也可以直接把你的看法说出来", text)


class ConstraintsDedupTests(unittest.TestCase):
    def test_rendered_sections_are_stripped(self):
        stripped = _constraints_for_prompt(
            {"material": "很长的素材段", "scene_world": "场景世界",
             "user_must_speak": False})
        self.assertNotIn("material", stripped)
        self.assertNotIn("scene_world", stripped)
        self.assertEqual(stripped, {"user_must_speak": False})

    def test_other_keys_untouched(self):
        original = {"max_teaching_acts": 1, "turn_contract": {"a": 1}}
        self.assertEqual(_constraints_for_prompt(original), original)

    def test_non_dict_is_empty(self):
        self.assertEqual(_constraints_for_prompt(None), {})
        self.assertEqual(_constraints_for_prompt([1, 2]), {})


# ── 全文覆盖：分段多轮分析（2026-09-21）──────────────────────────────────
#
# 原实现只把**前 12,000 字**送进模型，长素材的后段从没被读过（"知道个大概"）。
# 用户口径："要彻底了解到这个视频的内容，而不是这种囫囵吞枣"，且接受多花时间。

class WindowTests(unittest.TestCase):
    def test_short_text_is_one_window(self):
        self.assertEqual(material_analyzer._windows("短素材", 12000), ["短素材"])

    def test_long_text_splits_and_covers_everything(self):
        text = "\n".join("第%d段。" % i + "正文。" * 200 for i in range(1, 9))
        windows = material_analyzer._windows(text, 2000)
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(len(w) <= 2000 for w in windows))
        # 每段都还在（不丢内容）——逐段找一个特征串
        joined = "".join(windows)
        for i in range(1, 9):
            self.assertIn("第%d段。" % i, joined)

    def test_overlong_paragraph_still_bounded(self):
        windows = material_analyzer._windows("啊" * 5000, 1000)
        self.assertTrue(all(len(w) <= 1000 for w in windows))
        self.assertEqual(sum(len(w) for w in windows), 5000)


class MergeDeepTests(unittest.TestCase):
    def _parts(self, count: int) -> list:
        return [{"title": "标题%d" % i,
                 "topics": ["窗口%d主题" % i],
                 "viewpoints": ["窗口%d观点" % i],
                 "gaps": ["窗口%d悬念" % i]} for i in range(1, count + 1)]

    def test_every_window_contributes(self):
        """轮转取件：每个窗口都进得来。从头截断会让产物全来自素材开头。"""
        merged = material_analyzer._merge_deep(self._parts(5))
        self.assertEqual(len(merged["topics"]), 5)
        self.assertIn("窗口5主题", merged["topics"])

    def test_respects_field_cap(self):
        parts = [{"topics": ["窗口%d主题%d" % (w, i) for i in range(1, 9)]}
                 for w in range(1, 4)]
        merged = material_analyzer._merge_deep(parts)
        self.assertLessEqual(len(merged["topics"]),
                             material_analyzer._DEEP_LIST_MAX)

    def test_dedupes(self):
        merged = material_analyzer._merge_deep(
            [{"topics": ["同一条"]}, {"topics": ["同一条"]}])
        self.assertEqual(merged["topics"], ["同一条"])

    def test_title_taken_from_first_window_that_has_one(self):
        merged = material_analyzer._merge_deep([{}, {"title": "有标题"}])
        self.assertEqual(merged["title"], "有标题")

    def test_empty_parts_is_empty(self):
        self.assertEqual(material_analyzer._merge_deep([]), {})

    def test_source_map_requires_exact_body_quotes_and_derives_video_time(self):
        transcript = (
            "[speaker=speaker_0; time_ms=1000-1800] Dylan Faden is below.\n"
            "[speaker=speaker_1; time_ms=1900-2700] Jesse, I'm still looking for you.\n"
        )
        payload = {
            "voice_source_spans": [{
                "start_ms": 1000, "end_ms": 1800,
                "source_type": "in_material_dialogue", "confidence": "high",
                "evidence": "Dylan Faden is below.",
            }],
            "source_map": {
                "entities": [{
                    "name": "Dylan", "aliases": [{
                        "name": "Dylan Faden",
                        "evidence": "Dylan Faden is below.",
                    }],
                    "role": "The person Zoe is trying to reach.",
                    "evidence": "Dylan Faden is below.",
                }],
                "beats": [{
                    "order": 2, "event": "Dylan continues looking for Jesse.",
                    "effect": "His search remains open.",
                    "evidence": "Jesse, I'm still looking for you.",
                    "start_ms": 999999,
                }],
                "open_threads": [{
                    "question": "Will Dylan find Jesse?",
                    "evidence": "Jesse, I'm still looking for you.",
                }],
            },
        }
        parsed = material_analyzer._parse_deep_json(
            json.dumps(payload), source_text=transcript)
        source_map = parsed["source_map"]
        self.assertTrue(source_map["validated"])
        self.assertEqual(source_map["entities"][0]["aliases"][0]["name"],
                         "Dylan Faden")
        self.assertEqual(source_map["entities"][0]["evidence"]["start_ms"], 1000)
        self.assertEqual(source_map["entities"][0]["evidence"]["source_type"],
                         "in_material_dialogue")
        self.assertEqual(source_map["beats"][0]["evidence"]["start_ms"], 1900)
        self.assertNotEqual(source_map["beats"][0]["evidence"]["start_ms"], 999999)

    def test_source_map_rejects_comment_only_quotes(self):
        body = "[speaker=speaker_0; time_ms=100-500] Dylan is still searching."
        combined = body + material_analyzer.COMMENT_MARKER + "Hiss theory from viewer comments."
        payload = {
            "source_map": {
                "entities": [{
                    "name": "Hiss", "aliases": [], "role": "A viewer theory.",
                    "evidence": "Hiss theory from viewer comments.",
                }],
                "beats": [{
                    "order": 1, "event": "The Hiss is involved.",
                    "effect": "It explains the crisis.",
                    "evidence": "Hiss theory from viewer comments.",
                }],
                "open_threads": [],
            },
        }
        parsed = material_analyzer._parse_deep_json(
            json.dumps(payload), source_text=combined)
        self.assertNotIn("source_map", parsed)

    def test_source_map_uses_existing_validated_voice_spans_when_body_has_no_tags(self):
        body = "Dylan tells Zoe that the research team is not where he expected."
        known_spans = [{
            "start_ms": 12000, "end_ms": 16000,
            "source_type": "in_material_dialogue", "confidence": "high",
            "evidence": "Dylan tells Zoe that the research team is not where he expected.",
        }]
        payload = {"source_map": {"entities": [], "beats": [{
            "order": 1, "event": "Dylan reaches the team's equipment but not the team.",
            "effect": "The search must continue.",
            "evidence": "the research team is not where he expected",
        }], "open_threads": []}}
        parsed = material_analyzer._parse_deep_json(
            json.dumps(payload), source_text=body,
            known_voice_source_spans=known_spans)
        proof = parsed["source_map"]["beats"][0]["evidence"]
        self.assertEqual(proof["start_ms"], 12000)
        self.assertEqual(proof["end_ms"], 16000)
        self.assertEqual(proof["source_type"], "in_material_dialogue")

    def test_source_map_does_not_accept_alias_without_same_quote_link(self):
        transcript = (
            "Dylan goes below.\n"
            "迪伦听见Zoe在无线电里说话。\n"
        )
        payload = {"source_map": {"entities": [{
            "name": "Dylan", "aliases": [{
                "name": "迪伦", "evidence": "迪伦听见Zoe在无线电里说话。",
            }], "role": "The person who goes below.",
            "evidence": "Dylan goes below.",
        }], "beats": [], "open_threads": []}}
        parsed = material_analyzer._parse_deep_json(
            json.dumps(payload), source_text=transcript)
        self.assertEqual(parsed["source_map"]["entities"][0]["aliases"], [])

    def test_source_map_merge_keeps_window_sequence_when_some_events_are_untimed(self):
        def beat(order, event, quote, start_ms=None):
            evidence = {"quote": quote, "source_type": "body"}
            if start_ms is not None:
                evidence["start_ms"] = start_ms
            return {"order": order, "event": event, "effect": "", "evidence": evidence}

        merged = material_analyzer._merge_source_maps([
            {"source_map": {"validated": True, "beats": [
                beat(1, "first", "first event quote", 1000),
                beat(2, "second", "second event quote"),
            ]}},
            {"source_map": {"validated": True, "beats": [
                beat(1, "third", "third event quote", 3000),
                beat(2, "fourth", "fourth event quote"),
            ]}},
        ])
        self.assertEqual([item["event"] for item in merged["beats"]],
                         ["first", "second", "third", "fourth"])
        self.assertEqual([item["order"] for item in merged["beats"]], [1, 2, 3, 4])

    def test_voice_sources_merge_by_speaker_and_keep_one_consensus(self):
        merged = material_analyzer._merge_deep([
            {"voice_sources": [{
                "speaker_id": "speaker_0", "source_type": "creator_narration",
                "confidence": "high", "evidence": "Let me explain the map.",
            }]},
            {"voice_sources": [{
                "speaker_id": "speaker_0", "source_type": "creator_narration",
                "confidence": "medium", "evidence": "I will go this way.",
            }, {
                "speaker_id": "speaker_1", "source_type": "in_material_dialogue",
                "confidence": "high", "evidence": "Zoe, I think I found it.",
            }]},
        ])
        self.assertEqual(len(merged["voice_sources"]), 2)
        narrator = merged["voice_sources"][0]
        self.assertEqual(narrator["speaker_id"], "speaker_0")
        self.assertEqual(narrator["source_type"], "creator_narration")
        self.assertEqual(narrator["confidence"], "medium")
        self.assertEqual(narrator["evidence"], [
            "Let me explain the map.", "I will go this way.",
        ])

    def test_voice_source_disagreement_becomes_uncertain_not_a_false_consensus(self):
        merged = material_analyzer._merge_deep([
            {"voice_sources": [{
                "speaker_id": "speaker_1", "source_type": "in_material_dialogue",
                "confidence": "high", "evidence": "Zoe, I found something.",
            }]},
            {"voice_sources": [{
                "speaker_id": "speaker_1", "source_type": "quoted_or_recorded_audio",
                "confidence": "medium", "evidence": "Recorded message begins.",
            }]},
        ])
        source = merged["voice_sources"][0]
        self.assertEqual(source["source_type"], "uncertain")
        self.assertEqual(source["confidence"], "low")
        self.assertEqual(source["alternatives"], [
            "in_material_dialogue", "quoted_or_recorded_audio",
        ])
        self.assertIn("判断不一致", material_analyzer.format_voice_source(source))

    def test_legacy_voice_source_strings_still_normalize(self):
        parsed = material_analyzer._parse_deep_json(
            '{"voice_sources":["speaker=0：UP主旁白，置信度高；证据：Let me explain."]}')
        self.assertEqual(parsed["voice_sources"][0]["speaker_id"], "speaker_0")
        self.assertEqual(parsed["voice_sources"][0]["source_type"], "creator_narration")
        self.assertEqual(parsed["voice_sources"][0]["confidence"], "high")

    def test_global_voice_evidence_is_bounded_and_spans_the_full_source(self):
        lines = ["[speaker=0; time_ms=%d-%d] utterance_%d" %
                 (i * 100, i * 100 + 80, i) for i in range(20)]
        evidence = material_analyzer._voice_source_evidence("\n".join(lines))
        self.assertLessEqual(evidence.count("utterance_"), 6)
        self.assertIn("speaker_0", evidence)
        self.assertIn("utterance_0", evidence)
        self.assertIn("utterance_19", evidence)

    def test_same_asr_id_can_have_multiple_supported_voice_sources(self):
        parsed = material_analyzer._parse_deep_json(
            '{"voice_sources":['
            '{"speaker_id":"speaker_0","source_type":"creator_narration",'
            '"confidence":"medium","evidence":"I will go this way",'
            '"alternatives":["in_material_dialogue"]},'
            '{"speaker_id":"speaker_0","source_type":"in_material_dialogue",'
            '"confidence":"medium","evidence":"Remember your father",'
            '"alternatives":["creator_narration"]}]}')
        rows = parsed["voice_sources"]
        self.assertEqual([row["source_type"] for row in rows], [
            "creator_narration", "in_material_dialogue",
        ])
        self.assertTrue(all("alternatives" not in row for row in rows))
        self.assertTrue(all("判断不一致" not in material_analyzer.format_voice_source(row)
                            for row in rows))

    def test_voice_source_spans_are_time_and_quote_validated(self):
        transcript = (
            "[speaker=speaker_0; time_ms=100-500] I can explain the map.\n"
            "[speaker=speaker_0; time_ms=600-900] Jesse, I'm here.\n"
        )
        parsed = material_analyzer._parse_deep_json(
            '{"voice_source_spans":['
            '{"start_ms":100,"end_ms":500,"source_type":"creator_narration",'
            '"confidence":"high","evidence":"I can explain the map."},'
            '{"start_ms":600,"end_ms":900,"source_type":"in_material_dialogue",'
            '"confidence":"medium","evidence":"Jesse, I\'m here."},'
            '{"start_ms":101,"end_ms":500,"source_type":"creator_narration",'
            '"confidence":"high","evidence":"I can explain the map."},'
            '{"start_ms":100,"end_ms":500,"source_type":"creator_narration",'
            '"confidence":"high","evidence":"not in this line"}]}',
            source_text=transcript)
        rows = parsed["voice_source_spans"]
        self.assertEqual([(row["start_ms"], row["end_ms"], row["source_type"])
                          for row in rows], [
                              (100, 500, "creator_narration"),
                              (600, 900, "in_material_dialogue"),
                          ])
        self.assertIn("100-500ms", material_analyzer.format_voice_source_span(rows[0]))
        self.assertIn("创作者旁白/解说", material_analyzer.format_voice_source_span(rows[0]))

    def test_voice_source_spans_merge_by_time_not_speaker_id(self):
        merged = material_analyzer._merge_deep([
            {"voice_source_spans": [{
                "start_ms": 100, "end_ms": 500,
                "source_type": "creator_narration", "confidence": "high",
                "evidence": "I can explain the map.",
            }]},
            {"voice_source_spans": [{
                "start_ms": 600, "end_ms": 900,
                "source_type": "in_material_dialogue", "confidence": "medium",
                "evidence": "Jesse, I'm here.",
            }]},
        ])
        self.assertEqual([(row["start_ms"], row["source_type"])
                          for row in merged["voice_source_spans"]], [
                              (100, "creator_narration"),
                              (600, "in_material_dialogue"),
                          ])


class MultiPassAnalysisTests(unittest.IsolatedAsyncioTestCase):
    """长素材必须**分段都读一遍**，而不是只读开头。"""

    LONG = "\n".join("第%d段。" % i + "正文内容。" * 400 for i in range(1, 9))

    async def test_deep_reads_every_window(self):
        seen = []

        async def fake_deep(text, *, kind, title, partial=False,
                            source_evidence_text=""):
            seen.append(text[:8])
            return {"topics": [text[:8]]}

        with mock.patch.object(material_analyzer, "_deep_once", fake_deep):
            merged = await material_analyzer.analyze_deep(self.LONG, kind="link")
        self.assertGreater(len(seen), 1, "长素材必须拆成多个窗口")
        self.assertEqual(len(merged["topics"]), len(seen),
                         "每个窗口的主题都要进产物")
        self.assertTrue(all(part.startswith("第") for part in seen),
                        "窗口内容来自原文而不是重复同一段")

    async def test_short_text_uses_single_call(self):
        """过了深析触发阈值（1200 字）但不到一个窗口（12000 字）→ 一次调用。"""
        seen = []

        async def fake_deep(text, *, kind, title, partial=False,
                            source_evidence_text=""):
            seen.append(partial)
            return {"topics": ["x"]}

        short = "一段不长不短的素材。" * 130
        self.assertGreater(len(short), material_analyzer.DEEP_TRIGGER_CHARS)
        self.assertLessEqual(len(short), material_analyzer.DEEP_INPUT_CHARS)
        with mock.patch.object(material_analyzer, "_deep_once", fake_deep):
            await material_analyzer.analyze_deep(short, kind="text")
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0], "单窗口不标 partial")

    async def test_conflicting_speaker_labels_trigger_one_source_wide_review(self):
        long_text = "\n".join("第%d段。" % i + "正文内容。" * 200
                               for i in range(1, 20))
        calls = []

        async def fake_deep(text, *, kind, title, partial=False,
                            source_evidence_text=""):
            calls.append(text[:6])
            source_type = ("creator_narration" if len(calls) == 1
                           else "in_material_dialogue")
            return {"topics": [text[:6]], "voice_sources": [{
                "speaker_id": "speaker_0", "source_type": source_type,
                "confidence": "medium", "evidence": text[:12],
            }]}

        reconcile = mock.AsyncMock(return_value=[{
            "speaker_id": "speaker_0", "source_type": "creator_narration",
            "confidence": "high", "evidence": "evenly sampled full-source evidence",
        }, {
            "speaker_id": "speaker_0", "source_type": "in_material_dialogue",
            "confidence": "medium", "evidence": "a distinct embedded line",
        }])
        with mock.patch.object(material_analyzer, "_deep_once", fake_deep), \
             mock.patch.object(material_analyzer, "_reconcile_voice_sources", reconcile):
            merged = await material_analyzer.analyze_deep(
                long_text, kind="video", title="a long clip")
        self.assertGreater(len(calls), 1)
        reconcile.assert_awaited_once_with(long_text, title="a long clip")
        self.assertEqual([row["source_type"] for row in merged["voice_sources"]], [
            "creator_narration", "in_material_dialogue",
        ])
        self.assertEqual({row["speaker_id"] for row in merged["voice_sources"]},
                         {"speaker_0"})

    async def test_summary_reads_every_window_then_merges(self):
        seen = []

        async def fake_summary(text, kind, *, partial=False):
            seen.append(partial)
            return "分段提要%d" % len(seen)

        async def fake_merge(parts, kind):
            return "归并后的全文提要"

        with mock.patch.object(material_analyzer, "_summary_once", fake_summary), \
             mock.patch.object(material_analyzer, "_summary_merge", fake_merge):
            got = await material_analyzer.summarize_material(self.LONG, kind="link")
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(seen), "分段读的那几遍都要标 partial")
        self.assertEqual(got, "归并后的全文提要")

    async def test_one_window_failure_does_not_kill_the_rest(self):
        calls = {"n": 0}

        async def flaky(text, *, kind, title, partial=False,
                        source_evidence_text=""):
            calls["n"] += 1
            if calls["n"] == 1:
                return {}
            return {"topics": ["窗口%d" % calls["n"]]}

        with mock.patch.object(material_analyzer, "_deep_once", flaky):
            merged = await material_analyzer.analyze_deep(self.LONG, kind="link")
        self.assertTrue(merged.get("topics"), "一个窗口失败不该毁掉整份分析")


class PromptFormatTests(unittest.TestCase):
    """提示词拼接的回归钉：`"a" "b" % x` 的 `%` 只作用于**最后一段字面量**，
    中间插一个 `+ scope +` 就会把格式化切断（本批真踩到过：TypeError 被
    `_deep_once` 的 except 吞成"没有分析"，表现是深析静默消失）。"""

    def _assert_clean(self, prompt: str, *must_contain: str):
        self.assertNotRegex(prompt, r"%[sdr]", "提示词里有没被替换的占位符")
        for needle in must_contain:
            self.assertIn(needle, prompt)

    def test_deep_prompt_fills_every_placeholder(self):
        whole = material_analyzer._deep_prompt("link", "某视频", "正文内容" * 20)
        self._assert_clean(whole, "《某视频》", "正文内容")
        self.assertIn("不要把译名另列为新人", whole)
        self.assertIn("最终 order 不能重置或倒序", whole)
        self.assertNotIn("其中一段", whole)

    def test_deep_prompt_partial_marks_the_scope(self):
        part = material_analyzer._deep_prompt("link", "某视频", "正文内容" * 20,
                                              partial=True)
        self._assert_clean(part, "《某视频》", "正文内容", "其中一段")

    def test_summary_prompt_fills_every_placeholder(self):
        whole = material_analyzer._summary_prompt("素材正文" * 20, "link")
        self._assert_clean(whole, "素材正文")
        self.assertNotIn("其中一段", whole)

    def test_summary_prompt_partial_forbids_guessing_the_rest(self):
        part = material_analyzer._summary_prompt("素材正文" * 20, "link",
                                                 partial=True)
        self._assert_clean(part, "素材正文", "其中一段", "不要猜测其他部分")


class DeepBudgetTests(unittest.TestCase):
    """深析注入按字段均分：整段截断会把排在后面的维度永远切掉。"""

    ALL_FIELDS = ("主题", "不同立场/观点", "可聊的角度", "人物/主体",
                  "时间线", "没讲清的")

    def _analysis(self) -> dict:
        return {field: ["第%d条分析内容大概四十个字左右的一段话" % i
                        for i in range(1, 9)]
                for field in ("topics", "people", "timeline",
                              "viewpoints", "hooks", "gaps")}

    def _labels(self, text: str) -> list:
        return [line.split("：", 1)[0] for line in text.split("\n")[1:]
                if "：" in line]

    def test_all_six_fields_survive_the_budget(self):
        text = TutorAgent._material_analysis_text(
            self._analysis(), budget=TutorAgent.MATERIAL_DEEP_MAX_CHARS)
        self.assertLessEqual(len(text), TutorAgent.MATERIAL_DEEP_MAX_CHARS)
        for label in self.ALL_FIELDS:
            self.assertIn(label, self._labels(text),
                          "字段 %s 被预算切掉了" % label)

    def test_no_budget_keeps_everything(self):
        """不传 budget 时逐字不变（老调用点零影响）。"""
        text = TutorAgent._material_analysis_text(self._analysis())
        self.assertGreater(len(text), TutorAgent.MATERIAL_DEEP_MAX_CHARS)
        for label in self.ALL_FIELDS:
            self.assertIn(label, self._labels(text))

    def test_empty_analysis_is_empty(self):
        self.assertEqual(TutorAgent._material_analysis_text({}), "")
        self.assertEqual(TutorAgent._material_analysis_text(None), "")


class ReasoningEffortWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_robust_client_forwards_reasoning_effort(self):
        from llm_client import RobustLLMClient
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None),
            finish_reason="stop",
        )])
        create = mock.AsyncMock(return_value=response)
        fake = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        client = RobustLLMClient()
        with mock.patch.object(client, "_get_client", return_value=fake):
            result = await client.chat_completion(
                [{"role": "user", "content": "test"}],
                model="deepseek-v4.1-flash", base_url="https://opencode.ai/zen/go/v1",
                api_key="test-key", max_tokens=6000,
                extra_body={"thinking": {"type": "enabled"}},
                reasoning_effort="low",
            )
        self.assertEqual(result, "ok")
        self.assertEqual(create.await_args.kwargs["reasoning_effort"], "low")
        self.assertEqual(create.await_args.kwargs["max_tokens"], 6000)

    def test_voice_source_assessment_survives_into_material_context(self):
        text = TutorAgent._material_analysis_text({
            "voice_sources": [
                "speaker_0：较可能是UP主旁白，置信度中；证据：讲解操作目的。",
            ],
        })
        self.assertIn("声音来源线索（分析判断）", text)
        self.assertIn("speaker_0", text)
        self.assertIn("不是素材原文里的事实", text)

    def test_multiple_source_roles_for_one_speaker_survive_into_context(self):
        text = TutorAgent._material_analysis_text({
            "voice_sources": [
                {"speaker_id": "speaker_0", "source_type": "creator_narration",
                 "confidence": "medium", "evidence": ["I will go this way."]},
                {"speaker_id": "speaker_0", "source_type": "in_material_dialogue",
                 "confidence": "medium", "evidence": ["Remember your father."]},
            ],
        })
        self.assertIn("创作者旁白/解说", text)
        self.assertIn("素材内角色对白", text)
        self.assertEqual(text.count("speaker_0"), 2)

    def test_voice_source_spans_survive_into_single_and_multi_material_context(self):
        span = {"start_ms": 100, "end_ms": 500,
                "source_type": "creator_narration", "confidence": "high",
                "evidence": "I can explain the map."}
        text = TutorAgent._material_analysis_text({"voice_source_spans": [span]})
        self.assertIn("按时间标注的声音来源片段", text)
        self.assertIn("100-500ms", text)
        self.assertIn("来源片段是逐句归属边界", text)
        info = {
            "multi": True,
            "sources": [{"material_id": 9, "title": "Gameplay clip",
                         "source_coverage": {"kind": "video",
                                             "body_available": True},
                         "voice_source_spans": [span]}],
            "evidence_blocks": [{"material_id": 9, "block_no": 1,
                                 "span_kind": "body",
                                 "text": "[speaker=speaker_0; time_ms=100-500] I can explain the map."}],
        }
        ctx = SessionContext(session_id="voice-source-span", user_id="u", target_lang="en")
        ctx.material_context = info
        packet = TutorAgent._material_evidence_packet(ctx, "What is the host saying?")
        rendered = TutorAgent._material_text_multi(info, packet)
        self.assertIn("视频约00:00", rendered)
        self.assertIn("创作者旁白/解说", rendered)
        self.assertIn("来源片段是逐句归属边界", rendered)

    def test_deepseek_deep_analysis_enables_thinking_by_default(self):
        from engine import material_analyzer
        fake = mock.AsyncMock(return_value='{"topics": ["x"]}')
        with mock.patch.dict(os.environ, {
            material_analyzer.DEEP_MODEL_ENV: "deepseek-v4.1-flash",
            material_analyzer.DEEP_THINKING_ENV: "true",
        }, clear=False), \
             mock.patch("llm_client.DEEPSEEK_BASE_URL", "https://api.deepseek.com"), \
             mock.patch("llm_client.robust_llm_client.chat_completion", new=fake):
            result = __import__("asyncio").run(
                material_analyzer._deep_once("正文", kind="link", title="素材"))
        self.assertEqual(result.get("topics"), ["x"])
        self.assertEqual(fake.await_args.kwargs["extra_body"],
                         {"thinking": {"type": "enabled"}})
        self.assertEqual(fake.await_args.kwargs["max_tokens"], 16000)
        self.assertEqual(fake.await_args.kwargs["reasoning_effort"], "low")

    def test_diarized_transcript_is_sent_as_labeled_analysis_evidence(self):
        from engine import material_analyzer
        text = material_analyzer.format_transcript_with_speakers("旁白对白", [
            {"text": "旁白", "speaker_id": "speaker_0", "start_ms": 0, "end_ms": 500},
            {"text": "对白", "speaker_id": "speaker_1", "start_ms": 500, "end_ms": 900},
        ])
        self.assertIn("[speaker=speaker_0; time_ms=0-500] 旁白", text)
        self.assertIn("[speaker=speaker_1; time_ms=500-900] 对白", text)

    def test_comment_tail_is_excluded_before_speaker_coverage_is_checked(self):
        from engine import material_analyzer
        combined = "旁白正文" + material_analyzer.COMMENT_MARKER + "观众评论"
        body = material_analyzer.material_analysis_body(combined)
        rendered = material_analyzer.format_transcript_with_speakers(body, [
            {"text": "旁白正文", "speaker_id": "speaker_0",
             "start_ms": 0, "end_ms": 500},
        ])
        self.assertNotIn("观众评论", body)
        self.assertIn("speaker=speaker_0", rendered)

    def test_voice_source_map_is_available_to_multi_source_tape(self):
        from engine.material_analyzer import _tape_plan_prompt
        info = {
            "multi": True,
            "sources": [{
                "material_id": 9,
                "title": "Gameplay clip",
                "creator_display_name": "纯黑",
                "creator_display_name_source": "user_confirmed",
                "voice_sources": [
                    "speaker_0：较可能是UP主旁白，置信度中；证据：讲解操作目的。",
                ],
                "voice_source_spans": [{
                    "start_ms": 100, "end_ms": 500,
                    "source_type": "creator_narration",
                    "confidence": "high",
                    "evidence": "Pope sent the research team below.",
                }],
                "analysis_source_map": {
                    "validated": True,
                    "entities": [{
                        "name": "Pope", "aliases": [],
                        "role": "She assigns the team to investigate.",
                        "evidence": {"quote": "Pope sent the research team below.",
                                     "start_ms": 100, "end_ms": 500,
                                     "source_type": "creator_narration"},
                    }],
                    "beats": [{
                        "order": 1, "event": "Pope sends the team below.",
                        "effect": "The investigation moves underground.",
                        "evidence": {"quote": "Pope sent the research team below.",
                                     "start_ms": 100, "end_ms": 500,
                                     "source_type": "creator_narration"},
                    }],
                },
                "analysis_navigation": {"timeline": ["Legacy navigation must be suppressed."]},
            }],
            "evidence_blocks": [{
                "material_id": 9, "block_no": 1, "span_kind": "body",
                "text": "A narrator explains the gameplay.",
            }],
        }
        ctx = SessionContext(session_id="voice-source-map", user_id="u", target_lang="en")
        ctx.material_context = info
        packet = TutorAgent._material_evidence_packet(ctx, "What is happening in the clip?")
        rendered = TutorAgent._material_text_multi(info, packet)
        self.assertIn("声音来源线索", rendered)
        self.assertIn("speaker_0", rendered)
        self.assertIn("发布者账号名：纯黑", rendered)
        self.assertIn("用户已确认", rendered)
        self.assertIn("正文证据导航", rendered)
        self.assertIn("约00:00", rendered)
        self.assertIn("Pope sends the team below", rendered)
        self.assertNotIn("This is the exact saved transcript quote", rendered)
        self.assertNotIn("Legacy navigation must be suppressed", rendered)
        self.assertIn("来源边界", rendered)
        self.assertIn("不能据此确认它与更大作品世界观的关系", rendered)
        self.assertIn("描述这段解说时直接自然使用账号名一次", rendered)
        self.assertIn("不代表具体语句的说话者", rendered)
        prompt = _tape_plan_prompt([{
            "material_id": 9, "title": "Gameplay clip", "kind": "video",
            "summary": "Gameplay with commentary.",
            "voice_sources": info["sources"][0]["voice_sources"],
            "blocks": [{"block_no": 1, "span_kind": "body",
                        "text": "A narrator explains the gameplay."}],
        }])
        self.assertIn("声源线索（自动分析判断", prompt)
        self.assertIn("置信度中", prompt)


class MaterialVoiceCompletenessTests(unittest.TestCase):
    def test_story_map_uses_source_order_over_model_order_or_timestamps(self):
        beats = []
        for source_index in reversed(range(6)):
            beats.append({
                "order": len(beats) + 1,
                "event": "Beat %d" % source_index,
                "effect": "",
                "evidence": {
                    "quote": "Exact source quote for beat %d." % source_index,
                    "source_offset": source_index * 100,
                    # A conflicting timestamp must not override the body order.
                    "start_ms": (5 - source_index) * 1000,
                    "end_ms": (6 - source_index) * 1000,
                },
            })
        cleaned = TutorAgent._material_source_map({
            "source_map": {"validated": True, "beats": beats},
        })
        self.assertEqual([row["event"] for row in cleaned["beats"]],
                         ["Beat %d" % index for index in range(6)])
        self.assertTrue(all("source_offset" not in row["evidence"]
                            for row in cleaned["beats"]))

    def test_rendered_story_map_keeps_whole_rows_and_late_beats(self):
        beats = []
        for index in reversed(range(10)):
            beats.append({
                "order": len(beats) + 1,
                "event": "Beat %d advances the investigation through a distinct event." % index,
                "effect": "The team learns one new piece of evidence.",
                "evidence": {
                    "quote": "This is the exact saved transcript quote for event %d." % index,
                    "source_offset": index * 500,
                    "source_type": "in_material_dialogue",
                    "start_ms": index * 60000,
                    "end_ms": index * 60000 + 3000,
                },
            })
        info = {
            "sources": [{
                "material_id": 61, "title": "P5", "analysis_source_map": {
                    "validated": True, "beats": beats,
                },
            }],
            "evidence_blocks": [],
        }
        rendered = TutorAgent._material_text_multi(info, info)
        self.assertIn("Beat 0 advances", rendered)
        self.assertIn("Beat 9 advances", rendered)
        self.assertNotIn("source_offset", rendered)
        map_section = rendered.split("正文证据导航", 1)[1].split("\n\n", 1)[0]
        beat_rows = [line.strip() for line in map_section.splitlines()
                     if line.strip().split(".", 1)[0].isdigit()]
        self.assertEqual(len(beat_rows), 10, beat_rows)
        self.assertTrue(all("素材内对白" in row and "→" in row
                            for row in beat_rows), beat_rows)
        self.assertTrue(all("This is the exact saved transcript quote" not in row
                            for row in beat_rows), beat_rows)
        positions = [int(row.split(".", 1)[0]) for row in beat_rows]
        self.assertEqual(positions, sorted(positions))

    def test_saved_story_navigation_reaches_multi_tape_context(self):
        class FakeL2:
            async def get_material_tape_build(self, user_id, conv_id=""):
                return {"status": "ready", "source_ids": [61], "summary": "A story segment."}

            async def get_material(self, user_id, material_id, with_text=False):
                return {
                    "title": "Control gameplay P5", "kind": "video",
                    "summary": "Dylan follows the research team's trail.",
                    "analysis": {
                        "people": [
                            "Pope: sends the research team to gather data.",
                            "Zoe: Dylan's contact over the radio.",
                            "Dr. Melanie Grist: studies the overlapping timelines.",
                        ],
                        "timeline": [
                            "The team enters the underground area and goes missing.",
                            "Grist asks for relays to study the Oldest House's resistance.",
                            "New data returns; Dylan leaves a Jesse-related question unresolved.",
                        ],
                        "viewpoints": ["A speculative comment that should not become plot fact."],
                    },
                }

            async def list_material_source_blocks(self, user_id, material_ids):
                return [{
                    "material_id": 61, "block_no": 1, "span_kind": "body",
                    "text": (
                        "Pope sent the research team to gather data. Zoe contacts Dylan by radio. "
                        "Dr. Melanie Grist asks for relays to study the Oldest House's resistance. "
                        "New data arrives, while Dylan leaves what he saw about Jesse unexplained."
                    ),
                }]

        agent = TutorAgent.__new__(TutorAgent)
        agent.l2 = FakeL2()
        bundle = __import__("asyncio").run(
            agent._material_tape_bundle("user", "material-conv"))

        self.assertTrue(bundle.get("multi"))
        self.assertEqual(bundle["sources"][0]["analysis_navigation"]["people"][0],
                         "Pope: sends the research team to gather data.")
        self.assertNotIn("viewpoints", bundle["sources"][0]["analysis_navigation"],
                         "unattributed viewpoints are not part of the plot/role index")

        ctx = SessionContext(session_id="material-story-map", user_id="user", target_lang="en")
        ctx.material_context = bundle
        packet = TutorAgent._material_evidence_packet(
            ctx, "Can you connect the events and explain Zoe's role?")
        rendered = TutorAgent._material_text_multi(bundle, packet)
        self.assertIn("素材理解导航", rendered)
        self.assertIn("事件顺序线索", rendered)
        self.assertIn("Zoe: Dylan's contact over the radio", rendered)
        self.assertIn("不完整复述同一段对白", rendered)
        self.assertIn("人物身份、关系、先后和结论都要用本轮正文证据核实", rendered)
        self.assertNotIn("speculative comment", rendered)

    def test_material_reasoning_labels_inferred_motives(self):
        app = Path(__file__).resolve().parents[1]
        root = app.parents[1]
        director = (app / "agent" / "conversation_director.py").read_text(encoding="utf-8")
        control = (app / "agent" / "tutor_agent.py").read_text(encoding="utf-8")
        rule = "由相邻细节推出来的原因要标明是解读"
        self.assertIn(rule, director)
        self.assertIn(rule, control)
        source_rule = "只有对应的素材内对白片段自身明确表达该事实"
        self.assertIn(source_rule, director)
        self.assertIn(source_rule, control)
        epistemic_rule = "来源身份只回答‘谁在说’"
        self.assertIn(epistemic_rule, director)
        self.assertIn(epistemic_rule, control)
        relevance_rule = "只引用直接支持或反驳该判断的来源片段"
        speaker_rule = "只有原文明确标出说话者才能把话归给具体角色"
        self.assertIn(relevance_rule, director)
        self.assertIn(relevance_rule, control)
        self.assertIn(speaker_rule, director)
        self.assertIn(speaker_rule, control)

    def test_provider_segments_do_not_end_a_material_tape_turn(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / "voice" / "doubao-session-server.mjs").read_text(encoding="utf-8")
        self.assertIn(
            "sessionCassetteMode === 'expression' || sessionIsMaterialTape",
            source,
        )
        self.assertIn(
            "sessionCassetteMode !== 'expression' &&\n            !sessionIsMaterialTape",
            source,
        )

    def test_controlled_reply_paths_do_not_silently_drop_units_after_four(self):
        app = Path(__file__).resolve().parents[1]
        source = (app / "agent" / "tutor_agent.py").read_text(
            encoding="utf-8"
        )
        start = source.index("    async def respond_turn_stream(")
        next_method = source.find("\n    async def ", start + 1)
        stream_body = source[start:next_method if next_method >= 0 else len(source)]
        self.assertNotIn("units[:4]", stream_body)




class MaterialTapeStoryNavigationTests(unittest.TestCase):
    def test_incomplete_video_absence_gate_is_source_scoped(self):
        from engine.material_claim_scope import incomplete_primary_source, asserts_absence_in_unverified_video
        sources = [{"material_id": 1, "title": "Room", "source_coverage": {"kind": "video", "body_available": None}},
                   {"material_id": 2, "title": "Research", "source_coverage": {"kind": "video", "body_available": True}}]
        blocks = [{"material_id": 1, "text": "room kitchen layout", "span_kind": "legacy_unattributed"},
                  {"material_id": 2, "text": "research setbacks recovery", "span_kind": "body"}]
        self.assertEqual(incomplete_primary_source(sources, blocks, "room layout")["material_id"], 1)
        self.assertEqual(incomplete_primary_source(sources, blocks, "research recovery"), {})
        self.assertTrue(asserts_absence_in_unverified_video("The video doesn't walk through the layout."))
        self.assertFalse(asserts_absence_in_unverified_video("The available page text does not describe the layout."))
        self.assertFalse(asserts_absence_in_unverified_video("I cannot confirm whether the video explains it."))

    def test_legacy_video_recovery_retains_text_without_claiming_transcript(self):
        from engine.material_analyzer import recover_legacy_source_blocks
        spans, analysis = recover_legacy_source_blocks({"source_ref": "https://www.bilibili.com/video/BV1234/", "raw_text": "A room introduction.", "blocks": ["A room introduction."], "analysis": {}})
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["text"], "A room introduction.")
        self.assertEqual(spans[0]["span_kind"], "legacy_unattributed")
        self.assertIsNone(analysis["source_coverage"]["body_available"])

    def test_legacy_text_and_verified_video_keep_body_provenance(self):
        from engine.material_analyzer import recover_legacy_source_blocks
        for row in [{"source_ref": "https://example.org/essay", "raw_text": "An essay.", "analysis": {}},
                    {"source_ref": "https://www.bilibili.com/video/BV1234/", "raw_text": "A dialogue.", "analysis": {"source_coverage": {"kind": "video", "body_available": True}}}]:
            spans, _ = recover_legacy_source_blocks(row)
            self.assertEqual(spans[0]["span_kind"], "body")

    def test_unknown_source_is_not_classified_by_title(self):
        rendered = TutorAgent._material_text_multi({}, {"sources": [{"material_id": 3, "title": "Room", "source_coverage": {"kind": "video", "body_available": None}}]}, query="How does it connect?")
        self.assertIn("旧素材覆盖未核验", rendered)
        self.assertIn("不把资料不足当成内容无关", rendered)

    def test_analysis_and_runtime_share_scope_contract(self):
        from engine.material_claim_scope import MATERIAL_CLAIM_SCOPE_RULE
        from engine.material_analyzer import _tape_plan_prompt
        source = {"material_id": 1, "title": "Essay", "blocks": [{"text": "A person recovered.", "span_kind": "body"}]}
        self.assertIn(MATERIAL_CLAIM_SCOPE_RULE, _tape_plan_prompt([source]))
        rendered = TutorAgent._material_text_multi({}, {"sources": [source], "lines": ["A person recovered."]}, query="What follows from this?")
        self.assertIn(MATERIAL_CLAIM_SCOPE_RULE, rendered)

    def test_unwatched_nonstory_material_keeps_body_evidence(self):
        rendered = TutorAgent._material_text_multi({}, {
            "sources": [{"material_id": 1, "title": "Essay", "summary": "Research setbacks."}],
            "groups": [{"title": "Research", "summary": "Earlier setbacks and later recognition.", "material_ids": [1]}],
            "lines": ["[1:1 body] Earlier she was discouraged; later she received an award."],
        }, query="这个视频我没看过，先讲讲内容。")
        self.assertIn("Earlier she was discouraged", rendered)
        self.assertNotIn("以上时间线来自", rendered)

    def test_plain_materials_receive_comparison_and_source_authority_rules(self):
        rendered = TutorAgent._material_text_multi({}, {
            "sources": [{"material_id": 1, "title": "First", "summary": "A story."},
                        {"material_id": 2, "title": "Second", "summary": "A commentary."}],
            "groups": [], "evidence": [],
        }, query="How do these connect?")
        self.assertIn("共同主题不等于相同动机", rendered)
        self.assertIn("不把解说视频整体称为", rendered)

    def test_build_analysis_receives_cached_background_with_separate_authority(self):
        from engine.material_analyzer import _tape_plan_prompt
        prompt = _tape_plan_prompt([{
            "material_id": 7, "title": "Mira expedition", "kind": "video",
            "blocks": [{"text": "Mira finds an abandoned camp.", "span_kind": "body"}],
            "source_map": {"entities": [{"name": "Mira"}]},
            "external_context": {"status": "ready", "sources": [{
                "url": "https://example.org/atlas", "snippet": "Atlas is a game.",
                "body_text": "Mira and Lena are siblings."}]},
        }])
        self.assertIn("Mira and Lena are siblings.", prompt)
        self.assertIn("https://example.org/atlas", prompt)
        self.assertIn("不能替代本段情节证据", prompt)
        self.assertIn("Mira finds an abandoned camp.", prompt)

    def test_cached_background_selects_matching_body_passage_from_later_source(self):
        context = {"status": "ready", "sources": [
            {"url": "https://example.org/a", "snippet": "Atlas has beautiful architecture."},
            {"url": "https://example.org/b", "snippet": "Atlas has several locations."},
            {"url": "https://example.org/c", "snippet": "Atlas follows an expedition.",
             "body_text": "Read related news today. Mira and Lena are siblings. Their parents worked at the harbor."}]}
        rendered = TutorAgent._material_external_context_text(
            context, query="How are Mira and Lena related?", budget=600)
        self.assertIn("Mira and Lena are siblings.", rendered)
        self.assertNotIn("beautiful architecture", rendered)
        self.assertNotIn("Read related news", rendered)
        self.assertLessEqual(len(rendered), 600)

    def test_valid_repaired_material_reply_stays_available_to_followups(self):
        from unittest.mock import patch
        from agent.tutor_agent import TutorAgent
        prepared = {"constraints": {"material_tape": True}}
        with patch.object(TutorAgent, "_controlled_reply_violations", return_value=[]):
            self.assertFalse(TutorAgent._repaired_reply_needs_history_exclusion(
                "The team is trapped across timelines.", prepared))
            self.assertTrue(TutorAgent._repaired_reply_needs_history_exclusion(
                "I don't have a complete answer to that yet.", prepared))
            self.assertTrue(TutorAgent._repaired_reply_needs_history_exclusion(
                "A repaired chat answer.", {"constraints": {}}))
        with patch.object(TutorAgent, "_controlled_reply_violations", return_value=["unfinished"]):
            self.assertTrue(TutorAgent._repaired_reply_needs_history_exclusion(
                "The team is", prepared))

    def source(self):
        first = "Mira finds the expedition equipment, but the researchers are missing."
        last = "The relay delivers new data, but the expedition's fate is still unknown."
        return {
            "material_id": 92, "title": "An expedition episode", "kind": "video",
            "summary": "An old compressed summary.",
            "blocks": [{"block_no": 1, "span_kind": "body", "text": first},
                       {"block_no": 2, "span_kind": "body",
                        "text": "An unrelated corridor. " * 150 + last},
                       {"block_no": 3, "span_kind": "comment",
                        "text": "The expedition was rescued by an unseen helicopter."}],
            "source_map": {"validated": True, "beats": [
                {"order": 1, "event": "New data arrives without resolving the expedition's fate.",
                 "evidence": {"quote": last}},
                {"order": 2, "event": "Equipment is found without the researchers.",
                 "evidence": {"quote": first}}]},
        }

    def test_complete_body_navigation_preserves_late_evidence_in_source_order(self):
        prompt = material_analyzer._tape_plan_prompt([self.source()])
        nav = prompt.split("正文剧情索引", 1)[1].split("来源块", 1)[0]
        self.assertIn("The relay delivers new data", nav)
        self.assertLess(nav.index("Equipment is found"), nav.index("New data arrives"))
        self.assertIn("不是额外原文", nav)
        self.assertNotIn("An old compressed summary.", prompt)

    def test_comment_and_invented_evidence_cannot_enter_story_navigation(self):
        source = self.source()
        source["source_map"]["beats"] += [
            {"event": "A helicopter rescued them.", "evidence": {
                "quote": source["blocks"][2]["text"]}},
            {"event": "A secret door was opened.", "evidence": {
                "quote": "A secret door opened into the final laboratory."}},
        ]
        nav = material_analyzer._tape_plan_prompt([source]).split(
            "正文剧情索引", 1)[1].split("来源块", 1)[0]
        self.assertNotIn("helicopter", nav)
        self.assertNotIn("secret door", nav)
        source["source_map"]["validated"] = False
        self.assertNotIn("正文剧情索引", material_analyzer._tape_plan_prompt([source]))


    def context(self):
        source = self.source()
        source["analysis_source_map"] = source.pop("source_map")
        source["summary"] = "The hero has already found everyone and defeated the final boss."
        return {"multi": True, "summary": "The expedition continues; 评论区提出玩家猜测。",
                "groups": [], "hooks": [], "sources": [source],
                "evidence_blocks": [{"material_id": 92, **block}
                                    for block in source["blocks"]]}

    def test_body_question_does_not_resurrect_legacy_story_summary(self):
        info = self.context()
        ctx = SessionContext(session_id="legacy-story-summary", user_id="u", target_lang="en")
        ctx.material_context = info
        packet = TutorAgent._material_evidence_packet(ctx, "Who is Mira?")
        text = TutorAgent._material_text_multi(info, packet, query="Who is Mira?")
        self.assertNotIn("already found everyone", packet["summary"])
        self.assertNotIn("already found everyone", text)
        self.assertIn("researchers are missing", text)

    def test_recap_keeps_cached_background_for_a_new_viewer(self):
        info = self.context()
        info["sources"][0]["external_context"] = {
            "status": "ready", "queries": ["expedition game background"],
            "sources": [{"title": "Official expedition game overview",
                         "url": "https://example.com/expedition",
                         "snippet": "Mira leads the expedition in a world changed by a strange force."}]}
        query = "I haven't played this game. What's this episode about?"
        ctx = SessionContext(session_id="recap-background", user_id="u", target_lang="en")
        ctx.material_context = info
        packet = TutorAgent._material_evidence_packet(ctx, query)
        text = TutorAgent._material_text_multi(info, packet, query=query)
        self.assertIn("Mira", packet["external_background"])
        self.assertIn("理解本段所需的外部背景", text)
        self.assertIn("Mira leads the expedition", text)
        self.assertIn("不是本段剧情", text)
        self.assertNotIn("already found everyone", text)


class MaterialTapeStoryBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_story_map_reaches_actual_tape_builder(self):
        import main
        source = MaterialTapeStoryNavigationTests().source()
        analysis = {"source_map": source["source_map"], "external_context": {
            "schema_version": material_analyzer.EXTERNAL_CONTEXT_SCHEMA_VERSION}}
        store = SimpleNamespace(
            get_material=mock.AsyncMock(return_value={"title": source["title"],
                "kind": "video", "summary": source["summary"], "analysis": analysis}),
            list_material_source_blocks=mock.AsyncMock(return_value=source["blocks"]),
            finish_material_tape_build=mock.AsyncMock(return_value=True),
            fail_material_tape_build=mock.AsyncMock())
        builder = mock.AsyncMock(return_value={"summary": "The expedition makes progress.",
            "groups": [], "hooks": [], "top_chunks": []})
        with mock.patch.object(main, "l2", store), mock.patch.object(
                material_analyzer, "build_material_tape_plan", builder):
            result = await main._run_material_tape_build("qa-user", 3,
                conv_id="qa-story_A", name="Story test", source_ids=[92])
        self.assertTrue(result["ok"])
        sent = builder.call_args.args[0]
        self.assertEqual(sent[0]["source_map"], source["source_map"])
        self.assertIn("正文剧情索引", material_analyzer._tape_plan_prompt(sent))
        store.fail_material_tape_build.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class ExternalRefreshRetentionTests(unittest.TestCase):
    def test_failed_policy_refresh_retains_original_evidence(self):
        from engine.material_analyzer import merge_external_context, EXTERNAL_CONTEXT_QUERY_POLICY
        source = {"url": "https://example.org/study", "page_text": "original measurements"}
        result = merge_external_context({"status": "ready", "sources": [source]},
                                        {"status": "no_relevant_results", "sources": []})
        self.assertEqual(result["sources"], [source])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["query_policy"], EXTERNAL_CONTEXT_QUERY_POLICY)
        self.assertEqual(result["refresh_status"], "no_relevant_results")

    def test_refresh_deduplicates_urls_and_keeps_new_evidence_first(self):
        from engine.material_analyzer import merge_external_context
        result = merge_external_context({"sources": [{"url": "https://example.org/a"}]},
             {"status": "ready", "sources": [{"url": "https://example.org/a", "snippet": "updated"},
                                                {"url": "https://example.org/b"}]})
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["snippet"], "updated")


class ExternalNaturalChineseQueryTests(unittest.TestCase):
    def test_entity_in_a_chinese_sentence_matches_cached_context(self):
        context = {"status": "ready", "sources": [{"url": "https://example.org/interview",
                   "title": "数学家周宁的采访", "snippet": "周宁谈到学习和研究经历。"}]}
        self.assertTrue(TutorAgent._external_context_needed("这个讲周宁的视频，有没有其他资料补充她的经历？", context))
        self.assertFalse(TutorAgent._external_context_needed("球队后来换了什么阵型？", context))


class CachedBackgroundQuestionSelectionTests(unittest.TestCase):
    def test_shared_background_selects_the_question_not_the_first_paragraph(self):
        sources = [{"title": "研究者访谈", "external_context": {"status": "ready", "sources": [{
            "url": "https://example.org/interview", "title": "研究者的经历",
            "snippet": "这是一篇介绍研究者获奖感受的采访。",
            "body_text": "作者信息与获奖仪式介绍。\n转系过程很辛苦，需要同时学习两个专业的课程。\n宿舍环境与休息安排介绍。"}]}}]
        answer = TutorAgent._material_cached_background(sources, "转系过程为什么很辛苦？")
        self.assertIn("两个专业", answer)
        self.assertNotIn("作者信息", answer)


class MultiSourceCreatorAttributionTests(unittest.TestCase):
    def test_one_known_handle_must_not_rename_other_sources_narrators(self):
        ctx = SessionContext(session_id="multi-creator", user_id="u", target_lang="en")
        ctx.material_context = {"multi": True, "sources": [{"material_id": 1}, {
            "material_id": 2, "creator_display_name": "FoodHost", "voice_source_spans": [{
            "source_type": "creator_narration", "confidence": "high"}]}]}
        reply = "The narrator in the research video describes emotional support."
        self.assertEqual(TutorAgent._prefer_creator_handle_for_narration(reply, ctx), reply)


class MaterialResponseBudgetTests(unittest.TestCase):
    def test_short_chinese_content_question_is_not_a_greeting_budget(self):
        prepared = {"user_input_chars": 22, "constraints": {"material_tape": True, "material_source_count": 1}}
        with mock.patch.dict(os.environ, {"MATERIAL_RESPONSE_MAX_TOKENS": "768"}):
            self.assertEqual(TutorAgent._spoken_response_token_budget(prepared, 320), 512)
            prepared["constraints"]["material_source_count"] = 3
            self.assertEqual(TutorAgent._spoken_response_token_budget(prepared, 320), 768)
        self.assertEqual(TutorAgent._spoken_response_token_budget({"user_input_chars": 22}, 320), 160)

    def test_material_explicit_budget_override_is_respected(self):
        with mock.patch.dict(os.environ, {"MATERIAL_RESPONSE_MAX_TOKENS": "200"}):
            self.assertEqual(TutorAgent._spoken_response_token_budget({"constraints": {"material_tape": True}}, 320), 200)


class ExternalQueryPurposeTests(unittest.TestCase):
    def test_verification_query_survives_comment_supplement(self):
        from engine.material_analyzer import _supplement_comment_search_query
        queries = ["Example background official", "growth mindset independent trial"]
        self.assertEqual(_supplement_comment_search_query(
            queries, "Example", {}, ["Hiss is confusing"],
            protected_queries=[queries[1]]), queries)

    def test_typed_plan_keeps_purpose_and_accepts_legacy_queries(self):
        from engine.material_analyzer import _normalize_external_query_plan
        purposes = {}
        queries = _normalize_external_query_plan([
            {"query": "growth mindset trial", "purpose": "claim_verification"},
            "Example biography"], purposes)
        self.assertEqual(queries, ["growth mindset trial", "Example biography"])
        self.assertEqual(purposes[queries[0]], "claim_verification")
        self.assertEqual(purposes[queries[1]], "background")


class MaterialComparisonContextTests(unittest.TestCase):
    def test_material_comparison_reserves_a_synthesis_unit(self):
        contract = TutorAgent._build_reply_contract(
            policy=SimpleNamespace(response_length="1-3 short sentences"),
            move=SimpleNamespace(action="", feedback_mode="", user_must_speak=False, prompt_hint=""),
            constraints={"material_tape": True, "material_source_count": 2,
                         "max_explanation_units": 1},
            turn_decision={"primary_frame": "general", "response_sequence": ["respond_to_latest_input"],
                           "primary_goal": "Compare both stories"},
            director={"content_need": "What does each story show?"},
        )
        self.assertEqual(contract["teaching"]["max_explanation_units"], 3)

        many_sources = TutorAgent._build_reply_contract(
            policy=SimpleNamespace(response_length="1-3 short sentences"),
            move=SimpleNamespace(action="", feedback_mode="", user_must_speak=False, prompt_hint=""),
            constraints={"material_tape": True, "material_source_count": 8,
                         "max_explanation_units": 1},
            turn_decision={"primary_frame": "general", "response_sequence": ["respond_to_latest_input"],
                           "primary_goal": "Compare each story"},
            director={"content_need": "What does each source show?"},
        )
        self.assertEqual(many_sources["teaching"]["max_explanation_units"], 3)

        ordinary = TutorAgent._build_reply_contract(
            policy=SimpleNamespace(response_length="1-3 short sentences"),
            move=SimpleNamespace(action="", feedback_mode="", user_must_speak=False, prompt_hint=""),
            constraints={"material_tape": True, "material_source_count": 2,
                         "max_explanation_units": 1},
            turn_decision={"primary_frame": "general", "response_sequence": ["respond_to_latest_input"],
                           "primary_goal": "Tell me what happened in the second story"},
            director={"content_need": "Answer the question"},
        )
        self.assertEqual(ordinary["teaching"]["max_explanation_units"], 1)

    def test_comparison_preserves_both_long_source_contexts(self):
        ctx = SessionContext(session_id="comparison", user_id="u", target_lang="en")
        ctx.material_context = {
            "sources": [{"material_id": 1, "title": "First"}, {"material_id": 2, "title": "Second"}],
            "evidence_blocks": [
                {"material_id": mid, "block_no": 1, "span_kind": "body",
                 "text": "Their explanation of adversity. " + "context " * 180}
                for mid in (1, 2)]}
        packet = TutorAgent._material_evidence_packet(ctx, "Compare their explanations of adversity")
        self.assertEqual(packet["refs"], ["1:1", "2:1"])

    def test_deep_parser_preserves_query_purpose(self):
        import json
        from engine.material_analyzer import _parse_deep_json
        plan = [{"query": "growth mindset trial", "purpose": "claim_verification"}]
        parsed = _parse_deep_json(json.dumps({"external_context_queries": plan}))
        self.assertEqual(parsed["external_context_queries"], plan)


class MaterialComparisonCoverageTests(unittest.TestCase):
    def test_comparison_includes_closing_advice_without_query_keywords(self):
        ctx = SessionContext(session_id="compare-qualifier", user_id="u", target_lang="en")
        ctx.material_context = {
            "sources": [{"material_id": 1, "title": "First"}, {"material_id": 2, "title": "Second"}],
            "evidence_blocks": [
                {"material_id": 1, "block_no": 1, "span_kind": "body", "text": "Hard times need persistence."},
                {"material_id": 2, "block_no": 1, "span_kind": "body", "text": "Hard times are not the end."},
                {"material_id": 2, "block_no": 2, "span_kind": "body", "text": "Beliefs can change."},
                {"material_id": 2, "block_no": 3, "span_kind": "body", "text": "Seek help from friends and professionals."}]}
        packet = TutorAgent._material_evidence_packet(ctx, "Compare their views on hard times")
        self.assertIn("2:3", packet["refs"])


class MaterialCommentContextGuardTests(unittest.TestCase):
    def test_people_in_body_discussion_are_not_commenters(self):
        from agent.tutor_agent import _material_comment_claim_overgeneralizes
        self.assertFalse(_material_comment_claim_overgeneralizes(
            "People generally need both practical support and hope. They believe change is possible.",
            "So it is not only endurance or mindset, right?"))

    def test_actual_viewer_generalization_still_rejected(self):
        from agent.tutor_agent import _material_comment_claim_overgeneralizes
        self.assertTrue(_material_comment_claim_overgeneralizes(
            "Most viewers agree with the speaker.", "What do the comments say?"))


class MaterialScopedOutputContractTests(unittest.TestCase):
    def test_material_contract_removes_fixed_sentence_compression(self):
        from agent.tutor_agent import _OUTPUT_CONTRACT_CACHE, _scope_material_output_contract
        scoped = _scope_material_output_contract(_OUTPUT_CONTRACT_CACHE, True)
        self.assertNotIn("只输出1到3", scoped)
        self.assertNotIn("用一到两句讲完", scoped)
        self.assertIn("USER_INTENT_SOURCE", scoped)
        self.assertIn("每个短句必须独立完整", scoped)

    def test_ordinary_contract_stays_identical(self):
        from agent.tutor_agent import _OUTPUT_CONTRACT_CACHE, _scope_material_output_contract
        self.assertEqual(_scope_material_output_contract(_OUTPUT_CONTRACT_CACHE, False), _OUTPUT_CONTRACT_CACHE)

    def test_research_query_keeps_findings_and_limitations(self):
        from engine.material_analyzer import select_external_reference_passages
        context = {"status":"ready", "sources":[{
            "title":"A school mindset trial", "url":"https://doi.org/example",
            "evidence_kind":"research_abstract", "retrieval_query":"成长性思维 growth mindset school trial",
            "snippet":"We examine school achievement.",
            "body_text":"We examine school achievement. " + "Methods were randomized. " * 20 +
                        "Effects depended on school context and did not apply to every student."}]}
        chosen = select_external_reference_passages(context, "成长性思维有研究证据吗？")
        self.assertIn("did not apply to every student", chosen[0]["quote"])
        ctx = TutorAgent._clean_material_external_context(context)
        self.assertEqual(ctx["sources"][0]["evidence_kind"], "research_abstract")
