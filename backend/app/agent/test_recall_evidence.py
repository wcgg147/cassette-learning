import unittest

from agent.recall_evidence import (
    Evidence,
    build_evidence,
    build_topic_index,
    coverage,
    index_overlap,
    looks_like_reference_question,
    render_evidence,
    render_topic_index,
    should_offer,
)


def archive(sid, summary, **kw):
    return {"session_id": sid, "summary": summary, **kw}


class ShouldOfferTests(unittest.TestCase):
    def test_route_episodic_offers(self):
        self.assertTrue(should_offer(route_mode="episodic_recall"))

    def test_any_candidate_source_offers(self):
        self.assertTrue(should_offer(candidate_count=1))
        self.assertTrue(should_offer(related_entity_count=1))

    def test_open_threads_alone_do_not_offer(self):
        self.assertFalse(should_offer(open_thread_count=1))
        self.assertTrue(should_offer(open_thread_count=1, user_text="我那个麻烦事后来怎么样了？"))

    def test_topic_overlap_offers(self):
        self.assertTrue(should_offer(topic_overlap=True))
        self.assertTrue(should_offer(topic_overlap=True, user_text="我房租最后怎么定的？"))
        archives = [{"summary": "跟房东谈房租，房东同意维持原价，不涨"}]
        self.assertTrue(index_overlap("我房租最后怎么定的？", archives))
        self.assertTrue(index_overlap("年糕后来好点了吗？", [{"summary": "猫叫年糕，开始吃东西"}]))
        self.assertFalse(index_overlap("今天天气怎么样？", archives))

    def test_empty_everything_skips(self):
        self.assertFalse(should_offer())

    def test_indexical_question_offers_index(self):
        self.assertTrue(should_offer(user_text="我那个麻烦事后来怎么样了？"))
        self.assertTrue(should_offer(user_text="家里那个小家伙还好吗？"))
        self.assertTrue(should_offer(user_text="我早上常去那家店叫什么来着？"))

    def test_plain_smalltalk_does_not_offer(self):
        self.assertFalse(should_offer(user_text="你还好吗？"))
        self.assertFalse(should_offer(user_text="今天天气不错。"))
        self.assertFalse(should_offer(user_text="你之前说慢慢来，我觉得挺对。"))

    def test_topic_question_without_reference_offers(self):
        text = "换工作的事有下文吗？"
        self.assertFalse(looks_like_reference_question(text))
        self.assertTrue(should_offer(user_text=text))

    def test_immediate_context_beats_topic_overlap(self):
        for text in ("刚才那个词什么意思？", "你上一句说的那个是什么？",
                     "这个说法对吗？", "刚才提到的那个例子能再说一遍吗？"):
            self.assertFalse(should_offer(topic_overlap=True, user_text=text), text)


class ReferenceQuestionTests(unittest.TestCase):
    def test_indexical_questions(self):
        for text in ("我那个麻烦事后来怎么样了？", "家里那个小家伙还好吗？",
                     "我那个新爱好还在坚持吗？", "我早上常去那家店叫什么来着？",
                     "我那个换工作的事有下文吗？"):
            self.assertTrue(looks_like_reference_question(text), text)

    def test_statements_and_greetings_are_not_questions(self):
        for text in ("你之前说慢慢来，我觉得挺对。", "你还好吗？",
                     "我之前和朋友聊过这家咖啡店。", "说个笑话吧。"):
            self.assertFalse(looks_like_reference_question(text), text)

    def test_cross_language_indexical_questions(self):
        for text in ("How's the little one at home?",
                     "What was that coffee shop I go to?",
                     "Did I tell you about my new hobby?",
                     "Do you remember the rent thing I mentioned?"):
            self.assertTrue(looks_like_reference_question(text), text)

    def test_ordinary_english_question_does_not_offer(self):
        self.assertFalse(looks_like_reference_question("Do you like coffee?"))
        self.assertFalse(looks_like_reference_question("What's your name?"))

    def test_immediate_context_references_do_not_offer(self):
        for text in ("刚才那个词什么意思？", "你上一句说的那个是什么？",
                     "这个说法对吗？", "刚才提到的那个例子能再说一遍吗？"):
            self.assertFalse(looks_like_reference_question(text), text)


class TopicIndexTests(unittest.TestCase):
    def test_caps_and_extracts_topic(self):
        archives = [{"session_id": f"s{i}", "summary": f"CEFR: A2 | 具体事件：话题{i}的细节 | 其他"}
                    for i in range(15)]
        items = build_topic_index(archives, max_items=12)
        self.assertEqual(len(items), 12)
        self.assertEqual(items[0].content, "话题0的细节")
        self.assertIn("archive:s0", items[0].source)

    def test_dedupes_session(self):
        archives = [{"session_id": "s1", "summary": "具体事件：房租"},
                    {"session_id": "s1", "summary": "具体事件：房租"}]
        self.assertEqual(len(build_topic_index(archives)), 1)

    def test_index_line_carries_last_state(self):
        items = build_topic_index([{
            "session_id": "s1",
            "summary": "用户跟房东谈房租，房东要涨15%",
            "last_user_text": "房东后来同意不涨了，维持原价。",
        }])
        self.assertIn("最后：", items[0].content)
        self.assertIn("同意不涨", items[0].content)

    def test_empty_index_renders_nothing(self):
        self.assertEqual(render_topic_index([]), "")

    def test_index_contract_present(self):
        text = render_topic_index([Evidence("猫：年糕开始吃东西", "archive:s1")])
        self.assertIn("话题索引", text)
        self.assertIn("不要拿别的事顶上", text)
        self.assertIn("archive:s1", text)
        self.assertIn("计划", text)
        self.assertIn("不要罗列索引", text)


class CoverageTests(unittest.TestCase):
    def test_same_sentence_is_full_coverage(self):
        self.assertGreaterEqual(coverage("我那个麻烦事后来怎么样了", "我那个麻烦事后来怎么样了"), 0.95)

    def test_unrelated_is_low(self):
        self.assertLess(coverage("我那个麻烦事后来怎么样了", "下个月去新加坡出差"), 0.2)

    def test_short_query_is_zero(self):
        self.assertEqual(coverage("好", "好"), 0.0)


class BuildEvidenceTests(unittest.TestCase):
    def test_caps_at_two_and_preserves_order(self):
        items = build_evidence(
            route_mode="episodic_recall",
            cross_session_recall=[archive("s1", "房租谈成了维持原价"),
                                  archive("s2", "猫开始吃东西了"),
                                  archive("s3", "吉他练到能弹一小段")],
        )
        self.assertEqual(len(items), 2)
        self.assertIn("s1", items[0].source)
        self.assertIn("s2", items[1].source)

    def test_drops_self_echo(self):
        items = build_evidence(
            user_text="我那个麻烦事后来怎么样了",
            route_mode="episodic_recall",
            cross_session_recall=[archive("s1", "我那个麻烦事后来怎么样了"),
                                  archive("s2", "房租谈成了维持原价")],
        )
        self.assertEqual(len(items), 1)
        self.assertIn("房租", items[0].content)

    def test_dedupes_same_source_and_content(self):
        items = build_evidence(
            route_mode="episodic_recall",
            cross_session_recall=[archive("s1", "房租谈成了维持原价"),
                                  archive("s1", "房租谈成了维持原价")],
        )
        self.assertEqual(len(items), 1)

    def test_open_thread_before_entity(self):
        items = build_evidence(
            open_threads=[{"thread_id": "t1", "topic": "房租", "detail": "还没定"}],
            related_entities=[{"entity_id": "e1", "name": "年糕", "detail": "猫"}],
        )
        self.assertIn("thread:t1", items[0].source)
        self.assertIn("entity:e1", items[1].source)

    def test_max_items_zero(self):
        self.assertEqual(build_evidence(route_mode="episodic_recall",
                                        cross_session_recall=[archive("s1", "x")],
                                        max_items=0), [])

    def test_status_and_reason_are_carried(self):
        items = build_evidence(
            route_mode="episodic_recall",
            cross_session_recall=[archive("s1", "房租谈成了维持原价", status="changed")],
        )
        self.assertEqual(items[0].status, "changed")
        self.assertEqual(items[0].reason, "explicit")


class RenderTests(unittest.TestCase):
    def test_empty_packet_renders_nothing(self):
        self.assertEqual(render_evidence([]), "")

    def test_contract_and_provenance_are_present(self):
        text = render_evidence([Evidence("房租谈成了维持原价", "archive:s1", "2026-10-20", "changed")])
        self.assertIn("不是待办", text)
        self.assertIn("archive:s1", text)
        self.assertIn("不要拿另一件事顶上", text)
        self.assertNotIn("未找到", text)

    def test_no_record_search_wording(self):
        text = render_evidence([Evidence("房租谈成了维持原价", "archive:s1")])
        self.assertNotIn("没找到相关记录", text)
        self.assertNotIn("检索", text)


if __name__ == "__main__":
    unittest.main()
