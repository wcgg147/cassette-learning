# -*- coding: utf-8 -*-
"""0919-01 §4 第 1 期（表达层这一条腿）的回归测试。

覆盖两件事（框架档 `plans/0919-01-表达层与情绪层框架方案.md` §4 第 1 期）：

1.1 修 `bugreports/0917-01-入戏只会宣告不会演.md`：
  - 合同加元叙述禁令（`[No Meta]`）；
  - 入戏/切角轮首句必须是角色内行动句或台词（`[Scene Entry]`，逐轮段）；
  - 现实流程要有实体动作；
  - 事后标记走既有 `_controlled_reply_violations` 同族判定（不新增调用、不整轮重试）。

1.2 合同分层：「AI 对**内容**有立场」（`[Stance - content]`）与
「对**语言**不评判」（`[Non-Judgment - language]`）各归各的段、各带正例。

§9 A/B 两件（同一框架档的收尾）：
  - `MachineLineSplitTests`：机器行（USER_INTENT/SOURCE/TERM/WANT_TO_SAY）粘行时
    **正文绝不为空**（第 1 期活体 8 轮命中 2 轮，非本期引入）；
  - `EmptyPromiseWindowTests`：空承诺判据从"单轮无实体"改成"相邻两轮窗口"
    （推进步骤口径，`bugreports/0917-01` §5-4 的残留）。

样例一律取自档案 §五 的**用户真机原话**（不另编）。
"""
from __future__ import annotations

import json
import unittest
import config  # noqa: E402


_PF_ORIG = None


def setUpModule():
    """本模块测「人格先行关闭」时的原始闲聊盘结构。

    ⛔ 用 setUpModule 而非模块级赋值：模块级赋值会在 import 时污染同进程
    所有 import 本模块（拿 control_ctx/stable_prepared）的测试。
    """
    global _PF_ORIG
    _PF_ORIG = getattr(config, "CHAT_TAPE_PERSONA_FIRST_ENABLED", True)
    config.CHAT_TAPE_PERSONA_FIRST_ENABLED = False


def tearDownModule():
    global _PF_ORIG
    if _PF_ORIG is not None:
        config.CHAT_TAPE_PERSONA_FIRST_ENABLED = _PF_ORIG

import agent.tutor_agent as tutor_agent
from agent import reply_quality
from agent.tutor_agent import (
    SessionContext,
    TutorAgent,
    _ADVANCE_STEP_RE,
    _DURATION_RECHECK_RE,
    _EMPTY_PROMISE_RE,
    _META_SELF_REPORT_RE,
    _ROLE_REMINDER_RE,
    _SELF_INTRO_RE,
    _USER_DURATION_RE,
    _empty_promise_across_turns,
    _promise_without_advance_step,
)
from agent.learning_orchestrator import LearningState

# ── 档案 §一 的四条真机回复（原样，不改写）──────────────────────────────────
USER_1 = "I ordered a latte on my phone earlier, the number is six five, can you give it to me?"
REPLY_1 = ("Oh, I see. You mean you ordered a latte on my phone earlier, and your order "
           "number is six five. I'll check if that's ready for you.")
USER_2 = "I ordered it fifty minutes ago so I think this drink is already ready"
REPLY_2 = ("You ordered this drink fifty minutes ago. Let me check how long it's been "
           "since you placed the order.")
REPLY_3 = ("I'm Alice, your friend who's chatting with you now. I'm here to chat with you "
           "about whatever you want, like your latte order or games.")
USER_4 = "刚才我们聊的时候，你不是进了咖啡店店员那个身份吗？"
REPLY_4 = ("哦，我刚才没切换成咖啡厅店员的身份，现在我来配合你。"
           "我这就帮你查六五号订单的拿铁，稍等一下。")
# 用户给出的"现实中该有的样子"（档案 §二）→ 合法台词基准
GOOD_LINE = "六五号拿铁是吧？我看一下——稍等，我给您出张小票。"


def flow_prepared(action: str = "continue_roleplay_scene") -> dict:
    """现实场景里"对方角色说话"那一轮的 prepared（确定性桩）。"""
    return {
        "policy": {"mode": "flow", "response_goal": "完成点单"},
        "learning_move": {"stage": "flow", "action": action, "goal": "完成点单"},
        "learner_state": {"communicative_intent": "request_help"},
        "constraints": {
            "conversation_director": {
                "speaker_position": "counterpart", "interaction_frame": "in_role",
            },
            "turn_contract": {
                "decision": {"primary_frame": "in_role", "speaker_position": "counterpart"}
            },
        },
    }


class MetaNarrativeGuardTests(unittest.TestCase):
    """1.1 的事后标记：真机三条各命中对应判据，合法台词零命中。

    **真机 ① 单轮仍不命中，且必须如实标出来**：`REPLY_1`（"You mean you ordered…
    I'll check if that's ready for you."）——
      · "空承诺"（§9 B 改成相邻两轮窗口后）要求**两轮都**只有承诺、没有推进步骤。
        它自己那一轮命中（复述物不再算推进步骤），但它的上一轮是句普通问句，
        窗口不成立；
      · 0916-27"复述当回应"要求零新内容词，而它带了 check/ready 两个新词。
    所以单轮的它落在"宁漏不误伤"的漏里；但**它后面再接一轮同样的空承诺**就命中
    （见 test_real_reply_1_alone_is_not_flagged_but_the_stall_pair_is）。
    """

    def test_each_real_reply_hits_its_own_rule(self) -> None:
        cases = [
            (REPLY_2, USER_2, "语义倒挂"),
            (REPLY_3, USER_2, "中途不要自我介绍"),
            (REPLY_4, USER_4, "元叙述"),
        ]
        previous = "What would you like to order?"
        for reply, user, needle in cases:
            with self.subTest(needle=needle):
                violations = TutorAgent._controlled_reply_violations(
                    reply, flow_prepared(), user, previous
                )
                self.assertTrue(
                    any(needle in item for item in violations),
                    f"未命中『{needle}』：{violations}",
                )

    def test_real_reply_1_alone_is_not_flagged_but_the_stall_pair_is(self) -> None:
        """真机 ① 单轮不判（§9 B 的相邻两轮窗口），**连着来两轮就判**。

        它自己那轮命中了单轮判据（"复述物不算推进步骤"），差的是"上一轮也命中"
        这个条件；把真机 ① 放到上一轮位置，本轮的同类空承诺就成立。

        注意本轮**不**用真机① 那句原话：它整句就嵌在 `REPLY_1` 里，
        `_controlled_reply_violations` 会先被 `_cross_turn_duplicate_ratio >= 0.8`
        那条判据**提前返回**（只报"几乎相同"，:7455 的既有行为），轮不到空承诺。
        这里换一句语义等价的承诺句，把窗口本身测出来。
        """
        self.assertEqual([], TutorAgent._controlled_reply_violations(
            REPLY_1, flow_prepared(), USER_1, "What would you like to order?",
        ))
        # 相邻两轮：上一轮 = 真机①（承诺 + 没有推进步骤）→ 本轮判违规
        self.assertTrue(any("空承诺" in v for v in TutorAgent._controlled_reply_violations(
            "Okay, let me look into that for you.", flow_prepared(), USER_1, REPLY_1,
        )))

    def test_user_approved_real_flow_line_is_clean(self) -> None:
        """档案 §二 用户给的三步形态（角色内动作 + 实体动作）不得被判违规。"""
        violations = TutorAgent._controlled_reply_violations(
            GOOD_LINE, flow_prepared(), USER_1, "What would you like to order?"
        )
        self.assertEqual([], violations)

    def test_meta_narrative_forms_from_the_archive(self) -> None:
        for text in [
            REPLY_4,
            "我来配合你，我这就切换成店员的身份。",
            "现在我来扮演咖啡店店员。",
            "我刚才没有切换成店员身份，现在补上。",
        ]:
            with self.subTest(text=text[:18]):
                self.assertTrue(_META_SELF_REPORT_RE.search(text), text)

    def test_legit_role_line_is_not_meta_narrative(self) -> None:
        self.assertFalse(_META_SELF_REPORT_RE.search(GOOD_LINE))
        self.assertFalse(_META_SELF_REPORT_RE.search("Your order's ready — six five."))

    def test_promise_predicate_only_counts_advance_steps(self) -> None:
        """**单轮**判据只问"有没有推进步骤"，不问"有没有可指认物"（§9 B）。

        收窄是双向的：复述物（编号、饮品、数字）**不再**算作动作——真机① 的
        `I'll check if that's ready for you.` 复述了 latte / six five，旧口径
        （复述物也算实体）把它漏掉，新口径单轮就命中。至于"单轮命中算不算违规"，
        由相邻两轮窗口决定（见 EmptyPromiseWindowTests）。
        """
        self.assertTrue(_promise_without_advance_step("I'll check if that's ready for you."))
        self.assertTrue(_promise_without_advance_step("让我看看。"))
        # 复述物不是推进步骤（旧口径会放过这两句）
        self.assertTrue(_promise_without_advance_step("Six five — let me check that for you."))
        self.assertTrue(_promise_without_advance_step("六五号拿铁是吧？我看一下。"))
        self.assertTrue(_promise_without_advance_step("我看一下。你的饮料是拿铁咖啡对吗？"))
        # 注："我来**看**一下"（中间多一个"来"）**不在** `_EMPTY_PROMISE_RE` 的承诺词表里
        # （词表只收"我/让我 + (去)? + 查/看/看看/确认"）。这是既有口径，本期不动；
        # 用户认可的三步形态里写的是"我看一下"，照旧被收。
        self.assertFalse(_EMPTY_PROMISE_RE.search("我来看一下。"))
        # 推进步骤在场 → 不是空承诺
        self.assertFalse(_promise_without_advance_step(GOOD_LINE))  # 我给您出张小票
        self.assertFalse(_promise_without_advance_step(
            "Let me check the pickup area for number six five."
        ))
        self.assertFalse(_promise_without_advance_step("麻烦您扫一下码，我这就去看。"))
        # 没有承诺话术 → 与空承诺无关
        self.assertFalse(_promise_without_advance_step(
            "Try saying: Excuse me, do you have a charger?"
        ))
        # 礼貌式"我这就帮您查"由元叙述判据管（不是承诺词表），但必须命中
        self.assertTrue(_META_SELF_REPORT_RE.search("我这就帮您查。"))
        self.assertTrue(_META_SELF_REPORT_RE.search("我这就帮你查六五号订单的拿铁。"))

    def test_spoken_number_is_not_an_advance_step(self) -> None:
        """口语编号是"复述可指认物"，按 §9 B 不再算推进步骤。

        Phase 1 曾把"成串数字"（six five / sixty-five / 六五号）当实体，为的是
        别让 `Six five — let me check that for you.` 这句**合法台词**被判空承诺。
        §9 B 把口径收窄后，这条保护改由**窗口**承担：单轮不判，连着两轮才判。
        副作用如实钉在这里——店员连着两轮只念编号 + 承诺会被标记一次
        （代价是白跑一次本地修复，不是答错）。
        """
        self.assertFalse(_ADVANCE_STEP_RE.search("Six five — let me check that for you."))
        self.assertFalse(_ADVANCE_STEP_RE.search("Let me check for sixty-five."))
        self.assertFalse(_ADVANCE_STEP_RE.search("六五号拿铁是吧？我看一下。"))
        # 单轮：上一轮是普通问句 → 窗口不成立
        self.assertFalse(_empty_promise_across_turns(
            "Six five — let me check that for you.", "What would you like to order?",
        ))
        # 连着两轮 → 判（口径收窄的已知代价，不是意外）
        self.assertTrue(_empty_promise_across_turns(
            "Six five, let me check that for you.",
            "Six five — let me check that for you.",
        ))

    def test_advance_step_is_narrow_and_does_not_swallow_restatement(self) -> None:
        """推进步骤词表要窄：只收"把流程往前推"的动作，不收复述物/话题词。"""
        for text in ["取餐台", "扫码", "递单纸", "出张小票", "麻烦您扫一下码"]:
            with self.subTest(text=text):
                self.assertTrue(_ADVANCE_STEP_RE.search(text))
        for text in [
            "你的号码是六五号", "六五号拿铁是吧", "Six five",
            "One moment — let me check for you.",
        ]:
            with self.subTest(text=text):
                self.assertFalse(_ADVANCE_STEP_RE.search(text))

    def test_duration_restated_as_pending_check(self) -> None:
        self.assertTrue(_USER_DURATION_RE.search(USER_2))
        self.assertTrue(_DURATION_RECHECK_RE.search(REPLY_2))
        # 只给事实、不把它当待查项 → 不判违规
        self.assertFalse(TutorAgent._controlled_reply_violations(
            "Fifty minutes — it's ready then, six five.",
            flow_prepared(), USER_2, "What would you like to order?",
        ))

    def test_midway_self_introduction_flagged_but_not_opening(self) -> None:
        self.assertTrue(_SELF_INTRO_RE.search(REPLY_3))
        # 开场轮自我介绍是合同允许的（previous_reply 为空 → 不判）
        opening = TutorAgent._controlled_reply_violations(
            "Hi! I'm Alice, your English tutor. What should I call you?",
            {"learning_move": {}, "constraints": {}}, "hello", "",
        )
        self.assertEqual([], opening)

    def test_no_false_positive_on_ordinary_lines(self) -> None:
        for text in [
            "I'm sorry, I don't have that number. Let me look again.",
            "I'm looking for your order now.",
            "这里用 do you have 来询问店里是否有货，比 have 更自然。",
            "Sure, let's switch. Do you like noodles or rice better?",
        ]:
            with self.subTest(text=text[:22]):
                self.assertFalse(_SELF_INTRO_RE.search(text))

    def test_role_reminder_detector_is_narrow(self) -> None:
        self.assertTrue(_ROLE_REMINDER_RE.search(USER_4))
        self.assertTrue(_ROLE_REMINDER_RE.search("you're the barista, right?"))
        for text in ["我今天想聊点别的，换个话题吧。", "这个咖啡好喝。", "我们继续聊吧。"]:
            with self.subTest(text=text):
                self.assertFalse(_ROLE_REMINDER_RE.search(text))


class EmptyPromiseWindowTests(unittest.TestCase):
    """§9 B：空承诺从"单轮无实体"改成"**相邻两轮窗口**"。

    用户验收口径：**相邻两轮命中判违规、单轮不判**。
    单轮里"我这就去查"可以是正常人的过渡话；连着两轮只承诺、一步没推进，
    才是真缺陷。这也是「六五号拿铁是吧？我看一下」不再被误伤的原因。
    """

    PREV_QUESTION = "What would you like to order?"

    def _flagged(self, reply: str, previous: str) -> bool:
        return any("空承诺" in v for v in TutorAgent._controlled_reply_violations(
            reply, flow_prepared(), USER_1, previous,
        ))

    def test_single_turn_is_never_a_violation(self) -> None:
        """单轮不判——包括用户亲自认可的三步形态与真机① 的原句。"""
        for text in [
            "I'll check if that's ready for you.",
            "六五号拿铁是吧？我看一下。",
            "让我看看。",
        ]:
            with self.subTest(text=text[:20]):
                self.assertFalse(_empty_promise_across_turns(text, self.PREV_QUESTION))
                self.assertFalse(self._flagged(text, self.PREV_QUESTION))

    def test_two_adjacent_stalls_are_a_violation(self) -> None:
        """相邻两轮都只有承诺、都没有推进步骤 → 判违规（真机 ①② 相邻那一对）。"""
        previous = REPLY_1  # 真机①："… I'll check if that's ready for you."
        reply = REPLY_2     # 真机②："… Let me check how long it's been …"
        self.assertTrue(_empty_promise_across_turns(reply, previous))
        self.assertTrue(self._flagged(reply, previous))

    def test_one_advance_step_in_either_turn_clears_it(self) -> None:
        """两轮里只要有一轮落了推进步骤（取餐台/扫码/递单/小票），窗口就不成立。"""
        # 本轮落子
        self.assertFalse(self._flagged(GOOD_LINE, REPLY_1))
        # 上一轮落子
        self.assertFalse(self._flagged(
            "I'll check for you.", "Let me check the pickup area — yes, six five is here.",
        ))
        # 上一轮为空（首轮）→ 没有"相邻"可言
        self.assertFalse(_empty_promise_across_turns("I'll check for you.", ""))


class MachineLineSplitTests(unittest.TestCase):
    """§9 A：机器行粘行不再把正文整行吃掉（硬要求：剥完 candidate 不得为空）。

    三类构造输入逐个钉住：(a) 正常两行 + 正文；(b) SOURCE 与正文粘行；
    (c) 只有两行、没有正文。三类都**不许**出现"静默空串"。
    """

    def test_a_normal_two_lines_plus_body(self) -> None:
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: Do you have a name?\n"
            "USER_INTENT_SOURCE: 你有名字吗\n"
            "\n"
            "Sure — do you want to give me one?\n"
        )
        self.assertEqual("Sure — do you want to give me one?", body.strip())
        self.assertEqual("Do you have a name?", intent)
        self.assertEqual("你有名字吗", span)

    def test_b_source_glued_to_the_body(self) -> None:
        """粘行时**只剥标签本身**：正文非空且正确，出处置空（→ 意图不可信降级）。

        出处置空不是缺陷：plans/0915-23 的设计里，交代不出真实出处的意图行
        就是不可信的，调用方会退回用户原话——这正是想要的降级方向。
        """
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: Do you have a name?\n"
            "USER_INTENT_SOURCE: 你有名字吗 Sure — do you want to give me one?"
        )
        self.assertTrue(body.strip(), "正文被吃空了")
        self.assertIn("Sure — do you want to give me one?", body)
        self.assertNotIn("USER_INTENT_SOURCE", body)
        # 意图行本身没被 SOURCE 污染
        self.assertEqual("Do you have a name?", intent)
        self.assertEqual("", span)

    def test_b_real_dump_shape_both_lines_on_one_line(self) -> None:
        """历史 dump `_dsf8.txt` 的实测形态：两行 + 正文全挤在一行。

        `USER_INTENT: 问价格怎么说 USER_INTENT_SOURCE: mixed You just said …`
        老写法会把整行收成 source_span、正文变空。这里钉住正文被救回来。
        """
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: 问价格怎么说 USER_INTENT_SOURCE: mixed "
            "You just said it perfectly — 'How much is it?' Now try: "
            "'How much is it for a large?'"
        )
        self.assertTrue(body.strip(), "正文被吃空了")
        self.assertIn("How much is it for a large?", body)
        self.assertNotIn("USER_INTENT_SOURCE", body)
        self.assertEqual("问价格怎么说", intent)
        self.assertEqual("", span)

    def test_c_machine_lines_only_no_body(self) -> None:
        """只有两行、没有正文：**不是静默空串**，而是"标签剥掉、内容留给正文"，
        出处置空 → 调用方按"意图不可信"退回用户原话。"""
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: Hi there\nUSER_INTENT_SOURCE: 你好"
        )
        self.assertTrue(body.strip(), "静默空串 = 这一轮没话说，正是要避免的")
        self.assertNotIn("USER_INTENT_SOURCE", body)
        self.assertEqual("Hi there", intent)
        self.assertEqual("", span, "出处置空才会让调用方走意图降级（兜底）")

    def test_clean_shapes_are_unchanged(self) -> None:
        """干净形态零改动的护栏（既有 test_transcript_fidelity / test_term_anchor 同口径）。"""
        # 缺 SOURCE 行：意图照收、出处为空（触发保真校验的拒绝路径）
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: Hi there\n\nHello!\n"
        )
        self.assertEqual("Hi there", intent)
        self.assertEqual("", span)
        self.assertIn("Hello!", body)
        # SOURCE 行的下一行不是正文（这里是 TERM 行）——仍算干净，出处照收
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: Evening! Did you play COD lately?\n"
            "USER_INTENT_SOURCE: 你最近玩使命召唤了吗\n"
            "TERM: COD=Call of Duty\n\nEvening! Did you play COD lately?"
        )
        self.assertEqual("你最近玩使命召唤了吗", span)
        self.assertIn("TERM: COD=Call of Duty", body)

    def test_sibling_splitters_never_empty_the_body(self) -> None:
        """同族剥行函数同一要求（§9 A.2）：剥完不得为空。"""
        rest, anchors = TutorAgent._split_term_line("TERM: COD=Call of Duty")
        self.assertTrue(rest.strip(), "TERM 行独占全文时正文被吃空了")
        self.assertEqual("COD=Call of Duty", rest.strip())
        self.assertEqual({"COD": "Call of Duty"}, anchors)
        rest, want = TutorAgent._split_want_to_say_line("WANT_TO_SAY: toilet paper")
        self.assertTrue(rest.strip(), "WANT_TO_SAY 行独占全文时正文被吃空了")
        self.assertIsNotNone(want)
        self.assertEqual("toilet paper", want["english"])

    # ── 真机 2026-10-09：重复机器行块必须丢干净（来自 alice_chat_validation_fix）──

    def test_d_duplicated_machine_block_is_dropped_entirely(self) -> None:
        """重复块（真机 2026-10-09）：同一段两行块被连写多遍时，余下的整块必须丢干净。

        现场：模型把 USER_INTENT / USER_INTENT_SOURCE 两行块连写 5 遍；只摘一遍的
        旧写法把余下 4 块留在正文里，`_collapse_repeated_reply` 压成一份后
        **连标签一起念给学习者**（气泡里出现 "USER_INTENT: … USER_INTENT_SOURCE: …"）。
        这里钉住：正文里一个标签都不许剩。这一格允许正文为空——重复块本身不是
        正文，由上游导演草稿兜底，绝不为了"不空"把机器行念出去。
        """
        one = (
            "USER_INTENT: But the weather forecast says heavy rain. "
            "Do you think I should still go?\n"
            "USER_INTENT_SOURCE: 但天气预报说下大雨。你觉得我还该去吗\n"
        )
        body, intent, span = TutorAgent._split_all_user_intent_lines(one * 5)
        self.assertEqual("", body.strip(), "重复块必须整段丢弃，不能留给 TTS")
        self.assertNotIn("USER_INTENT", body)
        self.assertEqual(
            "But the weather forecast says heavy rain. Do you think I should still go?",
            intent,
        )

    def test_d2_duplicated_block_before_a_real_body_keeps_the_body(self) -> None:
        """重复块后面还有真正的口语回复：只丢重复块，正文原样留下。"""
        one = "USER_INTENT: Do you have a name?\nUSER_INTENT_SOURCE: 你有名字吗\n"
        body, intent, span = TutorAgent._split_all_user_intent_lines(
            one * 3 + "\nSure — do you want to give me one?\n"
        )
        self.assertEqual("Sure — do you want to give me one?", body.strip())
        self.assertNotIn("USER_INTENT", body)
        self.assertEqual("Do you have a name?", intent)

    def test_d3_single_pass_function_behaviour_is_untouched(self) -> None:
        """护栏：单遍函数（`_split_user_intent_line`）的行为逐字不变。

        不动点逻辑只加在包装里，老函数的形状照旧（防止这次修完把老口径改掉）。
        """
        body, intent, span = TutorAgent._split_user_intent_line(
            "USER_INTENT: Hi there\nUSER_INTENT_SOURCE: 你好\n\nHello!\n"
        )
        self.assertEqual("Hello!", body.strip())
        self.assertEqual("Hi there", intent)
        self.assertEqual("你好", span)
        # 只有两行、没有正文：仍是"标签剥掉、内容留给正文"（不是空串）
        body2, intent2, span2 = TutorAgent._split_user_intent_line(
            "USER_INTENT: Hi there\nUSER_INTENT_SOURCE: 你好"
        )
        self.assertTrue(body2.strip())
        self.assertNotIn("USER_INTENT_SOURCE", body2)
        self.assertEqual("", span2)


def control_ctx(roleplay_turn: int, role: str = "咖啡店店员") -> tuple[TutorAgent, SessionContext]:
    agent = TutorAgent.__new__(TutorAgent)
    ctx = SessionContext(
        session_id="expr-layer", user_id="expr-layer-user", target_lang="en",
        scene={"name": "coffee_shop", "goal": "点单"},
    )
    ctx.learning_state = LearningState(stage="flow", mission="点单")
    ctx.roleplay_turn = roleplay_turn
    ctx.scenario_role = role
    return agent, ctx


def stable_prepared(action: str) -> dict:
    prepared = flow_prepared(action)
    prepared["constraints"]["turn_contract"]["latest_input"] = "…"
    return prepared


class SceneEntrySectionTests(unittest.TestCase):
    """1.1 的形态段：入戏/切角轮挂上，其余轮零字符增量。"""

    def _prompt(self, prepared: dict, turn: int, transcript: str = "") -> str:
        agent, ctx = control_ctx(turn)
        return agent._build_control_prompt(ctx, prepared, transcript)

    def test_present_on_scene_establish_turn(self) -> None:
        prompt = self._prompt(stable_prepared("establish_roleplay_scene"), 0)
        self.assertIn("[Scene Entry]", prompt)
        self.assertIn("第一句必须就是角色内的行动句或台词", prompt)
        self.assertIn("咖啡店店员", prompt)

    def test_present_when_user_reminds_about_the_role(self) -> None:
        prompt = self._prompt(stable_prepared("continue_roleplay_scene"), 5, USER_4)
        self.assertIn("[Scene Entry]", prompt)

    def test_present_on_first_counterpart_turn(self) -> None:
        prompt = self._prompt(stable_prepared("supply_counterpart_turn"), 1)
        self.assertIn("[Scene Entry]", prompt)

    def test_absent_five_turns_into_the_role(self) -> None:
        prompt = self._prompt(stable_prepared("continue_roleplay_scene"), 5, "six five please")
        self.assertNotIn("[Scene Entry]", prompt)

    def test_sits_after_turn_contract_not_in_stable_prefix(self) -> None:
        """逐轮变的规矩必须挂在 turn_contract_section 之后（缓存前缀口径）。"""
        prompt = self._prompt(stable_prepared("establish_roleplay_scene"), 0)
        self.assertLess(prompt.index("[Latest Turn Contract"), prompt.index("[Scene Entry]"))
        self.assertLess(prompt.index("[Role] 只输出说给用户听的话"), prompt.index("[Scene Entry]"))


class ContractLayeringTests(unittest.TestCase):
    """1.2 合同分层：内容有立场 / 语言不评判，各归各的段、各带正例。"""

    def _all_four_branches(self, action: str = "engage_general_topic") -> list[tuple[str, str]]:
        out = []
        original = tutor_agent._PROMPT_CACHE_ORDER
        prepared_base = stable_prepared(action)
        for authoritative in (True, False):
            for cache in (1, 0):
                prepared = {
                    **prepared_base,
                    "constraints": {**prepared_base["constraints"]},
                }
                if authoritative:
                    prepared["conversation_director"] = {
                        "_meta": {"source": "director_model"}, "response_need": "listen_and_continue",
                    }
                else:
                    prepared["conversation_director"] = {"response_need": "listen_and_continue"}
                agent, ctx = control_ctx(4)
                try:
                    tutor_agent._PROMPT_CACHE_ORDER = cache
                    out.append((
                        f"authoritative={authoritative}/cache={cache}",
                        agent._build_control_prompt(ctx, prepared, ""),
                    ))
                finally:
                    tutor_agent._PROMPT_CACHE_ORDER = original
        return out

    def test_both_layers_present_in_all_four_branches(self) -> None:
        branches = self._all_four_branches()
        self.assertEqual(4, len(branches))
        for label, prompt in branches:
            with self.subTest(branch=label):
                self.assertIn("[Stance - content]", prompt)
                self.assertIn("[Non-Judgment - language]", prompt)
                self.assertIn("[No Meta]", prompt)

    def test_each_layer_carries_a_positive_example(self) -> None:
        for label, prompt in self._all_four_branches():
            with self.subTest(branch=label):
                stance = prompt[prompt.index("[Stance - content]"):]
                stance = stance[:stance.index("[Non-Judgment")]
                non_judgment = prompt[prompt.index("[Non-Judgment - language]"):]
                non_judgment = non_judgment[:non_judgment.index("[No Meta]")]
                self.assertIn("✅", stance, "立场段缺正例")
                self.assertIn("✅", non_judgment, "不评判段缺正例")
                # 三层各有不同的作用域，且互相点名（不写清分层必然混）
                self.assertIn("内容", stance)
                self.assertIn("语言", non_judgment)

    def test_non_judgment_keeps_g1_three_rules_and_correction_rhythm(self) -> None:
        for label, prompt in self._all_four_branches():
            with self.subTest(branch=label):
                for needle in ["用户跳过的话题不绕回来", "只纠第一次",
                               "直接换方向", "绝不要求用户重说一遍", "不评判 ≠ 不纠错"]:
                    self.assertIn(needle, prompt)

    def test_stable_layers_come_before_per_turn_sections(self) -> None:
        """稳定段（立场/不评判/元叙述禁令）必须进公共前缀，不能在逐轮段之后。"""
        for label, prompt in self._all_four_branches():
            with self.subTest(branch=label):
                for tag in ["[Stance - content]", "[Non-Judgment - language]", "[No Meta]"]:
                    self.assertLess(prompt.index(tag), prompt.index("[Latest Turn Contract"))


class AcceptanceListTests(unittest.TestCase):
    """`bugreports/0917-01` §五 的五条验收 + 用户追加的一条，逐条落成断言。

    每条都注明**由谁承担**：第 1 期新增的判据，还是既有家族（0916-27 三律）本就有。
    写清归属是为了将来回归失败时知道该改哪一处。
    """

    PREV = "What would you like to order?"

    def test_s5_1_no_declaration_of_role_or_plan(self) -> None:
        """[ ] 重放同一段对话：回复里不得出现"我来配合你/我这就切换/我这就帮你查"。

        承担者：第 1 期新增 `[No Meta]`（生成侧）+ `_META_SELF_REPORT_RE`（事后标记）。
        """
        for text in [REPLY_4, "我来配合你，这就查六五号。", "现在我来扮演咖啡店店员。"]:
            with self.subTest(text=text[:16]):
                self.assertTrue(_META_SELF_REPORT_RE.search(text))
        self.assertFalse(TutorAgent._controlled_reply_violations(
            GOOD_LINE, flow_prepared(), USER_4, self.PREV,
        ))

    def test_s5_2_action_line_when_user_reminds_the_role(self) -> None:
        """[ ] 提醒入戏后，**必须**出现角色内台词（"六五号拿铁是吧？我看一下"级形态）。

        生成侧：第 1 期新增 `[Scene Entry]`（提醒轮必挂，见 SceneEntrySectionTests）；
        事后侧：空承诺判据。活体验证（2026-09-19，真模型）提醒轮实测得
        “六五号拿铁是吧？我看一下——稍等，我去取餐台找找。”——正是这一形态。
        """
        agent, ctx = control_ctx(5)
        prompt = agent._build_control_prompt(
            ctx, stable_prepared("continue_roleplay_scene"), USER_4,
        )
        self.assertIn("[Scene Entry]", prompt)
        self.assertFalse(_promise_without_advance_step(GOOD_LINE))

    def test_s5_3_user_given_fact_not_treated_as_pending(self) -> None:
        """[ ] 不得把用户刚说过的事实当成"待查项"（"让我查下过了多久"）。

        承担者：第 1 期新增 `_USER_DURATION_RE` × `_DURATION_RECHECK_RE`。
        """
        self.assertTrue(_USER_DURATION_RE.search(USER_2))
        self.assertTrue(_DURATION_RECHECK_RE.search(REPLY_2))
        violations = TutorAgent._controlled_reply_violations(
            REPLY_2, flow_prepared(), USER_2, self.PREV,
        )
        self.assertTrue(any("语义倒挂" in v for v in violations), violations)

    def test_s5_4_restatement_must_not_carry_the_whole_turn(self) -> None:
        """[ ] 复述句不得承担整轮回复（要么给新信息，要么别复述）。

        覆盖是四段拼的（§9 B 补上了真机① 那一段）：
          · 纯复述（整轮零新内容词）→ 既有 0916-27 三律抓得住（断言 ①）；
          · 真机 ②（复述 + 时长倒查）→ "语义倒挂"抓住（断言 ②）；
          · 真机 ①（复述 + 空承诺）→ **相邻两轮窗口**：单轮不判（断言 ③），
            它后面再接一轮同类空承诺就判（断言 ⑤，§9 B 新增）；
          · 提示侧 [Scene Entry]"禁止先复述用户刚说的话"只覆盖入戏/切角轮（断言 ④）。
        残留如实记：**孤立一轮**的"复述 + 空承诺"仍不判——用户口径是宁漏不误伤，
        单轮的"我这就去查"可以是正常过渡话，这条漏是口径本身的一部分。
        """
        # ① 纯复述形态：既有三律抓得到
        pure = "Oh, you ordered a latte on your phone earlier, and the number is six five."
        self.assertTrue(any("复述当回应" in v for v in reply_quality.reply_quality_violations(
            USER_1, pure, raw_text="", scene_role=True,
        )))
        # ② 真机② 的形态：语义倒挂抓住
        self.assertTrue(any("语义倒挂" in v for v in TutorAgent._controlled_reply_violations(
            REPLY_2, flow_prepared(), USER_2, self.PREV,
        )))
        # ③ 真机① 单轮：不判（上一轮不是承诺句，窗口不成立）
        self.assertEqual([], TutorAgent._controlled_reply_violations(
            REPLY_1, flow_prepared(), USER_1, self.PREV,
        ))
        # ④ 提示侧：入戏轮禁止首句复述
        agent, ctx = control_ctx(0)
        entry = agent._build_control_prompt(
            ctx, stable_prepared("establish_roleplay_scene"), "", )
        self.assertIn("禁止先复述用户刚说的话", entry)
        # ⑤ §9 B：真机① 那一轮后面再接一轮空承诺 → 命中（卡住不动这一形态）
        self.assertTrue(any("空承诺" in v for v in TutorAgent._controlled_reply_violations(
            "Okay, let me look into that for you.", flow_prepared(), USER_1, REPLY_1,
        )))

    def test_s5_5_no_midway_self_introduction(self) -> None:
        """[ ] 对话中途不得出现自我介绍。

        承担者：第 1 期新增 `_SELF_INTRO_RE`（仅 previous_reply 非空时判）。
        """
        self.assertTrue(_SELF_INTRO_RE.search(REPLY_3))
        self.assertTrue(any("自我介绍" in v for v in TutorAgent._controlled_reply_violations(
            REPLY_3, flow_prepared(), USER_2, self.PREV,
        )))

    def test_s5_6_stance_on_content_without_judging_language(self) -> None:
        """[＋] AI 表达自己的观点，但不评判用户的语言错误（用户追加的一条）。

        两层必须**同时在场且各归各段**——只写一层就会出现用户点名的两种坏结果：
        写成"随和"→ 连语言错误都不纠了；写成"爱发表意见"→ 开始杠用户的观点。
        （活体验证：用户说 "I think morning shift is better than night shift"
        且句里带 in morning 这类小错，实测回复只回内容、未点评语言。）
        """
        agent, ctx = control_ctx(4)
        prompt = agent._build_control_prompt(
            ctx, stable_prepared("engage_general_topic"), "", )
        # [Stance] 段里**点名**了不评判段（"另见 [Non-Judgment - language]"），
        # 所以切片要跳过这处交叉引用、取第二次出现才是真正的段头。
        start = prompt.index("[Stance - content]")
        first_nj = prompt.index("[Non-Judgment - language]")
        second_nj = prompt.index("[Non-Judgment - language]", first_nj + 1)
        stance = prompt[start:second_nj]
        non_judgment = prompt[second_nj:]
        non_judgment = non_judgment[:non_judgment.index("[No Meta]")]
        # 立场：明确允许不同意、禁止顺从式附和，并交叉引用语言层（分层要互相点名，
        # 否则模型会把两层混成一层）
        for needle in ["可以不同意", "不要为了顺从而附和", "另见 [Non-Judgment - language]"]:
            self.assertIn(needle, stance)
        # 不评判：明确不等于不纠错，纠错节奏仍由 [Teaching] 管
        for needle in ["只纠第一次", "不评判 ≠ 不纠错", "[Teaching]"]:
            self.assertIn(needle, non_judgment)



# ═══════════════════════════════════════════════════════════════════════════
# 第 2 期 B 批（plans/0919-02 §4.4/§4.5）：练表达盘 + 支架递减
#
# 这一批把"练表达"从概念落成**第三种盘型**。后端可离线断言的是两件事：
#   1. 档位判定必须是**纯函数**（`engine/expression_scaffold.py`）：三态、连续性、
#      封顶封底——档位由代码算才权威，模型只许引用（§4.5）；
#   2. 练习盘的盘内合同（`[Expression Practice]`）必须挂上**四个 prompt 分支**、
#      且**闲聊盘逐字零增量**（照 `[Scene Entry]` 同款空串过滤，缓存前缀不受影响，§5 硬约束 1）。
# 前端接线（盘型入口 / 显式提交按钮 / 20s 兜底）是静态契约，落点在同批新增的
# `test_runner/test_expression_tape_wiring.py`（与 `test_continue_tape_wiring` 同一形状）。
# ═══════════════════════════════════════════════════════════════════════════

from engine import expression_scaffold as _scaffold


def _scaffold_state(tier: int, **over) -> dict:
    """构造一份档位状态（`tier` 直接给定，不走 decide_tier）。"""
    state = {
        "help_tier": tier, "met_streak": 0, "miss_streak": 0,
        "last_reason": "", "observations": 0,
    }
    state.update(over)
    return state


# 一次"**达标**"的对比载荷（按 2026-09-20 修订后的判据：密度上升 + details 不降）。
# 旧写法只给 `{"verdict": {"available": True, "concise_progress": True}}`——
# 新判据读的是 `delta.density`，缺了它会被当"不可用" ⇒ 一次写都不发。
_MET_COMPARE: dict = {
    "verdict": {"available": True},
    "delta": {"density": 1.20, "details": 14},
}


def expression_ctx(tier: int = 1) -> tuple[TutorAgent, SessionContext]:
    """练习盘的 (agent, ctx)：盘型只认 `cassette_mode`，档位先落在 ctx 上。"""
    agent, ctx = control_ctx(4)
    ctx.cassette_mode = "expression"
    ctx.expression_scaffold = _scaffold_state(tier)
    return agent, ctx


class ExpressionScaffoldTierOneTests(unittest.TestCase):
    """档 1 发出去的东西**不许给判据送分**（plans/0920-02 §8.6）。

    档 1 原先是"给 3-5 个关键词"，实测那些词（含内容词）**直接进 `details` 分子**：
    同一件事两遍的密度 2.00 → 3.00、判据送 `met=True / density_up` ⇒ "达标"里混进了
    "他把我们给的词念了一遍"（`bugreports/probe_opening_keywords_metrics_0920.py`）。
    现在档 1 只发：骨架（中文元提示）+ 连接词，且每个连接词都必须是**衔接词表认的**——
    衔接词只作观测量、不进 `details`/密度。这里**逐个重算**钉住，不靠注释。
    """

    def test_every_tier_one_connective_scores_zero_details(self) -> None:
        from engine.chunk_evidence import count_expression_metrics

        for phrase in _scaffold.SCAFFOLD_TIER1_CONNECTIVES:
            with self.subTest(phrase=phrase):
                metrics = count_expression_metrics(phrase.capitalize() + ".")
                self.assertGreaterEqual(
                    int(metrics["connectives"]), 1,
                    "%r 不在衔接词表里 —— 档 1 发的词必须是 `_CONNECTIVE_*` 认的衔接词"
                    % phrase)
                self.assertEqual(
                    0, int(metrics["details"]),
                    "%r 会进 details 分子 ⇒ 档 1 又在给判据送分" % phrase)

    def test_the_frame_is_meta_guidance_not_content(self) -> None:
        """骨架说的是"怎么组织"，不是"讲什么"：不得含任何目标语言内容词。"""
        frame = _scaffold.SCAFFOLD_TIER1_FRAME
        self.assertNotRegex(frame, r"[A-Za-z]",
                            "骨架里混进了目标语言（它只该描述结构，不该给要用的词）")
        for needle in ("先", "再", "最后"):
            self.assertIn(needle, frame)

    def test_tier_one_instruction_forbids_content_words(self) -> None:
        """指示必须**写明禁给内容词**——这是 §8.6 的护栏，不是措辞偏好。

        指示放宽一次（回到"给关键词或词组"），档 1 就又变成给判据送分。
        """
        row = _scaffold.tier_row(1)
        self.assertIn("原样照抄", row["instruction"])
        self.assertIn("只给连接词", row["instruction"])
        self.assertIn("内容词", row["instruction"])
        # 样例里那两样必须**来自常量**（单一事实源），不是手抄的字面串
        self.assertIn(_scaffold.SCAFFOLD_TIER1_FRAME, row["example"])
        for phrase in _scaffold.SCAFFOLD_TIER1_CONNECTIVES:
            self.assertIn(phrase, row["example"])


class ExpressionScaffoldTierTests(unittest.TestCase):
    """§4.5 档位判定：初始 1、连续两次达标 ↓帮助、连续两次不达标 ↑帮助、量不到不动。

    **看数先看这句**：`help_tier` 是"帮助由多到少"的序号——
    1 = 帮助最多（给三步骨架 + 几个固定连接词），4 = 帮助最少（无提示）。
    所以 §5 验收那句"档位下降一档（帮助减少）"= `help_tier` **+1**。
    """

    def test_initial_tier_is_one_most_help(self) -> None:
        """§4.5：初始档 = 1（帮助最多）。缺失/坏输入一律退回 1，绝不抛。"""
        self.assertEqual(1, _scaffold.initial_state()["help_tier"])
        self.assertEqual(1, _scaffold.state_from_row({})["help_tier"])
        self.assertEqual(1, _scaffold.state_from_row(None)["help_tier"])
        self.assertEqual(
            1, _scaffold.state_from_row({"last_evidence": "not json"})["help_tier"])
        self.assertEqual(1, _scaffold.clamp_tier(None))
        self.assertEqual(1, _scaffold.clamp_tier("??"))

    def test_two_consecutive_met_lower_help_by_one_tier(self) -> None:
        """§5 验收原句：连讲两次达标后档位下降一档（帮助减少）。"""
        state = _scaffold.decide_tier(_scaffold.initial_state(), True, "concise_progress")
        self.assertEqual(1, state["help_tier"], "一次达标不降档——判据是连续两次")
        self.assertEqual(1, state["met_streak"])
        state = _scaffold.decide_tier(state, True, "concise_progress")
        self.assertEqual(2, state["help_tier"], "连续两次达标 → 帮助减少一档")
        self.assertEqual(0, state["met_streak"], "降档后连胜从零重算（不许攒着）")

    def test_two_consecutive_miss_raise_help_back(self) -> None:
        """连续两次不达标 → 帮助增加一档（`help_tier` -1）。"""
        state = _scaffold.decide_tier(_scaffold.initial_state(), True, "")
        state = _scaffold.decide_tier(state, True, "")           # → 第 2 档
        self.assertEqual(2, state["help_tier"])
        state = _scaffold.decide_tier(state, False, "details_dropped")
        self.assertEqual(2, state["help_tier"], "一次不达标不升帮助")
        state = _scaffold.decide_tier(state, False, "details_dropped")
        self.assertEqual(1, state["help_tier"], "连续两次不达标 → 帮助增加一档")

    def test_alternating_verdicts_never_reach_a_streak_of_two(self) -> None:
        """达标/不达标交错 → 两条连胜互斥清零，谁都不攒到 2，档位不动。"""
        state = _scaffold.initial_state()
        for met in (True, False, True, False, True):
            state = _scaffold.decide_tier(state, met, "")
            self.assertEqual(1, state["help_tier"], "交错时档位不动")
        self.assertLess(state["met_streak"], 2)
        self.assertLess(state["miss_streak"], 2)

    def test_unavailable_neither_advances_nor_resets(self) -> None:
        """`met=None`（密度不可用 / 跨口径）**既不推进也不清连胜**。

        这是 §4.2「拿不到就标 unavailable，不伪造」在档位层的同一条纪律：
        不许把"量不到"读成"他没做到"。
        """
        state = _scaffold.decide_tier(_scaffold.initial_state(), True, "")
        self.assertEqual(1, state["met_streak"])
        state = _scaffold.decide_tier(state, None, _scaffold.REASON_VERDICT_UNAVAILABLE)
        self.assertEqual(1, state["met_streak"], "量不到：已验证的那次连胜保留")
        self.assertEqual(1, state["help_tier"], "量不到：档位不动")
        self.assertEqual(1, state["observations"], "不可用不计入 observations")
        # 再补一次达标 → 仍按"连续两次"降档（上一次达标没被抹掉）
        state = _scaffold.decide_tier(state, True, "")
        self.assertEqual(2, state["help_tier"], "不可用不改判据：仍凑满连续两次")

    def test_tier_is_clamped_at_both_ends(self) -> None:
        """封顶 4 / 封底 1（越界输入也夹住）。"""
        top = _scaffold_state(4)
        top = _scaffold.decide_tier(top, True, "")
        top = _scaffold.decide_tier(top, True, "")
        self.assertEqual(4, top["help_tier"], "已在最末档：不得越界到 5")
        bottom = _scaffold.decide_tier(_scaffold_state(1), False, "")
        bottom = _scaffold.decide_tier(bottom, False, "")
        self.assertEqual(1, bottom["help_tier"], "已在第一档：不得越界到 0")
        self.assertEqual(4, _scaffold.clamp_tier(99))
        self.assertEqual(1, _scaffold.clamp_tier(-3))

    @staticmethod
    def _compare(*, d_first, d_second, det_first, det_second, available=True) -> dict:
        """造一份 F14 对比载荷的最小形状（只需 `verdict` + `delta` 两处）。"""
        return {
            "verdict": {
                "available": available,
                # 这一格**故意留着旧字段**：判据修订后本模块不再读它，
                # 留着是为了证明"读的是密度差、不是 concise_progress"。
                "concise_progress": None,
                "reason": "fixture",
            },
            "delta": {
                "details": det_second - det_first,
                "density": (d_second - d_first) if available else None,
                "density_text": "",
            },
        }

    def test_met_needs_a_usable_density(self) -> None:
        """密度不可用 / 跨口径 → `None`（**不是** False）——不推进档位、不伪造。"""
        self.assertIsNone(_scaffold.met_from_compare(None))
        self.assertIsNone(_scaffold.met_from_compare({}))
        self.assertIsNone(_scaffold.met_from_compare({"verdict": {"available": False}}))
        self.assertIsNone(_scaffold.met_from_compare(
            {"verdict": {"available": False}, "delta": {"density": 9.0, "details": 9}}))
        # available=True 却没有密度差（不该出现）→ 也按不可用，不拿句数凑判定
        self.assertIsNone(_scaffold.met_from_compare(
            {"verdict": {"available": True}, "delta": {"density": None, "details": 3}}))

    def test_density_up_with_details_kept_is_met(self) -> None:
        """**修订后的主路径**：细节变多、密度上升（句数可以不降）就算达标。

        这正是盘内邀请（"再多讲点细节"）指向的方向；旧判据在这里判 False。
        """
        met, reason = _scaffold.met_and_reason_from_compare(
            self._compare(d_first=5.20, d_second=6.67, det_first=26, det_second=40))
        self.assertIs(True, met)
        self.assertEqual(_scaffold.REASON_DENSITY_UP, reason)

    def test_density_up_but_details_dropped_is_not_met(self) -> None:
        """**保留的护栏**：靠"少讲内容"换来的密度上升不算达标。

        载荷取 §9.3 那条实测原值（英文两遍 5 句/26 细节 → 2 句/22 细节，
        密度 5.20 → 11.00）——修订**不改**它的结论，改的只是"句数下降"这个
        额外条件。
        """
        met, reason = _scaffold.met_and_reason_from_compare(
            self._compare(d_first=5.20, d_second=11.00, det_first=26, det_second=22))
        self.assertIs(False, met, "细节丢了就不该算达标")
        self.assertEqual(_scaffold.REASON_DETAILS_DROPPED, reason)

    def test_flat_or_falling_density_is_not_met(self) -> None:
        """密度持平/下降 → 不达标（原因串分开，便于复盘"为什么没动档"）。"""
        flat, flat_reason = _scaffold.met_and_reason_from_compare(
            self._compare(d_first=3.00, d_second=3.00, det_first=9, det_second=9))
        self.assertIs(False, flat)
        self.assertEqual(_scaffold.REASON_DENSITY_FLAT, flat_reason)
        # 细节**一个没少**、只是讲得更松（密度下降）→ 原因是 density_down，不是 details_dropped
        down, down_reason = _scaffold.met_and_reason_from_compare(
            self._compare(d_first=4.00, d_second=2.50, det_first=12, det_second=12))
        self.assertIs(False, down)
        self.assertEqual(_scaffold.REASON_DENSITY_DOWN, down_reason)

    def test_the_old_rule_would_have_failed_this(self) -> None:
        """回归钉子：**加细节但没变短**这个方向必须能达标。

        旧判据（`sentences_down and details_not_down`）在这个方向上恒为 False
        ——而盘内提示邀请的正是"add more details"，于是"连续两次达标"不可达、
        档位冻结。这条测试就是那次修订的判据本身。
        """
        compare = self._compare(
            d_first=1.50, d_second=3.67, det_first=3, det_second=11)
        compare["verdict"]["sentences_down"] = False   # 句数**上升**了
        met, _ = _scaffold.met_and_reason_from_compare(compare)
        self.assertIs(True, met, "句数上升但密度上升：修订后必须算达标")

    def test_real_compare_payload_end_to_end(self) -> None:
        """吃**真实** `compare_expression_metrics` 的输出（不是手造形状）。

        活体之外最容易漏的一环：判据读的键名（`delta.density` / `delta.details`）
        与 F14 实际产出的键名对不上 → 键写错时 `None` 会被读成"不可用"而静默冻结。
        """
        from engine.chunk_evidence import (
            compare_expression_metrics,
            count_expression_metrics,
        )
        first = count_expression_metrics(
            "I went home. I was tired.")
        second = count_expression_metrics(
            "I went home after work. I was really tired because I had a long day. "
            "My boss gave me three tasks.")
        compare = compare_expression_metrics(first, second)
        # 先钉语料形状（语料漂了就报错，别让判据在错误的输入上"通过"）
        self.assertGreater(compare["delta"]["density"] or 0, 0,
                           "语料不再构成密度上升，这条用例失去鉴别力")
        self.assertGreaterEqual(compare["delta"]["details"], 0)
        met, reason = _scaffold.met_and_reason_from_compare(compare)
        self.assertIs(True, met)
        self.assertEqual(_scaffold.REASON_DENSITY_UP, reason)

    def test_met_from_compare_shell_agrees_with_the_reasoned_version(self) -> None:
        """兼容壳只丢原因串，三态必须与带原因的版本逐例一致。"""
        for compare in (
            None, {},
            {"verdict": {"available": False}},
            self._compare(d_first=5.20, d_second=6.67, det_first=26, det_second=40),
            self._compare(d_first=5.20, d_second=11.00, det_first=26, det_second=22),
            self._compare(d_first=3.00, d_second=3.00, det_first=9, det_second=9),
        ):
            with self.subTest(compare=compare):
                self.assertIs(
                    _scaffold.met_and_reason_from_compare(compare)[0],
                    _scaffold.met_from_compare(compare))

    def test_state_round_trips_through_json_string_evidence(self) -> None:
        """`last_evidence`：sqlite 存 TEXT、pg 存 jsonb —— 两种形态都要读回同一份状态。"""
        state = _scaffold.decide_tier(_scaffold.initial_state(), True, "concise_progress")
        state = _scaffold.decide_tier(state, True, "concise_progress")       # → 第 2 档
        evidence = _scaffold.evidence_from_state(state)
        self.assertEqual(2, evidence["help_tier"])
        as_dict = _scaffold.state_from_row({"last_evidence": evidence})
        as_json = _scaffold.state_from_row({"last_evidence": json.dumps(evidence)})
        self.assertEqual(as_dict, as_json)
        self.assertEqual(2, as_dict["help_tier"])


class ExpressionScaffoldWriteBackTests(unittest.TestCase):
    """`_record_expression_scaffold`：只在练习盘推进，且**量不到时不写库**（§4.2）。"""

    class _L2:
        """最小 l2 桩：只实现档位用到的两个方法，记录每一次写。"""

        def __init__(self):
            self.row: dict = {}
            self.upserts: list = []

        async def get_learning_state(self, user_id: str, item_type: str) -> dict:
            return dict(self.row)

        async def upsert_learning_state(self, user_id, item_type, mastery, **changes):
            self.upserts.append(
                {"user_id": user_id, "item_type": item_type,
                 "mastery": mastery, "changes": changes})
            self.row = {
                "immediate_success": int(self.row.get("immediate_success") or 0)
                + int(changes.get("immediate_success") or 0),
                "failed_attempts": int(self.row.get("failed_attempts") or 0)
                + int(changes.get("failed_delta") or 0),
            }

    def _agent(self, l2):
        agent, ctx = expression_ctx(1)
        agent.l2 = l2
        return agent, ctx

    def test_met_writes_back_tier_state_and_success_counter(self) -> None:
        import asyncio

        l2 = self._L2()
        agent, ctx = self._agent(l2)
        compare = _MET_COMPARE
        asyncio.run(agent._record_expression_scaffold(ctx, compare))
        self.assertEqual(1, len(l2.upserts), "达标要写回一次")
        written = l2.upserts[0]
        self.assertEqual(_scaffold.SCAFFOLD_ITEM_TYPE, written["item_type"])
        self.assertEqual("expr-layer-user", written["user_id"])
        self.assertEqual(1, written["changes"]["immediate_success"])
        self.assertEqual(0, written["changes"]["failed_delta"])
        # 写回的 last_evidence 带档位与连胜，供下次开会话读回
        self.assertEqual(1, written["changes"]["last_evidence"]["help_tier"])
        self.assertEqual(1, written["changes"]["last_evidence"]["met_streak"])
        # 原因串必须与判定**同源**（修订后由 `expression_scaffold` 自己产出）——
        # 分开取会让台账出现"达标，但原因写着 not_shorter / concise_progress"。
        self.assertEqual(
            _scaffold.REASON_DENSITY_UP,
            written["changes"]["last_evidence"]["last_reason"])
        # ctx 上的状态就地推进（本轮后续 prompt 立刻用新档位）
        self.assertEqual(1, ctx.expression_scaffold["help_tier"])

    def test_unavailable_compare_writes_nothing(self) -> None:
        """密度不可用 → **一次写都不发**（不许伪造达标、也不许记一笔失败）。"""
        import asyncio

        l2 = self._L2()
        agent, ctx = self._agent(l2)
        asyncio.run(agent._record_expression_scaffold(
            ctx, {"verdict": {"available": False}}))
        self.assertEqual([], l2.upserts, "量不到不许写库")
        self.assertEqual(1, ctx.expression_scaffold["help_tier"])

    def test_two_met_writes_land_on_tier_two(self) -> None:
        """写侧与纯函数一致：两次达标之后存库的档位是第 2 档（§5 验收的那条）。"""
        import asyncio

        l2 = self._L2()
        agent, ctx = self._agent(l2)
        compare = _MET_COMPARE
        asyncio.run(agent._record_expression_scaffold(ctx, compare))
        asyncio.run(agent._record_expression_scaffold(ctx, compare))
        self.assertEqual(2, ctx.expression_scaffold["help_tier"])
        self.assertEqual(2, l2.upserts[-1]["changes"]["last_evidence"]["help_tier"])
        self.assertEqual(1, l2.upserts[-1]["changes"]["immediate_success"])

    def test_write_failure_never_blocks_the_turn(self) -> None:
        """写库抛错 → 静默（档位永远不能阻塞对话路径）。"""
        import asyncio

        class _Boom(self._L2):
            async def upsert_learning_state(self, *a, **kw):
                raise RuntimeError("db down")

        agent, ctx = self._agent(_Boom())
        asyncio.run(agent._record_expression_scaffold(ctx, _MET_COMPARE))
        self.assertEqual(1, ctx.expression_scaffold["help_tier"], "内存态照常推进")


class RetellLoopGateTests(unittest.TestCase):
    """首遍门槛按盘型分叉（2026-09-20 修订，plans/0920-02）。

    为什么需要它：门槛被写死在 `_chunk_turn_layer` 的 `elif` 里时**没法单测**，
    于是"练习盘的中文讲述永远开不了闭环"这类缺陷只能等真机发现。抽成
    `_opens_retell_loop` 后，两半各自可断言。
    """

    @staticmethod
    def _agent_ctx(mode: str):
        agent, ctx = control_ctx(4)
        ctx.cassette_mode = mode
        return agent, ctx

    def test_chat_tape_requires_an_explicit_feedback_request(self) -> None:
        """闲聊盘：长分享本身不触发表达分析，明确请求时才打开反馈。"""
        from engine.chunk_evidence import count_expression_metrics
        agent, ctx = self._agent_ctx("resume")
        two = count_expression_metrics("I went home. I was tired.")
        self.assertEqual(2, two["sentences"])
        self.assertFalse(agent._opens_retell_loop(ctx, two, "i went home i was tired"))
        three = count_expression_metrics("I went home. I was tired. I slept well.")
        self.assertGreaterEqual(three["sentences"], 3)
        self.assertFalse(
            agent._opens_retell_loop(ctx, three, "i went home i was tired i slept"))
        self.assertTrue(agent._opens_retell_loop(ctx, two, "Any feedback on how I said that?"))

    def test_practice_tape_accepts_a_short_passage(self) -> None:
        """练习盘：两句话（甚至只有细节、没有句末标点）就算"一段"，开闭环。

        ASR 对中文常常整段不带句末标点 → `sentences` 退化成 1，这是练习盘
        "讲一段 → 重讲"闭环在中文下**根本起不来的**直接原因。
        """
        from engine.chunk_evidence import count_expression_metrics
        agent, ctx = self._agent_ctx("expression")
        two = count_expression_metrics("I went home. I was tired.")
        self.assertTrue(agent._opens_retell_loop(ctx, two, "i went home i was tired"))
        # 无句末标点（中文转写的常见形态）：sentences=1，但 details 有内容 → 仍算一段
        no_punct = count_expression_metrics("我昨天去了咖啡馆排了二十分钟")
        self.assertEqual(1, no_punct["sentences"])
        self.assertTrue(agent._opens_retell_loop(ctx, no_punct, "我昨天去了咖啡馆排了二十分钟"))

    def test_an_explicit_request_opens_the_loop_on_both_tapes(self) -> None:
        """口语里明确要反馈 → 两盘一律放行（与盘型无关的老口径，不许被分叉弄丢）。

        语料挑一个**两盘门槛都过不去**的：`嗯。` 没有任何可量到的细节
        （`details=0`、`sentences=1`）——练习盘那边 `looks_like_a_passage` 的
        `details >= 1` 也不成立，所以能真正看出"是口语请求放行的"。
        """
        from engine.chunk_evidence import count_expression_metrics
        tiny = count_expression_metrics("嗯。")
        self.assertEqual(0, tiny["details"], "语料不再是无细节样本，这条用例失去鉴别力")
        self.assertEqual(1, tiny["sentences"])
        for mode in ("resume", "expression"):
            with self.subTest(mode=mode):
                agent, ctx = self._agent_ctx(mode)
                self.assertFalse(agent._opens_retell_loop(ctx, tiny, "嗯"))
                self.assertTrue(
                    agent._opens_retell_loop(ctx, tiny, "how did it sound?"))


class ExpressionPracticeSectionTests(unittest.TestCase):
    """§4.4 盘内合同：练习盘挂 `[Expression Practice]`，闲聊盘**逐字零增量**。"""

    def _branches(self, mode: str, tier: int = 1) -> list[tuple[str, str]]:
        """四个 prompt 分支（权威/回落 × 缓存序 1/0），与 ContractLayeringTests 同款。"""
        out: list[tuple[str, str]] = []
        original = tutor_agent._PROMPT_CACHE_ORDER
        prepared_base = stable_prepared("engage_general_topic")
        for authoritative in (True, False):
            for cache in (1, 0):
                prepared = {
                    **prepared_base,
                    "constraints": {**prepared_base["constraints"]},
                }
                if authoritative:
                    prepared["conversation_director"] = {
                        "_meta": {"source": "director_model"},
                        "response_need": "listen_and_continue",
                    }
                else:
                    prepared["conversation_director"] = {"response_need": "listen_and_continue"}
                agent, ctx = control_ctx(4)
                ctx.cassette_mode = mode
                ctx.expression_scaffold = _scaffold_state(tier)
                try:
                    tutor_agent._PROMPT_CACHE_ORDER = cache
                    out.append((
                        "mode=%s/authoritative=%s/cache=%s" % (mode, authoritative, cache),
                        agent._build_control_prompt(ctx, prepared, ""),
                    ))
                finally:
                    tutor_agent._PROMPT_CACHE_ORDER = original
        return out

    def test_present_on_expression_tape_in_all_four_branches(self) -> None:
        """§5 硬约束 1：新合同段必须四处都有——缺一处就是"某条分支静默没有这盘规矩"。"""
        branches = self._branches("expression")
        self.assertEqual(4, len(branches))
        for label, prompt in branches:
            with self.subTest(branch=label):
                self.assertIn("[Expression Practice]", prompt)
                self.assertIn("[Expression Scaffold]", prompt)
                # §4.6：显式提交口径必须写进合同（不是静音自动提交）
                self.assertIn("提交由前端显式按钮触发", prompt)
                # §4.3 闭环：他讲的时候不抢话，提交后再回应
                self.assertIn("不要抢话", prompt)
                # §4.4：分析只给一个最值得改的点（复用现役规矩，不另立一套）
                self.assertIn("最值得改的点", prompt)
                # §9：他问题再多也只提一个点（四个分支都要有，缺一处＝某条链路上模型又会并列几条）
                self.assertIn("同一轮只提一个点", prompt)
                # §4.5 纪律：档位由代码算、不许报数字、不许自行升降
                self.assertIn("不得自行升降", prompt)

    def test_absent_on_non_expression_tape_in_all_four_branches(self) -> None:
        """非练习盘四个分支都不许出现这一段（否则闲聊盘会收到练习盘规矩）。"""
        for label, prompt in self._branches("resume"):
            with self.subTest(branch=label):
                self.assertNotIn("[Expression Practice]", prompt)
                self.assertNotIn("[Expression Scaffold]", prompt)

    def test_off_tape_prompt_is_byte_identical_whatever_the_scaffold_holds(self) -> None:
        """非练习盘：档位载入与否，prompt **逐字不变**（缓存前缀口径的硬护栏）。

        若实现漏掉盘型判断，这里会多出一段 → 闲聊盘每轮 prompt 都变，公共前缀报废。
        """
        agent, ctx = control_ctx(4)
        prepared = stable_prepared("engage_general_topic")
        ctx.expression_scaffold = {}
        before = agent._build_control_prompt(ctx, prepared, "")
        ctx.expression_scaffold = _scaffold_state(3, met_streak=5, miss_streak=0)
        after = agent._build_control_prompt(ctx, prepared, "")
        self.assertEqual(before, after)

    def test_section_helper_is_empty_off_tape(self) -> None:
        """`_expression_practice_section` 在非练习盘返回**空串**（被空段过滤吃掉）。"""
        agent, ctx = control_ctx(4)
        self.assertEqual("", agent._expression_practice_section(ctx))
        self.assertEqual("", agent._expression_scaffold_hint(ctx))
        ctx.cassette_mode = "expression"
        self.assertNotEqual("", agent._expression_practice_section(ctx))
        self.assertNotEqual("", agent._expression_scaffold_hint(ctx))

    def test_section_sits_after_turn_contract_not_in_stable_prefix(self) -> None:
        """逐轮变的规矩挂在 `turn_contract` 之后，不得进稳定缓存前缀。"""
        for label, prompt in self._branches("expression"):
            with self.subTest(branch=label):
                self.assertLess(
                    prompt.index("[Latest Turn Contract"),
                    prompt.index("[Expression Practice]"),
                )
                self.assertLess(
                    prompt.index("[Role] 只输出说给用户听的话"),
                    prompt.index("[Expression Practice]"),
                )

    def test_scaffold_text_follows_the_help_tier(self) -> None:
        """档位变了，"这一档只给什么帮助"跟着变；四档话术各就各位、互不相同。"""
        seen: dict = {}
        for tier in (1, 2, 3, 4):
            agent, ctx = expression_ctx(tier)
            prompt = agent._build_control_prompt(
                ctx, stable_prepared("engage_general_topic"), "")
            seen[tier] = prompt
            self.assertIn("第 " + str(tier) + " 档", prompt)
            # 该档的"帮助形态"摘要必须原样落地（四档四句，不能渲染成同一句）
            self.assertIn(_scaffold.tier_row(tier)["summary"], prompt)
        self.assertEqual(4, len(set(seen.values())))
        self.assertNotIn(_scaffold.tier_row(1)["summary"], seen[2])
        self.assertNotIn(_scaffold.tier_row(4)["summary"], seen[3])

    def test_task_hint_carries_the_current_tier(self) -> None:
        """给任务时附带的档位注（§4.5）：只说"只给一种帮助"，并禁止模型自报数字。"""
        agent, ctx = expression_ctx(2)
        hint = agent._expression_scaffold_hint(ctx)
        self.assertIn("只给一种帮助", hint)
        self.assertIn(_scaffold.tier_row(2)["summary"], hint)
        note = agent._expression_scaffold_hint(ctx, advance_note=True)
        self.assertIn("你不得自行升降", note)

    def test_pick_one_point_prioritises_the_tier_driving_dims(self) -> None:
        """§9 定案：他问题再多也只提一个点，且**优先挑进判据的那两维**。

        跨文件对账——两维的名字**从判据模块取**（`TIER_DRIVING_DIM_LABELS`），
        不在测试里手写：判据哪天换了维度，这里必须当场红，而不是话术静默停在旧维度上。
        为什么要连"不许并列几条"一起钉：只写"只提一个点"的话，模型会把优先级读成
        "可以全都提、只是排序不同"——那就退回了原来的问题。
        """
        agent, ctx = control_ctx(4)
        ctx.cassette_mode = "expression"
        ctx.expression_scaffold = _scaffold_state(1)
        section = agent._expression_practice_section(ctx)
        for key, label in _scaffold.TIER_DRIVING_DIM_LABELS.items():
            with self.subTest(dim=key):
                self.assertIn(label, section)
        self.assertIn("同一轮只提一个点", section)
        self.assertIn("只挑优先级最高的那一个", section)
        self.assertIn("不要并列几条", section)
        # 另一半也要在：只上屏的那几维"可以提，但不动档位"——少了这半句，
        # 模型会把"优先"读成"只许"，第二梯队就再也提不了了。
        self.assertIn("代码不判、档位也不因它们动", section)
        # 键必须是判据**真正读**的 delta 字段：按源码对账，不在测试里再抄一份
        # "应该读什么"（判据换了字段而话术没换，这里就要红）。
        import inspect
        judge_src = inspect.getsource(_scaffold.met_and_reason_from_compare)
        for key in _scaffold.TIER_DRIVING_DIM_LABELS:
            with self.subTest(delta_field=key):
                self.assertIn('delta.get("%s")' % key, judge_src)


class _PolicyStub:
    """`TutorPolicy` 的最小替身：本层只读/写它的 `correction_budget`。"""

    def __init__(self, budget: int = 3) -> None:
        self.correction_budget = budget


class ExpressionArgumentFocusRoutingTests(unittest.TestCase):
    """§4.2.2／§4.2.3／§4.2.5 的**接点**：`prepare_turn` 交给合同的那份 `constraints`。

    纯层的取舍规则（伪证据丢弃、下限优先、画像只在同档平手时生效）已由
    `test_runner/test_expression_argument.py` 逐条钉死。这里只管**接线**三件事，
    因为这三件错了纯层再对也不生效：

      1. `ctx.conversation_plan` 可能来自缓存／记忆恢复 ⇒ 焦点必须在**本轮原话**上
         复验一次，不能只信上游；
      2. 焦点要真的落进 `constraints["argument_focus"]`（逐轮段才有东西可渲染）；
      3. 练习盘把语言层让位：非主动问语言点时预算归零 + 英文优先；⚠ 且**不许就地改
         `ctx.conversation_plan`**（它被缓存复用，改了下一轮会带着上一轮的覆盖）。
    """

    ZH = "我觉得在家办公更好，因为我每天省下两个小时的通勤时间。"
    MIXED = "I think 在家办公 is better because 我每天省两个小时的通勤时间。"
    TRANSCRIPT = (
        "我觉得在家办公更好，因为我每天省下两个小时的通勤时间。"
        "而且通勤很累，所以我更喜欢在家办公。"
    )
    # 首讲夹具：主张弱、证据 ok。焦点应为 claim。
    PROBE = {"weak": ["claim"], "ok": ["evidence"], "na": ["warrant", "qualifier", "rebuttal"],
             "focus_key": "claim", "evidence": "我觉得在家办公更好"}

    def _ctx(self, probe: dict | None = None, profile_weak=None):
        agent, ctx = expression_ctx(1)
        ctx.conversation_plan = {"argument_probe": dict(probe)} if probe else {}
        ctx.expression_profile_weak = list(profile_weak or [])
        return agent, ctx

    @staticmethod
    def _constraints() -> dict:
        return {
            "adaptive_language_support": {
                "presentation_goal": "中文优先", "roleplay_presentation": "mixed",
            },
            "conversation_director": {"language_presentation": "mixed"},
        }

    def _apply(self, ctx, constraints, policy, *, transcript, explicit=False):
        agent = TutorAgent.__new__(TutorAgent)
        agent._apply_expression_argument_focus(
            ctx, constraints, policy,
            user_transcript=transcript, explicit_language_request=explicit,
        )

    # ── §5 第 7 条：中文／中英夹杂照旧进同一逻辑反馈路径，且语言层让位 ──────────

    def test_zh_first_turn_zeroes_the_correction_budget(self) -> None:
        """用户没主动问语言点 ⇒ 这一轮不是语言课：两层预算都归零。"""
        ctx_agent, ctx = self._ctx(self.PROBE)
        constraints, policy = self._constraints(), _PolicyStub(3)
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        self.assertEqual(0, constraints["correction_budget"])
        self.assertEqual(0, policy.correction_budget, "policy 不归零 ⇒ 两层 prompt 自相矛盾")
        # 中文原话也能选出一个逻辑焦点 —— 不因为文字是中文就中断内容反馈
        self.assertEqual("claim", constraints["argument_focus"]["focus_key"])
        self.assertTrue(constraints["argument_focus"]["from_probe"])

    def test_mixed_script_goes_down_the_same_path(self) -> None:
        """中英夹杂同样是**内容**反馈，不是要求先翻译。"""
        mixed_probe = {
            "weak": ["evidence"], "ok": ["claim"],
            "focus_key": "evidence", "evidence": "我每天省两个小时的通勤时间",
        }
        _agent, ctx = self._ctx(mixed_probe)
        constraints, policy = self._constraints(), _PolicyStub(3)
        self._apply(ctx, constraints, policy, transcript=self.MIXED)
        self.assertEqual(0, constraints["correction_budget"])
        self.assertEqual("evidence", constraints["argument_focus"]["focus_key"])

    def test_explicit_language_question_keeps_the_budget(self) -> None:
        """他主动问词／句 ⇒ 只这一轮放行语言问答，预算不动。"""
        _agent, ctx = self._ctx(self.PROBE)
        constraints, policy = self._constraints(), _PolicyStub(3)
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT, explicit=True)
        self.assertNotIn("correction_budget", constraints)
        self.assertEqual(3, policy.correction_budget)
        # 但英文优先的呈现目标仍要在（答完回英文）
        self.assertIn("英文", constraints["adaptive_language_support"]["presentation_goal"])

    def test_adaptive_presentation_is_english_first_and_target_only(self) -> None:
        _agent, ctx = self._ctx(self.PROBE)
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        adaptive = constraints["adaptive_language_support"]
        self.assertIn("英文", adaptive["presentation_goal"])
        self.assertEqual("target_only", adaptive["roleplay_presentation"])
        self.assertEqual("target_only",
                         constraints["conversation_director"]["language_presentation"])

    def test_cached_plan_is_never_mutated_in_place(self) -> None:
        """§4.2.5：`conversation_plan` 会被缓存复用 ⇒ 只能改 `constraints` 的副本。"""
        plan = {"argument_probe": dict(self.PROBE), "language_presentation": "mixed"}
        _agent, ctx = self._ctx()
        ctx.conversation_plan = plan
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        self.assertIs(plan, ctx.conversation_plan)
        self.assertEqual("mixed", plan["language_presentation"],
                         "就地改了缓存 plan ⇒ 下一轮从缓存恢复会带着上一轮的覆盖")
        self.assertEqual("target_only",
                         constraints["conversation_director"]["language_presentation"])

    # ── §5 第 5／6 条：本轮复验 + 无缺口时不许虚构诊断 ─────────────────────────

    def test_fabricated_evidence_drops_the_focus_entirely(self) -> None:
        """导演给了本轮原话里找不到的引文 ⇒ 整条 probe 丢弃、不进 `constraints`。"""
        bad = dict(self.PROBE, evidence="I never said this sentence at all")
        _agent, ctx = self._ctx(bad)
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        focus = constraints["argument_focus"]
        self.assertFalse(focus["from_probe"])
        self.assertEqual("", focus["focus_key"])
        self.assertEqual({}, focus["dims"])

    def test_no_weak_dim_leaves_an_empty_focus(self) -> None:
        """全 ok ⇒ 本轮没有可行动缺口：焦点空串，靠段里的"先问一个澄清"兜住。"""
        ok_probe = {"ok": ["claim", "evidence", "warrant", "qualifier", "rebuttal"]}
        _agent, ctx = self._ctx(ok_probe)
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        focus = constraints["argument_focus"]
        self.assertTrue(focus["from_probe"])
        self.assertEqual("", focus["focus_key"])

    # ── §5 第 9 条：画像弱项只在**同档平手**时生效 ─────────────────────────────

    def test_profile_breaks_the_tie_only_inside_this_turns_pool(self) -> None:
        """§5 第 9 条：下限两维都缺、画像近期是 evidence ⇒ 挑 evidence（不是固定序的 claim）。

        ⚠ 这条夹具**故意不带**导演建议：`focus_key` 是同档平手时**更高**的偏好
        （纯层 `test_director_hint_only_wins_a_tie` 已钉），带了就测不到画像序了。
        """
        both = {"weak": ["claim", "evidence"], "evidence": "我觉得在家办公更好"}
        _agent, ctx = self._ctx(both, profile_weak=["evidence", "warrant"])
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        focus = constraints["argument_focus"]
        self.assertEqual("evidence", focus["focus_key"])
        self.assertEqual(["evidence", "warrant"], focus["profile_weak"])

    def test_current_turn_beats_the_stale_profile(self) -> None:
        """本轮主张不清、证据已可信 ⇒ 转去补主张；旧证据弱项不许压过本轮。"""
        _agent, ctx = self._ctx(self.PROBE, profile_weak=["evidence"])
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        self.assertEqual("claim", constraints["argument_focus"]["focus_key"])

    def test_profile_weak_is_capped_at_five(self) -> None:
        """逐轮段有界（§4.2.4）：画像最多带 5 个维度状态 + 1 条近期证据。"""
        dims = ["claim", "evidence", "warrant", "qualifier", "rebuttal", "claim"]
        _agent, ctx = self._ctx(self.PROBE, profile_weak=dims)
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        self.assertLessEqual(len(constraints["argument_focus"]["profile_weak"]), 5)
        self.assertLessEqual(len(constraints["argument_focus"]["evidence"]), 140)
        self.assertEqual(set(constraints["argument_focus"]["dims"]),
                         {"claim", "evidence", "warrant", "qualifier", "rebuttal"})

    # ── 逐轮段真的挂上去了，且**不列五维清单**（§5 第 5 条）────────────────────

    def test_focus_section_renders_exactly_one_dimension(self) -> None:
        from engine import expression_argument as ea

        agent, ctx = self._ctx(self.PROBE)
        constraints, policy = self._constraints(), _PolicyStub()
        self._apply(ctx, constraints, policy, transcript=self.TRANSCRIPT)
        prepared = stable_prepared("engage_general_topic")
        prepared["constraints"] = {**prepared["constraints"], **constraints}
        prompt = agent._build_control_prompt(ctx, prepared, self.TRANSCRIPT)
        self.assertIn("[Expression Focus]", prompt)
        self.assertIn("本轮**只提这一个**焦点", prompt)
        hits = [label for label in ea.DIMENSION_LABELS.values() if label in prompt]
        self.assertEqual([ea.DIMENSION_LABELS["claim"]], hits,
                         "逐轮段把五维列成清单了 ⇒ 回到「报总分总」的老路")

    def test_off_tape_prompt_is_byte_identical_whatever_the_profile_weak_holds(self) -> None:
        """非练习盘：画像弱项在不在，prompt **逐字不变**（§5 第 1 条护栏）。"""
        agent, ctx = control_ctx(4)
        prepared = stable_prepared("engage_general_topic")
        ctx.expression_profile_weak = []
        before = agent._build_control_prompt(ctx, prepared, "")
        ctx.expression_profile_weak = ["claim", "evidence"]
        after = agent._build_control_prompt(ctx, prepared, "")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
