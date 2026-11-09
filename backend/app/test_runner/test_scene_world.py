# -*- coding: utf-8 -*-
"""场景世界锚（plans/0916-03）的确定性回归测试。

背景（11 组 39 轮实测，`test_runner/diag_scene_depth.py` 见日志
`logs/scene_depth.txt`）：

- 进场景后 AI 从不贡献世界事实——咖啡店 7 组"具体名词/轮"全部 0.0，
  所有具体信息都来自用户自己的话，AI 只是镜子；
- 没有坐标：同一个动作（点单时报姓名）在英语国家是对的、在国内是错的，
  而 `[Anti-Parroting]` 曾把"咖啡店问名字"写成普适正确示范，
  于是国内用户被索要姓名——用户原话"这完完全全就是架空事件"；
- 没有进度与能力边界：问已答过的问题、付款前就送客、
  服务员被问配方直接编、"我去问一下"却没有下文。

本文件只做**纯逻辑断言**：不连库、不调模型、不碰真实用户。
`scene_world` 是纯静态数据 + 纯函数，接入点用 `TutorAgent.__new__` 绕过构造，
所以整份测试是确定性的、可离线跑。

跑法：cd chanpinzhujian/api && python -m unittest test_runner.test_scene_world -v
"""

from __future__ import annotations

import unittest

from agent import scene_world
from agent.tutor_agent import SessionContext, TutorAgent

DOMESTIC = scene_world.LOCALE_DOMESTIC
ENGLISH = scene_world.LOCALE_ENGLISH

CAFE_BRIEF = "我们在一家咖啡店，你是店员，我是顾客。"
CAFE_BRIEF_CN = "我们模拟在国内的咖啡店点单，你是店员，我是顾客。"
RESTAURANT_BRIEF = "我们在一家餐厅，你是服务员，我是顾客。"


def roleplay_ctx(brief: str, messages=(), turn: int = 0) -> SessionContext:
    """构造一个"已进入场景"的会话上下文。"""
    ctx = SessionContext(session_id="scene-session", user_id="scene-user", target_lang="en")
    ctx.active_scene = "open_roleplay"
    ctx.scenario_brief = brief
    ctx.scenario_role = "店员"
    ctx.learner_role = "顾客"
    ctx.scenario_step = "active_exchange"
    ctx.roleplay_turn = turn
    ctx.scene_context_set = True
    ctx.messages = [{"role": r, "content": c} for r, c in messages]
    return ctx


def new_agent() -> TutorAgent:
    """不跑 __init__ 的 agent：世界锚只用到 ctx 与静态方法。"""
    return TutorAgent.__new__(TutorAgent)


# ── 1. 卡片完整性 ────────────────────────────────────────────────
class CardIntegrityTests(unittest.TestCase):
    def test_every_card_has_five_anchors(self):
        self.assertTrue(scene_world.CARDS, "至少要有咖啡店/餐厅两张卡")
        for key, card in scene_world.CARDS.items():
            with self.subTest(card=key):
                self.assertTrue(card["name"])
                self.assertTrue(card["place"], "场所面目不能空")
                self.assertGreaterEqual(len(card["phases"]), 4, "现实环节要成链")
                self.assertGreaterEqual(len(card["events"]), 3, "世界要能自己制造处境")
                self.assertGreaterEqual(len(card["ladder"]), 3, "服务链至少三层")
                self.assertTrue(card["relay"], "转达规则不能空")
                self.assertTrue(card["locales"])

    def test_ladder_entries_declare_capability_boundary(self):
        """服务链每层必须写清'能答什么 / 答不了什么'，否则模型会自己编。"""
        for key, card in scene_world.CARDS.items():
            for item in card["ladder"]:
                with self.subTest(card=key, role=item.get("role")):
                    for field in ("role", "holds", "can", "cannot"):
                        self.assertTrue(str(item.get(field) or "").strip(), f"缺 {field}")

    def test_top_of_ladder_can_refund(self):
        """退款/免单必须在链上有人能接（否则闭环到顶也是空话）。"""
        for key, card in scene_world.CARDS.items():
            joined = " ".join(
                f"{item['can']}" for item in card["ladder"]
            )
            with self.subTest(card=key):
                self.assertIn("退款", joined)

    def test_both_locales_are_specified(self):
        for key, card in scene_world.CARDS.items():
            for locale in (DOMESTIC, ENGLISH):
                with self.subTest(card=key, locale=locale):
                    loc = card["locales"].get(locale)
                    self.assertIsNotNone(loc)
                    for field in ("order", "pay", "pickup", "name_policy", "tip", "extra"):
                        self.assertTrue(str(loc.get(field) or "").strip(), f"缺 {field}")


# ── 2. 坐标判定 ──────────────────────────────────────────────────
class LocaleTests(unittest.TestCase):
    def test_domestic_hints(self):
        for text in ["我们模拟在国内的咖啡店点单", "就用瑞幸那种模式", "扫码点单，微信支付"]:
            with self.subTest(text=text):
                self.assertEqual(scene_world.detect_locale(text), DOMESTIC)

    def test_english_hints(self):
        for text in ["我们模拟在美国的一家咖啡店点单", "我在英国留学", "We are in the US."]:
            with self.subTest(text=text):
                self.assertEqual(scene_world.detect_locale(text), ENGLISH)

    def test_domestic_wins_when_both_present(self):
        """国内线索优先：用户身处国内，缺的正是模型不会的那套流程。"""
        self.assertEqual(scene_world.detect_locale("我在美国体验过，但还是说国内吧"), DOMESTIC)

    def test_default_and_empty(self):
        self.assertEqual(scene_world.detect_locale(CAFE_BRIEF), DOMESTIC, "无线索默认国内")
        self.assertEqual(scene_world.detect_locale("", default=ENGLISH), ENGLISH)
        self.assertEqual(scene_world.detect_locale(CAFE_BRIEF, default=""), "", "空 default 便于只测线索")


# ── 3. 卡片解析（只负责"进去之后是哪张卡"） ───────────────────────
class ResolveTests(unittest.TestCase):
    def test_hits_expected_card(self):
        self.assertEqual(scene_world.resolve(CAFE_BRIEF), "cafe")
        self.assertEqual(scene_world.resolve(RESTAURANT_BRIEF), "restaurant")
        self.assertEqual(scene_world.resolve("我们练习在餐厅点菜。"), "restaurant")

    def test_more_hits_wins(self):
        self.assertEqual(scene_world.resolve("在餐厅点菜，服务员拿菜单"), "restaurant")

    def test_no_card(self):
        for text in ["", "我们聊聊你最近怎么样", "随便聊聊天"]:
            with self.subTest(text=text):
                self.assertIsNone(scene_world.resolve(text))

    def test_mention_without_scene_is_not_the_gate(self):
        """resolve 是纯函数：它只说"像哪张卡"，不做"要不要进场景"的判断。

        门（IntentRouter）把"我喜欢喝咖啡"挡在场景之外，所以这里命中 cafe
        不会造成误伤——调用点 `_refresh_scene_world` 已用 scene_context_set 收口。
        """
        self.assertEqual(scene_world.resolve("我喜欢喝咖啡"), "cafe")
        ctx = roleplay_ctx(CAFE_BRIEF)
        ctx.scene_context_set = False
        ctx.scenario_brief = "我喜欢喝咖啡"
        new_agent()._refresh_scene_world(ctx, "我喜欢喝咖啡")
        self.assertEqual(ctx.scene_world_key, "", "没进场景就不该写世界锚")


# ── 4. 姓名硬规则（用户举报的"让我报名字"根因） ────────────────────
class NamePolicyTests(unittest.TestCase):
    def test_domestic_never_asks_for_name(self):
        """国内坐标下任何卡片都不得引导模型索取姓名。

        咖啡店写的是"不索取姓名"（国内以取餐码/订单号识别）；
        餐厅写的是"一般不登记姓名"（等位可能留手机号尾号）。
        措辞可以不同，但必须是"不"——绝不能出现英语国家那句问杯名的示范。
        """
        for key in scene_world.CARDS:
            locale_key = scene_world.CARDS[key]["locales"][DOMESTIC]
            text = scene_world.render(key, DOMESTIC)
            with self.subTest(card=key):
                self.assertNotIn("What name", text)
                self.assertIn("不", locale_key["name_policy"], "国内姓名规则必须是否定式")
                self.assertIn("姓名", locale_key["name_policy"])

    def test_english_may_ask_for_cup_name(self):
        text = scene_world.render("cafe", ENGLISH)
        self.assertIn("What name", text)

    def test_render_name_rule(self):
        self.assertIn("不得索取姓名", scene_world.render_name_rule(DOMESTIC))
        self.assertIn("What name", scene_world.render_name_rule(ENGLISH))


# ── 5. 已确认信息：确定性抽取（治"问已答过的问题"、提前送客） ──────
class ConfirmedDetailsTests(unittest.TestCase):
    def test_captures_order_specs(self):
        found = scene_world.extract_confirmed_details(["Medium, oat milk, no sugar."])
        joined = " | ".join(found).lower()
        self.assertIn("杯型=medium", joined)
        self.assertIn("奶类=oat milk", joined)
        self.assertIn("甜度/冰量=no sugar", joined)

    def test_captures_pickup_number_and_count(self):
        found = scene_world.extract_confirmed_details(["My pickup number is fifty-six."])
        self.assertTrue(any("取餐号" in item for item in found), found)
        found = scene_world.extract_confirmed_details(["Two lattes, please."])
        self.assertTrue(any("数量" in item for item in found), found)

    def test_skips_pure_chinese(self):
        """纯中文句子不抓：用户用中文求助时，句子里的词不是英文订单事实。"""
        self.assertEqual(scene_world.extract_confirmed_details(["我要大杯的热的"]), [])

    def test_captures_mixed_language(self):
        found = scene_world.extract_confirmed_details(["我要 medium 的，for here"])
        joined = " | ".join(found)
        self.assertIn("杯型=medium", joined)
        self.assertIn("堂食/外带=for here", joined)

    def test_dedupes_and_caps(self):
        found = scene_world.extract_confirmed_details(
            ["Medium, oat milk.", "Medium again, oat milk."]
        )
        self.assertEqual(len([x for x in found if x.startswith("杯型=")]), 1)
        many = scene_world.extract_confirmed_details([f"Medium {i}" for i in range(20)])
        self.assertLessEqual(len(many), 10)

    def test_ignores_non_order_chatter(self):
        self.assertEqual(scene_world.extract_confirmed_details(["I'll pay by card."]), [])


# ── 6. 渲染：五件套都要落地，且不超预算 ───────────────────────────
class RenderTests(unittest.TestCase):
    def test_unknown_card_renders_empty(self):
        self.assertEqual(scene_world.render("no_such_card", DOMESTIC), "")

    def test_contains_all_five_anchors(self):
        text = scene_world.render("cafe", DOMESTIC)
        self.assertIn("[Scene World", text)
        self.assertIn("这是哪儿：", text)
        self.assertIn("坐标：", text)
        self.assertIn("现实环节", text)
        self.assertIn("此刻可发生的事", text)
        self.assertIn("对面角色与能力边界", text)
        self.assertIn("转达规则", text)
        self.assertIn("服务链闭环", text)

    def test_ladder_roles_are_rendered(self):
        card = scene_world.CARDS["restaurant"]
        text = scene_world.render("restaurant", DOMESTIC)
        for item in card["ladder"]:
            self.assertIn(item["role"], text)

    def test_closure_is_explicit(self):
        """说"我去问一下"必须在同一轮带回答案——否则这句话本身就是幻觉来源。"""
        text = scene_world.render("restaurant", DOMESTIC)
        self.assertIn("必须在同一轮里给出结果", text)
        self.assertIn("绝不允许只说去问", text)

    def test_confirmed_is_injected_and_forbids_reasking(self):
        text = scene_world.render("cafe", DOMESTIC, confirmed=["杯型=medium", "奶类=oat milk"])
        self.assertIn("禁止再问一遍", text)
        self.assertIn("杯型=medium", text)

    def test_opening_hint_only_on_first_turn(self):
        self.assertIn("【开场】", scene_world.render("cafe", DOMESTIC, turn=0))
        self.assertNotIn("【开场】", scene_world.render("cafe", DOMESTIC, turn=3))

    def test_length_is_bounded(self):
        """纯静态注入，但也不能长到挤占既有 prompt 预算。"""
        text = scene_world.render("restaurant", ENGLISH, confirmed=["杯型=medium"])
        self.assertLess(len(text), 2000, f"实际 {len(text)} 字符")

    def test_on_scene_terms(self):
        self.assertIn("food", scene_world.on_scene_terms("restaurant"))
        self.assertIn("食物", scene_world.on_scene_terms("cafe"))
        self.assertEqual(scene_world.on_scene_terms(""), "")
        self.assertEqual(scene_world.on_scene_terms("no_such_card"), "")


# ── 7. 接入 TutorAgent：写入、幂等、不改坐标就不重置 ────────────────
class RefreshSceneWorldTests(unittest.TestCase):
    def setUp(self):
        self.agent = new_agent()

    def test_writes_key_and_locale(self):
        ctx = roleplay_ctx(CAFE_BRIEF)
        self.agent._refresh_scene_world(ctx, "I'd like a latte, please.")
        self.assertEqual(ctx.scene_world_key, "cafe")
        self.assertEqual(ctx.scene_world_locale, DOMESTIC)

    def test_noop_without_scene(self):
        ctx = roleplay_ctx(CAFE_BRIEF)
        ctx.scene_context_set = False
        self.agent._refresh_scene_world(ctx, "I'd like a latte.")
        self.assertEqual(ctx.scene_world_key, "")
        self.assertEqual(ctx.scene_world_locale, "")

    def test_collects_confirmed_details_from_recent_user_turns(self):
        ctx = roleplay_ctx(CAFE_BRIEF, messages=[
            ("user", "I'd like a latte, please."),
            ("assistant", "Sure. What size?"),
            ("user", "Medium, oat milk, no sugar."),
        ])
        self.agent._refresh_scene_world(ctx, "Medium, oat milk, no sugar.")
        joined = " | ".join(ctx.scene_world_details).lower()
        self.assertIn("杯型=medium", joined)

    def test_details_accumulate_across_turns(self):
        ctx = roleplay_ctx(CAFE_BRIEF, messages=[("user", "Medium, oat milk.")])
        self.agent._refresh_scene_world(ctx, "Medium, oat milk.")
        first = list(ctx.scene_world_details)
        ctx.messages.append({"role": "user", "content": "That's all, to go."})
        self.agent._refresh_scene_world(ctx, "That's all, to go.")
        self.assertTrue(set(first).issubset(set(ctx.scene_world_details)), "旧确认不能被冲掉")
        self.assertTrue(any("堂食/外带" in item for item in ctx.scene_world_details))

    def test_key_is_sticky_within_scene(self):
        """同一盘里不因为某轮措辞变化就把世界锚换掉。"""
        ctx = roleplay_ctx(CAFE_BRIEF)
        self.agent._refresh_scene_world(ctx, "I'd like a latte.")
        ctx.scene_world_details = ["杯型=medium"]
        self.agent._refresh_scene_world(ctx, "菜单上还有别的吗")
        self.assertEqual(ctx.scene_world_key, "cafe")
        self.assertEqual(ctx.scene_world_details, ["杯型=medium"], "换卡才清空已确认")

    def test_locale_can_switch_mid_conversation(self):
        ctx = roleplay_ctx(CAFE_BRIEF)
        self.agent._refresh_scene_world(ctx, "I'd like a latte.")
        self.assertEqual(ctx.scene_world_locale, DOMESTIC)
        self.agent._refresh_scene_world(ctx, "其实我是在美国的一家咖啡店")
        self.assertEqual(ctx.scene_world_locale, ENGLISH)
        self.assertEqual(ctx.scene_world_key, "cafe", "换坐标不该换卡")

    def test_render_needs_scene(self):
        ctx = roleplay_ctx(CAFE_BRIEF)
        self.assertEqual(TutorAgent._scene_world_text(ctx), "", "还没解析出卡片时为空")
        self.agent._refresh_scene_world(ctx, "I'd like a latte.")
        self.assertIn("[Scene World", TutorAgent._scene_world_text(ctx))

    def test_rendered_text_follows_locale_for_name_rule(self):
        ctx = roleplay_ctx(CAFE_BRIEF_CN)
        self.agent._refresh_scene_world(ctx, "Hi, I'd like a latte.")
        self.assertEqual(ctx.scene_world_locale, DOMESTIC)
        self.assertNotIn("What name", TutorAgent._scene_world_text(ctx))


# ── 8. 注入生成侧 prompt：两条分支都必须带上（实测曾只挂长分支） ─────
def roleplay_prepared(brief: str, world_key: str = "", locale: str = "",
                      director_authoritative: bool = False) -> dict:
    constraints = {
        "scene_roleplay": {"brief": brief, "world_key": world_key, "locale": locale},
    }
    if world_key:
        constraints["scene_world"] = scene_world.render(world_key, locale or DOMESTIC)
    prepared = {
        "policy": {"mode": "flow", "teaching_act": "listen_and_deepen"},
        "learning_move": {"stage": "flow", "action": "listen_and_deepen"},
        "learner_state": {"language_mode": "target_language"},
        "constraints": constraints,
    }
    if director_authoritative:
        # 实测：director_model 成功时 `_build_control_prompt` 走紧凑分支，
        # 若 [Scene World] 只挂在长分支的 [Constraints] 里，正常语音链路根本收不到。
        prepared["conversation_director"] = {
            "_meta": {"source": "director_model"},
            "response_sequence": ["answer_user"],
        }
    return prepared


class ControlPromptTests(unittest.TestCase):
    def setUp(self):
        self.agent = new_agent()
        self.ctx = roleplay_ctx(CAFE_BRIEF_CN, turn=1)
        self.agent._refresh_scene_world(self.ctx, "Hi, I'd like a latte, please.")

    def _prompt(self, director_authoritative: bool) -> str:
        prepared = roleplay_prepared(
            self.ctx.scenario_brief,
            self.ctx.scene_world_key,
            self.ctx.scene_world_locale,
            director_authoritative,
        )
        return self.agent._build_control_prompt(self.ctx, prepared)

    def test_both_branches_carry_scene_world(self):
        # 注意：稳定契约里本来就引用过 "[Scene World]" 这个词（[Reality] 与
        # [Anti-Parroting] 的坐标规则），所以这里要用卡片渲染头的唯一串判断，
        # 否则测的是"常规规则在不在"，而不是"世界锚注入了没有"。
        for authoritative in (False, True):
            prompt = self._prompt(authoritative)
            with self.subTest(director_authoritative=authoritative):
                self.assertIn("[Scene World -", prompt)
                self.assertIn("这是哪儿：", prompt)

    def test_prompt_carries_world_facts_and_service_ladder(self):
        prompt = self._prompt(True)
        self.assertIn("现实环节", prompt)
        self.assertIn("店长", prompt, "服务链要出现在真正发给模型的 prompt 里")
        self.assertIn("服务链闭环", prompt)

    def test_domestic_prompt_forbids_asking_name(self):
        prompt = self._prompt(True)
        self.assertIn("不索取姓名", prompt)
        self.assertNotIn("What name", prompt)

    def test_english_prompt_allows_cup_name(self):
        ctx = roleplay_ctx("我们模拟在美国的一家咖啡店点单，你是店员，我是顾客。", turn=1)
        self.agent._refresh_scene_world(ctx, "Hi, I'd like a latte.")
        prepared = roleplay_prepared(
            ctx.scenario_brief, ctx.scene_world_key, ctx.scene_world_locale, True
        )
        prompt = self.agent._build_control_prompt(ctx, prepared)
        self.assertIn("What name", prompt)

    def test_non_scene_turn_has_no_scene_world(self):
        ctx = SessionContext(session_id="plain", user_id="plain", target_lang="en")
        prompt = self.agent._build_control_prompt(ctx, roleplay_prepared("闲聊"))
        self.assertNotIn("[Scene World -", prompt)
        self.assertNotIn("这是哪儿：", prompt)

    def test_coordinate_rules_only_appear_inside_a_scene(self):
        """坐标规则是场景专属的，绝不进非场景轮。

        这不是洁癖：稳定契约的长度直接进每轮 prompt 预算，
        `test_control_context_budget` 盯的就是那个数字。把进场景才用得上的
        规则无条件写进契约，会让每次闲聊都白背一段场景规则。
        """
        plain = SessionContext(session_id="plain", user_id="plain", target_lang="en")
        plain_prompt = self.agent._build_control_prompt(plain, roleplay_prepared("闲聊"))
        self.assertNotIn("是否索取姓名由", plain_prompt)
        self.assertIn("或问名字", plain_prompt, "非场景轮保留原来的中性示例")

        scene_prompt = self._prompt(True)
        self.assertIn("是否索取姓名由", scene_prompt)
        self.assertNotIn("或问名字", scene_prompt, "进场景后必须换成坐标规则")


# ── 9. 话题漂移护栏不能误伤场景内的正当内容 ────────────────────────
class TopicDriftGuardTests(unittest.TestCase):
    ON_SCENE_REPLY = "Excellent choice. Our food is cooked to order, so it takes a few minutes."
    OFF_SCENE_REPLY = "By the way, have you seen that new movie about cats?"

    def _violations(self, reply: str, world_key: str) -> list:
        prepared = roleplay_prepared(RESTAURANT_BRIEF, world_key=world_key)
        return TutorAgent._controlled_reply_violations(reply, prepared, "What do you recommend?")

    def test_on_scene_food_is_not_flagged(self):
        """世界锚定义了"什么是这个场景里的话题"：餐厅谈 food 不是跑题。"""
        found = self._violations(self.ON_SCENE_REPLY, "restaurant")
        self.assertFalse(any("旧话题" in item for item in found), found)

    def test_without_world_key_it_would_have_been_flagged(self):
        """回归护栏：说明 `on_scene_terms` 是真的在起作用，不是空操作。"""
        found = self._violations(self.ON_SCENE_REPLY, "")
        self.assertTrue(any("旧话题" in item for item in found), found)

    def test_truly_off_scene_topic_is_still_flagged(self):
        """收紧必须只针对场景内词汇；真跑题（电影/猫）仍要被拦。"""
        found = self._violations(self.OFF_SCENE_REPLY, "restaurant")
        self.assertTrue(any("旧话题" in item for item in found), found)


if __name__ == "__main__":
    unittest.main(verbosity=2)
