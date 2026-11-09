# -*- coding: utf-8 -*-
"""练表达盘的跨文件接线静态契约（plans/0919-02 §4.3/§4.4/§4.6）。

这一批把"练表达"从概念落成一个**盘型**（第三种盘）。链条横跨四处，任何一处漏掉
都**不会报错**，只会静默失效——正是本项目反复踩的那类坑。同类先例：
`test_runner/test_continue_tape_wiring.py`（拆续盘的接线）。

  1. **盘型**：`main.py` 的 `POST /expression/start` 立一盘零消息的带，
     `conversation_meta.scene='expression'`（沿用 `save_conversation` 的第 6 个
     位置参数，**免迁移、不加数据库列**，D006）；刻意**不复用 `_create_material_tape`**
     （那个函数会写 material_items 并触发素材分析，练习盘没有素材）。
  2. **开场**：`opening-line` 的 `otype` 链里必须有 `expression` 分支——开场要给
     **第一个小任务 + 当前支架档位**（§4.4）；落进通用池就退化成"今天过得怎么样"
     式闲聊问句，与练习盘形态完全不符。判定顺序必须是 素材 > 练习盘 > 场景。
  3. **盘内合同**：`tutor_agent._build_control_prompt` 的**四处** prompt 分支都要挂
     `expression_section`——少一处 == 某条分支静默没有练习盘规矩（§5 硬约束 1）。
     首遍门槛与 §4.2 的度量口径：**闲聊盘不动**；练习盘侧 2026-09-20 起按盘型
     分叉（plans/0920-02），见 `test_first_pass_gate_splits_by_tape_type`。
  4. **显式提交**（§4.6）：前端练习盘关掉 2.4s/4.2s 静音自动提交（换成 20s 兜底），
     并给一枚"说完了"按钮；**闲聊盘的 2.4s/4.2s 一字不改**（§4.2 非目标）。
  5. **web-dist** 必须是当前母版构建出来的——忘了重建，真机拿到的还是旧前端
     （同一坑 `test_continue_tape_wiring` 已踩过一次）。

纯静态、不联网、不改任何数据。
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest
from pathlib import Path

# 开源包布局（相对本文件 backend/app/test_runner/）：
#   APP/     = backend/app        （main.py、agent/、engine/ 在这里）
#   ROOT/    = opensource/        （frontend/、voice/ 在这里）
# 原仓路径是 chanpinzhujian/api/... 与 design-review/...、deployment/...，
# 拆包后目录改名了；这里按新布局定位，语义（读源码做静态契约断言）一字不变。
HERE = os.path.dirname(os.path.abspath(__file__))       # backend/app/test_runner
APP = os.path.dirname(HERE)                             # backend/app
ROOT = os.path.dirname(os.path.dirname(APP))            # opensource/
MAIN_PY = os.path.join(APP, "main.py")
TUTOR_PY = os.path.join(APP, "agent", "tutor_agent.py")
SCAFFOLD_PY = os.path.join(APP, "engine", "expression_scaffold.py")
MJS = os.path.join(ROOT, "voice", "doubao-session-server.mjs")
MASTER = os.path.join(ROOT, "frontend", "cassette-learning-interaction-review.html")
WEB_DIST = os.path.join(ROOT, "frontend", "web-dist", "index.html")
BUILD_WEB = os.path.join(ROOT, "frontend", "build_web.py")


def read(path: str) -> str:
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


def build_expected() -> str:
    """母版经 build_web.py 的 8 处替换后应当长什么样（与 build_web 单一事实源）。"""
    import contextlib
    import importlib.util

    build_web = BUILD_WEB
    spec = importlib.util.spec_from_file_location("build_web_probe_expr", build_web)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    with contextlib.redirect_stdout(io.StringIO()):
        return module.build(MASTER)


class ExpressionTapeTypeTests(unittest.TestCase):
    """① 盘型 + ② 开场：main.py 侧的两段接线。"""

    @classmethod
    def setUpClass(cls):
        cls.py = read(MAIN_PY)

    def test_start_route_exists_and_returns_conv_id_and_title(self) -> None:
        self.assertIn('@app.post("/expression/start")', self.py,
                      "开练习盘的端点不见了")
        self.assertIn("class ExpressionStartReq(", self.py)
        block = re.search(
            r'@app\.post\("/expression/start"\)(.*?)(?=\n@app\.|\Z)', self.py, re.S)
        self.assertIsNotNone(block)
        self.assertIn("_create_expression_tape(", block.group(1),
                      "端点必须调建盘函数（否则只回 200 没有盘）")

    def test_creating_the_tape_does_not_reuse_the_material_tape(self) -> None:
        """§4.3：练习盘**不复用 `_create_material_tape`**（否则素材语义被整盘带进来）。"""
        block = re.search(
            r"async def _create_expression_tape\((.*?)(?=\n@app\.post)", self.py, re.S)
        self.assertIsNotNone(block, "_create_expression_tape 不见了")
        # docstring 里**刻意**提到了"不复用 `_create_material_tape`"——那是说明文字，
        # 不是调用。断言只能看代码体，否则自己把注释判成违规。
        doc = re.search(r'"""(.*?)"""', block.group(1), re.S)
        code = block.group(1).replace(doc.group(0), "") if doc else block.group(1)
        self.assertNotIn("_create_material_tape(", code)
        self.assertNotIn("add_material(", code)
        self.assertNotIn("material_items", code)
        # 盘型靠 save_conversation 的第 6 个位置参数（scene）带过去 —— 免迁移
        self.assertIn('"English", "B1", "expression", []', code,
                      "盘型必须落在 conversation_meta.scene='expression'")
        # 临时标题：先给一个能认出来的名字，真标题由既有归档链后换（§4.3）
        self.assertIn("练表达 · ", code, "临时标题不见了")

    def test_conv_id_keeps_the_material_tape_shape(self) -> None:
        """conv_id 与素材带同形（`xxx_A`，前端按 _A/_B 拆带与面）。"""
        self.assertIn('return "expr-%s_A" % uuid.uuid4().hex[:10]', self.py)

    def test_opening_otype_chain_has_an_expression_branch_before_scene(self) -> None:
        """§4.4：判定顺序 素材 > 练习盘 > 场景 —— 练习盘分支必须在 scene 之前。

        plans/0923-06 §4.1.3 起，练习盘开场**直出库内审定文本**（不再走开场 LLM）。
        """
        expr_branch_token = 'if otype == "expression":' if 'if otype == "expression":' in self.py else 'elif otype == "expression":'
        self.assertIn(expr_branch_token, self.py)
        expr_at = self.py.index(expr_branch_token)
        scene_at = self.py.index('elif otype == "scene":')
        self.assertLess(expr_at, scene_at, "练习盘分支必须排在场景分支之前")
        branch = self.py[expr_at:scene_at]
        self.assertIn("_expression_opening_payload(", branch)

    def test_expression_branch_does_not_use_the_opening_llm_or_scaffold(self) -> None:
        """§4.1.3（**反转自** plans/0919-02 §4.4）：首讲前不给框架、不走生成。

        旧契约要求练习盘开场携带支架档位并调 `render_opening_scaffold`；0923-06
        明确反转为「库内审定文本直出」——用户首讲前不给短语清单／三步框架／五维术语。
        这条静态钉子就是那次反转：少了它，旧写法会悄悄长回来。
        """
        expr_branch_token = 'if otype == "expression":' if 'if otype == "expression":' in self.py else 'elif otype == "expression":'
        expr_at = self.py.index(expr_branch_token)
        branch = self.py[expr_at:self.py.index('elif otype == "scene":')]
        self.assertNotIn("render_opening_scaffold(", branch)
        self.assertNotIn("get_learning_state(", branch)
        self.assertNotIn("chat_completion(", branch)
        self.assertIn("_expression_opening_payload(", branch)

    def test_expression_opening_carries_source_metadata_not_in_tts_text(self) -> None:
        """§4.1.4：来源标题／URL 只作元数据（`topic`），**不拼进 TTS 文本**。"""
        self.assertIn('"source": "expression_topic"', self.py)
        self.assertIn('"topic": meta', self.py)

    def test_source_readonly_route_exists(self) -> None:
        """§4.1.4：新增本盘只读来源查询，供云端恢复后的旧盘显示出处。"""
        self.assertIn('@app.get("/expression/topic-source/{user_id}")', self.py)
        self.assertIn("此前话题无来源记录", self.py)

    def test_expression_tape_detection_uses_scene_and_leaves_the_sentinel_alone(self) -> None:
        """盘型识别用 `scene` 取值；**不碰 `material_keywords` 哨兵**（§4.3 红线）。

        素材盘的问候禁令、[Material] 注入全靠那个哨兵；练习盘一旦写了它，
        整盘会收到"只谈这份素材"的规矩。
        """
        self.assertIn('str(scene or "").strip().lower() == "expression"', self.py)
        marker = self.py.index('str(scene or "").strip().lower() == "expression"')
        neighborhood = self.py[max(0, marker - 900):marker + 400]
        self.assertNotIn("material_keywords =", neighborhood)
        self.assertNotIn("material_keywords=", neighborhood)

    def test_quality_check_takes_an_expression_mode_flag(self) -> None:
        """开场质检：练习盘按"给任务"形态判（不误杀长句、不拿问候规则卡它）。"""
        self.assertIn("expression_mode: bool = False", self.py)
        self.assertIn('expression_mode=(otype == "expression")', self.py,
                      "质检调用没把盘型传进去 → 练习盘会被按闲聊口径卡")

    def test_all_fallback_paths_land_on_the_sourced_fallback_card(self) -> None:
        """兜底不得落通用池（§4.4）：所有出口都回到**有出处保底卡**。

        plans/0923-06 §4.1.3：旧的无源模板（`Let's practice. Tell me about one
        thing…`）统一改成库内审定、有真实来源的保底卡。
        """
        self.assertIn("def _expression_opening_fallback(", self.py)
        self.assertGreaterEqual(self.py.count("_expression_opening_fallback("), 3,
                                "兜底路径数不对：练习盘开场失败时会掉进通用池")
        self.assertNotIn("Let's practice. Tell me about one thing that happened", self.py)


class ExpressionTapeContractTests(unittest.TestCase):
    """③ 盘内合同：四处 prompt 分支都要挂上，红线一处不碰。"""

    @classmethod
    def setUpClass(cls):
        cls.py = read(TUTOR_PY)

    def test_section_is_built_once_and_used_in_all_four_branch_sites(self) -> None:
        """四处分支（权威/回落 × 缓存序 1/0）都挂 `expression_section`。"""
        self.assertIn("expression_section = self._expression_practice_section(ctx)", self.py)
        # 3 处列表 + 1 处赋值 = 4
        self.assertEqual(
            4, self.py.count("expression_section"),
            "expression_section 的落点数不对 —— 某条 prompt 分支会静默没有练习盘规矩")
        self.assertIn("def _expression_practice_section(", self.py)

    def test_expression_tape_is_recognised_only_by_cassette_mode(self) -> None:
        self.assertIn(
            'return str(getattr(ctx, "cassette_mode", "") or "") == "expression"', self.py)

    def test_session_start_accepts_the_expression_mode(self) -> None:
        self.assertIn('"fresh", "expression"}', self.py,
                      "/session/start 的 context_mode 白名单没收 expression")

    def test_first_pass_gate_splits_by_tape_type(self) -> None:
        """首遍门槛按最新盘型合同分流：闲聊需明确请求，练习盘按段落判。

        原来的闲聊 `sentences >= 3` 门槛会把普通长分享变成 F14 结构反馈，
        与 plans/0923-07 的闲聊不主动分析相冲突。修正后：

          · 闲聊盘只有明确请求反馈才开环；
          · 练习盘改用 `looks_like_a_passage` —— 豆包 ASR 对中文常常整段不带句末
            标点（`sentences` 退化成 1）⇒ 沿用 `>=3` 让**中文讲述的闭环根本不启动**；
          · 两条分支必须落在 `_opens_retell_loop` 里，不能在调用点写三元式
            （写死了就没法单测，也看不出哪一半属于哪个盘型）。
        """
        self.assertIn("return False", self.py[
            self.py.index("def _opens_retell_loop("):self.py.index("def _apply_learning_move(")],
            "闲聊盘的三句门槛仍可能把普通分享变成反馈任务")
        self.assertIn(
            "elif self._opens_retell_loop(ctx, metrics, user_transcript):", self.py,
            "_chunk_turn_layer 没走分叉函数（首遍门槛又被写回原地了）")
        block = re.search(
            r"def _opens_retell_loop\((.*?)(?=\n    @staticmethod)", self.py, re.S)
        self.assertIsNotNone(block, "_opens_retell_loop 不见了")
        body = block.group(1)
        self.assertIn("looks_like_a_passage(metrics)", body,
                      "练习盘首遍门槛没接上 looks_like_a_passage")
        self.assertIn("asks_for_expression_feedback(transcript)", body,
                      "口语明确要反馈时两盘都该放行，这条丢了")
        self.assertIn("if self._is_expression_tape(ctx):", body,
                      "练习盘判据没按盘型走")

    def test_scaffold_hooks_are_wired_on_both_hint_paths_and_the_recorder(self) -> None:
        """档位附注挂在**两条**给任务/分析的路径上；对比结果落一处记录。"""
        self.assertEqual(
            2, self.py.count("self._expression_scaffold_hint("),
            "档位附注的挂点数不对（对比轮 + 首遍轮各一处）")
        self.assertEqual(1, self.py.count("self._record_expression_scaffold("))
        # 字段叫 expression_scaffold（无下划线）；带下划线的是方法名。
        self.assertIn("expression_scaffold: Dict[str, Any] = field(default_factory=dict)", self.py)

    def test_verdict_line_is_same_source_as_the_tier(self) -> None:
        """提示侧的"进步判定"必须与支架档位**同一次判定**（plans/0920-02 残留项）。

        2026-09-20 之前，档位用 scaffold 的密度判据、提示却读 F14 的
        `verdict.reason`（语义＝"简洁进步"：句数下降且 details 没丢）。而盘内首遍
        提示明确邀请"再多讲点细节"——用户照做（5句/26细节 → 6句/40细节）时 F14 判
        `not_shorter` ＝"否（句数没有下降）"，档位那边却记了一次**达标** ⇒ 同一段
        提示里两套判定，AI 会当面否认系统认下的进步。这里把同源关系钉死：
        提示侧不许再接回 F14 的判据表。
        """
        self.assertNotIn("_VERDICT_TEXT", self.py,
                         "提示侧又接回 F14 的简洁判据了（两套判定会互相打脸）")
        self.assertIn("def _expr_progress_text(", self.py)
        self.assertIn(
            "from engine.expression_scaffold import met_and_reason_from_compare", self.py)
        self.assertIn("met_and_reason_from_compare(cmp)", self.py)
        self.assertIn("代码判定进步（判据与档位同源）：", self.py)

    def test_progress_text_covers_every_scaffold_reason(self) -> None:
        """话术表必须覆盖 scaffold 会产出的**每一个**原因串（跨文件对账）。

        两边各存一份字符串常量，谁单方面改名都不会报错——只会让模型看到裸原因码
        （`density_up` 这种），或更糟：判据达标而提示写"不可用"。所以这里从
        scaffold **导入**常量来对，不在源码里数键名。
        """
        sys.path.insert(0, os.path.dirname(HERE))
        from engine.expression_scaffold import (  # noqa: PLC0415
            REASON_DENSITY_DOWN,
            REASON_DENSITY_FLAT,
            REASON_DENSITY_UP,
            REASON_DETAILS_DROPPED,
            REASON_NO_COMPARE,
            REASON_VERDICT_UNAVAILABLE,
        )

        block = re.search(r"_PROGRESS_TEXT = \{(.*?)\n        \}", self.py, re.S)
        self.assertIsNotNone(block, "话术表 _PROGRESS_TEXT 不见了")
        keys = set(re.findall(r'"([a-z_]+)":', block.group(1)))
        for reason in (REASON_DENSITY_UP, REASON_DENSITY_FLAT, REASON_DENSITY_DOWN,
                       REASON_DETAILS_DROPPED, REASON_NO_COMPARE,
                       REASON_VERDICT_UNAVAILABLE):
            self.assertIn(reason, keys, "话术表缺 " + reason)
        # 不可用那两类原因串是 F14 侧透传过来的（scaffold 不自己造），也得有词
        for passthrough in ("caliber_mismatch", "density_unavailable"):
            self.assertIn(passthrough, keys, "话术表缺 " + passthrough)


class ExpressionScaffoldModuleTests(unittest.TestCase):
    """§4.5 落差层必须是**纯函数模块**（零 LLM 调用、可离线单测）。"""

    @classmethod
    def setUpClass(cls):
        cls.py = read(SCAFFOLD_PY)

    def test_module_is_pure_and_llm_free(self) -> None:
        for forbidden in ("llm_client", "openai", "requests", "asyncio"):
            self.assertNotIn(forbidden, self.py,
                             "落差层必须零 LLM / 零 IO（档位由代码算才权威）")

    def test_four_tiers_and_both_streaks_are_declared(self) -> None:
        for needle in ("SCAFFOLD_MIN_TIER = 1", "SCAFFOLD_MAX_TIER = 4",
                       "TIER_DOWN_STREAK = 2", "TIER_UP_STREAK = 2",
                       'SCAFFOLD_ITEM_TYPE = "expression_practice"'):
            self.assertIn(needle, self.py)


class VoiceProxyModeTests(unittest.TestCase):
    """语音代理的 contextMode 白名单必须收 `expression`（否则整盘退化成空白带）。"""

    @classmethod
    def setUpClass(cls):
        cls.mjs = read(MJS)

    def test_cassette_mode_whitelist_includes_expression(self) -> None:
        self.assertIn(
            '["resume", "continuation", "continuation_candidate", "fresh", "expression"]',
            self.mjs,
            "代理不认 contextMode='expression' → 练习盘会被当闲聊盘起会话，"
            "盘内合同整段失效")
        self.assertRegex(
            self.mjs,
            r'const cassetteMode = \[[^\]]*"expression"\]\.includes\(config\.contextMode\)')


class ExpressionOpeningQualityCheckTests(unittest.TestCase):
    """§4.4 开场质检：练习盘的开场是**任务**，可以是祈使句（没问号也算过）。

    这是一条**活体踩出来的**回归（2026-09-19）：模型给出

        Tell me one thing you did today. Use words like morning, work, lunch, home.

    ——形态与方案 §4.4 自己的样例（"讲一件你最近遇到的有意思的事"）一致，还正确
    带上了第 1 档的关键词帮助，却被 `no_question` 判违规；两次都不过 → 整盘掉进
    模板兜底，模型那句更好的开场被丢掉。根因是把问号判据只放宽了"多问"那一半，
    没管"零问号"那一半。所以这里把**零问号**这条钉住，同时钉住"别的盘仍要问句"。
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import main  # noqa: PLC0415  （既有约定：开场质检的单测直接 import 主模块）

        # 必须包 staticmethod：裸函数挂成类属性会变成**绑定方法**，`self` 悄悄占掉
        # 第一个参数（text），断言全都在拿 TestCase 对象当文案——这个坑真踩过一次。
        cls.qc = staticmethod(main._opening_quality_check)

    def test_imperative_task_without_a_question_mark_passes(self) -> None:
        for task in (
            "Tell me one thing you did today. Use words like morning, work, lunch, home.",
            "Tell me about one thing that happened to you recently. Last week, ...",
            "Talk about something interesting that happened to you recently.",
        ):
            with self.subTest(task=task):
                self.assertEqual("", self.qc(task, [], expression_mode=True))

    def test_task_with_a_question_mark_also_passes(self) -> None:
        for task in (
            "Tell me about one thing that happened to you recently - what happened?",
            "What did you do today? Use words like morning, work, lunch.",
        ):
            with self.subTest(task=task):
                self.assertEqual("", self.qc(task, [], expression_mode=True))

    def test_multiple_questions_are_allowed_on_the_practice_tape(self) -> None:
        # 两个问号才算"多问"——练习盘允许（任务句 + 一句帮助式追问）
        text = "Tell me about your weekend. What did you do? Who was with you?"
        self.assertEqual("", self.qc(text, [], expression_mode=True))
        # 别的盘仍是"恰好一问"（这条豁免只给练习盘）
        self.assertEqual("multi_question", self.qc(text, []))

    def test_off_tape_still_requires_a_question(self) -> None:
        task = "Tell me one thing you did today."
        self.assertEqual("no_question", self.qc(task, []))

    def test_greeting_is_still_rejected_on_the_practice_tape(self) -> None:
        """豁免的只有问句判据；"不问候"这条与语言无关的硬线不动。"""
        text = "Hi there! Tell me about one thing that happened to you recently."
        self.assertEqual("greets", self.qc(text, [], expression_mode=True))

    def test_grand_topic_is_still_rejected_on_the_practice_tape(self) -> None:
        """同样保留：练习盘开场不得问爱好/梦想/自我介绍这类宏大话题（§4.4）。"""
        text = "Tell me about your hobbies and your dreams for the future."
        self.assertEqual("grand_topic", self.qc(text, [], expression_mode=True))


class ExpressionFrontendWiringTests(unittest.TestCase):
    """④ 显式提交 + ⑤ 母版/web-dist 一致性（前端侧）。"""

    def _check_wiring(self, html: str, where: str) -> None:
        # 盘型入口按钮
        self.assertIn('id="exprTape"', html, where + "：开练习盘的按钮不见了")
        # 显式提交按钮：默认必须收起（非练习盘不许出现）
        submit = re.search(r'<button[^>]*id="exprSubmit"[^>]*>', html)
        self.assertIsNotNone(submit, where + '："说完了"按钮不见了')
        self.assertIn("hidden", submit.group(0),
                      where + '：非练习盘默认收起 —— 没 hidden 就是全盘都亮着')
        # 2026-09-20 **二次收口**（用户亲测后定调）：**说明文案整个删掉**，
        # 整条只在"正在采集"时出现。原先按盘型判 → 没录音时会冒出一个只有文案、
        # 没有按钮的空盒子（用户截图反馈："这个文本提示没必要，只需要说话的时候
        # 有那个说完了的按钮"）。所以这里反过来钉**删除**，防止哪天又被加回来：
        #   ① 元素 `#exprSubmitHint`、CSS `.expr-submit-hint`、JS 常量 `EXPR_HINT`
        #      三处都不得复活（一个残留就会让空盒子重新出现）；
        #   ② 显隐判据**只有一条** `running`（练习盘 && 采集中 && 未提交），
        #      整条/计时/按钮全跟它走 —— 分两级判就又会出空盒子。
        self.assertNotIn("exprSubmitHint", html,
                         where + "：说明文案已按用户要求删除，元素/CSS/JS 都不许复活")
        self.assertNotIn("expr-submit-hint", html,
                         where + "：旧提示的 CSS 规则还在（元素没了、样式留着＝死代码）")
        self.assertNotIn("讲多久都行，停顿没关系", html,
                         where + "：旧说明文案随元素一起删除")
        self.assertIn('if(strip)strip.hidden=!running', html,
                      where + "：整条必须跟 running 走，否则没录音时冒出只有文案的空盒子")
        # §4.8 长录制呈现（2026-09-20 收口版：三件一处管显隐）。这里钉**行为**、
        # 不钉实现函数名——原版 `paintExprRecTimer` 已被 2fc90140 并进
        # `syncExpressionControls` 并挪进对话窗口。要紧的是这条：
        # 提交键与计时必须绑"**真的在采集中**"，不能只绑盘型——只绑盘型就会出现
        # "柜子里/没在录音时按钮也亮着、点了被前置判断挡回来"，用户感受是"点了没反应"
        # （B 批原版就是这个行为，2fc90140 修掉）。两条都钉住，防回归。
        self.assertIn('id="exprRecTimer"', html)
        self.assertIn("已录 ", html)
        self.assertIn("submit.hidden=!running", html)
        self.assertIn("timer.hidden=!running", html)
        # 显隐同步：换带时按盘型亮/灭
        self.assertIn("function isExpressionTape(", html)
        self.assertIn("function syncExpressionControls(", html)
        self.assertIn("syncExpressionControls();", html,
                      where + "：renderTape 没同步显隐 → 换到闲聊盘按钮还亮着")
        # 建盘：POST /expression/start，并把 scene='expression' 交给合并
        self.assertIn("'/expression/start'", html)
        self.assertIn("scene:'expression',messages:[]", html,
                      where + "：建盘后没带盘型 → 前端不认它是练习盘")
        # §4.3：盘型归一（theme='expression' → contextMode='expression'）
        self.assertIn("contextMode:'expression'", html)
        self.assertIn("isExprTape?'expression'", html,
                      where + "：__tapeContext 没按 theme 归一盘型")
        # §4.6 静音窗：练习盘 20s 兜底；闲聊盘 2.4s/4.2s
        silence_ok = (
            "_exprTape ? 20000 : (soundsUnfinished ? 4200 : 2400)" in html or
            "_exprTape ? 20000 : (_chatTape ? 4200 : (soundsUnfinished ? 4200 : 2400))" in html
        )
        self.assertTrue(
            silence_ok,
            where + "：静音窗三元式不对 —— 要么练习盘没 20s 兜底，要么闲聊盘被改了口径"
        )
        # 提交走既有 stop_speech 收口，reason='manual'（不新增协议字段）。
        # ⚠ 收口已经**挪进第一个 <script> 的语音桥**（`commitTurnManually`）——
        # 练习盘的「说完了」不能再从块 2 直接发（那里拿不到 `commitSent`/`ws`，
        # 正是"commitSent is not defined"那次的根因）。所以断言按桥里的写法钉。
        self.assertIn("type: 'stop_speech', reason: 'manual'", html,
                      where + "：手动提交没有走 stop_speech/manual 这条既有收口")
        # plans/0923-06 §4.1.4：练习盘标题旁的**来源外链**（只读元数据，不拼进 TTS 文本）
        self.assertIn('id="tapeSource"', html, where + "：来源外链元素不见了")
        self.assertIn("function renderTapeSource(", html)
        self.assertIn("function ensureTapeSource(", html)
        self.assertIn("topic: data.topic || null", html,
                      where + "：开场响应没把来源元信息带到前端")
        # ⚠ 键名**不能**写块 1 的私有 `configuredConversationKey`（跨块裸引用 = 运行时
        # ReferenceError），必须走块 2 自己的同源公式 `activeConversationKey()`
        # （= `convId || id + '_A'`，与块 1 :1421/:2669、块 2 :3013 逐字同源）。
        self.assertIn("exprTopicByKey[activeConversationKey()]=payload.topic||null", html,
                      where + "：来源没按 conv_id 分开存 —— 换带会把上一盘的出处留在新盘上")
        # 只认 https，且标题必须用 textContent 填（来源标题是**数据**：innerHTML 会被注入）
        self.assertIn("/^https:\\/\\//i.test(url)", html,
                      where + "：来源 URL 没做 https 白名单")
        self.assertIn("a.textContent='Source · '+title", html,
                      where + "：来源标题必须用 textContent 填")
        self.assertIn("/expression/topic-source/", html,
                      where + "：没接只读来源查询（云端恢复的旧盘就永远显示不出出处）")
        self.assertIn("此前话题无来源记录", html,
                      where + "：历史无分配必须如实说明，不许补造「当时已看到」的出处")
        # 换带 / 进场都要同步（含"无记录"态）
        self.assertIn("renderTapeSource();", html)
        self.assertIn("ensureTapeSource();", html)

    def test_master_carries_the_full_wiring(self) -> None:
        self._check_wiring(read(MASTER), "母版")

    def test_web_dist_carries_the_full_wiring(self) -> None:
        self._check_wiring(read(WEB_DIST), "web-dist")

    def test_web_dist_matches_current_master(self) -> None:
        self.assertEqual(
            build_expected(), read(WEB_DIST),
            "web-dist 过期：重跑 python deployment/build_web.py")


if __name__ == "__main__":
    unittest.main()
