# -*- coding: utf-8 -*-
"""人格能力边界（2026-10-30）：遇到超出她生活经验的东西，她像个不懂的人那样反应。

跑法：cd chanpinzhujian/api && PYTHONPATH=. py -3.12 -m unittest agent.test_companion_limits

背景（真机 QQ）：主人发 UTF-8 十六进制，她回"这串十六进制我认得，翻出来是'我爱你'"——
真人不会心算编码。条款写在 `agent/companion_contract.render_companion_limits()`，
同时进会话级 `_build_system_prompt` 与逐轮 `_build_control_prompt`（只改一处不生效）。
本文件只守"它还在、且在三条链路里都在"，具体效果看真机回复，不靠断言词表。
"""

import os
import sys
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql://llm_user:***@localhost:5432/llm_os")
os.environ.setdefault("MEMORY_BACKEND", "local")
sys.path.insert(0, ".")

from agent.companion_contract import (  # noqa: E402
    render_companion_limits,
    render_companion_social_contract,
)
from agent.test_communication_style_wiring import Ctx, _bare_tutor  # noqa: E402
from agent.test_expression_contract import control_ctx, stable_prepared  # noqa: E402

MARK = "[Companion limits]"


class CompanionLimitsTests(unittest.TestCase):
    def test_block_is_reaction_not_prohibition(self) -> None:
        text = render_companion_limits()
        # 三条具体反例（few-shot）在：编码 / 写代码 / 冷门百科
        self.assertIn("十六进制", text)
        self.assertIn("写代码", text)
        self.assertIn("冷门百科", text)
        # 分寸：听得懂技术话题，只是不动手（不许写成"她不懂技术"）
        self.assertIn("听得懂", text)
        # 反应方式而不是禁令清单：不出现"严禁/禁止/必须"
        for word in ("严禁", "禁止", "必须"):
            self.assertNotIn(word, text)

    def test_shared_contract_carries_it(self) -> None:
        """Director / 逐轮控制面 / QQ 接入层共用这一份。"""
        self.assertIn(MARK, render_companion_social_contract())

    def test_session_prompt_carries_it_on_chat_tape(self) -> None:
        prompt = _bare_tutor()._build_system_prompt(Ctx())
        self.assertIn(MARK, prompt)
        # 它在 [Becoming] 之后，与"她是长出来的"同一段人格层
        self.assertLess(prompt.index("[Becoming]"), prompt.index(MARK))

    def test_material_tape_stays_byte_clean(self) -> None:
        class MaterialCtx(Ctx):
            material_context = {"title": "t"}

        prompt = _bare_tutor()._build_system_prompt(MaterialCtx())
        self.assertNotIn(MARK, prompt)
        self.assertNotIn("[Becoming]", prompt)

    def test_control_prompt_carries_it(self) -> None:
        agent, ctx = control_ctx(1)
        prompt = agent._build_control_prompt(ctx, stable_prepared("guide"), "")
        self.assertIn(MARK, prompt)


if __name__ == "__main__":
    unittest.main()
