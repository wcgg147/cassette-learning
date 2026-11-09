# -*- coding: utf-8 -*-
"""闲聊盘诚实合同拆分（2026-10-22，plans/1022-01 点3）。

⑧（不编生活细节）是诚实底线，默认开；
⑥⑦⑨⑩（被质疑先接住/给决定/不安慰/先给判断）是语气硬门，保持默认关。
本测试钉住拆分，防止以后又把 ⑨/⑩ 混回默认开的那段。
"""

from __future__ import annotations

import unittest

from agent import tutor_agent as tutor


class ChatTapeHonestySplitTest(unittest.TestCase):
    def test_no_fabrication_contains_only_rule_eight(self) -> None:
        text = tutor._CHAT_TAPE_NO_FABRICATION
        self.assertIn("不编生活细节", text)
        self.assertNotIn("不安慰", text)
        self.assertNotIn("先给判断", text)

    def test_tone_contract_keeps_nine_and_ten(self) -> None:
        text = tutor._CHAT_TAPE_HONESTY_CONTRACT
        self.assertIn("⑨", text)
        self.assertIn("⑩", text)
        self.assertNotIn("⑧ 不编生活细节", text)


if __name__ == "__main__":
    unittest.main()
