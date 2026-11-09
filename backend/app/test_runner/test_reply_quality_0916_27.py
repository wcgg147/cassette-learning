# -*- coding: utf-8 -*-
"""plans/0916-27 回复质量律：律 0 对题 + 三律（单测）。

用户原话（2026-09-17）："我说我昨天吃的什么，他就以为我在聊前天的事情——
这是精确度的问题。" 本文件把四类判据钉死，并用**用户的原话当用例**。

跑法：cd chanpinzhujian/api && python test_runner/test_reply_quality_0916_27.py
"""
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, ".")

from agent import reply_quality as rq  # noqa: E402

_results = []


def check(label, cond, detail=""):
    _results.append(bool(cond))
    print(f"{label} ->", "PASS" if cond else f"FAIL {detail}")


def main():
    print("\n[1] 律 0·时间锚对齐（用户原话用例）")
    user = "What did I eat yesterday?"
    raw = "我昨天吃的什么"
    bad = "You were talking about work the day before yesterday, right?"
    good = "You had noodles yesterday — did you enjoy them?"
    _viol = rq.topic_alignment_violations(user, bad, raw)
    check("答成前天 → 判越界（时间锚）",
          any(v.startswith("对题失败·时间锚") for v in _viol), f"{_viol}")
    check("谈昨天 → 不误伤",
          not any(v.startswith("对题失败·时间锚") for v in rq.topic_alignment_violations(user, good, raw)),
          f"{rq.topic_alignment_violations(user, good, raw)}")
    check("明确更正（Did you mean yesterday?）→ 放行",
          not rq.topic_alignment_violations(user, "Did you mean yesterday? Let me answer that.", raw),
          "")
    check("同一时间锚同时出现 → 放行",
          not any(v.startswith("对题失败·时间锚") for v in
                  rq.topic_alignment_violations(user, "You asked about yesterday, not the day before.", raw)),
          "")

    print("\n[2] 律 0·核心对象对齐")
    guitar_user = "I want to talk about my guitar lesson last week"
    off_topic = "That sounds nice. How do you usually spend your weekends?"
    on_topic = "Your guitar lesson last week — did you enjoy it?"
    viol_off = rq.topic_alignment_violations(guitar_user, off_topic)
    check("通篇不提用户谈的对象 → 判越界",
          any(v.startswith("对题失败·核心对象") for v in viol_off), f"{viol_off}")
    check("提到该对象 → 不误伤",
          not rq.topic_alignment_violations(guitar_user, on_topic), "")

    print("\n[3] 律 0·问句必须被回答 / 自述理解过程")
    only_question = "What do you think you ate?"
    check("整段只有反问 → 判越界",
          any(v.startswith("对题失败·只反问") for v in rq.topic_alignment_violations(user, only_question, raw)),
          "")
    check("先答再反问 → 不误伤",
          not rq.topic_alignment_violations(user, "You had ramen yesterday. Was it good?", raw),
          "")
    check("自述理解过程（用户要的是答案）→ 判越界",
          any(v.startswith("对题失败·自述理解过程") for v in rq.topic_alignment_violations(
              user, "I thought you meant you went to work yesterday. Let me re-explain.", raw)),
          "")

    print("\n[4] 三律·语言增量 / 复述当回应")
    liked = "I made dumplings with my mom"
    check("纯过渡句（零信息）→ 判越界",
          any(v.startswith("增量为零") for v in rq.reply_quality_violations(liked, "Got it!")),
          "")
    check("带具体内容 → 不误伤",
          not rq.reply_quality_violations(liked, "Dumplings! What did you put inside?"),
          "")
    echo = "So you mean you went to the beach with your brother last weekend?"
    check("复述当回应（So you mean… 换词抛回，零新内容）→ 判越界",
          any(v.startswith("增量不足·复述") for v in rq.reply_quality_violations(
              "I went to the beach with my brother last weekend", echo)),
          f"{rq.reply_quality_violations('I went to the beach with my brother last weekend', echo)}")
    check("低置信确认轮：复述确认不罚（合同要求的）",
          not rq.reply_quality_violations(
              "I went to the beach with my brother last weekend", echo, confirmation_turn=True),
          "")
    check("戏内角色人格：短台词不判零信息（Here you are.）",
          not rq.reply_quality_violations("I'd like a coffee please", "Here you are.", scene_role=True),
          "")

    print("\n[5] 三律·承认收窄（只认 AI 自己的错，且不许自证清白开场）")
    correction_raw = "我没说你昨天上班了，我是说我妈"
    excuse = "I don't have any record of you saying that. Let's talk about something else."
    check("用户纠错后以'我没有记录…'开头 → 判越界",
          any(v.startswith("承认律") for v in rq.reply_quality_violations(
              "I never said I went to work yesterday", excuse, raw_text=correction_raw)),
          "")
    check("先认下再修正 → 不误伤",
          not rq.reply_quality_violations(
              "I never said I went to work yesterday",
              "You're right, I got that wrong — you were talking about your mom.",
              raw_text=correction_raw),
          "")

    print("\n[6] 兜底确认（绝不硬答）")
    confirm = rq.alignment_confirm_reply(user, bad, raw)
    check("时间锚被换 → 给出一句简短确认", bool(confirm) and "yesterday" in confirm, f"{confirm!r}")
    check("没有违规 → 不给确认",
          rq.alignment_confirm_reply(user, good, raw) is None, "")

    print("\n[7] alignment_state 裁定（进 turn_contract.alignment 的那份）")
    state = rq.alignment_state(user, raw)
    check("时间锚 = yesterday", state.get("time_anchors") == ["yesterday"], f"{state}")
    check("问句标记", state.get("is_question") is True, f"{state}")
    check("普通陈述不产生时间锚", rq.alignment_state("I like coffee", "我喜欢咖啡").get("time_anchors") == [], "")

    print(f"\n结论：{sum(_results)}/{len(_results)} 通过")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())


# 双模：也能被 run_suite（unittest）收进来，避免这套判据只活在"手动跑"里。
import unittest  # noqa: E402


class ReplyQuality0916Tests(unittest.TestCase):
    def test_all_cases_pass(self):
        self.assertEqual(0, main())
