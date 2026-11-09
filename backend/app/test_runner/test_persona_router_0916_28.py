# -*- coding: utf-8 -*-
"""plans/0916-28 人格路由与记忆分区（单测）。

用户拍板（2026-09-17）：①先做人格路由、不拆 agent（同一模型做不同的事）；
②`origin` 两级标签（real/scene）+ 按"盘"(conv_id)作用域。

跑法：cd chanpinzhujian/api && python test_runner/test_persona_router_0916_28.py
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, ".")

from agent import persona_router as pr  # noqa: E402
from agent.tutor_agent import SessionContext, TutorAgent  # noqa: E402

_results = []


def check(label, cond, detail=""):
    _results.append(bool(cond))
    print(f"{label} ->", "PASS" if cond else f"FAIL {detail}")


def fake_ctx(**kwargs):
    ctx = SessionContext(session_id="s", user_id="zz-selftest-persona", target_lang="en")
    for key, value in kwargs.items():
        setattr(ctx, key, value)
    return ctx


def main():
    print("\n[1] 人格路由三态（确定性、用户零操作）")
    out_scene = pr.resolve_persona("I'd like a latte please", scene_active=False)
    check("未进场景 → 朋友人格（模式 A）", out_scene["mode"] == "companion", f"{out_scene}")
    in_scene = pr.resolve_persona("I'd like a latte please", scene_active=True)
    check("在戏里推进情节 → 角色人格（模式 B）", in_scene["mode"] == "scene_role", f"{in_scene}")
    check("角色人格：教学退到最低", in_scene["teaching_posture"] == "low", f"{in_scene}")
    exit_cue = pr.resolve_persona("等一下，这句怎么说？", scene_active=True, raw_text="等一下，这句怎么说？")
    check("显式出戏问句 → 外教视角（半入戏）", exit_cue["mode"] == "tutor_sidecar", f"{exit_cue}")
    lang_q = pr.resolve_persona(
        "what does latte mean", scene_active=True,
        turn_contract={"language_question": {"kind": "term_meaning"}})
    check("场景内问词义（language_question）→ 外教视角", lang_q["mode"] == "tutor_sidecar", f"{lang_q}")
    started = pr.resolve_persona("你来当咖啡店店员，我们开始", scene_active=True, scene_changed=True)
    check("刚入戏那一句 → 角色人格（不被同一句里的『怎么说』带偏）",
          started["mode"] == "scene_role", f"{started}")

    print("\n[2] 人格规范块与 prompt 段")
    companion = pr.persona_constraint(out_scene, learner_name="Jason", tutor_name="Alice")
    section = pr.render_persona_section(companion)
    # 2026-10-10 契约更新（plans/1010-01 §181/§219）：companion 不再假定"已经是
    # 熟人朋友"，熟悉度改由 Relationship arc 契约决定 ⇒ 断言从"含朋友口径"
    # 改为"含称呼、不冒称熟人、并挂关系契约"。旧断言与新的验收标准直接冲突。
    check("模式 A 段文本含称呼、不冒称熟人、并挂关系契约",
          "Jason" in section and "Alice" in section
          and "朋友" not in section and "熟人" not in section
          and "Relationship arc" in section, section)
    scene_payload = pr.persona_constraint(
        in_scene, scene="咖啡店点单", tutor_role="店员", learner_role="顾客")
    scene_section = pr.render_persona_section(scene_payload)
    check("模式 B 段文本含角色与场景",
          "店员" in scene_section and "咖啡店点单" in scene_section, scene_section)
    check("模式 B 不问什么都不写戏外内容", "朋友" not in scene_section, scene_section)
    check("空约束 → 空段（零字符增量）", pr.render_persona_section({}) == "", "")

    print("\n[3] 入戏开关复用既有形态（归档定标签用）")
    check("盘里有『你来当…店员』→ 场景盘",
          pr.looks_like_scene_tape([
              {"role": "user", "content": "我来当顾客，你来当咖啡店店员"},
              {"role": "assistant", "content": "Sure, what can I get you?"},
          ]) is True, "")
    check("普通闲聊盘 → 不是场景盘",
          pr.looks_like_scene_tape([
              {"role": "user", "content": "我昨天吃了拉面"},
              {"role": "assistant", "content": "That sounds great."},
          ]) is False, "")
    check("只看用户消息（AI 台词不算入戏请求）",
          pr.looks_like_scene_tape([
              {"role": "assistant", "content": "假装我们是店员和顾客"},
          ]) is False, "")

    print("\n[4] 记忆分区：两级标签 + 按盘作用域")
    tagged = pr.tag_origin("我点了冰美式", pr.SCENE_ORIGIN)
    origin, body = pr.split_origin(tagged)
    check("打标签/剥标签往返", origin == "scene" and body == "我点了冰美式", f"{tagged!r}")
    check("幂等：重复打不叠加", pr.split_origin(pr.tag_origin(tagged, "scene"))[1] == "我点了冰美式", "")
    check("旧数据（无前缀）按 real 处理", pr.split_origin("老数据")[0] == "real", "")

    entities = [
        {"etype": "thing", "name": "iced americano", "detail": pr.tag_origin("我点了冰美式", "scene"), "conv_id": "cafe-1"},
        {"etype": "thing", "name": "燕麦拿铁", "detail": pr.tag_origin("我点了燕麦拿铁", "scene"), "conv_id": "cafe-1"},
        {"etype": "person", "name": "Jason", "detail": pr.tag_origin("用户的名字", "real"), "conv_id": "chat-9"},
    ]
    friend_view = pr.filter_entities_for_persona(entities, persona_mode="companion", conv_id="chat-9")
    names = [e["name"] for e in friend_view]
    check("戏外人格：别的盘的戏内事实被挡住",
          "iced americano" not in names and "燕麦拿铁" not in names, f"{names}")
    check("戏外人格：真实事实保留", "Jason" in names, f"{names}")
    check("读侧剥掉前缀（模型看不到 [scene] 标记）",
          all("[scene]" not in str(e.get("detail")) for e in friend_view), f"{friend_view}")
    scene_view = pr.filter_entities_for_persona(entities, persona_mode="scene_role", conv_id="cafe-2")
    check("戏内人格：不串戏（别盘的戏内事实一律按盘挡住）",
          "iced americano" not in [e["name"] for e in scene_view], f"{[e['name'] for e in scene_view]}")
    check("戏内人格：真实事实仍跨盘可见",
          "Jason" in [e["name"] for e in scene_view], f"{[e['name'] for e in scene_view]}")
    same_tape_view = pr.filter_entities_for_persona(entities, persona_mode="companion", conv_id="cafe-1")
    check("本盘作用域：自己那盘的戏内事实仍可读",
          "iced americano" in [e["name"] for e in same_tape_view], "")

    print("\n[5] 接线：事实卡按人格分区（戏内事实不跨人格外泄）")
    agent = TutorAgent.__new__(TutorAgent)
    ctx = fake_ctx(scene_context_set=True, persona_state={"mode": "scene_role"})
    agent._remember_session_fact(ctx, "我点了冰美式")
    check("戏内事实带 scene 标签",
          pr.split_origin(ctx.session_facts[0])[0] == "scene", f"{ctx.session_facts}")
    check("戏内人格读得到",
          agent._persona_scoped_facts(ctx) == ["我点了冰美式"], f"{agent._persona_scoped_facts(ctx)}")
    ctx.persona_state = {"mode": "companion"}
    check("切回朋友人格后读不到（不是用户的真实生活）",
          agent._persona_scoped_facts(ctx) == [], f"{agent._persona_scoped_facts(ctx)}")
    ctx2 = fake_ctx(scene_context_set=False)
    agent._remember_session_fact(ctx2, "我昨天吃了拉面")
    check("戏外事实带 real 标签且任何人格都读得到",
          pr.split_origin(ctx2.session_facts[0])[0] == "real"
          and agent._persona_scoped_facts(ctx2) == ["我昨天吃了拉面"], f"{ctx2.session_facts}")

    print("\n[6] 接线：律 0 对题裁定进合同 / 实体读侧按人格过滤")
    ctx3 = fake_ctx(last_raw_transcript="我昨天吃的什么")
    contract = agent._alignment_contract("What did I eat yesterday?", ctx3)
    check("时间锚进合同（模型生成前就知道要对准昨天）",
          contract.get("time_anchor") == "yesterday", f"{contract}")
    check("合同带原文（中文侧比对用）", contract.get("user_raw") == "我昨天吃的什么", f"{contract}")
    plain = agent._alignment_contract("ok", fake_ctx(last_raw_transcript=""))
    check("无锚无对象的不提问轮 → 空合同（零增量）", plain == {}, f"{plain}")
    ctx4 = fake_ctx(conversation_id="cafe-2", memory_entities=entities,
                    persona_state={"mode": "companion"})
    scoped = agent._persona_scoped_entities(ctx4)
    check("实体读侧按盘过滤（别的盘的戏内事实进不了 prompt）",
          [e["name"] for e in scoped] == ["Jason"], f"{[e['name'] for e in scoped]}")
    ctx4b = fake_ctx(conversation_id="cafe-1", memory_entities=entities,
                     persona_state={"mode": "companion"})
    check("实体读侧：本盘戏内事实保留（作用域=盘，不是人格）",
          "iced americano" in [e["name"] for e in agent._persona_scoped_entities(ctx4b)], "")

    print("\n[7] 注入池：到期复习项进机会池（需要巩固的）")
    ctx5 = fake_ctx(due_reviews=[{"item_type": "en_past_tense", "label": "过去时"}])
    rows = __import__("asyncio").run(agent._chunk_rows(ctx5))
    check("due review 合成为池行（status=activated / source_type=due_review）",
          any(r.get("source_type") == "due_review" and r.get("status") == "activated"
              and r.get("norm") == "过去时" for r in rows), f"{rows}")

    print(f"\n结论：{sum(_results)}/{len(_results)} 通过")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())


# 双模：也能被 run_suite（unittest）收进来。
import unittest  # noqa: E402


class PersonaRouter0916Tests(unittest.TestCase):
    def test_all_cases_pass(self):
        self.assertEqual(0, main())
