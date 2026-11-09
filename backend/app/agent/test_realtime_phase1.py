import asyncio
import unittest

try:
    from .tutor_agent import SessionContext, TutorAgent
except ImportError:
    from agent.tutor_agent import SessionContext, TutorAgent


class RealtimePhaseOneTests(unittest.TestCase):
    def test_embedded_gloss_cleanup_preserves_complete_bilingual_units(self):
        self.assertEqual(
            TutorAgent._normalize_embedded_glosses(
                "张先生，我来check（检查）您的预订，在system（系统）里。"
            ),
            "张先生，我来检查您的预订，在系统里。",
        )
        complete = "Can I pay in cash?（我可以用现金付款吗？）"
        self.assertEqual(TutorAgent._normalize_embedded_glosses(complete), complete)

    def test_stream_forwards_real_prepare_progress_before_reply(self):
        async def run():
            agent = TutorAgent.__new__(TutorAgent)
            ctx = SessionContext(session_id="s", user_id="u", target_lang="en")
            agent._sessions = {"s": ctx}

            async def prepare(_session_id, _text, progress_hook=None):
                if progress_hook:
                    progress_hook({
                        "type": "progress",
                        "phase": "memory_lookup",
                        "text": "我找一下你提到的那盘记录。",
                        "spoken": True,
                    })
                return {
                    "session_id": "s",
                    "policy": {},
                    "learning_move": {},
                    "learning_state": {},
                    "learner_state": {},
                    "constraints": {"session_memory": {"route": {}, "recall": []}},
                    "turn_execution": {
                        "route_tier": "fast",
                        "planning_route": "fast_path",
                    },
                    "conversation_director": {
                        "reply_draft": "Cash means 现金。",
                        "interaction_frame": "language_sidecar",
                        "language_presentation": "target_only",
                        "_meta": {"source": "local_fastpath", "latency_ms": 0},
                    },
                }

            async def record(*_args, **_kwargs):
                return None

            agent.prepare_turn = prepare
            agent._record_metering_event = record
            agent._controlled_reply_violations = lambda *_args, **_kwargs: []
            agent._enforce_controlled_reply = (
                lambda reply, *_args, **_kwargs: reply
            )

            events = []
            async for event in agent.respond_turn_stream(
                "s", "What does cash mean?", "turn-1"
            ):
                events.append(event)

            self.assertEqual(events[0]["type"], "progress")
            self.assertEqual(events[0]["phase"], "memory_lookup")
            self.assertEqual(events[1]["type"], "meta")
            self.assertEqual(events[1]["turn_execution"]["route_tier"], "fast")
            self.assertTrue(any(item["type"] == "phrase" for item in events))
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(events[-1]["generation"]["route_tier"], "fast")
            self.assertIn("prepare_latency_ms", events[-1]["generation"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
