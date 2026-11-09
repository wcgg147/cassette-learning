"""Outcome-based learning loop for language tutoring.

The error fingerprint tells us what the learner got wrong.  This service keeps
the missing second half of the learning signal: whether the learner can use the
target structure again immediately, in a new context, and after a delay.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, Iterable, Optional


_PROBE_LIBRARY = {
    "en_past_tense": ("immediate_recall", "请把刚才那句话用过去式再说一遍。不要看答案。"),
    "en_third_person_s": ("immediate_recall", "换一个主语是 he 或 she 的句子，再说一遍。"),
    "en_articles": ("immediate_recall", "请自己造一个含 a/an/the 的短句。"),
    "en_plurals": ("immediate_recall", "请换一个复数名词，把句子再说一遍。"),
    "en_present_perfect": ("immediate_recall", "请用 have/has + 过去分词描述一次经历。"),
    "en_prepositions": ("immediate_recall", "请换一个场景，用正确的介词再说一遍。"),
    "ja_wa_ga": ("immediate_recall", "请换一个主语，自己选择「は」或「が」再说一遍。"),
    "ja_ni_de": ("immediate_recall", "请换一个地点，自己选择「に」或「で」再说一遍。"),
    "ja_te_form": ("immediate_recall", "请把另一个动词变成て形，再造一个短句。"),
}


class LearningOutcomeEngine:
    """Turns diagnosed errors into observable learning evidence."""

    def __init__(self, l2: Any):
        self.l2 = l2

    async def due_reviews(self, user_id: str, limit: int = 3) -> list[dict]:
        if not hasattr(self.l2, "get_due_learning_reviews"):
            return []
        return await self.l2.get_due_learning_reviews(user_id, limit=limit)

    async def record_errors(
        self, user_id: str, session_id: str, errors: Iterable[Dict[str, Any]], turn_index: int
    ) -> Optional[dict]:
        errors = list(errors)
        if not errors:
            return None
        # One focused probe per turn avoids turning speaking practice into a quiz.
        error = errors[0]
        item_type = error.get("type", "unknown")
        state = await self._get_state(user_id, item_type)
        mastery = max(0.02, float(state.get("mastery_probability", 0.35)) - 0.18)
        await self._save(
            user_id=user_id,
            item_type=item_type,
            mastery_probability=mastery,
            detected_delta=1,
            next_review_at=datetime.now(timezone.utc),
            last_evidence={"kind": "error", "session_id": session_id, "turn": turn_index, "excerpt": error.get("excerpt", "")},
        )
        if error.get("defer_probe"):
            return None
        return self._probe_for(item_type, stage="immediate_recall", error=error)

    async def score_probe(
        self,
        user_id: str,
        session_id: str,
        probe: Optional[Dict[str, Any]],
        current_errors: Iterable[Dict[str, Any]],
        turn_index: int,
    ) -> Optional[dict]:
        """Score only an explicitly issued probe; ordinary chat is not evidence."""
        if not probe:
            return None
        item_type = probe.get("item_type", "")
        if not item_type:
            return None
        # Absence of a detected error is not evidence of mastery. A greeting,
        # silence, or an answer that avoids the target form must not pass a
        # retrieval check merely because the rule detector found nothing.
        transcript = str(probe.get("response_text") or "")
        if transcript and not self._attempts_item(item_type, transcript):
            return {
                "stage": probe.get("stage", "immediate_recall"),
                "result": "not_attempted",
                "item_type": item_type,
                "next_probe": probe,
            }
        failed = any(e.get("type") == item_type for e in current_errors)
        stage = probe.get("stage", "immediate_recall")
        state = await self._get_state(user_id, item_type)
        old = float(state.get("mastery_probability", 0.35))
        now = datetime.now(timezone.utc)
        stability = max(0.25, float(state.get("stability_days", 1.0) or 1.0))
        difficulty = min(10.0, max(1.0, float(state.get("difficulty", 5.0) or 5.0)))
        if failed:
            mastery = max(0.02, old - 0.12)
            difficulty = min(10.0, difficulty + 0.6)
            stability = max(0.25, stability * 0.55)
            review_at = now + timedelta(minutes=10 if stage != "delayed_recall" else 60)
            await self._save(
                user_id, item_type, mastery, failed_delta=1,
                next_review_at=review_at, stability_days=stability,
                difficulty=difficulty, last_review_at=now, review_delta=1,
                last_evidence={"kind": stage, "result": "incorrect", "session_id": session_id, "turn": turn_index},
            )
            return {
                "stage": stage, "result": "incorrect", "item_type": item_type,
                "mastery_probability": round(mastery, 2), "difficulty_band": "scaffold",
                "next_review_at": review_at.isoformat(),
            }

        gain = {"immediate_recall": 0.18, "transfer": 0.25, "delayed_recall": 0.32}.get(stage, 0.15)
        mastery = min(0.98, old + gain)
        success_field = {"immediate_recall": "immediate_success", "transfer": "transfer_success", "delayed_recall": "delayed_success"}.get(stage, "immediate_success")
        next_stage = "transfer" if stage == "immediate_recall" else "delayed_recall"
        if stage == "immediate_recall":
            # Immediate repetition proves recall in the same context, not durable memory.
            review_at = now
            difficulty = max(1.0, difficulty - 0.1)
        else:
            # FSRS-inspired, evidence-driven interval: successful transfer/delayed
            # recall grows stability; difficult items grow more conservatively.
            stage_gain = 1.45 if stage == "transfer" else 1.9
            difficulty_factor = 1.15 - (difficulty - 1.0) * 0.045
            stability = min(365.0, max(1.0, stability * stage_gain * difficulty_factor + 0.5))
            difficulty = max(1.0, difficulty - (0.25 if stage == "delayed_recall" else 0.15))
            interval_days = self._interval_for_retention(stability, desired_retention=0.90)
            review_at = now + timedelta(days=interval_days)
        await self._save(
            user_id, item_type, mastery, **{success_field: 1},
            next_review_at=review_at, stability_days=stability,
            difficulty=difficulty, last_review_at=now, review_delta=1,
            last_evidence={"kind": stage, "result": "correct", "session_id": session_id, "turn": turn_index},
        )
        return {
            "stage": stage,
            "result": "correct",
            "item_type": item_type,
            "mastery_probability": round(mastery, 2),
            "difficulty_band": self._difficulty_band(mastery, difficulty),
            "next_review_at": review_at.isoformat(),
            "next_probe": self._probe_for(item_type, stage=next_stage) if next_stage == "transfer" else None,
        }

    @staticmethod
    def _attempts_item(item_type: str, transcript: str) -> bool:
        """Conservative proof that a response actually used the probed form.

        This is intentionally narrower than general grammar assessment. It is
        a guard against false-positive learning evidence, not a proficiency
        scorer.
        """
        import re

        text = (transcript or "").strip().lower()
        if not text:
            return False
        patterns = {
            "en_past_tense": r"\b(did|didn't|was|were|went|came|had|made|took|saw|said|bought|\w+ed)\b",
            "en_third_person_s": r"\b(he|she|it)\s+(?:\w+s|does|has|is)\b",
            "en_articles": r"\b(a|an|the)\s+\w+",
            "en_plurals": r"\b(?:\w+s|children|people|men|women|mice|feet|teeth)\b",
            "en_present_perfect": r"\b(?:have|has)\s+(?:\w+ed|been|gone|done|seen|eaten|written|made|taken)\b",
            "en_prepositions": r"\b(?:in|on|at|to|for|from|with|about|of|by|into|over|under)\b",
            "ja_wa_ga": r"[はが]",
            "ja_ni_de": r"[にで]",
            "ja_te_form": r"て",
        }
        pattern = patterns.get(item_type)
        return bool(re.search(pattern, text, re.IGNORECASE)) if pattern else len(text) >= 3

    @staticmethod
    def _interval_for_retention(stability_days: float, desired_retention: float = 0.90) -> int:
        """Translate stability into a bounded interval at the desired retention.

        Stability is defined at 90% recall. Keeping the formula explicit makes
        the schedule testable and replaceable by a full FSRS implementation.
        """
        retention = min(0.97, max(0.80, desired_retention))
        interval = stability_days * (math.log(retention) / math.log(0.90))
        return max(1, min(365, int(round(interval))))

    @staticmethod
    def _difficulty_band(mastery: float, difficulty: float) -> str:
        if mastery < 0.45 or difficulty >= 7.0:
            return "scaffold"
        if mastery >= 0.78 and difficulty <= 5.5:
            return "stretch"
        return "same_level"

    async def _get_state(self, user_id: str, item_type: str) -> dict:
        if hasattr(self.l2, "get_learning_state"):
            return await self.l2.get_learning_state(user_id, item_type) or {}
        return {}

    async def _save(self, user_id: str, item_type: str, mastery_probability: float, **changes: Any) -> None:
        if hasattr(self.l2, "upsert_learning_state"):
            await self.l2.upsert_learning_state(user_id, item_type, mastery_probability, **changes)

    def _probe_for(self, item_type: str, stage: str, error: Optional[Dict[str, Any]] = None) -> dict:
        default = "请不用看刚才的答案，换一个新句子再表达一次。"
        _, prompt = _PROBE_LIBRARY.get(item_type, ("immediate_recall", default))
        if stage == "transfer":
            prompt = "换到一个不同场景，不提示规则，自己用这个结构说一句话。"
        elif stage == "delayed_recall":
            prompt = "这是昨天练过的点。请在新场景中自然说一句，不需要解释规则。"
        return {
            "item_type": item_type,
            "stage": stage,
            "prompt": prompt,
            "error_excerpt": (error or {}).get("excerpt", ""),
        }
