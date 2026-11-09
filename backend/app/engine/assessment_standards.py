"""Versioned, evidence-gated learning assessment standards.

The language model may propose teaching language, but it does not get to
declare mastery. This module turns the durable evidence already recorded by
``LearningOutcomeEngine`` into deterministic stages and next-step gates.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, Iterable


SOP_VERSION = "learning-assessment-sop-v1.0"


class LearningAssessmentStandards:
    """Apply one inspectable rubric to language and speech evidence."""

    MASTERY_PROBABILITY = 0.80
    ADVANCE_WITH_REVIEW_PROBABILITY = 0.58
    MASTERY_ACCURACY = 0.80
    ADVANCE_ACCURACY = 0.65
    MIN_MASTERY_REVIEWS = 3
    MIN_SPEECH_SAMPLES = 3
    PRONUNCIATION_THRESHOLD = 75.0
    PROSODY_THRESHOLD = 70.0

    def __init__(self, l2: Any):
        self.l2 = l2

    @classmethod
    def evaluate_item(cls, state: Dict[str, Any] | None) -> Dict[str, Any]:
        """Return a deterministic stage and recommendation for one item."""
        state = dict(state or {})
        if not state or not state.get("item_type"):
            return {
                "standard_version": SOP_VERSION,
                "item_type": str(state.get("item_type") or ""),
                "stage": "unobserved",
                "decision": "insufficient_evidence",
                "new_content_eligible": False,
                "reason": "没有可验证的使用机会，不能判定掌握或不掌握",
                "missing_gates": ["structured_learning_evidence"],
                "evidence": {},
            }

        immediate = cls._integer(state.get("immediate_success"))
        transfer = cls._integer(state.get("transfer_success"))
        delayed = cls._integer(state.get("delayed_success"))
        failed = cls._integer(state.get("failed_attempts"))
        detected = cls._integer(state.get("detected_count"))
        reviews = cls._integer(state.get("review_count"))
        mastery = cls._number(state.get("mastery_probability"), 0.0)
        successes = immediate + transfer + delayed
        checks = successes + failed
        accuracy = successes / checks if checks else None

        mastery_gates = {
            "mastery_probability": mastery >= cls.MASTERY_PROBABILITY,
            "transfer_success": transfer >= 1,
            "delayed_success": delayed >= 1,
            "review_count": reviews >= cls.MIN_MASTERY_REVIEWS,
            "scored_accuracy": accuracy is not None and accuracy >= cls.MASTERY_ACCURACY,
        }
        mastered = all(mastery_gates.values())
        advance_with_review = (
            not mastered
            and mastery >= cls.ADVANCE_WITH_REVIEW_PROBABILITY
            and transfer >= 1
            and accuracy is not None
            and accuracy >= cls.ADVANCE_ACCURACY
        )

        if mastered:
            stage, decision = "stable_mastery", "advance"
            reason = "已通过独立迁移、延迟回忆和最低准确率门槛"
        elif advance_with_review:
            stage, decision = "transferred", "advance_with_review"
            reason = "已能换场景独立使用，可以学习新内容，同时保留间隔复习"
        elif delayed:
            stage, decision = "retained_but_unstable", "targeted_review"
            reason = "已有延迟回忆证据，但总体准确率或掌握置信度仍未达标"
        elif transfer:
            stage, decision = "early_transfer", "delayed_check"
            reason = "已经完成一次迁移，需要延迟复测后才能判定稳定掌握"
        elif immediate:
            stage, decision = "supported_recall", "transfer_check"
            reason = "只证明当前语境下能复现，下一步应换场景验证"
        elif detected or failed:
            stage, decision = "observed_gap", "scaffold_and_retry"
            reason = "已经观察到错误，但还没有有效的成功证据"
        else:
            stage, decision = "exposed", "collect_evidence"
            reason = "只有接触记录，没有足够的独立使用证据"

        return {
            "standard_version": SOP_VERSION,
            "item_type": str(state.get("item_type") or ""),
            "stage": stage,
            "decision": decision,
            "new_content_eligible": bool(mastered or advance_with_review),
            "reason": reason,
            "missing_gates": [name for name, passed in mastery_gates.items() if not passed],
            "evidence": {
                "detected_errors": detected,
                "immediate_success": immediate,
                "transfer_success": transfer,
                "delayed_success": delayed,
                "failed_attempts": failed,
                "review_count": reviews,
                "scored_accuracy": round(accuracy, 3) if accuracy is not None else None,
                "mastery_probability": round(mastery, 3),
            },
            "next_review_at": cls._text(state.get("next_review_at")),
        }

    async def build_decision_packet(self, user_id: str, limit: int = 6) -> Dict[str, Any]:
        """Build the small assessment packet that is safe to put in a prompt."""
        progress: Iterable[Dict[str, Any]] = []
        if hasattr(self.l2, "get_learning_progress"):
            progress = await self.l2.get_learning_progress(user_id, limit=max(limit * 3, 12))
        assessments = [self.evaluate_item(item) for item in progress]
        now = datetime.now(timezone.utc)

        def due(item: Dict[str, Any]) -> bool:
            value = item.get("next_review_at")
            if not value:
                return False
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed <= now
            except (TypeError, ValueError):
                return False

        priority = {
            "scaffold_and_retry": 0, "targeted_review": 1,
            "delayed_check": 2, "transfer_check": 3,
            "collect_evidence": 4, "advance_with_review": 5, "advance": 6,
        }
        assessments.sort(key=lambda item: (priority.get(item["decision"], 9), item["item_type"]))
        focus = [item for item in assessments if item["decision"] not in {"advance", "advance_with_review"}][:limit]
        do_not_repeat = [
            item["item_type"] for item in assessments
            if item["new_content_eligible"] and not due(item)
        ][:limit]
        review_now = [item["item_type"] for item in assessments if due(item)][:limit]
        return {
            "standard_version": SOP_VERSION,
            "rule": "只根据结构化证据判定；证据不足时不得宣称掌握；已可进阶项目不要机械重复",
            "focus_items": [self._compact_item(item) for item in focus],
            "review_now": review_now,
            "do_not_repeat": do_not_repeat,
            "speech": await self.evaluate_speech(user_id),
            "observed_item_count": len(assessments),
        }

    async def evaluate_speech(self, user_id: str) -> Dict[str, Any]:
        rows = []
        if hasattr(self.l2, "get_speech_assessments"):
            rows = await self.l2.get_speech_assessments(user_id, limit=20)
        return {
            "pronunciation": self._speech_metric(rows, "pronunciation_accuracy", self.PRONUNCIATION_THRESHOLD),
            "prosody": self._speech_metric(rows, "prosody_score", self.PROSODY_THRESHOLD),
            "fluency": self._fluency_metric(rows),
        }

    @classmethod
    def _speech_metric(cls, rows: Iterable[Dict[str, Any]], field: str, threshold: float) -> Dict[str, Any]:
        samples, calibrated = [], []
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            samples.append(cls._number(value, 0.0))
            source = cls._source_dict(row.get("assessment_source"))
            if source.get("rubric_version") and source.get("provider"):
                calibrated.append(cls._number(value, 0.0))
        if not samples:
            return {"decision": "insufficient_evidence", "sample_count": 0, "reason": "没有供应商评分"}
        if len(calibrated) < cls.MIN_SPEECH_SAMPLES:
            return {
                "decision": "insufficient_evidence",
                "sample_count": len(samples),
                "calibrated_sample_count": len(calibrated),
                "reason": "评分样本不足或缺少供应商与量表版本，不能判定达标",
            }
        recent = calibrated[: cls.MIN_SPEECH_SAMPLES]
        average = sum(recent) / len(recent)
        return {
            "decision": "meets_standard" if average >= threshold else "needs_practice",
            "sample_count": len(samples),
            "calibrated_sample_count": len(calibrated),
            "recent_average": round(average, 1),
            "threshold": threshold,
        }

    @classmethod
    def _fluency_metric(cls, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        observed = [row for row in rows if row.get("fluency_score") is not None]
        if not observed:
            return {"decision": "insufficient_evidence", "sample_count": 0, "reason": "没有可比较的语速与停顿样本"}
        comparable = []
        for row in observed:
            source = cls._source_dict(row.get("assessment_source"))
            if source.get("task_standard") and source.get("rubric_version"):
                comparable.append(cls._number(row.get("fluency_score"), 0.0))
        if len(comparable) < cls.MIN_SPEECH_SAMPLES:
            return {
                "decision": "tracked_not_benchmarked",
                "sample_count": len(observed),
                "comparable_sample_count": len(comparable),
                "reason": "已记录流利度趋势，但缺少至少3个同任务、同量表样本，不能判及格",
            }
        return {
            "decision": "trend_available",
            "sample_count": len(observed),
            "comparable_sample_count": len(comparable),
            "recent_average": round(sum(comparable[:3]) / 3, 1),
        }

    @staticmethod
    def _compact_item(item: Dict[str, Any]) -> Dict[str, Any]:
        evidence = item.get("evidence") or {}
        return {
            "item": item.get("item_type"), "stage": item.get("stage"),
            "next": item.get("decision"), "accuracy": evidence.get("scored_accuracy"),
            "reason": item.get("reason"),
        }

    @staticmethod
    def _source_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _number(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _integer(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
