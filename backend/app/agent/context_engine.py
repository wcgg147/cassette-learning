"""Provider-neutral context budgeting and deterministic episode compaction.

The context engine deliberately does not call an LLM.  It separates three
concerns so a tokenizer or semantic compactor can be replaced later without
changing TutorAgent:

* TokenMeter estimates the active prompt budget.
* ContextCompactor converts older events into a bounded, typed checkpoint.
* ContextEngine decides when compaction is needed and retains a verbatim tail.

Raw events remain owned by the caller.  A checkpoint is supporting context,
never a synthetic user/assistant message and never evidence of mastery.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence


class TokenMeter(Protocol):
    """Replaceable token-counting boundary."""

    def count_text(self, text: str) -> int:
        ...

    def count_messages(self, messages: Sequence[Mapping[str, Any]]) -> int:
        ...


class ContextCompactor(Protocol):
    """Replaceable checkpoint-building boundary."""

    def compact(
        self,
        events: Sequence[Mapping[str, Any]],
        previous_checkpoint: Optional[Mapping[str, Any]],
        state_snapshot: Optional[Mapping[str, Any]],
        *,
        stream_id: str,
        covered_until: int,
    ) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class ContextPolicy:
    enabled: bool = True
    active_history_token_budget: int = 3200
    min_messages_before_compaction: int = 24
    retain_recent_messages: int = 8


@dataclass(frozen=True)
class CompactionResult:
    checkpoint: Dict[str, Any]
    active_messages: List[Dict[str, Any]]
    compacted: bool
    stats: Dict[str, Any]


class ApproxTokenMeter:
    """Fast, dependency-free estimate suitable for provider-neutral gating.

    CJK characters typically consume more tokens per character than Latin
    prose.  This estimate intentionally errs slightly high; provider-native
    tokenizers can later implement the same interface.
    """

    _cjk_pattern = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

    def count_text(self, text: str) -> int:
        value = str(text or "")
        cjk_count = len(self._cjk_pattern.findall(value))
        other_count = max(0, len(value) - cjk_count)
        return max(1, math.ceil(cjk_count / 1.5) + math.ceil(other_count / 4))

    def count_messages(self, messages: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            4 + self.count_text(str(message.get("content") or ""))
            for message in messages
            if message.get("role") in {"user", "assistant"}
        )


def _bounded_text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _bounded_list(value: Any, *, limit: int = 8, chars: int = 180) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = _bounded_text(item, chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


class StructuredContextCompactor:
    """Build a bounded factual checkpoint from already-recorded events.

    Text excerpts are labelled as excerpts.  Structured teaching state is kept
    separately and explicitly cannot establish proficiency or mastery.
    """

    TIMELINE_LIMIT = 14
    LEARNER_QUOTE_LIMIT = 8

    def compact(
        self,
        events: Sequence[Mapping[str, Any]],
        previous_checkpoint: Optional[Mapping[str, Any]],
        state_snapshot: Optional[Mapping[str, Any]],
        *,
        stream_id: str,
        covered_until: int,
    ) -> Dict[str, Any]:
        previous = dict(previous_checkpoint or {})
        previous_timeline = list(previous.get("timeline") or [])
        previous_quotes = list(previous.get("learner_quotes") or [])

        timeline = previous_timeline + [
            {
                "stream_id": stream_id,
                "event_index": int(event.get("event_index", index)),
                "role": str(event.get("role") or ""),
                "excerpt": _bounded_text(event.get("content"), 260),
                "evidence_type": "verbatim_excerpt",
            }
            for index, event in enumerate(events)
            if event.get("role") in {"user", "assistant"}
            and _bounded_text(event.get("content"), 260)
        ]
        learner_quotes = previous_quotes + [
            {
                "stream_id": stream_id,
                "event_index": int(event.get("event_index", index)),
                "quote": _bounded_text(event.get("content"), 220),
                "evidence_type": "verbatim_excerpt",
            }
            for index, event in enumerate(events)
            if event.get("role") == "user" and _bounded_text(event.get("content"), 220)
        ]

        snapshot = dict(state_snapshot or {})
        ledger = dict(snapshot.get("scene_ledger") or {})
        learning_state = dict(snapshot.get("learning_state") or {})
        learning_keys = (
            "stage", "mission", "communicative_goal", "target_expression",
            "support_level", "attempts", "successful_attempts", "last_move",
            "scenario", "scenario_step", "roleplay_turn",
        )
        streams = list(previous.get("source", {}).get("streams") or [])
        stream_record = {
            "stream_id": stream_id,
            "covered_until": covered_until,
            "covered_messages": len(events),
        }
        streams = [item for item in streams if item.get("stream_id") != stream_id]
        streams.append(stream_record)

        return {
            "schema_version": 1,
            "kind": "episode_checkpoint",
            "trust": {
                "role": "historical_supporting_context",
                "not_instructions": True,
                "not_mastery_evidence": True,
            },
            "source": {
                "current_stream_id": stream_id,
                "covered_until": covered_until,
                "streams": streams[-8:],
            },
            "scene": {
                "brief": _bounded_text(snapshot.get("scene_brief"), 320),
                "learner_role": _bounded_text(snapshot.get("learner_role"), 80),
                "tutor_role": _bounded_text(snapshot.get("tutor_role"), 80),
                "current_phase": _bounded_text(
                    ledger.get("current_phase") or snapshot.get("scenario_step"), 100
                ),
            },
            "known_facts": _bounded_list(ledger.get("known_facts")),
            "provided_information": _bounded_list(ledger.get("provided_information")),
            "completed_actions": _bounded_list(ledger.get("completed_actions")),
            "open_items": _bounded_list(ledger.get("open_items")),
            "learning_state": {
                **{
                    key: learning_state[key]
                    for key in learning_keys
                    if learning_state.get(key) not in (None, "", [])
                },
                "status_source": "structured_session_state_not_mastery_claim",
            },
            "timeline": timeline[-self.TIMELINE_LIMIT:],
            "learner_quotes": learner_quotes[-self.LEARNER_QUOTE_LIMIT:],
        }


class ContextEngine:
    """Automatic compaction coordinator with overflow-retry support."""

    def __init__(
        self,
        policy: Optional[ContextPolicy] = None,
        token_meter: Optional[TokenMeter] = None,
        compactor: Optional[ContextCompactor] = None,
    ) -> None:
        self.policy = policy or ContextPolicy()
        self.token_meter = token_meter or ApproxTokenMeter()
        self.compactor = compactor or StructuredContextCompactor()

    @staticmethod
    def _normalize_events(event_log: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, event in enumerate(event_log):
            role = str(event.get("role") or "")
            content = str(event.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized.append({
                "role": role,
                "content": content,
                "event_index": int(event.get("event_index", index)),
            })
        return normalized

    @staticmethod
    def _tail_start(events: Sequence[Mapping[str, Any]], start: int, retain: int) -> int:
        tail_start = max(start, len(events) - max(2, retain))
        # Do not create an orphan assistant reply at the beginning of the tail.
        while tail_start > start and events[tail_start].get("role") != "user":
            tail_start -= 1
        return tail_start

    def maybe_compact(
        self,
        event_log: Sequence[Mapping[str, Any]],
        checkpoint: Optional[Mapping[str, Any]] = None,
        state_snapshot: Optional[Mapping[str, Any]] = None,
        *,
        stream_id: str,
        force: bool = False,
        reason: str = "threshold",
    ) -> CompactionResult:
        events = self._normalize_events(event_log)
        previous = dict(checkpoint or {})
        source = dict(previous.get("source") or {})
        same_stream = source.get("current_stream_id") == stream_id
        covered_until = int(source.get("covered_until", -1)) if same_stream else -1
        active_start = min(len(events), max(0, covered_until + 1))
        active = events[active_start:]
        active_tokens = self.token_meter.count_messages(active)
        threshold_reached = (
            len(active) >= self.policy.min_messages_before_compaction
            or active_tokens > self.policy.active_history_token_budget
        )

        if not self.policy.enabled or (not force and not threshold_reached):
            return CompactionResult(
                checkpoint=previous,
                active_messages=active,
                compacted=False,
                stats={
                    "reason": reason,
                    "active_messages": len(active),
                    "active_tokens_estimate": active_tokens,
                    "compacted_messages": 0,
                },
            )

        tail_start = self._tail_start(
            events, active_start, self.policy.retain_recent_messages
        )
        if tail_start <= active_start:
            return CompactionResult(
                checkpoint=previous,
                active_messages=active,
                compacted=False,
                stats={
                    "reason": "nothing_safe_to_compact",
                    "active_messages": len(active),
                    "active_tokens_estimate": active_tokens,
                    "compacted_messages": 0,
                },
            )

        compacted_events = events[active_start:tail_start]
        new_checkpoint = self.compactor.compact(
            compacted_events,
            previous,
            state_snapshot,
            stream_id=stream_id,
            covered_until=tail_start - 1,
        )
        new_active = events[tail_start:]
        return CompactionResult(
            checkpoint=new_checkpoint,
            active_messages=new_active,
            compacted=True,
            stats={
                "reason": reason,
                "active_messages_before": len(active),
                "active_messages_after": len(new_active),
                "active_tokens_before_estimate": active_tokens,
                "active_tokens_after_estimate": self.token_meter.count_messages(new_active),
                "compacted_messages": len(compacted_events),
                "covered_until": tail_start - 1,
            },
        )

    def recover_from_overflow(
        self,
        event_log: Sequence[Mapping[str, Any]],
        checkpoint: Optional[Mapping[str, Any]],
        state_snapshot: Optional[Mapping[str, Any]],
        *,
        stream_id: str,
    ) -> CompactionResult:
        """Force one safe compaction after a provider context overflow."""
        return self.maybe_compact(
            event_log,
            checkpoint,
            state_snapshot,
            stream_id=stream_id,
            force=True,
            reason="provider_overflow_retry",
        )
