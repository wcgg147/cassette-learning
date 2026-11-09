"""Privacy-safe trace fields for one tutor turn.

The trace explains which controller route, memory route and response path were
used without persisting learner transcripts, prompts or model output.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


def _labels(values: Iterable[Any], limit: int = 8) -> list[str]:
    result: list[str] = []
    for value in values or []:
        label = str(value or "").strip()[:64]
        if label and label not in result:
            result.append(label)
        if len(result) >= limit:
            break
    return result


def build_turn_decision_trace(
    prepared: Dict[str, Any] | None,
    *,
    director_source: str = "",
    response_route: str = "",
    generation_path: str = "",
    validation_violations: int = 0,
    validation_violation_kinds: Iterable[Any] = (),
) -> Dict[str, Any]:
    """Return bounded, non-content observability for a prepared turn."""
    prepared = prepared or {}
    constraints = prepared.get("constraints") or {}
    contract = constraints.get("turn_contract") or {}
    decision = prepared.get("turn_controller") or contract.get("decision") or {}
    session_memory = constraints.get("session_memory") or {}
    memory_route = session_memory.get("route") or {}
    recalls = session_memory.get("recall") or []

    confidence = decision.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None

    return {
        "decision_version": str(decision.get("version") or "")[:48],
        "decision_frame": str(decision.get("primary_frame") or "")[:48],
        "turn_acts": _labels(decision.get("acts") or []),
        "response_sequence": _labels(decision.get("response_sequence") or []),
        "route_confidence": round(float(confidence), 3) if confidence is not None else None,
        "strong_reasoning_required": bool(decision.get("strong_reasoning_required", False)),
        "planning_route": str(decision.get("planning_route") or "fast_path")[:48],
        "routing_reasons": _labels(decision.get("routing_reasons") or []),
        "memory_route_mode": str(memory_route.get("mode") or "global_only")[:48],
        "memory_precision": str(memory_route.get("precision") or "semantic")[:32],
        "memory_recall_count": min(len(recalls), 16),
        "director_source": str(director_source or "")[:64],
        "response_route": str(response_route or "")[:48],
        "generation_path": str(generation_path or "")[:64],
        "validation_violation_count": max(0, min(int(validation_violations or 0), 32)),
        # 只留类别标签（违规文本的前半句），不含用户内容——用于回答
        # "86% 的回复被修，究竟是被哪条规则修的"。
        "validation_violation_kinds": _labels(validation_violation_kinds, 4),
    }
