"""Privacy-safe normalization for turn/session metering events.

The contract records observable learning and provider usage facts.  It never
stores transcripts, raw audio, credentials, prices, or inferred billing data.
"""
from __future__ import annotations

from typing import Any, Dict


SCHEMA_VERSION = 1
METERING_FIELDS = {
    "schema_version", "user_id", "session_id", "turn_id", "event_scope",
    "request_status", "provider", "model", "request_count", "retry_count",
    "input_tokens", "output_tokens", "total_tokens", "token_usage_status",
    "asr_input_duration_ms", "asr_usage_status", "tts_output_duration_ms",
    "tts_character_count", "tts_usage_status", "canceled", "failure_code",
    "structured_probe_issued", "structured_probe_result",
    "learning_evidence_valid", "metadata",
}
METADATA_FIELDS = {
    "turn_index", "session_turn_count", "session_entry_mode", "source",
    "first_text_latency_ms", "first_audio_latency_ms", "response_duration_ms",
    "first_user_speech_latency_ms", "asr_duration_ms", "canceled_turns",
    "failed_turns", "successful_turns", "metered_turns",
    "generation_path", "backend_total_latency_ms", "director_latency_ms",
    "director_source", "director_failure_code", "web_search_used",
    "web_search_result_count", "web_search_provider",
    "decision_version", "decision_frame", "turn_acts", "response_sequence",
    "route_confidence", "strong_reasoning_required", "planning_route",
    "routing_reasons", "memory_route_mode", "memory_precision",
    "memory_recall_count", "response_route", "validation_violation_count",
    # plans/1017-02 §8.50：记"这一轮被哪条规则修过"（只存类别名，不含原话）。
    # **必须登记**——本白名单的语义是"不登记就静默丢弃"。
    "validation_violation_kinds",
    "cached_tokens",
    # plans/0916-22 修复0：录音完整性台账（免迁移，进 metadata）。
    "head_trim_ms", "tail_trim_ms", "end_by_silence", "end_mid_speech",
    "asr_segments_merged", "transcript_chars", "transcript_words",
    "suspect_truncation",
    # plans/0916-14 G1 行为原则：事后标记 + 开口指标（嵌套 dict，免迁移）。
    "g1",
    # plans/0919-03 §4.1–4.4 情绪层：轮级情绪状态 + 事后标记（嵌套 dict，免迁移）。
    # **必须登记**——本白名单的语义是"不登记就静默丢弃"，漏登记会让标记
    # 一直在算、一直落不了库（0919-01 §0.2 记过同一个坑）。
    "emotion",
    # plans/0919-02 §4.7 能力画像（盘内趋势）：练习盘逐轮的五维读数
    # （嵌套 dict，免迁移）。**只记 ok/weak 两态，不含原话**；只有练习盘会写，
    # 其它盘不发送该键 ⇒ 零增量。
    "expr_arg",
}
VALID_SCOPES = {"turn", "session"}
VALID_REQUEST_STATUSES = {
    "unavailable", "pending", "success", "failed", "canceled",
    "completed", "completed_with_failures",
}
VALID_USAGE_STATUSES = {"unavailable", "observed", "reported", "partial"}
VALID_PROBE_RESULTS = {"not_issued", "pending", "correct", "incorrect", "not_attempted"}


def _bounded_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, min(int(value), 2_147_483_647))


def normalize_metering_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return only contract fields, with unavailable provider data explicit."""
    source = {key: value for key, value in (event or {}).items() if key in METERING_FIELDS}
    normalized: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "user_id": str(source.get("user_id") or "")[:128],
        "session_id": str(source.get("session_id") or "")[:128],
        "turn_id": str(source.get("turn_id") or "")[:128],
        "event_scope": source.get("event_scope") if source.get("event_scope") in VALID_SCOPES else "turn",
        "request_status": source.get("request_status") if source.get("request_status") in VALID_REQUEST_STATUSES else "unavailable",
        "provider": str(source.get("provider") or "")[:64],
        "model": str(source.get("model") or "")[:128],
        "request_count": _bounded_int(source.get("request_count")) or 0,
        "retry_count": _bounded_int(source.get("retry_count")) or 0,
        "input_tokens": _bounded_int(source.get("input_tokens")),
        "output_tokens": _bounded_int(source.get("output_tokens")),
        "total_tokens": _bounded_int(source.get("total_tokens")),
        "token_usage_status": source.get("token_usage_status") if source.get("token_usage_status") in VALID_USAGE_STATUSES else "unavailable",
        "asr_input_duration_ms": _bounded_int(source.get("asr_input_duration_ms")),
        "asr_usage_status": source.get("asr_usage_status") if source.get("asr_usage_status") in VALID_USAGE_STATUSES else "unavailable",
        "tts_output_duration_ms": _bounded_int(source.get("tts_output_duration_ms")),
        "tts_character_count": _bounded_int(source.get("tts_character_count")),
        "tts_usage_status": source.get("tts_usage_status") if source.get("tts_usage_status") in VALID_USAGE_STATUSES else "unavailable",
        "canceled": bool(source.get("canceled", False)),
        "failure_code": str(source.get("failure_code") or "")[:96],
        "structured_probe_issued": bool(source.get("structured_probe_issued", False)),
        "structured_probe_result": source.get("structured_probe_result") if source.get("structured_probe_result") in VALID_PROBE_RESULTS else "not_issued",
        "learning_evidence_valid": bool(source.get("learning_evidence_valid", False)),
    }
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    normalized["metadata"] = {
        key: metadata[key] for key in METADATA_FIELDS if key in metadata
    }
    if normalized["token_usage_status"] == "reported" and all(
        normalized[key] is None for key in ("input_tokens", "output_tokens", "total_tokens")
    ):
        normalized["token_usage_status"] = "unavailable"
    if normalized["asr_input_duration_ms"] is None:
        normalized["asr_usage_status"] = "unavailable"
    if normalized["tts_output_duration_ms"] is None and normalized["tts_character_count"] is None:
        normalized["tts_usage_status"] = "unavailable"
    return normalized


def merge_metering_event(existing: Dict[str, Any] | None, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge cumulative observations without letting success erase cancel/failure."""
    current = normalize_metering_event(existing or {})
    incoming = normalize_metering_event(updates)
    merged = dict(current)
    for key in (
        "schema_version", "user_id", "session_id", "turn_id", "event_scope",
        "provider", "model", "input_tokens", "output_tokens", "total_tokens",
        "asr_input_duration_ms", "tts_output_duration_ms", "tts_character_count",
        "failure_code", "structured_probe_result",
    ):
        value = incoming.get(key)
        if value not in (None, ""):
            merged[key] = value
    merged["request_count"] = max(current.get("request_count", 0), incoming.get("request_count", 0))
    merged["retry_count"] = max(current.get("retry_count", 0), incoming.get("retry_count", 0))
    merged["canceled"] = bool(current.get("canceled") or incoming.get("canceled"))
    if "structured_probe_issued" in updates:
        merged["structured_probe_issued"] = incoming["structured_probe_issued"]
    if "learning_evidence_valid" in updates:
        merged["learning_evidence_valid"] = incoming["learning_evidence_valid"]
    for key in ("token_usage_status", "asr_usage_status", "tts_usage_status"):
        rank = {"unavailable": 0, "observed": 1, "partial": 2, "reported": 3}
        merged[key] = max((current.get(key, "unavailable"), incoming.get(key, "unavailable")), key=lambda value: rank[value])
    status_rank = {
        "unavailable": 0, "pending": 1, "success": 2, "completed": 3,
        "completed_with_failures": 4, "failed": 5, "canceled": 6,
    }
    merged["request_status"] = max(
        (current.get("request_status", "unavailable"), incoming.get("request_status", "unavailable")),
        key=lambda value: status_rank[value],
    )
    if merged["canceled"] and merged.get("event_scope") == "turn":
        merged["request_status"] = "canceled"
    merged["metadata"] = {**current.get("metadata", {}), **incoming.get("metadata", {})}
    return normalize_metering_event(merged)
