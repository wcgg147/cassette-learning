"""Transparent speech assessment for the real-time tutor.

The client/provider supplies timing and optional pronunciation scores.  We do
not fabricate phoneme accuracy from plain text: unavailable dimensions are
returned as ``None`` with their source marked as unavailable.
"""
from __future__ import annotations

import re
from typing import Any, Dict


class SpeechAssessmentEngine:
    async def assess(self, user_id: str, session_id: str, payload: Dict[str, Any], store: Any) -> Dict[str, Any]:
        transcript = (payload.get("transcript") or "").strip()
        duration = float(payload.get("duration_seconds") or 0)
        raw_pauses = payload.get("pause_count")
        pauses = int(raw_pauses) if raw_pauses is not None else None
        context = payload.get("assessment_context") or {}
        target_lang = str(context.get("target_lang") or "").lower()
        english_words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", transcript)
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", transcript))
        if target_lang == "en":
            words = len(english_words)
            language_eligible = words >= 2 and len("".join(english_words)) >= max(4, chinese_chars)
        elif target_lang:
            # Other target languages need their own tokenizer/rubric.  Do not
            # silently score them with the English whitespace heuristic.
            words = 0
            language_eligible = False
        else:
            # Legacy clients omitted target_lang.  Infer conservatively so a
            # long Chinese process comment cannot become a bogus English WPM.
            words = len(english_words)
            language_eligible = words >= 2 and chinese_chars == 0
        words_per_minute = (
            round(words / duration * 60, 1)
            if duration > 0 and language_eligible else None
        )
        fluency = None
        fluency_source = "unavailable"
        if words_per_minute is not None:
            pace = max(0.0, 1.0 - abs(words_per_minute - 115) / 115)
            if pauses is None:
                fluency = round(pace * 100, 1)
                fluency_source = "speech_rate_and_asr_transcript"
            else:
                pause_factor = max(0.0, 1.0 - pauses / max(words, 1))
                fluency = round((pace * 0.65 + pause_factor * 0.35) * 100, 1)
                fluency_source = "timing_pauses_and_asr_transcript"

        provider = payload.get("provider_scores") or {}
        source = {
            "fluency_score": fluency_source,
            "pause_count": "client_or_provider" if pauses is not None else "unavailable",
            "pronunciation_accuracy": "provider" if provider.get("pronunciation_accuracy") is not None else "unavailable",
            "prosody_score": "provider" if provider.get("prosody_score") is not None else "unavailable",
            "language_gate": (
                f"eligible_{target_lang or 'inferred_english'}"
                if language_eligible else "insufficient_target_language_speech"
            ),
        }
        # A score without its provider, rubric version and comparable task is
        # observable data, but not a fair pass/fail standard.
        for key in ("provider", "rubric_version", "task_standard"):
            value = str(context.get(key) or "").strip()
            if value:
                source[key] = value[:120]
        result = {
            "transcript": transcript,
            "duration_seconds": duration or None,
            "word_count": words,
            "pause_count": pauses,
            "words_per_minute": words_per_minute,
            "fluency_score": fluency,
            "pronunciation_accuracy": provider.get("pronunciation_accuracy"),
            "prosody_score": provider.get("prosody_score"),
            "assessment_source": source,
        }
        if hasattr(store, "record_speech_assessment"):
            await store.record_speech_assessment(user_id, session_id, result)
        return result
