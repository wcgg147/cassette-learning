"""
Learning Engine - Chinese native speakers learning Japanese + English.
Detects language-specific errors and writes to L2 memory.
"""

import uuid
import re
from typing import Optional
from memory.provider import MemoryEntry as MemoryFact, maybe_await

try:
    from ..memory.l3_archive import L3Archive
except ImportError:
    from memory.l3_archive import L3Archive


class LearningEngine:

    def __init__(self, l1, l2, l3: L3Archive):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self._target_lang = None  # "ja" or "en", set per session

    def set_target_language(self, lang: str):
        """Set target language for this session: 'ja' or 'en'."""
        self._target_lang = lang

    async def analyze_utterance(
        self, user_id: str, transcript: str, session_id: str,
        target_lang: Optional[str] = None,
    ) -> list[dict]:
        """Detect errors in user utterance, write to L2.

        ``target_lang`` is passed per request so concurrent English and
        Japanese sessions cannot overwrite one another through the legacy
        mutable ``_target_lang`` fallback.
        """
        language = target_lang or self._target_lang or "en"
        if language == "ja":
            errors = self._detect_japanese_errors(transcript)
        else:
            errors = self._detect_english_errors(transcript)
        errors = [self._enrich_diagnostic(error, transcript) for error in errors]

        # Write errors to L2 using backend-specific methods
        for err in errors:
            await self.persist_diagnostic(user_id, err, transcript)
        return errors

    async def persist_diagnostic(
        self, user_id: str, diagnostic: dict, transcript: str = ""
    ) -> None:
        """Persist a rule- or director-backed language diagnostic once.

        Open-ended naturalness judgements come from the conversation director,
        while textbook patterns come from the local detector.  Both must reach
        the same learner evidence store; otherwise the tutor can correct a turn
        audibly while the long-term assessment remains empty.
        """
        try:
            err_type = diagnostic.get("type", "unknown")
            excerpt = diagnostic.get("excerpt", transcript[:200])
            if hasattr(self.l2, "add_error"):
                await maybe_await(self.l2.add_error(user_id, err_type, excerpt))
            else:
                await self.l2.add_fact(user_id, MemoryFact(
                    entry_id=str(uuid.uuid4()),
                    user_id=user_id,
                    entry_type="error",
                    content=f"{err_type}: {excerpt}"[:200],
                    weight=1.0,
                ))
        except Exception:
            pass

    def _enrich_diagnostic(self, error: dict, transcript: str) -> dict:
        """Expose a stable diagnostic contract to the client and outcome loop.

        Rule detection remains the low-latency first pass.  Each finding carries
        its confidence and evidence source so a later LLM/speech assessor can
        augment it without pretending that every judgement is certain.
        """
        error_type = error.get("type", "unknown")
        language = "ja" if error_type.startswith("ja_") else "en"
        category = "grammar"
        if "cn_" in error_type:
            category = "l1_transfer"
        elif "keigo" in error_type:
            category = "pragmatics"
        correction = self._suggest_correction(error_type, error.get("excerpt", ""))
        return {
            **error,
            "diagnostic_id": str(uuid.uuid4()),
            "error_type": error_type,
            "language": language,
            "category": category,
            "span": error.get("excerpt", transcript[:100]),
            "correction": correction,
            "confidence": 0.92 if language == "ja" else 0.88,
            "evidence_source": "rule_detector",
            "requires_confirmation": False,
        }

    @staticmethod
    def _suggest_correction(error_type: str, excerpt: str) -> str:
        replacements = {
            "didn't went": "didn't go", "didn't came": "didn't come",
            "did went": "did go", "did came": "did come", "goed": "went",
            "buyed": "bought", "writed": "wrote", "a apple": "an apple",
            "a hour": "an hour", "a umbrella": "an umbrella", "a orange": "an orange",
            "childs": "children", "mouses": "mice", "sheeps": "sheep",
            "deers": "deer", "two person": "two people", "have went": "have gone",
            "has went": "has gone", "have ate": "have eaten", "has ate": "has eaten",
            "i'm not used the": "I didn't use the", "i am not used the": "I didn't use the",
            "i not used the": "I didn't use the", "i'm not use the": "I didn't use the",
            "i am not use the": "I didn't use the",
            "married with": "married to", "depend of": "depend on",
            "listen music": "listen to music", "discuss about": "discuss",
            "my card can't pay": "My card isn't working.",
            "my card cannot pay": "My card isn't working.",
        }
        lower = excerpt.lower()
        for wrong, right in replacements.items():
            if wrong in lower:
                return right
        return "请根据本轮教学反馈自行改写。"

    # ── Japanese error detection ─────────────────────────

    def _detect_japanese_errors(self, text: str) -> list[dict]:
        """Detect common Japanese errors for Chinese learners."""
        errors = []

        # Particle confusion (は/が, に/で, を etc.)
        patterns_ja = {
            "wa_ga_confusion": [
                (r"私が(.{1,20})です", "wa_ga", "可能用が代替了は。主语是主题时用「は」"),
                (r"(.{2,5})が(.{2,10})が", "wa_ga", "一个句子中出现两个が"),
            ],
            "ni_de_confusion": [
                (r"学校に勉強", "ni_de", "动作发生场所用「で」，不是「に」"),
                (r"東京で行く", "ni_de", "移动目的地用「に」，不是「で」"),
            ],
            "te_form": [
                (r"買いて", "te_form", "五段动词て形变化错误，应为「買って」"),
                (r"書って", "te_form", "五段动词て形变化错误，应为「書いて」"),
                (r"行きて", "te_form", "五段动词连用形错误，应为「行って」"),
            ],
            "ta_form": [
                (r"書いたです", "ta_form", "た形+です不自然，应为「書きました」"),
            ],
            "keigo": [
                (r"先生が言う", "keigo", "对老师应使用敬语「おっしゃる」"),
                (r"社長に会う", "keigo", "对上司应使用谦让语「お目にかかる」"),
            ],
            "transitive_intransitive": [
                (r"ドアを開く", "trans_intrans", "自动词「開く」应用が，或改用他动词「開ける」"),
            ],
            "adjective_conjugation": [
                (r"美味いです", "adj_conj", "い形容詞直接+です即可，不用变"),
            ],
            "chinese_word_misuse": [
                (r"爱人", "cn_misuse", "日语的「愛人」是情人的意思，不是配偶"),
                (r"手紙", "cn_misuse", "日语的「手紙」是信，不是厕纸"),
                (r"勉強", "cn_misuse", "日语的「勉強」是学习，不是勉强"),
                (r"大丈夫", "cn_misuse", "日语的「大丈夫」是没问题，不是男子汉"),
            ],
        }

        for error_type, patterns in patterns_ja.items():
            for pattern, tag, hint in patterns:
                if re.search(pattern, text):
                    errors.append({
                        "type": f"ja_{tag}",
                        "excerpt": text[:100],
                        "hint": hint,
                    })

        return errors

    # ── English error detection ──────────────────────────

    def _detect_english_errors(self, text: str) -> list[dict]:
        """Detect common English errors for Chinese learners."""
        errors = []
        lower = " " + text.lower() + " "

        patterns_en = {
            "past_tense": ["didn't went", "didn't came", "did went", "did came", "goed", "buyed", "writed"],
            "third_person_s": ["he go ", "she go ", "he have ", "she have ", "he don't", "she don't"],
            "articles": ["a apple", "a hour", "a umbrella", "a orange"],
            "plurals": ["childs", "mouses", "sheeps", "deers", "two person"],
            "present_perfect": ["have went", "has went", "have ate", "has ate"],
            "negative_past": [
                "i'm not used the", "i am not used the", "i not used the",
                "i'm not use the", "i am not use the",
            ],
            "prepositions": ["married with", "depend of", "listen music", "discuss about"],
            # A card is the payment instrument; it does not itself "pay".
            # This high-confidence collocation is common in live travel/shop
            # practice and is safe to detect without an LLM round-trip.
            "payment_collocation": ["my card can't pay", "my card cannot pay"],
            "cn_english": ["people mountain people sea", "good good study", "no zuo no die"],
        }

        for error_type, phrases in patterns_en.items():
            for phrase in phrases:
                if phrase in lower:
                    idx = lower.index(phrase)
                    excerpt = text[max(0, idx - 20):idx + len(phrase) + 20].strip()
                    errors.append({"type": f"en_{error_type}", "excerpt": excerpt})
                    break

        # 高频结构错误（固定短语清单盖不住）：进行时漏 be 动词。
        # 真机实录 "我 i i watching a short video in bilibili" 未被检出，
        # 纠错链路全程空转。代词（可带一次口误重复）后直接跟 V-ing 即命中；
        # "i am watching" / "i enjoy watching" 均不匹配。
        # 排除 -ing 结尾的普通名词/代词，防 "i something" 类误报。
        _ING_NON_VERB = {
            "something", "anything", "everything", "nothing", "thing", "things",
            "king", "during", "morning", "evening", "offing", "king",
        }
        for m in re.finditer(
            r"\b(i|you|we|they|he|she|it)\s+(?:\1\s+)?([a-z]+ing)\b", lower
        ):
            if m.group(2) in _ING_NON_VERB:
                continue
            idx = m.start()
            excerpt = text[max(0, idx - 20):idx + len(m.group(0)) + 20].strip()
            errors.append({
                "type": "en_missing_be_progressive",
                "excerpt": excerpt,
            })
            break

        return errors

    # ── Mark mastered ────────────────────────────────────

    async def mark_mastered(self, user_id: str, expression: str, confidence: float = 0.8):
        """Mark an expression as mastered."""
        try:
            if hasattr(self.l2, "add_mastered"):
                self.l2.add_mastered(user_id, expression, confidence)
        except Exception:
            pass

    async def archive_dialogue(
        self, user_id: str, session_id: str,
        transcript: list[dict], metadata: Optional[dict] = None,
    ) -> str:
        return await self.l3.archive_session(user_id, session_id, transcript, metadata)

    async def get_frozen_profile(self, user_id: str) -> dict:
        return await self.l1.get_frozen_snapshot(user_id)

    async def search_memory(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        return await self.l2.search_history(user_id, query, top_k)
