"""Asynchronous, evidence-gated cross-session memory curation.

The curator is intentionally outside the realtime response path.  It proposes
compact candidates from the learner's own words; deterministic code validates
the evidence and decides which explicit, low-risk items may enter relationship
memory.  Assessment, CEFR, mastery and pronunciation remain owned by their
separate evidence-based engines.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List

logger = logging.getLogger("memory_curator")


class MemoryCuratorAgent:
    VERSION = "memory-curator-v2"
    ALLOWED_ASSISTANT_PERSONA_KINDS = {
        "experience", "preference", "ability", "trait", "language_environment",
    }
    ALLOWED_KINDS = {
        "learning_preference",
        "next_session_intent",
        "self_reported_weakness",
        "avoid_pattern",
        "preferred_tone",
        "engagement_topic",
        "open_loop",
    }
    AUTO_ACCEPT_KINDS = {
        "learning_preference",
        "next_session_intent",
        "self_reported_weakness",
        "avoid_pattern",
        "open_loop",
        "engagement_topic",
    }
    FUTURE_MARKERS = (
        "下次", "下一次", "下回", "以后", "明天", "下周", "下个月", "准备", "打算",
        "要去", "next time", "next session", "tomorrow", "next week", "plan to",
        "going to", "when we practice again",
    )
    PREFERENCE_MARKERS = (
        "更想练", "想多练", "想重点练", "希望练", "以后继续练", "以后想练", "我喜欢", "我更喜欢",
        "want to practice", "would like to practice", "want to focus on", "prefer",
    )
    WEAKNESS_MARKERS = (
        "薄弱", "英语不好", "口语不好", "语法不好", "发音不好", "说不来",
        "不会说", "不会讲", "老是说错", "总说错", "经常说错", "容易说错",
        "struggle with", "weak at", "keep getting", "often get wrong",
        "my english is weak", "past tense", "时态",
    )
    AVOID_MARKERS = (
        "不要", "别再", "不想", "不需要", "太多", "讨厌", "don't", "do not", "stop",
    )

    def __init__(self, *, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    async def curate(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        existing_memory: Dict[str, Any] | None = None,
        handoff: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        bounded = self._bounded_messages(messages)
        user_texts = [m["content"] for m in bounded if m["role"] == "user"]
        base = {
            "version": self.VERSION,
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "status": "disabled",
            "candidates": [],
            "accepted_items": [],
            "rejected_count": 0,
            "assistant_persona_facts": [],
            "assistant_persona_rejected_count": 0,
            "latency_ms": 0,
        }
        if not user_texts:
            base["status"] = "no_user_evidence"
            return base

        deterministic = self.deterministic_candidates(user_texts)
        deterministic_validated, deterministic_accepted, deterministic_rejected = (
            self.validate_candidates(deterministic, user_texts)
        )

        try:
            from config import (
                MEMORY_CURATOR_API_KEY,
                MEMORY_CURATOR_BASE_URL,
                MEMORY_CURATOR_ENABLED,
                MEMORY_CURATOR_MAX_TOKENS,
                MEMORY_CURATOR_MODEL,
                MEMORY_CURATOR_TIMEOUT,
            )
            from llm_client import robust_llm_client
        except ImportError:
            from ..config import (
                MEMORY_CURATOR_API_KEY,
                MEMORY_CURATOR_BASE_URL,
                MEMORY_CURATOR_ENABLED,
                MEMORY_CURATOR_MAX_TOKENS,
                MEMORY_CURATOR_MODEL,
                MEMORY_CURATOR_TIMEOUT,
            )
            from ..llm_client import robust_llm_client

        if not MEMORY_CURATOR_ENABLED or not MEMORY_CURATOR_API_KEY:
            base.update({
                "status": "fallback_completed" if deterministic_validated else "disabled",
                "candidates": deterministic_validated,
                "accepted_items": deterministic_accepted,
                "rejected_count": deterministic_rejected,
                "fallback_reason": "curator_disabled",
            })
            base["latency_ms"] = int((time.perf_counter() - started) * 1000)
            return base

        client = self._llm_client or robust_llm_client
        handoff_data = handoff or {}
        curate_alice_persona = bool(handoff_data.get("_curate_alice_persona"))
        persona_output_contract = (
            ',"assistant_persona_facts":[{"kind":"experience|preference|ability|trait|language_environment",'
            '"facet_key":"stable-topic-slug","evidence_quote":"exact assistant quote",'
            '"event_index":integer}]'
            if curate_alice_persona else ""
        )
        prompt = (
            "你是语言学习产品的后台记忆整理器，不参与回复用户。用户记忆候选只依据 role=user；"
            "不得把 assistant 的话当作用户事实。"
            "输出严格 JSON：{\"candidates\":[{\"kind\":...,\"value\":...,\"evidence_quote\":...,"
            "\"confidence\":0到1,\"durability_days\":整数,\"reason\":...}],\"episode_summary\":...,"
            "\"contradictions\":[...],\"handoff_patch\":{}"
            + persona_output_contract + "}。"
            "kind 只能是 learning_preference/next_session_intent/self_reported_weakness/avoid_pattern/"
            "preferred_tone/engagement_topic/open_loop。evidence_quote 必须逐字来自对应 user 消息文本。"
            "该文本可能是语音转写或精修后的显示文本；不得声称它必然是用户音频逐字原话。"
            "engagement_topic 记录用户的兴趣与持续投入：既包括明确表达的喜欢/不喜欢/持续兴趣，"
            "也包括行为信号——反复做、主动投入时间金钱、惯例性提及的事（如每月多次爬山、熬夜看某类纪录片、"
            "长期在练的技能）。value 要保留具体对象与用户给出的理由，不得把单次反应泛化成普遍人格偏好；"
            "证据必须逐字来自用户消息。"
            "不要把 assistant 的猜测当成用户事实；不要判断或升级 CEFR、掌握度、发音、语调、语速；"
            "不要保存一次性的场景台词、姓名房号票号等敏感或短期信息；没有可靠候选就返回空数组。"
        )
        if curate_alice_persona:
            prompt += (
                "另整理 Alice 的自述，但只从 role=assistant 的消息提取，并写入 assistant_persona_facts；"
                "只收 Alice 对自己明确说出的经历、偏好、能力、长期特质或语言环境。"
                "一次明确的个人经历可以立即作为本用户关系中的叙事事实保存，不要求重复确认；"
                "它表示 Alice 曾这样自述，不表示现实世界已核实。value 不需要改写，系统会保存原证据。"
                "每条必须给出原 assistant 消息中的连续原文 evidence_quote，以及该消息的 event_index；"
                "不能引用或推断 role=user 内容，不能把用户经历移给 Alice，不能从出生地/城市推断能力或性格。"
                "不要收录单轮疲劳、心情、寒暄、假设角色扮演、系统身份声明或不完整句；"
                "facet_key 表示可跨会话复用的人格维度/主题，不是单次事件或单件物品的 ID；"
                "不同经历若确实体现同一稳定维度，应复用同一 facet_key，但每段经历必须独立保留原句、会话与轮次，不能合并成同一事件。"
                "只在共同维度有证据时复用标签；彼此无关的自述必须分开，单次事件不得泛化成长期性格。"
                "参考 existing_memory.alice_persona_facts 做跨会话标签对齐；新自述与已存事实体现同一维度时优先复用已有 facet_key。"
                "例如，不同旧物的独立经历可以都归到 sentimental-keepsakes，但每件物品仍是一条独立证据；不要把这个示例当成固定分类表。"
                "没有已有匹配时，facet_key 仍取可复用的一般维度，不用具体物品或单次事件命名。facet_key 用简短稳定的小写英文 slug。没有明确自述就返回空数组。"
            )
        payload = {
            "messages": bounded,
            "existing_memory": self._bounded_existing_memory(existing_memory or {}),
            "handoff": self._bounded_handoff(handoff_data),
        }
        try:
            raw = await client.chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                model=MEMORY_CURATOR_MODEL,
                base_url=MEMORY_CURATOR_BASE_URL,
                api_key=MEMORY_CURATOR_API_KEY,
                max_tokens=MEMORY_CURATOR_MAX_TOKENS,
                temperature=0.1,
                # 后台链路：走独立闸位，避免与实时语音回复抢 llm_client 的 2 个名额
                gate="background",
                # MEMORY_CURATOR_MODEL 常与导演共用 deepseek 系推理模型；不关
                # thinking 时推理 token 会吃光 max_tokens，输出为空导致
                # LLMCompletionError，跨会话记忆整理静默失败。
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                max_retries=1,
                timeout=MEMORY_CURATOR_TIMEOUT,
            )
            parsed = self._parse_json(raw)
            validated, accepted, rejected = self.validate_candidates(
                parsed.get("candidates") or [], user_texts
            )
            assistant_persona_facts, assistant_persona_rejected = (
                self.validate_assistant_persona_facts(
                    parsed.get("assistant_persona_facts") or [], bounded
                ) if curate_alice_persona else ([], 0)
            )
            base.update({
                "status": "completed",
                "candidates": validated,
                "accepted_items": accepted,
                "rejected_count": rejected,
                "episode_summary": str(parsed.get("episode_summary") or "")[:500],
                "contradictions": self._bounded_strings(parsed.get("contradictions"), 6, 240),
                "handoff_patch": self._safe_handoff_patch(parsed.get("handoff_patch")),
                "assistant_persona_facts": assistant_persona_facts,
                "assistant_persona_rejected_count": assistant_persona_rejected,
            })
        except Exception as exc:
            logger.warning("memory curator failed session=%s error=%s", session_id, type(exc).__name__)
            base.update({
                "status": "fallback_completed" if deterministic_validated else "failed",
                "failure_code": type(exc).__name__,
                "candidates": deterministic_validated,
                "accepted_items": deterministic_accepted,
                "rejected_count": deterministic_rejected,
                "fallback_reason": "llm_failure",
            })
        base["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return base

    @classmethod
    def validate_assistant_persona_facts(
        cls, candidates: Iterable[Dict[str, Any]], messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        """Accept only self-narrative quotes found verbatim in assistant turns.

        The stored value is the source quote itself, never the model's paraphrase;
        this keeps extraction from adding details that Alice did not say.
        """
        assistant_messages = []
        for ordinal, message in enumerate(messages):
            if str(message.get("role") or "") != "assistant":
                continue
            content = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
            if not content:
                continue
            try:
                event_index = int(message.get("event_index"))
            except (TypeError, ValueError):
                event_index = ordinal
            assistant_messages.append((content, event_index))

        accepted: List[Dict[str, Any]] = []
        rejected = 0
        seen = set()
        for raw in list(candidates or [])[:12]:
            if not isinstance(raw, dict):
                rejected += 1
                continue
            kind = str(raw.get("kind") or "").strip().lower()
            evidence = re.sub(r"\s+", " ", str(raw.get("evidence_quote") or "")).strip()
            facet = re.sub(r"[^a-z0-9_-]+", "-", str(raw.get("facet_key") or "").lower()).strip("-_")[:80]
            if (kind not in cls.ALLOWED_ASSISTANT_PERSONA_KINDS or len(evidence) < 12
                    or len(evidence) > 600 or not facet):
                rejected += 1
                continue
            if not re.search(r"\b(?:i|i['’]m|i['’]ve|i['’]ll|my|mine|we|our)\b|我|我的|我们", evidence, re.I):
                rejected += 1
                continue
            matching = [
                event_index for content, event_index in assistant_messages
                if evidence.casefold() in content.casefold()
            ]
            if not matching:
                rejected += 1
                continue
            signature = (kind, facet, evidence.casefold())
            if signature in seen:
                continue
            seen.add(signature)
            try:
                proposed_index = int(raw.get("event_index"))
            except (TypeError, ValueError):
                proposed_index = matching[-1]
            event_index = proposed_index if proposed_index in matching else matching[-1]
            # Reject obvious cut-off fragments such as “I wanted to move because”.
            if re.search(r"\b(?:and|but|because|so|or|a|an|the|to|of|with)$", evidence, re.I):
                rejected += 1
                continue
            accepted.append({
                "kind": kind,
                "facet_key": facet,
                "value": evidence,
                "evidence_quote": evidence,
                "event_index": event_index,
                "confidence": 0.62,
            })
        return accepted, rejected

    @classmethod
    def deterministic_candidates(cls, user_texts: List[str]) -> List[Dict[str, Any]]:
        """Extract only explicit, low-risk learner statements without an LLM.

        This intentionally narrow fallback makes a statement such as "以后多练
        酒店场景和过去式" survive an immediate cassette switch even when the
        semantic curator provider is unavailable.
        """
        candidates: List[Dict[str, Any]] = []
        next_markers = (
            "下次", "下一次", "下回", "以后继续练", "以后如果继续",
            "以后我还想", "以后我也还想", "以后想练", "下次再练", "next time", "next session",
        )
        teaching_terms = (
            "练", "学习", "英语", "英文", "语言", "过去式", "时态", "语法",
            "表达", "发音", "纠正", "跟读", "回复", "中英", "场景",
            "practice", "learn", "english", "grammar", "pronunciation", "correct",
        )

        for raw_text in user_texts[-24:]:
            evidence = re.sub(r"\s+", " ", str(raw_text or "")).strip()[:240]
            if not evidence:
                continue
            lower = evidence.lower()
            is_learning_statement = any(term in lower for term in teaching_terms)
            if any(marker in lower for marker in cls.PREFERENCE_MARKERS):
                candidates.append({
                    "kind": "learning_preference", "value": evidence,
                    "evidence_quote": evidence, "confidence": 0.98,
                    "durability_days": 365, "reason": "explicit learner preference",
                })
            if is_learning_statement and any(marker in lower for marker in next_markers):
                candidates.append({
                    "kind": "next_session_intent", "value": evidence,
                    "evidence_quote": evidence, "confidence": 0.98,
                    "durability_days": 120, "reason": "explicit future learning intent",
                })
            if cls._has_explicit_signal("self_reported_weakness", evidence):
                candidates.append({
                    "kind": "self_reported_weakness", "value": evidence,
                    "evidence_quote": evidence, "confidence": 0.98,
                    "durability_days": 180, "reason": "explicit self-reported weakness",
                })
            if (
                is_learning_statement
                and any(marker in lower for marker in cls.AVOID_MARKERS)
                and re.search(r"(?:不要|别再|不想|不需要|don['’]?t|do not|stop).{0,80}", lower)
            ):
                candidates.append({
                    "kind": "avoid_pattern", "value": evidence,
                    "evidence_quote": evidence, "confidence": 0.97,
                    "durability_days": 180, "reason": "explicit teaching-process boundary",
                })

        unique = []
        seen = set()
        for item in candidates:
            signature = (item["kind"], item["value"])
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(item)
        return unique[:12]

    @classmethod
    def validate_candidates(
        cls, candidates: Iterable[Dict[str, Any]], user_texts: List[str]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        validated: List[Dict[str, Any]] = []
        accepted: List[Dict[str, Any]] = []
        rejected = 0
        now = datetime.now(timezone.utc)
        for raw in list(candidates)[:16]:
            if not isinstance(raw, dict):
                rejected += 1
                continue
            kind = str(raw.get("kind") or "").strip()
            value = re.sub(r"\s+", " ", str(raw.get("value") or "")).strip()[:240]
            evidence = re.sub(r"\s+", " ", str(raw.get("evidence_quote") or "")).strip()[:240]
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
            except (TypeError, ValueError):
                confidence = 0.0
            evidence_matches = bool(evidence) and any(
                evidence in re.sub(r"\s+", " ", text) for text in user_texts
            )
            if kind not in cls.ALLOWED_KINDS or not value or not evidence_matches:
                rejected += 1
                continue
            days = cls._clamp_days(raw.get("durability_days"), kind)
            # 显式记忆请求（"Remember this / 记住…"）是用户能给出的最强确认：用户明确
            # 要求记住的内容必须落库，不能因为措辞不是 "I like X" 而被门槛挡掉。
            # plans/1012 实测：「Remember this: my favorite food is ramen, and my cat's
            # name is Momo.」被抽为 engagement_topic、confidence 0.95、evidence 逐字匹配，
            # 但 _has_explicit_signal 只认 like/love/喜欢 这类动词 ⇒ 永不 auto_accept
            # ⇒ accepted=[] ⇒ 用户明确投喂的信息跨会话即失忆。
            explicit_memory_request = cls._has_explicit_memory_request(evidence)
            auto_accepted = (
                explicit_memory_request
                and confidence >= 0.9
            ) or (
                kind in cls.AUTO_ACCEPT_KINDS
                and confidence >= 0.9
                and cls._has_explicit_signal(kind, evidence)
            )
            candidate = {
                "id": f"curator_{uuid.uuid4().hex[:12]}",
                "kind": kind,
                "value": value,
                "evidence_quote": evidence,
                "confidence": round(confidence, 3),
                "durability_days": days,
                "reason": str(raw.get("reason") or "")[:200],
                "status": "accepted" if auto_accepted else "candidate",
            }
            validated.append(candidate)
            if auto_accepted:
                accepted.append({
                    "id": candidate["id"],
                    "key": kind,
                    "value": value,
                    "confidence": round(confidence, 3),
                    "source": "memory_curator_explicit_user_evidence",
                    "evidence": evidence,
                    "created_at": now.isoformat(),
                    "expires_at": (now + timedelta(days=days)).isoformat(),
                    "confirmed_by_user": True,
                })
        return validated, accepted, rejected

    @classmethod
    def _has_explicit_signal(cls, kind: str, evidence: str) -> bool:
        lower = evidence.lower()
        if kind == "next_session_intent" or kind == "open_loop":
            return any(marker in lower for marker in cls.FUTURE_MARKERS) or any(
                marker in lower for marker in ("以后我还想", "下次再练")
            )
        if kind == "learning_preference":
            return any(marker in lower for marker in cls.PREFERENCE_MARKERS)
        if kind == "self_reported_weakness":
            return (
                any(marker in lower for marker in cls.WEAKNESS_MARKERS)
                or any(marker in lower for marker in ("经常", "总是", "老是", "容易"))
                and any(marker in lower for marker in ("错", "不会", "不对"))
            )
        if kind == "avoid_pattern":
            return any(marker in lower for marker in cls.AVOID_MARKERS)
        if kind == "engagement_topic":
            # 2026-10-25 画像偏科修复：兴趣判定归模型（curator 本身就是 LLM 调用，
            # kind=engagement_topic 已是它的判断结果），这里不再用词面动词闸收窄——
            # 实测 586 行里兴趣仅 16 行（87% 是 last_topic/心情流水），"我周末常去
            # 爬山"这类高频行为信号因不含"喜欢"二字永不入库，她学不会用户的真实
            # 兴趣。边界仍由外层两道硬闸守：evidence 逐字匹配用户原话 + conf≥0.9
            # （Momo 类脑补由证据闸挡；垃圾值另有 looks_like_garbage_value 闸）。
            return True
        return False

    @staticmethod
    def _has_explicit_memory_request(evidence: str) -> bool:
        """用户明确要求「记住」某事的措辞 —— 独立于 kind 的最强确认信号。

        plans/1012：此前 engagement_topic 的 _has_explicit_signal 只认 like/love/喜欢
        这类动词，导致「Remember this: my favorite food is ramen…」抽得出来却进不了
        accepted（accepted=[]），用户明确投喂的信息跨会话即失忆。
        这里只认祈使式显式记忆请求，不扩大 engagement_topic 的常规判定。
        """
        return bool(re.search(
            r"\bremember\s+(?:this|that|it)\b|\bplease\s+remember\b|"
            r"\b(?:don'?t|do\s+not)\s+forget\b|\bkeep\s+(?:this|that|it)\s+in\s+mind\b|"
            r"\bnote\s+(?:this|that|it)\s+down\b|\bmemorize\b|"
            r"记住|记一下|记下这个|帮我记|别忘|不要忘|记住这个",
            evidence, re.IGNORECASE,
        ))

    @staticmethod
    def _clamp_days(value: Any, kind: str) -> int:
        defaults = {
            "next_session_intent": 120,
            "open_loop": 90,
            "learning_preference": 365,
            "self_reported_weakness": 180,
            "avoid_pattern": 180,
            "engagement_topic": 180,
        }
        try:
            days = int(value)
        except (TypeError, ValueError):
            days = defaults.get(kind, 30)
        return max(1, min(days, 730))

    @staticmethod
    def _bounded_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in messages[-24:]:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                bounded_item: Dict[str, Any] = {"role": role, "content": content[:900]}
                try:
                    if item.get("event_index") is not None:
                        bounded_item["event_index"] = int(item.get("event_index"))
                except (TypeError, ValueError):
                    pass
                result.append(bounded_item)
        return result

    @staticmethod
    def _bounded_existing_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
        allowed = (
            "learning_preference", "next_session_intent", "self_reported_weakness",
            "avoid_patterns", "preferred_tone", "open_loop", "engagement_topics",
        )
        bounded = {key: memory.get(key) for key in allowed if memory.get(key)}
        persona_facts = []
        for row in memory.get("alice_persona_facts") or []:
            if not isinstance(row, dict) or row.get("status", "active") != "active":
                continue
            kind = str(row.get("kind") or "").strip().lower()[:80]
            facet = str(row.get("facet_key") or "").strip().lower()[:120]
            value = re.sub(r"\s+", " ", str(row.get("value_text") or "")).strip()[:240]
            if kind not in MemoryCuratorAgent.ALLOWED_ASSISTANT_PERSONA_KINDS or not facet or not value:
                continue
            persona_facts.append({"kind": kind, "facet_key": facet, "value_text": value})
            if len(persona_facts) >= 16:
                break
        if persona_facts:
            bounded["alice_persona_facts"] = persona_facts
        return bounded

    @staticmethod
    def _bounded_handoff(handoff: Dict[str, Any]) -> Dict[str, Any]:
        bounded = {
            "scene": handoff.get("scene") or {},
            "conversation_plan": handoff.get("conversation_plan") or {},
            "scene_ledger": handoff.get("scene_ledger") or {},
        }
        if handoff.get("_curate_alice_persona"):
            bounded["_curate_alice_persona"] = True
        return bounded

    @staticmethod
    def _parse_json(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            value = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(0))
                except (TypeError, ValueError, json.JSONDecodeError):
                    value = MemoryCuratorAgent._recover_truncated_object(match.group(0))
            else:
                value = MemoryCuratorAgent._recover_truncated_object(text)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _recover_truncated_object(raw: str) -> Dict[str, Any]:
        """Salvage complete top-level members from a token-truncated JSON object.

        deepseek 系推理模型即使关闭 thinking 也可能在长 JSON 输出时截断；
        保留完整的靠前字段（candidates 等）比整段丢弃更安全。
        """
        text = str(raw or "").strip()
        start = text.find("{")
        if start < 0:
            return {}
        text = text[start:]
        in_string = False
        escaped = False
        depth = 0
        top_level_commas: List[int] = []
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth = max(0, depth - 1)
            elif char == "," and depth == 1:
                top_level_commas.append(index)

        candidates = [text.rstrip().rstrip(",") + "}"]
        candidates.extend(text[:index].rstrip() + "}" for index in reversed(top_level_commas))
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value:
                return value
        return {}

    @staticmethod
    def _bounded_strings(value: Any, limit: int, chars: int) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:chars] for item in value[:limit] if str(item).strip()]

    @staticmethod
    def _safe_handoff_patch(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        allowed = {"current_phase", "next_counterpart_action", "open_items", "completed_actions"}
        return {key: value[key] for key in allowed if key in value}
