"""
Profile Evolution Engine

Analyzes conversation sessions and incrementally builds a rich user profile.
Designed to be called after each session (on_session_end hook).

Integrated with L2SQLiteMemory system for persistent storage.
"""

import re
import json
import logging
from difflib import SequenceMatcher
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("profile_evolution")

try:
    from ..memory.l2_sqlite import L2SQLiteMemory
    from ..memory.provider import maybe_await
    from ..agent.intent_router import IntentRouter, TutorIntent
except ImportError:
    from memory.l2_sqlite import L2SQLiteMemory
    from memory.provider import maybe_await
    from agent.intent_router import IntentRouter, TutorIntent


from llm_client import background_llm_job


def _cjk_ratio(text: str) -> float:
    """中文字符占比——用来判断"这句是中文还是英文"，比整串语言检测便宜可靠。"""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cjk = sum(1 for ch in letters if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(letters)


def _match_language(text: str, quote: str) -> str:
    """让沉淀下来的文本跟她原话同语言（plans/1017-02 §8.45）。

    实测：她英文说「I like things mild…」，提炼出的 `text` 却是中文
    「喜欢清淡口味」——日后被提起时会在英文回复里蹦中文。
    提示词里已要求同语言，但仍会漏；这里做代码兜底：语言不一致时**用她的原话**，
    宁可少一层概括，也不要在英文回复里插一句中文。
    """
    if not text:
        return quote
    text_cjk, quote_cjk = _cjk_ratio(text), _cjk_ratio(quote)
    if (quote_cjk < 0.2 and text_cjk > 0.3) or (quote_cjk > 0.8 and text_cjk < 0.2):
        return quote
    return text


class ProfileEvolutionEngine:
    """
    Analyzes session data and incrementally updates user profiles.
    Each session refines the profile rather than rebuilding from scratch.

    Uses L2SQLiteMemory for persistent storage of:
    - error_fingerprints
    - mastered_expressions
    - session_archives
    - style_parameters
    """

    # Batch processing thresholds
    BATCH_SIZE = 25
    BATCH_THRESHOLD = 50
    COMPRESSION_THRESHOLD = 100

    # 表达习惯信号（plans/0919-03 §4.6）。这是**软适配信号**，只决定 AI
    # "怎么说"，不是能力评分，不得用于评价用户。
    #
    # 词表一律走词边界匹配：此前用 `w in text` 子串判定，"or" 会命中
    # for/story/work、"so" 会命中 some/also，实测 5 条普通陈述句即把
    # hesitant 打满 5/5、analytical 3/5 —— 标签恒为"犹豫型"，
    # 接进 prompt 就是给每个用户恒定误适配。
    #
    # 中文侧刻意不做词表：没有分词器时中文只能子串匹配，正是此次修掉的
    # 缺陷。中文只认标点（全角 ！？），标点不需要分词。
    COMM_STYLE_EN_WORDS = {
        "expressive": ("really", "very", "super", "absolutely", "totally"),
        "analytical": ("because", "since", "therefore", "however", "although", "but", "so", "if"),
        "hesitant": ("um", "uh", "maybe", "perhaps", "not sure", "i think"),
    }
    # 标签成型的门槛：会话太短或信号太弱宁可不给，也不要拿噪声引导模型。
    COMM_STYLE_MIN_MESSAGES = 3
    COMM_STYLE_MIN_RATIO = 0.25
    COMM_STYLE_MAX_LABELS = 2

    # CEFR vocabulary markers for rough estimation
    CEFR_MARKERS = {
        "A1": ["hello", "thank", "yes", "no", "please", "sorry", "good", "bad", "like", "want"],
        "A2": ["because", "but", "also", "think", "know", "feel", "usually", "maybe", "help", "need"],
        "B1": ["however", "although", "probably", "actually", "basically", "definitely", "suggest", "consider", "describe", "explain"],
        "B2": ["nevertheless", "furthermore", "consequently", "implications", "perspective", "analyze", "comprehensive", "significant", "substantial", "hypothetically"],
        "C1": ["paradigm", "nuanced", "juxtaposition", "pragmatic", "ubiquitous", "ambivalent", "elucidate", "paradoxical", "dichotomy", "hegemonic"],
        "C2": ["epistemological", "ontological", "hermeneutic", "teleological", "syllogism", "solipsism", "aperçu", "Weltschmerz", "Zeitgeist", "Sonderweg"],
    }

    # Topic keywords for interest extraction
    TOPIC_KEYWORDS = {
        "technology": ["computer", "software", "app", "AI", "robot", "internet", "digital", "programming", "code", "data", "tech"],
        "travel": ["travel", "flight", "airport", "hotel", "passport", "visabooking", "destination", "sightseeing", "tourist"],
        "food": ["coffee", "restaurant", "food", "delicious", "cook", "menu", "eat", "drink", "tea", "cake", "pizza", "sushi"],
        "movies": ["movie", "film", "watch", "actor", "actress", "cinema", "series", "netflix", "director", "scene"],
        "music": ["music", "song", "sing", "guitar", "piano", "concert", "band", "listen", "album", "spotify"],
        "sports": ["sport", "football", "basketball", "run", "gym", "exercise", "fitness", "swim", "tennis", "yoga"],
        "work": ["work", "office", "meeting", "boss", "colleague", "project", "deadline", "career", "job", "company", "interview"],
        "study": ["study", "exam", "university", "class", "homework", "professor", "student", "learn", "school", "book"],
        "gaming": ["game", "play", "gaming", "console", "steam", "rpg", "multiplayer", "esports", "twitch", "streamer"],
        "fashion": ["fashion", "clothing", "style", "wear", "brand", "shopping", "outfit", "trend", "designer"],
        "relationships": ["friend", "family", "relationship", "dating", "love", "marriage", "parent", "children", "partner"],
        "shopping": ["buy", "shop", "price", "discount", "cart", "payment", "refund", "store", "expensive", "cheap"],
        "weather": ["weather", "rain", "sunny", "cold", "hot", "warm", "snow", "temperature", "wind", "cloudy"],
    }

    def __init__(self, l2_memory: Optional[L2SQLiteMemory] = None):
        """
        Initialize with L2SQLiteMemory instance via dependency injection.

        Args:
            l2_memory: L2SQLiteMemory instance for persistent storage.
                       If None, profile evolution operates in read-only mode.
        """
        self.l2 = l2_memory
        self.intent_router = IntentRouter()

    async def _guarded_write(self, user_id: str, token: str, writer,
                             conv_id: str = ""):
        """在删除闸门内执行一处**既有**写入（plans/0923-04 §4.4 阶段 2）。

        与直接 `await writer()` 完全等价，除非该用户的写入令牌已被全删/删盘撤销。
        没有令牌、后端不支持租约、或闸门自身出错时一律放行——即回退到原行为
        （§4.5「新读口异常时整盘回旧路径」）。被拒写时返回 `None`。
        """
        if (not str(token or "") or self.l2 is None
                or not hasattr(self.l2, "write_lease_active")):
            return await writer()
        try:
            from memory.learner_profile import ProfileMemory
        except Exception:
            return await writer()
        result = await ProfileMemory(self.l2).guarded_write(
            user_id, [token], writer, source_conv_id=conv_id
        )
        return result.get("value") if result.get("ok") else None

    @background_llm_job
    async def on_session_end(self, user_id: str, session_id: str,
                             conversation_messages: list[dict],
                             session_meta: Optional[dict] = None):
        """
        Called when a session ends. Analyzes the conversation and updates the profile.

        Flow:
        1. Get historical memory context from L2
        2. Extract user messages
        3. Batch process if needed (>50 messages) or compress if >100
        4. Sync results to L2 memory tables
        5. Save session archive
        """
        if not conversation_messages:
            return

        # 写入租约闸门（plans/0923-04 §4.4 阶段 2）：**在任何写入和分析之前**登记
        # 本次会话末作业的令牌。登记必须发生在开头——放到分析之后，一次"删盘→分析
        # 跑完→登记"就会拿到全新令牌，反而把迟到写入放进来。
        # 令牌面**不含原话**，只用于"来源盘是否已被撤销"的判定。
        meta = session_meta or {}
        lease_token = ""
        lease_conv = str(meta.get("conversation_id") or "")
        try:
            if self.l2 is not None:
                from memory.learner_profile import ProfileMemory
                lease_token = await ProfileMemory(self.l2).admit_write(
                    user_id, source_conv_id=lease_conv
                )
        except Exception:
            lease_token = ""

        # Step 1: Get historical memory context
        memory_ctx = await self.l2.get_memory_context(user_id) if self.l2 else {}

        # Step 2: Extract user messages only
        user_messages = [
            m for m in conversation_messages
            if m.get("role") == "user" and m.get("content")
        ]
        if not user_messages:
            return

        # Step 3: Compress long conversations if needed
        if len(user_messages) > self.COMPRESSION_THRESHOLD:
            user_messages = await self._compress_long_conversation(user_messages)

        # ── 非教学盘分盘（2026-10-25）：素材盘/闲聊盘的会话末**不做教学评估**
        # （CEFR/错误/纠错/掌握表达——这些是给「练语言」服务的；素材盘聊素材、闲聊盘
        # 陪人，评 CEFR/错误是误套教学模板，且与人格先行「收掉教学服务」一致）。
        # 练习盘仍保留完整教学分析。轻量 analysis 只保留后续归档必需的无害字段，
        # 教学画像（_sync_to_memory）、CEFR 前后置归档值在非教学盘整体跳过。
        _is_teaching = bool(meta.get("is_expression_tape") or False)
        if not _is_teaching:
            analysis = {
                "message_count": len(user_messages),
                "topics_detected": [],
                "error_patterns": {},
                "correction_preference": "",
            }
        else:
            # Step 4: Batch process if needed
            if len(user_messages) > self.BATCH_THRESHOLD:
                analysis = await self._analyze_in_batches(user_messages, memory_ctx)
            else:
                analysis = self._analyze_session(user_messages, memory_ctx)

        # Cross-session summaries must come from explicit session state, not
        # an arbitrary slice of raw ASR utterances. Raw ASR can contain partial
        # words, corrections and product feedback that should not become memory.
        event_parts = [
            str(meta.get("scene_brief") or "").strip(),
            str(meta.get("target_goal") or "").strip(),
            str(meta.get("current_phase") or "").strip(),
        ]
        event_parts = [item for item in event_parts if item]
        if event_parts:
            analysis["event_summary"] = "；".join(dict.fromkeys(event_parts))[:240]

        # Step 5: Enrich analysis with historical data（仅教学盘——非教学盘不做历史教学增强）
        if _is_teaching:
            analysis = self._enrich_from_memory(memory_ctx, analysis)

        # Step 6: Sync to L2 memory tables（仅教学盘——非教学盘已收掉教学画像写入）
        if _is_teaching:
            await self._guarded_write(
                user_id, lease_token,
                lambda: self._sync_to_memory(user_id, analysis), lease_conv,
            )

        # Step 7: Save session archive
        if _is_teaching:
            cefr_before = self._guess_current_cefr(memory_ctx)
            cefr_after = analysis.get("cefr_estimate", (cefr_before, 0.5))[0]
        else:
            # 非教学盘：不评 CEFR（cefr 字段给空）
            cefr_before = ""
            cefr_after = ""

        if self.l2:
            # 归档摘要：练习盘维持原逻辑（默认模板，开关开才走 LLM）；素材盘/闲聊盘
            # **直接走语义摘要**（_content_summary）——这样归档有真实内容、不再套
            # "练习目标语言/CEFR/recast"教学模板，回忆检索才记得住聊了素材什么。
            summary_text = ""
            _archive_llm_on = False
            if _is_teaching:
                try:
                    from config import ARCHIVE_SUMMARY_LLM_ENABLED as _archive_llm_on
                except Exception:
                    _archive_llm_on = False
                if _archive_llm_on:
                    summary_text = await self._content_summary(conversation_messages)
            else:
                summary_text = await self._content_summary(conversation_messages)
            # ── 她说过的话沉淀成候选事实（§8.41）：只在会话末，不进热路径。
            # §8.58：**只在闲聊盘做**——素材盘/练习盘里她多半在讨论素材
            # （实测游记里的 "a small room in Singapore" 被抽成了她的 origin=Singapore）。
            _chat_tape = meta.get("companion_chat")
            try:
                existing_facets = []
                if hasattr(self.l2, "list_persona_facets"):
                    existing_facets = await self.l2.list_persona_facets(user_id)
                existing_facts = []
                if hasattr(self.l2, "list_companion_persona_facts"):
                    existing_facts = await self.l2.list_companion_persona_facts(
                        user_id, limit=200) or []
                if _chat_tape is False:
                    distilled = []
                    print("[profile] self-fact skipped：非闲聊盘（§8.58）")
                else:
                    distilled = await self._distill_self_facts(
                        conversation_messages, existing_facets=existing_facets,
                        existing_facts=existing_facts,
                        source_conv_id=lease_conv or str(session_id or ""))
            except Exception:
                distilled = []
            if distilled:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    from memory.companion_persona import observation_to_fact
                    stamp = _dt.now(_tz.utc).isoformat()
                    rows = []
                    for item in distilled:
                        # 仲裁只认 trait/preference/ability/language_environment/current_status，
                        # 所以 stance 归到 trait，否则它永远不参与升格。
                        kind = ("preference" if item["kind"] == "preference" else "trait")
                        fact = observation_to_fact(
                            {"category": kind,
                             "facet": item["facet_key"],
                             "value": item["value_text"]},
                            user_id=user_id,
                            conv_id=lease_conv or str(session_id or ""),
                            turn=int(len(conversation_messages)),
                            observed_at=stamp,
                            evidence_quote=item["quote"],
                            confidence=0.55,
                        )
                        fact["status"] = "candidate"   # ⛔ 先候选；跨会话重复才升 active
                        fact["source_kind"] = "session_self_statement"
                        rows.append(fact)
                    # ⛔ 必须 guard_identity=True：传 False 会跳过整段版本仲裁，
                    # candidate 就永远升不上去（§8.41 实测踩到）。
                    written = await self.l2.upsert_companion_persona_facts(
                        user_id, rows, guard_identity=True)
                    print(f"[profile] self-facts candidate written: {written}")
                except Exception as exc:
                    print(f"[profile] self-fact persist failed: {type(exc).__name__}")
            # §8.91c：**已关闭自动降级**。随机 50 对实测：模型语义判据精确率只有 33%
            # （二分类 6%），自动改库会误杀真事实。改为：
            #   ① 写入时用字面规则挡（精确率 100%，零成本，见 tutor_agent）；
            #   ② 其余可疑项由 `tools/echo_review_sweep.py` 出清单，人工确认后再降。
            # 保留 `_review_echo_disclosures` 供人工/脚本调用，但**不在会话末自动执行**。
            # （原调用点用 conv_id 匹配；时序不稳，且判据精度不够——见上。）

            # ── 未闭环的事（§8.56）：会话末抽"他提过、还没有下文的事"。
            # 1 次调用，不在语音路径上；默认关。
            try:
                from config import OPEN_THREADS_ENABLED as _threads_on
            except Exception:
                _threads_on = False
            unanswered_q: dict = {}
            if _threads_on:
                try:
                    try:
                        from engine.life_line_gen import refresh_life_line
                        # §8.79：**只在闲聊盘生成**——练习盘/素材盘结束也生成她的"这周"
                        # 没有意义，还白花一次调用（2026-10-18 自查发现）。
                        if _chat_tape:
                            await refresh_life_line(self.l2, user_id)
                    except Exception:
                        pass
                    opened, resolved_topics, unanswered_q = await self._distill_open_threads(
                        conversation_messages)
                    if opened:
                        written_threads = await self.l2.upsert_open_threads(
                            user_id, opened, source_conv_id=str(session_id or ""))
                        print(f"[profile] open threads written: {written_threads}")
                    if resolved_topics:
                        from agent.open_threads import score_thread
                        existing = await self.l2.list_open_threads(user_id, limit=20)
                        closing = [
                            t["thread_id"] for t in existing
                            if any(score_thread(t, topic) >= 2 for topic in resolved_topics)
                        ]
                        if closing:
                            await self.l2.close_open_threads(
                                user_id, closing, reason="session_end_resolved")
                            print(f"[profile] open threads closed: {len(closing)}")
                except Exception as exc:
                    print(f"[profile] open-thread failed: {type(exc).__name__}")

            # ── 她问过、他没答的问题（§8.76）：纯规则，零模型调用。
            try:
                from config import COMPANION_OPEN_QUESTIONS_ENABLED as _oq_on
            except Exception:
                _oq_on = False
            if _oq_on:
                try:
                    # 判断"她这问题算不算没被回答"交给模型（同一次抽取，零额外调用）；
                    # ⛔ 不能用关键词打分：他把"爱吃鱼"答了，词面上也匹配不到 "what to eat"。
                    from agent.open_questions import question_record
                    kept = question_record(
                        str(unanswered_q.get("text") or ""),
                        str(unanswered_q.get("topic") or "")) if _threads_on else None
                    if kept:
                        await self.l2.upsert_open_question(
                            user_id, kept, str(session_id or ""))
                        print(f"[profile] open question kept: {kept['text'][:40]}")
                    await self.l2.expire_open_questions(user_id, days=3)
                except Exception as exc:
                    print(f"[profile] open-question failed: {type(exc).__name__}")

            # 问过的线程回写（§8.60）：存在库里，下一盘/重启后都知道别重复问。
            _asked = list(meta.get("open_threads_asked") or [])
            if _asked and hasattr(self.l2, "mark_thread_asked"):
                try:
                    await self.l2.mark_thread_asked(user_id, _asked)
                except Exception as exc:
                    print(f"[profile] asked writeback failed: {type(exc).__name__}")

            # 留言箱（§8.60）：明天/后天到期的事，提前留一条。
            try:
                from config import COMPANION_OUTBOX_ENABLED as _outbox_on
            except Exception:
                _outbox_on = False
            if _outbox_on and hasattr(self.l2, "enqueue_outbox"):
                try:
                    mid = await self._enqueue_due_outbox(user_id, str(session_id or ""))
                    if mid:
                        print(f"[outbox] queued {mid}")
                except Exception as exc:
                    print(f"[outbox] failed: {type(exc).__name__}")

            # 只记内容事件（±1），由 closeness_score 带半衰期累加 ⇒ 会升也会降。
            try:
                delta_infos = await self._closeness_deltas(conversation_messages)
            except Exception:
                delta_infos = []
            if delta_infos:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    import uuid as _uuid
                    stamp = _dt.now(_tz.utc)
                    events = []
                    for index, info in enumerate(delta_infos):
                        if not info.get("delta"):
                            continue
                        events.append({
                            "id": f"closeness_{_uuid.uuid4().hex[:10]}",
                            "key": "alice_closeness_delta",
                            "kind": "closeness_delta",
                            "value": float(info["delta"]),
                            "summary": str(info.get("reason") or "")[:200],
                            "confidence": 0.8,
                            "source": "session_end_closeness",
                            "source_conv_id": str(session_id or ""),
                            "source_turn": index,
                            "created_at": stamp.isoformat(),
                            "observed_at": stamp.isoformat(),
                            "confirmed_by_user": False,
                        })
                    if events:
                        await self.l2.upsert_relationship_items(user_id, events)
                except Exception as exc:
                    print(f"[profile] closeness delta persist failed: "
                          f"{type(exc).__name__}")
            # 必须等待落库：postgres 后端这三个写入器是 coroutine，直接调用
            # 只会得到一个从未执行的协程（RuntimeWarning: never awaited），
            # 归档静默丢失 → 该会话在跨会话检索里永远查不到（0915-11）。
            # 摘要同步做向量 embedding（plans/0915-06；main.py:2940 同款保守模式）：
            # 让历史检索的向量通道生效——只有语义摘要、没向量的行，用户"换措辞"
            # 回忆（如"糯米团子"→"酒酿圆子"、"看花田"→"北海道薰衣草"）会漏召回。
            # 失败静默退回纯稀疏，不影响归档本身。
            _archive_summary = summary_text or self._generate_summary(analysis)
            _summary_embedding = None
            if _archive_summary:
                try:
                    from rag.shared_embedder import embed_text as _embed
                    _summary_embedding = await _embed(_archive_summary)
                except Exception:
                    _summary_embedding = None
            # 双后端签名兼容：SQLite/PG 的 save_session_archive 应有 summary_embedding
            # 参数；个别旧实现没有该参数时按 main.py 惯例剔除，避免 TypeError 丢归档。
            try:
                import inspect as _inspect
                _sig = _inspect.signature(self.l2.save_session_archive)
                _archive_kw = (
                    {"summary_embedding": _summary_embedding}
                    if "summary_embedding" in _sig.parameters else {}
                )
            except (TypeError, ValueError):
                _archive_kw = {}
            await self._guarded_write(
                user_id, lease_token,
                lambda: maybe_await(
                    self.l2.save_session_archive(
                        user_id=user_id,
                        session_id=session_id,
                        summary=_archive_summary,
                        error_count=sum(analysis.get("error_patterns", {}).values()),
                        cefr_before=cefr_before,
                        cefr_after=cefr_after,
                        **_archive_kw,
                    )
                ),
                lease_conv,
            )

    # ── Batch Processing ─────────────────────────────────

    async def _analyze_in_batches(self, user_messages: list[dict],
                                   memory_ctx: dict) -> dict:
        """
        Process user messages in batches for long conversations.
        Each batch of BATCH_SIZE messages is analyzed independently,
        then results are merged.
        """
        batches = [
            user_messages[i:i + self.BATCH_SIZE]
            for i in range(0, len(user_messages), self.BATCH_SIZE)
        ]

        results = []
        for batch in batches:
            batch_analysis = self._analyze_session(batch, memory_ctx)
            results.append(batch_analysis)

        return self._merge_batch_results(results)

    def _merge_batch_results(self, results: list[dict]) -> dict:
        """
        Merge multiple batch analysis results into a single analysis.
        Weighted by message count for CEFR and error aggregation.
        """
        if not results:
            return self._analyze_session([], {})

        if len(results) == 1:
            return results[0]

        merged = {
            "cefr_estimate": self._weighted_cefr_merge(results),
            "topics_detected": [],
            "error_patterns": {},
            "communication_style": {"direct": 0, "expressive": 0, "analytical": 0, "hesitant": 0},
            "correction_preference": "recast",
            "english_ratio": 0.0,
            "message_count": 0,
            "avg_message_length": 0.0,
        }

        all_topics = []
        correction_votes = {}
        total_chars = 0
        total_messages = 0

        for r in results:
            # Merge topics (deduplicate preserving order)
            for topic in r.get("topics_detected", []):
                if topic not in all_topics:
                    all_topics.append(topic)

            # Merge error patterns (sum counts)
            for err_type, count in r.get("error_patterns", {}).items():
                merged["error_patterns"][err_type] = merged["error_patterns"].get(err_type, 0) + count

            # Merge communication style signals (sum)
            comm = r.get("communication_style", {})
            for key in merged["communication_style"]:
                merged["communication_style"][key] += comm.get(key, 0)

            # Vote on correction preference
            pref = r.get("correction_preference", "recast")
            correction_votes[pref] = correction_votes.get(pref, 0) + 1

            # Weighted english ratio
            msg_count = r.get("message_count", 1)
            merged["english_ratio"] += r.get("english_ratio", 0.5) * msg_count
            total_messages += msg_count
            total_chars += r.get("avg_message_length", 0) * msg_count

        merged["topics_detected"] = all_topics[:5]

        # Correction preference by majority vote
        if correction_votes:
            merged["correction_preference"] = max(correction_votes, key=correction_votes.get)

        # Weighted average for english ratio
        merged["english_ratio"] = round(merged["english_ratio"] / max(total_messages, 1), 2)
        merged["message_count"] = total_messages
        merged["avg_message_length"] = total_chars / max(total_messages, 1)

        return merged

    async def _compress_long_conversation(self, user_messages: list[dict]) -> list[dict]:
        """
        Compress conversations with >100 user messages.
        Generates an intermediate summary message representing the bulk,
        then keeps the most recent messages for fine-grained analysis.
        """
        if len(user_messages) <= self.COMPRESSION_THRESHOLD:
            return user_messages

        # Keep last 50 messages for detailed analysis
        recent = user_messages[-50:]
        early = user_messages[:-50]

        # Generate summary from early messages
        early_text = " ".join([m.get("content", "")[:100] for m in early])
        early_topics = self._extract_topics(early_text)
        early_errors = self._extract_error_patterns(early)

        # Create a synthetic summary message
        summary_message = {
            "role": "user",
            "content": f"[Session summary: {len(early)} earlier messages. "
                      f"Topics: {', '.join(early_topics[:3])}. "
                      f"Key errors: {', '.join(list(early_errors.keys())[:3])}]"
        }

        return [summary_message] + recent

    # ── Enrichment & Sync ─────────────────────────────────

    def _enrich_from_memory(self, memory_ctx: dict, analysis: dict) -> dict:
        """
        Combine L2 historical data with current session analysis.
        Historical data provides smoothing and trend detection.
        """
        if not memory_ctx:
            return analysis

        # Enrich error patterns with historical weights
        historical_errors = {
            e["error_type"]: e for e in memory_ctx.get("errors", [])
        }

        enriched_errors = dict(analysis.get("error_patterns", {}))
        for err_type, hist in historical_errors.items():
            if err_type in enriched_errors:
                # Boost count for recurring errors
                enriched_errors[err_type] = enriched_errors[err_type] + hist.get("count", 0)
            elif hist.get("weight", 0) > 0.3:
                # DECAY: Keep historically significant errors but decay count if not in current session
                decayed_count = int(hist.get("count", 0) * 0.7)
                if decayed_count > 0:
                    enriched_errors[err_type] = decayed_count

        analysis["error_patterns"] = enriched_errors

        # Enrich topics with recent session data
        recent_topics = set()
        for session in memory_ctx.get("recent_sessions", []):
            summary = session.get("summary", "")
            if summary:
                recent_topics.update(self._extract_topics(summary))

        current_topics = set(analysis.get("topics_detected", []))
        combined_topics = list(current_topics) + list(recent_topics - current_topics)
        analysis["topics_detected"] = combined_topics[:5]

        # Store historical context for downstream use
        analysis["historical_total_sessions"] = memory_ctx.get("total_sessions", 0)

        return analysis

    async def _sync_to_memory(self, user_id: str, analysis: dict):
        """
        Write analysis results back to L1 + L2 memory tables.
        Updates L1 weakness fields, L2 error_fingerprints, mastered_expressions,
        and style_parameters.
        """
        if not self.l2:
            logger.warning("_sync_to_memory: self.l2 is None, skipping write")
            return

        error_patterns = analysis.get("error_patterns", {})
        mastered_list = analysis.get("mastered_expressions", [])
        cefr_estimate, cefr_conf = analysis.get("cefr_estimate", ("", 0.0))

        # ── L1 画像字段写入 ──
        top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        recent_weakness = ", ".join(f"{err}({cnt})" for err, cnt in top_errors) if top_errors else ""

        # 提取画像完整字段
        topics_detected = analysis.get("topics_detected", [])
        comm_style = analysis.get("communication_style", {})
        # 表达习惯标签（plans/0919-03 §4.6）：此前只用来推 learning_style，
        # 四维原值算完即弃，读侧因此无值可注入。
        comm_label = self._distill_communication_style(
            comm_style, analysis.get("message_count", 0)
        )
        # 根据沟通风格推断 learning_style
        if comm_style.get("analytical", 0) > max(comm_style.get("direct", 0), comm_style.get("expressive", 0)):
            learning_style = "analytical"
        elif comm_style.get("expressive", 0) > comm_style.get("direct", 0):
            learning_style = "expressive"
        else:
            learning_style = "balanced"
        # 置信度基于消息量和错误覆盖
        msg_count = analysis.get("message_count", 0)
        confidence = min(0.9, 0.3 + msg_count * 0.03)
        # review_queue: 将高频未掌握错误加入复习队列
        review_queue = ",".join([err for err, cnt in top_errors if cnt >= 2])

        try:
            if hasattr(self.l2, "update_l1_profile"):
                await self.l2.update_l1_profile(user_id, {
                    "cefr": cefr_estimate,
                    "recent_weakness": recent_weakness,
                    "preferred_topics": topics_detected[:5],
                    "learning_style": learning_style,
                    "communication_style": comm_label,
                    "review_queue": review_queue,
                    "confidence_score": round(confidence, 2),
                })
                logger.debug(f"_sync_to_memory [{user_id}]: L1 updated — CEFR={cefr_estimate}, weakness={recent_weakness}, topics={topics_detected[:3]}, style={learning_style}, comm={comm_label}")
        except Exception as e:
            logger.warning(f"_sync_to_memory [{user_id}]: L1 update failed: {e}")

        # ── L2 error_fingerprints ──
        # Overwrite exact counts and weights using set_error_fingerprint to reflect decay
        memory_ctx = await self.l2.get_memory_context(user_id) if self.l2 else {}
        hist_error_types = {e["error_type"] for e in memory_ctx.get("errors", [])}

        for error_type, count in error_patterns.items():
            weight = min(1.0, 0.5 + count * 0.1)
            if hasattr(self.l2, "set_error_fingerprint"):
                await maybe_await(self.l2.set_error_fingerprint(user_id, error_type, count, weight))
            else:
                if count > 0:
                    await maybe_await(self.l2.add_error(user_id, error_type, "", weight))

        # Remove completely resolved weaknesses
        for error_type in hist_error_types:
            if error_type not in error_patterns or error_patterns[error_type] <= 0:
                if hasattr(self.l2, "set_error_fingerprint"):
                    await maybe_await(self.l2.set_error_fingerprint(user_id, error_type, 0, 0.0))

        # ── L2 mastered_expressions ──
        for expression, confidence in mastered_list:
            await maybe_await(self.l2.add_mastered(
                user_id=user_id,
                expression=expression,
                confidence=confidence
            ))

        # ── L2 correction_style ──
        correction_pref = analysis.get("correction_preference", "recast")
        if correction_pref and correction_pref != "recast":
            await self._update_style_parameter(user_id, "correction_style", correction_pref)

    async def _update_style_parameter(self, user_id: str, param_name: str, value):
        """
        Update a single style_parameter field in L2.
        Creates the row if it doesn't exist.
        """
        if not self.l2:
            return

        try:
            await maybe_await(self.l2.set_style_parameter(user_id, param_name, value))
        except Exception:
            pass  # style_parameters update is best-effort

    # ── Session Analysis ──────────────────────────────────

    def _analyze_session(self, user_messages: list[dict],
                         memory_ctx: Optional[dict] = None) -> dict:
        """Run all extraction analyses on the session."""
        all_text = " ".join([m.get("content", "") for m in user_messages])

        return {
            "cefr_estimate": self._estimate_cefr(all_text, memory_ctx),
            "topics_detected": self._extract_topics(all_text),
            "error_patterns": self._extract_error_patterns(user_messages),
            "mastered_expressions": self._extract_mastered_expressions(user_messages),
            "communication_style": self._analyze_communication_style(user_messages),
            "correction_preference": self._infer_correction_preference(user_messages),
            "english_ratio": self._calc_english_ratio(all_text),
            "message_count": len(user_messages),
            "avg_message_length": sum(len(m.get("content", "")) for m in user_messages) / max(len(user_messages), 1),
        }

    def _estimate_cefr(self, text: str, memory_ctx: Optional[dict] = None) -> tuple[str, float]:
        """CEFR estimate based on vocabulary, sentence complexity, and structure."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Factor 1: Vocabulary marker matches per level
        level_scores = {}
        for level, markers in self.CEFR_MARKERS.items():
            matches = sum(1 for m in markers if m in text_lower or m in words)
            level_scores[level] = matches

        # Find highest level with matches
        vocab_level = "A1"
        best_score = 0
        for level in ["C2", "C1", "B2", "B1", "A2", "A1"]:
            if level_scores.get(level, 0) > best_score:
                best_score = level_scores[level]
                vocab_level = level

        # Factor 2: Sentence complexity (avg words per sentence, subordinate clauses)
        level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        has_subordinate = bool(re.search(r'\b(because|although|if|when|while|since|that|which|who)\b', text_lower))
        has_complex_struct = bool(re.search(r'\b(however|nevertheless|furthermore|consequently|despite|in spite of)\b', text_lower))

        if avg_sentence_len > 20 and has_complex_struct:
            complexity_level = "B2"
        elif avg_sentence_len > 15 and has_subordinate:
            complexity_level = "B1"
        elif avg_sentence_len > 10:
            complexity_level = "A2"
        else:
            complexity_level = "A1"

        # Factor 3: Pick the higher of vocab and complexity estimates
        vocab_idx = level_order.index(vocab_level) if vocab_level in level_order else 0
        complexity_idx = level_order.index(complexity_level) if complexity_level in level_order else 0
        current_idx = max(vocab_idx, complexity_idx)

        # Smoothing: blend history CEFR and current CEFR
        hist_cefr = self._guess_current_cefr(memory_ctx)
        hist_idx = level_order.index(hist_cefr) if hist_cefr in level_order else 0

        total_sessions = memory_ctx.get("total_sessions", 0) if memory_ctx else 0
        if total_sessions <= 1:
            smoothed_idx = current_idx
        else:
            # 70% history weight, 30% current session weight to avoid fluctuations
            smoothed_idx = round(0.7 * hist_idx + 0.3 * current_idx)

        smoothed_idx = max(0, min(len(level_order) - 1, smoothed_idx))
        best_level = level_order[smoothed_idx]

        # Calculate confidence
        sample_confidence = min(len(words) / 200, 1.0)
        match_confidence = min(best_score / 5, 1.0)
        complexity_confidence = min(avg_sentence_len / 25, 1.0)
        confidence = (sample_confidence + match_confidence + complexity_confidence) / 3

        return best_level, round(confidence, 2)

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topics of interest from conversation content."""
        text_lower = text.lower()
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                topic_scores[topic] = score

        # Sort by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_topics[:5]]

    def _extract_error_patterns(self, user_messages: list[dict]) -> dict:
        """Extract recurring error patterns from user messages."""
        patterns = {}
        for msg in user_messages:
            content = msg.get("content", "")
            if not content:
                continue
            # Check for known patterns from the learning engine
            for pattern_name, pattern_regex in self._english_error_patterns().items():
                if re.search(pattern_regex, content, re.IGNORECASE):
                    patterns[pattern_name] = patterns.get(pattern_name, 0) + 1
        return patterns

    # ── Mastered Expression Detection ─────────────────────

    MASTERED_PATTERNS = {
        # A1/A2 基础语法标记
        "simple_past_correct": (r"\b(went|saw|bought|did|made|took|played|liked|loved|wanted|came|had|was|were)\b", 0.5, "simple_past_verbs"),
        "modal_verbs_correct": (r"\b(can|could|should|must|will|would)\s+[a-z]{2,}\b", 0.5, "modal_helper_verbs"),
        "simple_comparative": (r"\b(better|worse|bigger|smaller|older|younger|more\s+[a-z]+)\s+than\b", 0.55, "simple_comparisons"),
        # B1+ 连接词正确使用
        "however_clause": (r"\b[Hh]owever\s*,", 0.7, "however_connector"),
        "although_clause": (r"\b[Aa]lthough\s+\w+", 0.7, "although_subordinate"),
        "despite_phrase": (r"\b[Dd]espite\s+(the|my|his|her|our|their)\s+\w+", 0.75, "despite_phrase"),
        "in_spite_of": (r"\b[Ii]n\s+spite\s+of\b", 0.75, "in_spite_of"),
        # B2+ 复杂句型
        "present_perfect": (r"\b(have|has|had)\s+\w+ed\b", 0.6, "present_perfect"),
        "present_perfect_continuous": (r"\b(have|has)\s+been\s+\w+ing\b", 0.8, "present_perfect_continuous"),
        "past_perfect": (r"\bhad\s+\w+ed\b", 0.75, "past_perfect"),
        "passive_voice": (r"\b(is|are|was|were|been|being)\s+\w+ed\s+by\b", 0.7, "passive_voice"),
        "conditional_2nd": (r"\b[Ii]f\s+(I|we|they|he|she)\s+\w+ed\s*,\s+(I|we|they|he|she)\s+(would|could|might)\s+\w+", 0.85, "second_conditional"),
        "relative_clause": (r"\b(which|that|who|whom|whose)\s+\w+", 0.6, "relative_clause"),
        # C1+ 高级表达
        "nevertheless": (r"\b[Nn]evertheless\s*,", 0.85, "nevertheless"),
        "furthermore": (r"\b[Ff]urthermore\s*,", 0.85, "furthermore"),
        "consequently": (r"\b[Cc]onsequently\s*,", 0.85, "consequently"),
        "not_only_but": (r"\b[Nn]ot\s+only\s+.+\s+but\s+also\b", 0.85, "not_only_but_also"),
        "would_rather": (r"\bwould\s+rather\s+\w+", 0.8, "would_rather"),
        "used_to": (r"\bused\s+to\s+\w+", 0.65, "used_to"),
        # 自然口语流利标记
        "fluency_marker": (r"\b(you know|I mean|sort of|kind of|to be honest|frankly)\b", 0.6, "fluency_marker"),
        "tag_question": (r"\b(isn'?t|aren'?t|don'?t|doesn'?t|didn'?t|won'?t|can'?t)\s+(it|he|she|they|we|you)\b", 0.7, "tag_question"),
        "phrasal_verb": (r"\b(figure out|look forward to|catch up|give up|take off|put off|turn out|come across)\b", 0.7, "phrasal_verb"),
    }

    def _extract_mastered_expressions(self, user_messages: list[dict]) -> list[tuple[str, float]]:
        """Detect correctly used complex structures, return (label, confidence) pairs."""
        mastered = []
        seen = set()
        for msg in user_messages:
            content = msg.get("content", "")
            if not content or len(content) < 5:
                continue
            for label, (pattern, confidence, display) in self.MASTERED_PATTERNS.items():
                if label in seen:
                    continue
                if re.search(pattern, content, re.IGNORECASE):
                    mastered.append((display, confidence))
                    seen.add(label)
        return mastered

    def _english_error_patterns(self) -> dict:
        """English error patterns common for Chinese learners."""
        return {
            # 冠词缺失
            "article_missing": r"\b(I|we|they|he|she)\s+(want|need|have|like|go|eat|drink|play|buy|see|read)\s+([a-z]+)",
            "article_missing_be": r"\b(I|he|she|this|that)\s+(am|is|are|was|were)\s+(student|teacher|doctor|engineer|nurse)\b",
            # 复数缺失
            "plural_missing": r"\b(many|some|two|three|several|few|a lot of)\s+([a-z]+[^s])\b",
            "plural_missing_be": r"\b(one|two|three|four|five)\s+(year|month|day|week|hour|minute|book|apple|friend)s?\s+(old|ago|later)\b",
            # 主谓一致
            "subject_verb": r"\b(I|we|they)\s+(is|was|has|does)\b",
            "subject_verb_he": r"\b(he|she|it)\s+(are|were|have|do)\b",
            # 时态错误
            "tense_past": r"\b(yesterday|last (week|month|year)|ago)\b.*\b(I|we|they|he|she)\s+(go|come|eat|see|buy|do|make|take)\b",
            "tense_present_perfect": r"\b(already|just|yet|ever|never)\b.*\b(I|we|they|he|she)\s+(went|came|ate|saw|bought|did|made|took)\b",
            "tense_future": r"\b(tomorrow|next (week|month|year)|soon|later)\b.*\b(I|we|they|he|she)\s+(went|came|ate|saw|bought|did|made|took)\b",
            # 词汇/直译
            "literal_translation": r"\b(cold coffee|open the light|close the light|no can|very convenient|play computer|play phone|learn English very good|open TV|close TV)\b",
            "chinglish_very": r"\bvery\s+(like|want|need|happy|sad|good|bad|big|small)\b(?!\s+(much|well|good|bad))",
            "chinglish_no_can": r"\b(no can|no do|no have|no know|no understand|no like)\b",
            # 词序错误
            "word_order": r"\b(very like|very want|not can|not have|can not|all are|all is)\b",
            "word_order_adverb": r"\b(I|we|they|he|she)\s+(very|quite|so)\s+(much|good|happy|tired|busy)\b",
            # 介词错误
            "prep_wrong": r"\b(good in|good at English|married with|different with| angry to|angry with you for|depend on you for|discuss about|explain about|ask about you for)\b",
            # 缺少 be 动词
            "missing_be": r"\b(I|we|they|he|she)\s+(very|quite|so|too|really)\s+(happy|sad|tired|busy|hungry|thirsty|good|bad|beautiful|handsome)\b",
            # 双重否定
            "double_negative": r"\b(don'?t|doesn'?t|didn'?t|not|never)\s+\b\w*\s*(no|not|never|nothing|nobody|none)\b",
            # 动名词/不定式混淆
            "gerund_infinitive": r"\b(enjoy|finish|practice|mind|avoid)\s+to\s+\w+",
            "gerund_infinitive_2": r"\b(want|decide|hope|plan|agree|refuse|promise)\s+\w+ing\b",
            # 条件句错误
            "conditional_error": r"\bif\s+(I|we|they|he|she)\s+(will|would|can|could|shall|should)\s+\w+",
            # 比较级错误
            "comparative_error": r"\b(more|most)\s+(better|worse|bigger|smaller|older|younger|easier|harder|faster|slower)\b",
        }

    @classmethod
    def _has_style_word(cls, text: str, dimension: str) -> bool:
        """按词边界匹配该维度的标记词（子串匹配见 COMM_STYLE_EN_WORDS 注释）。"""
        return any(
            re.search(rf"\b{re.escape(word)}\b", text)
            for word in cls.COMM_STYLE_EN_WORDS.get(dimension, ())
        )

    def _analyze_communication_style(self, user_messages: list[dict]) -> dict:
        """Extract communication style signals from user messages.

        口径见 COMM_STYLE_EN_WORDS：软适配信号，四维计数，词边界匹配。
        """
        style = {"direct": 0, "expressive": 0, "analytical": 0, "hesitant": 0}

        for msg in user_messages:
            content = msg.get("content", "") or ""
            text = content.lower()

            # Direct: short imperative-like messages. 全角问号同样算提问
            # ——豆包 ASR 对中文吐全角（plans/0919-01 §0.1），只认 "?" 会把
            # 中文提问误判成"直接"。
            if len(content) < 20 and "?" not in content and "？" not in content:
                style["direct"] += 1
            # Expressive: emotional language, exclamation marks, repetition.
            # 标点中英皆认，不需要分词。
            if ("!" in content or "！" in content
                    or self._has_style_word(text, "expressive")):
                style["expressive"] += 1
            # Analytical: longer messages with reasoning
            if len(content) > 50 and self._has_style_word(text, "analytical"):
                style["analytical"] += 1
            # Hesitant: hedging language, incomplete sentences
            if self._has_style_word(text, "hesitant"):
                style["hesitant"] += 1

        return style

    @classmethod
    def _distill_communication_style(cls, style: dict, message_count: int) -> str:
        """把四维计数压成一个稳定的短标签（如 "analytical+hesitant"）。

        无足够信号时返回空串——宁可不适配，也不要拿噪声去引导模型。
        标签只说"怎么说"，不说"是什么样的人"（plans/0919-03 §4.6 分层注意）。
        """
        if not style or not message_count or message_count < cls.COMM_STYLE_MIN_MESSAGES:
            return ""
        ranked = sorted(
            ((name, count) for name, count in style.items() if count > 0),
            key=lambda kv: (-kv[1], kv[0]),
        )
        picked = [
            name for name, count in ranked
            if count >= message_count * cls.COMM_STYLE_MIN_RATIO
        ]
        return "+".join(picked[:cls.COMM_STYLE_MAX_LABELS])

    def _infer_correction_preference(self, user_messages: list[dict]) -> str:
        """
        Infer how the user prefers to be corrected based on their reactions.
        Options: recast, explicit, hint, no_correction
        """
        for msg in user_messages:
            text = msg.get("content", "").lower()
            # User asks for explicit correction
            if any(w in text for w in ["explain", "why", "what's wrong", "correct", "错误", "纠正"]):
                return "explicit"
            # User seems frustrated with too much correction
            if any(w in text for w in ["don't correct", "别纠正", "不要纠正", "just talk", "自然"]):
                return "no_correction"
            # User asks for hints
            if any(w in text for w in ["hint", "clue", "提示", "暗示"]):
                return "hint"

        return "recast"

    def _calc_english_ratio(self, text: str) -> float:
        """Calculate the ratio of English words to total content."""
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if not words and not chinese_chars:
            return 0.5
        total = len(words) + len(chinese_chars)
        return round(len(words) / total, 2) if total > 0 else 0.5

    # ── Helper Methods ────────────────────────────────────

    def _guess_current_cefr(self, memory_ctx: dict) -> str:
        """Infer current CEFR level from historical session archives."""
        if not memory_ctx or not memory_ctx.get("recent_sessions"):
            return "A1"

        # Use most recent session's cefr_after
        for session in memory_ctx["recent_sessions"]:
            if session.get("cefr_after"):
                return session["cefr_after"]

        return "A1"

    def _weighted_cefr_merge(self, results: list[dict]) -> tuple[str, float]:
        """Merge CEFR estimates weighted by message count."""
        level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        weighted_levels = {}

        for r in results:
            level, conf = r.get("cefr_estimate", ("A1", 0.5))
            msg_count = r.get("message_count", 1)
            idx = level_order.index(level) if level in level_order else 0
            weighted_levels[idx] = weighted_levels.get(idx, 0) + conf * msg_count

        if not weighted_levels:
            return "A1", 0.5

        best_idx = max(weighted_levels, key=weighted_levels.get)
        total_weight = sum(weighted_levels.values())
        confidence = weighted_levels[best_idx] / total_weight if total_weight > 0 else 0.5

        return level_order[best_idx], round(confidence, 2)

    ARCHIVE_SUMMARY_SYSTEM = (
        "你是会话归档员。把这段语言学习对话压缩成一条中文摘要。\n"
        "摘要供日后回忆检索用，必须包含：聊了什么主题、发生了哪些具体事件、"
        "提到的具体名词（食物/地点/人名/作品/数字等）、有没有没聊完的事。\n"
        "第一行写标签：<上位词/主题>：<具体内容>；例如「新爱好：吉他（手指疼，和弦还没顺）」"
        "「麻烦事：房租（房东最终同意不涨）」「小家伙：猫 年糕（术后恢复，昨天开始吃东西）」"
        "「朋友：杭州（周末来，计划本帮菜，还没定餐厅）」。标签要写**用户可能说的叫法**，"
        "同一件事可以带一个别名，例如「学吉他（新爱好）：手指疼、和弦弹不好」"
        "「房租（麻烦事）：房东最终同意不涨」。标签只写可能的候选，不要编。\n"
        "最后一条用户消息里的事要写到哪一步或结果（例如“最后房东同意不涨”“其中一家约了下周面试”），"
        "不要只写开头。\n"
        "一到两句话、140 字以内、第三人称。不同的人要分开写清，不要混成一个人。\n"
        "只输出摘要本身，不要前缀、不要解释。"
    )

    # §8.28：同一批事实，换个口吻写。材料决定语气——照念档案体就会变成逐条汇报。
    ARCHIVE_SUMMARY_SYSTEM_ALICE = (
        "你在替一个叫 Alice 的人写她**自己的记忆**，不是写会议纪要。\n"
        "用她的口吻、第一人称（「我」）记下这段对话里**值得记住的那几件事**：\n"
        "对方经历了什么、提到过谁和哪儿、有什么没说完。\n"
        "像她事后在心里嘀咕那样写：两三句，口语，能带一点她自己的态度（好笑、心疼、好奇）。\n"
        "事实必须准确，名词（食物/地点/人名/数字）不能改；但**不要列举、不要分条、不要写成报告**。\n"
        "只输出这段记忆本身，不要前缀、不要解释。"
    )

    def _archive_summary_prompt(self) -> str:
        try:
            from config import ARCHIVE_SUMMARY_VOICE as _voice
        except Exception:
            _voice = "archive"
        # 检索要的是"能被匹配到的名词"，人格要的是"念出来像人"；两者都保留，只换语气。
        return (self.ARCHIVE_SUMMARY_SYSTEM_ALICE
                if str(_voice).lower() == "alice"
                else self.ARCHIVE_SUMMARY_SYSTEM)

    CLOSENESS_SYSTEM = (
        "你在给一段对话打一个**关系变化**的分。用户的原话是："
        "「关系不是单纯聊得多久就能定下来，同事天天说也不见得关系有多好，以及关系是会变化的」。\n"
        "所以只按**内容**判断，不看聊了多少轮：\n"
        "  +1：用户主动分享了私人的、不只是寒暄的事（自己的处境、难处、在意的人或事）；\n"
        "  +1：用户明确表达了关心/依赖/信任（担心累不累、说只跟你讲这些）；\n"
        "  −1：用户明确拉开距离（不想说、让你别问、觉得你越界、说你不像人/别装了）；\n"
        "   0：普通闲聊、只有寒暄或事务性内容——**聊多久都是 0**。\n"
        "输出 JSON：{\"delta\": -1|0|1, \"reason\": \"一句话，引用对话里的依据\"}。\n"
        "证据不足一律 0；不要因为聊得投机就给分。只输出 JSON。"
    )

    async def _closeness_deltas(self, messages: list[dict]) -> list[dict]:
        """把一场会话切成若干窗口分别判分（plans/1017-02 §8.38）。

        为什么不是一场一个分：真实 330 条那场里**既有交心也有翻脸**，
        一个净分表达不了"先近后远"；208 条那场只判出"聊天气"。
        规则：≤80 条只判一次；否则按 ≤120 条切成最多 3 段（头/中/尾），每段各判一次。
        """
        items = list(messages or [])
        if len(items) <= 80:
            single = await self._closeness_delta(items)
            return [single] if single else []
        size = 120
        windows = [items[i:i + size] for i in range(0, len(items), size)]
        if len(windows) > 3:
            windows = [windows[0], windows[len(windows) // 2], windows[-1]]
        results = []
        for window in windows:
            if len(window) < 6:
                continue
            result = await self._closeness_delta(window)
            if result and result.get("delta"):
                results.append(result)
        return results

    async def _closeness_delta(self, messages: list[dict]) -> dict:
        """会话结束给一次关系变化打分（plans/1017-02 §8.38）。

        ⛔ 不进语音热路径（只在会话末跑一次）；失败返回 {}，调用方不写任何东西。
        """
        try:
            from config import RELATIONSHIP_CLOSENESS_FROM_CONTENT_ENABLED as _on
        except Exception:
            _on = False
        if not _on:
            return {}
        lines = []
        window = list(messages or [])
        # §8.38 实测教训：只取最后 40 条时，一场 208 条的对话被判成"只聊天气"——
        # 前面的内容全被丢了。改成**头尾取样**（开头 12＋结尾 28），和归档摘要同口径。
        if len(window) > 40:
            window = window[:12] + window[-28:]
        for message in window:
            role = "用户" if str(message.get("role")) == "user" else "AI"
            text = " ".join(str(message.get("content") or "").split())[:300]
            if text:
                lines.append(f"{role}: {text}")
        if len(lines) < 3:
            return {}
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            # ⛔ 必须带模型名：`thinking_disabled_extra_body(None)` 认不出供应商 ⇒
            # 关不掉推理 ⇒ 每次会话末都可能等十几秒（§8.24 实测 0 字/9.1s 同一坑）。
            content = await robust_llm_client.chat_completion(
                [
                    {"role": "system", "content": self.CLOSENESS_SYSTEM},
                    {"role": "user", "content": "对话记录：\n" + "\n".join(lines)},
                ],
                model=_model, temperature=0.0, max_tokens=300,
                extra_body=thinking_disabled_extra_body(_model),
            )
            payload = json.loads(str(content or "{}"))
            delta = max(-1, min(1, int(payload.get("delta") or 0)))
            return {"delta": delta,
                    "reason": str(payload.get("reason") or "")[:200]}
        except Exception:
            return {}

    SELF_FACT_SYSTEM = (
        "你在一段对话之后，整理**这个 AI 自己**说过的话里，哪些是关于她本人的、能长期成立的。\n"
        "只收三类：preference（喜欢/讨厌什么）、trait（她是什么样的人）、stance（她怎么看事）。\n"
        "硬规则：\n"
        "  1. 必须是她**自己**说的话，不是用户说的，也不是关于用户的；\n"
        "  2. 必须**长期成立**——「今天开了个会」「刚才有点累」这类一次性的不要；\n"
        "  3. `quote` 必须是她说过的**原话片段**（≥6 字，逐字），能对回对话；\n"
        "  4. 已经知道的、重复的不要；最多 3 条，证据不足就返回空数组；\n"
        "  5. `text` **必须用她说这句话时用的语言**（她说英文就写英文）。"
        "自述日后会被她自然提起，语言不一致会让她在英文回复里蹦出中文。\n"
        "输出 JSON 数组：[{\"facet\":\"slow-paced\",\"kind\":\"trait\","
        "\"text\":\"讨厌被催，宁可自己慢一点\",\"quote\":\"原话片段\",\"echo\":false}]\n"
        "  6. `text` 必须是**第一人称**（I / my / 我）。写成「She likes…」这种关于她的叙述一律不合格。\n"
        "  6b. **只收她主动提起的**：如果这句话是在**接他刚说的同一件事**（同意、共鸣、同一个立场），一律不要——哪怕它看起来像她的偏好。判据：没有他那句话，她还会主动说这句吗？会 ⇒ 收；不会 ⇒ 丢。\n"
        "  7. `facet` 用短横线小写英文，**尽量复用这几个**："
        "morning-routine / sleep / coffee / food-preference / work-rhythm / energy / "
        "social / taste / mood / values；别拿整句当 facet。\n"
        "`echo` 表示：这句话是在**顺着用户刚说的同一件事往下接**（映照），"
        "还是她**自己独立的**偏好/习惯。\n"
        "  例：用户说「我今天开了个会」，她答「I had one today too」→ echo=true；\n"
        "  例：用户说「晚上吃什么」，她答「I don't do well with spicy food」→ echo=false。\n"
        "注意跨语言也要判：用户说中文、她答英文，同样按**意思**判。\n"
        "示例：\n"
        "  输入：用户「周末想点个外卖」／AI「I'd skip the Sichuan place. "
        "Spicy food and I don't get along at all.」\n"
        "  输出：[{\"facet\":\"food-preference\",\"kind\":\"preference\","
        "\"text\":\"Spicy food and I don't get along\","
        "\"quote\":\"Spicy food and I don't get along at all\",\"echo\":false}]\n"
        "  输入：用户「今天开会了吗」／AI「I had a meeting today that ran forty minutes.」\n"
        "  输出：[]（一次性，不算）\n"
        "⚠ 只要她表达了对某类事物的喜好／厌恶／习惯，就**应该**抽出来；"
        "不要因为句子短、话题小就返回空数组。\n"
        "只输出 JSON。"
    )

    OPEN_THREAD_SYSTEM = (
        "你在对话结束后，找出**用户提过、但还没有下文的事**——只留下一盘还能自然接上的。\n"
        "只收三类：① 他打算做/正在犹豫的事（换工作、要不要去、周末安排）；\n"
        "② 他在等结果、担心的事（面试、体检、等回复）；③ 说了开头没说完的事。\n"
        "硬规则：\n"
        "  1. 必须是**用户**的事，不是 AI 自己的；\n"
        "  2. 必须**还没有结果**——用户已经说清结局的，放进 resolved；\n"
        "  3. `topic` 用 4-10 个字的短语（跟用户原话的语言），`detail` 一句话补背景；\n"
        "  4. `quote` 是用户原话片段（≥4 字，逐字）；\n"
        "  5. 最多 3 条；没有就返回空数组。宁缺毋滥，寒暄和小事不要记；\n"
        "  7. **她问过、而用户没答的问题**：如果 AI 在最后问了用户一个具体问题，"
        "而用户之后没有回答（只是嗯了一声或换了话题），就把那一句**原样**抄进 "
        "`unanswered_question`；用户答过、或问题只是寒暄（如「最近怎么样」）就留空；同时给 `unanswered_topic`：这件事的**双语**短标签（4-10 字，先中文后英文，用 | 分隔）。\n"
        "  6. **有时间的事必须给时间**：`due_hint` 抄用户原话（如「后天下午」「周日」），"
        "`due_at` 换算成绝对日期（按北京时间；今天是 {today}）；没有时间就都留空。\n"
        "输出 JSON：{\"open\":[{\"topic\":\"换工作\",\"detail\":\"在犹豫要不要跳槽\","
        "\"quote\":\"我最近在想换工作\",\"due_hint\":\"\",\"due_at\":\"\"}],"
        "\"resolved\":[\"体检\"],\"unanswered_question\":\"\",\"unanswered_topic\":\"\"}\n"
        "只输出 JSON。"
    )

    async def _distill_open_threads(
        self, messages: list[dict]
    ) -> tuple[list[dict], list[str], dict]:
        """抽出"他提过、还没下文的事"（§8.56）＋"她问过、他没答的问题"（§8.76）。

        返回 (要存的线程, 要关掉的话题, {"text": 原问句, "topic": 双语短标签})。
        """
        pairs = []
        for message in list(messages or [])[-40:]:
            text = " ".join(str(message.get("content") or "").split())[:300]
            if not text:
                continue
            label = "用户" if str(message.get("role")) == "user" else "AI"
            pairs.append(f"{label}: {text}")
        if not pairs:
            return [], [], {}
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            raw = await robust_llm_client.chat_completion(
                [
                    # `{today}` 必须换成真实日期，否则"后天"没法换算成绝对日期。
                    {"role": "system", "content": self.OPEN_THREAD_SYSTEM.replace(
                        "{today}", datetime.now().strftime("%Y-%m-%d"))},
                    {"role": "user", "content": "对话：\n" + "\n".join(pairs)},
                ],
                model=_model, temperature=0.0, max_tokens=400,
                extra_body=thinking_disabled_extra_body(_model),
            )
            parsed = json.loads(str(raw or "{}"))
        except Exception as exc:
            import os as _os
            if _os.getenv("RELATIONSHIP_DEBUG_STAGE"):
                print(f"[DBG] open-thread distill failed: {type(exc).__name__}: {exc}")
            return [], [], {}
        if not isinstance(parsed, dict):
            return [], [], {}
        opened: list[dict] = []
        for item in (parsed.get("open") or []):
            if not isinstance(item, dict):
                continue
            topic = " ".join(str(item.get("topic") or "").split())[:80]
            if not topic:
                continue
            opened.append({
                "topic": topic,
                "detail": " ".join(str(item.get("detail") or "").split())[:240],
                "quote": " ".join(str(item.get("quote") or "").split())[:200],
                "due_hint": " ".join(str(item.get("due_hint") or "").split())[:60],
                # ⛔ 只认 YYYY-MM-DD；模型给了别的一律丢空（宁可不到点提醒，也别按乱日期触发）。
                "due_at": (lambda raw: raw if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw) else "")(
                    " ".join(str(item.get("due_at") or "").split())[:10]),
            })
        resolved = [" ".join(str(x).split())[:80] for x in (parsed.get("resolved") or [])
                    if str(x).strip()]
        unanswered = {
            "text": " ".join(str(parsed.get("unanswered_question") or "").split())[:160],
            "topic": " ".join(str(parsed.get("unanswered_topic") or "").split())[:80],
        }
        return opened[:3], resolved[:3], unanswered

    OUTBOX_SYSTEM = (
        "你要给一个朋友留一条消息——他明天可能不在，回来时会看到。\n"
        "你**不是**提醒服务：不要复述安排细节（时间、票、地点、待办），不要说「别忘了」。\n"
        "你要表达的只是：你记得这件事，你在意它对他意味着什么。\n"
        "规则：英文；一到两句；像朋友随口留的一句；不解释、不列点、最多一个问句。\n"
        "不要编造他没说过的事。\n"
        "反例（像客服）：\n"
        "  Hey, the show is Sunday night and your ticket is already sorted.\n"
        "  Just a reminder to bring your tickets tomorrow.\n"
        "正例（像朋友）：\n"
        "  Tomorrow's the thing you've been waiting for, right? Hope it's as good as you expect.\n"
        "  I keep thinking about tomorrow for you. Tell me how it goes after.\n"
        "只输出消息本身，不要引号。"
    )

    async def _enqueue_due_outbox(self, user_id: str, session_id: str = "") -> str:
        """把"明天/后天到期的事"提前写成一条留言（§8.60）。

        为什么在**会话末**做：不需要常驻定时器，也不需要系统推送权限；
        用户下次打开时，这条消息已经在箱子里了。
        """
        from datetime import date, timedelta

        rows = await self.l2.list_open_threads(user_id, limit=10)
        today = datetime.now(timezone.utc).date()
        soon = [r for r in rows if str(r.get("due_at") or "")[:10] in {
            (today + timedelta(days=1)).isoformat(),
            (today + timedelta(days=2)).isoformat(),
        }]
        if not soon:
            return ""
        thread = soon[0]
        body = f"{thread.get('topic')} — {thread.get('detail') or ''}".strip(" —")
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            text = await robust_llm_client.chat_completion(
                [
                    {"role": "system", "content": self.OUTBOX_SYSTEM},
                    {"role": "user", "content": "明天这件事：" + body[:160]},
                ],
                model=_model, temperature=0.7, max_tokens=120,
                extra_body=thinking_disabled_extra_body(_model),
            )
        except Exception as exc:
            print(f"[outbox] compose failed: {type(exc).__name__}")
            return ""
        msg = " ".join(str(text or "").strip().strip('"').split())[:400]
        if len(msg) < 8:
            return ""
        return await self.l2.enqueue_outbox(
            user_id, msg, reason="due:" + str(thread.get("topic") or "")[:40],
            conv_id=str(session_id or ""), due_at=str(thread.get("due_at") or ""),
        )

    async def _distill_self_facts(self, messages: list[dict],
                                  existing_facets: list[str] | None = None,
                                  existing_facts: list[dict] | None = None,
                                  source_conv_id: str = "") -> list[dict]:
        """把她自己说过的话沉淀成**候选事实**（plans/1017-02 §8.41）。

        为什么要这一步：契约能调语气，但**她讲自己的事**受限于材料——目前只有 5 条底子。
        让她自己说过的稳定偏好/态度沉淀下来，人格才有积累。三道闸在代码里做，不靠模型自觉：
          ① quote 必须逐字出现在**她自己的**消息里；② 不许涉及"你/your"（那是关于用户的）；
          ③ 不许是"今天/刚才"这类一次性事件。
        落库用 `status=candidate`，跨会话重复由既有 `arbitrate_version_change` 升 active。
        """
        try:
            from config import PERSONA_SELF_FACT_ENABLED as _on
        except Exception:
            _on = False
        if not _on:
            return []
        said = [str(m.get("content") or "") for m in (messages or [])
                if str(m.get("role")) != "user" and str(m.get("content") or "").strip()]
        if not said:
            return []
        # 要把**用户那一边**也带上，模型才判得出"她在接话"；只喂她自己的话，
        # 跨语言的映照根本看不出来（§8.41 盲区）。
        pairs = []
        for message in list(messages or [])[-40:]:
            text = " ".join(str(message.get("content") or "").split())[:300]
            if not text:
                continue
            label = "用户" if str(message.get("role")) == "user" else "AI"
            pairs.append(f"{label}: {text}")
        transcript = "\n".join(pairs)
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            raw = await robust_llm_client.chat_completion(
                [
                    {"role": "system", "content": self.SELF_FACT_SYSTEM},
                    {"role": "user", "content":
                        (("已有的 facet（**同一件事必须沿用已有的那个 slug**，"
                          "不要换个说法新建）：" + "、".join(existing_facets) + "\n\n")
                         if existing_facets else "")
                        + "她说过的话：\n" + transcript},
                ],
                model=_model, temperature=0.0, max_tokens=600,
                extra_body=thinking_disabled_extra_body(_model),
            )
            payload = json.loads(str(raw or "[]"))
        except Exception as exc:
            import os as _os
            if _os.getenv("RELATIONSHIP_DEBUG_STAGE"):
                print(f"[DBG] self-fact distill failed: {type(exc).__name__}: {exc}")
            return []
        if not isinstance(payload, list):
            return []
        blob = "\n".join(said)
        user_blob = "\n".join(str(m.get("content") or "") for m in (messages or [])
                             if str(m.get("role")) == "user")

        def _bigrams(text: str) -> set[str]:
            clean = "".join(ch for ch in text.lower() if ch.isalnum())
            return {clean[i:i + 2] for i in range(len(clean) - 1)}

        user_grams = _bigrams(user_blob)
        # 兜底句不是她"说的话"：生成失败/超时后系统塞的那句
        # （`response_failure_generic_fallback` 等），不该沉淀成她的人格。
        fallback_lines = (
            "got away from me for a second",
            "couldn't put that together",
        )
        episodic = ("今天", "刚才", "刚刚", "昨晚", "今晚", "today", "just now",
                    "tonight", "yesterday", "earlier")
        rows = []
        drops = {"空/过短": 0, "原话不匹配": 0, "兜底句": 0, "关于用户": 0,
                 "一次性": 0, "回音": 0, "第三人称": 0,
                 "与已有事实雷同": 0, "本盘已记过": 0}
        # §8.58 判重（有依据后才加）：实测同一盘里她被写了两遍（两条 1.00 相似、
        # 相隔 3 分钟），同会话同 facet 另有 9 条冗余。判重只为掐掉**冗余写入**，
        # 跨会话同 facet 必须留着——那是"候选→active"升格唯一的证据。
        _existing = [f for f in (existing_facts or []) if isinstance(f, dict)]

        def _norm(text: str) -> str:
            return "".join(ch for ch in str(text or "").lower() if ch.isalnum())

        def _preceding_user(msgs: list[dict], quote: str) -> str:
            """找这句话对应的"他那句"：她的消息里含 quote 时，取它前面最近的一条用户消息。"""
            last = ""
            head = quote[:24]
            for message in msgs or []:
                text = " ".join(str(message.get("content") or "").split())
                if str(message.get("role")) == "user":
                    last = text
                elif head and head in text:
                    return last[:300]
            return last[:300]

        _existing_norm = [(_norm(f.get("value_text")), f) for f in _existing]
        for item in payload[:5]:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("text") or "").split())[:200]
            quote = " ".join(str(item.get("quote") or "").split())[:200]
            if len(text) < 4 or len(quote) < 6:
                drops["空/过短"] += 1
                continue
            if quote not in blob:                     # ① 原话对不上 ⇒ 丢
                drops["原话不匹配"] += 1
                continue
            if any(line in quote.lower() for line in fallback_lines):  # 兜底句 ⇒ 丢
                drops["兜底句"] += 1
                continue
            if "你" in text or "you" in text.lower():  # ② 说成关于用户的 ⇒ 丢
                drops["关于用户"] += 1
                continue
            # ③ 一次性：原话和提炼后都要查——只查 text 会漏（原话里有 today，
            # 但提炼成"开会经常没结论"就看不出来了）。
            haystack = f"{text} {quote}".lower()
            if any(word in haystack for word in episodic):  # ③ 一次性 ⇒ 丢
                drops["一次性"] += 1
                continue
            # ④ 回音：两种判法并用——
            #    (a) 模型自己标 `echo`（跨语言也生效，这是原来那个字面闸的盲区）；
            #    (b) 同语言时再用字符二元组兜一道（便宜、不依赖模型）。
            if item.get("echo") is True:
                drops["回音"] += 1
                continue
            grams = _bigrams(quote)
            if grams and user_grams:
                overlap = len(grams & user_grams) / max(1, len(grams))
                if overlap >= 0.5:
                    drops["回音"] += 1
                    continue
            _facet = str(item.get("facet") or text)[:160]
            _value = _match_language(text, quote)
            # §8.91b：值必须是她的**自述口吻**。实测出过 "She's slow in the mornings…"
            # 被当她的自述存进去（和叙事人称错位同一个病），这种一律不收。
            if re.match(r"^(she|he|her|his|alice)", _value.strip(), re.I) or                     _value.lstrip().startswith(("她", "他")):
                drops["第三人称"] += 1
                continue
            # §8.58 判重（两条，都有实测依据）：
            #  ① 与已有事实文本雷同（实测两对 1.00 相似，相隔 3 分钟写了两遍）；
            #  ② 本盘同 facet 已经写过（实测同会话内 9 条冗余）——
            #     ⛔ 只掐"同一盘内"，跨会话同 facet 是升格证据，必须留。
            _nv = _norm(_value)
            if _nv and any(
                len(other) >= 12 and min(len(_nv), len(other)) >= 12
                and SequenceMatcher(None, _nv, other).ratio() >= 0.90
                for other, _ in _existing_norm
            ):
                drops["与已有事实雷同"] += 1
                continue
            if source_conv_id and any(
                str(f.get("facet_key") or "") == _facet
                and str(f.get("source_conv_id") or "") == source_conv_id
                for _, f in _existing_norm
            ):
                drops["本盘已记过"] += 1
                continue
            rows.append({
                "kind": str(item.get("kind") or "trait")[:20],
                "facet_key": _facet,
                "value_text": _value,
                # §8.91d：回音判据要"他说了什么"——这句话前面最近的一条用户消息
                "user_msg": _preceding_user(messages, quote),
                "quote": quote,
            })
        rows = rows[:3]
        # 二审：换一套问法独立判一次，两次都说是她的才留（§8.41）。
        # 失败返回 None ⇒ 不做二次收紧，也不放宽（按原样留，候选态仍是兜底）。
        if rows:
            keep = await self._verify_self_facts(rows, messages)
            if keep is not None:
                dropped = len(rows) - len(keep)
                rows = [rows[index] for index in keep]
                if dropped:
                    print(f"[self-fact] 二审丢掉 {dropped} 条")
        if rows:
            print(f"[self-fact] 留 {len(rows)} 条："
                  + "；".join(r["value_text"][:24] for r in rows))
        elif any(drops.values()):
            reason = "／".join(f"{key} {value}" for key, value in drops.items() if value)
            print(f"[self-fact] 全被拦：{reason}")
        return rows

    VERIFY_SYSTEM = "".join((
        "判断**是不是同一个属性**：用户那句和她这句，是不是在说同一件事、同一个话题面。\n",
        "只判属性，不管对错、不管像不像、也不管谁更合理。\n",
        "用户「我夏天之前不开空调，怕吹」／她「I'd rather be a little warm than sit in a blast of cold air」\n",
        "⇒ same=true（都在说吹冷风这件事）\n",
        "用户「今天好累」／她「I've been learning guitar, my fingers are killing me」\n",
        "⇒ same=false（他在说累，她在说吉他，是两个属性）\n",
        "用户说中文、她答英文也按**意思**判。\n",
        "输出 JSON 数组：[{\"i\":1,\"same\":true},{\"i\":2,\"same\":false}]。只输出 JSON。",
    ))

    ECHO_CLASSIFY_SYSTEM = "".join((
        "判断这段对话里她这句话属于哪一类：\n",
        "  A = 他先说了**他自己的事**，她跟着说同一件（附和/共鸣）⇒ 不是她的事实；\n",
        "  B = 他**在问她**，她在回答 ⇒ 是她的事实；\n",
        "  C = 她自己主动提起（他没提过这件事）⇒ 是她的事实。\n",
        "只判这一件事，不管像不像、也不管说得对不对。\n",
        "输出 JSON：[{\"i\":1,\"kind\":\"A\"},{\"i\":2,\"kind\":\"B\"}]。只输出 JSON。",
    ))

    async def _review_echo_disclosures(self, user_id: str, session_id: str) -> int:
        """会话末复核：**每轮规则抽取**写下的自述里，哪些其实是"接他刚说的那件事"（§8.91c）。

        为什么要单独一道：`agent/tutor_agent.py` 那条路（写 `assistant_self_disclosure`）用的
        `is_recalled_persona_echo` 只比"她这句 vs 她以前说过的"，**不看用户当前说了什么**
        ⇒ "他先说不开空调、她说我宁愿热点"会被当成新事实，一天下来把她长成只会附和的自己。

        做法：拿库里已经存着的 `interaction_context`（=当轮用户原话）和 `value_text` 成对，
        用**已标定**的"同一属性"判据（接话 3/3 判出、自己说起 0/3 误判）复核；
        判为回音的**降级为 superseded，不删除**。
        成本：会话末最多 1 次调用；轮次路径 0。
        """
        if not session_id or not hasattr(self.l2, "list_companion_persona_facts"):
            return 0
        # ⚠ 每轮自述是**延后落库**的：会话末立刻查经常 rows=0（实测）。
        # 所以空的时候等 2 秒重试一次；仍空就放弃（下一盘再补）。
        rows: list[dict] = []
        for attempt in range(2):
            try:
                rows = await self.l2.list_companion_persona_facts(user_id, limit=200) or []
            except Exception as exc:
                return 0
            if rows:
                break
            if attempt == 0:
                import asyncio as _asyncio
                await _asyncio.sleep(2.0)
        try:
            from agent.life_line import asked_about_her
        except Exception:  # pragma: no cover
            def asked_about_her(_text: str) -> bool:  # type: ignore
                return False
        pending = []
        for row in rows:
            if str(row.get("status")) != "active":
                continue
            if str(row.get("source_kind")) != "assistant_self_disclosure":
                continue
            if str(row.get("source_conv_id") or "") != str(session_id):
                continue
            cue = str(row.get("interaction_context") or "").strip()
            value = str(row.get("value_text") or "").strip()
            if not cue or len(value) < 8:
                continue
            # ⚠ 不再用"他那句含你/you 就跳过"的粗规则：三分类判据自己会判 B（他在问她）。
            # 只跳过明显空白的行。
            pending.append((str(row.get("fact_id") or ""), cue, value))
        if not pending:
            return 0
        listing = "\n".join(
            f"{n}. 用户：{cue[:160]}\n   她：{value[:160]}"
            for n, (_, cue, value) in enumerate(pending, 1))
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            raw = await robust_llm_client.chat_completion(
                [
                    {"role": "system", "content": self.ECHO_CLASSIFY_SYSTEM},
                    {"role": "user", "content": listing},
                ],
                model=_model, temperature=0.0, max_tokens=200,
                extra_body=thinking_disabled_extra_body(_model),
            )
            picked = json.loads(str(raw or "[]"))
            if not isinstance(picked, list):
                print(f"[DBG] echo-review parse: not a list: {str(raw)[:120]}", flush=True)
                return 0
        except Exception as exc:
            print(f"[DBG] echo-review judge failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return 0
        echo_ids = []
        for entry in picked:
            if not isinstance(entry, dict) or str(entry.get("kind") or "").strip().upper() != "A":
                continue
            try:
                index = int(entry.get("i")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(pending):
                echo_ids.append(pending[index][0])
        if not echo_ids or not hasattr(self.l2, "demote_companion_persona_facts"):
            return 0
        try:
            demoted = await self.l2.demote_companion_persona_facts(
                user_id, echo_ids, reason="echo_of_user_turn")
        except Exception as exc:
            print(f"[DBG] echo-review demote failed: {type(exc).__name__}: {str(exc)[:160]}",
                  flush=True)
            return 0
        print(f"[self-fact] 回音复核：降级 {demoted} 条（共查 {len(pending)}）")
        return int(demoted or 0)


    async def _verify_self_facts(self, rows: list[dict],
                                 messages: list[dict]) -> list[int] | None:
        """二审（§8.41 → §8.91d）：只收"她自己先提起的"。

        判据从"是不是她的"改成**"他那句和这句是不是同一个属性"**——
        前者模型永远答"是"（她的偏好也确实说得通），后者有分辨力：
        实测接话 3/3 判出、自己说起 0/3 误判（唯一条误判是"你最近在忙什么"，
        那是**在问她**，走下面的例外直接收）。

        返回保留下来的下标；失败返回 None（调用方按"没验"处理，不放宽也不收紧）。
        """
        if not rows:
            return []
        try:
            from agent.life_line import asked_about_her
        except Exception:  # pragma: no cover
            def asked_about_her(_text: str) -> bool:  # type: ignore
                return False
        keep: list[int] = []
        pending: list[tuple[int, str, str]] = []
        for index, row in enumerate(rows):
            user_msg = str(row.get("user_msg") or "")
            if user_msg and asked_about_her(user_msg):
                keep.append(index)      # 他问的是她 ⇒ 她本来就在说自己
                continue
            pending.append((index, user_msg, str(row.get("quote") or "")))
            if __import__("os").getenv("PERSONA_DEBUG_PACKET"):
                print(f"[DBG] echo-pair: user={user_msg[:40]!r} quote={str(row.get('quote'))[:40]!r}",
                      flush=True)
        if not pending:
            return keep
        listing = "\n".join(
            f"{n}. 用户：{user_msg[:160]}\n   她：{quote[:160]}"
            for n, (_, user_msg, quote) in enumerate(pending, 1))
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            raw = await robust_llm_client.chat_completion(
                [
                    {"role": "system", "content": self.VERIFY_SYSTEM},
                    {"role": "user", "content": listing},
                ],
                model=_model, temperature=0.0, max_tokens=200,
                extra_body=thinking_disabled_extra_body(_model),
            )
            picked = json.loads(str(raw or "[]"))
            if not isinstance(picked, list):
                return None
            echoed: set[int] = set()
            for entry in picked:
                if isinstance(entry, dict):
                    if str(entry.get("kind") or "").strip().upper() == "A":
                        try:
                            echoed.add(int(entry.get("i")) - 1)
                        except (TypeError, ValueError):
                            continue
                else:
                    try:
                        echoed.add(int(str(entry).strip()) - 1)
                    except (TypeError, ValueError):
                        continue
            keep.extend(index for n, (index, _, _) in enumerate(pending)
                        if n not in echoed)
            return sorted(keep)
        except Exception:
            return None

    async def _content_summary(self, messages: list[dict]) -> str:
        """用 LLM 写**有内容**的归档摘要（plans/1017-02 §8.24）。

        关掉的路径完全不碰这里；打开后失败一律返回空串，由调用方回落模板。
        ⛔ 不写库、不加埋点：只产出一段文本。
        """
        try:
            from config import ARCHIVE_SUMMARY_MAX_MESSAGES as _limit
        except Exception:
            _limit = 40
        lines = []
        window = list(messages or [])
        # §8.29：长对话不能只取尾部 —— 330 条的会话里，头 290 条会被整段丢掉，
        # 摘要就只剩"最后聊了什么"。改成**头尾取样**：开头 12 条（定基调、立人物）
        # ＋ 结尾（最近说了什么），中间折中。实测 330 条会话：只取尾部时摘要只讲最后的话题。
        limit = max(2, int(_limit))
        if len(window) > limit:
            head_n = min(12, limit // 3)
            window = window[:head_n] + window[-(limit - head_n):]
        for message in window:
            role = "用户" if str(message.get("role")) == "user" else "AI"
            text = " ".join(str(message.get("content") or "").split())[:400]
            if text:
                lines.append(f"{role}: {text}")
        if len(lines) < 2:
            return ""
        try:
            from llm_client import robust_llm_client, thinking_disabled_extra_body
            try:
                from config import MEMORY_CURATOR_MODEL as _model
            except Exception:
                _model = None
            # ⛔ 必须关推理：实测同一段对话，不关 = **0 字 / 9.1s**（推理 token 吃光预算），
            # 关掉 = 180 字 / 2.6s（plans/1017-02 §8.24）。空回复会被当成失败回落模板，
            # 表面上"没坏"，实际上这个开关等于白开。
            extra = thinking_disabled_extra_body(_model)
            if hasattr(robust_llm_client, "chat_completion"):
                content = await robust_llm_client.chat_completion(
                    [
                        {"role": "system", "content": self._archive_summary_prompt()},
                        {"role": "user", "content": "对话记录：\n" + "\n".join(lines)},
                    ],
                    model=_model, temperature=0.2, max_tokens=700,
                    extra_body=extra,
                )
                summary = str(content or "").strip()
                if len(summary) >= 20:
                    return summary[:600]
        except Exception:
            return ""
        return ""

    def _generate_summary(self, analysis: dict) -> str:
        """Generate a human-readable session summary with factual event details."""
        parts = []

        event_summary = analysis.get("event_summary")
        if event_summary:
            parts.append(f"具体事件：{event_summary[:160]}")

        cefr, conf = analysis.get("cefr_estimate", ("unknown", 0))
        parts.append(f"CEFR: {cefr} (conf={conf})")

        topics = analysis.get("topics_detected", [])
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")

        errors = analysis.get("error_patterns", {})
        if errors:
            top_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:3]
            parts.append(f"Errors: {', '.join(f'{k}({v})' for k, v in top_errors)}")

        parts.append(f"Messages: {analysis.get('message_count', 0)}")
        parts.append(f"English ratio: {analysis.get('english_ratio', 0)}")

        pref = analysis.get("correction_preference", "recast")
        parts.append(f"Correction: {pref}")

        return " | ".join(parts)
