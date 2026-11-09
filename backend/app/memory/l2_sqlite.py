"""
L2 Memory Layer -- SQLite (替代PostgreSQL)

用SQLite数据库存储结构化事实，零外部依赖。
路径：./data/memory.db

适用于开发和测试环境，不需要安装PostgreSQL。

注意：SQLite不支持pgvector，embedding字段改用JSON存储。
相似度检索改用简单的字符串匹配或跳过。
"""

import sqlite3
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

try:
    from .provider import MemoryEntry as MemoryFact
    from .history_retrieval import rank_history_sparse
    from .entities import (
        assemble_entity_neighborhood,
        entity_neighborhood_ids,
        entity_search_terms,
        merge_entity_search_observations,
        prepare_entity_rows,
        prepare_entity_relation_rows,
        stable_entity_id,
    )
    from .relationship_merge import merge_relationship_items
    from . import learner_profile as _learner_profile
except ImportError:
    from provider import MemoryEntry as MemoryFact
    from memory.history_retrieval import rank_history_sparse
    from memory.entities import (
        assemble_entity_neighborhood,
        entity_neighborhood_ids,
        entity_search_terms,
        merge_entity_search_observations,
        prepare_entity_rows,
        prepare_entity_relation_rows,
        stable_entity_id,
    )
    from memory.relationship_merge import merge_relationship_items
    from memory import learner_profile as _learner_profile


# style_parameters 的可写列白名单。列名会被拼进 SQL，所以**必须**白名单，
# 不能让调用方直接透传（否则是注入口）。l2_postgres 保持同一份清单。
STYLE_PARAMETER_COLUMNS = (
    "formality", "warmth", "humor", "directness", "correction_style",
    "error_tolerance", "challenge_level", "preferred_length", "language_mix",
    "example_frequency", "opening_style", "followup_style", "version",
    "feedback_count",
)


def _timestamp_without_timezone(value):
    """Normalize TIMESTAMP (without time zone) values to naive UTC."""
    if value is None:
        return None
    parsed = _learner_profile.parse_dt(value)
    return parsed.replace(tzinfo=None) if parsed else value


class L2SQLiteMemory:
    """
    L2结构化事实存储 - SQLite版。
    
    替代L2PostgresMemory，使用SQLite数据库，无需PostgreSQL。
    
    支持：
    - error_fingerprints: 错误指纹存储
    - mastered_expressions: 已掌握表达存储
    - session_archives: 会话归档
    - methodology_calls: 教学法调用记录
    """

    def __init__(self, db_path: str = "./data/memory.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_excerpt TEXT,
                count INTEGER DEFAULT 1,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, error_type)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mastered_expressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                expression TEXT NOT NULL,
                mastered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL DEFAULT 0.8
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS methodology_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                instruction TEXT,
                example_dialogue TEXT,
                cefr_level TEXT,
                scene_tags TEXT,
                embedding TEXT,
                boost_score REAL DEFAULT 0.0,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS methodology_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                methodology_id INTEGER,
                cooperation_score REAL,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_daily_state (
                user_id TEXT NOT NULL,
                date TEXT DEFAULT CURRENT_DATE,
                review_queue TEXT,
                session_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, date)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_archives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT UNIQUE,
                s3_key TEXT,
                duration_seconds INTEGER,
                cefr_before TEXT,
                cefr_after TEXT,
                error_count INTEGER,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                dream_level TEXT,
                changes TEXT,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                layer TEXT NOT NULL,
                language TEXT NOT NULL,
                skill TEXT NOT NULL,
                source TEXT,
                source_note TEXT DEFAULT '',
                keywords TEXT,
                content TEXT NOT NULL,
                embedding TEXT,
                boost_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                chunk_id TEXT,
                session_id TEXT,
                cooperation_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                seq INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conv_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                target_lang TEXT DEFAULT 'English',
                level TEXT DEFAULT '',
                scene TEXT DEFAULT 'daily_talk',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hidden_at TIMESTAMP DEFAULT NULL,
                expires_at TIMESTAMP DEFAULT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cassette_handoffs (
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                handoff_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, conv_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_curator_runs (
                run_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                conversation_id TEXT DEFAULT '',
                trigger_name TEXT NOT NULL,
                status TEXT NOT NULL,
                candidates_json TEXT NOT NULL DEFAULT '[]',
                accepted_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_curator_user_time "
            "ON memory_curator_runs(user_id, created_at DESC)"
        )
        # 兼容旧表：补字段（SQLite ALTER 失败会抛，捕获即可）
        try:
            cursor.execute("ALTER TABLE conversation_meta ADD COLUMN hidden_at TIMESTAMP DEFAULT NULL")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE conversation_meta ADD COLUMN expires_at TIMESTAMP DEFAULT NULL")
        except Exception:
            pass

        # 练习盘话题分配（plans/0923-06 §4.1.2）：建盘当时的来源快照，只服务练习盘。
        # 与 db/migrate_expression_topics.sql / db/init.sql 同形；幂等可重复初始化。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expression_topic_assignments (
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                source_title TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                source_date TEXT NOT NULL DEFAULT '',
                prompt_text TEXT NOT NULL DEFAULT '',
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, conv_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_expr_topic_assign_user_time "
            "ON expression_topic_assignments(user_id, assigned_at DESC)"
        )

        # ── Loop Engineering Tables ─────────────────────────

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS style_parameters (
                user_id TEXT PRIMARY KEY,
                formality REAL DEFAULT 0.5,
                warmth REAL DEFAULT 0.5,
                humor REAL DEFAULT 0.3,
                directness REAL DEFAULT 0.5,
                correction_style TEXT DEFAULT 'recast',
                error_tolerance REAL DEFAULT 0.5,
                challenge_level REAL DEFAULT 0.5,
                preferred_length INTEGER DEFAULT 30,
                language_mix REAL DEFAULT 0.5,
                example_frequency REAL DEFAULT 0.5,
                opening_style TEXT DEFAULT 'suggest',
                followup_style TEXT DEFAULT 'expand',
                version INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_count INTEGER DEFAULT 0
            )
        """)

        # Loop Engineering: prompt 自动优化 patches（审计驱动）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_patches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                patch_id TEXT UNIQUE NOT NULL,
                session_id TEXT,
                audit_recommendation_id TEXT,
                root_cause TEXT DEFAULT '',
                action TEXT NOT NULL,
                layer TEXT NOT NULL,
                from_value TEXT,
                to_value TEXT,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                verified_sessions INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                applied_at TIMESTAMP,
                failed_at TIMESTAMP,
                rolled_back_at TIMESTAMP
            )
        """)
        try:
            cursor.execute("ALTER TABLE prompt_patches ADD COLUMN root_cause TEXT DEFAULT ''")
        except Exception:
            pass
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_patches_user ON prompt_patches(user_id, status)")

        # Outcome-based learning loop: error detection alone is not mastery.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_item_states (
                user_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                mastery_probability REAL DEFAULT 0.35,
                detected_count INTEGER DEFAULT 0,
                immediate_success INTEGER DEFAULT 0,
                transfer_success INTEGER DEFAULT 0,
                delayed_success INTEGER DEFAULT 0,
                failed_attempts INTEGER DEFAULT 0,
                next_review_at TIMESTAMP,
                stability_days REAL DEFAULT 1.0,
                difficulty REAL DEFAULT 5.0,
                last_review_at TIMESTAMP,
                review_count INTEGER DEFAULT 0,
                last_evidence TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, item_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_reviews ON learning_item_states(user_id, next_review_at)")
        for column_sql in (
            "ALTER TABLE learning_item_states ADD COLUMN stability_days REAL DEFAULT 1.0",
            "ALTER TABLE learning_item_states ADD COLUMN difficulty REAL DEFAULT 5.0",
            "ALTER TABLE learning_item_states ADD COLUMN last_review_at TIMESTAMP",
            "ALTER TABLE learning_item_states ADD COLUMN review_count INTEGER DEFAULT 0",
        ):
            try:
                cursor.execute(column_sql)
            except Exception:
                pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speech_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                transcript TEXT,
                duration_seconds REAL,
                word_count INTEGER,
                pause_count INTEGER,
                words_per_minute REAL,
                fluency_score REAL,
                pronunciation_accuracy REAL,
                prosody_score REAL,
                assessment_source TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_speech_assessments_user ON speech_assessments(user_id, created_at DESC)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_turn_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                asr_duration_ms INTEGER,
                first_text_latency_ms INTEGER,
                first_audio_latency_ms INTEGER,
                response_duration_ms INTEGER,
                first_user_speech_latency_ms INTEGER,
                session_entry_mode TEXT DEFAULT 'fallback',
                stopped_by_user INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_voice_metrics_session ON voice_turn_metrics(session_id, turn_index)")
        for migration in (
            "ALTER TABLE voice_turn_metrics ADD COLUMN first_user_speech_latency_ms INTEGER",
            "ALTER TABLE voice_turn_metrics ADD COLUMN session_entry_mode TEXT DEFAULT 'fallback'",
        ):
            try:
                cursor.execute(migration)
            except Exception:
                pass

        # Append/update one privacy-safe contract row per turn and one aggregate
        # row per session.  Provider usage stays nullable when it is not
        # reported; no transcript, raw audio, credential, or inferred price is
        # stored in this table.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metering_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_version INTEGER NOT NULL DEFAULT 1,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL DEFAULT '',
                event_scope TEXT NOT NULL DEFAULT 'turn',
                request_status TEXT NOT NULL DEFAULT 'unavailable',
                provider TEXT DEFAULT '',
                model TEXT DEFAULT '',
                request_count INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                token_usage_status TEXT NOT NULL DEFAULT 'unavailable',
                asr_input_duration_ms INTEGER,
                asr_usage_status TEXT NOT NULL DEFAULT 'unavailable',
                tts_output_duration_ms INTEGER,
                tts_character_count INTEGER,
                tts_usage_status TEXT NOT NULL DEFAULT 'unavailable',
                canceled INTEGER DEFAULT 0,
                failure_code TEXT DEFAULT '',
                structured_probe_issued INTEGER DEFAULT 0,
                structured_probe_result TEXT DEFAULT 'not_issued',
                learning_evidence_valid INTEGER DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, turn_id, event_scope)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metering_session ON metering_events(session_id, event_scope, turn_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metering_user_time ON metering_events(user_id, created_at DESC)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                turn_index INTEGER DEFAULT 0,
                feedback_type TEXT NOT NULL,
                signal_value REAL DEFAULT 0.0,
                context TEXT DEFAULT '{}',
                timestamp REAL DEFAULT 0.0
            )
        """)

        # ── 自动化审计与自修复系统表（SQLite 兼容版） ──────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_audit_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                session_started_at TIMESTAMP NOT NULL,
                session_ended_at TIMESTAMP,
                audit_status TEXT NOT NULL DEFAULT 'pending',
                audit_triggered_at TIMESTAMP,
                audit_completed_at TIMESTAMP,
                issues_found TEXT,  -- JSON string
                modifications_made TEXT,  -- JSON string
                related_chunk_ids TEXT,  -- JSON string of list
                audit_version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                modification_id TEXT NOT NULL UNIQUE,
                issue_id TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                old_value TEXT,  -- JSON string
                new_value TEXT,  -- JSON string
                reasoning TEXT,
                triggered_by TEXT DEFAULT 'auto',
                status TEXT DEFAULT 'pending',
                validation_turns INTEGER DEFAULT 0,
                validation_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rolled_back_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                trigger_turn INTEGER NOT NULL,
                user_message TEXT,
                system_response TEXT,
                context_summary TEXT,
                related_chunk_ids TEXT,  -- JSON string of list
                processed_for_evolution INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Relationship memory is intentionally separate from mastery/error
        # memory.  It stores only compact conversational continuity signals.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationship_memory (
                user_id TEXT PRIMARY KEY,
                last_topic TEXT DEFAULT '',
                open_loop TEXT DEFAULT '',
                recent_mood TEXT DEFAULT '',
                preferred_tone TEXT DEFAULT '',
                engagement_topics TEXT DEFAULT '[]',
                avoid_patterns TEXT DEFAULT '[]',
                last_user_message TEXT DEFAULT '',
                last_assistant_response TEXT DEFAULT '',
                memory_items TEXT DEFAULT '[]',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            cursor.execute("ALTER TABLE relationship_memory ADD COLUMN memory_items TEXT DEFAULT '[]'")
        except Exception:
            pass

        # Alice 的用户隔离自我事实与可重建成长摘编。字段与
        # db/migrate_companion_persona.sql 的 Postgres 结构同形。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_persona_facts (
                user_id TEXT NOT NULL,
                persona_key TEXT NOT NULL DEFAULT 'alice',
                fact_id TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT '',
                facet_key TEXT NOT NULL DEFAULT '',
                value_text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                source_kind TEXT NOT NULL DEFAULT 'unknown',
                source_conv_id TEXT,
                source_turn INTEGER,
                source_message_ref TEXT NOT NULL DEFAULT '',
                evidence_quote TEXT NOT NULL DEFAULT '',
                interaction_context TEXT NOT NULL DEFAULT '',
                interaction_context_source TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                observed_at TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                recorded_at TEXT NOT NULL DEFAULT '',
                time_basis TEXT NOT NULL DEFAULT '',
                supersedes TEXT NOT NULL DEFAULT '',
                supporting_fact_ids TEXT NOT NULL DEFAULT '[]',
                related_entity_ids TEXT NOT NULL DEFAULT '[]',
                legacy_fact_id TEXT NOT NULL DEFAULT '',
                source_occurrence_id TEXT NOT NULL DEFAULT '',
                hidden_at TEXT,
                PRIMARY KEY (user_id, fact_id),
                UNIQUE (user_id, source_occurrence_id)
            )
        """)
        # 未闭环的事（plans/1017-02 §8.56）：与 Postgres 同形，默认关。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_supply_log (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT '',
                ref_id TEXT NOT NULL DEFAULT '',
                ref_hint TEXT NOT NULL DEFAULT '',
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, session_id, kind, ref_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_life_items (
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT '',
                aliases TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                source TEXT NOT NULL DEFAULT 'weekly',
                created_at TEXT NOT NULL DEFAULT '',
                topic TEXT NOT NULL DEFAULT '',
                hits INTEGER NOT NULL DEFAULT 1,
                strength REAL NOT NULL DEFAULT 1.0,
                depth REAL NOT NULL DEFAULT 1.0,
                last_recalled_at TEXT NOT NULL DEFAULT '',
                recalls INTEGER NOT NULL DEFAULT 0,
                ref TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                actions_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (user_id, item_id)
            )
        """)
        life_columns = {row[1] for row in cursor.execute("PRAGMA table_info(companion_life_items)")}
        for column, definition in (
            ("aliases", "TEXT NOT NULL DEFAULT ''"),
            ("source", "TEXT NOT NULL DEFAULT 'legacy_unknown'"),
            ("topic", "TEXT NOT NULL DEFAULT ''"),
            ("hits", "INTEGER NOT NULL DEFAULT 1"),
            ("strength", "REAL NOT NULL DEFAULT 1.0"),
            ("depth", "REAL NOT NULL DEFAULT 1.0"),
            ("last_recalled_at", "TEXT NOT NULL DEFAULT ''"),
            ("recalls", "INTEGER NOT NULL DEFAULT 0"),
            ("ref", "TEXT NOT NULL DEFAULT ''"),
            ("title", "TEXT NOT NULL DEFAULT ''"),
            ("author", "TEXT NOT NULL DEFAULT ''"),
            ("detail", "TEXT NOT NULL DEFAULT ''"),
            ("actions_json", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if column not in life_columns:
                cursor.execute(f"ALTER TABLE companion_life_items ADD COLUMN {column} {definition}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_open_questions (
                user_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                topic TEXT NOT NULL DEFAULT '',
                source_conv_id TEXT,
                asked_at TEXT NOT NULL DEFAULT '',
                shown_at TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, question_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_open_threads (
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                source_conv_id TEXT,
                source_quote TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                closed_at TEXT NOT NULL DEFAULT '',
                closed_reason TEXT NOT NULL DEFAULT '',
                due_hint TEXT NOT NULL DEFAULT '',
                due_at TEXT NOT NULL DEFAULT '',
                asked_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, thread_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_open_threads "
            "ON companion_open_threads(user_id, status, last_seen_at DESC)"
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_outbox (
                user_id TEXT NOT NULL,
                msg_id TEXT NOT NULL,
                conv_id TEXT,
                text TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                due_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                read_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, msg_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_outbox_unread "
            "ON companion_outbox(user_id, read_at, created_at DESC)"
        )
        for column, declaration in (
            ("interaction_context", "TEXT NOT NULL DEFAULT ''"),
            ("interaction_context_source", "TEXT NOT NULL DEFAULT ''"),
            ("support_count", "INTEGER NOT NULL DEFAULT 1"),
        ):
            try:
                cursor.execute(
                    f"ALTER TABLE companion_persona_facts ADD COLUMN {column} {declaration}"
                )
            except Exception:
                pass
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_persona_current "
            "ON companion_persona_facts(user_id, persona_key, status, facet_key, recorded_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_companion_persona_source "
            "ON companion_persona_facts(user_id, source_conv_id)"
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companion_persona_snapshots (
                user_id TEXT NOT NULL,
                persona_key TEXT NOT NULL DEFAULT 'alice',
                narrative TEXT NOT NULL DEFAULT '',
                supporting_fact_ids TEXT NOT NULL DEFAULT '[]',
                snapshot_version TEXT NOT NULL DEFAULT '',
                source_hash TEXT NOT NULL DEFAULT '',
                generated_at TEXT,
                legacy_backfill_version TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, persona_key)
            )
        """)

        # 图谱-lite 实体表（plans/0915-09/13）：与 postgres 的 memory_entities
        # 方法面对齐——缺一个就是 API 路由 500 或被 hasattr 兜成静默空值。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conv_id TEXT DEFAULT '',
                etype TEXT DEFAULT 'other',
                name TEXT NOT NULL,
                detail TEXT DEFAULT '',
                entity_id TEXT DEFAULT '',
                evidence_source TEXT DEFAULT 'unknown',
                confidence REAL DEFAULT 0.0,
                evidence_ref TEXT DEFAULT '',
                source_message_seq INTEGER DEFAULT NULL,
                evidence_quote TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, etype, name, conv_id)
            )
        """)
        for column, declaration in (
            ("entity_id", "TEXT DEFAULT ''"),
            ("evidence_source", "TEXT DEFAULT 'unknown'"),
            ("confidence", "REAL DEFAULT 0.0"),
            ("evidence_ref", "TEXT DEFAULT ''"),
            ("source_message_seq", "INTEGER DEFAULT NULL"),
            ("evidence_quote", "TEXT DEFAULT ''"),
        ):
            try:
                cursor.execute(f"ALTER TABLE memory_entities ADD COLUMN {column} {declaration}")
            except Exception:
                pass
        cursor.execute(
            "SELECT user_id, conv_id, etype, name FROM memory_entities "
            "WHERE entity_id IS NULL OR entity_id = ''"
        )
        for row in cursor.fetchall():
            entity_id = stable_entity_id(
                str(row["user_id"]), str(row["etype"] or "other"),
                str(row["name"] or ""), str(row["conv_id"] or ""),
            )
            cursor.execute(
                "UPDATE memory_entities SET entity_id = ? "
                "WHERE user_id = ? AND conv_id = ? AND etype = ? AND name = ? "
                "AND (entity_id IS NULL OR entity_id = '')",
                (entity_id, row["user_id"], row["conv_id"], row["etype"], row["name"]),
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_entities_user_name "
            "ON memory_entities(user_id, name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_entities_user_entity "
            "ON memory_entities(user_id, entity_id)"
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entity_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                edge_key TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id TEXT DEFAULT '',
                object_value TEXT DEFAULT '',
                evidence_source TEXT DEFAULT 'unknown',
                evidence_ref TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                source_message_seq INTEGER DEFAULT NULL,
                evidence_quote TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, conv_id, edge_key)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_entity_relations_subject "
            "ON memory_entity_relations(user_id, subject_entity_id, created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_entity_relations_object "
            "ON memory_entity_relations(user_id, object_entity_id, created_at DESC)"
        )
        try:
            cursor.execute(
            "ALTER TABLE memory_entity_relations ADD COLUMN source_message_seq INTEGER DEFAULT NULL"
            )
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE memory_entity_relations ADD COLUMN evidence_quote TEXT DEFAULT ''")
        except Exception:
            pass

        # 词块本（plans/0916-11）：与 postgres 的 chunk_items 方法面对齐
        # （0916-09 的教训：只写一边会让功能在开发机静默失效）。
        # use_events / status_history 用 TEXT 存 JSON（postgres 是 jsonb），
        # 读写一律走 engine.chunk_evidence.loads_json_list 容错。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                norm TEXT NOT NULL,
                kind TEXT DEFAULT 'collocation',
                status TEXT DEFAULT 'candidate',
                source_type TEXT DEFAULT 'tape',
                source_id TEXT DEFAULT '',
                source_turn INTEGER,
                context_snippet TEXT DEFAULT '',
                gloss TEXT DEFAULT '',
                example TEXT DEFAULT '',
                slot TEXT DEFAULT '',
                reason TEXT DEFAULT '',
                use_count INTEGER DEFAULT 0,
                use_events TEXT DEFAULT '[]',
                status_history TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_use_at TIMESTAMP,
                UNIQUE (user_id, norm)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_items_user_status "
            "ON chunk_items(user_id, status)"
        )

        # 素材（plans/0916-21 D4/F3）：与 postgres 的 material_items 方法面对齐。
        # blocks 在本后端是 JSON 文本（PG 那边是 jsonb），两边读写都走
        # loads_json_list，所以返回给前端的形状一致；本地/云端体验不能分叉。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS material_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                kind TEXT DEFAULT 'text',
                title TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',
                raw_text TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                char_count INTEGER DEFAULT 0,
                material_conv_id TEXT DEFAULT '',
                status TEXT DEFAULT 'ready',
                error TEXT DEFAULT '',
                blocks TEXT DEFAULT '[]',
                analysis TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_material_items_user_kind "
            "ON material_items(user_id, kind, created_at DESC)"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_material_items_conv "
            "ON material_items(user_id, material_conv_id) "
            "WHERE material_conv_id <> ''"
        )
        # 补列：老库的 material_items 没有 analysis（plans/0916-24 深析产物），
        # CREATE TABLE IF NOT EXISTS 对已存在的表不生效，只能 ALTER 一次。
        for migration in (
            "ALTER TABLE material_items ADD COLUMN analysis TEXT DEFAULT '{}'",
        ):
            try:
                cursor.execute(migration)
            except Exception:
                pass

        # 素材库（plans/0923-08）：两张素材专用表，与 PG 侧
        # db/migrate_material_library.sql / init.sql 同形同义（列名一致，
        # JSON 列在本后端是 TEXT，读写一律走 loads_json_list）。
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS material_source_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                material_id INTEGER NOT NULL,
                block_no INTEGER NOT NULL,
                span_kind TEXT DEFAULT 'body',
                text TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_material_source_blocks_key "
            "ON material_source_blocks(user_id, material_id, block_no)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_material_source_blocks_user "
            "ON material_source_blocks(user_id, material_id)"
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS material_tape_builds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                conv_id TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                source_ids TEXT DEFAULT '[]',
                summary TEXT DEFAULT '',
                groups TEXT DEFAULT '[]',
                hooks TEXT DEFAULT '[]',
                top_chunks TEXT DEFAULT '[]',
                error TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_material_tape_builds_request "
            "ON material_tape_builds(user_id, request_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_material_tape_builds_conv "
            "ON material_tape_builds(user_id, conv_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_material_tape_builds_status "
            "ON material_tape_builds(user_id, status, updated_at DESC)"
        )
        # 旧库回填游标（幂等补列）：与 PG 侧同一列名
        for migration in (
            "ALTER TABLE material_items ADD COLUMN blocks_backfilled_at TIMESTAMP DEFAULT NULL",
        ):
            try:
                cursor.execute(migration)
            except Exception:
                pass

        # 统一学习者画像（plans/0923-04 阶段 1）：两表 + 三索引，DDL 全幂等，
        # 与 PG 侧 db/migrate_learner_profile.sql 同源同形。
        cursor.executescript(_learner_profile.SQLITE_DDL)

        # Versioned PG counterpart: migrate_conversation_turn_ids_20261025.sql.
        if "turn_id" not in {row[1] for row in cursor.execute("PRAGMA table_info(conversation_messages)")}:
            cursor.execute("ALTER TABLE conversation_messages ADD COLUMN turn_id TEXT NOT NULL DEFAULT ''")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_messages_turn "
            "ON conversation_messages(user_id, conv_id, turn_id) WHERE turn_id<>''"
        )


        conn.commit()
        conn.close()

    async def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # add_fact 的权威实现在文件下方（贴近 get_mastered），那里读的是
    # `fact.entry_type`。此处原先还有一份读 `fact.fact_type` 的同名副本，
    # 被下方定义覆盖成死代码 —— 已删除，避免以后有人误改到这一份。

    async def get_errors(self, user_id: str, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, error_type, error_excerpt, count, weight, last_seen_at
            FROM error_fingerprints
            WHERE user_id = ? AND weight > 0.05
            ORDER BY weight DESC LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def add_error(self, user_id: str, error_type: str, error_excerpt: str = "", weight: float = 1.0):
        """Add or update an error fingerprint."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO error_fingerprints (user_id, error_type, error_excerpt, count, weight)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT(user_id, error_type) DO UPDATE SET
               count = error_fingerprints.count + 1,
               weight = min(1.0, error_fingerprints.weight + 0.1),
               last_seen_at = datetime("now")""",
            (user_id, error_type, error_excerpt, weight)
        )
        conn.commit()

    def add_mastered(self, user_id: str, expression: str, confidence: float = 0.8):
        """Mark an expression as mastered."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO mastered_expressions (user_id, expression, confidence) VALUES (?, ?, ?)",
            (user_id, expression, confidence)
        )
        conn.commit()

    async def get_learning_state(self, user_id: str, item_type: str) -> dict:
        row = self._get_conn().execute(
            "SELECT * FROM learning_item_states WHERE user_id = ? AND item_type = ?",
            (user_id, item_type),
        ).fetchone()
        return dict(row) if row else {}

    async def insert_feedback_signal(self, signal: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO feedback_signals
               (session_id, user_id, turn_index, feedback_type, signal_value, context, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.get("session_id", ""), signal.get("user_id", ""),
                signal.get("turn_index", 0), signal.get("feedback_type", ""),
                signal.get("signal_value", 0.0),
                json.dumps(signal.get("context", {}), ensure_ascii=False),
                signal.get("timestamp", 0.0),
            ),
        )
        conn.commit()

    async def insert_voice_turn_metric(self, user_id: str, session_id: str, metrics: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO voice_turn_metrics
               (user_id, session_id, turn_index, asr_duration_ms, first_text_latency_ms,
                first_audio_latency_ms, response_duration_ms, first_user_speech_latency_ms,
                session_entry_mode, stopped_by_user)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, session_id, int(metrics.get("turn_index", 0)),
             metrics.get("asr_duration_ms"), metrics.get("first_text_latency_ms"),
             metrics.get("first_audio_latency_ms"), metrics.get("response_duration_ms"),
             metrics.get("first_user_speech_latency_ms"), metrics.get("session_entry_mode", "fallback"),
             1 if metrics.get("stopped_by_user") else 0),
        )
        conn.commit()

    async def upsert_metering_event(self, event: dict) -> dict:
        """Persist one cumulative turn/session event without sensitive payloads."""
        from .metering_contract import merge_metering_event, normalize_metering_event

        incoming = normalize_metering_event(event)
        conn = self._get_conn()
        existing_row = conn.execute(
            """SELECT * FROM metering_events
               WHERE session_id = ? AND turn_id = ? AND event_scope = ?""",
            (incoming["session_id"], incoming["turn_id"], incoming["event_scope"]),
        ).fetchone()
        existing = dict(existing_row) if existing_row else {}
        if existing:
            try:
                existing["metadata"] = json.loads(existing.get("metadata") or "{}")
            except Exception:
                existing["metadata"] = {}
        merged = merge_metering_event(existing, incoming)
        conn.execute(
            """INSERT INTO metering_events
               (schema_version, user_id, session_id, turn_id, event_scope,
                request_status, provider, model, request_count, retry_count,
                input_tokens, output_tokens, total_tokens, token_usage_status,
                asr_input_duration_ms, asr_usage_status, tts_output_duration_ms,
                tts_character_count, tts_usage_status, canceled, failure_code,
                structured_probe_issued, structured_probe_result,
                learning_evidence_valid, metadata, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(session_id, turn_id, event_scope) DO UPDATE SET
                 schema_version=excluded.schema_version, user_id=excluded.user_id,
                 request_status=excluded.request_status, provider=excluded.provider,
                 model=excluded.model, request_count=excluded.request_count,
                 retry_count=excluded.retry_count, input_tokens=excluded.input_tokens,
                 output_tokens=excluded.output_tokens, total_tokens=excluded.total_tokens,
                 token_usage_status=excluded.token_usage_status,
                 asr_input_duration_ms=excluded.asr_input_duration_ms,
                 asr_usage_status=excluded.asr_usage_status,
                 tts_output_duration_ms=excluded.tts_output_duration_ms,
                 tts_character_count=excluded.tts_character_count,
                 tts_usage_status=excluded.tts_usage_status, canceled=excluded.canceled,
                 failure_code=excluded.failure_code,
                 structured_probe_issued=excluded.structured_probe_issued,
                 structured_probe_result=excluded.structured_probe_result,
                 learning_evidence_valid=excluded.learning_evidence_valid,
                 metadata=excluded.metadata, updated_at=CURRENT_TIMESTAMP""",
            (
                merged["schema_version"], merged["user_id"], merged["session_id"],
                merged["turn_id"], merged["event_scope"], merged["request_status"],
                merged["provider"], merged["model"], merged["request_count"],
                merged["retry_count"], merged["input_tokens"], merged["output_tokens"],
                merged["total_tokens"], merged["token_usage_status"],
                merged["asr_input_duration_ms"], merged["asr_usage_status"],
                merged["tts_output_duration_ms"], merged["tts_character_count"],
                merged["tts_usage_status"], 1 if merged["canceled"] else 0,
                merged["failure_code"], 1 if merged["structured_probe_issued"] else 0,
                merged["structured_probe_result"], 1 if merged["learning_evidence_valid"] else 0,
                json.dumps(merged["metadata"], ensure_ascii=False),
            ),
        )
        conn.commit()
        return merged

    async def get_session_metering(self, session_id: str) -> list[dict]:
        rows = self._get_conn().execute(
            """SELECT * FROM metering_events WHERE session_id = ?
               ORDER BY CASE event_scope WHEN 'turn' THEN 0 ELSE 1 END, id""",
            (session_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except Exception:
                item["metadata"] = {}
            result.append(item)
        return result

    async def finalize_session_metering(self, user_id: str, session_id: str, turn_count: int) -> dict:
        turns = [item for item in await self.get_session_metering(session_id) if item["event_scope"] == "turn"]
        def total(field: str) -> int | None:
            values = [int(item[field]) for item in turns if item.get(field) is not None]
            return sum(values) if values else None
        token_states = {item.get("token_usage_status", "unavailable") for item in turns}
        token_status = "reported" if token_states == {"reported"} and turns else ("partial" if token_states & {"reported", "partial"} else "unavailable")
        providers = {item.get("provider") for item in turns if item.get("provider")}
        models = {item.get("model") for item in turns if item.get("model")}
        asr_values = [item for item in turns if item.get("asr_input_duration_ms") is not None]
        tts_values = [item for item in turns if item.get("tts_output_duration_ms") is not None or item.get("tts_character_count") is not None]
        failed_turns = sum(1 for item in turns if item.get("request_status") == "failed")
        canceled_turns = sum(1 for item in turns if item.get("canceled"))
        successful_turns = sum(1 for item in turns if item.get("request_status") == "success")
        probe_results = {item.get("structured_probe_result") for item in turns}
        session_probe_result = "correct" if "correct" in probe_results else (
            "incorrect" if "incorrect" in probe_results else (
                "not_attempted" if any(bool(item.get("structured_probe_issued")) for item in turns) else "not_issued"
            )
        )
        event = {
            "user_id": user_id, "session_id": session_id, "turn_id": "__session__",
            "event_scope": "session",
            "request_status": "completed_with_failures" if failed_turns or canceled_turns else "completed",
            "provider": next(iter(providers)) if len(providers) == 1 else ("multiple" if providers else ""),
            "model": next(iter(models)) if len(models) == 1 else ("multiple" if models else ""),
            "request_count": sum(int(item.get("request_count") or 0) for item in turns),
            "retry_count": sum(int(item.get("retry_count") or 0) for item in turns),
            "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "total_tokens": total("total_tokens"), "token_usage_status": token_status,
            "asr_input_duration_ms": total("asr_input_duration_ms"),
            "asr_usage_status": "observed" if asr_values else "unavailable",
            "tts_output_duration_ms": total("tts_output_duration_ms"),
            "tts_character_count": total("tts_character_count"),
            "tts_usage_status": "observed" if tts_values else "unavailable",
            "canceled": canceled_turns > 0,
            "structured_probe_issued": any(bool(item.get("structured_probe_issued")) for item in turns),
            "structured_probe_result": session_probe_result,
            "learning_evidence_valid": any(bool(item.get("learning_evidence_valid")) for item in turns),
            "metadata": {
                "session_turn_count": int(turn_count), "metered_turns": len(turns),
                "successful_turns": successful_turns, "failed_turns": failed_turns,
                "canceled_turns": canceled_turns,
            },
        }
        return await self.upsert_metering_event(event)

    async def get_recent_feedback_signals(self, user_id: str, limit: int = 20) -> list[dict]:
        rows = self._get_conn().execute(
            """SELECT session_id, user_id, turn_index, feedback_type,
                      signal_value, context, timestamp
               FROM feedback_signals WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [{
            "session_id": row[0], "user_id": row[1], "turn_index": row[2],
            "feedback_type": row[3], "signal_value": row[4],
            "context": json.loads(row[5] or "{}"), "timestamp": row[6],
        } for row in rows]

    async def upsert_learning_state(
        self, user_id: str, item_type: str, mastery_probability: float, **changes
    ) -> None:
        conn = self._get_conn()
        existing = await self.get_learning_state(user_id, item_type)
        values = {
            "detected_count": int(existing.get("detected_count", 0)) + int(changes.get("detected_delta", 0)),
            "immediate_success": int(existing.get("immediate_success", 0)) + int(changes.get("immediate_success", 0)),
            "transfer_success": int(existing.get("transfer_success", 0)) + int(changes.get("transfer_success", 0)),
            "delayed_success": int(existing.get("delayed_success", 0)) + int(changes.get("delayed_success", 0)),
            "failed_attempts": int(existing.get("failed_attempts", 0)) + int(changes.get("failed_delta", 0)),
            "next_review_at": _timestamp_without_timezone(changes.get("next_review_at")),
            "stability_days": float(changes.get("stability_days", existing.get("stability_days", 1.0) or 1.0)),
            "difficulty": float(changes.get("difficulty", existing.get("difficulty", 5.0) or 5.0)),
            "last_review_at": _timestamp_without_timezone(
                changes.get("last_review_at", existing.get("last_review_at"))
            ),
            "review_count": int(existing.get("review_count", 0)) + int(changes.get("review_delta", 0)),
            "last_evidence": json.dumps(changes.get("last_evidence", {}), ensure_ascii=False),
        }
        next_review = values["next_review_at"]
        if hasattr(next_review, "isoformat"):
            next_review = next_review.isoformat()
        conn.execute(
            """INSERT INTO learning_item_states
               (user_id, item_type, mastery_probability, detected_count, immediate_success,
                transfer_success, delayed_success, failed_attempts, next_review_at,
                stability_days, difficulty, last_review_at, review_count, last_evidence, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, item_type) DO UPDATE SET
                 mastery_probability=excluded.mastery_probability,
                 detected_count=excluded.detected_count,
                 immediate_success=excluded.immediate_success,
                 transfer_success=excluded.transfer_success,
                 delayed_success=excluded.delayed_success,
                 failed_attempts=excluded.failed_attempts,
                 next_review_at=excluded.next_review_at,
                 stability_days=excluded.stability_days,
                 difficulty=excluded.difficulty,
                 last_review_at=excluded.last_review_at,
                 review_count=excluded.review_count,
                 last_evidence=excluded.last_evidence,
                 updated_at=CURRENT_TIMESTAMP""",
            (user_id, item_type, mastery_probability, values["detected_count"], values["immediate_success"],
             values["transfer_success"], values["delayed_success"], values["failed_attempts"], next_review,
             values["stability_days"], values["difficulty"], values["last_review_at"], values["review_count"],
             values["last_evidence"]),
        )
        conn.commit()

    async def get_due_learning_reviews(self, user_id: str, limit: int = 3) -> list[dict]:
        rows = self._get_conn().execute(
            """SELECT * FROM learning_item_states
               WHERE user_id = ? AND next_review_at IS NOT NULL AND next_review_at <= CURRENT_TIMESTAMP
               ORDER BY mastery_probability ASC, next_review_at ASC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    async def get_learning_progress(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self._get_conn().execute(
            """SELECT item_type, mastery_probability, detected_count, immediate_success,
                      transfer_success, delayed_success, failed_attempts, next_review_at,
                      stability_days, difficulty, last_review_at, review_count, updated_at
               FROM learning_item_states WHERE user_id = ?
               ORDER BY mastery_probability ASC, updated_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    async def record_speech_assessment(self, user_id: str, session_id: str, result: dict) -> None:
        self._get_conn().execute(
            """INSERT INTO speech_assessments
               (user_id, session_id, transcript, duration_seconds, word_count, pause_count,
                words_per_minute, fluency_score, pronunciation_accuracy, prosody_score, assessment_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, session_id, result.get("transcript", ""), result.get("duration_seconds"),
             result.get("word_count"), result.get("pause_count"), result.get("words_per_minute"),
             result.get("fluency_score"), result.get("pronunciation_accuracy"), result.get("prosody_score"),
             json.dumps(result.get("assessment_source", {}), ensure_ascii=False)),
        )
        self._get_conn().commit()

    async def get_speech_assessments(self, user_id: str, limit: int = 20) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM speech_assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
        ).fetchall()
        return [dict(row) for row in rows]

    def set_error_fingerprint(self, user_id: str, error_type: str, count: int, weight: float):
        """Overwrite the exact count and weight of an error fingerprint, or delete it if count <= 0."""
        conn = self._get_conn()
        if count <= 0 or weight <= 0.05:
            conn.execute(
                "DELETE FROM error_fingerprints WHERE user_id = ? AND error_type = ?",
                (user_id, error_type)
            )
        else:
            conn.execute(
                """INSERT INTO error_fingerprints (user_id, error_type, count, weight)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, error_type) DO UPDATE SET
                   count = ?,
                   weight = ?,
                   last_seen_at = datetime("now")""",
                (user_id, error_type, count, weight, count, weight)
            )
        conn.commit()

    async def update_l1_profile(self, user_id: str, fields: dict):
        """
        Update L1 profile fields (cefr, recent_weakness) via file-based storage.
        Used by profile_evolution._sync_to_memory to persist CEFR and weakness
        after each session analysis.
        """
        try:
            from .l1_file import L1FileProfile
        except ImportError:
            from l1_file import L1FileProfile

        l1 = L1FileProfile()
        await l1.update_profile(user_id, fields)

    async def add_fact(self, user_id: str, fact):
        """Add a MemoryEntry fact - compatible with L2PostgresMemory interface."""
        self.add_error(user_id, fact.entry_type, fact.content, fact.weight)

    def save_session_archive(self, user_id: str, session_id: str,
                             summary: str, error_count: int,
                             cefr_before: str = "", cefr_after: str = ""):
        """Save a session archive summary."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO session_archives
            (user_id, session_id, summary, error_count, cefr_before, cefr_after, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, session_id, summary, error_count, cefr_before, cefr_after))
        conn.commit()

    def get_last_archive_time(self, user_id: str) -> dict:
        """最近一盘归档时间与摘要（方法面与 postgres 同步，plans/0915-17 时间系统）。

        sqlite 是 sync def——调用点统一走 memory.provider.maybe_await。
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT summary, created_at FROM session_archives "
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        except Exception:
            return {}
        if not row:
            return {}
        return {"summary": str(row[0] or ""), "created_at": str(row[1] or "")}

    async def save_entities(self, user_id: str, conv_id: str, entities: list[dict]) -> int:
        """实体写侧（与 L2PostgresMemory.save_entities 方法面一致，plans/0915-09）。

        幂等：UNIQUE(user_id, etype, name, conv_id)，同会话重写只更新 detail；
        同一会话同一泛称若对应多个不同 detail，先用稳定局部后缀拆开，避免静默覆盖；
        非法条目静默跳过，绝不抛错阻塞保存路径。
        """
        rows = prepare_entity_rows(user_id, conv_id, entities)
        if not rows:
            return 0
        conn = self._get_conn()
        conn.executemany(
            """INSERT INTO memory_entities
                   (user_id, conv_id, etype, name, detail, entity_id,
                    evidence_source, confidence, evidence_ref, source_message_seq, evidence_quote)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (user_id, etype, name, conv_id)
               DO UPDATE SET detail = CASE
                               WHEN excluded.detail = '' OR instr(memory_entities.detail, excluded.detail) > 0
                               THEN memory_entities.detail
                               ELSE substr(trim(memory_entities.detail || '；' || excluded.detail, '；'), 1, 600)
                             END,
                             evidence_source = excluded.evidence_source,
                             confidence = max(memory_entities.confidence, excluded.confidence),
                             evidence_ref = CASE WHEN excluded.evidence_ref = ''
                               THEN memory_entities.evidence_ref ELSE excluded.evidence_ref END,
                             source_message_seq = CASE WHEN excluded.evidence_ref = ''
                               THEN memory_entities.source_message_seq ELSE excluded.source_message_seq END,
                             evidence_quote = CASE WHEN excluded.evidence_ref = ''
                               THEN memory_entities.evidence_quote ELSE excluded.evidence_quote END""",
            rows,
        )
        conn.commit()
        return len(rows)

    async def save_entity_relations(self, user_id: str, conv_id: str,
                                    relations: list[dict]) -> int:
        """Append idempotent relation edges, accepting only this user's entity IDs."""
        rows = prepare_entity_relation_rows(user_id, conv_id, relations)
        if not rows:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        known_rows = cursor.execute(
            "SELECT DISTINCT entity_id FROM memory_entities WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        known = {str(row[0] or "") for row in known_rows}
        rows = [row for row in rows if row[3] in known and (not row[5] or row[5] in known)]
        if not rows:
            return 0
        cursor.executemany(
            """INSERT OR IGNORE INTO memory_entity_relations
                   (user_id, conv_id, edge_key, subject_entity_id, predicate,
                    object_entity_id, object_value, evidence_source, evidence_ref,
                    confidence, source_message_seq, evidence_quote)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return cursor.rowcount

    # ── 词块本（plans/0916-11，与 L2PostgresMemory 同方法面）────────────
    # 双后端硬约定（0916-09）：四个 chunk_* 方法两边必须成对实现——API 路由
    # 直接调 l2 方法，缺一个就是线上 500 或被 hasattr 兜成静默空值。

    async def save_chunk_candidates(self, user_id: str, conv_id: str,
                                    candidates: list[dict]) -> int:
        """候选落库（幂等）：UNIQUE(user_id, norm)。

        ON CONFLICT 只在已有行仍是 candidate 时刷新来源/语境信息
        （用户已收藏的条目绝不被重盘归纳改写）；非法条目（缺 text/norm 或
        kind/status 越界）静默跳过，绝不抛错阻塞保存路径。
        """
        from engine.chunk_evidence import CHUNK_KINDS, CHUNK_STATUSES
        rows = []
        for c in candidates or []:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text") or "").strip()[:80]
            norm = str(c.get("norm") or "").strip()[:120]
            kind = str(c.get("kind") or "collocation").strip().lower()
            if not text or not norm or kind not in CHUNK_KINDS:
                continue
            rows.append((
                user_id, text, norm, kind,
                str(c.get("source_type") or "tape")[:20],
                str(c.get("source_id") or str(conv_id or ""))[:120],
                int(c["source_turn"]) if str(c.get("source_turn") or "").strip().isdigit() else None,
                str(c.get("context_snippet") or "")[:200],
                str(c.get("gloss") or "")[:80],
                str(c.get("example") or "")[:200],
                str(c.get("slot") or "")[:40],
                str(c.get("reason") or "")[:160],
            ))
        if not rows:
            return 0
        conn = self._get_conn()
        conn.executemany(
            """INSERT INTO chunk_items (user_id, text, norm, kind, status,
                                        source_type, source_id, source_turn,
                                        context_snippet, gloss, example, slot, reason)
               VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, norm) DO UPDATE SET
                 text = excluded.text, kind = excluded.kind,
                 source_type = excluded.source_type, source_id = excluded.source_id,
                 source_turn = excluded.source_turn,
                 context_snippet = excluded.context_snippet,
                 gloss = excluded.gloss, example = excluded.example,
                 slot = excluded.slot, reason = excluded.reason,
                 updated_at = CURRENT_TIMESTAMP
               WHERE chunk_items.status = 'candidate'""",
            rows,
        )
        conn.commit()
        return len(rows)

    async def list_chunks(self, user_id: str, status: str | None = None,
                          limit: int = 200) -> list[dict]:
        """词块列表（本子读侧）：可选按状态过滤，新收的在前。"""
        safe_limit = max(1, min(int(limit), 500))
        conn = self._get_conn()
        cursor = conn.cursor()
        if status:
            cursor.execute(
                """SELECT * FROM chunk_items WHERE user_id = ? AND status = ?
                   ORDER BY updated_at DESC, id DESC LIMIT ?""",
                (user_id, str(status), safe_limit))
        else:
            cursor.execute(
                """SELECT * FROM chunk_items WHERE user_id = ?
                   ORDER BY updated_at DESC, id DESC LIMIT ?""",
                (user_id, safe_limit))
        return [dict(r) for r in cursor.fetchall()]

    async def list_chunks_for_notebook(self, user_id: str, bucket: str = "",
                                       offset: int = 0, limit: int = 50) -> dict:
        """词块本只读分页（plans/0923-05 §4.4）：按真实时间分桶 + 组内排序 + 切页。

        与旧 `list_chunks` **并行、互不影响**：这里不设 200/500 条截断，
        专门服务"放久的候选"折叠后仍能逐页取回。分桶/排序/切页全在
        `engine.chunk_evidence` 的共用纯函数里做（与 PG 侧同签名同返回键）。
        """
        from engine.chunk_evidence import page_notebook_chunks
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunk_items WHERE user_id = ? ORDER BY id",
                       (user_id,))
        rows = [dict(r) for r in cursor.fetchall()]
        return page_notebook_chunks(rows, bucket, offset=offset, limit=limit)

    async def update_chunk_status(self, user_id: str, chunk_id: str,
                                  status: str) -> bool:
        """状态流转（append-only：每次流转追加进 status_history 留底）。

        收藏 = candidate→saved；撤销收藏 = saved→candidate（照方案 §4.3，
        退回候选不删行）；机会轮练过 = saved→activated；会用 = →mastered。
        目标状态越界或行不存在返回 False。
        """
        from engine.chunk_evidence import CHUNK_STATUSES, loads_json_list
        status = str(status or "").strip().lower()
        if status not in CHUNK_STATUSES:
            return False
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, status, status_history FROM chunk_items "
            "WHERE user_id = ? AND id = ?",
            (user_id, str(chunk_id)))
        row = cursor.fetchone()
        if not row:
            return False
        history = loads_json_list(row["status_history"])
        from datetime import datetime, timezone
        history.append({
            "from": str(row["status"] or ""),
            "to": status,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        cursor.execute(
            """UPDATE chunk_items SET status = ?, status_history = ?,
               updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (status, json.dumps(history, ensure_ascii=False), row["id"]))
        conn.commit()
        return True

    async def record_chunk_use(self, user_id: str, chunk_id: str,
                               event: dict) -> dict:
        """记录一次真实使用（F13 会用确认的写侧，二期启用；一期先落方法面）。

        use_events 追加 [{conv_id, turn, via_opportunity, at}]，use_count +1，
        last_use_at 刷新。返回 {use_count, status} 供调用方做状态流转判定。
        """
        from engine.chunk_evidence import loads_json_list
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, use_count, use_events, status FROM chunk_items "
            "WHERE user_id = ? AND id = ?",
            (user_id, str(chunk_id)))
        row = cursor.fetchone()
        if not row:
            return {}
        events = loads_json_list(row["use_events"])
        from datetime import datetime, timezone
        ev = dict(event or {})
        ev.setdefault("at", datetime.now(timezone.utc).isoformat())
        events.append({
            "conv_id": str(ev.get("conv_id") or "")[:120],
            "turn": int(ev["turn"]) if str(ev.get("turn") or "").isdigit() else None,
            "via_opportunity": bool(ev.get("via_opportunity")),
            "at": str(ev.get("at") or ""),
        })
        use_count = int(row["use_count"] or 0) + 1
        cursor.execute(
            """UPDATE chunk_items SET use_count = ?, use_events = ?,
               last_use_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (use_count, json.dumps(events, ensure_ascii=False), row["id"]))
        conn.commit()
        return {"use_count": use_count, "status": str(row["status"] or "")}

    async def list_recent_openings(self, user_id: str, limit: int = 5) -> list[dict]:
        """最近 N 盘各自的首条 AI 消息（与 L2PostgresMemory 方法面一致，
        plans/0916-09 开场白跨盘去重）。本机开发环境常跑 local 后端，
        不实现的话去重会在开发机上静默失效。"""
        safe_limit = max(1, min(int(limit), 20))
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT conv_id, max(created_at) AS latest
               FROM conversation_messages
               WHERE user_id = ? AND role IN ('user', 'assistant')
                 AND length(coalesce(content, '')) > 0
               GROUP BY conv_id ORDER BY latest DESC LIMIT ?""",
            (user_id, safe_limit))
        convs = [dict(r) for r in cursor.fetchall()]
        openings: list[dict] = []
        for conv in convs:
            cursor.execute(
                """SELECT content, created_at FROM conversation_messages
                   WHERE user_id = ? AND conv_id = ? AND role = 'assistant'
                     AND length(coalesce(content, '')) > 0
                   ORDER BY seq ASC, id ASC LIMIT 1""",
                (user_id, str(conv["conv_id"] or "")))
            row = cursor.fetchone()
            if row:
                openings.append({
                    "conv_id": str(conv["conv_id"] or ""),
                    "opening": str(row["content"] or "").strip(),
                    "created_at": str(row["created_at"] or ""),
                })
        return openings

    async def conv_message_times(self, user_id: str, conv_id: str) -> list[dict]:
        """本盘每条落库消息的时间（plans/0916-23，与 L2PostgresMemory 方法面一致）。

        role/content/created_at 按 seq 升序——历史时间标签的数据源。
        """
        if not str(conv_id or "").strip():
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT role, content, created_at FROM conversation_messages
               WHERE user_id = ? AND conv_id = ?
               ORDER BY seq ASC""",
            (user_id, str(conv_id).strip()))
        return [dict(row) for row in cursor.fetchall()]

    async def conv_message_stats(self, user_id: str, conv_id: str) -> dict:
        """本盘消息统计（plans/0916-23 议题2）：总条数 + 用户轮数。"""
        if not str(conv_id or "").strip():
            return {"messages": 0, "user_turns": 0}
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT count(*) AS n,
                      sum(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_turns
               FROM conversation_messages
               WHERE user_id = ? AND conv_id = ?
                 AND role IN ('user', 'assistant')
                 AND length(coalesce(content, '')) > 0""",
            (user_id, str(conv_id).strip()))
        row = cursor.fetchone()
        if not row:
            return {"messages": 0, "user_turns": 0}
        return {
            "messages": int(row["n"] or 0),
            "user_turns": int(row["user_turns"] or 0),
        }

    async def count_conv_messages(self, user_id: str, conv_id: str) -> int:
        """一盘已有多少条落库消息（与 L2PostgresMemory 方法面一致，
        plans/0916-09：非空盘不再发开场白）。"""
        if not str(conv_id or "").strip():
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT count(*) AS n FROM conversation_messages
               WHERE user_id = ? AND conv_id = ?
                 AND role IN ('user', 'assistant')
                 AND length(coalesce(content, '')) > 0""",
            (user_id, str(conv_id).strip()))
        row = cursor.fetchone()
        return int(row["n"]) if row else 0

    async def last_conv_message_at(self, user_id: str, conv_id: str) -> str:
        """本盘最后一条落库消息的时间（ISO 字符串；没消息返回空串）。

        与 L2PostgresMemory 方法面一致（plans/0916-16 修复项 2，照 0916-09 的教训：
        只写一边会让开发机静默失效）。开场守卫靠它判断"是不是刚聊过"。
        返回的是 SQLite 的 CURRENT_TIMESTAMP 文本（UTC、无时区后缀），
        调用方 `_minutes_since_iso` 按 UTC 解析，两端口径一致。
        """
        if not str(conv_id or "").strip():
            return ""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT max(created_at) AS latest FROM conversation_messages
               WHERE user_id = ? AND conv_id = ?
                 AND role IN ('user', 'assistant')
                 AND length(coalesce(content, '')) > 0""",
            (user_id, str(conv_id).strip()))
        row = cursor.fetchone()
        return str(row["latest"] or "") if row else ""

    async def recent_conv_messages(self, user_id: str, conv_id: str,
                                   limit: int = 6) -> list[dict]:
        """Read a bounded same-tape dialogue window for a return opening."""
        if not str(conv_id or "").strip():
            return []
        cursor = self._get_conn().cursor()
        cursor.execute(
            """SELECT role, content FROM conversation_messages
               WHERE user_id = ? AND conv_id = ?
                 AND role IN ('user', 'assistant')
                 AND length(coalesce(content, '')) > 0
               ORDER BY seq DESC LIMIT ?""",
            (user_id, str(conv_id).strip(), max(1, min(int(limit), 12))),
        )
        return [dict(row) for row in reversed(cursor.fetchall())]

    async def search_entities_for_text(self, user_id: str, query_text: str,
                                       limit: int = 12) -> list[dict]:
        """Find older relevant entities beyond the startup cache without an LLM."""
        query_text = " ".join(str(query_text or "").split())[:1200]
        terms = entity_search_terms(query_text)
        safe_limit = max(1, min(int(limit), 24))
        clauses = ["(length(e.name) >= 2 AND instr(lower(?), lower(e.name)) > 0)"]
        params: list = [user_id, query_text.casefold()]
        for term in terms:
            clauses.append("(lower(e.name) LIKE ? OR lower(e.detail) LIKE ?)")
            pattern = f"%{term}%"
            params.extend((pattern, pattern))
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT e.conv_id, e.etype, e.name, e.detail, e.entity_id,
                       e.evidence_source, e.confidence, e.evidence_ref, e.evidence_quote,
                       e.source_message_seq, e.created_at, m.created_at AS observed_at
                FROM memory_entities e
                LEFT JOIN conversation_messages m
                  ON m.user_id = e.user_id AND m.conv_id = e.conv_id AND m.seq = e.source_message_seq
                WHERE e.user_id = ? AND ({' OR '.join(clauses)})
                ORDER BY CASE WHEN length(e.name) >= 2
                                   AND instr(lower(?), lower(e.name)) > 0
                              THEN 1 ELSE 0 END DESC,
                         COALESCE(m.created_at, e.created_at) DESC LIMIT 240""",
            tuple([*params, query_text.casefold()]),
        )
        entities = merge_entity_search_observations(
            [dict(row) for row in cursor.fetchall()], query_text, limit=safe_limit,
        )
        entity_ids = [str(entity["entity_id"]) for entity in entities]
        if not entity_ids:
            return []
        marks = ",".join("?" for _ in entity_ids)
        cursor.execute(
            f"""SELECT r.subject_entity_id, r.predicate, r.object_entity_id,
                       r.object_value, r.evidence_source, r.evidence_ref, r.confidence,
                       r.source_message_seq, r.evidence_quote, r.created_at,
                       m.created_at AS observed_at,
                       (SELECT e.name FROM memory_entities e
                        WHERE e.user_id = r.user_id AND e.entity_id = r.object_entity_id
                        ORDER BY e.created_at DESC LIMIT 1) AS object_name
                FROM memory_entity_relations r
                LEFT JOIN conversation_messages m
                  ON m.user_id = r.user_id AND m.conv_id = r.conv_id AND m.seq = r.source_message_seq
                WHERE r.user_id = ?
                  AND (r.subject_entity_id IN ({marks}) OR r.object_entity_id IN ({marks}))
                ORDER BY COALESCE(m.created_at, r.created_at) DESC LIMIT 120""",
            (user_id, *entity_ids, *entity_ids),
        )
        relation_rows = [dict(row) for row in cursor.fetchall()]
        neighbor_ids = entity_neighborhood_ids(
            set(entity_ids), relation_rows, limit=min(4, safe_limit),
        )
        neighbor_rows = []
        if neighbor_ids:
            neighbor_marks = ",".join("?" for _ in neighbor_ids)
            cursor.execute(
                f"""SELECT e.conv_id, e.etype, e.name, e.detail, e.entity_id,
                           e.evidence_source, e.confidence, e.evidence_ref,
                           e.evidence_quote, e.source_message_seq, e.created_at,
                           m.created_at AS observed_at
                    FROM memory_entities e
                    LEFT JOIN conversation_messages m
                      ON m.user_id=e.user_id AND m.conv_id=e.conv_id AND m.seq=e.source_message_seq
                    WHERE e.user_id=? AND e.entity_id IN ({neighbor_marks})
                    ORDER BY e.entity_id, COALESCE(m.created_at, e.created_at) DESC""",
                (user_id, *neighbor_ids),
            )
            best_neighbor_by_id: dict[str, dict] = {}
            for row in cursor.fetchall():
                neighbor = dict(row)
                neighbor_id = str(neighbor.get("entity_id") or "")
                if neighbor_id and neighbor_id not in best_neighbor_by_id:
                    best_neighbor_by_id[neighbor_id] = neighbor
            neighbor_rows = list(best_neighbor_by_id.values())
        return assemble_entity_neighborhood(
            entities, neighbor_rows, relation_rows,
            neighbor_limit=min(4, safe_limit),
        )

    async def list_entities(self, user_id: str, name: str | None = None,
                            limit: int = 50) -> list[dict]:
        """实体读侧（与 L2PostgresMemory.list_entities 方法面一致）：最新在前。"""
        safe_limit = max(1, min(int(limit), 200))
        conn = self._get_conn()
        cursor = conn.cursor()
        if name:
            cursor.execute(
                """SELECT e.conv_id, e.etype, e.name, e.detail, e.entity_id,
                          e.evidence_source, e.confidence, e.evidence_ref, e.evidence_quote,
                          e.source_message_seq, e.created_at, m.created_at AS observed_at
                   FROM memory_entities e
                   LEFT JOIN conversation_messages m
                     ON m.user_id = e.user_id AND m.conv_id = e.conv_id AND m.seq = e.source_message_seq
                   WHERE e.user_id = ? AND e.name LIKE ?
                   ORDER BY COALESCE(m.created_at, e.created_at) DESC LIMIT ?""",
                (user_id, f"%{name.strip()}%", safe_limit))
        else:
            cursor.execute(
                """SELECT e.conv_id, e.etype, e.name, e.detail, e.entity_id,
                          e.evidence_source, e.confidence, e.evidence_ref, e.evidence_quote,
                          e.source_message_seq, e.created_at, m.created_at AS observed_at
                   FROM memory_entities e
                   LEFT JOIN conversation_messages m
                     ON m.user_id = e.user_id AND m.conv_id = e.conv_id AND m.seq = e.source_message_seq
                   WHERE e.user_id = ?
                   ORDER BY COALESCE(m.created_at, e.created_at) DESC LIMIT ?""",
                (user_id, safe_limit))
        entities = [dict(row) for row in cursor.fetchall()]
        entity_ids = list(dict.fromkeys(
            str(row.get("entity_id") or "") for row in entities
            if str(row.get("entity_id") or "")
        ))
        if not entity_ids:
            return entities
        marks = ",".join("?" for _ in entity_ids)
        cursor.execute(
            f"""SELECT r.subject_entity_id, r.predicate, r.object_entity_id,
                       r.object_value, r.evidence_source, r.evidence_ref, r.confidence,
                       r.source_message_seq, r.evidence_quote, r.created_at, m.created_at AS observed_at,
                       (SELECT e.name FROM memory_entities e
                        WHERE e.user_id = r.user_id AND e.entity_id = r.object_entity_id
                        ORDER BY e.created_at DESC LIMIT 1) AS object_name
                FROM memory_entity_relations r
                LEFT JOIN conversation_messages m
                  ON m.user_id = r.user_id AND m.conv_id = r.conv_id AND m.seq = r.source_message_seq
                WHERE r.user_id = ? AND r.subject_entity_id IN ({marks})
                ORDER BY COALESCE(m.created_at, r.created_at) DESC LIMIT 120""",
            (user_id, *entity_ids),
        )
        relations_by_subject: dict[str, list[dict]] = {}
        for row in cursor.fetchall():
            relation = dict(row)
            relations_by_subject.setdefault(str(relation.pop("subject_entity_id")), []).append(relation)
        for entity in entities:
            entity["relations"] = relations_by_subject.get(str(entity.get("entity_id") or ""), [])
        return entities

    async def list_message_chunks(self, user_id: str, limit: int = 60) -> list[dict]:
        """原始消息进检索池（与 L2PostgresMemory.list_message_chunks 方法面一致，
        plans/0915-14 书§3.3.5 上下文感知检索）：滑窗切块 + 归档摘要前缀。
        sqlite 的 session_archives.session_id 直接存 conv_id 文本（无 uuid 映射）。
        """
        safe_limit = max(1, min(int(limit), 120))
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT conv_id, max(created_at) AS latest
               FROM conversation_messages
               WHERE user_id = ? AND role IN ('user', 'assistant')
                 AND length(coalesce(content, '')) > 0
               GROUP BY conv_id ORDER BY latest DESC LIMIT 8""",
            (user_id,))
        convs = [dict(r) for r in cursor.fetchall()]
        if not convs:
            return []
        cursor.execute(
            "SELECT session_id, summary FROM session_archives WHERE user_id = ?",
            (user_id,))
        summaries = {str(r["session_id"]): str(r["summary"] or "") for r in cursor.fetchall()}
        cursor.execute(
            "SELECT conv_id, name, scene FROM conversation_meta WHERE user_id = ?",
            (user_id,))
        metas = {
            str(r["conv_id"]): (str(r["name"] or ""), str(r["scene"] or ""))
            for r in cursor.fetchall()
        }
        rows: list[dict] = []
        for conv in convs:
            conv_id = str(conv["conv_id"] or "")
            cursor.execute(
                """SELECT role, content, created_at FROM conversation_messages
                   WHERE user_id = ? AND conv_id = ?
                     AND role IN ('user', 'assistant')
                     AND length(coalesce(content, '')) > 0
                   ORDER BY seq, created_at LIMIT 60""",
                (user_id, conv_id))
            msgs = [dict(r) for r in cursor.fetchall()]
            if not msgs:
                continue
            prefix = summaries.get(conv_id, "")
            if not prefix:
                name, scene = metas.get(conv_id, ("", ""))
                prefix = name or scene or "更早的一盘对话"
            prefix = " ".join(prefix.split())[:60]
            for start in range(0, len(msgs), 4):
                chunk = msgs[start:start + 6]
                if not chunk:
                    continue
                body = " / ".join(
                    ("用户" if m["role"] == "user" else "AI") + ":" + str(m["content"])
                    for m in chunk
                )
                if not body.strip():
                    continue
                rows.append({
                    "candidate_id": f"{conv_id}#msgs-{start}-{start + len(chunk) - 1}",
                    "session_id": conv_id,
                    "summary": f"旧磁带原文片段（出自：{prefix}）={body}"[:1800],
                    "searchable": f"【出自旧对话：{prefix}】" + body[:2400],
                    "error_count": 0,
                    "created_at": chunk[-1]["created_at"],
                    "source": "conversation_message_chunk",
                })
                if len(rows) >= safe_limit:
                    return rows
        return rows

    async def search_message_chunks(self, user_id: str, query: str,
                                    limit: int = 60) -> list[dict]:
        """Search query-centered windows across all saved messages for one user.

        This runs only for an explicit episodic-recall turn. The query is
        bounded, matching rows are ranked in SQL, and only nearby message
        windows are returned to the retriever.
        """
        from memory.entities import entity_search_terms
        from memory.history_retrieval import build_message_chunk_candidates

        query = " ".join(str(query or "").split())[:1200]
        terms = entity_search_terms(query, max_terms=12)
        if not terms:
            return []
        safe_limit = max(1, min(int(limit), 120))
        seed_limit = min(safe_limit, 60)
        params = {"user_id": user_id, "seed_limit": seed_limit,
                  "window_limit": seed_limit * 6}
        clauses = []
        score_parts = []
        for index, term in enumerate(terms):
            key = f"term{index}"
            params[key] = f"%{term}%"
            clause = f"lower(m.content) LIKE :{key}"
            clauses.append(clause)
            score_parts.append(f"CASE WHEN {clause} THEN 1 ELSE 0 END")
        score_expr = " + ".join(score_parts)
        cursor = self._get_conn().cursor()
        cursor.execute(
            f"""WITH matched AS (
                   SELECT m.user_id, m.conv_id, m.seq, m.created_at,
                          ({score_expr}) AS match_score
                   FROM conversation_messages m
                   WHERE m.user_id = :user_id
                     AND m.role IN ('user', 'assistant')
                     AND length(coalesce(m.content, '')) > 0
                     AND ({' OR '.join(clauses)})
                   ORDER BY match_score DESC, m.created_at DESC
                   LIMIT :seed_limit
               )
               SELECT matched.conv_id, matched.seq AS seed_seq,
                      matched.match_score, matched.created_at AS seed_created_at,
                      nearby.seq, nearby.role, nearby.content, nearby.created_at
               FROM matched
               JOIN conversation_messages nearby
                 ON nearby.user_id = matched.user_id
                AND nearby.conv_id = matched.conv_id
                AND nearby.seq BETWEEN matched.seq - 2 AND matched.seq + 3
                AND nearby.role IN ('user', 'assistant')
                AND length(coalesce(nearby.content, '')) > 0
               ORDER BY matched.match_score DESC, matched.created_at DESC,
                        matched.conv_id, matched.seq, nearby.seq
               LIMIT :window_limit""",
            params,
        )
        return build_message_chunk_candidates(
            [dict(row) for row in cursor.fetchall()], query, limit=safe_limit
        )


    async def get_mastered(self, user_id: str, limit: int = 50) -> list[dict]:
        """Get mastered expressions for a user."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT expression, confidence
            FROM mastered_expressions
            WHERE user_id = ?
            ORDER BY confidence DESC LIMIT ?""",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_id, summary, error_count,
                   cefr_before, cefr_after, duration_seconds, created_at
            FROM session_archives
            WHERE user_id = ? AND created_at > datetime('now', '-{} days')
            ORDER BY created_at DESC
        """.format(days), (user_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    async def increment_session_count(self, user_id: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_daily_state 
            (user_id, date, session_count, updated_at)
            VALUES (?, CURRENT_DATE,
                COALESCE((SELECT session_count FROM user_daily_state WHERE user_id=? AND date=CURRENT_DATE), 0) + 1,
                CURRENT_TIMESTAMP)
        """, (user_id, user_id))
        conn.commit()

    async def list_history_candidates(self, user_id: str, limit: int = 100, *,
                                      query: str = "") -> list[dict]:
        """Return bounded, evidence-bearing episode candidates for the retriever.

        `query` 与 Postgres 版同义：先把词面命中的老归档捞进池子，排在最新 N 条前面。
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        safe_limit = max(1, min(int(limit), 200))
        patterns: list = []
        if query:
            try:
                from .history_retrieval import lexical_tokens
            except Exception:
                try:
                    from memory.history_retrieval import lexical_tokens  # type: ignore
                except Exception:
                    lexical_tokens = None  # type: ignore
            if lexical_tokens is not None:
                try:
                    patterns = [
                        f"%{t}%" for t in lexical_tokens(query) if len(str(t)) >= 2
                    ][:8]
                except Exception:
                    patterns = []
        cursor.execute("""
            SELECT session_id, summary, error_count, created_at
            FROM session_archives
            WHERE user_id = ? AND summary IS NOT NULL
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, safe_limit))
        rows = [dict(row) for row in cursor.fetchall()]
        matched_rows: list = []
        if patterns:
            clause = " OR ".join(["summary LIKE ?"] * len(patterns))
            cursor.execute(f"""
                SELECT session_id, summary, error_count, created_at
                FROM session_archives
                WHERE user_id = ? AND summary IS NOT NULL AND ({clause})
                ORDER BY created_at DESC LIMIT ?
            """, (user_id, *patterns, safe_limit))
            matched_rows = [dict(row) for row in cursor.fetchall()]
        # A cassette handoff is the precise executable checkpoint. Session
        # archives are useful summaries, but they can omit the last unfinished
        # request. Search both stores only after the router has established that
        # the learner explicitly asked to recall an earlier cassette.
        cursor.execute(
            """SELECT conv_id, handoff_json, updated_at
               FROM cassette_handoffs
               WHERE user_id = ?
               ORDER BY updated_at DESC LIMIT ?""",
            (user_id, safe_limit),
        )
        handoff_rows = []
        for record in cursor.fetchall():
            try:
                handoff = json.loads(record["handoff_json"] or "{}")
            except (TypeError, ValueError):
                continue
            searchable, summary = self._cassette_handoff_text(handoff)
            if not searchable:
                continue
            handoff_rows.append({
                "session_id": str(record["conv_id"] or ""),
                "summary": summary,
                "searchable": searchable,
                "error_count": 0,
                "created_at": record["updated_at"],
                "source": "cassette_handoff",
            })
        seen: set = set()
        combined_rows = []
        for item in handoff_rows + [
            {**row, "searchable": str(row.get("summary") or ""), "source": "session_archive"}
            for row in list(matched_rows) + list(rows)
        ]:
            key = (item.get("source"), str(item.get("session_id") or ""))
            if key in seen:
                continue
            seen.add(key)
            combined_rows.append(item)
        combined_rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        matched_keys = {("session_archive", str(r.get("session_id") or "")) for r in matched_rows}
        # 命中组排前面，组内仍按新的在前（两次稳定排序）。
        combined_rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        combined_rows.sort(key=lambda item: (
            0 if (item.get("source"), str(item.get("session_id") or "")) in matched_keys else 1))
        return combined_rows[:safe_limit]

    async def search_history(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        candidates = await self.list_history_candidates(user_id, limit=100, query=query)
        ranked = rank_history_sparse(candidates, query, top_k=top_k)
        for row in ranked:
            row.pop("searchable", None)
            row["retrieval_mode"] = "sparse"
        return ranked

    @staticmethod
    def _cassette_handoff_text(handoff: dict) -> tuple[str, str]:
        """Return a searchable body plus a bounded learner-facing checkpoint."""
        scene = handoff.get("scene") or {}
        plan = handoff.get("conversation_plan") or {}
        ledger = handoff.get("scene_ledger") or {}

        def values(raw) -> list[str]:
            if isinstance(raw, list):
                return [str(item).strip() for item in raw if str(item).strip()]
            if raw in (None, ""):
                return []
            return [str(raw).strip()]

        fields = [
            ("场景", scene.get("brief") or scene.get("name")),
            ("角色", " / ".join(values(scene.get("learner_role")) + values(scene.get("tutor_role")))),
            ("当前步骤", scene.get("step") or plan.get("current_phase") or ledger.get("current_phase")),
            ("目标", scene.get("goal") or plan.get("real_world_goal") or plan.get("content_need")),
            ("下一动作", plan.get("next_counterpart_action")),
            ("待处理", "、".join(values(plan.get("open_items")) + values(ledger.get("open_items")))),
            ("已完成", "、".join(values(plan.get("completed_actions")) + values(ledger.get("completed_actions")))),
        ]
        checkpoint_parts = [
            f"{label}={str(value).strip()}"
            for label, value in fields if str(value or "").strip()
        ]
        recent_parts = []
        for item in list(handoff.get("recent_messages") or [])[-8:]:
            role = "用户" if item.get("role") == "user" else "AI"
            content = " ".join(str(item.get("content") or "").split())
            if content:
                recent_parts.append(f"{role}:{content[:240]}")
        checkpoint = "；".join(checkpoint_parts)
        recent = " / ".join(recent_parts)
        searchable = " ".join(part for part in (checkpoint, recent) if part).strip()
        summary = checkpoint
        if recent:
            summary = f"{checkpoint}；最近对话={recent}" if checkpoint else f"最近对话={recent}"
        return searchable[:2400], summary[:900]

    async def save_cassette_handoff(self, user_id: str, conv_id: str, handoff: dict) -> None:
        if not conv_id:
            return
        self._get_conn().execute(
            """INSERT INTO cassette_handoffs (user_id, conv_id, handoff_json, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, conv_id) DO UPDATE SET
                 handoff_json=excluded.handoff_json, updated_at=CURRENT_TIMESTAMP""",
            (user_id, conv_id, json.dumps(handoff or {}, ensure_ascii=False)),
        )
        self._get_conn().commit()

    async def load_cassette_handoff(self, user_id: str, conv_id: str) -> dict:
        if not conv_id:
            return {}
        row = self._get_conn().execute(
            "SELECT handoff_json FROM cassette_handoffs WHERE user_id=? AND conv_id=?",
            (user_id, conv_id),
        ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["handoff_json"] or "{}")
        except (TypeError, ValueError):
            return {}

    async def save_memory_curator_run(self, run: dict) -> None:
        self._get_conn().execute(
            """INSERT OR REPLACE INTO memory_curator_runs
               (run_id, user_id, session_id, conversation_id, trigger_name, status,
                candidates_json, accepted_json, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                str(run.get("run_id") or ""), str(run.get("user_id") or ""),
                str(run.get("session_id") or ""), str(run.get("conversation_id") or ""),
                str(run.get("trigger_name") or "session_end"), str(run.get("status") or "unknown"),
                json.dumps(run.get("candidates") or [], ensure_ascii=False),
                json.dumps(run.get("accepted_items") or [], ensure_ascii=False),
                json.dumps(run.get("metadata") or {}, ensure_ascii=False),
            ),
        )
        self._get_conn().commit()

    async def load_memory_curator_runs(self, user_id: str, limit: int = 20) -> list[dict]:
        rows = self._get_conn().execute(
            """SELECT * FROM memory_curator_runs WHERE user_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, max(1, min(int(limit), 100))),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for source, target, fallback in (
                ("candidates_json", "candidates", []),
                ("accepted_json", "accepted_items", []),
                ("metadata_json", "metadata", {}),
            ):
                try:
                    item[target] = json.loads(item.pop(source) or json.dumps(fallback))
                except (TypeError, ValueError):
                    item[target] = fallback
            result.append(item)
        return result

    @staticmethod
    def _lexical_units(text: str) -> set[str]:
        """Language-agnostic lexical units for short Chinese/English recalls."""
        lowered = (text or "").lower()
        english = {token for token in re.findall(r"[a-z0-9]+", lowered) if len(token) > 1}
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
        grams = {
            chinese[index:index + size]
            for size in (2, 3)
            for index in range(max(0, len(chinese) - size + 1))
        }
        return english | grams

    async def log_methodology_call(self, user_id: str, methodology_id: int, session_id: str, cooperation_score: float = 0.0) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO methodology_calls (user_id, methodology_id, session_id, cooperation_score)
            VALUES (?, ?, ?, ?)
        """, (user_id, methodology_id, session_id, cooperation_score))
        conn.commit()
        return str(cursor.lastrowid)

    async def store_embedding(self, name: str, instruction: str, embedding: list[float], cefr_level: str = "", scene_tags: list[str] = None, boost_score: float = 0.0, source: str = "") -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO methodology_vectors
            (name, instruction, example_dialogue, cefr_level, scene_tags, embedding, boost_score, source, updated_at)
            VALUES (?, ?, '', ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (name, instruction, cefr_level, json.dumps(scene_tags or []), json.dumps(embedding), boost_score, source))
        conn.commit()
        return str(cursor.lastrowid)

    async def get_daily_session_count(self, user_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(session_count, 0) as cnt
            FROM user_daily_state
            WHERE user_id = ? AND date = CURRENT_DATE
        """, (user_id,))
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    # ── Conversation persistence ─────────────────────────

    async def save_conversation(self, user_id: str, conv_id: str, name: str,
                                 target_lang: str, level: str, scene: str,
                                 messages: list[dict]):
        from memory.message_identity import assistant_turn_id
        # Append-only 策略：绝不删除已有消息，只追加 seq > 当前最大值的新消息
        # 这样即使前端多次 POST / 后端重启 / 竞态丢包，旧消息永远不丢
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO conversation_meta
            (user_id, conv_id, name, target_lang, level, scene, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (user_id, conv_id, name, target_lang, level, scene))
        # 查询当前已存的最大 seq
        cursor.execute(
            "SELECT COALESCE(MAX(seq), -1) AS max_seq FROM conversation_messages WHERE conv_id = ?",
            (conv_id,)
        )
        row = cursor.fetchone()
        max_seq = row["max_seq"] if row else -1
        # 只插入 seq > max_seq 的消息（按传入顺序递增）
        inserted = 0
        for i, msg in enumerate(messages):
            if i <= max_seq:
                continue
            ts = msg.get("timestamp") or msg.get("created_at") or datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO conversation_messages (user_id, conv_id, role, content, seq, created_at, turn_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, conv_id, msg.get("role", "user"), msg.get("content", ""), i, str(ts), assistant_turn_id(msg)))
            inserted += 1
        conn.commit()
        return {"inserted": inserted, "max_seq": max_seq}

    async def load_conversations(self, user_id: str, include_hidden: bool = False) -> list[dict]:
        """加载用户的会话列表。include_hidden=False 时只返回未隐藏的。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if include_hidden:
            cursor.execute("""
                SELECT conv_id, name, target_lang, level, scene, created_at, updated_at, hidden_at, expires_at
                FROM conversation_meta
                WHERE user_id = ? AND hidden_at IS NULL
                ORDER BY updated_at DESC
            """, (user_id,))
        else:
            # 默认只返回未隐藏且未过期的
            cursor.execute("""
                SELECT conv_id, name, target_lang, level, scene, created_at, updated_at, hidden_at, expires_at
                FROM conversation_meta
                WHERE user_id = ? AND hidden_at IS NULL
                ORDER BY updated_at DESC
            """, (user_id,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            conv = dict(row)
            cursor.execute("""
                SELECT role, content, created_at, turn_id FROM conversation_messages
                WHERE conv_id = ? ORDER BY seq ASC
            """, (conv["conv_id"],))
            msg_rows = cursor.fetchall()
            conv["messages"] = [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "turn_id": r["turn_id"],
                    "timestamp": r["created_at"],
                    "created_at": r["created_at"],
                }
                for r in msg_rows
            ]
            result.append(conv)
        return result

    # ── 删除一盘会话（回收站 / 彻底删除）────────────────────
    #
    # 与 L2PostgresMemory 方法面严格一致（同一个 API 路由会打到任一后端）。
    # 语义见 postgres 侧注释：删一盘必须同时清掉派生记忆（摘要/实体/交班条），
    # 否则「删掉了但 AI 还记得」。
    #
    # 键形态差异（这是本后端与 postgres 唯一的不同处）：sqlite 的
    # session_archives.session_id 直接存 conv_id 文本，没有 uuid5 映射，
    # 所以这里不能用 _as_uuid —— 用了就一行都删不掉。

    _CONV_DERIVED_TABLES = ("memory_entity_relations", "memory_entities", "cassette_handoffs")

    def _purge_conv_derived(self, cursor, user_id: str, conv_id: str) -> dict:
        """清掉一个会话派生出来的记忆（摘要/实体/交班条）。幂等，缺表不抛错。"""
        counts = {}
        for table in self._CONV_DERIVED_TABLES:
            try:
                cursor.execute(
                    f"DELETE FROM {table} WHERE user_id = ? AND conv_id = ?",
                    (user_id, conv_id))
                counts[table] = cursor.rowcount
            except Exception:
                counts[table] = 0
        try:
            cursor.execute(
                "DELETE FROM session_archives WHERE user_id = ? AND session_id = ?",
                (user_id, conv_id))
            counts["session_archives"] = cursor.rowcount
        except Exception:
            counts["session_archives"] = 0
        # plans/0923-06 §4.2.4／§5 第 12 条：与 postgres 侧同一条撤销 —— 练习盘会话末
        # 冷评估写入的五维证据，隐藏／彻底删该盘时必须沿来源删掉（列名 `source_conv_id`，
        # 不是 `conv_id`，故不能并入 `_CONV_DERIVED_TABLES`）。其它盘不写本表 ⇒ 零影响。
        try:
            cursor.execute(
                "DELETE FROM learner_profile_facts WHERE user_id = ? AND source_conv_id = ?",
                (user_id, conv_id))
            counts["learner_profile_facts"] = cursor.rowcount
        except Exception:
            counts["learner_profile_facts"] = 0
        # plans/0923-04 §4.4 阶段 2：与 postgres 侧同一条 —— 撤销这一盘的写入令牌，
        # 挡住"已登记、还没落库"的迟到作业在删盘后把该盘内容写回来。只置 revoked、
        # 不删行；别的盘零影响；缺表（老库未迁）不抛错。
        try:
            cursor.execute(
                "UPDATE learner_profile_write_leases SET status = 'revoked' "
                "WHERE user_id = ? AND status = 'active' AND scope_conv_id = ?",
                (user_id, conv_id))
            counts["write_leases_revoked"] = cursor.rowcount
        except Exception:
            counts["write_leases_revoked"] = 0
        return counts

    async def hide_conversation(self, user_id: str, conv_id: str, retention_days: int = 30) -> dict:
        """软删除（进回收站）：标记 hidden_at，并**立刻**清掉派生记忆。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversation_meta
            SET hidden_at = CURRENT_TIMESTAMP,
                expires_at = datetime('now', '+' || ? || ' days')
            WHERE user_id = ? AND conv_id = ?
        """, (retention_days, user_id, conv_id))
        existed = cursor.rowcount > 0
        dropped = self._purge_conv_derived(cursor, user_id, conv_id)
        try:
            cursor.execute(
                "SELECT memory_items FROM relationship_memory WHERE user_id=?", (user_id,)
            )
            row = cursor.fetchone()
            items = json.loads(row["memory_items"] or "[]") if row else []
            hidden_count = 0
            for item in items:
                if (isinstance(item, dict)
                        and str(item.get("source_conv_id") or "") == str(conv_id)
                        and not item.get("hidden_at")):
                    item["hidden_at"] = datetime.now(timezone.utc).isoformat()
                    hidden_count += 1
            if hidden_count:
                cursor.execute(
                    "UPDATE relationship_memory SET memory_items=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
                    (json.dumps(items, ensure_ascii=False), user_id),
                )
            dropped["relationship_memory_items_hidden"] = hidden_count
        except Exception:
            dropped["relationship_memory_items_hidden"] = 0
        cursor.execute(
            "UPDATE companion_persona_facts SET hidden_at=CURRENT_TIMESTAMP "
            "WHERE user_id=? AND source_conv_id=?", (user_id, conv_id),
        )
        dropped["companion_persona_facts_hidden"] = max(0, cursor.rowcount)
        cursor.execute(
            """UPDATE companion_persona_snapshots
               SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                   source_hash='', generated_at=NULL
               WHERE user_id=? AND persona_key='alice'""", (user_id,),
        )
        conn.commit()
        return {"status": "hidden", "conv_id": conv_id, "existed": existed,
                "retention_days": retention_days, "dropped": dropped}

    async def restore_conversation(self, user_id: str, conv_id: str) -> dict:
        """从回收站恢复：只把隐藏标记摘掉，原文原样回来。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversation_meta
            SET hidden_at = NULL, expires_at = NULL
            WHERE user_id = ? AND conv_id = ?
        """, (user_id, conv_id))
        existed = cursor.rowcount > 0
        try:
            cursor.execute(
                "SELECT memory_items FROM relationship_memory WHERE user_id=?", (user_id,)
            )
            row = cursor.fetchone()
            items = json.loads(row["memory_items"] or "[]") if row else []
            restored = 0
            for item in items:
                if (isinstance(item, dict)
                        and str(item.get("source_conv_id") or "") == str(conv_id)
                        and item.pop("hidden_at", None)):
                    restored += 1
            if restored:
                cursor.execute(
                    "UPDATE relationship_memory SET memory_items=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
                    (json.dumps(items, ensure_ascii=False), user_id),
                )
        except Exception:
            pass
        cursor.execute(
            "UPDATE companion_persona_facts SET hidden_at=NULL "
            "WHERE user_id=? AND source_conv_id=?", (user_id, conv_id),
        )
        cursor.execute(
            """UPDATE companion_persona_snapshots
               SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                   source_hash='', generated_at=NULL
               WHERE user_id=? AND persona_key='alice'""", (user_id,),
        )
        conn.commit()
        return {"status": "restored", "conv_id": conv_id, "existed": existed,
                "derived_memory": "dropped_on_hide"}

    async def purge_conversation(self, user_id: str, conv_id: str) -> dict:
        """物理删除一盘会话的全部行（不可恢复，幂等）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        deleted = self._purge_conv_derived(cursor, user_id, conv_id)
        cursor.execute(
            "DELETE FROM companion_persona_facts WHERE user_id=? AND source_conv_id=?",
            (user_id, conv_id),
        )
        deleted["companion_persona_facts"] = max(0, cursor.rowcount)
        cursor.execute(
            """UPDATE companion_persona_snapshots
               SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                   source_hash='', generated_at=NULL
               WHERE user_id=? AND persona_key='alice'""", (user_id,),
        )
        # plans/0923-08 §4.2：素材盘成盘记录是**该盘**的派生提要/分组/话题快照。
        # 与 expression_topic_assignments 同规矩 —— **故意不放进 hide 的派生清理**：
        # 隐藏后恢复要能重新读到这盘的盘面（新构建会换 conv_id，删了就永久打不开）；
        # 隐藏期间靠读取侧的 hidden 过滤保证不可见。**彻底删**（本方法 / 过期清理）删本表行。
        for table in ("conversation_messages", "conversation_meta",
                      "expression_topic_assignments", "material_tape_builds"):
            cursor.execute(
                f"DELETE FROM {table} WHERE user_id = ? AND conv_id = ?",
                (user_id, conv_id))
            deleted[table] = cursor.rowcount
        conn.commit()
        return {"conv_id": conv_id, "deleted": deleted,
                "rows": sum(deleted.values())}

    async def purge_expired_conversations(self, user_id: str = None) -> dict:
        """物理删除已过期的隐藏会话。返回 {deleted, purged_conv_ids}。

        原文与派生记忆一起清（旧版只删 conversation_meta，会留下永久删不掉的
        对话原文，与隐私承诺相冲 —— postgres 侧早就修了，这里补齐对齐）。
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        if user_id:
            cursor.execute("""
                SELECT user_id, conv_id FROM conversation_meta
                WHERE hidden_at IS NOT NULL AND expires_at < datetime('now') AND user_id = ?
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT user_id, conv_id FROM conversation_meta
                WHERE hidden_at IS NOT NULL AND expires_at < datetime('now')
            """)
        expired = [(row["user_id"], row["conv_id"]) for row in cursor.fetchall()]
        for uid, conv_id in expired:
            self._purge_conv_derived(cursor, uid, conv_id)
            cursor.execute(
                "DELETE FROM companion_persona_facts WHERE user_id=? AND source_conv_id=?",
                (uid, conv_id),
            )
            cursor.execute(
                """UPDATE companion_persona_snapshots
                   SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                       source_hash='', generated_at=NULL
                   WHERE user_id=? AND persona_key='alice'""", (uid,),
            )
            for table in ("conversation_messages", "conversation_meta",
                          "expression_topic_assignments", "material_tape_builds"):
                cursor.execute(
                    f"DELETE FROM {table} WHERE user_id = ? AND conv_id = ?",
                    (uid, conv_id))
        conn.commit()
        return {"deleted": len(expired),
                "purged_conv_ids": [conv_id for _, conv_id in expired]}

    # ── 练习盘话题分配（plans/0923-06 §4.1.2）──────────────────────
    #
    # 只服务练习盘（scene='expression'）。存建盘**当时的来源快照**，故后续扩库／改文案／
    # 热点过期都不会改写旧盘的出处。**故意不放进 hide 的派生清理**：隐藏盘要能在恢复后
    # 重新读到原话题快照（§4.4）；隐藏期间靠读取侧的 hidden 过滤保证不可见。
    # 彻底删（purge_conversation / purge_expired_conversations）会删掉本表行。

    async def put_expression_topic_assignment(self, user_id: str, conv_id: str,
                                              assignment: dict) -> dict:
        """写入一盘的建盘快照。同一 (user_id, conv_id) **幂等且不覆写**。

        不覆写是关键（§4.1.2「同盘重试读取原分配，不重新抽题」）：已分配过的盘
        再次调用只会得到 `exists`，原出处不被后来的抽题改掉。
        """
        data = assignment or {}
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO expression_topic_assignments
                (user_id, conv_id, topic_id, source_title, source_url,
                 source_date, prompt_text, assigned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    COALESCE(NULLIF(?, ''), CURRENT_TIMESTAMP))
        """, (
            user_id, conv_id,
            str(data.get("topic_id") or ""),
            str(data.get("source_title") or ""),
            str(data.get("source_url") or ""),
            str(data.get("source_date") or ""),
            str(data.get("prompt_text") or ""),
            str(data.get("assigned_at") or ""),
        ))
        inserted = cursor.rowcount > 0
        conn.commit()
        return {"status": "assigned" if inserted else "exists",
                "conv_id": conv_id, "inserted": inserted}

    async def get_expression_topic_assignment(self, user_id: str, conv_id: str,
                                              include_hidden: bool = False) -> dict:
        """读一盘的分配快照。隐藏盘默认**不可见**（§4.4），恢复后才可见。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        sql = """
            SELECT a.user_id, a.conv_id, a.topic_id, a.source_title, a.source_url,
                   a.source_date, a.prompt_text, a.assigned_at
            FROM expression_topic_assignments a
            LEFT JOIN conversation_meta m
                   ON m.user_id = a.user_id AND m.conv_id = a.conv_id
            WHERE a.user_id = ? AND a.conv_id = ?
        """
        if not include_hidden:
            # 无 meta 行（LEFT JOIN 出 NULL）不算隐藏 —— 只有明确 hidden_at 非空才挡住
            sql += " AND m.hidden_at IS NULL"
        cursor.execute(sql, (user_id, conv_id))
        row = cursor.fetchone()
        return dict(row) if row else {}

    async def list_recent_expression_topic_assignments(self, user_id: str,
                                                       limit: int = 5,
                                                       include_hidden: bool = True) -> list:
        """最近若干盘的分配（新盘据此排除重复的 topic_id，§4.1.2）。"""
        try:
            n = max(1, int(limit))
        except (TypeError, ValueError):
            n = 5
        conn = self._get_conn()
        cursor = conn.cursor()
        sql = """
            SELECT a.conv_id, a.topic_id, a.assigned_at
            FROM expression_topic_assignments a
            LEFT JOIN conversation_meta m
                   ON m.user_id = a.user_id AND m.conv_id = a.conv_id
            WHERE a.user_id = ?
        """
        if not include_hidden:
            sql += " AND m.hidden_at IS NULL"
        sql += " ORDER BY a.assigned_at DESC, a.conv_id DESC LIMIT ?"
        cursor.execute(sql, (user_id, n))
        return [dict(r) for r in cursor.fetchall()]

    async def load_hidden_conversations(self, user_id: str) -> list[dict]:
        """加载被隐藏的会话列表（可恢复）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT conv_id, name, target_lang, level, scene, created_at, updated_at, hidden_at, expires_at
            FROM conversation_meta
            WHERE user_id = ? AND hidden_at IS NOT NULL
            ORDER BY hidden_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            conv = dict(row)
            cursor.execute("""
                SELECT role, content, created_at, turn_id FROM conversation_messages
                WHERE conv_id = ? ORDER BY seq ASC
            """, (conv["conv_id"],))
            msg_rows = cursor.fetchall()
            conv["messages"] = [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "turn_id": r["turn_id"],
                    "timestamp": r["created_at"],
                    "created_at": r["created_at"],
                }
                for r in msg_rows
            ]
            result.append(conv)
        return result

    async def get_relationship_memory(self, user_id: str) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM relationship_memory WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        result = dict(row)
        for key in ("engagement_topics", "avoid_patterns"):
            try:
                result[key] = json.loads(result.get(key) or "[]")
            except Exception:
                result[key] = []
        try:
            items = json.loads(result.get("memory_items") or "[]")
        except Exception:
            items = []
        now = datetime.now(timezone.utc)
        active = []
        superseded_history = []
        hidden_history = []
        from memory.companion_persona import is_legacy_persona_memory_key
        for item in items:
            try:
                if item.get("hidden_at"):
                    hidden_history.append(item)
                    continue
                if is_legacy_persona_memory_key(item.get("key")):
                    continue
                # 0915-12 流水账：与 postgres 读侧同款消解（superseded 留底但
                # 不进 live 视图），保持双后端方法面一致。
                if item.get("superseded_by"):
                    superseded_history.append(item)
                    continue
                expires = item.get("expires_at")
                if expires and datetime.fromisoformat(expires.replace("Z", "+00:00")) <= now:
                    continue
            except Exception:
                continue
            active.append(item)
        result["memory_items"] = active
        result["superseded_items"] = superseded_history
        result["hidden_items"] = hidden_history
        legacy_engagement_topics = result.get("engagement_topics") or []
        legacy_avoid_patterns = result.get("avoid_patterns") or []
        result["engagement_topics"] = list(dict.fromkeys(
            [str(item.get("value") or "").strip() for item in reversed(active)
             if item.get("key") == "engagement_topic" and str(item.get("value") or "").strip()]
            + [str(value).strip() for value in legacy_engagement_topics if str(value).strip()]
        ))[:8]
        result["avoid_patterns"] = list(dict.fromkeys(
            str(value).strip() for value in legacy_avoid_patterns if str(value).strip()
        ))
        # Only evidence-bearing, non-expired items are authoritative for prompts.
        if active:
            result["last_topic"] = ""
            result["open_loop"] = ""
            result["recent_mood"] = ""
            result["preferred_tone"] = ""
        for item in active:
            key = item.get("key")
            if key in (
                "last_topic", "open_loop", "recent_mood", "preferred_tone",
                "learning_preference", "self_reported_weakness", "next_session_intent",
                "next_session_intent_consumed",
                # 身份记忆（0915-04，postgres 侧已有）：名字浮到顶层供 prompt
                # 身份行读取——此前 sqlite 侧漏了提升，属双后端方法面缺口。
                "learner_name", "tutor_name",
                # 戏剧别名（plans/0915-16）：与 postgres 侧同步提升
                "learner_roleplay_alias", "tutor_roleplay_alias",
            ):
                result[key] = item.get("value", "")
            elif key == "avoid_pattern":
                result.setdefault("avoid_patterns", []).append(item.get("value", ""))
        result["avoid_patterns"] = list(dict.fromkeys(v for v in result.get("avoid_patterns", []) if v))[-8:]
        # Compatibility for explicit future-learning statements stored before
        # next_session_intent became a separate slot.  Broad interests are not
        # promoted; the learner's own wording must contain a future marker.
        preference = str(result.get("learning_preference") or "")
        if (
            not result.get("next_session_intent")
            and result.get("next_session_intent_consumed") != preference
            and any(
            marker in preference.lower()
            for marker in ("下次", "下一次", "下回", "以后如果继续", "以后继续练", "next time", "next session")
            )
        ):
            result["next_session_intent"] = preference
        return result

    async def upsert_relationship_items(self, user_id: str, items: list[dict]) -> None:
        current = await self.get_relationship_memory(user_id)
        # merge 基底必须是全量（live + superseded 留底），与 postgres 侧一致。
        base = list(current.get("memory_items", [])) + list(current.get("superseded_items", []))
        base += list(current.get("hidden_items", []))
        merged = merge_relationship_items(base, items)
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO relationship_memory (user_id, memory_items, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET memory_items=excluded.memory_items, updated_at=CURRENT_TIMESTAMP""",
            (user_id, json.dumps(merged, ensure_ascii=False)),
        )
        conn.commit()

    async def delete_relationship_memory_item(self, user_id: str, memory_key: str) -> int:
        current = await self.get_relationship_memory(user_id)
        # 0915-12：真删覆盖留底条目（与 postgres 侧一致）。
        before = (list(current.get("memory_items", []))
                  + list(current.get("superseded_items", []))
                  + list(current.get("hidden_items", [])))
        after = [item for item in before if item.get("id") != memory_key]
        if len(after) == len(before):
            return 0
        self._get_conn().execute(
            "UPDATE relationship_memory SET memory_items = ?, updated_at=CURRENT_TIMESTAMP WHERE user_id = ?",
            (json.dumps(after, ensure_ascii=False), user_id),
        )
        self._get_conn().commit()
        return 1

    # ── Alice 自我人格记忆（Postgres 方法面完全一致）────────────────────

    @staticmethod
    def _companion_persona_row(row) -> dict:
        if not row:
            return {}
        from engine.chunk_evidence import loads_json_list
        item = dict(row)
        item["supporting_fact_ids"] = loads_json_list(item.get("supporting_fact_ids"))
        item["related_entity_ids"] = loads_json_list(item.get("related_entity_ids"))
        return item

    async def ensure_companion_persona_legacy_migrated(self, user_id: str) -> int:
        """Idempotently import old Alice rows and user-confirmed archived identity claims."""
        from memory.companion_persona import (
            LEGACY_BACKFILL_VERSION, legacy_items_to_facts,
            legacy_confirmed_identity_to_facts,
        )
        from engine.chunk_evidence import loads_json_list
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT legacy_backfill_version FROM companion_persona_snapshots "
            "WHERE user_id = ? AND persona_key = 'alice'", (user_id,),
        )
        marker = cursor.fetchone()
        if marker and marker["legacy_backfill_version"] == LEGACY_BACKFILL_VERSION:
            return 0
        cursor.execute("SELECT memory_items FROM relationship_memory WHERE user_id = ?", (user_id,))
        legacy = cursor.fetchone()
        items = loads_json_list(legacy["memory_items"] if legacy else [])
        facts = legacy_items_to_facts(items)
        written = await self.upsert_companion_persona_facts(user_id, facts, guard_identity=False)
        cursor.execute(
            """SELECT m.conv_id, m.seq, m.content, m.created_at,
                      u.seq AS confirmed_by_seq, u.content AS confirmation
               FROM conversation_messages m
               JOIN conversation_meta c
                 ON c.user_id=m.user_id AND c.conv_id=m.conv_id
               JOIN conversation_messages u
                 ON u.user_id=m.user_id AND u.conv_id=m.conv_id
                AND u.seq=m.seq + 1 AND u.role='user'
               WHERE m.user_id=? AND m.role='assistant'
                 AND c.hidden_at IS NULL
                 AND LOWER(COALESCE(c.scene,'')) IN ('daily','daily_talk','free_talk')
                 AND (LOWER(m.content) LIKE '%from%' OR LOWER(m.content) LIKE '%grew up%'
                      OR LOWER(m.content) LIKE '%born in%' OR LOWER(m.content) LIKE '%live in%'
                      OR LOWER(m.content) LIKE '%singaporean%')
               ORDER BY m.created_at ASC, m.seq ASC""", (user_id,),
        )
        archive_facts = legacy_confirmed_identity_to_facts(
            [dict(row) for row in cursor.fetchall()], user_id=str(user_id),
        )
        written += await self.upsert_companion_persona_facts(
            user_id, archive_facts, guard_identity=True,
        )
        cursor.execute(
            """INSERT INTO companion_persona_snapshots
                   (user_id, persona_key, legacy_backfill_version)
               VALUES (?, 'alice', ?)
               ON CONFLICT(user_id, persona_key) DO UPDATE SET
                   legacy_backfill_version=excluded.legacy_backfill_version""",
            (user_id, LEGACY_BACKFILL_VERSION),
        )
        conn.commit()
        return int(written or 0)

    async def upsert_companion_persona_facts(
        self, user_id: str, facts: list[dict], guard_identity: bool = True,
        source_conv_id: str = "",
    ) -> int:
        if not facts:
            return 0
        from engine.chunk_evidence import loads_json_list
        conn = self._get_conn()
        cursor = conn.cursor()
        written = 0
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_id = str(fact.get("fact_id") or fact.get("id") or "").strip()
            value = str(fact.get("value_text") or fact.get("value") or "").strip()[:1000]
            if not fact_id or not value:
                continue
            persona_key = str(fact.get("persona_key") or "alice").strip()[:80]
            kind = str(fact.get("kind") or "").strip()[:80]
            facet = str(fact.get("facet_key") or "").strip()[:180]
            status = str(fact.get("status") or "active").strip()
            if status not in {"active", "candidate", "superseded", "forgotten"}:
                status = "candidate"
            # A durable identity slot may accumulate corroboration, but normal model
            # output cannot replace an already established origin/residence.
            if guard_identity and kind == "identity" and facet in {"origin", "residence"}:
                cursor.execute(
                    """SELECT value_text FROM companion_persona_facts
                       WHERE user_id=? AND persona_key=? AND kind='identity' AND facet_key=?
                         AND status='active' AND hidden_at IS NULL AND valid_to=''
                       ORDER BY recorded_at ASC LIMIT 1""",
                    (user_id, persona_key, facet),
                )
                established = cursor.fetchone()
                if established and str(established["value_text"]).casefold() != value.casefold():
                    status = "candidate"
            supersedes = str(fact.get("supersedes") or "")[:180]
            valid_from = str(fact.get("valid_from") or "")[:80]
            try:
                support_count_val = max(1, int(fact.get("support_count") or 1))
            except (TypeError, ValueError):
                support_count_val = 1
            if guard_identity and kind in {"trait", "preference", "ability", "language_environment", "current_status"} and facet:
                from memory.companion_persona import arbitrate_version_change
                cursor.execute(
                    """SELECT fact_id, value_text, status, source_conv_id, observed_at, valid_from, support_count FROM companion_persona_facts
                       WHERE user_id=? AND persona_key=? AND kind=? AND facet_key=?
                         AND status IN ('active','candidate') AND hidden_at IS NULL""",
                    (user_id, persona_key, kind, facet),
                )
                versions = [dict(row) for row in cursor.fetchall()]
                same_value_sources = [
                    str(row.get("source_conv_id") or "") for row in versions
                    if str(row.get("value_text") or "").casefold() == value.casefold()
                ]
                candidate_facet_sources = [
                    str(row.get("source_conv_id") or "") for row in versions
                    if row.get("status") == "candidate"
                ]
                same_value_sources = list(dict.fromkeys(
                    same_value_sources + candidate_facet_sources
                ))
                transition = arbitrate_version_change(
                    {**fact, "kind": kind, "facet_key": facet, "value_text": value,
                     "status": status, "supersedes": supersedes, "valid_from": valid_from},
                    versions, same_value_sources,
                )
                status = str(transition.get("status") or status)
                supersedes = str(transition.get("supersedes") or supersedes)[:180]
                valid_from = str(transition.get("valid_from") or valid_from)[:80]
                support_count_val = max(
                    support_count_val,
                    int(transition.get("support_count") or 1),
                )
                change_time = str(transition.get("change_time") or "")
                supersede_ids = list(transition.get("supersede_ids") or [])
                promote_ids = list(transition.get("promote_ids") or [])
                supersede_cand_ids = list(transition.get("supersede_candidate_ids") or [])
                ts_mark = (
                    change_time
                    or str(fact.get("recorded_at") or "")[:80]
                    or datetime.now(timezone.utc).isoformat()
                )
                if change_time and supersede_ids:
                    cursor.executemany(
                        """UPDATE companion_persona_facts SET status='superseded', valid_to=?
                           WHERE user_id=? AND persona_key=? AND fact_id=? AND status='active'""",
                        [(change_time, user_id, persona_key, old_id) for old_id in supersede_ids],
                    )
                if supersede_cand_ids:
                    cursor.executemany(
                        """UPDATE companion_persona_facts SET status='superseded', valid_to=?
                           WHERE user_id=? AND persona_key=? AND fact_id=? AND status='candidate'""",
                        [(ts_mark, user_id, persona_key, cid) for cid in supersede_cand_ids],
                    )
                if promote_ids:
                    cursor.executemany(
                        """UPDATE companion_persona_facts
                           SET status='active', valid_to='',
                               support_count=MAX(support_count, ?)
                           WHERE user_id=? AND persona_key=? AND fact_id=? AND status='candidate'""",
                        [(support_count_val, user_id, persona_key, pid) for pid in promote_ids],
                    )
            supporting = loads_json_list(fact.get("supporting_fact_ids"))
            related = loads_json_list(fact.get("related_entity_ids"))
            occurrence = str(fact.get("source_occurrence_id") or fact_id).strip()[:240]
            confidence = fact.get("confidence")
            try:
                confidence = max(0.0, min(1.0, float(confidence or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            cursor.execute(
                """INSERT OR IGNORE INTO companion_persona_facts (
                       user_id, persona_key, fact_id, kind, facet_key, value_text,
                       status, source_kind, source_conv_id, source_turn, source_message_ref,
                       evidence_quote, interaction_context, interaction_context_source,
                       confidence, observed_at, valid_from, valid_to,
                       recorded_at, time_basis, supersedes, supporting_fact_ids,
                       related_entity_ids, legacy_fact_id, source_occurrence_id, hidden_at,
                       support_count
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, persona_key, fact_id, kind, facet, value, status,
                    str(fact.get("source_kind") or fact.get("source") or "unknown")[:80],
                    str(fact.get("source_conv_id") or "")[:160] or None,
                    fact.get("source_turn"), str(fact.get("source_message_ref") or "")[:200],
                    str(fact.get("evidence_quote") or fact.get("evidence") or "")[:1000],
                    str(fact.get("interaction_context") or "")[:500],
                    str(fact.get("interaction_context_source") or "")[:40],
                     confidence,
                    str(fact.get("observed_at") or "")[:80],
                    valid_from,
                    str(fact.get("valid_to") or "")[:80],
                    str(fact.get("recorded_at") or fact.get("created_at") or "")[:80],
                    str(fact.get("time_basis") or "")[:40],
                    supersedes,
                    json.dumps(supporting, ensure_ascii=False),
                    json.dumps(related, ensure_ascii=False),
                    str(fact.get("legacy_fact_id") or "")[:180],
                    occurrence,
                    str(fact.get("hidden_at") or "")[:40] or None,
                    support_count_val,
                ),
            )
            written += max(0, cursor.rowcount)
            # Version a state slot, not a history of distinct experiences.
            # An empty facet cannot identify a slot; other kinds remain separate.
            if status == "active" and cursor.rowcount and kind != "experience" and facet:
                cursor.execute(
                    """UPDATE companion_persona_facts
                       SET status='superseded', valid_to=?
                       WHERE user_id=? AND persona_key=? AND kind=? AND facet_key=?
                         AND fact_id<>? AND status='active'""",
                    (str(fact.get("recorded_at") or "")[:80]
                     or __import__("datetime").datetime.now(
                         __import__("datetime").timezone.utc).isoformat(),
                     user_id, persona_key, kind, facet, fact_id),
                )
        if written:
            cursor.execute(
                """UPDATE companion_persona_snapshots
                   SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                       source_hash='', generated_at=NULL
                   WHERE user_id=? AND persona_key='alice'""", (user_id,),
            )
        conn.commit()
        return written

    async def upsert_open_threads(
        self, user_id: str, threads: list[dict], source_conv_id: str = "",
    ) -> int:
        """未闭环的事（plans/1017-02 §8.56）——SQLite 与 Postgres 同形。"""
        import uuid as _uuid
        from datetime import datetime as _dt, timezone as _tz
        rows = [t for t in (threads or [])
                if isinstance(t, dict) and str(t.get("topic") or "").strip()]
        if not rows:
            return 0
        now = _dt.now(_tz.utc).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        for item in rows:
            topic = str(item.get("topic") or "").strip()[:80]
            thread_id = str(item.get("thread_id") or "").strip() or (
                "th_" + _uuid.uuid5(_uuid.NAMESPACE_URL, topic.casefold()).hex[:12]
            )
            cursor.execute(
                """INSERT INTO companion_open_threads
                   (user_id, thread_id, topic, detail, status, source_conv_id,
                    source_quote, created_at, last_seen_at)
                   VALUES (?,?,?,?,'open',?,?,?,?)
                   ON CONFLICT(user_id, thread_id) DO UPDATE SET
                     detail=excluded.detail, last_seen_at=excluded.last_seen_at,
                     status='open', closed_at='', closed_reason=''""",
                (user_id, thread_id, topic, str(item.get("detail") or "")[:240],
                 source_conv_id or None, str(item.get("quote") or "")[:200], now, now),
            )
        conn.commit()
        return len(rows)

    async def list_open_threads(self, user_id: str, limit: int = 8) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, topic, detail, source_quote, last_seen_at"
            " FROM companion_open_threads WHERE user_id=? AND status='open'"
            " ORDER BY last_seen_at DESC LIMIT ?",
            (user_id, max(1, min(50, int(limit or 8)))),
        )
        return [dict(row) for row in cursor.fetchall()]

    async def close_open_threads(
        self, user_id: str, thread_ids: list[str], reason: str = "",
    ) -> int:
        from datetime import datetime as _dt, timezone as _tz
        ids = [str(x) for x in (thread_ids or []) if str(x).strip()]
        if not ids:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE companion_open_threads SET status='closed', closed_at=?,"
            " closed_reason=? WHERE user_id=? AND thread_id=?",
            [(_dt.now(_tz.utc).isoformat(), str(reason or "")[:120], user_id, tid)
             for tid in ids],
        )
        conn.commit()
        return len(ids)

    async def list_recent_life_items(self, user_id: str, limit: int = 5) -> list[dict]:
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            "SELECT item_id, text, kind, valid_to, aliases FROM companion_life_items"
            " WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (user_id, max(1, limit)))
        return [dict(row) for row in cursor.fetchall()]

    async def log_supply(self, user_id: str, session_id: str, kind: str,
                         ref_id: str, ref_hint: str = "") -> None:
        from datetime import datetime as _dt, timezone as _tz
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO companion_supply_log"
            " (user_id, session_id, kind, ref_id, ref_hint, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, str(session_id or ""), str(kind or ""), str(ref_id or ""),
             str(ref_hint or "")[:160], _dt.now(_tz.utc).isoformat()))
        conn.commit()

    async def mark_supply_hits(self, user_id: str, session_id: str,
                              user_texts: list[str]) -> int:
        texts = [str(t or "") for t in (user_texts or []) if str(t or "").strip()]
        if not texts:
            return 0
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            "SELECT kind, ref_id, ref_hint FROM companion_supply_log"
            " WHERE user_id=? AND session_id=? AND used=0",
            (user_id, str(session_id or "")))
        hits = 0
        for row in cursor.fetchall():
            if any(self._supply_overlap(str(row["ref_hint"] or ""), t) for t in texts):
                cursor.execute(
                    "UPDATE companion_supply_log SET used=1"
                    " WHERE user_id=? AND session_id=? AND kind=? AND ref_id=?",
                    (user_id, str(session_id or ""), row["kind"], row["ref_id"]))
                hits += 1
        conn.commit()
        return hits

    @staticmethod
    def _supply_overlap(hint: str, text: str) -> bool:
        import re as _re

        def grams(value: str) -> set:
            out: set = set()
            clean = _re.sub(r"[^一-鿿a-zA-Z ]", "", value or "")
            for token in _re.findall(r"[一-鿿]+", clean):
                out.update(token[i:i + 2] for i in range(len(token) - 1))
            out.update(w.lower() for w in _re.findall(r"[a-zA-Z]{4,}", clean))
            return out

        return bool(grams(hint) & grams(text))

    async def replace_life_items(self, user_id: str, items: list[dict],
                                 days: int = 7) -> int:
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        now = _dt.now(_tz.utc)
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            "UPDATE companion_life_items SET status='expired'"
            " WHERE user_id=? AND status='active' AND kind <> 'reading'", (user_id,))
        rows = [x for x in (items or []) if isinstance(x, dict) and str(x.get("text") or "").strip()]
        for item in rows[:5]:
            item_id = str(item.get("item_id") or "").strip() or (
                "life_" + str(abs(hash(str(item.get("text")))))[:10])
            cursor.execute(
                """INSERT INTO companion_life_items
                   (user_id, item_id, text, kind, valid_from, valid_to, created_at, aliases, source)
                   VALUES (?,?,?,?,?,?,?,?,'weekly')
                   ON CONFLICT(user_id, item_id) DO UPDATE SET
                     text=excluded.text, kind=excluded.kind,
                     valid_to=excluded.valid_to, status='active', aliases=excluded.aliases, source='weekly'""",
                (user_id, item_id, str(item.get("text") or "")[:160],
                 str(item.get("kind") or "")[:40], now.isoformat(),
                 (now + _td(days=days)).isoformat(), now.isoformat(),
                 str(item.get("aliases") or "")[:120]))
        conn.commit()
        return len(rows[:5])

    async def append_life_item(self, user_id: str, item: dict, days: int = 3,
                              merge_days: int = 7) -> int:
        """Append reading without replacing distinct external items or weekly narratives."""
        import hashlib
        from datetime import timedelta
        text = str((item or {}).get("text") or "").strip()
        if not text:
            return 0
        now = datetime.now(timezone.utc)
        ref = str((item or {}).get("ref") or "").strip()[:200]
        item_id = str(item.get("item_id") or "").strip() or (
            ("read_" + hashlib.md5(ref.encode("utf-8")).hexdigest()[:10]) if ref
            else ("read_" + str(abs(hash(text)))[:10])
        )
        topic = str((item or {}).get("topic") or "").strip()[:40]
        title = str((item or {}).get("title") or "").strip()[:160]
        author = str((item or {}).get("author") or "").strip()[:80]
        detail = str((item or {}).get("detail") or "").strip()[:400]
        aliases = str((item or {}).get("aliases") or "").strip()[:240]
        depth = float((item or {}).get("depth") or 1.0)
        raw_actions = (item or {}).get("actions_json") or (item or {}).get("actions") or {}
        if isinstance(raw_actions, dict):
            actions_str = json.dumps(raw_actions, ensure_ascii=False) if raw_actions else "{}"
        else:
            actions_str = str(raw_actions or "{}").strip()[:1200] or "{}"

        conn = self._get_conn()
        cursor = conn.cursor()
        if ref:
            cursor.execute(
                """SELECT item_id, strength FROM companion_life_items
                   WHERE user_id=? AND status='active' AND ref=?
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, ref),
            )
            same_ref = cursor.fetchone()
            if same_ref:
                new_strength = min(10.0, float(same_ref["strength"] or 1.0) + 0.6 * depth)
                cursor.execute(
                    """UPDATE companion_life_items
                       SET hits = hits + 1, strength = ?, depth = MAX(depth, ?),
                           text = ?, aliases = CASE WHEN ? <> '' THEN ? ELSE aliases END,
                           created_at = ?, valid_to = ?, status = 'active',
                           topic = CASE WHEN ? <> '' THEN ? ELSE topic END,
                           title = CASE WHEN ? <> '' THEN ? ELSE title END,
                           author = CASE WHEN ? <> '' THEN ? ELSE author END,
                           detail = CASE WHEN ? <> '' THEN ? ELSE detail END,
                           actions_json = CASE WHEN ? NOT IN ('', '{}') THEN ? ELSE actions_json END
                       WHERE user_id = ? AND item_id = ?""",
                    (
                        new_strength, depth, text[:200], aliases, aliases,
                        now.isoformat(), (now + timedelta(days=days)).isoformat(),
                        topic, topic, title, title, author, author, detail, detail,
                        actions_str, actions_str, user_id, same_ref["item_id"],
                    ),
                )
                conn.commit()
                return 1
            if topic and merge_days > 0:
                cursor.execute(
                    """UPDATE companion_life_items
                       SET strength = MIN(10.0, strength + 0.25)
                       WHERE user_id=? AND status='active' AND topic=?
                         AND ref <> ? AND created_at > ?""",
                    (user_id, topic, ref, (now - timedelta(days=merge_days)).isoformat()),
                )
        elif topic and merge_days > 0:
            cursor.execute(
                """SELECT item_id, strength FROM companion_life_items
                   WHERE user_id=? AND status='active' AND topic=?
                     AND ref='' AND created_at > ?
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, topic, (now - timedelta(days=merge_days)).isoformat()),
            )
            prior = cursor.fetchone()
            if prior:
                new_strength = min(10.0, float(prior["strength"] or 1.0) + 0.6 * depth)
                cursor.execute(
                    """UPDATE companion_life_items
                       SET hits = hits + 1, strength = ?, text = ?, aliases = ?,
                           created_at = ?, valid_to = ?, status = 'active'
                       WHERE user_id = ? AND item_id = ?""",
                    (new_strength, text[:200], aliases, now.isoformat(),
                     (now + timedelta(days=days)).isoformat(), user_id, prior["item_id"]),
                )
                conn.commit()
                return 1

        conn.execute(
            """INSERT INTO companion_life_items
               (user_id,item_id,text,kind,aliases,valid_from,valid_to,status,source,created_at,
                topic,hits,strength,depth,ref,title,author,detail,actions_json)
               VALUES (?,?,?,?,?,?,?,'active',?,?,?,1,?,?,?,?,?,?,?)
               ON CONFLICT(user_id,item_id) DO UPDATE SET
                 text=excluded.text,
                 aliases=CASE WHEN excluded.aliases <> '' THEN excluded.aliases ELSE companion_life_items.aliases END,
                 valid_to=excluded.valid_to, status='active',
                 topic=CASE WHEN excluded.topic <> '' THEN excluded.topic ELSE companion_life_items.topic END,
                 ref=CASE WHEN excluded.ref <> '' THEN excluded.ref ELSE companion_life_items.ref END,
                 title=CASE WHEN excluded.title <> '' THEN excluded.title ELSE companion_life_items.title END,
                 author=CASE WHEN excluded.author <> '' THEN excluded.author ELSE companion_life_items.author END,
                 detail=CASE WHEN excluded.detail <> '' THEN excluded.detail ELSE companion_life_items.detail END,
                 actions_json=CASE WHEN excluded.actions_json NOT IN ('', '{}') THEN excluded.actions_json ELSE companion_life_items.actions_json END""",
            (
                user_id, item_id, text[:200], str(item.get("kind") or "reading")[:40],
                aliases, now.isoformat(), (now + timedelta(days=days)).isoformat(),
                str(item.get("source") or "reading")[:20], now.isoformat(),
                topic, depth, depth, ref, title, author, detail, actions_str,
            ),
        )
        conn.commit()
        return 1

    async def record_life_action(
        self, user_id: str, *, ref: str = "", item_id: str = "",
        actions: dict | None = None, comment: str = "",
        author: str = "", title: str = "",
    ) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        ref_clean = str(ref or "").strip()[:200]
        iid_clean = str(item_id or "").strip()
        row = None
        if iid_clean:
            cursor.execute(
                """SELECT item_id, aliases, actions_json, author, title, strength, depth
                   FROM companion_life_items WHERE user_id=? AND item_id=?""",
                (user_id, iid_clean),
            )
            row = cursor.fetchone()
        if not row and ref_clean:
            ref_base = ref_clean.split("?", 1)[0].split("#", 1)[0].rstrip("/")
            bv_match = re.search(r"(BV[0-9A-Za-z]{8,14})", ref_clean)
            bv_tok = bv_match.group(1) if bv_match else ""
            cursor.execute(
                """SELECT item_id, aliases, actions_json, author, title, strength, depth
                   FROM companion_life_items
                   WHERE user_id=?
                     AND (ref=? OR ref=? OR (? <> '' AND ref LIKE '%' || ? || '%'))
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, ref_clean, ref_base, bv_tok, bv_tok),
            )
            row = cursor.fetchone()
        if not row:
            return 0
        try:
            merged = json.loads(row["actions_json"] or "{}")
            if not isinstance(merged, dict):
                merged = {}
        except Exception:
            merged = {}
        act = actions or {}
        for flag in ("like", "coin", "fav", "follow"):
            if act.get(flag):
                merged[flag] = True
        if str(act.get("why") or "").strip():
            merged["why"] = str(act.get("why")).strip()[:120]
        c_text = str(comment or "").strip()[:200]
        if c_text:
            comments = [str(x) for x in (merged.get("comments") or []) if str(x).strip()]
            if c_text not in comments:
                comments.append(c_text)
            merged["comments"] = comments[-5:]

        alias_parts = [str(row["aliases"] or "").strip()]
        eff_author = str(author or row["author"] or "").strip()[:80]
        eff_title = str(title or row["title"] or "").strip()[:160]
        if eff_author and eff_author not in alias_parts[0]:
            alias_parts.append(eff_author)
        for flag, kw in (
            ("follow", "关注 UP主 follow"),
            ("fav", "收藏 bookmark saved"),
            ("coin", "投币 coin"),
            ("like", "点赞 liked"),
        ):
            if merged.get(flag) and kw.split()[0] not in alias_parts[0]:
                alias_parts.append(kw)
        if merged.get("comments") and "评论" not in alias_parts[0]:
            alias_parts.append("评论 留言 comment")
        new_aliases = " ".join(p for p in alias_parts if p).strip()[:240]

        heavy = bool(c_text or merged.get("coin") or merged.get("fav") or merged.get("follow"))
        boost = 0.85 if heavy else 0.4
        min_depth = 1.35 if heavy else 1.15
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """UPDATE companion_life_items
               SET actions_json = ?, aliases = ?,
                   author = CASE WHEN ? <> '' THEN ? ELSE author END,
                   title = CASE WHEN ? <> '' THEN ? ELSE title END,
                   strength = MIN(10.0, strength + ?),
                   depth = MAX(depth, ?),
                   last_recalled_at = ?
               WHERE user_id = ? AND item_id = ?""",
            (
                json.dumps(merged, ensure_ascii=False), new_aliases,
                eff_author, eff_author, eff_title, eff_title,
                boost, min_depth, now_iso, user_id, row["item_id"],
            ),
        )
        conn.commit()
        return 1

    async def has_fresh_life_item(self, user_id: str, source: str, hours: int = 20) -> bool:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        row = self._get_conn().execute(
            "SELECT count(*) FROM companion_life_items WHERE user_id=? AND source=? AND created_at>?",
            (user_id, str(source), cutoff)).fetchone()
        return bool(row[0])

    async def list_life_items(self, user_id: str, limit: int = 3,
                              shared_user_id: str = "") -> list[dict]:
        from datetime import datetime as _dt, timezone as _tz
        conn = self._get_conn(); cursor = conn.cursor()
        now_dt = _dt.now(_tz.utc)
        now_iso = now_dt.isoformat()
        pool_limit = max(1, limit) * 4

        def _row_score(r: dict) -> tuple:
            try:
                ts_raw = str(r.get("last_recalled_at") or r.get("created_at") or "")
                ts = _dt.fromisoformat(ts_raw.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_tz.utc)
                age_days = max(0.0, (now_dt - ts).total_seconds() / 86400.0)
            except Exception:
                age_days = 0.0
            s = float(r.get("strength") or 1.0)
            d = max(0.5, float(r.get("depth") or 1.0))
            eff = s * ((1.0 + age_days / max(0.05, s * d)) ** (-0.7))
            return (1 if str(r.get("kind") or "") != "reading" else 0, eff, str(r.get("created_at") or ""))

        shared_uid = str(shared_user_id or "").strip()
        if shared_uid and shared_uid != str(user_id):
            cursor.execute(
                """SELECT user_id AS owner_id, item_id, text, kind, source, valid_to,
                          aliases, topic, hits, strength, depth, last_recalled_at,
                          created_at, ref, title, author, detail, actions_json
                   FROM companion_life_items
                   WHERE user_id IN (?, ?) AND status='active' AND valid_to > ?""",
                (user_id, shared_uid, now_iso),
            )
            raw_rows = [dict(r) for r in cursor.fetchall()]
            raw_rows.sort(
                key=lambda r: (
                    1 if (str(r.get("owner_id") or "") == str(user_id) and str(r.get("kind") or "") != "reading") else 0,
                    *_row_score(r),
                ),
                reverse=True,
            )
            preferred_owners: dict[str, str] = {}
            for r in sorted(raw_rows, key=lambda x: 0 if str(x.get("owner_id") or "") == str(user_id) else 1):
                dk = f"ref:{r['ref']}" if str(r.get("ref") or "").strip() else f"id:{r['item_id']}"
                preferred_owners.setdefault(dk, str(r.get("owner_id") or ""))
            out: list[dict] = []
            seen: set[str] = set()
            for r in raw_rows:
                dk = f"ref:{r['ref']}" if str(r.get("ref") or "").strip() else f"id:{r['item_id']}"
                if dk in seen or preferred_owners.get(dk) != str(r.get("owner_id") or ""):
                    continue
                seen.add(dk)
                r.pop("owner_id", None)
                out.append(r)
                if len(out) >= pool_limit:
                    break
            return out

        cursor.execute(
            """SELECT item_id, text, kind, source, valid_to, aliases, topic, hits,
                      strength, depth, last_recalled_at, created_at, ref,
                      title, author, detail, actions_json
               FROM companion_life_items
               WHERE user_id=? AND status='active' AND valid_to > ?""",
            (user_id, now_iso),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        rows.sort(key=_row_score, reverse=True)
        return rows[:pool_limit]

    async def list_growth_facts_for_merge(self, user_id: str, limit: int = 200) -> list[dict]:
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            """SELECT fact_id, kind, facet_key, value_text, status, support_count,
                      recorded_at, source_conv_id
               FROM companion_persona_facts
               WHERE user_id=? AND persona_key='alice'
                 AND hidden_at IS NULL AND valid_to=''
                 AND status IN ('active','candidate')
                 AND facet_key LIKE 'growth:%'
               ORDER BY recorded_at DESC LIMIT ?""",
            (user_id, max(1, min(400, int(limit)))),
        )
        return [dict(r) for r in cursor.fetchall()]

    async def bump_persona_support(self, user_id: str, fact_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """UPDATE companion_persona_facts
               SET support_count = support_count + 1, status = 'active'
               WHERE user_id=? AND fact_id=?""",
            (user_id, fact_id),
        )
        conn.commit()

    async def set_persona_fact_value(
        self, user_id: str, fact_id: str, value: str, *,
        persona_key: str = "alice", kind: str = "preference", facet_key: str = "",
    ) -> int:
        val = str(value or "").strip()[:1000]
        if not user_id or not fact_id or not val:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            """UPDATE companion_persona_facts
               SET value_text=?, status='active', valid_to='', recorded_at=?
               WHERE user_id=? AND persona_key=? AND fact_id=?""",
            (val, now, user_id, str(persona_key), str(fact_id)),
        )
        if cursor.rowcount > 0:
            conn.commit()
            return cursor.rowcount
        cursor.execute(
            """INSERT OR IGNORE INTO companion_persona_facts
               (user_id, persona_key, fact_id, kind, facet_key, value_text,
                status, source_kind, source_conv_id, source_turn,
                source_message_ref, evidence_quote, interaction_context,
                supporting_fact_ids, related_entity_ids, confidence,
                recorded_at, valid_from, valid_to, hidden_at, observed_at,
                source_occurrence_id, support_count)
               VALUES (?, ?, ?, ?, ?, ?, 'active', 'reading', '', 0,
                       '', '', '', '[]', '[]', 0.9, ?, '', '', NULL, ?, ?, 1)""",
            (user_id, str(persona_key), str(fact_id), str(kind), str(facet_key)[:180],
             val, now, now, str(fact_id)),
        )
        conn.commit()
        return 1

    async def list_life_topics(self, user_id: str, limit: int = 20) -> list[str]:
        conn = self._get_conn(); cursor = conn.cursor()
        cursor.execute(
            """SELECT topic, count(*) AS n FROM companion_life_items
               WHERE user_id=? AND topic <> ''
               GROUP BY topic ORDER BY n DESC LIMIT ?""",
            (user_id, max(1, limit)),
        )
        return [str(r["topic"]) for r in cursor.fetchall()]

    async def upsert_open_question(self, user_id: str, question: dict,
                                   source_conv_id: str = "") -> None:
        """她问过、他没答的问题（§8.76）——与 Postgres 同形，最多留三条。"""
        from datetime import datetime as _dt, timezone as _tz
        text = str(question.get("text") or "").strip()[:160]
        qid = str(question.get("question_id") or "").strip()
        if not text or not qid:
            return
        now = _dt.now(_tz.utc).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO companion_open_questions
               (user_id, question_id, text, topic, source_conv_id, asked_at, created_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id, question_id) DO NOTHING""",
            (user_id, qid, text, str(question.get("topic") or "")[:80],
             source_conv_id or None, now, now),
        )
        cursor.execute(
            """UPDATE companion_open_questions SET status='expired'
               WHERE user_id=? AND status='pending' AND question_id NOT IN (
                   SELECT question_id FROM companion_open_questions
                   WHERE user_id=? AND status='pending'
                   ORDER BY asked_at DESC LIMIT 3)""",
            (user_id, user_id),
        )
        conn.commit()

    async def list_open_questions(self, user_id: str, limit: int = 3) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question_id, text, topic, asked_at FROM companion_open_questions"
            " WHERE user_id=? AND status='pending' ORDER BY asked_at DESC LIMIT ?",
            (user_id, max(1, min(10, int(limit or 3)))),
        )
        return [dict(row) for row in cursor.fetchall()]

    async def expire_open_questions(self, user_id: str, days: int = 3) -> int:
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        cutoff = (_dt.now(_tz.utc) - _td(days=days)).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE companion_open_questions SET status='expired'"
            " WHERE user_id=? AND status='pending' AND asked_at < ?",
            (user_id, cutoff),
        )
        conn.commit()
        return cursor.rowcount or 0

    async def mark_question_shown(self, user_id: str, question_id: str) -> None:
        from datetime import datetime as _dt, timezone as _tz
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE companion_open_questions SET shown_at=?, status='asked'"
            " WHERE user_id=? AND question_id=? AND status='pending'",
            (_dt.now(_tz.utc).isoformat(), user_id, question_id),
        )
        conn.commit()


    async def enqueue_outbox(self, user_id: str, text: str, *, reason: str = "",
                             conv_id: str = "", due_at: str = "",
                             msg_id: str = "") -> str:
        """她的留言箱（§8.60）——SQLite 与 Postgres 同形。"""
        import uuid as _uuid
        from datetime import datetime as _dt, timezone as _tz
        body = " ".join(str(text or "").split())[:400]
        if not body:
            return ""
        now = _dt.now(_tz.utc).isoformat()
        mid = msg_id or ("ob_" + _uuid.uuid4().hex[:12])
        conn = self._get_conn()
        cursor = conn.cursor()
        if reason:
            cursor.execute(
                "SELECT count(*) FROM companion_outbox"
                " WHERE user_id=? AND reason=? AND created_at >= ?",
                (user_id, reason, now[:10]))
            if cursor.fetchone()[0]:
                return ""
        cursor.execute(
            "INSERT OR IGNORE INTO companion_outbox"
            " (user_id, msg_id, conv_id, text, reason, due_at, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (user_id, mid, conv_id or None, body, reason[:80], due_at[:32], now))
        conn.commit()
        return mid

    async def list_outbox_unread(self, user_id: str, limit: int = 5) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT msg_id, text, reason, due_at, created_at FROM companion_outbox"
            " WHERE user_id=? AND read_at='' ORDER BY created_at DESC LIMIT ?",
            (user_id, max(1, min(20, int(limit or 5)))))
        return [dict(row) for row in cursor.fetchall()]

    async def mark_outbox_read(self, user_id: str, msg_id: str) -> int:
        from datetime import datetime as _dt, timezone as _tz
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE companion_outbox SET read_at=? WHERE user_id=? AND msg_id=? AND read_at=''",
            (_dt.now(_tz.utc).isoformat(), user_id, str(msg_id or "")))
        conn.commit()
        return 1

    async def demote_companion_persona_facts(
            self, user_id: str, fact_ids: list[str], reason: str = "") -> int:
        """把指定事实降级（§8.91c 回音复核用）：status=superseded，不删除。"""
        ids = [str(x) for x in (fact_ids or []) if str(x).strip()]
        if not ids:
            return 0
        from datetime import datetime as _dt, timezone as _tz
        conn = self._get_conn(); cursor = conn.cursor()
        marks = ','.join('?' for _ in ids)
        cursor.execute(
            "UPDATE companion_persona_facts SET status='superseded', valid_to=? "
            f"WHERE user_id=? AND fact_id IN ({marks}) AND status='active'",
            (_dt.now(_tz.utc).isoformat(), user_id, *ids))
        written = max(0, cursor.rowcount)
        conn.commit()
        return written

    async def list_companion_persona_facts(
        self, user_id: str, query: str = "", limit: int = 48,
    ) -> list[dict]:
        from memory.companion_persona import query_terms, wants_persona_history
        conn = self._get_conn()
        cursor = conn.cursor()
        params: list = [user_id]
        include_history = wants_persona_history(query)
        status_clause = "status IN ('active','superseded')" if include_history else "status='active' AND valid_to=''"
        where = f"user_id=? AND persona_key='alice' AND {status_clause} AND hidden_at IS NULL"
        terms = sorted(query_terms(query))
        if terms and not include_history:
            where += " AND (" + " OR ".join(
                "value_text LIKE ? OR facet_key LIKE ?" for _ in terms
            ) + ")"
            for term in terms:
                params.extend((f"%{term}%", f"%{term}%"))
        params.append(max(1, min(200, int(limit or 48))))
        cursor.execute(
            f"SELECT * FROM companion_persona_facts WHERE {where} "
            "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, "
            "CASE kind WHEN 'identity' THEN 0 ELSE 1 END, recorded_at DESC, fact_id "
            "LIMIT ?", tuple(params),
        )
        return [self._companion_persona_row(row) for row in cursor.fetchall()]

    async def get_companion_persona_snapshot(self, user_id: str) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM companion_persona_snapshots WHERE user_id=? AND persona_key='alice'",
            (user_id,),
        )
        return self._companion_persona_row(cursor.fetchone())

    async def save_companion_persona_snapshot(
        self, user_id: str, snapshot: dict,
    ) -> bool:
        from engine.chunk_evidence import loads_json_list
        ids = list(dict.fromkeys(
            str(value) for value in loads_json_list((snapshot or {}).get("supporting_fact_ids"))
            if str(value)
        ))
        narrative = str((snapshot or {}).get("narrative") or "").strip()[:600]
        if not narrative or not ids:
            return False
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) AS n FROM companion_persona_facts WHERE user_id=? "
            f"AND persona_key='alice' AND status IN ('active','superseded') "
            f"AND hidden_at IS NULL AND fact_id IN ({','.join('?' for _ in ids)})",
            (user_id, *ids),
        )
        if int(cursor.fetchone()["n"] or 0) != len(set(ids)):
            return False
        cursor.execute(
            """INSERT INTO companion_persona_snapshots
                   (user_id, persona_key, narrative, supporting_fact_ids,
                    snapshot_version, source_hash, generated_at)
               VALUES (?, 'alice', ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, persona_key) DO UPDATE SET
                   narrative=excluded.narrative,
                   supporting_fact_ids=excluded.supporting_fact_ids,
                   snapshot_version=excluded.snapshot_version,
                   source_hash=excluded.source_hash,
                   generated_at=excluded.generated_at""",
            (user_id, narrative, json.dumps(ids, ensure_ascii=False),
             str(snapshot.get("snapshot_version") or "")[:80],
             str(snapshot.get("source_hash") or "")[:80]),
        )
        conn.commit()
        return True

    async def invalidate_companion_persona_snapshot(self, user_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """UPDATE companion_persona_snapshots
               SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                   source_hash='', generated_at=NULL
               WHERE user_id=? AND persona_key='alice'""", (user_id,),
        )
        conn.commit()

    async def set_companion_persona_source_hidden(
        self, user_id: str, conv_id: str, hidden: bool,
    ) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE companion_persona_facts SET hidden_at=? WHERE user_id=? AND source_conv_id=?",
            ((datetime.now(timezone.utc).isoformat() if hidden else None), user_id, conv_id),
        )
        changed = max(0, cursor.rowcount)
        cursor.execute(
            """UPDATE companion_persona_snapshots
               SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                   source_hash='', generated_at=NULL
               WHERE user_id=? AND persona_key='alice'""", (user_id,),
        )
        conn.commit()
        return changed

    async def purge_companion_persona_source(self, user_id: str, conv_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM companion_persona_facts WHERE user_id=? AND source_conv_id=?",
            (user_id, conv_id),
        )
        deleted = max(0, cursor.rowcount)
        cursor.execute(
            """UPDATE companion_persona_snapshots
               SET narrative='', supporting_fact_ids='[]', snapshot_version='',
                   source_hash='', generated_at=NULL
               WHERE user_id=? AND persona_key='alice'""", (user_id,),
        )
        conn.commit()
        return deleted

    # ── 统一学习者画像（plans/0923-04 阶段 1）────────────────────────────────
    # ⚠ 方法面（名字、参数、返回键）**必须**与 l2_postgres 逐一同形：
    # 缺一个方法 = 线上 500，或在 hasattr 处静默变空值。
    # evidence_refs / supporting_fact_ids 是 JSON 文本列，一律经 loads_json_list
    # 解码后返回，两个后端形状一致。

    async def upsert_learner_profile_facts(self, user_id: str, facts: list[dict]) -> int:
        """幂等投影：同 (user_id, fact_key, fact_subkey, source_occurrence_id) 不新增版本。

        返回**新写入**行数（重复投影返回 0）——调用方可据此判断"这次是否真有新证据"。
        """
        if not facts:
            return 0
        conn = self._get_conn()
        cursor = conn.cursor()
        written = 0
        for fact in facts:
            if not isinstance(fact, dict) or not fact.get("fact_key"):
                continue
            cursor.execute(
                """INSERT INTO learner_profile_facts (
                       fact_id, user_id, dimension, fact_key, fact_subkey, value_text,
                       source_pipeline, source_table, source_id, source_anchor_id,
                       source_occurrence_id, extractor_version, source_conv_id,
                       source_session_id, evidence_quote, evidence_refs, evidence_kind,
                       confidence, observed_at, valid_from, valid_to, recorded_at,
                       time_basis, expires_at, status
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(user_id, fact_key, fact_subkey, source_occurrence_id)
                   DO NOTHING""",
                (
                    str(fact.get("fact_id") or ""),
                    user_id,
                    fact.get("dimension"),
                    fact.get("fact_key"),
                    str(fact.get("fact_subkey") or ""),
                    str(fact.get("value_text") or ""),
                    str(fact.get("source_pipeline") or ""),
                    str(fact.get("source_table") or ""),
                    str(fact.get("source_id") or ""),
                    str(fact.get("source_anchor_id") or ""),
                    str(fact.get("source_occurrence_id") or ""),
                    str(fact.get("extractor_version") or ""),
                    fact.get("source_conv_id"),
                    fact.get("source_session_id"),
                    str(fact.get("evidence_quote") or ""),
                    json.dumps(list(fact.get("evidence_refs") or []), ensure_ascii=False),
                    str(fact.get("evidence_kind") or "unknown"),
                    fact.get("confidence"),
                    fact.get("observed_at"),
                    fact.get("valid_from"),
                    fact.get("valid_to"),
                    str(fact.get("recorded_at") or ""),
                    str(fact.get("time_basis") or "unknown"),
                    fact.get("expires_at"),
                    str(fact.get("status") or "candidate"),
                ),
            )
            written += cursor.rowcount or 0
        conn.commit()
        return written

    async def list_learner_profile_facts(
        self, user_id: str, *, statuses: Optional[tuple] = None
    ) -> list[dict]:
        """读该用户全部画像事实行（**含历史版本**，按证据时间升序稳定排序）。"""
        from engine.chunk_evidence import loads_json_list
        conn = self._get_conn()
        cursor = conn.cursor()
        sql = "SELECT * FROM learner_profile_facts WHERE user_id = ?"
        params: list = [user_id]
        if statuses:
            marks = ",".join("?" for _ in statuses)
            sql += f" AND status IN ({marks})"
            params.extend(list(statuses))
        sql += " ORDER BY COALESCE(observed_at, '') ASC, recorded_at ASC, fact_id ASC"
        cursor.execute(sql, tuple(params))
        rows = []
        for row in cursor.fetchall():
            item = dict(row)
            item["evidence_refs"] = loads_json_list(item.get("evidence_refs"))
            rows.append(item)
        return rows

    async def put_learner_profile_narrative(
        self, user_id: str, dimension: str, text: str,
        supporting_fact_ids: Optional[list] = None,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO learner_profile_narratives
                   (user_id, dimension, text, supporting_fact_ids, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, dimension) DO UPDATE SET
                   text=excluded.text,
                   supporting_fact_ids=excluded.supporting_fact_ids,
                   updated_at=CURRENT_TIMESTAMP""",
            (user_id, str(dimension or ""), str(text or ""),
             json.dumps(list(supporting_fact_ids or []), ensure_ascii=False)),
        )
        conn.commit()

    async def list_learner_profile_narratives(self, user_id: str) -> list[dict]:
        from engine.chunk_evidence import loads_json_list
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM learner_profile_narratives WHERE user_id = ? ORDER BY dimension ASC",
            (user_id,),
        )
        rows = []
        for row in cursor.fetchall():
            item = dict(row)
            item["supporting_fact_ids"] = loads_json_list(item.get("supporting_fact_ids"))
            rows.append(item)
        return rows

    # ── 写入租约（plans/0923-04 §4.4 阶段 2「旧写手写入闸门」）─────────────────
    # ⚠ 方法面必须与 l2_postgres 逐一同形。本地后端只有一个连接 ⇒
    # "同一写事务串行"天然成立，不需要 PG 那样的 advisory lock。

    async def register_write_lease(self, user_id: str, token: str,
                                   scope_conv_id: str = "") -> bool:
        """登记一条写入令牌（幂等）。顺带清掉本用户超过 TTL 的旧行（表有界）。"""
        token = str(token or "").strip()
        if not token:
            return False
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM learner_profile_write_leases "
            "WHERE user_id = ? AND created_at < datetime('now', ?)",
            (user_id, f"-{int(_learner_profile.WRITE_LEASE_TTL_HOURS)} hours"),
        )
        cursor.execute(
            """INSERT INTO learner_profile_write_leases
                   (user_id, token, scope_conv_id, status, created_at)
               VALUES (?, ?, ?, 'active', CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, token) DO UPDATE SET
                   status = 'active',
                   scope_conv_id = excluded.scope_conv_id,
                   created_at = CURRENT_TIMESTAMP""",
            (user_id, token, str(scope_conv_id or "")),
        )
        conn.commit()
        return True

    async def write_lease_active(self, user_id: str, tokens: list[str]) -> bool:
        """全部令牌都仍是本用户的 active 租约才通过；表缺失/异常 ⇒ 放行（见 PG 侧说明）。"""
        wanted = sorted({str(t).strip() for t in (tokens or []) if str(t or "").strip()})
        if not wanted:
            return False
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            marks = ",".join("?" for _ in wanted)
            cursor.execute(
                f"""SELECT COUNT(*) AS n FROM learner_profile_write_leases
                     WHERE user_id = ? AND status = 'active'
                       AND token IN ({marks})""",
                tuple([user_id] + wanted),
            )
            row = cursor.fetchone()
        except Exception:
            return True
        return int(((row["n"] if row else 0) or 0)) == len(wanted)

    async def revoke_write_leases(self, user_id: str, scope_conv_id: str = "") -> int:
        """把令牌置为 revoked（不删行）。scope 为空 = 该用户全部。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        scope = str(scope_conv_id or "")
        if scope:
            cursor.execute(
                "UPDATE learner_profile_write_leases SET status = 'revoked' "
                "WHERE user_id = ? AND status = 'active' AND scope_conv_id = ?",
                (user_id, scope),
            )
        else:
            cursor.execute(
                "UPDATE learner_profile_write_leases SET status = 'revoked' "
                "WHERE user_id = ? AND status = 'active'",
                (user_id,),
            )
        count = cursor.rowcount
        conn.commit()
        return count

    async def clear_write_leases(self, user_id: str) -> int:
        """物理清空该用户的令牌行（全删的最后一步）。幂等。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM learner_profile_write_leases WHERE user_id = ?", (user_id,))
        count = cursor.rowcount
        conn.commit()
        return count

    # ── forget 控制行（plans/0923-04 §4.3，阶段 2 补全）───────────────────────
    # ⚠ 方法面必须与 l2_postgres 逐一同形。控制行与画像事实**同表**（status='forget'），
    # 不含原话与值：只存 fact_key / fact_subkey / 旧来源锚点。

    async def put_learner_profile_forget(self, user_id: str, fact_key: str,
                                         fact_subkey: str = "",
                                         source_anchor_id: str = "",
                                         evidence_refs: Optional[list] = None) -> None:
        """写一条 forget 控制行（幂等）。

        source_occurrence_id 用 `forget:<fact_key>` 当管道名派生 ⇒ 重复 forget
        同一 (键, 子键, 锚点) 落在同一行，不膨胀。行内 value/quote 恒空；
        evidence_refs 只留**被删行的来源引用 ID**（方案 §4.3 允许「旧来源锚点/
        消息 ID」）——供召回按被忘引用过滤，不含原话与值。
        """
        now = datetime.now(timezone.utc).isoformat()
        occurrence = _learner_profile.source_occurrence_id(
            f"forget:{str(fact_key or '')}", str(source_anchor_id or ""),
            str(fact_subkey or ""),
        )
        refs_json = json.dumps(
            [str(r) for r in (evidence_refs or []) if str(r or "").strip()][:32],
            ensure_ascii=False)
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO learner_profile_facts (
                   fact_id, user_id, dimension, fact_key, fact_subkey, value_text,
                   source_pipeline, source_table, source_id, source_anchor_id,
                   source_occurrence_id, extractor_version, source_conv_id,
                   source_session_id, evidence_quote, evidence_refs, evidence_kind,
                   confidence, observed_at, valid_from, valid_to, recorded_at,
                   time_basis, expires_at, status
               ) VALUES (?,?,NULL,?,?, '', 'forget', '', '', ?, ?, 'v1',
                         NULL, NULL, '', ?, 'control',
                         NULL, ?, ?, NULL, ?, 'unknown', NULL, 'forget')
               ON CONFLICT(user_id, fact_key, fact_subkey, source_occurrence_id)
               DO NOTHING""",
            (
                uuid.uuid4().hex,
                user_id,
                str(fact_key or ""),
                str(fact_subkey or ""),
                str(source_anchor_id or ""),
                occurrence,
                refs_json,
                now, now, now,
            ),
        )
        conn.commit()

    async def list_learner_profile_forgets(self, user_id: str) -> list[dict]:
        """该用户全部 forget 控制行（投影抑制用；**不含原话与值**）。"""
        from engine.chunk_evidence import loads_json_list
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM learner_profile_facts "
            "WHERE user_id = ? AND status = 'forget' "
            "ORDER BY recorded_at ASC, fact_id ASC",
            (user_id,),
        )
        rows = []
        for row in cursor.fetchall():
            item = dict(row)
            item["evidence_refs"] = loads_json_list(item.get("evidence_refs"))
            rows.append(item)
        return rows

    async def delete_learner_profile_facts(self, user_id: str, fact_key: str,
                                           fact_subkey: str = "") -> int:
        """物理删某逻辑事实的**全部版本**（不含 forget 控制行）。返回删除行数。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        sql = ("DELETE FROM learner_profile_facts "
               "WHERE user_id = ? AND fact_key = ? AND status != 'forget'")
        params: list = [user_id, str(fact_key or "")]
        if str(fact_subkey or ""):
            sql += " AND fact_subkey = ?"
            params.append(str(fact_subkey))
        cursor.execute(sql, tuple(params))
        count = cursor.rowcount
        conn.commit()
        return count

    async def list_profile_sweep_users(self) -> list[str]:
        """需要画像补偿投影的用户（有关系记忆行或已有画像行的用户，去重升序）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id FROM relationship_memory "
            "UNION SELECT user_id FROM learner_profile_facts ORDER BY user_id ASC"
        )
        return [str(row[0]) for row in cursor.fetchall() if row[0]]

    async def upsert_relationship_memory(self, user_id: str, updates: dict) -> None:
        current = await self.get_relationship_memory(user_id)
        merged = {**current, **{k: v for k, v in updates.items() if v is not None}}
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO relationship_memory
                (user_id, last_topic, open_loop, recent_mood, preferred_tone,
                 engagement_topics, avoid_patterns, last_user_message,
                 last_assistant_response, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                last_topic=excluded.last_topic,
                open_loop=excluded.open_loop,
                recent_mood=excluded.recent_mood,
                preferred_tone=excluded.preferred_tone,
                engagement_topics=excluded.engagement_topics,
                avoid_patterns=excluded.avoid_patterns,
                last_user_message=excluded.last_user_message,
                last_assistant_response=excluded.last_assistant_response,
                updated_at=CURRENT_TIMESTAMP
        """, (
            user_id,
            merged.get("last_topic", ""),
            merged.get("open_loop", ""),
            merged.get("recent_mood", ""),
            merged.get("preferred_tone", ""),
            json.dumps(merged.get("engagement_topics", []), ensure_ascii=False),
            json.dumps(merged.get("avoid_patterns", []), ensure_ascii=False),
            merged.get("last_user_message", ""),
            merged.get("last_assistant_response", ""),
        ))
        conn.commit()

    async def get_memory_context(self, user_id: str) -> dict:
        """Get shared memory context across all conversations for a user."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Top errors
        cursor.execute("""
            SELECT error_type, error_excerpt, count, weight
            FROM error_fingerprints
            WHERE user_id = ? AND weight > 0.1
            ORDER BY weight DESC LIMIT 10
        """, (user_id,))
        errors = [dict(r) for r in cursor.fetchall()]

        # Mastered expressions
        cursor.execute("""
            SELECT expression, confidence
            FROM mastered_expressions
            WHERE user_id = ?
            ORDER BY confidence DESC LIMIT 20
        """, (user_id,))
        mastered = [dict(r) for r in cursor.fetchall()]

        # Session count and recent summaries
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM session_archives
            WHERE user_id = ?
        """, (user_id,))
        total_sessions = cursor.fetchone()["cnt"]

        cursor.execute("""
            SELECT session_id, summary, cefr_before, cefr_after, error_count, created_at
            FROM session_archives
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 5
        """, (user_id,))
        recent_sessions = [dict(r) for r in cursor.fetchall()]

        relationship = await self.get_relationship_memory(user_id)
        return {
            "errors": errors,
            "mastered": mastered,
            "total_sessions": total_sessions,
            "recent_sessions": recent_sessions,
            "relationship": relationship,
        }

    async def get_style_parameters(self, user_id: str) -> dict:
        """读取 Loop Engineering 优化过的风格参数（version>0 表示已被反馈优化过）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM style_parameters WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    # ── Prompt Patches (Loop Engineering: 审计驱动的 prompt 自动优化) ──

    async def get_prompt_patches(self, user_id: str, status: str = "applied") -> list[dict]:
        """获取用户当前的 prompt patches（默认只取已应用的未验证 patch）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patch_id, user_id, session_id, audit_recommendation_id, root_cause,
                   action, layer, from_value, to_value, reason,
                   status, verified_sessions, max_retries,
                   created_at, applied_at, failed_at, rolled_back_at
            FROM prompt_patches
            WHERE user_id = ? AND status = ?
            ORDER BY applied_at DESC
            LIMIT 5
        """, (user_id, status))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    async def save_prompt_patch(self, patch: dict) -> str:
        """保存一条 prompt patch（insert or update）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        patch_id = patch.get("patch_id", "")
        cursor.execute("""
            INSERT OR REPLACE INTO prompt_patches
            (user_id, patch_id, session_id, audit_recommendation_id, root_cause,
             action, layer, from_value, to_value, reason,
             status, verified_sessions, max_retries,
             created_at, applied_at, failed_at, rolled_back_at)
            VALUES (?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    COALESCE(?, CURRENT_TIMESTAMP),
                    ?, ?, ?)
        """, (
            patch.get("user_id", ""),
            patch_id,
            patch.get("session_id", ""),
            patch.get("audit_recommendation_id", ""),
            patch.get("root_cause", ""),
            patch.get("action", ""),
            patch.get("layer", ""),
            patch.get("from_value", ""),
            patch.get("to_value", ""),
            patch.get("reason", ""),
            patch.get("status", "pending"),
            patch.get("verified_sessions", 0),
            patch.get("max_retries", 3),
            patch.get("created_at"),
            patch.get("applied_at"),
            patch.get("failed_at"),
            patch.get("rolled_back_at"),
        ))
        conn.commit()
        return patch_id

    async def batch_save_prompt_patches(self, patches: list[dict]) -> int:
        """批量保存 prompt patches。"""
        count = 0
        for patch in patches:
            await self.save_prompt_patch(patch)
            count += 1
        return count

    async def update_patch_status(self, patch_id: str, status: str,
                                   verified_sessions: int = None) -> bool:
        """更新单个 patch 的状态和已验证会话数。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if status == "applied":
            cursor.execute("""
                UPDATE prompt_patches SET status = ?, applied_at = ?
                WHERE patch_id = ?
            """, (status, now, patch_id))
        elif status == "failed":
            cursor.execute("""
                UPDATE prompt_patches SET status = ?, failed_at = ?
                WHERE patch_id = ?
            """, (status, now, patch_id))
        elif status == "rolled_back":
            cursor.execute("""
                UPDATE prompt_patches SET status = ?, rolled_back_at = ?
                WHERE patch_id = ?
            """, (status, now, patch_id))
        else:
            cursor.execute("""
                UPDATE prompt_patches SET status = ?, verified_sessions = COALESCE(verified_sessions, ?)
                WHERE patch_id = ?
            """, (status, verified_sessions, patch_id))
        conn.commit()
        return cursor.rowcount > 0

    async def mark_patches_applied(self, user_id: str, session_id: str) -> int:
        """Compatibility helper: explicit operators may start an experiment."""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE prompt_patches
            SET status = 'experiment', applied_at = ?, session_id = ?
            WHERE user_id = ? AND status = 'candidate'
        """, (now, session_id, user_id))
        conn.commit()
        return cursor.rowcount

    async def start_prompt_patch_experiment(self, user_id: str, patch_id: str) -> bool:
        """Explicitly promote one reviewed candidate into an experiment."""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """UPDATE prompt_patches SET status = 'experiment', applied_at = ?
               WHERE user_id = ? AND patch_id = ? AND status = 'candidate'""",
            (now, user_id, patch_id),
        )
        conn.commit()
        return cursor.rowcount == 1

    # ── 素材（plans/0916-21 D4/F3：粘贴/日记 + 网页链接 → 一盘能聊的素材带）──────
    # 方法面与 L2PostgresMemory **必须一致**（路由直接调 l2，缺一个就是线上 500
    # 或被 hasattr 兜成静默空值）。本后端的 user_id 不做归一（与全文件一致：
    # 存的就是调用方传来的原串）；PG 那边才走 _as_uuid。

    @staticmethod
    def _material_row(row, with_text: bool = True) -> dict:
        """行 → dict。with_text=False 用于列表（不回传正文，少一分泄露面）。"""
        from datetime import datetime as _dt
        if not row:
            return {}
        data = dict(row)
        data.pop("user_id", None)
        if with_text:
            from engine.chunk_evidence import loads_json_list
            data["blocks"] = loads_json_list(data.get("blocks"))
            data["raw_text"] = str(data.get("raw_text") or "")
        else:
            data.pop("raw_text", None)
            data.pop("blocks", None)
        # 深析产物（plans/0916-24）：本后端是 JSON 文本，与 PG 的 jsonb 同形状返回
        try:
            parsed = data.get("analysis")
            if isinstance(parsed, str):
                parsed = json.loads(parsed or "{}")
            data["analysis"] = parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            data["analysis"] = {}
        for key, value in list(data.items()):
            if isinstance(value, _dt):
                data[key] = value.isoformat()
        return data

    async def add_material(self, user_id: str, kind: str, title: str,
                           source_ref: str = "", raw_text: str = "",
                           summary: str = "", char_count: int = 0,
                           status: str = "ready", error: str = "",
                           blocks: list = None,
                           material_conv_id: str = "",
                           analysis: dict = None) -> int:
        """落一条素材，返回新行 id。失败件**必须**留 material_conv_id 为空。

        analysis（plans/0916-24）：深度分析产物（六字段框架，缺项空项正常）。
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO material_items
               (user_id, kind, title, source_ref, raw_text, summary, char_count,
                material_conv_id, status, error, blocks, analysis)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, str(kind or "text"), str(title or ""), str(source_ref or ""),
             str(raw_text or ""), str(summary or ""), int(char_count or 0),
             str(material_conv_id or ""), str(status or "ready"), str(error or ""),
             json.dumps(list(blocks or []), ensure_ascii=False),
             json.dumps(dict(analysis or {}), ensure_ascii=False)))
        conn.commit()
        return int(cursor.lastrowid)

    async def list_materials(self, user_id: str, kind: str = "",
                             limit: int = 20, with_text: bool = False) -> list:
        conn = self._get_conn()
        cursor = conn.cursor()
        if str(kind or "").strip():
            cursor.execute(
                "SELECT * FROM material_items WHERE user_id = ? AND kind = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user_id, str(kind).strip(), int(limit)))
        else:
            cursor.execute(
                "SELECT * FROM material_items WHERE user_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user_id, int(limit)))
        return [self._material_row(r, with_text=with_text) for r in cursor.fetchall()]

    async def get_material(self, user_id: str, material_id,
                           with_text: bool = True) -> dict:
        try:
            mid = int(str(material_id).strip())
        except (ValueError, TypeError):
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM material_items WHERE user_id = ? AND id = ?",
                       (user_id, mid))
        return self._material_row(cursor.fetchone(), with_text=with_text)

    async def get_material_by_conv(self, user_id: str, conv_id: str) -> dict:
        """按"它作为一盘带的 conv_id"取素材（F4：素材带进会话时注入 [Material]）。"""
        conv = str(conv_id or "").strip()
        if not conv:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM material_items WHERE user_id = ? AND material_conv_id = ? "
            "ORDER BY id DESC LIMIT 1", (user_id, conv))
        return self._material_row(cursor.fetchone())

    async def latest_diary_material(self, user_id: str,
                                    within_hours: int = 72) -> dict:
        """最近一篇日记级素材（F1 四级取材第 1 级）。超期/没有 → {}。

        注意用 UTC 对齐 SQLite 的 CURRENT_TIMESTAMP（本地时间会差 8 小时，
        窗口判据必须同一时基）。
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        cutoff = (_dt.now(_tz.utc) - _td(hours=int(within_hours))
                  ).strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM material_items WHERE user_id = ? AND kind = 'diary' "
            "AND status = 'ready' AND created_at >= ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1", (user_id, cutoff))
        return self._material_row(cursor.fetchone(), with_text=False)

    async def delete_material(self, user_id: str, material_id) -> bool:
        """删一条素材（界面入口是 V1.1，方法面先就位）。"""
        try:
            mid = int(str(material_id).strip())
        except (ValueError, TypeError):
            return False
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM material_items WHERE user_id = ? AND id = ?",
                       (user_id, mid))
        conn.commit()
        return cursor.rowcount > 0

    # ── 素材库（plans/0923-08）：先入库、选材后成盘 ────────────────────────
    # 方法面与 L2PostgresMemory **必须一致**（同名同参数同返回键）。

    @staticmethod
    def _tape_build_row(row) -> dict:
        from datetime import datetime as _dt
        from engine.chunk_evidence import loads_json_list
        if not row:
            return {}
        data = dict(row)
        data.pop("user_id", None)
        for key in ("source_ids", "groups", "hooks", "top_chunks"):
            data[key] = loads_json_list(data.get(key))
        for key, value in list(data.items()):
            if isinstance(value, _dt):
                data[key] = value.isoformat()
        return data

    async def list_material_library_page(self, user_id: str, cursor: str = "",
                                         section: str = "", limit: int = 20) -> dict:
        """素材库分页：稳定游标 = 上一页最后一条 id（倒序扫描）。

        section: '' 全部 / 'material' 一般素材（非日记）/ 'diary' 日记。
        返回 {"items": [...], "next_cursor": str, "has_more": bool}——
        items 不回 raw_text（少一分泄露面），形状与 PG 侧一致。
        """
        try:
            before = int(str(cursor).strip()) if str(cursor or "").strip() else 0
        except (ValueError, TypeError):
            before = 0
        n = max(1, min(int(limit or 20), 100))
        where = "user_id = ?"
        args: list = [user_id]
        if str(section or "").strip() == "material":
            where += " AND kind <> 'diary'"
        elif str(section or "").strip() == "diary":
            where += " AND kind = 'diary'"
        if before:
            where += " AND id < ?"
            args.append(before)
        args.append(n + 1)
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            f"""SELECT id, kind, title, source_ref, summary, char_count,
                       material_conv_id, status, error, created_at, updated_at
                FROM material_items WHERE {where}
                ORDER BY id DESC LIMIT ?""", tuple(args))
        rows = [self._material_row(r, with_text=False) for r in cur.fetchall()]
        has_more = len(rows) > n
        items = rows[:n]
        return {
            "items": items,
            "next_cursor": str(items[-1]["id"]) if (items and has_more) else "",
            "has_more": has_more,
        }

    async def put_material_source_blocks(self, user_id: str, material_id,
                                         blocks: list) -> int:
        """整份覆盖写入某素材的全量正文/转写块，返回写入块数。"""
        try:
            mid = int(str(material_id).strip())
        except (ValueError, TypeError):
            return 0
        rows = []
        for index, block in enumerate(list(blocks or [])):
            if isinstance(block, dict):
                text = str(block.get("text") or "")
                span = str(block.get("span_kind") or "body")
                no = block.get("block_no")
            else:
                text, span, no = str(block or ""), "body", None
            try:
                no = int(no) if no is not None else index + 1
            except (ValueError, TypeError):
                no = index + 1
            rows.append((no, span, text))
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM material_source_blocks WHERE user_id = ? AND material_id = ?",
                    (user_id, mid))
        for no, span, text in rows:
            cur.execute(
                """INSERT INTO material_source_blocks
                       (user_id, material_id, block_no, span_kind, text)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, material_id, block_no) DO UPDATE SET
                       span_kind = excluded.span_kind, text = excluded.text""",
                (user_id, mid, no, span, text))
        conn.commit()
        return len(rows)

    async def list_material_source_blocks(self, user_id: str, material_ids,
                                          limit_per_material: int = 0) -> list:
        """取若干素材的全文块（按素材、块号升序）。material_ids 传 int 或列表。
        limit_per_material<=0 = 每份取全部。返回行带 material_id。"""
        if isinstance(material_ids, (list, tuple, set)):
            ids = []
            for value in material_ids:
                try:
                    ids.append(int(str(value).strip()))
                except (ValueError, TypeError):
                    continue
        else:
            try:
                ids = [int(str(material_ids).strip())]
            except (ValueError, TypeError):
                return []
        if not ids:
            return []
        conn = self._get_conn()
        cur = conn.cursor()
        out: list = []
        cap = max(0, int(limit_per_material or 0))
        for mid in ids:
            if cap:
                cur.execute(
                    """SELECT material_id, block_no, span_kind, text
                       FROM material_source_blocks
                       WHERE user_id = ? AND material_id = ?
                       ORDER BY block_no ASC LIMIT ?""", (user_id, mid, cap))
            else:
                cur.execute(
                    """SELECT material_id, block_no, span_kind, text
                       FROM material_source_blocks
                       WHERE user_id = ? AND material_id = ?
                       ORDER BY block_no ASC""", (user_id, mid))
            for row in cur.fetchall():
                out.append({"material_id": int(row["material_id"]),
                            "block_no": int(row["block_no"]),
                            "span_kind": str(row["span_kind"] or "body"),
                            "text": str(row["text"] or "")})
        return out

    async def create_material_tape_build(self, user_id: str, request_id: str,
                                         source_ids: list, status: str = "pending",
                                         error: str = "") -> dict:
        """建一条成盘构建记录（request_id 对同一用户唯一，重放回原行）。"""
        req = str(request_id or "").strip()
        if not req:
            return {}
        ids = []
        for value in list(source_ids or []):
            try:
                ids.append(int(str(value).strip()))
            except (ValueError, TypeError):
                continue
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO material_tape_builds
                   (user_id, request_id, status, source_ids, error)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, request_id) DO NOTHING""",
            (user_id, req, str(status or "pending"),
             json.dumps(ids, ensure_ascii=False), str(error or "")))
        conn.commit()
        cur.execute("SELECT * FROM material_tape_builds WHERE user_id = ? AND request_id = ?",
                    (user_id, req))
        return self._tape_build_row(cur.fetchone())

    async def get_material_tape_build(self, user_id: str, build_id=0,
                                      request_id: str = "", conv_id: str = "") -> dict:
        conn = self._get_conn()
        cur = conn.cursor()
        if str(conv_id or "").strip():
            cur.execute(
                "SELECT * FROM material_tape_builds WHERE user_id = ? AND conv_id = ? "
                "ORDER BY id DESC LIMIT 1", (user_id, str(conv_id).strip()))
        elif str(request_id or "").strip():
            cur.execute(
                "SELECT * FROM material_tape_builds WHERE user_id = ? AND request_id = ?",
                (user_id, str(request_id).strip()))
        else:
            try:
                bid = int(str(build_id).strip())
            except (ValueError, TypeError):
                return {}
            cur.execute("SELECT * FROM material_tape_builds WHERE user_id = ? AND id = ?",
                        (user_id, bid))
        return self._tape_build_row(cur.fetchone())

    async def finish_material_tape_build(self, user_id: str, build_id,
                                         conv_id: str, name: str = "",
                                         summary: str = "", groups: list = None,
                                         hooks: list = None,
                                         top_chunks: list = None) -> bool:
        """一次提交写两处：零消息 scene='material' 的 conversation_meta + 构建行置 ready。

        本后端的连接天然串行，所以两句之间不会被别的写插进来；PG 侧用显式事务
        （见 l2_postgres.finish_material_tape_build）达到同一效果。
        """
        try:
            bid = int(str(build_id).strip())
        except (ValueError, TypeError):
            return False
        conv = str(conv_id or "").strip()
        if not conv:
            return False
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("BEGIN")
            cur.execute(
                """INSERT INTO conversation_meta (user_id, conv_id, name, target_lang, level, scene)
                   VALUES (?, ?, ?, ?, ?, 'material')
                   ON CONFLICT(conv_id) DO UPDATE SET
                       name = excluded.name, scene = 'material',
                       updated_at = CURRENT_TIMESTAMP""",
                (user_id, conv, str(name or ""), "English", ""))
            cur.execute(
                """UPDATE material_tape_builds SET
                       status = 'ready', conv_id = ?, summary = ?, groups = ?,
                       hooks = ?, top_chunks = ?, error = ''
                   WHERE user_id = ? AND id = ?""",
                (conv, str(summary or ""),
                 json.dumps(list(groups or []), ensure_ascii=False),
                 json.dumps(list(hooks or []), ensure_ascii=False),
                 json.dumps(list(top_chunks or []), ensure_ascii=False),
                 user_id, bid))
            changed = cur.rowcount
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        return changed > 0

    async def fail_material_tape_build(self, user_id: str, build_id,
                                       error: str = "") -> bool:
        try:
            bid = int(str(build_id).strip())
        except (ValueError, TypeError):
            return False
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE material_tape_builds SET status = 'failed', error = ? "
            "WHERE user_id = ? AND id = ?", (str(error or "")[:400], user_id, bid))
        conn.commit()
        return cur.rowcount > 0

    async def get_conversation_scene(self, user_id: str, conv_id: str) -> str:
        """按 (user_id, conv_id) 精确取盘型——会话热入口不许扫整柜。"""
        conv = str(conv_id or "").strip()
        if not conv:
            return ""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT scene FROM conversation_meta WHERE user_id = ? AND conv_id = ? "
                    "ORDER BY id DESC LIMIT 1", (user_id, conv))
        row = cur.fetchone()
        return str(row["scene"] or "") if row else ""

    async def mark_material_ready(self, user_id: str, material_id, *,
                                  title: str = "", summary: str = "",
                                  char_count: int = 0, blocks: list = None,
                                  analysis: dict = None) -> bool:
        """轮外分析完成后把库条置 ready（标题/摘要/块一次性补齐）。"""
        try:
            mid = int(str(material_id).strip())
        except (ValueError, TypeError):
            return False
        sets = ["status = 'ready'", "error = ''", "updated_at = CURRENT_TIMESTAMP"]
        args: list = []
        if str(title or "").strip():
            sets.append("title = ?")
            args.append(str(title).strip())
        if str(summary or "").strip():
            sets.append("summary = ?")
            args.append(str(summary))
        if int(char_count or 0) > 0:
            sets.append("char_count = ?")
            args.append(int(char_count))
        if blocks is not None:
            sets.append("blocks = ?")
            args.append(json.dumps(list(blocks or []), ensure_ascii=False))
        if analysis is not None:
            sets.append("analysis = ?")
            args.append(json.dumps(dict(analysis or {}), ensure_ascii=False))
        args.extend([user_id, mid])
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f"UPDATE material_items SET {', '.join(sets)} "
                    "WHERE user_id = ? AND id = ?", tuple(args))
        conn.commit()
        return cur.rowcount > 0

    async def mark_material_failed(self, user_id: str, material_id,
                                   error: str = "") -> bool:
        """轮外阶段失败：明确置 failed + 原因，条目不可选入盘（绝不产空带）。"""
        try:
            mid = int(str(material_id).strip())
        except (ValueError, TypeError):
            return False
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE material_items SET status = 'failed', error = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND id = ?",
            (str(error or "")[:400], user_id, mid))
        conn.commit()
        return cur.rowcount > 0

    async def set_material_source_ref(self, user_id: str, material_id,
                                      source_ref: str) -> bool:
        """补写一条库条目的 source_ref（上传件落盘后才知道最终路径）。"""
        try:
            mid = int(str(material_id).strip())
        except (ValueError, TypeError):
            return False
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE material_items SET source_ref = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE user_id = ? AND id = ?", (str(source_ref or ""), user_id, mid))
        conn.commit()
        return cur.rowcount > 0

    async def list_material_jobs(self, status: str = "processing",
                                 limit: int = 50) -> list:
        """取待续跑的导入作业（进程重启后按 ID 幂等续跑，plans/0923-08 §4.1）。

        返回行带 user_id 与 raw_text/source_ref —— 续跑要靠它们重放。
        """
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, user_id, kind, source_ref, raw_text, title, status
               FROM material_items WHERE status = ?
               ORDER BY id ASC LIMIT ?""",
            (str(status or "processing"), max(1, min(int(limit or 50), 200))))
        out = []
        for row in cur.fetchall():
            item = dict(row)
            item["user_id"] = str(item.get("user_id") or "")
            item["raw_text"] = str(item.get("raw_text") or "")
            out.append(item)
        return out

    async def list_material_tape_builds(self, status: str = "pending",
                                        limit: int = 20) -> list:
        """取待续跑的成盘构建（服务重启后接着跑，plans/0923-08 §4.2）。"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, request_id, source_ids FROM material_tape_builds "
            "WHERE status = ? ORDER BY id ASC LIMIT ?",
            (str(status or "pending"), max(1, min(int(limit or 20), 100))))
        from engine.chunk_evidence import loads_json_list
        out = []
        for row in cur.fetchall():
            item = dict(row)
            item["user_id"] = str(item.get("user_id") or "")
            item["source_ids"] = loads_json_list(item.get("source_ids"))
            out.append(item)
        return out

    async def export_user_data(self, user_id: str) -> dict:
        """Return the learner-owned L2 records without mutating them."""
        conn = self._get_conn()
        cursor = conn.cursor()
        tables = {
            "errors": "error_fingerprints", "mastered": "mastered_expressions",
            "learning_items": "learning_item_states", "speech_assessments": "speech_assessments",
            "style_parameters": "style_parameters", "prompt_patches": "prompt_patches",
            "conversations": "conversations", "conversation_meta": "conversation_meta",
            "conversation_messages": "conversation_messages",
            "session_archives": "session_archives", "relationship_memory": "relationship_memory",
            "companion_persona_facts": "companion_persona_facts",
            "companion_persona_snapshots": "companion_persona_snapshots",
            "companion_life_items": "companion_life_items",
            "metering_events": "metering_events", "memory_curator_runs": "memory_curator_runs",
            "materials": "material_items",
            "memory_entities": "memory_entities",
            "memory_entity_relations": "memory_entity_relations",
            # plans/0923-08 素材库：全量正文块与成盘提要也要能导出
            "material_source_blocks": "material_source_blocks",
            "material_tape_builds": "material_tape_builds",
            # plans/0923-04 §4.5：统一画像两表必须出现在导出里
            "profile_facts": "learner_profile_facts",
            "profile_narratives": "learner_profile_narratives",
            # plans/0923-06 §4.1.2：练习盘来源快照也要能导出
            "expression_topic_assignments": "expression_topic_assignments",
        }
        exported = {}
        for key, table in tables.items():
            try:
                cursor.execute(f"SELECT * FROM {table} WHERE user_id = ?", (user_id,))
                exported[key] = [dict(row) for row in cursor.fetchall()]
            except Exception:
                exported[key] = []
        return exported

    async def delete_user_data(self, user_id: str) -> dict:
        """Delete only rows belonging to one learner; caller must confirm separately."""
        conn = self._get_conn()
        cursor = conn.cursor()
        tables = ["error_fingerprints", "mastered_expressions", "learning_item_states",
                  "speech_assessments", "style_parameters", "prompt_patches", "conversations",
                  "conversation_meta", "conversation_messages", "session_archives",
                  "feedback_signals", "relationship_memory",
                  "companion_persona_facts", "companion_persona_snapshots", "companion_life_items",
                  "voice_turn_metrics", "metering_events", "memory_curator_runs", "cassette_handoffs",
                  "material_items", "material_source_blocks", "material_tape_builds",
                  "memory_entity_relations", "memory_entities",
                  "learner_profile_facts", "learner_profile_narratives",
                  "expression_topic_assignments"]
        deleted = {}
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                deleted[table] = cursor.rowcount
            except Exception:
                deleted[table] = 0
        conn.commit()
        return deleted

    async def rollback_failed_patches(self, user_id: str) -> int:
        """回滚验证失败次数超限的 patches（verified_sessions >= 3 且仍需手动回滚的在 audit 端处理）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE prompt_patches
            SET status = 'rolled_back', rolled_back_at = ?
            WHERE user_id = ? AND status = 'failed'
            AND verified_sessions >= max_retries
        """, (now, user_id))
        conn.commit()
        return cursor.rowcount

    async def get_pending_patches_count(self, user_id: str) -> int:
        """获取当前 pending + applied 但未验证的 patches 数量（用于安全检查：不超过5条）。"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM prompt_patches
            WHERE user_id = ? AND status IN ('pending', 'applied')
            AND verified_sessions < 3
        """, (user_id,))
        row = cursor.fetchone()
        return row["cnt"] if row else 0
