"""
Session Manager - Implements the Hermes Agent session lifecycle.

Hermes Agent 是 Nous Research 的开源 Agent 框架，提供以下核心能力：
- MEMORY.md: 永久知识、事实（L1 稳定画像）
- USER.md: 用户偏好、约束、学习风格
- learning.md: 临时 scratchpad，每次会话重置
- Nudge Engine: 基于回合数的渐进式触发机制（3/10/会话结束）

会话级上下文冻结机制（应用层业务逻辑，与 Redis RDB/AOF 持久化无关）：
1. Initialize: 加载会话级上下文冻结的 MEMORY.md + USER.md 到系统提示
2. During: 新学习内容写入 learning.md（临时 scratchpad）
3. End: Nudge Engine 将 learning.md 合成到 MEMORY.md 更新

会话生命周期：
1. 开始会话 → 调用 L1.get_frozen_snapshot() 获取会话级上下文冻结
2. 每轮对话 → 记录 learning.md，检查 Nudge Engine 阈值
3. 结束会话 → 合成 learning.md，更新 L1 画像

参考 Hermes Agent 官方文档：
"built-in learning loop — creates skills from experience, improves them during use"
"""

import json
from datetime import datetime, timezone
from typing import Optional

# 开源包只有本地后端；原仓这里按部署形态探测 Redis/Postgres 实现，已移除。
_L1RedisProfile = object
_L2PostgresMemory = object


# Turn-based Nudge thresholds (from Hermes Agent)
NUDGE_LIGHT_TURNS = 3
NUDGE_MEDIUM_TURNS = 10
NUDGE_HEAVY_DAILY = True


class SessionManager:
    """
    Hermes-style session manager.

    Tracks turns within a session and triggers Nudge Engine at
    progressive thresholds: 3 turns (light), 10 turns (medium),
    session end (heavy).
    """

    def __init__(
        self,
        l1: _L1RedisProfile,
        l2: _L2PostgresMemory,
    ):
        self.l1 = l1
        self.l2 = l2
        self._active_sessions: dict[str, dict] = {}

    # ── Session lifecycle ─────────────────────────────────

    async def start_session(self, user_id: str) -> dict:
        """
        Initialize a new session.
        Loads session-level context-frozen MEMORY.md + USER.md.
        Creates ephemeral learning.md scratchpad.

        注：此处的"上下文冻结"是应用层业务逻辑，在会话开始时锁定用户画像，
        以保证会话期间数据一致性，与 Redis 的 RDB/AOF 持久化快照机制完全不同。
        """
        # Load session-level context-frozen snapshots
        memory_md = await self._build_memory_md(user_id)
        user_md = await self._build_user_md(user_id)

        session_state = {
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc),
            "turn_count": 0,
            "learning_md": [],          # ephemeral scratchpad
            "memory_md": memory_md,     # session-level context-frozen
            "user_md": user_md,         # session-level context-frozen
            "nudge_level_reached": [],  # track which nudge levels fired
        }
        self._active_sessions[user_id] = session_state

        return {
            "memory_md": memory_md,
            "user_md": user_md,
            "learning_md": [],
        }

    async def record_turn(
        self,
        user_id: str,
        transcript: str,
        errors: list[dict],
    ) -> Optional[str]:
        """
        Record one conversation turn.
        Returns nudge level if triggered: "light" | "medium" | None.
        """
        if user_id not in self._active_sessions:
            return None

        session = self._active_sessions[user_id]
        session["turn_count"] += 1

        # Append learning note to scratchpad
        if errors:
            session["learning_md"].append({
                "turn": session["turn_count"],
                "errors": [e["type"] for e in errors],
                "excerpt": transcript[:200],
            })

        # Check Nudge Engine turn-based thresholds
        tc = session["turn_count"]
        if tc == NUDGE_LIGHT_TURNS and "light" not in session["nudge_level_reached"]:
            session["nudge_level_reached"].append("light")
            return "light"
        elif tc == NUDGE_MEDIUM_TURNS and "medium" not in session["nudge_level_reached"]:
            session["nudge_level_reached"].append("medium")
            return "medium"

        return None

    def get_turn_count(self, user_id: str) -> int:
        """Get current turn count for an active session."""
        session = self._active_sessions.get(user_id)
        return session["turn_count"] if session else 0

    def get_learning_count(self, user_id: str) -> int:
        """Get number of learning notes in current session."""
        session = self._active_sessions.get(user_id)
        return len(session["learning_md"]) if session else 0

    async def end_session(self, user_id: str) -> dict:
        """
        End session: trigger heavy nudge synthesis.
        Returns a summary of what changed.
        """
        if user_id not in self._active_sessions:
            return {"status": "no_active_session"}

        session = self._active_sessions.pop(user_id)

        # Synthesize learning.md into MEMORY.md
        summary = await self._synthesize(user_id, session)

        return {
            "status": "synthesized",
            "turns": session["turn_count"],
            "learnings": len(session["learning_md"]),
            "nudge_levels": session["nudge_level_reached"] + ["heavy"],
            "summary": summary,
        }

    # ── MEMORY.md / USER.md builders ──────────────────────

    async def _build_memory_md(self, user_id: str) -> str:
        """
        Build MEMORY.md from L1 stable profile + L2 facts.
        This is the session-level context-frozen snapshot of permanent knowledge.
        """
        profile = await self.l1.get_frozen_snapshot(user_id)
        errors = await self.l2.get_errors(user_id, limit=10)
        mastered = await self.l2.get_mastered(user_id, limit=10)

        sections = [
            f"# MEMORY.md - User {user_id}",
            f"CEFR Level: {profile.get('cefr', 'B1')}",
            f"Learning Style: {profile.get('learning_style', 'balanced')}",
            "",
            "## Current Weaknesses",
            profile.get("recent_weakness", "None identified"),
            "",
            "## Error Patterns",
        ]

        for e in errors:
            sections.append(
                f"- {e['error_type']}: {e['error_excerpt'] or '(no excerpt)'} "
                f"(count={e['count']}, weight={e['weight']:.2f})"
            )

        sections.append("")
        sections.append("## Mastered Expressions")
        for m in mastered:
            sections.append(
                f"- {m['expression']} (confidence={m['confidence']:.2f})"
            )

        return "\n".join(sections)

    async def _build_user_md(self, user_id: str) -> str:
        """
        Build USER.md from L1 preferences.
        This is the session-level context-frozen snapshot of user preferences/constraints.
        """
        profile = await self.l1.get_frozen_snapshot(user_id)
        topics = profile.get("preferred_topics", [])
        topics_str = ", ".join(topics) if topics else "general"

        return (
            f"# USER.md\n"
            f"Preferred Topics: {topics_str}\n"
            f"Learning Style: {profile.get('learning_style', 'balanced')}\n"
            f"CEFR Level: {profile.get('cefr', 'B1')}\n"
        )

    # ── Synthesis (heavy nudge) ───────────────────────────

    async def _synthesize(self, user_id: str, session: dict) -> str:
        """
        Synthesize learning.md into MEMORY.md updates.
        Hermes Agent calls this at session end (heavy nudge).
        """
        learnings = session["learning_md"]
        if not learnings:
            return "No learnings to synthesize."

        # Collect unique error types from this session
        error_types = set()
        for note in learnings:
            for err in note["errors"]:
                error_types.add(err)

        # Update L1 with new weakness
        weakness = ", ".join(sorted(error_types)) if error_types else ""
        if weakness:
            await self.l1.update_profile(user_id, {
                "recent_weakness": weakness,
            })

        return (
            f"Synthesized {len(learnings)} learning notes across "
            f"{session['turn_count']} turns. "
            f"Error types: {weakness or 'none'}."
        )
