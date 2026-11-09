import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class FeedbackSignal:
    """反馈信号"""
    session_id: str
    user_id: str
    turn_index: int = 0
    feedback_type: str = ""  # "explicit_positive", "explicit_negative", "implicit_continue", "implicit_abandon", "implicit_interrupt"
    signal_value: float = 0.0  # -1.0 to 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackCollector:
    """反馈采集器"""

    def __init__(self, l2_memory=None):
        self.l2 = l2_memory

    async def collect_implicit(
        self,
        session_id: str,
        user_id: str,
        turn_count: int,
        was_interrupted: bool = False,
        was_abandoned: bool = False,
        context: Dict[str, Any] = None,
    ) -> List[FeedbackSignal]:
        """采集隐式反馈"""
        signals = []

        # 会话长度反馈（轮次越多 = 越正面）
        if turn_count >= 8:
            signals.append(FeedbackSignal(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_count,
                feedback_type="implicit_continue",
                signal_value=0.7,
                context=context or {},
                timestamp=time.time(),
            ))
        elif turn_count >= 5:
            signals.append(FeedbackSignal(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_count,
                feedback_type="implicit_continue",
                signal_value=0.4,
                context=context or {},
                timestamp=time.time(),
            ))
        elif turn_count <= 2:
            signals.append(FeedbackSignal(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_count,
                feedback_type="implicit_abandon",
                signal_value=-0.5,
                context=context or {},
                timestamp=time.time(),
            ))

        # 打断反馈（负面）
        if was_interrupted:
            signals.append(FeedbackSignal(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_count,
                feedback_type="implicit_interrupt",
                signal_value=-0.3,
                context=context or {},
                timestamp=time.time(),
            ))

        # 保存到数据库
        for signal in signals:
            await self._save_signal(signal)

        return signals

    async def collect_explicit(
        self,
        session_id: str,
        user_id: str,
        turn_index: int,
        feedback_type: str,  # "positive" or "negative"
        context: Dict[str, Any] = None,
    ) -> FeedbackSignal:
        """采集显式反馈"""
        value = 1.0 if feedback_type == "positive" else -1.0

        signal = FeedbackSignal(
            session_id=session_id,
            user_id=user_id,
            turn_index=turn_index,
            feedback_type=f"explicit_{feedback_type}",
            signal_value=value,
            context=context or {},
            timestamp=time.time(),
        )

        await self._save_signal(signal)
        return signal

    async def _save_signal(self, signal: FeedbackSignal):
        """保存反馈信号到数据库"""
        if not self.l2:
            return

        try:
            if hasattr(self.l2, "insert_feedback_signal"):
                await self.l2.insert_feedback_signal(signal.to_dict())
                return
            conn = self.l2._get_conn()
            conn.execute(
                """INSERT INTO feedback_signals 
                   (session_id, user_id, turn_index, feedback_type, signal_value, context, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.session_id,
                    signal.user_id,
                    signal.turn_index,
                    signal.feedback_type,
                    signal.signal_value,
                    json.dumps(signal.context, ensure_ascii=False),
                    signal.timestamp,
                )
            )
            conn.commit()
        except Exception:
            pass

    async def get_recent_feedback(
        self, user_id: str, limit: int = 20
    ) -> List[FeedbackSignal]:
        """获取最近的反馈信号"""
        if not self.l2:
            return []

        try:
            if hasattr(self.l2, "get_recent_feedback_signals"):
                rows = await self.l2.get_recent_feedback_signals(user_id, limit=limit)
                return [FeedbackSignal(**row) for row in rows]
            conn = self.l2._get_conn()
            cursor = conn.execute(
                """SELECT session_id, user_id, turn_index, feedback_type, signal_value, context, timestamp
                   FROM feedback_signals
                   WHERE user_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (user_id, limit)
            )
            rows = cursor.fetchall()
            signals = []
            for row in rows:
                signals.append(FeedbackSignal(
                    session_id=row[0],
                    user_id=row[1],
                    turn_index=row[2],
                    feedback_type=row[3],
                    signal_value=row[4],
                    context=json.loads(row[5]) if row[5] else {},
                    timestamp=row[6],
                ))
            return signals
        except Exception:
            return []
