"""
MemoryProvider - Abstract base class aligned with Hermes Agent.
Based on: https://github.com/NousResearch/hermes-agent/blob/main/agent/memory_provider.py

Key lifecycle (called by MemoryManager in run_agent.py):
  initialize()          - connect, create resources, warm up
  system_prompt_block() - static text for the frozen system prompt
  prefetch(query)       - background recall before each turn
  sync_turn(user, asst) - async write after each turn
  get_tool_schemas()    - tool schemas to expose to the model
  handle_tool_call()    - dispatch a tool call
  shutdown()            - clean exit

Optional hooks (override to opt in):
  on_turn_start(turn, message, **kwargs) - per-turn tick
  on_session_end(messages)               - end-of-session Nudge Engine extraction
  on_session_switch(new_session_id)      - mid-process session rotation
  on_pre_compress(messages) -> str       - extract before context compression
  on_memory_write(action, target, content) - mirror built-in memory writes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import inspect

try:
    from ..config import MEMORY_BACKEND
except ImportError:
    from config import MEMORY_BACKEND


async def maybe_await(value: Any) -> Any:
    """等待一个「可能是 coroutine，也可能不是」的返回值。

    两个后端的同一批写入器签名一致但实现方式不同：
    - 本地后端（SQLite）里 add_error / add_mastered / set_error_fingerprint /
      save_session_archive 是同步 `def`；
    - redis 后端（Postgres）必须 await 才能真的落库。

    调用点统一写 `await maybe_await(self.l2.add_error(...))`，两边都成立。
    刻意**不用** fire-and-forget（create_task）：否则进程退出时写入会丢。
    """
    if inspect.isawaitable(value):
        return await value
    return value


def create_memory_backend() -> Tuple[Any, Any]:
    """
    根据 MEMORY_BACKEND 配置创建 L1 和 L2 记忆后端。
    
    返回：(L1实现, L2实现)
    
    MEMORY_BACKEND 选项：
    - "local": 使用本地文件存储（JSON + SQLite），零外部依赖
    - "redis": 使用 Redis + PostgreSQL（需要安装外部服务）
    """
    if MEMORY_BACKEND == "local":
        from .l1_file import L1FileProfile
        from .l2_sqlite import L2SQLiteMemory
        return L1FileProfile(), L2SQLiteMemory()
    raise RuntimeError("MEMORY_BACKEND 只支持 local；本开源包不含 PG/Redis 后端。")


@dataclass
class MemoryEntry:
    """Single memory entry, analogous to MemOS Memory Neuron."""
    entry_id: str
    content: str
    entry_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen_at: datetime = field(default_factory=datetime.now)


class MemoryProvider(ABC):
    """
    Hermes-aligned memory provider base class.

    Only ONE external plugin provider should be active at a time
    to prevent tool schema bloat and conflicting memory backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'pgvector', 'mem0', 'lance')."""

    # -- Core lifecycle --------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Check config, credentials, deps. No network calls."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """
        Initialize for a session. Called once at agent startup.
        kwargs: hermes_home, platform, agent_context, agent_identity,
                agent_workspace, parent_session_id, user_id, user_id_alt
        """

    def system_prompt_block(self) -> str:
        """
        STATIC text for frozen system prompt.
        Called during prompt assembly. Return "" to skip.
        Prefetch recall context is injected separately via prefetch().
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """
        Recall relevant context before a turn.
        Returns text to inject into the conversation context.
        Called by MemoryManager.prefetch_all().
        """
        return ""

    def sync_turn(self, user_msg: str, assistant_msg: str) -> None:
        """
        Async write after each turn. Non-blocking.
        Called by MemoryManager.sync_all().
        """

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for the model. Empty list = no tools."""
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call. Returns result string."""
        return ""

    def shutdown(self) -> None:
        """Clean exit: close connections, stop threads."""

    # -- Optional Nudge Engine hooks -------------------------------------

    def on_turn_start(self, turn: int, user_message: str, **kwargs) -> None:
        """Per-turn tick with runtime context."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """
        End-of-session Nudge Engine extraction.
        Hermes Agent calls this to synthesize learnings into permanent memory.
        This is where light/medium/heavy nudge logic fires.
        """

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        """Mid-process session rotation."""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract critical info BEFORE context compression.
        Returns summary to preserve across compression boundary.
        """
        return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes for external sync."""

    def backup_paths(self) -> List[str]:
        """Extra on-disk paths to include in backups."""
        return []
