"""Turn-level middleware — inspired by DeepAgents middleware pattern.

DeepAgents runs middleware on every step via on_chat_model_start/end, on_tool_start/end.
Adapted here for tutoring turns: before_turn (pre-process user utterance) /
after_turn (post-process assistant response). Middleware composes the processing
pipeline, making it pluggable instead of hardcoded inline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class TurnPhase(Enum):
    BEFORE = "before"  # Pre-turn: user utterance has arrived
    AFTER = "after"    # Post-turn: assistant has responded


@dataclass
class TurnContext:
    """Carrier object passed through the middleware chain.

    Mirrors DeepAgents' State that middlewares read/write.
    """
    session_id: str
    user_id: str
    target_lang: str
    user_transcript: str
    assistant_response: str = ""
    turn_count: int = 0
    # Mutable state that middlewares can annotate
    errors: List[Dict[str, Any]] = field(default_factory=list)
    new_skills: List[str] = field(default_factory=list)
    nudge_triggered: Optional[str] = None
    drift_detected: bool = False
    notes: List[str] = field(default_factory=list)  # Teaching notes from middleware
    # Hook for middleware to signal action
    terminate: bool = False   # Set True to skip remaining middleware


class TurnMiddleware(ABC):
    """Base class for turn-level middleware.

    Each middleware receives the TurnContext and can mutate it.
    Return the (possibly modified) context.  Raise TurnMiddleware.Skip
    to abort the chain for this phase.

    This is the same composition pattern as DeepAgents middleware:
    each middleware runs on every step, can read/write state,
    can short-circuit.
    """
    class Skip(Exception):
        """Raise to skip remaining middleware in current phase."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def before_turn(self, ctx: TurnContext) -> TurnContext:
        """Called before the assistant processes the user utterance."""
        return ctx

    @abstractmethod
    async def after_turn(self, ctx: TurnContext) -> TurnContext:
        """Called after the assistant has responded."""
        return ctx


class MiddlewarePipeline:
    """Runs a list of middleware in sequence for each phase.

    Equivalent to DeepAgents' AgentExecutor running middleware on each step.
    """

    def __init__(self, middlewares: Optional[List[TurnMiddleware]] = None):
        self._middlewares: List[TurnMiddleware] = middlewares or []

    def add(self, mw: TurnMiddleware) -> "MiddlewarePipeline":
        self._middlewares.append(mw)
        return self

    def extend(self, mws: List[TurnMiddleware]) -> "MiddlewarePipeline":
        self._middlewares.extend(mws)
        return self

    @property
    def middlewares(self) -> List[TurnMiddleware]:
        return list(self._middlewares)

    async def run_before(self, ctx: TurnContext) -> TurnContext:
        for mw in self._middlewares:
            try:
                ctx = await mw.before_turn(ctx)
                if ctx.terminate:
                    break
            except TurnMiddleware.Skip:
                break
        return ctx

    async def run_after(self, ctx: TurnContext) -> TurnContext:
        for mw in self._middlewares:
            try:
                ctx = await mw.after_turn(ctx)
                if ctx.terminate:
                    break
            except TurnMiddleware.Skip:
                break
        return ctx
