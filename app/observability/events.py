import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue

logger = logging.getLogger(__name__)

EventName = Literal[
    "agent_start", "agent_step", "llm_start", "llm_token", "llm_end", "tool_start",
    "tool_result", "memory_retrieved", "approval_required", "agent_error", "agent_complete",
]


class AgentEvent(BaseModel):
    run_id: UUID
    thread_id: str
    user_id: str
    sequence: int
    timestamp: datetime
    step: int
    event: EventName
    node: str
    tool: str | None = None
    latency_ms: float | None = None
    success: bool | None = None
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


EventSink = Callable[[AgentEvent], Awaitable[None]]


class RunEvents:
    def __init__(self, run_id: UUID, thread_id: str, user_id: str, sink: EventSink | None) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.user_id = user_id
        self._sink = sink
        self._sequence = 0
        self._step = 0

    @property
    def streaming(self) -> bool:
        return self._sink is not None

    async def emit(
        self, event: EventName, *, node: str, step: int | None = None, tool: str | None = None,
        latency_ms: float | None = None, success: bool | None = None, error: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        item = AgentEvent(
            run_id=self.run_id, thread_id=self.thread_id, user_id=self.user_id,
            sequence=self._sequence + 1, timestamp=datetime.now(UTC), event=event,
            step=self._step if step is None else step, node=node, tool=tool, latency_ms=latency_ms, success=success,
            error=error, metadata=metadata if metadata is not None else {},
        )
        if self._sink is not None:
            await self._sink(item)
        self._sequence = item.sequence
        self._step = item.step
        if event != "llm_token":
            logger.info("Agent event", extra={
                "agent_event": item.model_dump(mode="json", exclude={"metadata", "error", "tool"}),
            })
