from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.llm.model import ChatMessage
from app.observability.events import RunEvents


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, JsonValue]


class PendingApproval(ToolCall):
    approval_id: str


class ToolAttempt(TypedDict):
    attempt: int
    latency_ms: float
    error: str | None
    success: bool


class Observation(TypedDict):
    tool: str
    result: object
    success: bool
    attempts: list[ToolAttempt]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan: list[Annotated[str, Field(max_length=500)]] | None = Field(default=None, max_length=10)


class FinalDecision(Decision):
    action: Literal["final"]
    answer: str = Field(min_length=1)


class ToolDecision(Decision):
    action: Literal["tool"]
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, JsonValue]


class AgentDecision(BaseModel):
    decision: Annotated[FinalDecision | ToolDecision, Field(discriminator="action")]


class AgentContext(TypedDict):
    permissions: frozenset[str]
    conversation_message_limit: int
    events: RunEvents | None
    deadline: float


class AgentState(TypedDict):
    run_id: str
    thread_id: str
    user_id: str
    task: str
    messages: list[ChatMessage]
    conversation: list[ChatMessage]
    current_plan: list[str]
    current_step: int
    max_steps: int
    tool_calls: list[ToolCall]
    observations: list[Observation]
    proposed_tool: ToolCall | None
    retrieved_memories: list[str]
    pending_approval: PendingApproval | None
    remaining_seconds: float | None
    final_answer: str | None
    status: Literal["running", "completed", "failed", "awaiting_approval"]
    error: str | None
    metadata: dict[str, object]
