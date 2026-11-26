from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.agent.state import Observation, PendingApproval, ToolCall
from app.llm.model import ChatMessage

ThreadId = Annotated[str, StringConstraints(
    min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
)]
Task = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16000)]


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: ThreadId | None = None
    task: Task


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: UUID


class AgentRunResponse(BaseModel):
    run_id: UUID
    thread_id: str
    status: Literal["running", "completed", "failed", "awaiting_approval"]
    final_answer: str | None
    error: str | None
    current_plan: list[str]
    current_step: int
    max_steps: int
    messages: list[ChatMessage]
    tool_calls: list[ToolCall]
    observations: list[Observation]
    retrieved_memories: list[str]
    pending_approval: PendingApproval | None
