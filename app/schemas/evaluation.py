from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Score = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0)]
Phrase = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[Phrase, Field(max_length=255)]
    input: Phrase
    expected: Phrase
    case_sensitive: bool = False
    answer_from_tool_field: Phrase | None = None
    tools: list[Phrase] = Field(default_factory=list)
    criteria: list[Phrase] = Field(default_factory=list)
    expected_memories: list[Phrase] = Field(default_factory=list)
    max_steps: Count | None = None
    max_tool_calls: Count | None = None
    max_retries: Count | None = None
    max_latency_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None


class EvaluationMetrics(BaseModel):
    task_success: bool
    answer_quality: Score
    tool_accuracy: Score
    tool_success: Score
    trajectory_quality: Score
    memory_precision: Score
    memory_recall: Score
    efficiency: Score
    steps: Count
    tool_calls: Count
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    retry_count: Count
    errors: list[str]
    aggregate_score: Score


class EvaluationResult(BaseModel):
    case_id: str
    run_id: UUID | None = None
    status: Literal["running", "completed", "failed", "awaiting_approval"]
    passed: bool
    metrics: EvaluationMetrics


class StoredEvaluation(EvaluationResult):
    id: UUID
    created_at: datetime


class EvaluationReport(BaseModel):
    results: list[EvaluationResult]
    total: Count
    passed: Count
    failed: Count
    success_rate: Score
    average_latency_ms: float
    average_steps: float
    average_tool_calls: float
    average_aggregate_score: Score
