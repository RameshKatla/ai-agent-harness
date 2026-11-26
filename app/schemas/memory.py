from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints

MemoryContent = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class MemoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: MemoryContent


class MemoryRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    content: str
    created_at: datetime
    updated_at: datetime


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    thread_id: str
    task: str
    actions: list[str]
    result: str
    success: bool
    created_at: datetime
