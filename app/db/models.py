from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMENSIONS = 768


class Base(DeclarativeBase):
    pass


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base, Timestamped):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)


class Thread(Base, Timestamped):
    __tablename__ = "threads"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    id: Mapped[str] = mapped_column(String(255), primary_key=True)


class Message(Base, Timestamped):
    __tablename__ = "messages"
    __table_args__ = (ForeignKeyConstraint(
        ["user_id", "thread_id"], ["threads.user_id", "threads.id"], ondelete="CASCADE"
    ),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(255))
    thread_id: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)


class AgentRun(Base, Timestamped):
    __tablename__ = "agent_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id", "thread_id"], ["threads.user_id", "threads.id"], ondelete="CASCADE"
        ),
        Index("ix_agent_runs_thread_latest", "user_id", "thread_id", "created_at", "id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(255))
    thread_id: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32))
    state: Mapped[dict] = mapped_column(JSONB)


class AgentEvent(Base, Timestamped):
    __tablename__ = "agent_events"
    __table_args__ = (UniqueConstraint("run_id", "sequence"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    sequence: Mapped[int]
    event: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB)


class SemanticMemory(Base, Timestamped):
    __tablename__ = "memories"
    __table_args__ = (UniqueConstraint("user_id", "embedding_model", "memory_key"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    memory_key: Mapped[str] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(255))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Episode(Base, Timestamped):
    __tablename__ = "episodes"
    __table_args__ = (ForeignKeyConstraint(
        ["user_id", "thread_id"], ["threads.user_id", "threads.id"], ondelete="CASCADE"
    ),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    thread_id: Mapped[str] = mapped_column(String(255))
    task: Mapped[str] = mapped_column(Text)
    actions: Mapped[list[str]] = mapped_column(JSONB)
    result: Mapped[str] = mapped_column(Text)
    success: Mapped[bool]


class Evaluation(Base, Timestamped):
    __tablename__ = "evaluations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"))
    case_id: Mapped[str] = mapped_column(String(255))
    metrics: Mapped[dict] = mapped_column(JSONB)


class ToolCall(Base, Timestamped):
    __tablename__ = "tool_calls"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    tool: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict] = mapped_column(JSONB)
    observation: Mapped[dict] = mapped_column(JSONB)


class AuditLog(Base, Timestamped):
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB)
