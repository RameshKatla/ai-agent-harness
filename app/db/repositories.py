from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import MemoryError, StorageError
from app.db.models import (
    AgentEvent, AgentRun, AuditLog, Episode, Evaluation, Message, SemanticMemory, Thread,
    ToolCall, User,
)
from app.observability.events import AgentEvent as ObservedEvent
from app.schemas.agent import AgentRunResponse
from app.schemas.evaluation import EvaluationResult, StoredEvaluation
from app.schemas.memory import EpisodeRecord, MemoryRecord


async def ensure_thread(session: AsyncSession, user_id: str, thread_id: str) -> None:
    await session.execute(insert(User).values(id=user_id).on_conflict_do_nothing())
    await session.execute(
        insert(Thread).values(user_id=user_id, id=thread_id).on_conflict_do_nothing()
    )


class MemoryRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncSession]:
        try:
            async with self._sessions.begin() as session:
                yield session
        except (SQLAlchemyError, OSError):
            raise MemoryError("The memory database operation failed.") from None

    async def save_memory(
        self, *, user_id: str, content: str, key: str, model: str, embedding: list[float],
    ) -> MemoryRecord:
        async with self._transaction() as session:
            await session.execute(insert(User).values(id=user_id).on_conflict_do_nothing())
            stmt = insert(SemanticMemory).values(
                user_id=user_id, content=content, memory_key=key,
                embedding_model=model, embedding=embedding,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id", "embedding_model", "memory_key"],
                set_={
                    "content": stmt.excluded.content, "embedding": stmt.excluded.embedding,
                    "updated_at": func.now(),
                },
            ).returning(SemanticMemory)
            return MemoryRecord.model_validate((await session.scalars(stmt)).one())

    async def has_memories(self, user_id: str, model: str) -> bool:
        relevant = select(SemanticMemory.id).where(
            SemanticMemory.user_id == user_id, SemanticMemory.embedding_model == model,
        )
        async with self._transaction() as session:
            return bool(await session.scalar(select(relevant.exists())))

    async def search_memories(
        self, *, user_id: str, model: str, embedding: list[float],
        limit: int, min_similarity: float,
    ) -> list[MemoryRecord]:
        distance = SemanticMemory.embedding.cosine_distance(embedding)
        stmt = select(SemanticMemory).where(
            SemanticMemory.user_id == user_id,
            SemanticMemory.embedding_model == model,
            distance <= 1 - min_similarity,
        ).order_by(distance, SemanticMemory.updated_at.desc(), SemanticMemory.id).limit(limit)
        async with self._transaction() as session:
            return [MemoryRecord.model_validate(row) for row in await session.scalars(stmt)]

    async def list_memories(self, user_id: str, *, limit: int = 50, offset: int = 0) -> list[MemoryRecord]:
        stmt = select(SemanticMemory).where(SemanticMemory.user_id == user_id).order_by(
            SemanticMemory.updated_at.desc(), SemanticMemory.id,
        ).limit(limit).offset(offset)
        async with self._transaction() as session:
            return [MemoryRecord.model_validate(row) for row in await session.scalars(stmt)]

    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool:
        stmt = delete(SemanticMemory).where(
            SemanticMemory.user_id == user_id, SemanticMemory.id == memory_id,
        ).returning(SemanticMemory.id)
        async with self._transaction() as session:
            return await session.scalar(stmt) is not None

    async def save_episode(
        self, *, user_id: str, thread_id: str, task: str, actions: list[str],
        result: str, success: bool,
    ) -> EpisodeRecord:
        async with self._transaction() as session:
            await ensure_thread(session, user_id, thread_id)
            row = Episode(
                user_id=user_id, thread_id=thread_id, task=task, actions=actions,
                result=result, success=success,
            )
            session.add(row)
            await session.flush()
            return EpisodeRecord.model_validate(row)

    async def list_episodes(self, user_id: str, *, limit: int = 50) -> list[EpisodeRecord]:
        stmt = select(Episode).where(Episode.user_id == user_id).order_by(
            Episode.created_at.desc(), Episode.id,
        ).limit(limit)
        async with self._transaction() as session:
            return [EpisodeRecord.model_validate(row) for row in await session.scalars(stmt)]


class AgentRunRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(
        self, user_id: str, result: AgentRunResponse, *, finished_at: datetime,
        events: Sequence[ObservedEvent] = (), approval: tuple[UUID, bool] | None = None,
    ) -> None:
        try:
            async with self._sessions.begin() as session:
                await ensure_thread(session, user_id, result.thread_id)
                statement = insert(AgentRun).values(
                    id=result.run_id, user_id=user_id, thread_id=result.thread_id,
                    status=result.status, state=result.model_dump(mode="json"),
                    created_at=finished_at,
                )
                written = await session.scalar(statement.on_conflict_do_update(
                    index_elements=[AgentRun.id],
                    set_={
                        "status": statement.excluded.status,
                        "state": statement.excluded.state,
                        "created_at": statement.excluded.created_at,
                    },
                    where=(
                        (AgentRun.user_id == user_id)
                        & (AgentRun.thread_id == result.thread_id)
                        & (AgentRun.created_at < finished_at)
                    ),
                ).returning(AgentRun.id))
                if written is not None:
                    await self._save_history(session, user_id, result, events, approval)
        except (SQLAlchemyError, OSError):
            raise StorageError("Could not save the agent run.") from None

    @staticmethod
    async def _save_history(
        session: AsyncSession, user_id: str, result: AgentRunResponse,
        events: Sequence[ObservedEvent], approval: tuple[UUID, bool] | None,
    ) -> None:
        run_id = result.run_id
        await session.execute(delete(Message).where(Message.run_id == run_id))
        await session.execute(delete(ToolCall).where(ToolCall.run_id == run_id))
        last_user = max(
            (index for index, message in enumerate(result.messages) if message.role == "user"),
            default=0,
        )
        session.add_all([
            Message(run_id=run_id, user_id=user_id, thread_id=result.thread_id,
                    position=position, role=message.role, content=message.content)
            for position, message in enumerate(result.messages[last_user:])
        ])
        observations = result.model_dump(mode="json")["observations"]
        session.add_all([
            ToolCall(run_id=run_id, tool=call["name"], arguments=call["arguments"],
                     observation=observations[index] if index < len(observations) else {})
            for index, call in enumerate(result.tool_calls)
        ])
        offset = await session.scalar(
            select(func.coalesce(func.max(AgentEvent.sequence), 0)).where(AgentEvent.run_id == run_id)
        )
        session.add_all([
            AgentEvent(run_id=run_id, sequence=offset + index, event=event.event,
                       created_at=event.timestamp, payload=event.model_dump(
                           mode="json", exclude={"metadata", "error", "sequence"},
                       ))
            for index, event in enumerate(events, start=1)
        ])
        identifiers = {"run_id": str(run_id), "thread_id": result.thread_id}
        if approval is not None:
            approval_id, approved = approval
            session.add(AuditLog(
                user_id=user_id, action="approval_granted" if approved else "approval_rejected",
                payload={**identifiers, "approval_id": str(approval_id)},
            ))
        session.add(AuditLog(
            user_id=user_id, action="approval_requested" if result.pending_approval else "run_finished",
            payload={**identifiers, "status": result.status},
        ))

    async def latest(self, user_id: str, thread_id: str) -> AgentRunResponse | None:
        stmt = select(AgentRun.state).where(
            AgentRun.user_id == user_id, AgentRun.thread_id == thread_id,
        ).order_by(AgentRun.created_at.desc(), AgentRun.id.desc()).limit(1)
        try:
            async with self._sessions() as session:
                state = await session.scalar(stmt)
        except (SQLAlchemyError, OSError):
            raise StorageError("Could not read the agent run.") from None
        return AgentRunResponse.model_validate(state) if state is not None else None


class EvaluationRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, result: EvaluationResult) -> None:
        try:
            async with self._sessions.begin() as session:
                session.add(Evaluation(
                    run_id=result.run_id, case_id=result.case_id,
                    metrics=result.model_dump(mode="json", exclude={"case_id", "run_id"}),
                ))
        except (SQLAlchemyError, OSError):
            raise StorageError("Could not save the evaluation.") from None

    async def list(self, user_id: str, *, limit: int, offset: int) -> list[StoredEvaluation]:
        stmt = (
            select(Evaluation).join(AgentRun, Evaluation.run_id == AgentRun.id)
            .where(AgentRun.user_id == user_id)
            .order_by(Evaluation.created_at.desc(), Evaluation.id.desc())
            .limit(limit).offset(offset)
        )
        try:
            async with self._sessions() as session:
                rows = await session.scalars(stmt)
                return [StoredEvaluation(
                    **row.metrics, id=row.id, run_id=row.run_id, case_id=row.case_id, created_at=row.created_at,
                ) for row in rows]
        except (SQLAlchemyError, OSError):
            raise StorageError("Could not read evaluations.") from None
