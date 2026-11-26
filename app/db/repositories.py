from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import MemoryError, StorageError
from app.db.models import AgentRun, Episode, SemanticMemory, Thread, User
from app.schemas.agent import AgentRunResponse
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
    ) -> None:
        try:
            async with self._sessions.begin() as session:
                await ensure_thread(session, user_id, result.thread_id)
                statement = insert(AgentRun).values(
                    id=result.run_id, user_id=user_id, thread_id=result.thread_id,
                    status=result.status, state=result.model_dump(mode="json"),
                    created_at=finished_at,
                )
                await session.execute(statement.on_conflict_do_update(
                    index_elements=[AgentRun.id],
                    set_={
                        "status": statement.excluded.status,
                        "state": statement.excluded.state,
                        "created_at": statement.excluded.created_at,
                    },
                    where=(
                        (AgentRun.user_id == user_id)
                        & (AgentRun.thread_id == result.thread_id)
                        & (AgentRun.created_at <= finished_at)
                    ),
                ))
        except (SQLAlchemyError, OSError):
            raise StorageError("Could not save the agent run.") from None

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
