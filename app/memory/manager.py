import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Unpack
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import LLMError, MemoryError
from app.db.repositories import MemoryRepository
from app.llm.model import ChatMessage, LLMClient
from app.memory.episodic import EpisodeInput
from app.memory.semantic import Embedder
from app.memory.summarizer import summarize
from app.schemas.memory import EpisodeRecord, MemoryCreate, MemoryRecord

PREFERENCE = re.compile(r"my preferred ([^\r\n.!?]{1,120}?) is ([^\r\n!?]{1,240})", re.IGNORECASE)


class MemoryManager:
    def __init__(
        self, settings: Settings, repository: MemoryRepository, embedder: Embedder, llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedder = embedder
        self._llm = llm

    @classmethod
    def using_database(
        cls, settings: Settings, engine: AsyncEngine, embedder: Embedder, llm: LLMClient,
    ) -> "MemoryManager":
        return cls(settings, MemoryRepository(async_sessionmaker(engine)), embedder, llm)

    async def retrieve_relevant_memories(self, user_id: str, query: str) -> list[MemoryRecord]:
        model = self._embedder.model
        if not await self._repository.has_memories(user_id, model):
            return []
        embedding = await self._embedder.embed(query, query=True)
        return await self._repository.search_memories(
            user_id=user_id, model=model, embedding=embedding,
            limit=self._settings.memory_retrieval_limit,
            min_similarity=self._settings.memory_min_similarity,
        )

    async def save_semantic_memory(
        self, user_id: str, content: str, *, key: str | None = None,
    ) -> MemoryRecord:
        try:
            content = MemoryCreate(content=content).content
        except ValidationError:
            raise MemoryError("Semantic memory must contain 1 to 4000 characters.") from None
        if key is None:
            key = f"content:{sha256(content.encode()).hexdigest()}"
        embedding = await self._embedder.embed(content, query=False)
        return await self._repository.save_memory(
            user_id=user_id, content=content, key=key, model=self._embedder.model, embedding=embedding,
        )

    async def save_declared_preference(self, user_id: str, task: str) -> None:
        match = PREFERENCE.fullmatch(task.strip().removesuffix("."))
        if match:
            subject, value = (part.strip() for part in match.groups())
            if subject and value and not re.search(r"\.\s", value):
                key = "preference:" + " ".join(subject.casefold().split())
                await self.save_semantic_memory(user_id, f"Preferred {subject} is {value}.", key=key)

    async def save_episode(self, **episode: Unpack[EpisodeInput]) -> EpisodeRecord:
        return await self._repository.save_episode(**episode)

    async def list_episodes(self, user_id: str, *, limit: int = 50) -> list[EpisodeRecord]:
        return await self._repository.list_episodes(user_id, limit=limit)

    async def summarize_conversation(self, conversation: Sequence[ChatMessage]) -> str:
        try:
            return await summarize(self._llm, conversation)
        except LLMError:
            raise MemoryError("Could not summarize the conversation.") from None

    async def delete_memory(self, user_id: str, memory_id: UUID) -> bool:
        return await self._repository.delete_memory(user_id, memory_id)

    async def list_memories(self, user_id: str, *, limit: int = 50, offset: int = 0) -> list[MemoryRecord]:
        return await self._repository.list_memories(user_id, limit=limit, offset=offset)
