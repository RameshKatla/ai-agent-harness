import asyncio
import os
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import delete

from app.agent.executor import AgentExecutor
from app.core.config import Settings
from app.db.database import create_database_engine
from app.db.models import EMBEDDING_DIMENSIONS, User
from app.llm.ollama_client import OllamaLLM
from app.memory.manager import MemoryManager
from app.memory.semantic import OllamaEmbedder

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="requires migrated PostgreSQL"),
]


class TestEmbedder:
    __test__ = False
    model = "test-embedding-model"

    async def embed(self, text: str, *, query: bool) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[int("programming" in text.casefold())] = 1.0
        return vector


def test_postgres_persists_preferences_ranks_vectors_and_isolates_users():
    async def exercise():
        settings = Settings()
        engine = create_database_engine(settings)
        user_id, other_id = str(uuid4()), str(uuid4())
        try:
            def respond(request):
                body = request.content.decode()
                answer = "Your preferred language is Python." if "Saved memories" in body else "OK"
                return httpx.Response(200, json={
                    "message": {"role": "assistant", "content": answer}, "done": True,
                })

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
                llm = OllamaLLM(settings, http)
                manager = MemoryManager.using_database(settings, engine, TestEmbedder(), llm)
                executor = AgentExecutor(settings, llm, memory=manager)

                saved = await executor.run(
                    user_id=user_id, thread_id="same-thread",
                    task="My preferred programming language is Python.",
                )
                assert saved["status"] == "completed", saved["error"]
                unrelated = await manager.save_semantic_memory(user_id, "My favorite vegetable is kale.")
                other = await manager.save_semantic_memory(other_id, "Preferred programming language is Rust.")

                fresh = AgentExecutor(settings, llm, memory=manager)
                recalled = await fresh.run(
                    user_id=user_id, thread_id="new-thread",
                    task="What programming language do I prefer?",
                )
                assert recalled["retrieved_memories"] == ["Preferred programming language is Python."]
                assert recalled["final_answer"] == "Your preferred language is Python."
                assert (await manager.retrieve_relevant_memories(other_id, "programming")) == [other]
                assert await manager.delete_memory(other_id, unrelated.id) is False
                assert await manager.delete_memory(user_id, unrelated.id) is True
                assert await manager.delete_memory(user_id, unrelated.id) is False
                assert len(await manager.list_episodes(user_id)) == 2
                assert await manager.list_episodes(other_id) == []

                before, = await manager.list_memories(user_id)
                await manager.save_declared_preference(user_id, "My preferred programming language is Go.")
                after, = await manager.list_memories(user_id)
                assert after.id == before.id
                assert after.content == "Preferred programming language is Go."
        finally:
            async with engine.begin() as conn:
                await conn.execute(delete(User).where(User.id.in_([user_id, other_id])))
            await engine.dispose()

    asyncio.run(exercise())


@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="requires Ollama and embedding model")
def test_live_ollama_and_pgvector_recall_across_threads():
    async def exercise():
        settings = Settings()
        engine = create_database_engine(settings)
        user_id = str(uuid4())
        try:
            async with httpx.AsyncClient() as http:
                llm = OllamaLLM(settings, http)
                manager = MemoryManager.using_database(settings, engine, OllamaEmbedder(settings, http), llm)
                executor = AgentExecutor(settings, llm, memory=manager)
                first = await executor.run(
                    user_id=user_id, thread_id="record",
                    task="My preferred programming language is Python.",
                )
                assert first["status"] == "completed", first["error"]
                followup = await executor.run(
                    user_id=user_id, thread_id="recall",
                    task="What programming language do I prefer? Reply with the name.",
                )
                assert followup["status"] == "completed", followup["error"]
                assert followup["retrieved_memories"] == ["Preferred programming language is Python."]
                assert "python" in followup["final_answer"].casefold()
        finally:
            async with engine.begin() as conn:
                await conn.execute(delete(User).where(User.id == user_id))
            await engine.dispose()

    asyncio.run(exercise())
