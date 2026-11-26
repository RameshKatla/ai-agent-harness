import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import MemoryError
from app.db.models import EMBEDDING_DIMENSIONS
from app.db.repositories import MemoryRepository
from app.llm.model import ChatMessage, LLMClient
from app.memory.manager import MemoryManager
from app.memory.semantic import Embedder, OllamaEmbedder


def managers():
    settings = Settings(_env_file=None, memory_retrieval_limit=3, memory_min_similarity=0.7)
    repository = AsyncMock(spec=MemoryRepository)
    embedder = AsyncMock(spec=Embedder)
    embedder.model = "test-model"
    embedder.embed.return_value = [1.0, 0.0]
    llm = AsyncMock(spec=LLMClient)
    return MemoryManager(settings, repository, embedder, llm), repository, embedder, llm


def test_retrieve_only_relevant_model_and_user_memories():
    memory, repository, embedder, _ = managers()
    repository.has_memories.return_value = True
    repository.search_memories.return_value = []
    assert asyncio.run(memory.retrieve_relevant_memories("user", "query")) == []
    repository.has_memories.assert_awaited_once_with("user", "test-model")
    embedder.embed.assert_awaited_once_with("query", query=True)
    repository.search_memories.assert_awaited_once_with(
        user_id="user", model="test-model", embedding=[1.0, 0.0], limit=3, min_similarity=0.7,
    )


def test_no_embeddings_requested_when_user_has_no_memories():
    memory, repository, embedder, _ = managers()
    repository.has_memories.return_value = False
    assert asyncio.run(memory.retrieve_relevant_memories("user", "query")) == []
    embedder.embed.assert_not_awaited()


def test_only_explicit_declarations_produce_automatic_memory():
    memory, repository, _, _ = managers()
    asyncio.run(memory.save_declared_preference("user", "What is my preferred language?"))
    asyncio.run(memory.save_declared_preference(
        "user", "My preferred language is Python. Could you do something else?",
    ))
    asyncio.run(memory.save_declared_preference("user", "My preferred programming language is Python."))
    repository.save_memory.assert_awaited_once_with(
        user_id="user", content="Preferred programming language is Python.",
        key="preference:programming language", model="test-model", embedding=[1.0, 0.0],
    )


def test_blank_memory_rejected_before_embedding():
    memory, repository, embedder, _ = managers()
    with pytest.raises(MemoryError):
        asyncio.run(memory.save_semantic_memory("user", "  "))
    embedder.embed.assert_not_awaited()
    repository.save_memory.assert_not_awaited()


def test_summarization_passes_conversation_to_injected_model():
    memory, _, _, llm = managers()
    llm.chat.return_value = "The user chose Python."
    conversation = [ChatMessage(role="user", content="Let's use Python.")]
    assert asyncio.run(memory.summarize_conversation(conversation)) == "The user chose Python."
    assert llm.chat.await_args.args[0][-1].content == "user: Let's use Python."


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[0.0] * EMBEDDING_DIMENSIONS], [[float("inf")] * 768]])
def test_embedder_rejects_invalid_vectors(vectors):
    async def exercise():
        def respond(request):
            return httpx.Response(200, content=json.dumps({"embeddings": vectors}))

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            service = OllamaEmbedder(Settings(_env_file=None), client)
            with pytest.raises(MemoryError):
                await service.embed("query", query=True)

    asyncio.run(exercise())


def test_nomic_embedding_uses_distinct_task_prefixes():
    async def exercise():
        calls = []

        def respond(request):
            assert request.url.path == "/api/embed"
            body = json.loads(request.content)
            calls.append(body["input"])
            assert body["dimensions"] == EMBEDDING_DIMENSIONS
            return httpx.Response(200, json={"embeddings": [[1] + [0] * (EMBEDDING_DIMENSIONS - 1)]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            service = OllamaEmbedder(Settings(_env_file=None, ollama_embedding_model="nomic-embed-text"), client)
            assert (await service.embed("question", query=True))[0] == 1
            assert (await service.embed("fact", query=False))[0] == 1
        assert calls == ["search_query: question", "search_document: fact"]

    asyncio.run(exercise())
