import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.readiness import ollama_ready
from app.core.config import Settings
from app.main import create_app


@pytest.mark.parametrize(("postgres", "ollama", "status"), [
    (True, True, 200), (False, True, 503), (True, False, 503),
])
def test_dependency_readiness(monkeypatch, postgres, ollama, status):
    monkeypatch.setattr("app.main.postgres_ready", AsyncMock(return_value=postgres))
    monkeypatch.setattr("app.main.ollama_ready", AsyncMock(return_value=ollama))
    with TestClient(create_app(Settings(_env_file=None, redis_enabled=False))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == status


def test_ollama_readiness_requires_both_models():
    async def exercise():
        models = [{"name": "gemma4:e4b"}]
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": models}))
        settings = Settings(_env_file=None, ollama_model="gemma4:e4b", ollama_embedding_model="nomic-embed-text")
        async with httpx.AsyncClient(transport=transport) as http:
            assert not await ollama_ready(settings, http)
            models.append({"name": "nomic-embed-text:latest"})
            assert await ollama_ready(settings, http)

    asyncio.run(exercise())
