import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from app.api.dependencies import get_memory_manager
from app.core.config import Settings
from app.core.redis import RedisServices
from app.main import create_app
from app.memory.manager import MemoryManager
from app.observability.events import AgentEvent
from app.tools.weather import weather_tool

requires_redis = pytest.mark.skipif(os.getenv("RUN_REDIS_TESTS") != "1", reason="requires Redis")
AUTH = {"Authorization": "Bearer redis-test"}


def test_unavailable_redis_blocks_protected_requests_but_keeps_liveness(monkeypatch):
    redis = AsyncMock()
    redis.ping.side_effect = ConnectionError("private connection detail")
    redis.eval.side_effect = ConnectionError("private connection detail")
    monkeypatch.setattr(RedisServices, "connect", lambda settings: RedisServices(redis, settings))
    settings = Settings(_env_file=None, redis_enabled=True, api_token="redis-test")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/memory").status_code == 401
        response = client.get("/memory", headers=AUTH)
        assert response.status_code == 503
        assert response.json() == {
            "code": "rate_limit_unavailable",
            "detail": "Rate limiting is temporarily unavailable.",
            "request_id": response.headers["X-Request-ID"],
        }
    redis.aclose.assert_awaited_once()


@pytest.mark.external
@requires_redis
def test_rate_limit_is_shared_by_api_instances_and_scoped_by_user(monkeypatch):
    monkeypatch.setattr("app.main.postgres_ready", AsyncMock(return_value=True))
    monkeypatch.setattr("app.main.ollama_ready", AsyncMock(return_value=True))

    settings = Settings(
        redis_enabled=True, api_token="redis-test", api_user_id="first",
        redis_key_prefix=f"redis-test-{uuid4()}", rate_limit_requests=2, rate_limit_window_seconds=2,
    )

    def app_for(user_id):
        app = create_app(settings.model_copy(update={"api_user_id": user_id}))
        memory = AsyncMock(spec=MemoryManager)
        memory.list_memories.return_value = []
        app.dependency_overrides[get_memory_manager] = lambda: memory
        return app

    with (
        TestClient(app_for("first"), headers=AUTH) as first,
        TestClient(app_for("first"), headers=AUTH) as second,
        TestClient(app_for("second"), headers=AUTH) as other,
    ):
        assert first.get("/ready").status_code == 200
        assert first.get("/memory", headers={"Authorization": "Bearer invalid"}).status_code == 401
        assert first.get("/memory").status_code == 200
        assert second.get("/memory").status_code == 200
        rejected = first.get("/memory")
        assert rejected.status_code == 429
        assert int(rejected.headers["retry-after"]) in {1, 2}
        assert other.get("/memory").status_code == 200


@pytest.mark.external
@requires_redis
def test_rate_limit_counts_parallel_requests_atomically_and_expires():
    async def exercise():
        redis = RedisServices.connect(Settings(
            redis_key_prefix=f"redis-test-{uuid4()}",
            rate_limit_requests=5, rate_limit_window_seconds=1,
        ))
        try:
            decisions = await asyncio.gather(*(redis.retry_after("owner") for _ in range(12)))
            assert decisions.count(None) == 5
            assert decisions.count(1) == 7
            await asyncio.sleep(1.05)
            assert await redis.retry_after("owner") is None
        finally:
            await redis.close()

    asyncio.run(exercise())


def provider_calls(calls):
    def respond(request):
        calls.append(request.url.host)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [{
                "name": "New York", "latitude": 40.71, "longitude": -74.01,
            }]})
        return httpx.Response(200, json={
            "timezone": "America/New_York",
            "current": {"time": "2026-11-26T08:00", "temperature_2m": 8, "precipitation": 3, "weather_code": 61},
            "daily": {"time": ["2026-11-26"], "precipitation_probability_max": [90]},
        })
    return respond


@pytest.mark.external
@requires_redis
def test_weather_cache_is_shared_and_expires():
    async def exercise():
        settings = Settings(redis_key_prefix=f"redis-test-{uuid4()}")
        first, second = RedisServices.connect(settings), RedisServices.connect(settings)
        calls = []
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider_calls(calls))) as http:
                result = await weather_tool(http, cache=first, cache_ttl=1).invoke({"location": "New York"})
                assert await weather_tool(http, cache=second).invoke({"location": " new   York "}) == result
                assert len(calls) == 2
                await asyncio.sleep(1.05)
                assert await weather_tool(http, cache=second).invoke({"location": "New York"}) == result
                assert len(calls) == 4
        finally:
            await first.close()
            await second.close()

    asyncio.run(exercise())


def test_weather_falls_back_to_provider_when_redis_is_unavailable():
    async def exercise():
        client = AsyncMock()
        client.get.side_effect = ConnectionError("secret")
        client.set.side_effect = ConnectionError("secret")
        calls = []
        cache = RedisServices(client, Settings(_env_file=None))
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider_calls(calls))) as http:
            result = await weather_tool(http, cache=cache).invoke({"location": "New York"})
        assert result["location"] == "New York"
        assert len(calls) == 2

    asyncio.run(exercise())


@pytest.mark.external
@requires_redis
def test_pubsub_coordinates_run_lifecycle_without_broadcasting_private_payloads():
    async def exercise():
        settings = Settings(redis_key_prefix=f"redis-test-{uuid4()}")
        publisher, subscriber = RedisServices.connect(settings), RedisServices.connect(settings)
        start = AgentEvent(
            run_id=uuid4(), thread_id="thread", user_id="owner", sequence=1,
            timestamp=datetime.now(UTC), step=0, event="agent_start", node="executor",
            metadata={"request": "private content"},
        )
        try:
            async with (
                subscriber.subscribe("owner", "thread") as owner,
                subscriber.subscribe("different-owner", "thread") as other,
            ):
                await publisher.publish(start.model_copy(update={"event": "llm_token"}))
                await publisher.publish(start)
                event = await asyncio.wait_for(anext(owner), timeout=2)
                assert event.run_id == start.run_id
                assert event.event == "agent_start"
                assert "private content" not in event.model_dump_json()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(anext(other), timeout=0.1)
        finally:
            await publisher.close()
            await subscriber.close()

    asyncio.run(exercise())
