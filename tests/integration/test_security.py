import asyncio
import io
import json
import logging
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
from fastapi.testclient import TestClient

from app.agent.service import AgentService
from app.api.dependencies import get_agent_service, get_authenticator, get_memory_manager
from app.core.config import Settings
from app.core.exceptions import StorageError
from app.core.security import Authenticator, Principal
from app.main import create_app
from app.memory.manager import MemoryManager
from app.observability.logging import JsonFormatter

AUTH = {"Authorization": "Bearer test-private-token"}


def app_with(**settings):
    return create_app(Settings(
        _env_file=None, api_token="test-private-token", redis_enabled=False, **settings,
    ))


def test_requests_receive_unique_server_generated_ids():
    with TestClient(app_with()) as client:
        first = client.get("/health", headers={"X-Request-ID": "untrusted-id"})
        denied = client.get("/memory")
        missing = client.get("/unknown")
    ids = {response.headers["X-Request-ID"] for response in (first, denied, missing)}
    assert len(ids) == 3
    assert all(UUID(item).version == 4 for item in ids)
    assert denied.json()["request_id"] == denied.headers["X-Request-ID"]
    assert denied.headers["WWW-Authenticate"] == "Bearer"
    assert missing.json()["request_id"] == missing.headers["X-Request-ID"]


def test_body_at_limit_is_accepted_but_larger_body_is_rejected_before_handlers():
    body = b'{"content":"valid"}'
    app = app_with(max_request_body_bytes=len(body))
    service = AsyncMock(spec=MemoryManager)
    service.save_semantic_memory.return_value = {
        "id": "497f16b7-bdae-48a0-be36-7bc2d12b6a33",
        "user_id": "local",
        "content": "valid",
        "created_at": "2026-11-26T00:00:00Z",
        "updated_at": "2026-11-26T00:00:00Z",
    }
    app.dependency_overrides[get_memory_manager] = lambda: service
    headers = {**AUTH, "Content-Type": "application/json"}
    with TestClient(app) as client:
        assert client.post("/memory", content=body, headers=headers).status_code == 201
        response = client.post("/memory", content=body + b" ", headers=headers)
        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
    service.save_semantic_memory.assert_awaited_once_with("local", "valid")


def test_chunked_request_body_is_limited_before_the_stream_route_runs():
    async def exercise():
        app = app_with(max_request_body_bytes=32)
        agent = AsyncMock(spec=AgentService)
        app.dependency_overrides[get_agent_service] = lambda: agent

        async def chunks():
            yield b'{"task":"'
            yield b"x" * 20
            yield b"x" * 10
            raise AssertionError("The body reader should stop at the size limit.")

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agent/stream", headers={**AUTH, "Content-Type": "application/json"}, content=chunks(),
            )
        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"
        agent.stream.assert_not_called()

    asyncio.run(exercise())


def test_public_errors_and_request_logs_omit_untrusted_values():
    app = app_with()
    storage = AsyncMock(spec=AgentService)
    storage.get_thread.side_effect = StorageError("postgresql://private-password")
    app.dependency_overrides[get_agent_service] = lambda: storage
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())

    with TestClient(app) as client:
        root = logging.getLogger("app")
        root.addHandler(handler)
        try:
            validation = client.post(
                "/agent/run?raw-query-private", headers=AUTH,
                json={"task": "valid", "secret-field": "secret-value"},
            )
            storage_failure = client.get("/agent/secret-thread", headers=AUTH)
            storage.get_thread.side_effect = RuntimeError("unexpected-private-value")
            unexpected = client.get("/agent/secret-thread", headers=AUTH)
        finally:
            root.removeHandler(handler)

    assert validation.status_code == 422
    assert validation.json()["issues"] == [{"source": "body", "code": "extra_forbidden"}]
    assert storage_failure.status_code == 503
    assert storage_failure.json()["code"] == "storage_unavailable"
    assert unexpected.status_code == 500
    assert unexpected.json()["code"] == "internal_error"

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    access = [record for record in records if "http_request" in record]
    assert [record["http_request"]["route"] for record in access] == [
        "/agent/run", "/agent/{thread_id}", "/agent/{thread_id}",
    ]
    assert access[-1]["request_id"] == unexpected.headers["X-Request-ID"]
    published = "".join([validation.text, storage_failure.text, unexpected.text, output.getvalue()])
    for value in (
        "test-private-token", "private-password", "raw-query-private", "secret-field",
        "secret-value", "secret-thread", "unexpected-private-value",
    ):
        assert value not in published


def test_configured_tool_allowlist_restricts_the_principal():
    app = app_with(api_allowed_tools={"calculator"})
    agent = AsyncMock(spec=AgentService)
    agent.run.side_effect = StorageError()
    app.dependency_overrides[get_agent_service] = lambda: agent
    with TestClient(app) as client:
        assert client.post("/agent/run", headers=AUTH, json={"task": "task"}).status_code == 503
    assert agent.run.await_args.kwargs["permissions"] == frozenset({"calculator:use"})


def test_authentication_backend_can_supply_a_different_principal():
    app = app_with()
    backend = AsyncMock(spec=Authenticator)
    backend.configured = True
    backend.authenticate.return_value = Principal("custom-user", frozenset())
    agent = AsyncMock(spec=AgentService)
    agent.get_thread.return_value = None
    app.dependency_overrides[get_authenticator] = lambda: backend
    app.dependency_overrides[get_agent_service] = lambda: agent
    with TestClient(app) as client:
        assert client.get("/agent/thread", headers=AUTH).status_code == 404
    backend.authenticate.assert_awaited_once_with("test-private-token")
    agent.get_thread.assert_awaited_once_with(user_id="custom-user", thread_id="thread")
