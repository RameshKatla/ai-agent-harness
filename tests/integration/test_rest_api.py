from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.agent.service import AgentService
from app.api.dependencies import BUILTIN_PERMISSIONS, get_agent_service, get_memory_manager
from app.core.config import Settings
from app.core.exceptions import MemoryError, StorageError
from app.main import create_app
from app.memory.manager import MemoryManager
from app.schemas.agent import AgentRunResponse
from app.schemas.memory import MemoryRecord

AUTH = {"Authorization": "Bearer dev-test-token"}


@pytest.fixture
def api():
    app = create_app(Settings(_env_file=None, api_token="dev-test-token", api_user_id="owner"))
    service = AsyncMock(spec=AgentService)
    memory = AsyncMock(spec=MemoryManager)
    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_memory_manager] = lambda: memory
    with TestClient(app, headers=AUTH) as client:
        yield client, service, memory


def example_run():
    return AgentRunResponse(
        run_id=uuid4(), thread_id="work", status="completed", final_answer="100", error=None,
        current_plan=[], current_step=1, max_steps=10, messages=[], tool_calls=[],
        observations=[], retrieved_memories=[], pending_approval=None,
    )


def test_agent_post_and_get(api):
    client, service, _ = api
    run = example_run()
    service.run.return_value = run
    service.get_thread.return_value = run
    response = client.post("/agent/run", json={"task": " 25 * 4 "})
    assert response.status_code == 200
    assert response.json() == run.model_dump(mode="json")
    UUID(response.json()["run_id"])
    service.run.assert_awaited_once_with(
        user_id="owner", thread_id=None, task="25 * 4", permissions=BUILTIN_PERMISSIONS,
    )
    assert client.get("/agent/work").json() == response.json()
    service.get_thread.assert_awaited_once_with(user_id="owner", thread_id="work")


def test_thread_not_found_uses_authenticated_user(api):
    client, service, _ = api
    service.get_thread.return_value = None
    assert client.get("/agent/work").status_code == 404
    service.get_thread.assert_awaited_once_with(user_id="owner", thread_id="work")


@pytest.mark.parametrize("body", [
    {"task": "  "}, {"task": "a", "thread_id": "a/b"},
    {"task": "a", "user_id": "another-user"},
    {"task": "a", "permissions": ["admin:use"]},
])
def test_invalid_agent_run_input_is_rejected(api, body):
    client, service, _ = api
    assert client.post("/agent/run", json=body).status_code == 422
    service.run.assert_not_called()


def test_memory_endpoints(api):
    client, _, memory = api
    timestamp = datetime.now(UTC)
    record = MemoryRecord(
        id=uuid4(), user_id="owner", content="Prefers Python.",
        created_at=timestamp, updated_at=timestamp,
    )
    memory.save_semantic_memory.return_value = record
    memory.list_memories.return_value = [record]
    memory.delete_memory.side_effect = [True, False]

    response = client.post("/memory", json={"content": " Prefers Python. "})
    assert response.status_code == 201
    assert response.json() == record.model_dump(mode="json")
    memory.save_semantic_memory.assert_awaited_once_with("owner", "Prefers Python.")

    listing = client.get("/memory?limit=3&offset=5")
    assert listing.json() == [record.model_dump(mode="json")]
    memory.list_memories.assert_awaited_once_with("owner", limit=3, offset=5)

    deleted = client.delete(f"/memory/{record.id}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    memory.delete_memory.assert_awaited_once_with("owner", record.id)
    assert client.delete(f"/memory/{record.id}").status_code == 404


@pytest.mark.parametrize(("method", "path", "kwargs"), [
    ("GET", "/memory?limit=101", {}),
    ("DELETE", "/memory/invalid-uuid", {}),
    ("POST", "/memory", {"json": {"content": " ", "user_id": "other"}}),
])
def test_invalid_memory_input_is_rejected(api, method, path, kwargs):
    client, _, memory = api
    assert client.request(method, path, **kwargs).status_code == 422
    assert not memory.mock_calls


@pytest.mark.parametrize("header", ["", "Bearer incorrect", "Basic ZGV2"])
def test_protected_routes_require_authentication(api, header):
    client, service, memory = api
    response = client.get("/memory", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert not service.mock_calls
    assert not memory.mock_calls


def test_unconfigured_token_keeps_protected_routes_closed():
    with TestClient(create_app(Settings(_env_file=None, api_token=None))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/memory").status_code == 503


def test_storage_failure_returns_service_unavailable(api):
    client, service, memory = api
    service.get_thread.side_effect = StorageError("Could not read the agent run.")
    memory.list_memories.side_effect = MemoryError("The memory database operation failed.")
    assert client.get("/agent/work").status_code == 503
    response = client.get("/memory")
    assert response.status_code == 503
    assert response.json() == {"detail": "The memory database operation failed."}
