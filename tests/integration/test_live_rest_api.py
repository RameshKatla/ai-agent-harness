import asyncio
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.core.config import Settings
from app.db.database import create_database_engine
from app.db.models import User
from app.main import create_app

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_TESTS") != "1" or os.getenv("RUN_OLLAMA_TESTS") != "1",
        reason="requires migrated PostgreSQL and Ollama",
    ),
]


def test_live_rest_agent_and_memory():
    user_id, other_id = str(uuid4()), str(uuid4())
    settings = Settings(
        api_token="live-test-token", api_user_id=user_id,
        redis_enabled=os.getenv("RUN_REDIS_TESTS") == "1",
    )
    headers = {"Authorization": "Bearer live-test-token"}
    try:
        with TestClient(create_app(settings), headers=headers) as client:
            response = client.post("/agent/run", json={"task": "What is 25 * 4?"})
            assert response.status_code == 200, response.text
            run = response.json()
            assert run["status"] == "completed", run
            assert run["observations"] and run["observations"][0]["tool"] == "calculator", run
            assert run["observations"][0]["result"] == 100
            assert "100" in run["final_answer"]

            created = client.post("/memory", json={"content": "Preferred language is Python."})
            assert created.status_code == 201, created.text
            memory = created.json()
            assert [entry["id"] for entry in client.get("/memory").json()] == [memory["id"]]

        other = settings.model_copy(update={"api_user_id": other_id})
        with TestClient(create_app(other), headers=headers) as client:
            assert client.get(f'/agent/{run["thread_id"]}').status_code == 404
            assert client.get("/memory").json() == []
            assert client.delete(f'/memory/{memory["id"]}').status_code == 404

        with TestClient(create_app(settings), headers=headers) as client:
            assert client.get(f'/agent/{run["thread_id"]}').json() == run
            assert client.delete(f'/memory/{memory["id"]}').status_code == 204
            assert client.get("/memory").json() == []
    finally:
        async def cleanup():
            engine = create_database_engine(settings)
            try:
                async with engine.begin() as conn:
                    await conn.execute(delete(User).where(User.id.in_([user_id, other_id])))
            finally:
                await engine.dispose()

        asyncio.run(cleanup())
