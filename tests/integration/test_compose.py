import os
from uuid import uuid4

import httpx
import pytest

from app.core.config import Settings

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(os.getenv("RUN_COMPOSE_TESTS") != "1", reason="requires a running Compose stack"),
]


def test_compose_ready_and_memory_round_trip():
    settings = Settings()
    assert settings.api_token is not None
    base_url = os.getenv("COMPOSE_BASE_URL", "http://127.0.0.1:8000")
    headers = {"Authorization": "Bearer " + settings.api_token.get_secret_value()}
    with httpx.Client(base_url=base_url, timeout=90) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/memory").status_code == 401

        content = f"Compose smoke test {uuid4()}"
        response = client.post("/memory", headers=headers, json={"content": content})
        assert response.status_code == 201, response.text
        memory = response.json()
        try:
            assert memory["content"] == content
            listed = client.get("/memory", headers=headers).raise_for_status().json()
            assert any(record["id"] == memory["id"] for record in listed)
        finally:
            assert client.delete(f'/memory/{memory["id"]}', headers=headers).status_code == 204
