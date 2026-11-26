import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.mark.parametrize(("path", "status"), [("/health", "ok"), ("/ready", "ready")])
def test_health_routes(path, status):
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json() == {"status": status}
