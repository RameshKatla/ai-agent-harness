import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("MAX_AGENT_STEPS=8\nPOSTGRES_PASSWORD=local-secret\n")
    monkeypatch.setenv("MAX_AGENT_STEPS", "3")
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    settings = Settings(_env_file=dotenv)

    assert settings.max_agent_steps == 3
    assert settings.postgres_password.get_secret_value() == "local-secret"
    assert "local-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_agent_steps", 0),
        ("max_tool_calls", -1),
        ("max_retries", -1),
        ("agent_timeout_seconds", float("inf")),
        ("postgres_port", 65536),
    ],
)
def test_invalid_settings(name, value):
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, **{name: value})

    assert (name,) in {item["loc"] for item in error.value.errors()}


def test_secret_bearing_ollama_url_is_rejected_without_echoing_credentials():
    with pytest.raises(ValidationError) as failure:
        Settings(_env_file=None, ollama_base_url="http://user:private-password@ollama.test/")
    assert "private-password" not in str(failure.value)
    assert "cannot include credentials" in str(failure.value)
