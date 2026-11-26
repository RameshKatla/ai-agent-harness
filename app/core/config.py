from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    ollama_model: Annotated[str, Field(min_length=1)] = "gemma4:4b"
    ollama_embedding_model: Annotated[str, Field(min_length=1)] = "nomic-embed-text"
    postgres_host: str = "localhost"
    postgres_port: Annotated[int, Field(ge=1, le=65535)] = 5432
    postgres_db: str = "agentdb"
    postgres_user: str = "agent"
    postgres_password: SecretStr | None = None
    redis_url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"), repr=False
    )

    max_agent_steps: Annotated[int, Field(gt=0)] = 10
    max_tool_calls: Annotated[int, Field(ge=0)] = 5
    approval_required_tools: frozenset[str] = frozenset()
    max_retries: Annotated[int, Field(ge=0)] = 2
    agent_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 120
    memory_retrieval_limit: Annotated[int, Field(ge=1, le=20)] = 5
    memory_min_similarity: Annotated[float, Field(ge=0, le=1)] = 0.5
    max_conversation_turns: Annotated[int, Field(ge=1, le=100)] = 20
    api_token: SecretStr | None = None
    api_user_id: Annotated[str, Field(min_length=1, max_length=255)] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
