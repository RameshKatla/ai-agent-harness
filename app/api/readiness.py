import asyncio
from collections.abc import Awaitable, Callable

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings


class Readiness:
    def __init__(self, *checks: Callable[[], Awaitable[bool]]) -> None:
        self._checks = checks

    async def ready(self) -> bool:
        async def check(probe: Callable[[], Awaitable[bool]]) -> bool:
            try:
                async with asyncio.timeout(3):
                    return await probe()
            except Exception:
                return False

        return all(await asyncio.gather(*(check(probe) for probe in self._checks)))


async def postgres_ready(engine: AsyncEngine) -> bool:
    async with engine.connect() as connection:
        return await connection.scalar(text("SELECT 1")) == 1


async def ollama_ready(settings: Settings, http: httpx.AsyncClient) -> bool:
    response = await http.get(str(settings.ollama_base_url).rstrip("/") + "/api/tags")
    response.raise_for_status()
    available = {item["name"] for item in response.json()["models"]}

    def normalize(name: str) -> str:
        return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"

    return {normalize(settings.ollama_model), normalize(settings.ollama_embedding_model)} <= available
