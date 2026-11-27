import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from weakref import WeakValueDictionary

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings
from app.db.database import database_url
from app.llm.model import ChatMessage


def conversation_key(user_id: str, thread_id: str) -> str:
    return json.dumps([user_id, thread_id], separators=(",", ":"))


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(ChatMessage.__module__, ChatMessage.__name__)],
    )


class ShortTermMemory:
    def __init__(self) -> None:
        self.checkpointer = InMemorySaver(
            serde=checkpoint_serializer(),
        )
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())


class PostgresShortTermMemory:
    def __init__(self, checkpoints: AsyncConnectionPool, locks: AsyncConnectionPool) -> None:
        self.checkpointer = AsyncPostgresSaver(checkpoints, serde=checkpoint_serializer())
        self._locks = locks

    @classmethod
    @asynccontextmanager
    async def connect(cls, settings: Settings) -> AsyncIterator["PostgresShortTermMemory"]:
        url = database_url(settings).set(drivername="postgresql").render_as_string(hide_password=False)
        kwargs = {"autocommit": True, "row_factory": dict_row, "prepare_threshold": None}
        async with (
            AsyncConnectionPool(url, kwargs=kwargs, min_size=0, max_size=8, open=False) as checkpoints,
            AsyncConnectionPool(url, kwargs=kwargs, min_size=0, max_size=8, open=False) as locks,
        ):
            yield cls(checkpoints, locks)

    @asynccontextmanager
    async def lock(self, key: str) -> AsyncIterator[None]:
        lock_id = int.from_bytes(sha256(key.encode()).digest()[:8], "big", signed=True)
        async with self._locks.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))
            yield
