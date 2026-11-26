import hashlib
import json
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError
from redis.asyncio import Redis
from redis.asyncio.client import PubSub
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.exceptions import RedisUnavailableError
from app.observability.events import AgentEvent, EventName

logger = logging.getLogger(__name__)

# Setting the expiration and incrementing are indivisible across API workers.
_CONSUME = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('PTTL', KEYS[1])
if ttl < 0 then
    redis.call('PEXPIRE', KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end
return {count, ttl}
"""


class CoordinationEvent(BaseModel):
    event_id: UUID
    run_id: UUID
    thread_id: str
    user_id: str
    timestamp: datetime
    event: EventName
    success: bool | None = None


class RedisServices:
    def __init__(self, client: Redis, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @classmethod
    def connect(cls, settings: Settings) -> "RedisServices":
        client = Redis.from_url(
            str(settings.redis_url), socket_connect_timeout=1, socket_timeout=1,
            retry=Retry(NoBackoff(), 0),
        )
        return cls(client, settings)

    async def close(self) -> None:
        await self._client.aclose()

    async def ready(self) -> bool:
        try:
            return bool(await self._client.ping())
        except RedisError:
            return False

    def _key(self, kind: str, *parts: str) -> str:
        digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()
        return f"{self._settings.redis_key_prefix}:{kind}:{digest}"

    async def retry_after(self, user_id: str) -> int | None:
        try:
            count, ttl = await self._client.eval(
                _CONSUME, 1, self._key("limit", user_id),
                self._settings.rate_limit_window_seconds * 1000,
            )
        except RedisError:
            raise RedisUnavailableError("Rate limiting is temporarily unavailable.") from None
        if count <= self._settings.rate_limit_requests:
            return None
        return max(1, math.ceil(ttl / 1000))

    async def get_cached[Value: BaseModel](
        self, namespace: str, key: str, schema: type[Value],
    ) -> Value | None:
        try:
            raw = await self._client.get(self._key(f"cache:{namespace}", key))
            return schema.model_validate_json(raw) if raw is not None else None
        except (RedisError, ValidationError):
            logger.warning("Redis cache read unavailable")
            return None

    async def set_cached(self, namespace: str, key: str, value: BaseModel, *, ttl: int) -> None:
        try:
            await self._client.set(self._key(f"cache:{namespace}", key), value.model_dump_json(), ex=ttl)
        except RedisError:
            logger.warning("Redis cache write unavailable")

    async def publish(self, event: AgentEvent) -> None:
        if event.event not in {"agent_start", "approval_required", "agent_error", "agent_complete"}:
            return
        notification = CoordinationEvent(
            event_id=uuid4(), run_id=event.run_id, thread_id=event.thread_id, user_id=event.user_id,
            timestamp=event.timestamp, event=event.event, success=event.success,
        )
        try:
            await self._client.publish(
                self._key("events", event.user_id, event.thread_id), notification.model_dump_json(),
            )
        except RedisError:
            logger.warning("Redis event publish unavailable")

    @asynccontextmanager
    async def subscribe(self, user_id: str, thread_id: str) -> AsyncIterator[AsyncIterator[CoordinationEvent]]:
        async with self._client.pubsub() as pubsub:
            channel = self._key("events", user_id, thread_id)
            await pubsub.subscribe(channel)
            # Wait for Redis to acknowledge the subscription before returning to callers.
            acknowledgment = await pubsub.get_message(timeout=1)
            if acknowledgment is None or acknowledgment["type"] != "subscribe":
                raise RedisUnavailableError("Could not subscribe to run notifications.")
            async with aclosing(self._events(pubsub)) as events:
                yield events

    @staticmethod
    async def _events(pubsub: PubSub) -> AsyncGenerator[CoordinationEvent, None]:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
            if message is not None:
                try:
                    event = CoordinationEvent.model_validate_json(message["data"])
                except ValidationError:
                    logger.warning("Invalid Redis coordination event")
                else:
                    yield event
