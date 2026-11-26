import asyncio
import json
from weakref import WeakValueDictionary

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.llm.model import ChatMessage


def conversation_key(user_id: str, thread_id: str) -> str:
    return json.dumps([user_id, thread_id], separators=(",", ":"))


class ShortTermMemory:
    def __init__(self) -> None:
        self.checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[(ChatMessage.__module__, ChatMessage.__name__)],
            ),
        )
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())
