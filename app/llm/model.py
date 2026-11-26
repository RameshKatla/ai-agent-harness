from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


TokenSink = Callable[[str], Awaitable[None]]


class LLMClient(Protocol):
    async def chat(self, messages: Sequence[ChatMessage]) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncGenerator[str, None]: ...

    async def structured_output[T: BaseModel](
        self, messages: Sequence[ChatMessage], schema: type[T], *, on_token: TokenSink | None = None,
    ) -> T: ...
