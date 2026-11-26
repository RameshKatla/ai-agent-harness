import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.llm.model import ChatMessage, TokenSink


class ResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatFrame(BaseModel):
    message: ResponseMessage
    done: bool


class OllamaLLM:
    def __init__(self, settings: Settings, http: httpx.AsyncClient) -> None:
        self._http = http
        self._url = str(settings.ollama_base_url).rstrip("/") + "/api/chat"
        self._model = settings.ollama_model
        self._timeout = httpx.Timeout(settings.agent_timeout_seconds, connect=5.0)

    def _payload(
        self, messages: Sequence[ChatMessage], *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "stream": stream,
        }

    @staticmethod
    def _parse_frame(data: str | bytes) -> ChatFrame:
        try:
            frame = json.loads(data)
            if isinstance(frame, dict) and frame.get("error"):
                raise LLMError("Ollama reported a generation error.")
            return ChatFrame.model_validate(frame)
        except (ValueError, UnicodeError):
            raise LLMError("Ollama returned an invalid chat response.") from None

    async def _complete(self, payload: dict[str, Any]) -> str:
        try:
            response = await self._http.post(self._url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPError:
            raise LLMError("Ollama chat request failed.") from None

        frame = self._parse_frame(response.content)
        if not frame.done:
            raise LLMError("Ollama returned an incomplete chat response.")
        return frame.message.content

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        return await self._complete(self._payload(messages, stream=False))

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncGenerator[str, None]:
        return self._stream(self._payload(messages, stream=True))

    async def _stream(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        try:
            async with self._http.stream(
                "POST", self._url, json=payload,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    frame = self._parse_frame(line)
                    if frame.message.content:
                        yield frame.message.content
                    if frame.done:
                        return
        except httpx.HTTPError:
            raise LLMError("Ollama chat stream failed.") from None

        raise LLMError("Ollama chat stream ended before completion.")

    async def structured_output[T: BaseModel](
        self, messages: Sequence[ChatMessage], schema: type[T], *, on_token: TokenSink | None = None,
    ) -> T:
        payload = self._payload(messages, stream=on_token is not None)
        payload["format"] = schema.model_json_schema()
        payload["options"] = {"temperature": 0}
        if on_token is None:
            content = await self._complete(payload)
        else:
            chunks: list[str] = []
            async with aclosing(self._stream(payload)) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
                    await on_token(chunk)
            content = "".join(chunks)
        try:
            return schema.model_validate_json(content, strict=True)
        except ValidationError:
            raise LLMError("Ollama output did not match the requested schema.") from None
