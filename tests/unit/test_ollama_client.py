import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.llm.model import ChatMessage
from app.llm.ollama_client import OllamaLLM


class Answer(BaseModel):
    value: int


MESSAGES = [ChatMessage(role="user", content="What is 2 + 2?")]


def frame(content, *, done=True):
    return {"message": {"role": "assistant", "content": content}, "done": done}


def run(handler, operation):
    async def exercise():
        settings = Settings(_env_file=None, ollama_model="test-model",
                            ollama_base_url="http://example.test/ollama")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await operation(OllamaLLM(settings, http))

    return asyncio.run(exercise())


def test_chat_uses_configured_host_and_model():
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "http://example.test/ollama/api/chat"
        assert json.loads(request.content) == {
            "model": "test-model", "messages": [MESSAGES[0].model_dump()], "stream": False,
        }
        return httpx.Response(200, json=frame("4"))

    assert run(handler, lambda client: client.chat(MESSAGES)) == "4"


def test_structured_output_sends_schema_and_validates_response():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["format"] == Answer.model_json_schema()
        assert payload["options"] == {"temperature": 0}
        return httpx.Response(200, json=frame('{"value": 4}'))

    assert run(handler, lambda client: client.structured_output(MESSAGES, Answer)) == Answer(value=4)


def test_structured_output_rejects_invalid_field_type():
    def handler(request):
        return httpx.Response(200, json=frame('{"value": "4"}'))

    with pytest.raises(LLMError, match="schema"):
        run(handler, lambda client: client.structured_output(MESSAGES, Answer))


async def collect_stream(client):
    return [chunk async for chunk in client.stream(MESSAGES)]


def test_stream_yields_content_in_order_including_final_frame():
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        frames = [frame("hello", done=False), frame(" world")]
        return httpx.Response(200, text="\n\n".join(map(json.dumps, frames)))

    assert run(handler, collect_stream) == ["hello", " world"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503), "stream failed"),
        (httpx.Response(200, text="invalid json"), "invalid chat response"),
        (httpx.Response(200, json={"error": "private upstream detail"}), "generation error"),
        (httpx.Response(200, json=frame("partial", done=False)), "ended before completion"),
    ],
)
def test_stream_failures(response, message):
    with pytest.raises(LLMError, match=message):
        run(lambda request: response, collect_stream)


def test_chat_handles_network_timeout():
    def handler(request):
        raise httpx.ReadTimeout("private upstream detail")

    with pytest.raises(LLMError, match=r"^Ollama chat request failed\.$"):
        run(handler, lambda client: client.chat(MESSAGES))
