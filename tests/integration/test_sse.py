import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import ClientDisconnect

from app.agent.executor import AgentExecutor
from app.agent.service import AgentService
from app.api.dependencies import get_agent_service
from app.api.sse import EventStreamResponse
from app.core.config import Settings
from app.core.exceptions import StorageError
from app.db.repositories import AgentRunRepository
from app.llm.ollama_client import OllamaLLM
from app.main import create_app
from app.tools import build_tool_registry

AUTH = {"Authorization": "Bearer streaming-test"}


def parse_events(lines):
    event = {}
    for line in lines:
        if not line:
            if event:
                payload = json.loads(event["data"])
                assert event["event"] == payload["event"]
                assert event["id"] == f'{payload["run_id"]}:{payload["sequence"]}'
                yield payload
                event = {}
        elif not line.startswith(":"):
            key, value = line.split(": ", 1)
            event[key] = value


def frame(text, *, done=False):
    return json.dumps({
        "message": {"role": "assistant", "content": text},
        "done": done,
    }) + "\n"


class Frames(httpx.AsyncByteStream):
    def __init__(self, chunks, *, on_second=None):
        self._chunks = chunks
        self._on_second = on_second
        self.closed = False

    async def __aiter__(self):
        for index, chunk in enumerate(self._chunks):
            if index == 1 and self._on_second is not None:
                await self._on_second()
            yield frame(chunk, done=index == len(self._chunks) - 1).encode()

    async def aclose(self):
        self.closed = True


def test_http_stream_returns_ordered_trajectory_and_persists_the_run():
    async def exercise():
        def respond(request):
            body = json.loads(request.content)
            assert body["stream"] is True
            assert "format" in body
            if body["messages"][-1]["role"] == "tool":
                output = '{"decision":{"action":"final","answer":"100"}}'
            else:
                output = '{"decision":{"action":"tool","name":"calculator","arguments":{"expression":"25 * 4"}}}'
            return httpx.Response(200, stream=Frames([output[:17], output[17:]]))

        settings = Settings(_env_file=None, redis_enabled=False, api_token="streaming-test", api_user_id="owner")
        repo = AsyncMock(spec=AgentRunRepository)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as model_http:
            executor = AgentExecutor(
                settings, OllamaLLM(settings, model_http), build_tool_registry(model_http),
            )
            service = AgentService(executor, repo)
            app = create_app(settings)
            app.dependency_overrides[get_agent_service] = lambda: service
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                assert (await client.post("/agent/stream", json={"task": "task"})).status_code == 401
                assert (await client.post("/agent/stream", headers=AUTH, json={})).status_code == 422
                response = await client.post(
                    "/agent/stream", headers=AUTH, json={"task": "What is 25 * 4?", "thread_id": "math"},
                )
                schema = (await client.get("/openapi.json")).json()
                operation = schema["paths"]["/agent/stream"]["post"]
                assert set(operation["responses"]["200"]["content"]) == {"text/event-stream"}

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-request-id"]
        events = list(parse_events(response.text.splitlines()))
        assert [item["event"] for item in events] == [
            "agent_start", "memory_retrieved", "agent_step", "llm_start", "llm_token",
            "llm_token", "llm_end", "tool_start", "tool_result", "agent_step",
            "llm_start", "llm_token", "llm_token", "llm_end", "agent_complete",
        ]
        assert [item["sequence"] for item in events] == list(range(1, len(events) + 1))
        assert {item["run_id"] for item in events} == {events[0]["run_id"]}
        assert {item["thread_id"] for item in events} == {"math"}
        assert {item["user_id"] for item in events} == {"owner"}
        assert events[8]["metadata"]["observation"]["result"] == 100
        assert events[-1]["metadata"]["run"]["final_answer"] == "100"
        assert repo.save.await_args.args[1].model_dump(mode="json") == events[-1]["metadata"]["run"]

    asyncio.run(exercise())


def test_events_reach_client_before_model_finishes():
    async def exercise():
        delivered = asyncio.Event()

        async def wait_for_first_delivery():
            await asyncio.wait_for(delivered.wait(), timeout=2)

        stream = Frames(["hel", "lo"], on_second=wait_for_first_delivery)
        settings = Settings(_env_file=None)
        repo = AsyncMock(spec=AgentRunRepository)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        async with httpx.AsyncClient(transport=transport) as http:
            service = AgentService(AgentExecutor(settings, OllamaLLM(settings, http)), repo)
            async for event in service.stream(
                user_id="owner", thread_id=None, task="Say hello.", permissions=(),
            ):
                if event.event == "llm_token" and event.metadata["content"] == "hel":
                    assert not repo.save.called
                    delivered.set()
                if event.event == "agent_complete":
                    assert event.metadata["run"]["final_answer"] == "hello"
        assert delivered.is_set()
        assert stream.closed

    asyncio.run(exercise())


def test_disconnect_closes_model_stream_and_cancels_unfinished_run():
    async def exercise():
        never = asyncio.Event()
        stream = Frames(["one", "two"], on_second=never.wait)
        settings = Settings(_env_file=None)
        repo = AsyncMock(spec=AgentRunRepository)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        async with httpx.AsyncClient(transport=transport) as http:
            service = AgentService(AgentExecutor(settings, OllamaLLM(settings, http)), repo)
            response = EventStreamResponse(service.stream(
                user_id="owner", thread_id=None, task="task", permissions=(),
            ))

            async def send(message):
                if b"event: llm_token" in message.get("body", b""):
                    raise OSError("client disconnected")

            async def receive():
                pytest.fail("ASGI 2.4 detects disconnects on send")

            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(
                    response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send),
                    timeout=2,
                )
            assert stream.closed
            repo.save.assert_not_called()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["model", "storage"])
def test_stream_reports_terminal_failure(failure):
    async def exercise():
        settings = Settings(_env_file=None)
        repo = AsyncMock(spec=AgentRunRepository)
        if failure == "storage":
            repo.save.side_effect = StorageError("private upstream detail")
            response = httpx.Response(200, stream=Frames(["answer"]))
        else:
            response = httpx.Response(200, text='{"error":"private upstream detail"}\n')
        transport = httpx.MockTransport(lambda _: response)
        async with httpx.AsyncClient(transport=transport) as http:
            service = AgentService(AgentExecutor(settings, OllamaLLM(settings, http)), repo)
            events = [event async for event in service.stream(
                user_id="owner", thread_id="thread", task="task", permissions=(),
            )]
        assert [event.event for event in events[-2:]] == ["agent_error", "agent_complete"]
        assert events[-1].success is False
        assert events[-1].metadata["status"] == "failed"
        assert "private upstream detail" not in events[-2].error
        assert ("run" in events[-1].metadata) == (failure == "model")

    asyncio.run(exercise())


def test_stream_timeout_keeps_the_last_started_step_in_terminal_events():
    async def exercise():
        never = asyncio.Event()
        stream = Frames(["partial", "unused"], on_second=never.wait)
        settings = Settings(_env_file=None, agent_timeout_seconds=0.15)
        repo = AsyncMock(spec=AgentRunRepository)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        async with httpx.AsyncClient(transport=transport) as http:
            service = AgentService(AgentExecutor(settings, OllamaLLM(settings, http)), repo)
            events = [event async for event in service.stream(
                user_id="owner", thread_id=None, task="task", permissions=(),
            )]
        assert stream.closed
        assert [event.event for event in events[-2:]] == ["agent_error", "agent_complete"]
        assert events[-2].error == "Agent run timed out."
        assert events[-2].step == events[-1].step == 1
        assert events[-1].metadata["run"]["status"] == "failed"
        repo.save.assert_awaited_once()

    asyncio.run(exercise())
