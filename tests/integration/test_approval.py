import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from app.agent.executor import AgentExecutor
from app.agent.service import AgentService
from app.api.dependencies import get_agent_service
from app.core.config import Settings
from app.core.exceptions import ApprovalConflictError
from app.db.repositories import AgentRunRepository
from app.llm.ollama_client import OllamaLLM
from app.main import create_app
from app.tools.calculator import CALCULATOR
from app.tools.registry import ToolRegistry

AUTH = {"Authorization": "Bearer approval-test"}
PERMISSIONS = {"calculator:use"}


async def exercise(case, *, flagged=True, **options):
    executions = []
    requests = []

    async def calculate(arguments):
        executions.append(arguments.expression)
        return 100

    def respond(request):
        body = json.loads(request.content)
        last = body["messages"][-1]
        requests.append(last)
        if last["role"] == "tool":
            observation = json.loads(last["content"])
            answer = str(observation["result"])
            decision = {"action": "final", "answer": answer}
        else:
            expression = "25 * 4" if last["content"] != "invalid" else 7
            decision = {"action": "tool", "name": "calculator", "arguments": {"expression": expression}}
        payload = {
            "message": {"role": "assistant", "content": json.dumps({"decision": decision})},
            "done": True,
        }
        if body["stream"]:
            return httpx.Response(200, text=json.dumps(payload) + "\n")
        return httpx.Response(200, json=payload)

    settings = Settings(_env_file=None, api_token="approval-test", api_user_id="owner", **options)
    registry = ToolRegistry()
    registry.register(replace(CALCULATOR, requires_approval=flagged, implementation=calculate))
    repository = AsyncMock(spec=AgentRunRepository)
    repository.latest.return_value = None
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        executor = AgentExecutor(settings, OllamaLLM(settings, http), registry)
        service = AgentService(executor, repository)
        app = create_app(settings)
        app.dependency_overrides[get_agent_service] = lambda: service
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            await case(api, executor, executions, requests, repository)


@pytest.mark.parametrize("approved", [True, False])
def test_http_approval_preserves_run_id_and_executes_only_if_approved(approved):
    async def case(api, executor, executions, requests, repository):
        response = await api.post("/agent/run", headers=AUTH, json={"task": "calculate", "thread_id": "math"})
        assert response.status_code == 200, response.text
        paused = response.json()
        assert paused["status"] == "awaiting_approval", paused
        approval = paused["pending_approval"]
        assert approval["arguments"] == {"expression": "25 * 4"}
        assert not executions and len(requests) == 1
        repository.latest.return_value = repository.save.await_args.args[1]

        assert (await api.get("/agent/math", headers=AUTH)).json() == paused
        assert (await api.post("/agent/run", headers=AUTH, json={"task": "new", "thread_id": "math"})).status_code == 409
        path = "/agent/math/approve" if approved else "/agent/math/reject"
        assert (await api.post(path, json={"approval_id": approval["approval_id"]})).status_code == 401
        assert (await api.post(path, headers=AUTH, json={"approval_id": str(uuid4())})).status_code == 409
        body = {"approval_id": approval["approval_id"]}
        # Concurrent requests for the same ID may pass the API read, but only one can resume the graph.
        replies = await asyncio.gather(
            api.post(path, headers=AUTH, json=body),
            api.post(path, headers=AUTH, json=body),
        )
        assert sorted(reply.status_code for reply in replies) == [200, 409]
        run = next(reply.json() for reply in replies if reply.status_code == 200)
        assert run["run_id"] == paused["run_id"]
        assert run["status"] == "completed"
        assert run["pending_approval"] is None
        assert run["current_step"] == 2
        assert run["final_answer"] == ("100" if approved else "The user rejected this tool call.")
        assert executions == (["25 * 4"] if approved else [])
        assert repository.save.await_count == 2
        assert await executor.pending(user_id="owner", thread_id="math") is None

    asyncio.run(exercise(case))


def test_stream_ends_at_approval_and_exposes_the_paused_run():
    async def case(api, executor, executions, requests, repository):
        response = await api.post("/agent/stream", headers=AUTH, json={"task": "calculate"})
        data = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert data[-1]["event"] == "approval_required"
        assert data[-1]["metadata"]["run"]["status"] == "awaiting_approval"
        assert "tool_start" not in [event["event"] for event in data]
        assert not executions
        repo_run = repository.save.await_args.args[1]
        approval = data[-1]["metadata"]["approval_id"]
        assert approval == repo_run.pending_approval["approval_id"]
        assert (await api.get(f"/agent/{repo_run.thread_id}", headers=AUTH)).json() == data[-1]["metadata"]["run"]

    asyncio.run(exercise(case, flagged=False, approval_required_tools={"calculator"}))


def test_approval_is_user_scoped_and_permissions_are_rechecked_on_resume():
    async def case(api, executor, executions, requests, repository):
        paused = await executor.run(user_id="owner", thread_id="math", task="calculate", permissions=PERMISSIONS)
        approval = paused["pending_approval"]["approval_id"]
        with pytest.raises(ApprovalConflictError):
            await executor.resume(
                user_id="other", thread_id="math", approval_id=approval, approved=True, permissions=PERMISSIONS,
            )
        result = await executor.resume(user_id="owner", thread_id="math", approval_id=approval, approved=True)
        assert result["observations"][-1]["result"] == "Permission denied for the requested tool."
        assert not executions

    asyncio.run(exercise(case))


@pytest.mark.parametrize(("task", "permissions"), [("invalid", PERMISSIONS), ("calculate", ())])
def test_invalid_or_unauthorized_tool_calls_do_not_pause(task, permissions):
    async def case(api, executor, executions, requests, repository):
        result = await executor.run(user_id="owner", thread_id="math", task=task, permissions=permissions)
        assert result["status"] == "completed"
        assert result["pending_approval"] is None
        assert result["observations"][-1]["success"] is False
        assert not executions

    asyncio.run(exercise(case))


def test_human_wait_does_not_consume_active_run_budget():
    async def case(api, executor, executions, requests, repository):
        state = await executor.run(
            user_id="owner", thread_id="math", task="calculate", permissions=PERMISSIONS,
        )
        await asyncio.sleep(0.3)
        result = await executor.resume(
            user_id="owner", thread_id="math", approval_id=state["pending_approval"]["approval_id"],
            approved=True, permissions=PERMISSIONS,
        )
        assert result["final_answer"] == "100"

    asyncio.run(exercise(case, agent_timeout_seconds=0.25))
