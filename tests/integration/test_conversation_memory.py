import asyncio
import json
from collections.abc import Callable

import httpx

from app.agent.executor import AgentExecutor
from app.core.config import Settings
from app.llm.ollama_client import OllamaLLM
from app.tools import build_tool_registry


def exercise(case: Callable, *, with_tools=False, **limits):
    requests = []

    async def respond(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        requests.append(messages)
        task = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        if task == "fail":
            return httpx.Response(503)
        if task == "timeout":
            await asyncio.sleep(10)
        content = f"answer to {task}"
        if with_tools:
            if task == "calculate" and messages[-1]["role"] != "tool":
                decision = {"action": "tool", "name": "calculator", "arguments": {"expression": "2*2"}}
            else:
                decision = {"action": "final", "answer": content}
            content = json.dumps({"decision": decision})
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": content}, "done": True,
        })

    async def run():
        settings = Settings(_env_file=None, **limits)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            registry = build_tool_registry(http) if with_tools else None
            executor = AgentExecutor(settings, OllamaLLM(settings, http), registry)
            await case(executor)

    asyncio.run(run())
    return requests


async def turn(executor, task, *, user="user", thread="thread", permissions=()):
    return await executor.run(thread_id=thread, user_id=user, task=task, permissions=permissions)


def contents(messages):
    return [message["content"] for message in messages]


def test_conversation_history_is_scoped_to_user_and_thread():
    async def case(executor):
        await turn(executor, "first")
        await turn(executor, "other user", user="other")
        await turn(executor, "other thread", thread="other")
        result = await turn(executor, "follow-up")
        assert result["current_step"] == 1
        assert result["final_answer"] == "answer to follow-up"

    requests = exercise(case)
    assert [contents(request) for request in requests] == [
        ["first"], ["other user"], ["other thread"], ["first", "answer to first", "follow-up"],
    ]


def test_history_retains_recent_successful_turn_pairs():
    async def case(executor):
        for task in ("first", "second", "fail", "last"):
            await turn(executor, task)

    requests = exercise(case, max_conversation_turns=1)
    assert contents(requests[-1]) == ["second", "answer to second", "last"]


def test_timeout_does_not_add_a_partial_turn_to_history():
    async def case(executor):
        await turn(executor, "first")
        result = await turn(executor, "timeout")
        assert result["error"] == "Agent run timed out."
        assert (await turn(executor, "last"))["status"] == "completed"

    requests = exercise(case, agent_timeout_seconds=0.15)
    assert contents(requests[-1]) == ["first", "answer to first", "last"]


def test_concurrent_turns_on_the_same_thread_are_serialized():
    async def case(executor):
        results = await asyncio.gather(turn(executor, "first"), turn(executor, "second"))
        assert all(result["status"] == "completed" for result in results)

    requests = exercise(case)
    assert contents(requests[1]) == ["first", "answer to first", "second"]


def test_followup_starts_with_fresh_tool_budget_and_no_tool_transcript():
    async def case(executor):
        for _ in range(2):
            result = await turn(executor, "calculate", permissions={"calculator:use"})
            assert result["status"] == "completed"
            assert len(result["tool_calls"]) == 1
            assert result["observations"][0]["result"] == 4

    requests = exercise(case, with_tools=True, max_tool_calls=1)
    assert [message["role"] for message in requests[2]] == ["system", "user", "assistant", "user"]
    assert contents(requests[2][1:]) == ["calculate", "answer to calculate", "calculate"]
