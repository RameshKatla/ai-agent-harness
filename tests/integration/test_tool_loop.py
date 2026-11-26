import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from app.agent.executor import AgentExecutor
from app.core.config import Settings
from app.core.exceptions import ToolError, TransientToolError
from app.llm.ollama_client import OllamaLLM
from app.tools.calculator import CALCULATOR
from app.tools.registry import RetryPolicy, ToolRegistry


def tool_decision(name="calculator", arguments=None, **extra):
    return {
        "action": "tool",
        "name": name,
        "arguments": {"expression": "25 * 4"} if arguments is None else arguments,
        **extra,
    }


def final_decision(answer="100", **extra):
    return {"action": "final", "answer": answer, **extra}


def run_agent(decisions, *, tool=CALCULATOR, permissions=("calculator:use",), **limits):
    requests = []

    async def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        decision = await decisions(payload) if callable(decisions) else next(decisions)
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": json.dumps({"decision": decision})},
            "done": True,
        })

    async def run():
        settings = Settings(_env_file=None, **limits)
        registry = ToolRegistry()
        registry.register(tool)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            executor = AgentExecutor(settings, OllamaLLM(settings, http), registry)
            return await executor.run(
                thread_id="thread", user_id="user", task="What is 25 * 4?",
                permissions=permissions,
            )

    return asyncio.run(run()), requests


def test_tool_result_reaches_model_and_plan_can_be_revised():
    result, requests = run_agent(iter([
        tool_decision(plan=["calculate", "respond"]),
        final_decision(plan=["report the calculated result"]),
    ]), max_tool_calls=1, max_agent_steps=2)

    assert result["status"] == "completed"
    assert result["final_answer"] == "100"
    assert result["current_step"] == 2
    assert result["current_plan"] == ["report the calculated result"]
    assert result["observations"][0]["attempts"][0]["success"] is True
    assert len(result["tool_calls"]) == 1
    schema = requests[0]["format"]
    assert schema["$defs"]["CalculatorInput"]["required"] == ["expression"]
    assert schema["$defs"]["ToolChoice0"]["properties"]["name"]["const"] == "calculator"
    assert json.loads(requests[1]["messages"][-1]["content"]) == {
        "tool": "calculator", "success": True, "result": 100,
    }


@pytest.mark.parametrize(
    ("limits", "message", "steps", "calls"),
    [
        ({"max_agent_steps": 1}, "Agent step limit reached.", 1, 0),
        ({"max_agent_steps": 3}, "Agent step limit reached.", 3, 2),
        ({"max_tool_calls": 0}, "Tool-call limit reached.", 1, 0),
        ({"max_tool_calls": 1}, "Tool-call limit reached.", 2, 1),
    ],
)
def test_repeated_tool_proposals_stop_at_policy_limits(limits, message, steps, calls):
    async def always_tool(payload):
        return tool_decision()

    result, requests = run_agent(always_tool, **limits)

    assert result["status"] == "failed"
    assert result["error"] == message
    assert result["current_step"] == len(requests) == steps
    assert len(result["tool_calls"]) == calls


@pytest.mark.parametrize(
    ("decision", "permissions", "attempts", "message"),
    [
        (tool_decision(name="unknown"), ("calculator:use",), 0, "Unknown tool."),
        (tool_decision(), (), 0, "Permission denied for the requested tool."),
        (tool_decision(arguments={"expression": 1}), ("calculator:use",), 1,
         "Invalid arguments for the requested tool."),
    ],
)
def test_untrusted_tool_proposals_cannot_run(decision, permissions, attempts, message):
    async def forbidden(arguments):
        pytest.fail("The tool implementation should never have run")

    result, requests = run_agent(
        iter([decision, final_decision("Tool unavailable")]),
        tool=replace(CALCULATOR, implementation=forbidden, retry_policy=RetryPolicy(2)),
        permissions=permissions,
    )
    observation = result["observations"][0]
    assert observation["success"] is False
    assert len(observation["attempts"]) == attempts
    assert observation["result"] == message
    if not permissions:
        assert "Available tools: []" in requests[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    ("error", "run_retries", "tool_retries", "attempt_count", "success"),
    [
        (TransientToolError, 2, 2, 2, True),
        (TransientToolError, 0, 2, 1, False),
        (TransientToolError, 2, 0, 1, False),
        (ToolError, 2, 2, 1, False),
    ],
)
def test_retry_only_for_transient_failures(error, run_retries, tool_retries, attempt_count, success):
    calls = 0

    async def flaky(arguments):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error("upstream failure")
        return 100

    result, _ = run_agent(
        iter([tool_decision(), final_decision()]),
        tool=replace(CALCULATOR, implementation=flaky, retry_policy=RetryPolicy(tool_retries)),
        max_retries=run_retries,
    )
    observation = result["observations"][0]
    assert calls == len(observation["attempts"]) == attempt_count
    assert observation["success"] is success
    assert observation["attempts"][0]["error"] == "upstream failure"
    assert all(attempt["latency_ms"] >= 0 for attempt in observation["attempts"])


def test_run_timeout_includes_model_call():
    async def slow_model(payload):
        await asyncio.sleep(10)

    result, _ = run_agent(slow_model, agent_timeout_seconds=0.03)
    assert result["status"] == "failed"
    assert result["error"] == "Agent run timed out."


def test_run_timeout_preserves_completed_tool_and_cancels_next_model_call():
    async def slow_second_step(payload):
        if payload["messages"][-1]["role"] == "tool":
            await asyncio.sleep(10)
        return tool_decision()

    result, _ = run_agent(slow_second_step, agent_timeout_seconds=0.05)
    assert result["status"] == "failed"
    assert result["error"] == "Agent run timed out."
    assert result["current_step"] == 1
    assert result["observations"][0]["result"] == 100


def test_model_cannot_override_run_limits_in_decision():
    result, _ = run_agent(iter([tool_decision(max_tool_calls=100)]), max_tool_calls=0)
    assert result["status"] == "failed"
    assert result["tool_calls"] == []
    assert "schema" in result["error"]


def test_tool_deadline_can_retry_within_the_run_deadline():
    calls = 0

    async def slow_once(arguments):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(10)
        return 100

    result, _ = run_agent(
        iter([tool_decision(), final_decision()]),
        tool=replace(
            CALCULATOR, implementation=slow_once, timeout_seconds=0.01,
            retry_policy=RetryPolicy(1),
        ),
    )
    observation = result["observations"][0]
    assert calls == 2
    assert observation["attempts"][0]["error"] == "The tool exceeded its execution deadline."
    assert observation["attempts"][1]["success"] is True
