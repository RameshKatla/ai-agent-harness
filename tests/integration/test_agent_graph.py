import asyncio
import json

import httpx
import pytest

from app.agent.executor import AgentExecutor
from app.core.config import Settings
from app.llm.ollama_client import OllamaLLM


@pytest.mark.parametrize(
    ("status_code", "content", "expected_status"),
    [(200, "4", "completed"), (200, " ", "failed"), (503, "", "failed")],
)
def test_agent_graph(status_code, content, expected_status):
    def respond(request):
        assert json.loads(request.content)["messages"] == [
            {"role": "user", "content": "What is 2 + 2?"}
        ]
        return httpx.Response(
            status_code, json={"message": {"role": "assistant", "content": content}, "done": True}
        )

    async def exercise():
        settings = Settings(_env_file=None, max_agent_steps=3)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            executor = AgentExecutor(settings, OllamaLLM(settings, http))
            return await executor.run(thread_id="thread", user_id="user", task="What is 2 + 2?")

    result = asyncio.run(exercise())
    assert result["thread_id"] == "thread"
    assert result["user_id"] == "user"
    assert result["status"] == expected_status
    assert result["current_step"] == 1
    assert result["max_steps"] == 3
    if expected_status == "completed":
        assert result["final_answer"] == "4"
        assert result["messages"][-1].role == "assistant"
        assert result["error"] is None
    else:
        assert result["final_answer"] is None
        assert len(result["messages"]) == 1
        assert result["error"]
