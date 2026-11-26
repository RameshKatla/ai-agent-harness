import asyncio
import os

import httpx
import pytest

from app.core.config import Settings
from app.llm.model import ChatMessage
from app.llm.ollama_client import OllamaLLM


@pytest.mark.external
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="requires local Ollama")
def test_live_ollama_chat():
    async def smoke_test():
        async with httpx.AsyncClient() as http:
            client = OllamaLLM(Settings(), http)
            answer = await client.chat(
                [ChatMessage(role="user", content="What is 2 + 2? Reply with the digit only.")]
            )
            assert "4" in answer

    asyncio.run(smoke_test())


@pytest.mark.external
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="requires local Ollama")
def test_live_ollama_agent_graph():
    from app.agent.executor import AgentExecutor

    async def smoke_test():
        settings = Settings()
        async with httpx.AsyncClient() as http:
            executor = AgentExecutor(settings, OllamaLLM(settings, http))
            result = await executor.run(
                thread_id="live-smoke-test",
                user_id="test",
                task="What is 2 + 2? Reply with the digit only.",
            )
            assert result["status"] == "completed", result["error"]
            assert "4" in result["final_answer"]

    asyncio.run(smoke_test())


@pytest.mark.external
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="requires local Ollama")
def test_live_ollama_calculator_loop():
    from app.agent.executor import AgentExecutor
    from app.tools import build_tool_registry

    async def smoke_test():
        settings = Settings()
        async with httpx.AsyncClient() as http:
            executor = AgentExecutor(settings, OllamaLLM(settings, http), build_tool_registry(http))
            result = await executor.run(
                thread_id="live-calculator",
                user_id="test",
                task="What is 25 * 4?",
                permissions={"calculator:use"},
            )
            assert result["status"] == "completed", result["error"]
            assert result["observations"], result["final_answer"]
            assert result["observations"][0]["tool"] == "calculator"
            assert result["observations"][0]["result"] == 100
            assert "100" in result["final_answer"]

    asyncio.run(smoke_test())


@pytest.mark.external
@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="requires local Ollama")
def test_live_ollama_conversation_followup():
    from app.agent.executor import AgentExecutor

    async def smoke_test():
        settings = Settings()
        async with httpx.AsyncClient() as http:
            executor = AgentExecutor(settings, OllamaLLM(settings, http))
            first = await executor.run(
                thread_id="live-conversation",
                user_id="test",
                task="My project is code-named Blue Lantern. Reply only with OK.",
            )
            assert first["status"] == "completed", first["error"]
            followup = await executor.run(
                thread_id="live-conversation",
                user_id="test",
                task="What is my project's code name? Reply only with the name.",
            )
            assert followup["status"] == "completed", followup["error"]
            assert "blue lantern" in followup["final_answer"].casefold()

    asyncio.run(smoke_test())
