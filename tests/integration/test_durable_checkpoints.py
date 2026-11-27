import asyncio
import json
import os
from uuid import uuid4

import httpx
import pytest

from app.agent.executor import AgentExecutor
from app.core.config import Settings
from app.core.exceptions import ApprovalConflictError
from app.llm.ollama_client import OllamaLLM
from app.memory.short_term import PostgresShortTermMemory, conversation_key
from app.tools import build_tool_registry

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="requires migrated PostgreSQL"),
]


def test_approval_and_conversation_survive_new_worker():
    async def exercise():
        settings = Settings(approval_required_tools={"calculator"})
        user = str(uuid4())
        thread = "checkpoint-test"

        def respond(request):
            messages = json.loads(request.content)["messages"]
            if messages[-1]["role"] == "tool":
                decision = {"action": "final", "answer": "100"}
            elif messages[-1]["content"] == "Follow up.":
                assert any(message["content"] == "100" for message in messages[:-1])
                decision = {"action": "final", "answer": "Follow-up recalled 100."}
            else:
                decision = {"action": "tool", "name": "calculator", "arguments": {"expression": "25 * 4"}}
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": json.dumps({"decision": decision})}, "done": True,
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            llm = OllamaLLM(settings, http)
            tools = build_tool_registry(http)
            permissions = {"calculator:use"}
            async with PostgresShortTermMemory.connect(settings) as first:
                executor = AgentExecutor(settings, llm, tools, short_term=first)
                paused = await executor.run(thread_id=thread, user_id=user, task="Calculate.", permissions=permissions)
            assert paused["status"] == "awaiting_approval"
            approval_id = paused["pending_approval"]["approval_id"]

            async with PostgresShortTermMemory.connect(settings) as second, PostgresShortTermMemory.connect(settings) as third:
                try:
                    executor = AgentExecutor(settings, llm, tools, short_term=second)
                    other_worker = AgentExecutor(settings, llm, tools, short_term=third)
                    assert await executor.pending(thread_id=thread, user_id="another-user") is None
                    kwargs = {
                        "thread_id": thread, "user_id": user, "approval_id": approval_id,
                        "approved": True, "permissions": permissions,
                    }
                    outcomes = await asyncio.gather(
                        executor.resume(**kwargs), other_worker.resume(**kwargs), return_exceptions=True,
                    )
                    assert sum(isinstance(result, ApprovalConflictError) for result in outcomes) == 1
                    completed, = [result for result in outcomes if isinstance(result, dict)]
                    assert completed["final_answer"] == "100"
                    assert len(completed["observations"]) == 1
                    followup = await executor.run(thread_id=thread, user_id=user, task="Follow up.", permissions=permissions)
                    assert followup["final_answer"] == "Follow-up recalled 100."
                finally:
                    await second.checkpointer.adelete_thread(conversation_key(user, thread))

    asyncio.run(exercise())
