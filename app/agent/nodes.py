import json
from contextlib import aclosing
from asyncio import get_running_loop
from time import perf_counter
from uuid import uuid4

from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic_core import to_jsonable_python

from app.agent.policies import PolicyEngine
from app.agent.prompts import decision_schema, memory_context, tool_prompt
from app.agent.state import AgentContext, AgentDecision, AgentState, FinalDecision, ToolCall
from app.core.exceptions import LLMError, MemoryError, PolicyViolationError
from app.llm.model import ChatMessage, LLMClient
from app.memory.manager import MemoryManager
from app.tools.registry import ToolRegistry


class AgentNode:
    def __init__(
        self, llm: LLMClient, policies: PolicyEngine, registry: ToolRegistry | None
    ) -> None:
        self._llm = llm
        self._policies = policies
        self._registry = registry

    async def __call__(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        try:
            self._policies.check_agent_step(state)
        except PolicyViolationError as error:
            return {"status": "failed", "error": str(error)}
        step = state["current_step"] + 1
        messages = state["messages"]
        events = runtime.context["events"]
        output_format = "text" if self._registry is None else "json"
        if events is not None:
            await events.emit("agent_step", node="agent", step=step)
            await events.emit("llm_start", node="agent", step=step, metadata={"format": output_format})
        started = perf_counter()

        async def on_token(content: str) -> None:
            if events is not None:
                await events.emit(
                    "llm_token", node="agent", step=step,
                    metadata={"content": content, "format": output_format},
                )

        streaming = events is not None and events.streaming
        try:
            if self._registry is None:
                context = memory_context(state["retrieved_memories"])
                prompt = [ChatMessage(role="system", content=context), *messages] if context else messages
                if streaming:
                    chunks: list[str] = []
                    async with aclosing(self._llm.stream(prompt)) as stream:
                        async for chunk in stream:
                            chunks.append(chunk)
                            await on_token(chunk)
                    answer = "".join(chunks).strip()
                else:
                    answer = (await self._llm.chat(prompt)).strip()
                if not answer:
                    raise LLMError("The model returned an empty answer.")
                decision = FinalDecision(action="final", answer=answer)
            else:
                tools = self._registry.available(runtime.context["permissions"])
                observations = [message.content for message in messages if message.role == "tool"]
                decision = (await self._llm.structured_output(
                    [tool_prompt(tools, state["retrieved_memories"], observations), *messages],
                    decision_schema(tools), on_token=on_token if streaming else None,
                )).decision
        except LLMError as error:
            if events is not None:
                await events.emit(
                    "llm_end", node="agent", step=step, success=False,
                    latency_ms=(perf_counter() - started) * 1000, error=str(error),
                )
            return {"current_step": step, "status": "failed", "error": str(error)}

        if events is not None:
            await events.emit(
                "llm_end", node="agent", step=step, success=True,
                latency_ms=(perf_counter() - started) * 1000,
                metadata={"action": decision.action, "plan": decision.plan},
            )
        update: dict[str, object] = {"current_step": step}
        if decision.plan is not None:
            update["current_plan"] = decision.plan
        if isinstance(decision, FinalDecision):
            answer = ChatMessage(role="assistant", content=decision.answer)
            return update | {
                "messages": [*messages, answer],
                "final_answer": decision.answer,
                "status": "completed",
            }

        try:
            self._policies.check_tool_call(state | {"current_step": step})
        except PolicyViolationError as error:
            return update | {"status": "failed", "error": str(error)}

        call: ToolCall = {"name": decision.name, "arguments": decision.arguments}
        if self._policies.requires_approval(call, runtime.context["permissions"]):
            update |= {
                "pending_approval": {
                    "approval_id": str(uuid4()), "name": call["name"], "arguments": call["arguments"],
                },
                "remaining_seconds": max(0.0, runtime.context["deadline"] - get_running_loop().time()),
                "status": "awaiting_approval",
            }
        return update | {
            "proposed_tool": call,
            "tool_calls": [*state["tool_calls"], call],
            "messages": [*messages, ChatMessage(
                role="assistant", content=AgentDecision(decision=decision).model_dump_json()
            )],
        }


def await_approval(state: AgentState) -> dict[str, object]:
    pending = state["pending_approval"]
    assert pending is not None
    approved = interrupt(pending)
    update: dict[str, object] = {
        "status": "running", "pending_approval": None, "remaining_seconds": None,
    }
    if approved is True:
        return update
    return update | {
        "proposed_tool": None,
        "observations": [*state["observations"], {
            "tool": pending["name"], "result": "The user rejected this tool call.",
            "success": False, "attempts": [],
        }],
    }


class ToolNode:
    def __init__(self, policies: PolicyEngine) -> None:
        self._policies = policies

    async def __call__(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        call = state["proposed_tool"]
        assert call is not None
        events = runtime.context["events"]
        if events is not None:
            await events.emit(
                "tool_start", node="tool", step=state["current_step"], tool=call["name"],
                metadata={"arguments": call["arguments"]},
            )
        started = perf_counter()
        observation = await self._policies.execute(call, runtime.context["permissions"])
        if events is not None:
            await events.emit(
                "tool_result", node="tool", step=state["current_step"], tool=call["name"],
                latency_ms=(perf_counter() - started) * 1000, success=observation["success"],
                error=None if observation["success"] else str(observation["result"]),
                metadata={"observation": to_jsonable_python(observation, fallback=str)},
            )
        return {
            "proposed_tool": None,
            "observations": [*state["observations"], observation],
        }


def observe(state: AgentState) -> dict[str, object]:
    observation = state["observations"][-1]
    content = json.dumps({
        "tool": observation["tool"],
        "success": observation["success"],
        "result": to_jsonable_python(observation["result"], fallback=str),
    })
    return {"messages": [*state["messages"], ChatMessage(role="tool", content=content)]}


class MemoryNodes:
    def __init__(self, manager: MemoryManager | None) -> None:
        self._manager = manager

    async def load(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        started = perf_counter()
        memories: list[str] = []
        if self._manager is not None:
            try:
                matches = await self._manager.retrieve_relevant_memories(state["user_id"], state["task"])
                memories = [memory.content for memory in matches]
            except MemoryError as error:
                return {"status": "failed", "error": str(error)}
        if (events := runtime.context["events"]) is not None:
            await events.emit(
                "memory_retrieved", node="load_memory", success=True,
                latency_ms=(perf_counter() - started) * 1000,
                metadata={"count": len(memories), "memories": memories},
            )
        return {"retrieved_memories": memories}

    async def save(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, object]:
        if self._manager is not None:
            try:
                if state["status"] == "completed":
                    await self._manager.save_declared_preference(state["user_id"], state["task"])
                await self._manager.save_episode(
                    user_id=state["user_id"],
                    thread_id=state["thread_id"],
                    task=state["task"],
                    actions=[call["name"] for call in state["tool_calls"]],
                    result=state["final_answer"] or state["error"] or "",
                    success=state["status"] == "completed",
                )
            except MemoryError as error:
                return {"status": "failed", "error": str(error), "final_answer": None}
        if state["status"] != "completed":
            return {}
        conversation = [
            *state["conversation"],
            ChatMessage(role="user", content=state["task"]),
            state["messages"][-1],
        ]
        return {"conversation": conversation[-runtime.context["conversation_message_limit"]:]}
