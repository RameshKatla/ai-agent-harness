import asyncio
from collections.abc import Collection
from contextlib import aclosing
from typing import cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.policies import PolicyEngine
from app.agent.state import AgentState
from app.core.config import Settings
from app.core.exceptions import ApprovalConflictError
from app.llm.model import ChatMessage, LLMClient
from app.memory.manager import MemoryManager
from app.memory.short_term import ShortTermMemory, conversation_key
from app.observability.events import RunEvents
from app.tools.registry import ToolRegistry


class AgentExecutor:
    def __init__(
        self, settings: Settings, llm: LLMClient, registry: ToolRegistry | None = None,
        *, memory: MemoryManager | None = None,
    ) -> None:
        self._settings = settings
        policies = PolicyEngine(settings, registry if registry is not None else ToolRegistry())
        self._memory = ShortTermMemory()
        self._graph = build_graph(llm, policies, registry, self._memory.checkpointer, memory)

    def _config(self, key: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": key},
            "recursion_limit": 4 * self._settings.max_agent_steps + 3,
        }

    async def _advance(
        self, input_state: AgentState | Command, latest: AgentState, config: RunnableConfig,
        permissions: Collection[str], events: RunEvents | None, deadline: float,
    ) -> AgentState:
        try:
            async with asyncio.timeout_at(deadline):
                async with aclosing(self._graph.astream(
                    input_state,
                    config=config,
                    context={
                        "permissions": frozenset(permissions),
                        "events": events,
                        "conversation_message_limit": 2 * self._settings.max_conversation_turns,
                        "deadline": deadline,
                    },
                    stream_mode="values",
                    durability="sync",
                )) as states:
                    async for state in states:
                        if "thread_id" in state:
                            latest = cast(AgentState, state)
        except TimeoutError:
            latest = self._timed_out(latest)
            snapshot = await self._graph.aget_state(config)
            if snapshot.values.get("pending_approval") is not None:
                await self._graph.aupdate_state(
                    config, {
                        "status": "failed", "error": latest["error"], "pending_approval": None,
                        "remaining_seconds": None, "proposed_tool": None,
                    }, as_node="save_memory",
                )
        return latest

    @staticmethod
    def _timed_out(state: AgentState) -> AgentState:
        return state | {
            "status": "failed", "error": "Agent run timed out.", "pending_approval": None,
            "remaining_seconds": None, "proposed_tool": None, "final_answer": None,
        }

    async def run(
        self, *, thread_id: str, user_id: str, task: str,
        permissions: Collection[str] = (), events: RunEvents | None = None,
    ) -> AgentState:
        key = conversation_key(user_id, thread_id)
        config = self._config(key)
        task_message = ChatMessage(role="user", content=task)
        initial: AgentState = {
            "run_id": str(events.run_id if events is not None else uuid4()),
            "thread_id": thread_id,
            "user_id": user_id,
            "task": task,
            "messages": [task_message],
            "conversation": [],
            "current_plan": [],
            "current_step": 0,
            "max_steps": self._settings.max_agent_steps,
            "tool_calls": [],
            "observations": [],
            "retrieved_memories": [],
            "pending_approval": None,
            "remaining_seconds": None,
            "proposed_tool": None,
            "final_answer": None,
            "status": "running",
            "error": None,
            "metadata": {},
        }
        deadline = asyncio.get_running_loop().time() + self._settings.agent_timeout_seconds
        try:
            async with asyncio.timeout_at(deadline) as preparation:
                async with self._memory.lock(key):
                    previous = await self._graph.aget_state(config)
                    if previous.interrupts and previous.values.get("pending_approval") is not None:
                        raise ApprovalConflictError("This thread already has a pending approval.")
                    history = previous.values.get("conversation", [])
                    initial = initial | {"messages": [*history, task_message], "conversation": history}
                    preparation.reschedule(None)
                    return await self._advance(initial, initial, config, permissions, events, deadline)
        except TimeoutError:
            return self._timed_out(initial)

    async def pending(self, *, thread_id: str, user_id: str) -> AgentState | None:
        snapshot = await self._graph.aget_state(self._config(conversation_key(user_id, thread_id)))
        if snapshot.interrupts and snapshot.values.get("pending_approval") is not None:
            return cast(AgentState, snapshot.values)
        return None

    async def resume(
        self, *, thread_id: str, user_id: str, approval_id: str, approved: bool,
        permissions: Collection[str] = (), events: RunEvents | None = None,
    ) -> AgentState:
        key = conversation_key(user_id, thread_id)
        config = self._config(key)
        async with self._memory.lock(key):
            previous = await self._graph.aget_state(config)
            pending = previous.values.get("pending_approval")
            if pending is None or pending["approval_id"] != approval_id or not previous.interrupts:
                raise ApprovalConflictError("This approval is no longer pending.")
            remaining = previous.values["remaining_seconds"]
            deadline = asyncio.get_running_loop().time() + remaining
            return await self._advance(
                Command(resume=approved), cast(AgentState, previous.values), config,
                permissions, events, deadline,
            )
