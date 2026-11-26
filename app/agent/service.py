import asyncio
from collections.abc import AsyncGenerator, Collection
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

from anyio import CancelScope

from app.agent.executor import AgentExecutor
from app.agent.state import AgentState
from app.core.exceptions import ApprovalConflictError, StorageError
from app.db.repositories import AgentRunRepository
from app.observability.events import AgentEvent, EventSink, RunEvents
from app.schemas.agent import AgentRunResponse


class AgentService:
    def __init__(
        self, executor: AgentExecutor, runs: AgentRunRepository, *, observer: EventSink | None = None,
    ) -> None:
        self._executor = executor
        self._runs = runs
        self._observer = observer

    async def run(
        self, *, user_id: str, thread_id: str | None, task: str, permissions: Collection[str],
    ) -> AgentRunResponse:
        events = RunEvents(uuid4(), thread_id or str(uuid4()), user_id, sink=None, observer=self._observer)
        return await self._run(events, task, permissions)

    async def _run(
        self, events: RunEvents, task: str, permissions: Collection[str],
    ) -> AgentRunResponse:
        started = perf_counter()
        await events.emit("agent_start", node="executor")
        state = await self._executor.run(
            user_id=events.user_id, thread_id=events.thread_id, task=task,
            permissions=permissions, events=events,
        )
        return await self._publish(events, state, started)

    async def _publish(
        self, events: RunEvents, state: AgentState, started: float,
    ) -> AgentRunResponse:
        finished_at = datetime.now(UTC)
        response = AgentRunResponse.model_validate(state)
        await self._runs.save(events.user_id, response, finished_at=finished_at)
        if response.pending_approval is not None:
            pending = response.pending_approval
            await events.emit(
                "approval_required", node="approval", step=response.current_step, tool=pending["name"],
                metadata={
                    "approval_id": pending["approval_id"], "arguments": pending["arguments"],
                    "run": response.model_dump(mode="json"),
                },
            )
            return response
        if response.error is not None:
            await events.emit(
                "agent_error", node="executor",
                success=False, error=response.error,
            )
        await events.emit(
            "agent_complete", node="executor",
            success=response.status == "completed", latency_ms=(perf_counter() - started) * 1000,
            metadata={"status": response.status, "run": response.model_dump(mode="json")}
            if events.streaming else {"status": response.status},
        )
        return response

    async def stream(
        self, *, user_id: str, thread_id: str | None, task: str, permissions: Collection[str],
    ) -> AsyncGenerator[AgentEvent | None, None]:
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=32)
        events = RunEvents(uuid4(), thread_id or str(uuid4()), user_id, queue.put, observer=self._observer)

        async def produce() -> None:
            try:
                await self._run(events, task, permissions)
            except Exception as error:
                if isinstance(error, ApprovalConflictError):
                    message = "The thread is not ready for this approval action."
                elif isinstance(error, StorageError):
                    message = "The storage operation is temporarily unavailable."
                else:
                    message = "The agent stream failed."
                await events.emit("agent_error", node="executor", success=False, error=message)
                await events.emit(
                    "agent_complete", node="executor", success=False, metadata={"status": "failed"},
                )
            await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield None
                    continue
                if event is None:
                    return
                yield event
        finally:
            producer.cancel()
            with CancelScope(shield=True):
                await asyncio.gather(producer, return_exceptions=True)

    async def resolve_approval(
        self, *, user_id: str, thread_id: str, approval_id: UUID, approved: bool,
        permissions: Collection[str],
    ) -> AgentRunResponse | None:
        state = await self._executor.pending(user_id=user_id, thread_id=thread_id)
        if state is None:
            if await self._runs.latest(user_id, thread_id) is None:
                return None
            raise ApprovalConflictError("This approval is no longer pending.")
        events = RunEvents(UUID(state["run_id"]), thread_id, user_id, sink=None, observer=self._observer)
        started = perf_counter()
        result = await self._executor.resume(
            thread_id=thread_id, user_id=user_id, approval_id=str(approval_id),
            approved=approved, permissions=permissions, events=events,
        )
        return await self._publish(events, result, started)

    async def get_thread(self, *, user_id: str, thread_id: str) -> AgentRunResponse | None:
        pending = await self._executor.pending(user_id=user_id, thread_id=thread_id)
        if pending is not None:
            return AgentRunResponse.model_validate(pending)
        return await self._runs.latest(user_id, thread_id)
