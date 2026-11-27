import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.db.database import create_database_engine
from app.db.models import AgentEvent, AuditLog, Message, ToolCall, User
from app.db.repositories import AgentRunRepository
from app.llm.model import ChatMessage
from app.observability.events import AgentEvent as ObservedEvent
from app.schemas.agent import AgentRunResponse

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="requires migrated PostgreSQL"),
]


def test_history_persists_a_resumed_run_without_duplicates():
    async def exercise():
        engine = create_database_engine(Settings())
        user, thread = str(uuid4()), "history-test"
        run_id, approval_id = uuid4(), uuid4()
        now = datetime.now(UTC)
        sessions = async_sessionmaker(engine)
        repo = AgentRunRepository(sessions)
        tool_call = {"name": "calculator", "arguments": {"expression": "25 * 4"}}
        paused = AgentRunResponse(
            run_id=run_id, thread_id=thread, status="awaiting_approval", final_answer=None,
            error=None, current_plan=[], current_step=1, max_steps=10, tool_calls=[tool_call],
            observations=[], retrieved_memories=[], pending_approval={**tool_call, "approval_id": str(approval_id)},
            messages=[
                ChatMessage(role="user", content="Earlier turn"),
                ChatMessage(role="assistant", content="Earlier answer"),
                ChatMessage(role="user", content="25 * 4"),
            ],
        )
        final = paused.model_copy(update={
            "status": "completed", "pending_approval": None, "final_answer": "100",
            "messages": [*paused.messages, ChatMessage(role="assistant", content="100")],
            "observations": [{"tool": "calculator", "result": 100, "success": True, "attempts": []}],
        })
        event = ObservedEvent(
            run_id=run_id, thread_id=thread, user_id=user, sequence=1, timestamp=now,
            step=1, event="approval_required", node="approval", metadata={"secret": "private"},
        )
        try:
            await repo.save(user, paused, finished_at=now, events=[event])
            updated_at = now + timedelta(seconds=1)
            completed = event.model_copy(update={"event": "agent_complete"})
            for _ in range(2):
                await repo.save(user, final, finished_at=updated_at, events=[completed], approval=(approval_id, True))
            assert await repo.latest(user, thread) == final
            async with sessions() as session:
                messages = (await session.scalars(
                    select(Message).where(Message.run_id == run_id).order_by(Message.position)
                )).all()
                assert [row.content for row in messages] == ["25 * 4", "100"]
                call, = (await session.scalars(select(ToolCall).where(ToolCall.run_id == run_id))).all()
                assert call.observation["result"] == 100
                events = (await session.scalars(
                    select(AgentEvent).where(AgentEvent.run_id == run_id).order_by(AgentEvent.sequence)
                )).all()
                assert [(row.sequence, row.event) for row in events] == [(1, "approval_required"), (2, "agent_complete")]
                assert all("metadata" not in row.payload for row in events)
                audit = (await session.scalars(select(AuditLog).where(AuditLog.user_id == user))).all()
                assert {row.action for row in audit} == {"approval_requested", "approval_granted", "run_finished"}
                assert len(audit) == 3
        finally:
            async with engine.begin() as conn:
                await conn.execute(delete(AuditLog).where(AuditLog.user_id == user))
                await conn.execute(delete(User).where(User.id == user))
            await engine.dispose()

    asyncio.run(exercise())
