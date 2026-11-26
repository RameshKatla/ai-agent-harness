from fastapi import APIRouter, HTTPException

from app.api.dependencies import AgentDependency, Authenticated
from app.api.sse import EventStreamResponse
from app.schemas.agent import AgentRunRequest, AgentRunResponse, ApprovalRequest, ThreadId

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run")
async def run_agent(
    body: AgentRunRequest, principal: Authenticated, agent: AgentDependency,
) -> AgentRunResponse:
    return await agent.run(
        user_id=principal.user_id, thread_id=body.thread_id,
        task=body.task, permissions=principal.permissions,
    )


@router.post(
    "/stream",
    status_code=200,
    response_class=EventStreamResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def stream_agent(
    body: AgentRunRequest, principal: Authenticated, agent: AgentDependency,
) -> EventStreamResponse:
    return EventStreamResponse(agent.stream(
        user_id=principal.user_id, thread_id=body.thread_id,
        task=body.task, permissions=principal.permissions,
    ))


@router.get("/{thread_id}")
async def get_thread(
    thread_id: ThreadId, principal: Authenticated, agent: AgentDependency,
) -> AgentRunResponse:
    result = await agent.get_thread(user_id=principal.user_id, thread_id=thread_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return result



@router.post("/{thread_id}/approve")
async def approve_tool(
    thread_id: ThreadId, body: ApprovalRequest, principal: Authenticated, agent: AgentDependency,
) -> AgentRunResponse:
    return await resolve_tool(thread_id, body, principal, agent, approved=True)


@router.post("/{thread_id}/reject")
async def reject_tool(
    thread_id: ThreadId, body: ApprovalRequest, principal: Authenticated, agent: AgentDependency,
) -> AgentRunResponse:
    return await resolve_tool(thread_id, body, principal, agent, approved=False)


async def resolve_tool(
    thread_id: str, body: ApprovalRequest, principal: Authenticated, agent: AgentDependency,
    *, approved: bool,
) -> AgentRunResponse:
    result = await agent.resolve_approval(
        user_id=principal.user_id, thread_id=thread_id, approval_id=body.approval_id,
        approved=approved, permissions=principal.permissions,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return result
