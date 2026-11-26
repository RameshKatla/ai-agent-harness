from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.agent.service import AgentService
from app.core.security import Authenticator, Principal
from app.db.repositories import EvaluationRepository
from app.memory.manager import MemoryManager

bearer = HTTPBearer(auto_error=False)


def get_authenticator(request: Request) -> Authenticator:
    return request.app.state.authenticator


async def require_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    authenticator: Annotated[Authenticator, Depends(get_authenticator)],
) -> Principal:
    settings = request.app.state.settings
    if not authenticator.configured:
        raise HTTPException(status_code=503, detail="API access is not configured.")
    principal = await authenticator.authenticate(credentials.credentials) if credentials is not None else None
    if principal is None:
        raise HTTPException(
            status_code=401, detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    redis = request.app.state.redis_services if settings.redis_enabled else None
    if redis is not None and (retry_after := await redis.retry_after(principal.user_id)) is not None:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded.", headers={"Retry-After": str(retry_after)},
        )
    return principal


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory_manager


Authenticated = Annotated[Principal, Depends(require_principal)]
AgentDependency = Annotated[AgentService, Depends(get_agent_service)]
MemoryDependency = Annotated[MemoryManager, Depends(get_memory_manager)]


def get_evaluations(request: Request) -> EvaluationRepository:
    return request.app.state.evaluations


EvaluationDependency = Annotated[EvaluationRepository, Depends(get_evaluations)]
