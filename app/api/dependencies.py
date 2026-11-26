from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.agent.service import AgentService
from app.db.repositories import EvaluationRepository
from app.memory.manager import MemoryManager
from app.tools import BUILTIN_PERMISSIONS

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: str
    permissions: frozenset[str]


def require_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    settings = request.app.state.settings
    token = settings.api_token.get_secret_value() if settings.api_token else ""
    if not token:
        raise HTTPException(status_code=503, detail="API access is not configured.")
    if credentials is None or not compare_digest(credentials.credentials.encode(), token.encode()):
        raise HTTPException(
            status_code=401, detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(settings.api_user_id, BUILTIN_PERMISSIONS)


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
