from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "unavailable"]


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", responses={503: {"model": HealthResponse}})
async def ready(request: Request, response: Response) -> HealthResponse:
    redis = request.app.state.redis_services if request.app.state.settings.redis_enabled else None
    if redis is not None and not await redis.ready():
        response.status_code = 503
        return HealthResponse(status="unavailable")
    return HealthResponse(status="ready")
