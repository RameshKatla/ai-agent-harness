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
    if not await request.app.state.readiness.ready():
        response.status_code = 503
        return HealthResponse(status="unavailable")
    return HealthResponse(status="ready")
