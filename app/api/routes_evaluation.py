from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import Authenticated, EvaluationDependency
from app.schemas.evaluation import StoredEvaluation

router = APIRouter(tags=["evaluation"])


@router.get("/evaluations")
async def list_evaluations(
    principal: Authenticated,
    evaluations: EvaluationDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=2147483647)] = 0,
) -> list[StoredEvaluation]:
    return await evaluations.list(principal.user_id, limit=limit, offset=offset)
