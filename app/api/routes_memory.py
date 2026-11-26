from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from app.api.dependencies import Authenticated, MemoryDependency
from app.schemas.memory import MemoryCreate, MemoryRecord

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("")
async def list_memories(
    principal: Authenticated,
    memory: MemoryDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=2147483647)] = 0,
) -> list[MemoryRecord]:
    return await memory.list_memories(principal.user_id, limit=limit, offset=offset)


@router.post("", status_code=201)
async def create_memory(
    body: MemoryCreate, principal: Authenticated, memory: MemoryDependency,
) -> MemoryRecord:
    return await memory.save_semantic_memory(principal.user_id, body.content)


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: UUID, principal: Authenticated, memory: MemoryDependency,
) -> Response:
    if not await memory.delete_memory(principal.user_id, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return Response(status_code=204)
