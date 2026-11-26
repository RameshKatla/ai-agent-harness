import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.executor import AgentExecutor
from app.agent.service import AgentService
from app.api.routes_agent import router as agent_router
from app.api.routes_health import router as health_router
from app.api.routes_memory import router as memory_router
from app.core.config import Settings
from app.core.exceptions import ApprovalConflictError, MemoryError, StorageError
from app.db.database import create_database_engine
from app.db.repositories import AgentRunRepository
from app.llm.ollama_client import OllamaLLM
from app.memory.manager import MemoryManager
from app.memory.semantic import OllamaEmbedder
from app.observability.logging import configure_logging
from app.tools import build_tool_registry

logger = logging.getLogger(__name__)


async def storage_error(_request: Request, error: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


async def approval_conflict(_request: Request, error: Exception) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(error)})


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(config.log_level)
        engine = create_database_engine(config)
        try:
            async with httpx.AsyncClient() as http:
                llm = OllamaLLM(config, http)
                memory = MemoryManager.using_database(
                    config, engine, OllamaEmbedder(config, http), llm,
                )
                executor = AgentExecutor(config, llm, build_tool_registry(http), memory=memory)
                app.state.memory_manager = memory
                app.state.agent_service = AgentService(
                    executor, AgentRunRepository(async_sessionmaker(engine)),
                )
                logger.info("API started")
                yield
        finally:
            await engine.dispose()

    app = FastAPI(title="AI Agent Harness", lifespan=lifespan)
    app.state.settings = config
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(memory_router)
    app.add_exception_handler(MemoryError, storage_error)
    app.add_exception_handler(StorageError, storage_error)
    app.add_exception_handler(ApprovalConflictError, approval_conflict)
    return app
