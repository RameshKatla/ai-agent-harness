import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.executor import AgentExecutor
from app.agent.service import AgentService
from app.api.routes_agent import router as agent_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_health import router as health_router
from app.api.routes_memory import router as memory_router
from app.core.config import Settings
from app.core.exceptions import ApprovalConflictError, MemoryError, RedisUnavailableError, StorageError
from app.core.redis import RedisServices
from app.db.database import create_database_engine
from app.db.repositories import AgentRunRepository, EvaluationRepository
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
        async with AsyncExitStack() as stack:
            engine = create_database_engine(config)
            stack.push_async_callback(engine.dispose)
            redis = RedisServices.connect(config) if config.redis_enabled else None
            app.state.redis_services = redis
            if redis is not None:
                stack.push_async_callback(redis.close)
            http = await stack.enter_async_context(httpx.AsyncClient())
            llm = OllamaLLM(config, http)
            memory = MemoryManager.using_database(
                config, engine, OllamaEmbedder(config, http), llm,
            )
            registry = build_tool_registry(
                http, cache=redis, weather_cache_ttl=config.weather_cache_ttl_seconds,
            )
            executor = AgentExecutor(config, llm, registry, memory=memory)
            app.state.memory_manager = memory
            sessions = async_sessionmaker(engine)
            app.state.agent_service = AgentService(
                executor, AgentRunRepository(sessions),
                observer=redis.publish if redis is not None else None,
            )
            app.state.evaluations = EvaluationRepository(sessions)
            logger.info("API started")
            yield

    app = FastAPI(title="AI Agent Harness", lifespan=lifespan)
    app.state.settings = config
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(memory_router)
    app.include_router(evaluation_router)
    app.add_exception_handler(MemoryError, storage_error)
    app.add_exception_handler(StorageError, storage_error)
    app.add_exception_handler(RedisUnavailableError, storage_error)
    app.add_exception_handler(ApprovalConflictError, approval_conflict)
    return app
