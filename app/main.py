import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.exceptions import HTTPException

from app.agent.executor import AgentExecutor
from app.agent.service import AgentService
from app.api.errors import ErrorResponse, agent_error, http_error, validation_error
from app.api.middleware import RequestSecurityMiddleware
from app.api.readiness import Readiness, ollama_ready, postgres_ready
from app.api.routes_agent import router as agent_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_health import router as health_router
from app.api.routes_memory import router as memory_router
from app.core.config import Settings
from app.core.exceptions import AgentError
from app.core.redis import RedisServices
from app.core.security import ConfiguredBearerAuthenticator, Principal
from app.db.database import create_database_engine
from app.db.repositories import AgentRunRepository, EvaluationRepository
from app.llm.ollama_client import OllamaLLM
from app.memory.manager import MemoryManager
from app.memory.semantic import OllamaEmbedder
from app.memory.short_term import PostgresShortTermMemory
from app.observability.logging import configure_logging
from app.tools import BUILTIN_PERMISSIONS, build_tool_registry

logger = logging.getLogger(__name__)


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
            probes = [partial(postgres_ready, engine), partial(ollama_ready, config, http)]
            if redis is not None:
                probes.append(redis.ready)
            app.state.readiness = Readiness(*probes)
            llm = OllamaLLM(config, http)
            memory = MemoryManager.using_database(
                config, engine, OllamaEmbedder(config, http), llm,
            )
            registry = build_tool_registry(
                http, cache=redis, weather_cache_ttl=config.weather_cache_ttl_seconds,
            )
            short_term = await stack.enter_async_context(PostgresShortTermMemory.connect(config))
            executor = AgentExecutor(config, llm, registry, memory=memory, short_term=short_term)
            app.state.memory_manager = memory
            sessions = async_sessionmaker(engine)
            app.state.agent_service = AgentService(
                executor, AgentRunRepository(sessions),
                observer=redis.publish if redis is not None else None,
            )
            app.state.evaluations = EvaluationRepository(sessions)
            logger.info("API started")
            yield

    app = FastAPI(
        title="AI Agent Harness", lifespan=lifespan,
        responses={status: {"model": ErrorResponse} for status in (413, 422, 500)},
    )
    app.state.settings = config
    permitted = frozenset(
        permission for permission in BUILTIN_PERMISSIONS
        if config.api_allowed_tools is None or permission.split(":")[0] in config.api_allowed_tools
    )
    app.state.authenticator = ConfiguredBearerAuthenticator(
        config.api_token, Principal(config.api_user_id, permitted),
    )
    app.add_middleware(RequestSecurityMiddleware, max_body_bytes=config.max_request_body_bytes)
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(memory_router)
    app.include_router(evaluation_router)
    app.add_exception_handler(HTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(AgentError, agent_error)
    return app
