from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings


def database_url(settings: Settings) -> URL:
    password = settings.postgres_password
    return URL.create(
        "postgresql+asyncpg",
        username=settings.postgres_user,
        password=password.get_secret_value() if password else None,
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
    )


def create_database_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(database_url(settings), pool_pre_ping=True, hide_parameters=True)
