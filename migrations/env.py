import asyncio

from alembic import context
from sqlalchemy import Connection

from app.core.config import Settings
from app.db.database import create_database_engine, database_url
from app.db.models import Base


def migrate(connection: Connection | None = None) -> None:
    if connection is None:
        context.configure(
            url=database_url(Settings()), target_metadata=Base.metadata,
            literal_binds=True, dialect_opts={"paramstyle": "named"},
        )
    else:
        context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


async def migrate_online() -> None:
    engine = create_database_engine(Settings())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    migrate()
else:
    asyncio.run(migrate_online())
