"""Alembic environment.

The connection URL is **not** read from alembic.ini. It comes from the same
``DATABASE_URL`` setting the application uses, through the same normaliser — so
a URL that works for the app works for migrations, including Supabase URIs with
``?sslmode=require`` that asyncpg would otherwise reject.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.db.base import Base
from app.db.session import normalise_url

# Importing the models registers them on Base.metadata; without this,
# autogenerate produces an empty migration.
from app.db import models  # noqa: F401  # registers tables on Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

_url, _connect_args = normalise_url(settings.database_url)
if not _url:
    raise SystemExit(
        "DATABASE_URL is not set. Add it to .env before running migrations:\n"
        "  DATABASE_URL=postgresql://user:pass@host:5432/dbname"
    )

# NOT config.set_main_option(): alembic.ini is parsed by configparser, which
# treats '%' as interpolation syntax. Any percent-encoded character in the
# password — '@' becomes %40, and '@' is common in passwords — raises
# "invalid interpolation syntax" before a single migration runs. The engine is
# built directly from the URL below instead, so the value never touches
# configparser. Escaping to '%%' would also work but silently corrupts the
# value anywhere else the option is read.


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — useful for reviewing a change."""
    context.configure(
        url=_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(
        _url,
        poolclass=pool.NullPool,
        connect_args=_connect_args,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
