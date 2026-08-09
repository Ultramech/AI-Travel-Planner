"""Engine and session lifecycle.

The database is **optional**, exactly like Redis and Kafka elsewhere in this
project: with no ``DATABASE_URL`` the app still runs, still plans trips, and
simply loses durable history. That keeps `git clone && make api` working with
nothing installed, and it means a database outage degrades the service rather
than stopping it.

Connecting to Supabase (or any pgbouncer) has two traps that are expensive to
find at runtime, so both are handled here:

1. **``sslmode`` is a libpq parameter, not an asyncpg one.** Supabase hands you
   a URI ending in ``?sslmode=require``; passing that through to asyncpg raises
   ``TypeError: connect() got an unexpected keyword argument 'sslmode'``. We
   strip it and translate it into asyncpg's ``ssl`` argument.
2. **Transaction-mode pooling breaks prepared statements.** Supabase's pooler on
   port 6543 multiplexes connections, so asyncpg's cached prepared statements
   end up on the wrong backend and fail with ``prepared statement "__asyncpg_..."
   does not exist``. On that port we disable the statement cache and give each
   statement a unique name.
"""

from __future__ import annotations

import logging
import ssl as ssl_module
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import settings

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
_available: bool = False

# Ports where Supabase/pgbouncer run in transaction-pooling mode.
_TRANSACTION_POOLER_PORTS = {6543}


def normalise_url(url: str) -> tuple[str, dict[str, Any]]:
    """Return an asyncpg-compatible URL plus the connect args it needs.

    Accepts anything a user is likely to paste: ``postgres://``,
    ``postgresql://``, with or without ``?sslmode=``, Supabase pooler or direct.
    """
    if not url:
        return "", {}

    parts = urlsplit(url)

    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    elif scheme.startswith("postgresql+") and scheme != "postgresql+asyncpg":
        # psycopg2/psycopg URLs would work but block the event loop.
        logger.warning("Rewriting driver %r to asyncpg for async use.", scheme)
        scheme = "postgresql+asyncpg"

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    connect_args: dict[str, Any] = {}

    sslmode = query.pop("sslmode", None)
    if sslmode in ("require", "verify-ca", "verify-full"):
        ctx = ssl_module.create_default_context()
        if sslmode == "require":
            # Supabase presents a cert chain many local trust stores lack; the
            # transport is still encrypted, we just do not verify the peer.
            ctx.check_hostname = False
            ctx.verify_mode = ssl_module.CERT_NONE
        connect_args["ssl"] = ctx
    elif sslmode == "disable":
        connect_args["ssl"] = False

    # pgbouncer=true is a Prisma-ism; asyncpg does not know it.
    pgbouncer = query.pop("pgbouncer", None) == "true"

    port = parts.port
    if pgbouncer or port in _TRANSACTION_POOLER_PORTS:
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_cache_size"] = 0
        connect_args["prepared_statement_name_func"] = lambda: f"__tp_{uuid4().hex}__"
        logger.info("Transaction-pooling mode detected; prepared statements disabled.")

    rebuilt = urlunsplit(
        (scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )
    return rebuilt, connect_args


def checkpointer_dsn() -> str:
    """A libpq DSN for the LangGraph Postgres saver.

    The saver is built on **psycopg**, not asyncpg, so it wants the opposite of
    what :func:`normalise_url` produces: the plain ``postgresql://`` scheme with
    ``sslmode`` left in the query string. Handing it the SQLAlchemy URL is a
    silent-looking failure that surfaces much later as a connection error.
    """
    url = settings.database_url
    if not url:
        return ""

    parts = urlsplit(url)
    scheme = "postgresql"
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.pop("pgbouncer", None)
    # Supabase requires TLS; be explicit rather than relying on libpq defaults.
    if "sslmode" not in query and "supabase" in (parts.hostname or ""):
        query["sslmode"] = "require"
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _build_engine() -> Optional[AsyncEngine]:
    url, connect_args = normalise_url(settings.database_url)
    if not url:
        return None

    pooling: dict[str, Any]
    if connect_args.get("statement_cache_size") == 0:
        # Already behind an external pooler — a second pool on top of it just
        # holds server-side connections hostage.
        pooling = {"poolclass": NullPool}
    else:
        pooling = {
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_timeout": settings.db_pool_timeout,
            "pool_recycle": settings.db_pool_recycle,
            "pool_pre_ping": True,
        }

    return create_async_engine(
        url,
        echo=settings.db_echo,
        connect_args=connect_args,
        **pooling,
    )


def get_engine() -> Optional[AsyncEngine]:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = _build_engine()
        if _engine is not None:
            _sessionmaker = async_sessionmaker(
                _engine, expire_on_commit=False, class_=AsyncSession
            )
    return _engine


def db_enabled() -> bool:
    """True only once a connection has actually been proven to work."""
    return _available


def db_configured() -> bool:
    return bool(settings.database_url)


async def init_db() -> bool:
    """Open the pool and verify it. Never raises — the DB is optional."""
    global _available
    if not db_configured():
        logger.info("DATABASE_URL not set; running without durable persistence.")
        _available = False
        return False

    engine = get_engine()
    if engine is None:
        _available = False
        return False

    try:
        from sqlalchemy import text as sa_text

        async with engine.connect() as conn:
            version = await conn.scalar(sa_text("select version()"))
        _available = True
        logger.info("Database connected: %s", str(version)[:60])
        return True
    except Exception as exc:  # noqa: BLE001
        _available = False
        logger.warning(
            "Database unavailable (%s: %s); continuing without persistence.",
            type(exc).__name__,
            str(exc)[:200],
        )
        return False


async def shutdown_db() -> None:
    global _engine, _sessionmaker, _available
    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Engine dispose failed: %s", exc)
    _engine = None
    _sessionmaker = None
    _available = False


@asynccontextmanager
async def get_session() -> AsyncIterator[Optional[AsyncSession]]:
    """Yield a session, or ``None`` when persistence is off.

    Callers write ``async with get_session() as s: if s is None: return`` — the
    persistence layer is best-effort by design, so no call site should have to
    branch on configuration before it can ask for a session.
    """
    if not _available or _sessionmaker is None:
        yield None
        return

    session = _sessionmaker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def healthcheck() -> dict[str, Any]:
    if not db_configured():
        return {"configured": False, "ok": False, "detail": "DATABASE_URL not set"}
    engine = get_engine()
    if engine is None:
        return {"configured": True, "ok": False, "detail": "engine not built"}
    try:
        from sqlalchemy import text as sa_text

        async with engine.connect() as conn:
            await conn.scalar(sa_text("select 1"))
        return {"configured": True, "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:200]}
