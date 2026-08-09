"""Checkpointer lifecycle.

What was wrong before
---------------------
``build_trip_graph()`` created ``MemorySaver()`` **inside the function**, the
Streamlit script called ``build_trip_graph()`` on every rerun, and
``plan_my_trip`` generated a fresh ``uuid4()`` thread_id for every invocation.

So the checkpointer:
  * lived in a process that restarts on every UI interaction,
  * was a brand-new empty object each time,
  * and was keyed by an ID that was never reused.

Nothing was ever resumed, nothing survived a restart, and human-in-the-loop
could not work at all — ``interrupt()`` requires a durable checkpoint keyed by a
stable thread_id to have anything to resume *into*.

What it does now
----------------
* One checkpointer per process, opened at startup and closed at shutdown.
* ``thread_id == trip_id``, chosen by the client and stable across processes.
* SQLite by default (survives restarts, zero infrastructure), Redis when the
  official saver is installed, memory only for tests.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_stack: Optional[AsyncExitStack] = None
_checkpointer: Any = None
_active_backend: Optional[str] = None


def active_backend() -> Optional[str]:
    """Which backend is *actually* in use, not which one was requested.

    These differ whenever a backend fails and falls back, and reporting the
    configured value in /health turned that fallback into a silent lie — the
    endpoint claimed 'postgres' while state was going to a local SQLite file.
    """
    return _active_backend


async def _setup_shared(saver: Any) -> None:
    """Create the checkpoint tables, tolerating a concurrent creator.

    ``setup()`` is not safe to run from several processes at once: the API and
    every worker call it at startup, and on a shared database they race to
    insert the same migration row, so all but one crash with

        duplicate key value violates unique constraint "checkpoint_migrations_pkey"

    That is a race, not a broken database — the loser's tables exist and are
    perfectly usable. Treat it as success, then prove it with a real query
    rather than assuming.
    """
    try:
        await saver.setup()
        return
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        raced = "duplicate key" in message or "already exists" in message
        if not raced:
            raise
        logger.info("Checkpoint schema was created concurrently; verifying.")

    # Prove the schema is actually usable before accepting the race explanation.
    await saver.aget_tuple({"configurable": {"thread_id": "__setup_probe__"}})


async def get_checkpointer() -> Any:
    """Return the process-wide checkpointer, opening it on first use."""
    global _stack, _checkpointer, _active_backend
    if _checkpointer is not None:
        return _checkpointer

    _stack = AsyncExitStack()
    backend = settings.checkpointer_backend

    if backend == "postgres":
        # The only backend where two processes genuinely share one source of
        # truth. SQLite below is durable but single-node: a second worker on
        # another machine cannot see, let alone resume, a trip paused by the
        # first. With Postgres, `--scale worker=N` is a real scaling story.
        if not settings.database_url:
            logger.warning(
                "checkpointer_backend=postgres but DATABASE_URL is empty; "
                "falling back to SQLite."
            )
            backend = "sqlite"
        else:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                from app.db.session import checkpointer_dsn

                _checkpointer = await _stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(checkpointer_dsn())
                )
                await _setup_shared(_checkpointer)
                _active_backend = "postgres"
                logger.info("Checkpointer: Postgres (shared across workers)")
                return _checkpointer
            except ImportError:
                logger.warning(
                    "langgraph-checkpoint-postgres not installed; falling back to SQLite."
                )
                backend = "sqlite"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Postgres checkpointer failed (%s); falling back to SQLite.", exc
                )
                backend = "sqlite"

    if backend == "redis":
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver  # type: ignore

            _checkpointer = await _stack.enter_async_context(
                AsyncRedisSaver.from_conn_string(settings.redis_url)
            )
            await _checkpointer.asetup()
            _active_backend = "redis"
            logger.info("Checkpointer: Redis (%s)", settings.redis_url)
            return _checkpointer
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-redis not installed; falling back to SQLite."
            )
            backend = "sqlite"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis checkpointer failed (%s); falling back to SQLite.", exc)
            backend = "sqlite"

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = settings.checkpoint_path
            _checkpointer = await _stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )
            _active_backend = "sqlite"
            logger.info("Checkpointer: SQLite (%s)", path)
            return _checkpointer
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQLite checkpointer failed (%s); falling back to memory.", exc)

    from langgraph.checkpoint.memory import MemorySaver

    _checkpointer = MemorySaver()
    _active_backend = "memory"
    logger.warning(
        "Checkpointer: in-memory. State will NOT survive a restart and "
        "human-in-the-loop only works within this process."
    )
    return _checkpointer


async def close_checkpointer() -> None:
    global _stack, _checkpointer, _active_backend
    _active_backend = None
    if _stack is not None:
        try:
            await _stack.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Checkpointer close failed: %s", exc)
    _stack = None
    _checkpointer = None


def thread_config(trip_id: str, **extra: Any) -> dict[str, Any]:
    """Build the run config. ``thread_id`` is the trip id — stable and reusable."""
    return {
        "configurable": {"thread_id": trip_id, **extra},
        "recursion_limit": settings.graph_recursion_limit,
    }
