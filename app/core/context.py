"""Ambient request context.

The provider audit log needs to know which trip a call belongs to, but the
providers themselves take domain arguments (origin, date, persons) and should
not grow a ``trip_id`` parameter purely for bookkeeping — that would push a
persistence concern into every signature in ``app/providers/``.

A context variable carries it instead. ``asyncio`` copies the context into each
task, so the value set once in the planner is visible inside every concurrent
provider call fanned out beneath it, and two trips running in the same worker
never see each other's id.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

current_trip_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trip_id", default=None
)

# Audit writes are fire-and-forget so a slow database never adds latency to a
# provider call. Tasks are held here because asyncio only keeps a weak
# reference — without this they can be garbage-collected mid-flight.
_background: set[asyncio.Task[Any]] = set()


def get_trip_id() -> Optional[str]:
    return current_trip_id.get()


def spawn_background(coro: Any) -> None:
    """Run a best-effort side effect without blocking the caller."""
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:  # no running loop (sync context, e.g. some tests)
        coro.close()
        return
    _background.add(task)
    task.add_done_callback(_background.discard)


async def drain_background(timeout: float = 5.0) -> None:
    """Let in-flight audit writes finish before the process exits."""
    if not _background:
        return
    pending = list(_background)
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.debug("Dropped %d unfinished background writes at shutdown.", len(pending))
