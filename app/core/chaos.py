"""Fault injection, so "it degrades gracefully" is a claim you can check.

The whole project is built on the premise that any single upstream may be down:
Redis falls back to an in-process cache, Kafka falls back to inline execution,
a dead scraper is meant to cost you one mode rather than the trip. That premise
was never actually tested. It was tested by *accident* — Groq's daily token cap
ran out mid-session and revealed that a rate-limited model returned a blank
itinerary while the trip still reported "completed".

Waiting for real outages to find those is a poor strategy, so this makes them
reproducible. Every provider already funnels through ``@resilient``, which gives
one place to inject:

    CHAOS_PROVIDERS=etrain,google_flights   # these fail
    CHAOS_MODE=error                        # error | timeout | empty
    CHAOS_PROBABILITY=1.0                   # 0..1

Off unless explicitly configured, and it refuses to arm itself when
``APP_ENV=production`` — a fault injector nobody remembers enabling is
indistinguishable from a bug.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class ChaosError(RuntimeError):
    """Raised in place of a real provider call when chaos is armed."""


def _targets() -> set[str]:
    raw = (settings.chaos_providers or "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def armed() -> bool:
    if (settings.app_env or "").lower() == "production":
        return False
    return bool(_targets()) and settings.chaos_probability > 0


def status() -> dict[str, Any]:
    """Surfaced on /health so an armed injector is never a surprise."""
    return {
        "armed": armed(),
        "providers": sorted(_targets()),
        "mode": settings.chaos_mode,
        "probability": settings.chaos_probability,
    }


def should_fail(provider: str) -> bool:
    if not armed():
        return False
    targets = _targets()
    if "all" not in targets and provider.lower() not in targets:
        return False
    return random.random() < settings.chaos_probability


async def apply(provider: str) -> Optional[Any]:
    """Inject the configured fault for ``provider``.

    Returns a substitute result for ``empty`` mode, raises for the others, and
    returns ``None`` when nothing should be injected.
    """
    if not should_fail(provider):
        return None

    mode = (settings.chaos_mode or "error").lower()
    logger.warning("CHAOS: injecting %s into %s", mode, provider)

    if mode == "timeout":
        # Outlast any provider's own timeout so the caller's handling runs.
        await asyncio.sleep(3600)
        return None
    if mode == "empty":
        # The subtler failure: a provider that answers, successfully, with
        # nothing. This is the one that used to be misreported as real data.
        return {"count": 0, "results": [], "chaos": True}
    raise ChaosError(f"chaos: simulated {provider} failure")
