"""Retry, timeout and circuit-breaking for outbound calls.

The original code wrapped every provider in a bare ``try/except Exception`` and
turned failures into strings that were then fed to an LLM. That hides outages,
retries nothing, and lets one dead upstream (etrain.info being slow) stall a
whole request for minutes.

Here each provider gets:
  * a bounded timeout,
  * exponential backoff with jitter on transient errors,
  * a circuit breaker so a persistently-failing upstream is skipped instead of
    re-tried on every single request.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way callers should surface."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


class CircuitOpenError(ProviderError):
    def __init__(self, provider: str, retry_in: float) -> None:
        super().__init__(provider, f"circuit open, retrying in {retry_in:.0f}s")
        self.retry_in = retry_in


@dataclass
class CircuitBreaker:
    """Classic three-state breaker: closed -> open -> half-open."""

    name: str
    failure_threshold: int = 4
    recovery_seconds: float = 60.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_seconds:
            return "half-open"
        return "open"

    def check(self) -> None:
        if self.state == "open":
            assert self._opened_at is not None
            remaining = self.recovery_seconds - (time.monotonic() - self._opened_at)
            raise CircuitOpenError(self.name, max(remaining, 0.0))

    def record_success(self) -> None:
        if self._failures or self._opened_at:
            logger.info("Circuit %s recovered.", self.name)
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit %s OPEN after %d failures; pausing calls for %.0fs.",
                self.name,
                self._failures,
                self.recovery_seconds,
            )
        elif self.state == "half-open":
            self._opened_at = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "failures": self._failures}


_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    if name not in _BREAKERS:
        _BREAKERS[name] = CircuitBreaker(name=name, **kwargs)
    return _BREAKERS[name]


def all_breakers() -> list[dict[str, Any]]:
    return [b.snapshot() for b in _BREAKERS.values()]


def _result_count(result: Any) -> int | None:
    """Best-effort 'how much did this provider actually return'."""
    if isinstance(result, (list, tuple, set)):
        return len(result)
    if isinstance(result, dict):
        for key in ("results", "options", "flights", "trains", "buses", "hotels"):
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                return len(value)
    return None


def _audit(
    provider: str,
    operation: str,
    *,
    ok: bool,
    started: float,
    attempts: int,
    result: Any = None,
    error: str | None = None,
) -> None:
    """Write one row to the provider audit log, off the critical path.

    Imported lazily: ``app.db`` pulls in SQLAlchemy, and this module is imported
    by every provider, so a module-level import would make the dependency
    mandatory for a project whose database is explicitly optional.
    """
    try:
        from app.core.context import get_trip_id, spawn_background
        from app.db.repository import record_provider_call
        from app.db.session import db_enabled

        if not db_enabled():
            return

        spawn_background(
            record_provider_call(
                get_trip_id(),
                provider,
                ok=ok,
                operation=operation,
                duration_ms=int((time.perf_counter() - started) * 1000),
                result_count=_result_count(result),
                error=error,
                meta={"attempts": attempts},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Provider audit skipped for %s: %s", provider, exc)


def resilient(
    provider: str,
    *,
    timeout: float = 30.0,
    retries: int = 2,
    base_delay: float = 0.6,
    failure_threshold: int = 4,
    recovery_seconds: float = 60.0,
):
    """Wrap an async provider call with timeout + backoff + circuit breaker."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        breaker = get_breaker(
            provider,
            failure_threshold=failure_threshold,
            recovery_seconds=recovery_seconds,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            breaker.check()
            last_exc: Exception | None = None
            started = time.perf_counter()

            for attempt in range(retries + 1):
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout
                    )
                    breaker.record_success()
                    _audit(
                        provider,
                        func.__name__,
                        ok=True,
                        started=started,
                        attempts=attempt + 1,
                        result=result,
                    )
                    return result
                except asyncio.TimeoutError as exc:
                    last_exc = exc
                    logger.warning(
                        "%s timed out after %.0fs (attempt %d/%d).",
                        provider,
                        timeout,
                        attempt + 1,
                        retries + 1,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "%s failed (attempt %d/%d): %s",
                        provider,
                        attempt + 1,
                        retries + 1,
                        exc,
                    )

                if attempt < retries:
                    delay = base_delay * (2**attempt) + random.uniform(0, 0.4)
                    await asyncio.sleep(delay)

            breaker.record_failure()
            _audit(
                provider,
                func.__name__,
                ok=False,
                started=started,
                attempts=retries + 1,
                error=str(last_exc) or "unknown failure",
            )
            raise ProviderError(provider, str(last_exc) or "unknown failure")

        return wrapper

    return decorator


async def gather_providers(
    tasks: dict[str, Awaitable[Any]],
    *,
    overall_timeout: float = 90.0,
) -> dict[str, Any]:
    """Run independent provider calls concurrently, isolating failures.

    Returns ``{name: {"ok": bool, "data"|"error": ...}}``. One dead upstream can
    no longer take the whole itinerary down — a design property the original
    sequential code did not have.
    """
    names = list(tasks)
    pending = [asyncio.ensure_future(tasks[n]) for n in names]

    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=overall_timeout,
        )
    except asyncio.TimeoutError:
        # A per-provider timeout can still be beaten by N providers each
        # finishing just under theirs. This is the outer budget that keeps one
        # request from running unboundedly long.
        logger.warning(
            "Provider fan-out exceeded its %.0fs budget; cancelling stragglers.",
            overall_timeout,
        )
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    out: dict[str, Any] = {}
    for name, task in zip(names, pending):
        if task.cancelled():
            out[name] = {
                "ok": False,
                "error": f"timed out after {overall_timeout:.0f}s (overall budget)",
            }
            continue
        error = task.exception()
        if error is not None:
            out[name] = {"ok": False, "error": str(error)}
        else:
            out[name] = {"ok": True, "data": task.result()}
    return out
