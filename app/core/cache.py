"""Redis-backed cache with graceful in-process fallback.

Design notes
------------
* Every upstream call in this project is slow (Selenium: 10-40s, SerpAPI: 1-3s,
  Amadeus: 1-2s, WikiVoyage + embeddings: 10-60s) and most of them are billed
  per request. Caching is not an optimisation here, it is what makes the system
  usable at all.
* Keys are built from a *normalised* argument tuple so that "delhi" and
  " Delhi " hit the same entry.
* Single-flight locking prevents a cache stampede: when 50 users ask for
  "Goa hotels" at once we do one upstream call, not 50.
* Negative results are cached too (shorter TTL) so a city with no trains does
  not re-run a 40-second Selenium scrape on every request.
* If Redis is down the app keeps working against a bounded in-process dict.
  Degraded, never dead.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# "Nothing found" comes in two flavours and they deserve different retry
# windows:
#
#   transient  — the scrape threw, the site changed, the network blipped. Retry
#                soon, because the answer is probably wrong.
#   settled    — the provider answered clearly that it has no service on this
#                route/date. Retry slowly, because re-asking costs a 30-second
#                scrape to be told the same thing.
#
# Conflating them made a far-future bus search (operators only publish ~30 days
# ahead, so the answer is reliably "none") re-scrape for 30 seconds every 5
# minutes, forever.
_NEGATIVE_TTL = 300  # transient emptiness: retry in 5 minutes
_SETTLED_EMPTY_TTL = 3600  # a definitive "no service here": retry in an hour

# Refresh an entry in the background once it is this far through its TTL. 0.75
# leaves a quarter of the lifetime as headroom, so a slow 30-second scrape has
# time to finish before anyone would have hit an expired key.
_REFRESH_RATIO = 0.75
_LOCK_TIMEOUT = 90  # seconds a single-flight lock may be held
_LOCK_POLL = 0.25


class _LocalCache:
    """Bounded LRU used when Redis is unavailable."""

    def __init__(self, maxsize: int = 2048) -> None:
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[str]:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, payload = item
        if expires_at < time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return payload

    def set(self, key: str, value: str, ttl: int) -> None:
        self._data[key] = (time.time() + ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()


class CacheClient:
    """Thin async wrapper over redis.asyncio with a local fallback."""

    def __init__(self) -> None:
        self._redis: Any = None
        self._local = _LocalCache()
        self._healthy = False
        self._connect_lock = asyncio.Lock()
        self._loop: Any = None

    async def connect(self) -> None:
        """Open the Redis connection, rebuilding it if the loop changed.

        A redis.asyncio client binds to the event loop that created it. The
        service runs one loop for its lifetime so this never matters in
        production — but anything that runs several loops in one process (the
        test suite creates one per test, and the graph does the same during
        checkpointer teardown) inherits a client whose loop is closed. Every
        operation then fails quietly and the caller silently falls through to
        stale data, which is far worse than an obvious error.
        """
        if not settings.cache_enabled:
            logger.info("Cache disabled by configuration.")
            return
        async with self._connect_lock:
            running = asyncio.get_running_loop()
            if self._redis is not None and self._loop is not running:
                logger.debug("Event loop changed; rebuilding the Redis client.")
                stale, self._redis, self._healthy = self._redis, None, False
                try:
                    await stale.aclose()
                except Exception:  # noqa: BLE001 - the old loop is likely gone
                    pass
            if self._redis is not None:
                return
            self._loop = running
            try:
                import redis.asyncio as aioredis

                client = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=5,
                    health_check_interval=30,
                )
                await client.ping()
                self._redis = client
                self._healthy = True
                logger.info("Redis cache connected: %s", settings.redis_url)
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                self._healthy = False
                logger.warning(
                    "Redis unavailable (%s). Falling back to in-process cache.", exc
                )

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._redis = None
            self._healthy = False
            self._loop = None

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def ping(self) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            self._healthy = True
        except Exception:  # noqa: BLE001
            self._healthy = False
        return self._healthy

    # --- raw string ops ----------------------------------------------------

    async def get(self, key: str) -> Optional[str]:
        if not settings.cache_enabled:
            return None
        if self._redis is not None:
            try:
                return await self._redis.get(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis GET failed (%s); using local cache.", exc)
                self._healthy = False
        return self._local.get(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        if not settings.cache_enabled:
            return
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis SET failed (%s); using local cache.", exc)
                self._healthy = False
        self._local.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete(key)
            except Exception:  # noqa: BLE001
                pass
        self._local.delete(key)

    async def invalidate_prefix(self, prefix: str) -> int:
        """Delete every key under a namespace. Uses SCAN, never KEYS."""
        removed = 0
        if self._redis is not None:
            try:
                async for key in self._redis.scan_iter(match=f"{prefix}*", count=200):
                    await self._redis.delete(key)
                    removed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis invalidate failed: %s", exc)
        for key in [k for k in list(self._local._data) if k.startswith(prefix)]:
            self._local.delete(key)
            removed += 1
        return removed

    # --- JSON ops ----------------------------------------------------------

    async def get_json(self, key: str) -> Any:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            await self.delete(key)
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Value for %s is not JSON serialisable: %s", key, exc)
            return
        await self.set(key, payload, ttl)

    # --- single-flight lock ------------------------------------------------

    async def acquire_lock(self, key: str) -> bool:
        if self._redis is None:
            return True  # local cache: no cross-process stampede to prevent
        try:
            return bool(await self._redis.set(key, "1", ex=_LOCK_TIMEOUT, nx=True))
        except Exception:  # noqa: BLE001
            return True

    async def release_lock(self, key: str) -> None:
        await self.delete(key)

    # --- pub/sub fan-out ---------------------------------------------------

    async def publish(self, channel: str, message: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.publish(channel, message)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis publish failed: %s", exc)

    def pubsub(self):
        if self._redis is None:
            return None
        return self._redis.pubsub()


cache = CacheClient()


# ---------------------------------------------------------------------------
# Key building
# ---------------------------------------------------------------------------

def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in sorted(value.items())}
    return value


def build_key(namespace: str, *args: Any, **kwargs: Any) -> str:
    payload = json.dumps(
        {"a": _normalise(list(args)), "k": _normalise(kwargs)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"tp:v1:{namespace}:{digest}"


# Keys whose list value carries the actual results of a provider call. A
# payload where all of these are empty is a "found nothing" answer however much
# metadata surrounds it.
_RESULT_KEYS = ("trains", "buses", "flights", "hotels", "cabs", "options", "results")


def _is_empty(value: Any) -> bool:
    """Is this a 'nothing found' result, and therefore only worth caching briefly?

    The subtle case, and a real bug before this: the scrapers do not return a
    bare empty list when they find nothing. They return a populated dict —
    ``{"count": 0, "buses": [], "note": "no route"}`` — which is truthy and has
    a non-zero length, so it was cached under the **full** TTL. A route that
    momentarily returned nothing, or a scrape broken by a site change, was then
    pinned as "no trains available" for six hours.

    A short negative TTL means the next request retries soon, which is what you
    want for an answer that is probably transient.
    """
    if value is None:
        return True
    if isinstance(value, (list, str)) and len(value) == 0:
        return True
    if isinstance(value, dict):
        if len(value) == 0:
            return True
        if value.get("error"):
            return True
        # An explicit count of zero settles it.
        if value.get("count") == 0:
            return True
        present = [k for k in _RESULT_KEYS if k in value]
        if present and all(not value.get(k) for k in present):
            return True
    return False


def _is_settled_empty(value: Any) -> bool:
    """Did the provider *answer* that it has nothing, rather than merely fail?

    A provider that returns a structured payload with an explicit zero count and
    an explanatory note has genuinely looked and found nothing. A bare empty
    list, or a payload carrying an ``error``, is far more likely to be a
    transient failure worth retrying soon.
    """
    if not isinstance(value, dict) or not value:
        return False
    if value.get("error"):
        return False
    return value.get("count") == 0 and bool(value.get("note"))


def _effective_ttl(result: Any, ttl: int, negative_ttl: int) -> int:
    if _is_settled_empty(result):
        return max(negative_ttl, _SETTLED_EMPTY_TTL)
    if _is_empty(result):
        return negative_ttl
    return ttl


async def _store(key: str, result: Any, ttl: int, negative_ttl: int) -> None:
    """Write a cache entry, stamped so staleness can be judged on read."""
    await cache.set_json(
        key, {"v": result, "t": time.time()}, _effective_ttl(result, ttl, negative_ttl)
    )


def _is_stale(entry: Any, ttl: int, refresh_ratio: float) -> bool:
    if not isinstance(entry, dict) or refresh_ratio <= 0:
        return False
    written = entry.get("t")
    if not isinstance(written, (int, float)):
        # Written before stamping existed: refresh once so it gains a stamp.
        return True
    return (time.time() - written) > (ttl * refresh_ratio)


async def _spawn_refresh(
    func: Callable[..., Awaitable[Any]],
    key: str,
    ttl: int,
    negative_ttl: int,
    args: tuple,
    kwargs: dict,
) -> None:
    """Re-run a provider in the background and overwrite its cache entry.

    Deliberately fire-and-forget: the caller already has a usable answer, so a
    refresh must never add latency, and a refresh that fails must never surface
    as an error on a request that was otherwise served fine.
    """
    refresh_lock = f"{key}:refresh"
    if not await cache.acquire_lock(refresh_lock):
        return  # another reader is already refreshing this key

    async def _run() -> None:
        try:
            result = await func(*args, **kwargs)
            await _store(key, result, ttl, negative_ttl)
            logger.debug("cache REFRESHED %s", key)
        except Exception as exc:  # noqa: BLE001
            # Keep serving the stale value; it is better than nothing.
            logger.debug("background refresh failed for %s: %s", key, exc)
        finally:
            await cache.release_lock(refresh_lock)

    from app.core.context import spawn_background

    spawn_background(_run())


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def cached(
    namespace: str,
    ttl: int,
    *,
    negative_ttl: int = _NEGATIVE_TTL,
    single_flight: bool = True,
    refresh_ratio: float = _REFRESH_RATIO,
):
    """Cache an async function's JSON-serialisable return value.

    ``skip_cache=True`` may be passed by callers to force a refresh — used when
    a user gives feedback like "show me different hotels".

    **Stale-while-revalidate.** Once an entry is older than
    ``ttl * refresh_ratio`` it is still returned immediately, but a refresh is
    kicked off in the background. Without this, every entry eventually expires
    while someone is waiting on it, and that unlucky request pays the full
    30-second scrape. With it, a route that is queried regularly is refreshed
    off the critical path and users effectively always hit a warm cache.

    The refresh is guarded by its own short lock, so N concurrent readers of a
    stale key trigger exactly one re-scrape, not N.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            skip = bool(kwargs.pop("skip_cache", False))
            key = build_key(namespace, *args, **kwargs)

            if not skip:
                hit = await cache.get_json(key)
                if hit is not None:
                    logger.debug("cache HIT  %s", key)
                    if _is_stale(hit, ttl, refresh_ratio):
                        await _spawn_refresh(func, key, ttl, negative_ttl, args, kwargs)
                    return hit.get("v")  # type: ignore[return-value]

            lock_key = f"{key}:lock"
            got_lock = True
            if single_flight and not skip:
                got_lock = await cache.acquire_lock(lock_key)
                if not got_lock:
                    # Someone else is already fetching. Wait briefly for them.
                    deadline = time.time() + 30
                    while time.time() < deadline:
                        await asyncio.sleep(_LOCK_POLL)
                        hit = await cache.get_json(key)
                        if hit is not None:
                            logger.debug("cache HIT (after wait) %s", key)
                            return hit.get("v")  # type: ignore[return-value]
                    logger.debug("single-flight wait timed out for %s", key)

            try:
                logger.debug("cache MISS %s", key)
                result = await func(*args, **kwargs)
                await _store(key, result, ttl, negative_ttl)
                return result
            finally:
                if single_flight and got_lock and not skip:
                    await cache.release_lock(lock_key)

        wrapper.cache_namespace = namespace  # type: ignore[attr-defined]
        return wrapper

    return decorator
