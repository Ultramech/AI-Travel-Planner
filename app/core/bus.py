"""Kafka producer/consumer plus the event-publishing façade.

Why Kafka is here, and where it genuinely earns its place
--------------------------------------------------------
Planning a trip takes 30-120 seconds: Selenium scrapes, three LLM round-trips,
several paid APIs. Doing that inside a synchronous HTTP request (what the
Streamlit version did) means the request holds a worker for minutes, cannot be
retried, and loses everything if the process restarts.

So the topology is:

  ``trip.requests``  work queue. The API validates, persists intent and returns
                     202 immediately. Workers consume and run the LangGraph.
                     Partitioned by ``trip_id`` so all work for one trip is
                     ordered and lands on one consumer. Scaling = adding
                     workers, no code change.

  ``trip.events``    durable event log of every node transition. Powers the
                     live progress feed, and doubles as an audit/replay trail
                     (which node failed, how long each provider took).

  ``trip.dlq``       jobs that exhausted their retries, kept for inspection
                     rather than silently dropped.

Redis pub/sub sits alongside Kafka purely for SSE fan-out — it is the low
latency path to a connected browser; Kafka is the durable one.

If Kafka is not running the module degrades to "disabled" and the API runs the
graph in a background task instead, so the project still works on a laptop with
nothing but Python installed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from app.config import settings
from app.core.cache import cache
from app.core.events import (
    EventType,
    TripEvent,
    history_key,
    status_key,
    stream_channel,
)

logger = logging.getLogger(__name__)

_EVENT_HISTORY_TTL = 86400
_MAX_HISTORY = 400


class KafkaBus:
    """Lazily-connected aiokafka producer with a hard "optional" contract."""

    def __init__(self) -> None:
        self._producer: Any = None
        self._enabled = settings.kafka_enabled
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._producer is not None

    async def start(self) -> None:
        if not settings.kafka_enabled:
            logger.info("Kafka disabled by configuration.")
            self._enabled = False
            return
        async with self._lock:
            if self._producer is not None:
                return
            try:
                from aiokafka import AIOKafkaProducer

                producer = AIOKafkaProducer(
                    bootstrap_servers=settings.kafka_bootstrap_servers,
                    value_serializer=lambda v: v.encode("utf-8"),
                    key_serializer=lambda v: v.encode("utf-8") if v else None,
                    acks="all",  # durability over latency for a work queue
                    enable_idempotence=True,
                    request_timeout_ms=15000,
                    linger_ms=20,
                )
                await asyncio.wait_for(producer.start(), timeout=10)
                self._producer = producer
                self._enabled = True
                logger.info(
                    "Kafka producer connected: %s", settings.kafka_bootstrap_servers
                )
            except Exception as exc:  # noqa: BLE001
                self._enabled = False
                self._producer = None
                # A producer that failed to bootstrap still holds a client and
                # background tasks; dropping the reference leaks them
                # ("Unclosed AIOKafkaProducer").
                try:
                    await producer.stop()
                except Exception:  # noqa: BLE001
                    pass
                logger.warning(
                    "Kafka unavailable (%s). Falling back to in-process execution.",
                    exc,
                )

    async def stop(self) -> None:
        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:  # noqa: BLE001
                pass
            self._producer = None

    async def send(self, topic: str, value: str, key: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        try:
            await self._producer.send_and_wait(topic, value=value, key=key)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kafka send to %s failed: %s", topic, exc)
            return False


bus = KafkaBus()


# ---------------------------------------------------------------------------
# Event publishing — the single entry point used by graph nodes
# ---------------------------------------------------------------------------

async def publish_event(event: TripEvent) -> None:
    """Fan an event out to Kafka (durable) and Redis (live) and keep history."""
    payload = event.to_json()

    # 1. Durable log.
    await bus.send(settings.kafka_topic_events, payload, key=event.trip_id)

    # 2. Live fan-out to any SSE subscriber.
    await cache.publish(stream_channel(event.trip_id), payload)

    # 3. Replayable history so a client that connects late still sees progress.
    try:
        redis = cache._redis  # noqa: SLF001 - deliberate internal use
        if redis is not None:
            key = history_key(event.trip_id)
            await redis.rpush(key, payload)
            await redis.ltrim(key, -_MAX_HISTORY, -1)
            await redis.expire(key, _EVENT_HISTORY_TTL)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Event history append failed: %s", exc)


async def emit(
    trip_id: str,
    type_: EventType,
    message: str = "",
    node: Optional[str] = None,
    **payload: Any,
) -> None:
    await publish_event(
        TripEvent(
            trip_id=trip_id, type=type_, message=message, node=node, payload=payload
        )
    )


async def get_event_history(trip_id: str) -> list[str]:
    try:
        redis = cache._redis  # noqa: SLF001
        if redis is None:
            return []
        return await redis.lrange(history_key(trip_id), 0, -1)
    except Exception:  # noqa: BLE001
        return []


async def set_status(trip_id: str, status: str, **extra: Any) -> None:
    await cache.set_json(
        status_key(trip_id), {"status": status, **extra}, ttl=_EVENT_HISTORY_TTL
    )


async def get_status(trip_id: str) -> Optional[dict[str, Any]]:
    return await cache.get_json(status_key(trip_id))


# ---------------------------------------------------------------------------
# SSE source
# ---------------------------------------------------------------------------

async def subscribe(trip_id: str) -> AsyncIterator[str]:
    """Yield event JSON for a trip: replayed history first, then live."""
    for past in await get_event_history(trip_id):
        yield past

    pubsub = cache.pubsub()
    if pubsub is None:
        # No Redis: fall back to polling the history list.
        seen = len(await get_event_history(trip_id))
        while True:
            await asyncio.sleep(1.0)
            history = await get_event_history(trip_id)
            for item in history[seen:]:
                yield item
            seen = len(history)

    await pubsub.subscribe(stream_channel(trip_id))
    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=15.0
            )
            if message and message.get("data"):
                yield message["data"]
            else:
                yield ""  # heartbeat keeps proxies from killing the connection
    finally:
        try:
            await pubsub.unsubscribe(stream_channel(trip_id))
            await pubsub.aclose()
        except Exception:  # noqa: BLE001
            pass
