"""Kafka worker: consumes plan jobs and runs the graph.

Run with:  python -m app.workers.consumer

Why this process exists
-----------------------
Planning takes 30-120 s. Doing it inside the HTTP request means:
  * the client holds a connection open for minutes,
  * a deploy or crash loses the work entirely,
  * you cannot scale planning capacity independently of API capacity.

With the work on ``trip.requests``:
  * the API answers in milliseconds,
  * jobs survive a worker restart (the offset is only committed after success),
  * adding capacity is ``docker compose up --scale worker=4`` — the consumer
    group rebalances partitions automatically,
  * partitioning by ``trip_id`` keeps all work for one trip ordered on one
    consumer.

Delivery semantics: at-least-once. Offsets are committed only after the graph
run finishes, so a crash mid-plan replays the job. That is safe here because
the LangGraph checkpointer makes a replay resume from the last completed node
rather than redo the whole trip, and the Redis cache absorbs repeated provider
calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from typing import Any

from app.config import settings
from app.core.bus import bus, emit, set_status
from app.core.cache import cache
from app.core.events import EventType, PlanRequestMessage, TripStatus
from app.db.session import init_db, shutdown_db
from app.graph.builder import get_graph, shutdown_graph
from app.logging_conf import configure_logging
from app.services import planner

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
_shutdown = asyncio.Event()


async def handle_message(raw: str) -> None:
    """Process one job. Raises on failure so the caller can retry or DLQ it."""
    message = PlanRequestMessage.model_validate_json(raw)
    logger.info("Planning trip %s (attempt %d)", message.trip_id, message.attempt)
    await planner.start_trip(message.trip_id, message.payload)
    logger.info("Trip %s finished or paused for review", message.trip_id)


async def send_to_dlq(raw: str, error: str) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw[:500]}
    payload["_dlq_error"] = error
    await bus.send(settings.kafka_topic_dlq, json.dumps(payload),
                   key=payload.get("trip_id"))
    logger.error("Sent to DLQ: %s", error)


async def run_worker() -> None:
    configure_logging()

    if not settings.kafka_enabled:
        logger.error("KAFKA_ENABLED is false — nothing for this worker to consume.")
        return

    await cache.connect()
    await init_db()  # before the graph, so a Postgres checkpointer can open
    await bus.start()
    await get_graph()

    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        settings.kafka_topic_requests,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        value_deserializer=lambda v: v.decode("utf-8"),
        # Manual commit: only acknowledge a job once it is actually done.
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_interval_ms=600_000,  # a plan can legitimately take minutes
        session_timeout_ms=45_000,
    )

    try:
        await consumer.start()
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot reach Kafka at %s: %s",
                     settings.kafka_bootstrap_servers, exc)
        await _cleanup(None)
        return

    logger.info(
        "Worker listening on '%s' as group '%s'",
        settings.kafka_topic_requests, settings.kafka_consumer_group,
    )

    try:
        async for record in consumer:
            if _shutdown.is_set():
                break
            trip_id: Any = record.key
            try:
                await handle_message(record.value)
                await consumer.commit()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job failed for %s", trip_id)
                try:
                    message = PlanRequestMessage.model_validate_json(record.value)
                    attempt = message.attempt + 1
                except Exception:  # noqa: BLE001
                    message, attempt = None, MAX_ATTEMPTS

                if message is not None and attempt < MAX_ATTEMPTS:
                    message.attempt = attempt
                    logger.info("Requeueing %s (attempt %d)", message.trip_id, attempt)
                    await bus.send(
                        settings.kafka_topic_requests,
                        message.to_json(),
                        key=message.trip_id,
                    )
                else:
                    await send_to_dlq(record.value, str(exc))
                    if trip_id:
                        await set_status(str(trip_id), TripStatus.FAILED, error=str(exc))
                        await emit(str(trip_id), EventType.TRIP_FAILED, str(exc))
                # Commit either way: the job has been requeued or DLQ'd, so
                # replaying this offset would only duplicate it.
                await consumer.commit()
    finally:
        await _cleanup(consumer)


async def _cleanup(consumer: Any) -> None:
    logger.info("Worker shutting down…")
    if consumer is not None:
        with_suppress = getattr(consumer, "stop", None)
        if with_suppress:
            try:
                await consumer.stop()
            except Exception:  # noqa: BLE001
                pass
    await bus.stop()
    await cache.close()
    await shutdown_graph()
    await shutdown_db()


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown.set)
        except NotImplementedError:  # pragma: no cover - Windows
            pass

    try:
        loop.run_until_complete(run_worker())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
