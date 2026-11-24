"""HTTP surface.

The whole application used to be a Streamlit script — no API, no way for
anything else to consume it, and a 60-120 s plan blocking the UI thread.

Endpoints
---------
    POST   /api/v1/trips                  submit a plan (202 by default)
    GET    /api/v1/trips                  list & search past trips (durable)
    GET    /api/v1/trips/{id}             current state
    POST   /api/v1/trips/{id}/resume      answer a human-review interrupt
    GET    /api/v1/trips/{id}/stream      SSE progress feed
    GET    /api/v1/trips/{id}/history     checkpoint audit trail
    GET    /api/v1/trips/{id}/rationale   why this option was recommended
    GET    /api/v1/trips/{id}/providers   provider-call audit log
    GET    /api/v1/trips/{id}/itinerary   markdown export
    GET    /api/v1/providers/stats        provider reliability, aggregated
    GET    /api/v1/graph                  mermaid topology
    GET    /health  /health/live  /health/ready
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    HealthResponse,
    PlanTripRequest,
    ResumeTripRequest,
    TripAccepted,
    TripResponse,
)
from app.config import settings
from app.core.bus import bus, subscribe
from app.core.cache import cache
from app.core.events import EventType, PlanRequestMessage, TripStatus, new_id
from app.core.resilience import all_breakers
from app.db import repository
from app.db.session import healthcheck as db_healthcheck
from app.services import planner

logger = logging.getLogger(__name__)

router = APIRouter()
VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    from datetime import datetime, timezone

    from app.graph.checkpointer import active_backend as _active_checkpointer
    from app.providers.flights_router import _order, amadeus_available

    redis_ok = await cache.ping()
    flight_sources = _order()
    db = await db_healthcheck()
    dependencies = {
        "redis": "ok" if redis_ok else "unavailable (using in-process cache)",
        "kafka": "ok" if bus.enabled else "unavailable (running inline)",
        "database": (
            "ok"
            if db.get("ok")
            else ("not configured (no durable history)" if not db.get("configured")
                  else f"unavailable: {db.get('detail')}")
        ),
        # Report what is actually running, not what was asked for. A backend
        # that failed and fell back must not be able to hide behind config.
        "checkpointer": _active_checkpointer() or settings.checkpointer_backend,
        "checkpointer_configured": settings.checkpointer_backend,
        "llm_configured": settings.llm_configured,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.resolved_llm_model,
        "weather_configured": True,  # Open-Meteo, no key needed
        "weather_source": "open-meteo",
        "hotels_configured": bool(settings.serpapi_key),
        # Flights never need a key: Google Flights is the keyless default and
        # Amadeus is an optional upgrade to bookable offers.
        "flights": {
            "sources": flight_sources,
            "primary": flight_sources[0] if flight_sources else None,
            "amadeus_configured": amadeus_available(),
            "requires_api_key": False,
        },
        "scraping": "enabled" if settings.scraper_enabled else "disabled",
    }
    degraded = not settings.llm_configured or any(
        b["state"] != "closed" for b in all_breakers()
    )
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=VERSION,
        dependencies=dependencies,
        circuit_breakers=all_breakers(),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/health/live", tags=["health"])
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
async def readiness(response: Response) -> dict[str, object]:
    """Ready only when we can actually plan — i.e. the LLM is configured."""
    ready = settings.llm_configured
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "reason": None if ready else f"{settings.llm_key_name.upper()} not set"}


# ---------------------------------------------------------------------------
# Trips
# ---------------------------------------------------------------------------

@router.post(
    "/api/v1/trips",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=None,
    tags=["trips"],
)
async def create_trip(
    request: PlanTripRequest, background: BackgroundTasks, response: Response
):
    """Submit a trip.

    Async (default): validate, enqueue on Kafka, return 202 immediately. The
    client follows ``/stream`` for progress. This is the important difference
    from the Streamlit version, where the browser held an open connection for
    the full 60-120 s of scraping and LLM calls.

    Sync: run inline and return the first checkpoint. Useful for curl and tests.
    """
    trip_id = new_id("trip")
    payload = request.to_payload()

    if not request.async_mode:
        result = await planner.start_trip(trip_id, payload)
        response.status_code = status.HTTP_200_OK
        return TripResponse(**result)

    await planner.set_queued(trip_id, payload)

    message = PlanRequestMessage(trip_id=trip_id, payload=payload)
    queued = await bus.send(settings.kafka_topic_requests, message.to_json(), key=trip_id)

    if not queued:
        # Kafka unavailable — degrade to an in-process background task rather
        # than failing the request.
        logger.info("Kafka unavailable; running trip %s in-process.", trip_id)
        background.add_task(_run_inline, trip_id, payload)

    return TripAccepted(
        trip_id=trip_id,
        status=TripStatus.QUEUED.value,
        message="Trip queued. Follow the stream for progress."
        + ("" if queued else " (running in-process: Kafka unavailable)"),
        stream_url=f"/api/v1/trips/{trip_id}/stream",
        status_url=f"/api/v1/trips/{trip_id}",
    )


async def _run_inline(trip_id: str, payload: dict) -> None:
    try:
        await planner.start_trip(trip_id, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Inline trip %s failed", trip_id)


@router.get("/api/v1/trips", tags=["trips"])
async def list_trips(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    origin: str | None = Query(default=None),
    destination: str | None = Query(default=None),
    q: str | None = Query(default=None, description="Free text over origin/destination/request"),
    start_after: date | None = Query(default=None),
    start_before: date | None = Query(default=None),
):
    """Browse and search every trip ever planned.

    This is the endpoint the 24-hour Redis TTL made impossible. It reads the
    database directly, so it returns trips whose checkpoints and cache entries
    are long gone.
    """
    try:
        rows, total = await repository.list_trips(
            limit=limit,
            offset=offset,
            status=status_filter,
            origin=origin,
            destination=destination,
            q=q,
            start_after=start_after,
            start_before=start_before,
        )
    except repository.PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "returned": len(rows),
        "trips": rows,
    }


@router.get("/api/v1/trips/{trip_id}", response_model=TripResponse, tags=["trips"])
async def get_trip(trip_id: str) -> TripResponse:
    try:
        return TripResponse(**await planner.get_trip(trip_id))
    except planner.TripNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/trips/{trip_id}/resume", response_model=TripResponse, tags=["trips"]
)
async def resume_trip(trip_id: str, request: ResumeTripRequest) -> TripResponse:
    """Answer the pending human-review interrupt and continue the graph.

    This is what makes human-in-the-loop real: the graph paused at
    ``interrupt()``, its state is durably checkpointed under ``thread_id ==
    trip_id``, and this call resumes it — possibly in a different process, long
    after the original request finished.
    """
    try:
        result = await planner.resume_trip(trip_id, request.action, request.feedback)
        return TripResponse(**result)
    except planner.TripNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/v1/trips/{trip_id}/stream", tags=["trips"])
async def stream_trip(trip_id: str) -> EventSourceResponse:
    """Server-sent events: replayed history, then live progress."""

    async def generator():
        try:
            async for payload in subscribe(trip_id):
                if not payload:
                    yield {"event": "ping", "data": "{}"}
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                yield {"event": event.get("type", "message"), "data": payload}
                if event.get("type") in {
                    EventType.TRIP_COMPLETED.value,
                    EventType.TRIP_FAILED.value,
                }:
                    break
        except asyncio.CancelledError:
            logger.debug("SSE client disconnected from %s", trip_id)
            raise

    return EventSourceResponse(generator())


@router.get("/api/v1/trips/{trip_id}/history", tags=["trips"])
async def trip_history(trip_id: str, limit: int = Query(default=20, ge=1, le=100)):
    return {"trip_id": trip_id, "checkpoints": await planner.get_history(trip_id, limit)}


@router.get("/api/v1/trips/{trip_id}/rationale", tags=["trips"])
async def trip_rationale(trip_id: str):
    """Why this trip was recommended — the decision, opened up.

    An LLM recommendation you cannot audit is indistinguishable from one that
    is wrong, and over a day of running this system the model double-counted a
    group fare, read a 24-hour train as a 30-minute hop, and asserted that
    flights were unavailable on a route nothing had searched.

    So every recommendation carries its own working: each option with its real
    door-to-door cost and time, the weights applied for this traveller's budget
    level, which modes were skipped and *why*, which prices are estimates rather
    than quotes, and whether the model's prose actually matches the arithmetic.
    """
    try:
        trip = await planner.get_trip(trip_id)
    except planner.TripNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rationale = trip.get("decision_rationale")
    if not rationale:
        raise HTTPException(
            status_code=404,
            detail="No recommendation has been made for this trip yet.",
        )
    return {"trip_id": trip_id, **rationale}


@router.get("/api/v1/trips/{trip_id}/providers", tags=["trips"])
async def trip_provider_calls(
    trip_id: str, limit: int = Query(default=200, ge=1, le=1000)
):
    """Every outbound provider call this trip made, with latency and outcome."""
    try:
        return {
            "trip_id": trip_id,
            "calls": await repository.get_provider_calls(trip_id, limit),
        }
    except repository.PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/v1/providers/stats", tags=["ops"])
async def providers_stats(since_hours: int = Query(default=24, ge=1, le=8760)):
    """Aggregate reliability per provider — success rate and latency.

    The point of the audit log: 'the bus scraper is flaky' becomes a number you
    can set a circuit-breaker threshold from.
    """
    try:
        return {"since_hours": since_hours, "providers": await repository.provider_stats(since_hours)}
    except repository.PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/v1/trips/{trip_id}/itinerary", tags=["trips"])
async def export_itinerary(trip_id: str) -> Response:
    """Markdown export — the old UI offered a PDF download button whose payload
    was the literal string 'This would be the PDF content in a real app'."""
    try:
        trip = await planner.get_trip(trip_id)
    except planner.TripNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not trip.get("itinerary"):
        raise HTTPException(status_code=409, detail="Itinerary not generated yet")

    brief = trip.get("brief") or {}
    budget = trip.get("budget") or {}
    body = [
        f"# {brief.get('origin')} → {brief.get('destination')}",
        f"_{brief.get('start_date')} to {brief.get('end_date')} · "
        f"{brief.get('persons')} traveller(s)_\n",
        "## Transport\n",
        trip.get("transport_summary") or "_Not available._",
        "\n## Accommodation\n",
        trip.get("hotels_summary") or "_Not available._",
        "\n## Itinerary\n",
        trip["itinerary"],
    ]
    if budget.get("total_inr"):
        body.append(
            f"\n## Estimated cost\n\nTotal ₹{budget['total_inr']:,.0f} "
            f"(₹{budget.get('per_person_inr', 0):,.0f} per person)"
        )

    filename = f"{brief.get('origin', 'trip')}-{brief.get('destination', 'plan')}.md"
    return Response(
        content="\n".join(body),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

@router.get("/api/v1/graph", tags=["ops"])
async def graph_topology() -> dict[str, str]:
    from app.graph.builder import export_mermaid

    return {"format": "mermaid", "diagram": await export_mermaid()}


@router.delete("/api/v1/cache/{namespace}", tags=["ops"])
async def invalidate_cache(namespace: str) -> dict[str, object]:
    removed = await cache.invalidate_prefix(f"tp:v1:{namespace}")
    return {"namespace": namespace, "keys_removed": removed}
