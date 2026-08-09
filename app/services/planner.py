"""Trip planning service — the one place that drives the graph.

Both the HTTP layer and the Kafka worker call in here, so "start a trip" and
"resume a trip" behave identically no matter what triggered them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.types import Command

from app.core.bus import emit, get_status, set_status
from app.core.cache import cache
from app.core.context import current_trip_id, drain_background
from app.core.events import EventType, TripStatus, result_key
from app.db import repository
from app.graph.builder import get_graph
from app.graph.checkpointer import thread_config
from app.graph.state import new_state

logger = logging.getLogger(__name__)

_RESULT_TTL = 86400


class TripNotFound(LookupError):
    pass


def _coerce_interrupt(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def interrupt_from_result(result: Any) -> Optional[dict[str, Any]]:
    """Read a pending interrupt off an ``ainvoke`` return value.

    LangGraph >= 0.3 puts an ``__interrupt__`` key in the returned state;
    0.2.x does not surface it there at all. Handle both.
    """
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return _coerce_interrupt(getattr(interrupts[0], "value", interrupts[0]))


def interrupt_from_snapshot(snapshot: Any) -> Optional[dict[str, Any]]:
    """Read a pending interrupt off a state snapshot — works on every version."""
    if snapshot is None:
        return None
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            return _coerce_interrupt(getattr(item, "value", item))
    return None


async def pending_interrupt(graph: Any, config: dict[str, Any], result: Any = None):
    """The authoritative check for 'is this graph waiting on a human?'."""
    found = interrupt_from_result(result)
    if found is not None:
        return found
    try:
        return interrupt_from_snapshot(await graph.aget_state(config))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read state for interrupt detection: %s", exc)
        return None


def has_fatal_error(state: dict[str, Any]) -> bool:
    """Did the graph stop because it gave up, rather than because it finished?

    Reaching END is how *every* run terminates, including the fail-fast route
    for a request whose origin and destination could not be determined. Without
    this distinction such a trip is reported — and stored — as ``completed``
    with no itinerary attached.
    """
    return any(
        isinstance(err, dict) and err.get("fatal")
        for err in (state.get("errors") or [])
    )


def _fatal_message(state: dict[str, Any]) -> str:
    for err in state.get("errors") or []:
        if isinstance(err, dict) and err.get("fatal"):
            return str(err.get("message") or "Planning failed.")
    return "Planning failed."


def summarise_state(state: dict[str, Any]) -> dict[str, Any]:
    """The client-facing view of a trip. Messages stay internal."""
    return {
        "trip_id": state.get("trip_id"),
        "brief": state.get("brief") or {},
        "transport_strategy": state.get("transport_strategy"),
        "transport_options": state.get("transport_options") or {},
        "transport_summary": state.get("transport_summary"),
        "arrival_time": state.get("arrival_time"),
        "weather": state.get("weather") or [],
        "hotels": state.get("hotels"),
        "hotels_summary": state.get("hotels_summary"),
        "itinerary": state.get("itinerary"),
        "itinerary_meta": state.get("itinerary_meta") or {},
        "budget": state.get("budget") or {},
        "provider_health": state.get("provider_health") or {},
        "errors": state.get("errors") or [],
        "revision_count": state.get("revision_count", 0),
    }


async def _run(trip_id: str, payload: Any) -> dict[str, Any]:
    """Invoke or resume the graph and normalise the outcome."""
    graph = await get_graph()
    config = thread_config(trip_id)

    # Every provider call fanned out below this point inherits the context, so
    # the audit log knows which trip it belongs to without any provider taking
    # a trip_id argument.
    current_trip_id.set(trip_id)

    await set_status(trip_id, TripStatus.RUNNING)
    await emit(trip_id, EventType.TRIP_STARTED, "Planning started")

    await repository.record_status(trip_id, TripStatus.RUNNING.value)

    try:
        result = await graph.ainvoke(payload, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph run failed for %s", trip_id)
        await set_status(trip_id, TripStatus.FAILED, error=str(exc))
        await emit(trip_id, EventType.TRIP_FAILED, str(exc))
        await repository.record_status(trip_id, TripStatus.FAILED.value, error=str(exc))
        raise

    pending = await pending_interrupt(graph, config, result)
    snapshot = summarise_state(result)

    if pending:
        status = TripStatus.AWAITING_REVIEW
        snapshot["pending_review"] = pending
        await emit(trip_id, EventType.AWAITING_REVIEW, pending.get("question", "Review needed"))
    elif has_fatal_error(snapshot):
        status = TripStatus.FAILED
        await emit(trip_id, EventType.TRIP_FAILED, _fatal_message(snapshot))
    else:
        status = TripStatus.COMPLETED
        await emit(trip_id, EventType.TRIP_COMPLETED, "Planning complete")

    snapshot["status"] = status.value
    # Redis stays the fast path (24h TTL, serves the polling UI); Postgres is
    # the durable one. Writing both means an eviction costs latency, not data.
    await cache.set_json(result_key(trip_id), snapshot, ttl=_RESULT_TTL)
    await set_status(trip_id, status, has_pending_review=bool(pending))
    # Let the fire-and-forget provider-audit writes land before we claim the
    # trip is persisted; otherwise a caller reading /providers immediately
    # afterwards can see an incomplete log.
    await drain_background()
    await repository.persist_result(trip_id, snapshot)
    return snapshot


async def set_queued(trip_id: str, request: dict[str, Any]) -> None:
    """Record intent before the job leaves the API.

    Without this a client that polls ``GET /trips/{id}`` between the 202 and the
    worker picking the job up would get a 404 for its own trip.
    """
    await set_status(trip_id, TripStatus.QUEUED)
    # Durable first: if the process dies between here and Kafka, the request
    # still exists as a record rather than vanishing with the Redis key.
    await repository.record_requested(trip_id, request)
    await cache.set_json(
        result_key(trip_id),
        {
            "trip_id": trip_id,
            "status": TripStatus.QUEUED.value,
            "brief": {
                k: v for k, v in request.items()
                if k in {"origin", "destination", "start_date", "end_date", "persons"}
            },
        },
        ttl=_RESULT_TTL,
    )
    await emit(trip_id, EventType.TRIP_REQUESTED, "Trip queued")


async def start_trip(trip_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Begin planning. ``trip_id`` doubles as the checkpointer thread id."""
    brief_seed = {
        k: v
        for k, v in {
            "origin": request.get("origin"),
            "destination": request.get("destination"),
            "start_date": request.get("start_date"),
            "end_date": request.get("end_date"),
            "persons": request.get("persons"),
            "preferences": request.get("preferences"),
            "budget_level": request.get("budget_level"),
            "travel_class": request.get("travel_class"),
            "special_requirements": request.get("special_requirements"),
        }.items()
        if v not in (None, "", [])
    }

    state = new_state(
        trip_id=trip_id,
        raw_request=request.get("query") or _synthesise_query(brief_seed),
        brief=brief_seed,
    )
    # The row must exist *before* the graph runs, or the provider calls the
    # graph makes have no trip to attach to and land in the audit log
    # unattributed. No-ops when set_queued already inserted it.
    await repository.record_requested(trip_id, request)
    await emit(trip_id, EventType.TRIP_REQUESTED, "Trip requested", **brief_seed)
    return await _run(trip_id, state)


async def resume_trip(
    trip_id: str, action: str, feedback: Optional[str] = None
) -> dict[str, Any]:
    """Resume a graph paused at an ``interrupt()``."""
    graph = await get_graph()
    config = thread_config(trip_id)

    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.created_at:
        raise TripNotFound(f"No checkpoint found for trip {trip_id}")
    if not snapshot.next:
        raise ValueError(f"Trip {trip_id} has already finished; nothing to resume.")

    await emit(trip_id, EventType.TRIP_RESUMED, f"Resuming with '{action}'")
    return await _run(trip_id, Command(resume={"action": action, "feedback": feedback}))


async def get_trip(trip_id: str) -> dict[str, Any]:
    """Read a trip's current state, preferring the live checkpoint."""
    graph = await get_graph()
    config = thread_config(trip_id)

    try:
        snapshot = await graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Checkpoint read failed for %s: %s", trip_id, exc)
        snapshot = None

    if snapshot is not None and snapshot.values:
        view = summarise_state(snapshot.values)
        pending = interrupt_from_snapshot(snapshot)
        if pending:
            view["pending_review"] = pending
            view["status"] = TripStatus.AWAITING_REVIEW.value
        elif snapshot.next:
            view["status"] = TripStatus.RUNNING.value
        elif has_fatal_error(view):
            view["status"] = TripStatus.FAILED.value
        else:
            # An empty ``next`` does NOT prove the run finished. Between two
            # steps the checkpoint momentarily has state but no scheduled task,
            # and inferring "completed" from that reported a finished trip —
            # with no itinerary — seconds after submission, before the worker
            # had done any work.
            #
            # The status the planner recorded is authoritative while the run is
            # in flight; structural inference is only the fallback for old trips
            # whose status key has since expired.
            recorded = (await get_status(trip_id) or {}).get("status")
            if recorded in (TripStatus.QUEUED.value, TripStatus.RUNNING.value):
                view["status"] = recorded
            else:
                view["status"] = TripStatus.COMPLETED.value
        view["next_nodes"] = list(snapshot.next or ())
        return view

    cached_result = await cache.get_json(result_key(trip_id))
    if cached_result:
        return cached_result

    # Last resort, and the reason the database exists: checkpoints get pruned
    # and the Redis result key expires after 24h, but a completed plan should
    # still be retrievable a month later.
    try:
        stored = await repository.get_trip(trip_id)
    except repository.PersistenceUnavailable:
        stored = None
    if stored:
        stored["source"] = "database"
        return stored

    raise TripNotFound(f"Trip {trip_id} not found")


async def get_history(trip_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Checkpoint history — the audit trail the old MemorySaver could not keep."""
    graph = await get_graph()
    config = thread_config(trip_id)
    out: list[dict[str, Any]] = []
    try:
        async for snapshot in graph.aget_state_history(config, limit=limit):
            out.append(
                {
                    "checkpoint_id": snapshot.config.get("configurable", {}).get(
                        "checkpoint_id"
                    ),
                    "created_at": snapshot.created_at,
                    "next": list(snapshot.next or ()),
                    "step_keys": sorted(
                        k for k in (snapshot.values or {}) if k != "messages"
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("History read failed for %s: %s", trip_id, exc)
    return out


def _synthesise_query(brief: dict[str, Any]) -> str:
    parts = [
        f"Plan a trip from {brief.get('origin', '?')} to {brief.get('destination', '?')}"
    ]
    if brief.get("start_date"):
        parts.append(f"from {brief['start_date']}")
    if brief.get("end_date"):
        parts.append(f"to {brief['end_date']}")
    if brief.get("persons"):
        parts.append(f"for {brief['persons']} person(s)")
    if brief.get("preferences"):
        parts.append(f"interested in {', '.join(brief['preferences'])}")
    return " ".join(parts) + "."
