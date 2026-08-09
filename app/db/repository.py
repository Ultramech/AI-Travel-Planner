"""Data access. The only module that knows SQL exists.

Every public function here is **best-effort**: if persistence is off or the
database is briefly unreachable, it logs and returns rather than raising. A
planning run must not fail because the audit log was unavailable — the run is
the product, the record is bookkeeping.

The read paths (`list_trips`, `get_trip`) are the exception: a user explicitly
asking for their history should be told the truth when it cannot be served, so
those raise :class:`PersistenceUnavailable`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import Select, case, func, or_, select

from app.db.models import CostSnapshot, Itinerary, ProviderCall, Trip
from app.db.session import db_enabled, get_session

logger = logging.getLogger(__name__)


class PersistenceUnavailable(RuntimeError):
    """Raised by read paths when there is no database to read from."""


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


async def record_requested(trip_id: str, request: dict[str, Any]) -> None:
    """Insert the trip the moment the API accepts it.

    Written *before* the job is handed to Kafka, so a request is never lost in
    the window between the 202 and a worker picking it up — including when the
    worker never does.
    """
    if not db_enabled():
        return
    try:
        async with get_session() as session:
            if session is None:
                return
            existing = await session.get(Trip, trip_id)
            if existing is not None:
                return
            session.add(
                Trip(
                    id=trip_id,
                    status="queued",
                    origin=request.get("origin"),
                    destination=request.get("destination"),
                    start_date=_as_date(request.get("start_date")),
                    end_date=_as_date(request.get("end_date")),
                    persons=_as_int(request.get("persons")),
                    budget_level=request.get("budget_level"),
                    travel_class=request.get("travel_class"),
                    preferences=list(request.get("preferences") or []),
                    special_requirements=request.get("special_requirements"),
                    raw_request=request.get("query"),
                    brief={
                        k: v
                        for k, v in request.items()
                        if k
                        in {
                            "origin",
                            "destination",
                            "start_date",
                            "end_date",
                            "persons",
                            "preferences",
                            "budget_level",
                            "travel_class",
                        }
                        and v not in (None, "", [])
                    },
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_requested(%s) failed: %s", trip_id, exc)


async def record_status(
    trip_id: str, status: str, error: Optional[str] = None
) -> None:
    if not db_enabled():
        return
    try:
        async with get_session() as session:
            if session is None:
                return
            trip = await session.get(Trip, trip_id)
            if trip is None:
                return
            trip.status = status
            if error:
                trip.error = error[:4000]
            if status == "completed":
                trip.completed_at = datetime.now(timezone.utc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_status(%s) failed: %s", trip_id, exc)


async def persist_result(trip_id: str, snapshot: dict[str, Any]) -> None:
    """Write the outcome of a graph run: trip row, itinerary revision, costs.

    Idempotent per revision. An at-least-once Kafka redelivery re-runs the graph
    and lands here again; the unique ``(trip_id, revision)`` indexes mean that
    updates the existing revision instead of stacking duplicates.
    """
    if not db_enabled():
        return
    try:
        async with get_session() as session:
            if session is None:
                return

            trip = await session.get(Trip, trip_id)
            if trip is None:
                trip = Trip(id=trip_id, status=snapshot.get("status") or "running")
                session.add(trip)

            brief = snapshot.get("brief") or {}
            trip.status = snapshot.get("status") or trip.status
            trip.origin = brief.get("origin") or trip.origin
            trip.destination = brief.get("destination") or trip.destination
            trip.start_date = _as_date(brief.get("start_date")) or trip.start_date
            trip.end_date = _as_date(brief.get("end_date")) or trip.end_date
            trip.persons = _as_int(brief.get("persons")) or trip.persons
            trip.budget_level = brief.get("budget_level") or trip.budget_level
            trip.travel_class = brief.get("travel_class") or trip.travel_class
            if brief:
                trip.brief = {**(trip.brief or {}), **brief}

            trip.transport_strategy = snapshot.get("transport_strategy")
            trip.transport_options = snapshot.get("transport_options") or {}
            trip.transport_summary = snapshot.get("transport_summary")
            trip.recommended_transport = snapshot.get("recommended_transport")
            trip.arrival_time = snapshot.get("arrival_time")
            trip.weather = snapshot.get("weather")
            trip.hotels = snapshot.get("hotels")
            trip.hotels_summary = snapshot.get("hotels_summary")
            trip.provider_health = snapshot.get("provider_health") or {}
            trip.errors = snapshot.get("errors") or []
            revision = _as_int(snapshot.get("revision_count")) or 0
            trip.revision_count = revision
            if snapshot.get("status") == "completed":
                trip.completed_at = datetime.now(timezone.utc)

            # --- itinerary revision ---------------------------------------
            itinerary_text = snapshot.get("itinerary")
            if itinerary_text:
                existing = await session.scalar(
                    select(Itinerary).where(
                        Itinerary.trip_id == trip_id, Itinerary.revision == revision
                    )
                )
                if existing is None:
                    session.add(
                        Itinerary(
                            trip_id=trip_id,
                            revision=revision,
                            content=itinerary_text,
                            meta=snapshot.get("itinerary_meta") or {},
                        )
                    )
                else:
                    existing.content = itinerary_text
                    existing.meta = snapshot.get("itinerary_meta") or {}

            # --- cost snapshot --------------------------------------------
            budget = snapshot.get("budget") or {}
            if any(
                budget.get(k) for k in ("transport_inr", "hotel_inr", "onground_inr", "total_inr")
            ):
                existing_cost = await session.scalar(
                    select(CostSnapshot).where(
                        CostSnapshot.trip_id == trip_id,
                        CostSnapshot.revision == revision,
                    )
                )
                target = existing_cost or CostSnapshot(
                    trip_id=trip_id, revision=revision
                )
                target.currency = budget.get("currency") or "INR"
                target.transport_cost = _num(budget.get("transport_inr"))
                target.hotel_cost = _num(budget.get("hotel_inr"))
                target.onground_cost = _num(budget.get("onground_inr"))
                target.total_cost = _num(budget.get("total_inr"))
                target.per_person_cost = _num(budget.get("per_person_inr"))
                target.breakdown = budget
                if existing_cost is None:
                    session.add(target)
    except Exception as exc:  # noqa: BLE001
        logger.warning("persist_result(%s) failed: %s", trip_id, exc)


async def record_provider_call(
    trip_id: Optional[str],
    provider: str,
    *,
    ok: bool = True,
    operation: Optional[str] = None,
    duration_ms: Optional[int] = None,
    cache_hit: bool = False,
    result_count: Optional[int] = None,
    error: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Append one row to the provider audit log."""
    if not db_enabled():
        return
    try:
        async with get_session() as session:
            if session is None:
                return
            # A provider call can arrive before the trip row exists (or with no
            # trip at all, from a warm-up). Null the FK rather than lose the row.
            linked: Optional[str] = None
            if trip_id:
                linked = trip_id if await session.get(Trip, trip_id) else None
            session.add(
                ProviderCall(
                    trip_id=linked,
                    provider=provider,
                    operation=operation,
                    ok=ok,
                    cache_hit=cache_hit,
                    duration_ms=duration_ms,
                    result_count=result_count,
                    error=(error or None) and str(error)[:2000],
                    meta=meta or {},
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("record_provider_call(%s) failed: %s", provider, exc)


# --------------------------------------------------------------------------
# Read paths
# --------------------------------------------------------------------------


def _require_db() -> None:
    if not db_enabled():
        raise PersistenceUnavailable(
            "Trip history requires a database. Set DATABASE_URL and run "
            "`alembic upgrade head`."
        )


def _apply_filters(
    stmt: Select,
    *,
    status: Optional[str],
    origin: Optional[str],
    destination: Optional[str],
    q: Optional[str],
    start_after: Optional[date],
    start_before: Optional[date],
) -> Select:
    if status:
        stmt = stmt.where(Trip.status == status)
    if origin:
        stmt = stmt.where(func.lower(Trip.origin) == origin.lower())
    if destination:
        stmt = stmt.where(func.lower(Trip.destination) == destination.lower())
    if start_after:
        stmt = stmt.where(Trip.start_date >= start_after)
    if start_before:
        stmt = stmt.where(Trip.start_date <= start_before)
    if q:
        pattern = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(Trip.origin).like(pattern),
                func.lower(Trip.destination).like(pattern),
                func.lower(func.coalesce(Trip.raw_request, "")).like(pattern),
            )
        )
    return stmt


async def list_trips(
    *,
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    q: Optional[str] = None,
    start_after: Optional[date] = None,
    start_before: Optional[date] = None,
) -> tuple[list[dict[str, Any]], int]:
    """Page through trip history, newest first. Returns ``(rows, total)``."""
    _require_db()
    async with get_session() as session:
        if session is None:
            raise PersistenceUnavailable("No database session available.")

        filters = dict(
            status=status,
            origin=origin,
            destination=destination,
            q=q,
            start_after=start_after,
            start_before=start_before,
        )
        total = await session.scalar(
            _apply_filters(select(func.count()).select_from(Trip), **filters)
        )
        stmt = _apply_filters(select(Trip), **filters)
        stmt = stmt.order_by(Trip.created_at.desc()).limit(limit).offset(offset)
        rows = (await session.scalars(stmt)).all()
        return [_trip_summary(t) for t in rows], int(total or 0)


async def get_trip(trip_id: str) -> Optional[dict[str, Any]]:
    """Full stored record — the copy that outlives the Redis TTL."""
    _require_db()
    async with get_session() as session:
        if session is None:
            raise PersistenceUnavailable("No database session available.")
        trip = await session.get(Trip, trip_id)
        return _trip_detail(trip) if trip else None


async def get_provider_calls(
    trip_id: str, limit: int = 200
) -> list[dict[str, Any]]:
    _require_db()
    async with get_session() as session:
        if session is None:
            raise PersistenceUnavailable("No database session available.")
        rows = (
            await session.scalars(
                select(ProviderCall)
                .where(ProviderCall.trip_id == trip_id)
                .order_by(ProviderCall.created_at.desc())
                .limit(limit)
            )
        ).all()
        return [
            {
                "provider": r.provider,
                "operation": r.operation,
                "ok": r.ok,
                "cache_hit": r.cache_hit,
                "duration_ms": r.duration_ms,
                "result_count": r.result_count,
                "error": r.error,
                "ts": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


async def provider_stats(since_hours: int = 24) -> list[dict[str, Any]]:
    """Aggregate provider reliability — the payoff for keeping the audit log.

    ``since_hours`` genuinely bounds the window. An unbounded lifetime average
    would be useless for the question this answers: *is this provider healthy
    right now* — a scraper broken since this morning stays hidden behind months
    of past success.
    """
    _require_db()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    async with get_session() as session:
        if session is None:
            raise PersistenceUnavailable("No database session available.")
        stmt = (
            select(
                ProviderCall.provider,
                func.count().label("calls"),
                func.sum(case((ProviderCall.ok, 1), else_=0)).label("ok"),
                func.avg(ProviderCall.duration_ms).label("avg_ms"),
                func.max(ProviderCall.duration_ms).label("max_ms"),
            )
            .where(ProviderCall.created_at >= cutoff)
            .group_by(ProviderCall.provider)
            .order_by(func.count().desc())
        )
        rows = (await session.execute(stmt)).all()
        out = []
        for r in rows:
            calls = int(r.calls or 0)
            ok = int(r.ok or 0)
            out.append(
                {
                    "provider": r.provider,
                    "calls": calls,
                    "ok": ok,
                    "failed": calls - ok,
                    "success_rate": round(ok / calls, 3) if calls else None,
                    "avg_ms": round(float(r.avg_ms), 1) if r.avg_ms is not None else None,
                    "max_ms": int(r.max_ms) if r.max_ms is not None else None,
                }
            )
        return out


def _trip_summary(t: Trip) -> dict[str, Any]:
    latest_cost = t.cost_snapshots[-1] if t.cost_snapshots else None
    return {
        "trip_id": t.id,
        "status": t.status,
        "origin": t.origin,
        "destination": t.destination,
        "start_date": t.start_date.isoformat() if t.start_date else None,
        "end_date": t.end_date.isoformat() if t.end_date else None,
        "persons": t.persons,
        "total_inr": float(latest_cost.total_cost) if latest_cost and latest_cost.total_cost else None,
        "has_itinerary": bool(t.itineraries),
        "revision_count": t.revision_count,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
    }


def _trip_detail(t: Trip) -> dict[str, Any]:
    latest = t.itineraries[-1] if t.itineraries else None
    return {
        **_trip_summary(t),
        "brief": t.brief or {},
        "raw_request": t.raw_request,
        "preferences": t.preferences or [],
        "transport_summary": t.transport_summary,
        "transport_options": t.transport_options or {},
        "recommended_transport": t.recommended_transport,
        "arrival_time": t.arrival_time,
        "weather": t.weather or [],
        "hotels": t.hotels,
        "hotels_summary": t.hotels_summary,
        "itinerary": latest.content if latest else None,
        "itinerary_meta": latest.meta if latest else {},
        "itinerary_revisions": [
            {"revision": i.revision, "created_at": i.created_at.isoformat() if i.created_at else None}
            for i in t.itineraries
        ],
        "budget": (t.cost_snapshots[-1].breakdown if t.cost_snapshots else {}),
        "cost_history": [
            {
                "revision": c.revision,
                "total_inr": float(c.total_cost) if c.total_cost else None,
                "per_person_inr": float(c.per_person_cost) if c.per_person_cost else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in t.cost_snapshots
        ],
        "provider_health": t.provider_health or {},
        "errors": t.errors or [],
        "error": t.error,
    }
