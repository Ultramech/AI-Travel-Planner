"""The domain schema.

Four tables, each earning its place:

``trips``            the system of record. One row per planning request, from
                     the moment the API accepts it to whatever it finally
                     became. Previously this lived only in Redis under a 24h
                     TTL — a cache being used as a result store, where an
                     eviction silently destroyed a finished plan.

``itineraries``      the generated plan, **versioned**. A trip can be revised up
                     to three times; keeping every revision means "show me what
                     you proposed before I asked for something cheaper" is a
                     query rather than a lost message.

``provider_calls``   one row per outbound call to Amadeus/SerpAPI/a scraper:
                     latency, success, whether the cache served it. This is what
                     turns "the scrapers are flaky" into a number, and it is the
                     evidence base for tuning the cache TTLs and the circuit
                     breaker thresholds.

``cost_snapshots``   the computed budget at a point in time, kept per revision
                     so the price history of a trip is reconstructable.

JSONB rather than further normalisation for the provider payloads: they are
third-party shapes we do not control, read whole, and never join on. Columns
that *are* queried — status, origin, destination, dates — are real typed
columns with indexes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

# Portable JSON: JSONB on Postgres, plain JSON elsewhere (SQLite in tests).
JSONType = JSONB().with_variant(
    __import__("sqlalchemy").JSON(), "sqlite"
)


class Trip(Base, TimestampMixin):
    __tablename__ = "trips"

    # The trip id is chosen by the API and is also the LangGraph thread_id, so
    # this primary key is the join between the domain record and its execution
    # history. Not a surrogate integer — that would need a second lookup.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # --- the request ------------------------------------------------------
    origin: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    destination: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    start_date: Mapped[Optional[date]] = mapped_column(Date, index=True)
    end_date: Mapped[Optional[date]] = mapped_column(Date)
    persons: Mapped[Optional[int]] = mapped_column(Integer)
    budget_level: Mapped[Optional[str]] = mapped_column(String(32))
    travel_class: Mapped[Optional[str]] = mapped_column(String(32))
    preferences: Mapped[list[str]] = mapped_column(
        JSONType, nullable=False, server_default=text("'[]'")
    )
    special_requirements: Mapped[Optional[str]] = mapped_column(Text)
    raw_request: Mapped[Optional[str]] = mapped_column(Text)
    brief: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, server_default=text("'{}'")
    )

    # --- what the graph produced -----------------------------------------
    transport_strategy: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)
    transport_options: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, server_default=text("'{}'")
    )
    transport_summary: Mapped[Optional[str]] = mapped_column(Text)
    recommended_transport: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)
    arrival_time: Mapped[Optional[str]] = mapped_column(String(64))
    weather: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONType)
    hotels: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)
    hotels_summary: Mapped[Optional[str]] = mapped_column(Text)

    # --- diagnostics ------------------------------------------------------
    provider_health: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, server_default=text("'{}'")
    )
    errors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONType, nullable=False, server_default=text("'[]'")
    )
    error: Mapped[Optional[str]] = mapped_column(Text)
    revision_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    itineraries: Mapped[list["Itinerary"]] = relationship(
        back_populates="trip",
        cascade="all, delete-orphan",
        order_by="Itinerary.revision",
        lazy="selectin",
    )
    provider_calls: Mapped[list["ProviderCall"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", lazy="noload"
    )
    cost_snapshots: Mapped[list["CostSnapshot"]] = relationship(
        back_populates="trip",
        cascade="all, delete-orphan",
        order_by="CostSnapshot.revision",
        lazy="selectin",
    )

    __table_args__ = (
        CheckConstraint(
            "end_date IS NULL OR start_date IS NULL OR end_date >= start_date",
            name="end_after_start",
        ),
        # The list endpoint's default ordering. Postgres can walk this backwards
        # for "newest first" without a sort step.
        Index("ix_trips_created_at_desc", text("created_at DESC")),
        Index("ix_trips_status_created_at", "status", text("created_at DESC")),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Trip {self.id} {self.origin}->{self.destination} {self.status}>"


class Itinerary(Base, TimestampMixin):
    """A generated plan. Versioned — revisions are kept, never overwritten."""

    __tablename__ = "itineraries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str] = mapped_column(
        ForeignKey("trips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, server_default=text("'{}'")
    )

    trip: Mapped[Trip] = relationship(back_populates="itineraries")

    __table_args__ = (
        # One row per (trip, revision): makes re-running a revision idempotent
        # rather than silently duplicating the plan on an at-least-once redelivery.
        Index("uq_itineraries_trip_revision", "trip_id", "revision", unique=True),
    )


class ProviderCall(Base):
    """Audit log of every outbound provider call.

    Deliberately *not* using TimestampMixin — this table is append-only and
    high-volume, so it carries one timestamp, not two.
    """

    __tablename__ = "provider_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("trips.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    operation: Mapped[Optional[str]] = mapped_column(String(64))
    ok: Mapped[bool] = mapped_column(nullable=False, server_default=text("true"))
    cache_hit: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    result_count: Mapped[Optional[int]] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False, index=True
    )

    trip: Mapped[Optional[Trip]] = relationship(back_populates="provider_calls")

    __table_args__ = (
        # "how is provider X behaving lately" — the query this table exists for.
        Index("ix_provider_calls_provider_created_at", "provider", text("created_at DESC")),
    )


class CostSnapshot(Base):
    """The computed budget for a trip at one revision.

    ``Numeric(12, 2)`` not ``float``: money. The original code did arithmetic on
    floats parsed out of strings like "₹5,000" and displayed the result as a
    total.
    """

    __tablename__ = "cost_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str] = mapped_column(
        ForeignKey("trips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    currency: Mapped[str] = mapped_column(
        String(8), nullable=False, server_default=text("'INR'")
    )
    transport_cost: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    hotel_cost: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    onground_cost: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    total_cost: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    per_person_cost: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    breakdown: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    trip: Mapped[Trip] = relationship(back_populates="cost_snapshots")

    __table_args__ = (
        Index("uq_cost_snapshots_trip_revision", "trip_id", "revision", unique=True),
    )
