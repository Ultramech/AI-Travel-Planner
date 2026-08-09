"""Event model shared by the API, the graph and the Kafka worker."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class TripStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(StrEnum):
    TRIP_REQUESTED = "trip.requested"
    TRIP_STARTED = "trip.started"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    PROVIDER_CALLED = "provider.called"
    AWAITING_REVIEW = "trip.awaiting_review"
    TRIP_RESUMED = "trip.resumed"
    TRIP_COMPLETED = "trip.completed"
    TRIP_FAILED = "trip.failed"


class TripEvent(BaseModel):
    """One point on a trip's timeline.

    Published to Kafka (durable log — replay, analytics, audit) *and* to Redis
    pub/sub (low-latency fan-out to SSE clients). Two sinks, two jobs.
    """

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    trip_id: str
    type: EventType
    node: Optional[str] = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utcnow)

    def to_json(self) -> str:
        return self.model_dump_json()


class PlanRequestMessage(BaseModel):
    """The job placed on ``trip.requests``."""

    trip_id: str
    payload: dict[str, Any]
    attempt: int = 0
    enqueued_at: datetime = Field(default_factory=utcnow)

    def to_json(self) -> str:
        return self.model_dump_json()


def stream_channel(trip_id: str) -> str:
    return f"tp:stream:{trip_id}"


def status_key(trip_id: str) -> str:
    return f"tp:trip:{trip_id}:status"


def result_key(trip_id: str) -> str:
    return f"tp:trip:{trip_id}:result"


def history_key(trip_id: str) -> str:
    return f"tp:trip:{trip_id}:events"
