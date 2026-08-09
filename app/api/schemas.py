"""Request/response models. Validation happens at the edge, once."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PlanTripRequest(BaseModel):
    """Either give structured fields, or a free-text ``query``, or both."""

    query: Optional[str] = Field(
        default=None, max_length=2000,
        description="Free-text request, e.g. 'Delhi to Goa next Friday for 4 days'",
    )
    origin: Optional[str] = Field(default=None, max_length=100)
    destination: Optional[str] = Field(default=None, max_length=100)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    persons: int = Field(default=1, ge=1, le=20)
    preferences: list[str] = Field(default_factory=list, max_length=12)
    budget_level: Literal["low", "medium", "high"] = "medium"
    travel_class: Literal["economy", "business", "first"] = "economy"
    special_requirements: str = Field(default="", max_length=500)
    async_mode: bool = Field(
        default=True,
        description="Queue the job and return 202 (recommended). "
                    "False runs it inline and blocks until the first review.",
    )

    @field_validator("origin", "destination")
    @classmethod
    def _clean(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value else value

    @model_validator(mode="after")
    def _validate(self) -> "PlanTripRequest":
        if not self.query and not (self.origin and self.destination):
            raise ValueError(
                "Provide either 'query', or both 'origin' and 'destination'."
            )
        if self.origin and self.destination and (
            self.origin.lower() == self.destination.lower()
        ):
            raise ValueError("origin and destination must differ")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.start_date and self.start_date < date.today() - timedelta(days=1):
            raise ValueError("start_date is in the past")
        if self.start_date and self.end_date and (self.end_date - self.start_date).days > 30:
            raise ValueError("trips longer than 30 days are not supported")
        return self

    def to_payload(self) -> dict[str, Any]:
        data = self.model_dump(exclude={"async_mode"})
        for key in ("start_date", "end_date"):
            if data.get(key):
                data[key] = data[key].isoformat()
        return data


class ResumeTripRequest(BaseModel):
    """Answer a pending human-review interrupt."""

    action: Literal["approve", "revise_transport", "revise_hotels", "revise_plan"]
    feedback: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _feedback_required_for_revision(self) -> "ResumeTripRequest":
        if self.action != "approve" and not (self.feedback or "").strip():
            raise ValueError(f"'{self.action}' requires feedback describing the change")
        return self


class TripAccepted(BaseModel):
    trip_id: str
    status: str
    message: str
    stream_url: str
    status_url: str


class PendingReview(BaseModel):
    stage: str
    question: str
    options: list[str]
    revisions_remaining: Optional[int] = None
    transport_summary: Optional[str] = None
    itinerary: Optional[str] = None
    budget: Optional[dict[str, Any]] = None
    provider_health: Optional[dict[str, Any]] = None


class TripResponse(BaseModel):
    trip_id: Optional[str] = None
    status: str
    brief: dict[str, Any] = Field(default_factory=dict)
    transport_strategy: Optional[dict[str, Any]] = None
    transport_options: dict[str, Any] = Field(default_factory=dict)
    transport_summary: Optional[str] = None
    arrival_time: Optional[str] = None
    weather: list[dict[str, Any]] = Field(default_factory=list)
    hotels: Optional[dict[str, Any]] = None
    hotels_summary: Optional[str] = None
    itinerary: Optional[str] = None
    itinerary_meta: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    provider_health: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    revision_count: int = 0
    pending_review: Optional[PendingReview] = None
    next_nodes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    dependencies: dict[str, Any]
    circuit_breakers: list[dict[str, Any]]
    timestamp: datetime
