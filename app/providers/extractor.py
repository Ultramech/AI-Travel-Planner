"""Turn a free-text trip request into a validated structured brief.

Fixes versus the original ``trip_extractor.py``:
  * It asked one LLM call to do everything: parse dates, infer duration, *and*
    invent IATA codes for both cities. Airport resolution is now deterministic
    (``providers.geo``) and only the genuinely linguistic parts go to the LLM.
  * Nothing validated the output. ``duration`` could be ``"3 days"``,
    ``Persons`` could be ``None``, and downstream ``int(parsed["Persons"])``
    raised ``TypeError`` inside a graph node. Everything is validated and
    coerced here, once.
  * Dates had no sanity checks — an end date before the start date, or a date
    in the past, flowed straight into paid API calls.
  * ``ConversationBufferMemory`` was imported and never used; the module also
    re-imported its own dependencies inside the function body on every call.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings
from app.core.cache import cached
from app.providers import geo
from app.providers import llm as llm_provider

logger = logging.getLogger(__name__)

MAX_TRIP_DAYS = 30
MAX_PERSONS = 20


class TripBrief(BaseModel):
    """The validated trip request every downstream node relies on."""

    origin: str
    destination: str
    start_date: str
    end_date: str
    # No upper bound on the field itself: the dates are authoritative and
    # ``_coherent`` recomputes this. A bogus LLM duration must not block a
    # perfectly valid pair of dates.
    duration: int = Field(default=1, ge=1)
    persons: int = Field(default=1, ge=1, le=MAX_PERSONS)
    preferences: list[str] = Field(default_factory=list)
    budget_level: str = "medium"
    travel_class: str = "economy"
    special_requirements: str = ""

    # Resolved by geo, never by the LLM alone.
    origin_iata: Optional[str] = None
    destination_iata: Optional[str] = None
    origin_airport_city: Optional[str] = None
    destination_airport_city: Optional[str] = None
    # False when the airport/railhead belongs to a *different* city than the one
    # the user named — i.e. the journey needs a connecting leg. These were being
    # computed and then ignored; ``providers.journey`` now acts on them.
    origin_airport_exact: bool = True
    destination_airport_exact: bool = True
    origin_railhead_city: Optional[str] = None
    destination_railhead_city: Optional[str] = None
    origin_railhead_exact: bool = True
    destination_railhead_exact: bool = True

    @field_validator("origin", "destination")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("origin and destination are required")
        return value

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso_date(cls, value: str) -> str:
        datetime.strptime(str(value), "%Y-%m-%d")
        return str(value)

    @model_validator(mode="after")
    def _coherent(self) -> "TripBrief":
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError(f"end_date {self.end_date} is before start_date {self.start_date}")
        actual = (end - start).days + 1
        if actual > MAX_TRIP_DAYS:
            # Reject rather than silently truncate — quietly turning a 60-day
            # trip into a 30-day one produces a plan for a trip nobody asked for.
            raise ValueError(
                f"trip spans {actual} days; the maximum supported is {MAX_TRIP_DAYS}"
            )
        if actual != self.duration:
            # Dates are authoritative; duration is derived from them.
            object.__setattr__(self, "duration", actual)
        if self.origin.strip().lower() == self.destination.strip().lower():
            raise ValueError("origin and destination must differ")
        return self

    @property
    def is_past(self) -> bool:
        return datetime.strptime(self.start_date, "%Y-%m-%d").date() < date.today()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _coerce_int(value: Any, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        match = re.search(r"\d+", str(value))
        number = int(match.group()) if match else default
    return max(lo, min(number, hi))


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in re.split(r"[,;/]", str(value)) if p.strip()]


_SYSTEM = (
    "You extract structured travel details from user text. "
    "Reply with a single JSON object and nothing else. "
    "Never invent airport codes."
)


@cached("extract:trip", ttl=settings.ttl_llm_extract)
async def extract_from_text(user_input: str, today: Optional[str] = None) -> dict[str, Any]:
    """LLM extraction of the linguistic fields only."""
    reference = today or date.today().isoformat()
    prompt = f"""Extract travel details from the text below. Today is {reference}.

Return JSON with exactly these keys:
{{
  "origin": "<city travelling from>",
  "destination": "<city travelling to>",
  "start_date": "<YYYY-MM-DD>",
  "end_date": "<YYYY-MM-DD>",
  "persons": <integer>,
  "preferences": ["<interest>", ...],
  "budget_level": "low" | "medium" | "high",
  "travel_class": "economy" | "business" | "first",
  "special_requirements": "<free text or empty string>"
}}

Rules:
- Resolve relative dates ("next Friday", "in two weeks") against {reference}.
- If only a start date and a duration are given, compute end_date.
- If no dates are given at all, assume the trip starts 14 days from {reference}
  and lasts 3 days.
- end_date must be on or after start_date.

Text:
\"\"\"{user_input}\"\"\"
"""
    return await llm_provider.ask_json(prompt, system=_SYSTEM,
                                       model=settings.llm_extractor_model)


async def build_brief(
    *,
    user_input: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    persons: Optional[int] = None,
    preferences: Optional[list[str]] = None,
    budget_level: Optional[str] = None,
    travel_class: Optional[str] = None,
    special_requirements: Optional[str] = None,
) -> TripBrief:
    """Build a validated brief from structured fields, free text, or both.

    Explicit structured fields always win over anything the LLM inferred — the
    Streamlit form already knows the exact dates, so re-deriving them from a
    generated sentence (as the old ``plan_my_trip`` did) only adds error.
    """
    extracted: dict[str, Any] = {}
    needs_llm = user_input and not (origin and destination and start_date and end_date)
    if needs_llm:
        try:
            extracted = await extract_from_text(user_input) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM extraction failed: %s", exc)
            extracted = {}

    resolved_origin = (origin or extracted.get("origin") or "").strip()
    resolved_destination = (destination or extracted.get("destination") or "").strip()
    if not resolved_origin or not resolved_destination:
        raise ValueError(
            "Could not determine origin and destination. "
            "Provide them explicitly or state them clearly in the request."
        )

    default_start = (date.today() + timedelta(days=14)).isoformat()
    resolved_start = str(start_date or extracted.get("start_date") or default_start)[:10]
    resolved_end = str(end_date or extracted.get("end_date") or "")[:10]
    if not resolved_end:
        start_obj = datetime.strptime(resolved_start, "%Y-%m-%d").date()
        resolved_end = (start_obj + timedelta(days=2)).isoformat()

    start_obj = datetime.strptime(resolved_start, "%Y-%m-%d").date()
    end_obj = datetime.strptime(resolved_end, "%Y-%m-%d").date()
    if end_obj < start_obj:
        logger.warning("end_date %s < start_date %s; clamping.", resolved_end, resolved_start)
        end_obj = start_obj
        resolved_end = end_obj.isoformat()

    duration = (end_obj - start_obj).days + 1

    brief_kwargs: dict[str, Any] = {
        "origin": resolved_origin,
        "destination": resolved_destination,
        "start_date": resolved_start,
        "end_date": resolved_end,
        "duration": duration,
        "persons": _coerce_int(
            persons if persons is not None else extracted.get("persons"),
            1, 1, MAX_PERSONS,
        ),
        "preferences": _coerce_list(
            preferences if preferences else extracted.get("preferences")
        ),
        "budget_level": str(
            budget_level or extracted.get("budget_level") or "medium"
        ).lower(),
        "travel_class": str(
            travel_class or extracted.get("travel_class") or "economy"
        ).lower(),
        "special_requirements": str(
            special_requirements or extracted.get("special_requirements") or ""
        ).strip(),
    }

    brief = TripBrief(**brief_kwargs)

    # Deterministic airport resolution, in parallel.
    import asyncio

    src_info, dst_info, src_rail, dst_rail = await asyncio.gather(
        geo.resolve_airport(brief.origin),
        geo.resolve_airport(brief.destination),
        geo.resolve_railhead(brief.origin),
        geo.resolve_railhead(brief.destination),
        return_exceptions=True,
    )
    if isinstance(src_info, dict):
        brief.origin_iata = src_info.get("iata")
        brief.origin_airport_city = src_info.get("airport_city")
        brief.origin_airport_exact = bool(src_info.get("exact"))
    if isinstance(dst_info, dict):
        brief.destination_iata = dst_info.get("iata")
        brief.destination_airport_city = dst_info.get("airport_city")
        brief.destination_airport_exact = bool(dst_info.get("exact"))
    if isinstance(src_rail, dict):
        brief.origin_railhead_city = src_rail.get("station_city")
        brief.origin_railhead_exact = bool(src_rail.get("exact"))
    if isinstance(dst_rail, dict):
        brief.destination_railhead_city = dst_rail.get("station_city")
        brief.destination_railhead_exact = bool(dst_rail.get("exact"))

    return brief
