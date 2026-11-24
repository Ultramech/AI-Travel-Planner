"""Cab / road-transfer estimates.

The original ``find_cabs`` had a signature of
``(origin, destination, date=None, time=None, persons=1)`` but **every caller
passed ``persons`` as the fourth positional argument**, so the passenger count
landed in ``time`` and ``persons`` was always 1. It also hard-required Tavily
and returned raw search-engine output straight into an LLM prompt.

Here Tavily is optional: when it is configured we use it for live fare quotes,
and otherwise we fall back to a transparent distance-based estimate that is
clearly labelled as an estimate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import settings
from app.core.cache import cached
from app.core.resilience import resilient
from app.providers import llm as llm_provider

logger = logging.getLogger(__name__)

# Indicative Indian intercity rates, used only for the fallback estimate.
_RATE_CARD = {
    "hatchback": {"per_km": 13.0, "capacity": 4, "base": 500},
    "sedan": {"per_km": 16.0, "capacity": 4, "base": 600},
    "suv": {"per_km": 21.0, "capacity": 6, "base": 800},
    "tempo_traveller": {"per_km": 28.0, "capacity": 12, "base": 1500},
}


@cached("cabs:road", ttl=settings.ttl_cabs)
async def estimate_road_journey(origin: str, destination: str) -> dict[str, Any]:
    """Road distance **and driving time**. Cached hard — neither changes.

    The driving time was already being asked for and then thrown away, which
    mattered once connecting legs became real: dividing distance by an assumed
    speed says Shimla→Chandigarh is 2 hours, when 113 km of hill road is nearer
    4. Getting that wrong turns "leave by 08:00" into a missed flight.
    """
    try:
        data = await llm_provider.ask_json(
            f"What is the approximate road driving distance and typical driving "
            f"time from {origin} to {destination} in India? Account for terrain — "
            f"hill roads are much slower than highways.\n"
            'Reply with JSON only: {"km": <number>, "hours": <number>}'
        )
        km = float(data.get("km", 0) or 0)
        hours = float(data.get("hours", 0) or 0)
        if km <= 0:
            return {"km": None, "hours": None}
        if hours <= 0:
            hours = km / 45.0  # last resort: a plain intercity average
        return {"km": km, "hours": hours}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Road estimate failed for %s->%s: %s", origin, destination, exc)
        return {"km": None, "hours": None}


async def estimate_distance_km(origin: str, destination: str) -> Optional[float]:
    """Road distance only. Kept as the narrow call most callers want."""
    return (await estimate_road_journey(origin, destination)).get("km")


@cached("cabs", ttl=settings.ttl_cabs)
@resilient("cabs", timeout=45.0, retries=1)
async def search_cabs(
    origin: str,
    destination: str,
    travel_date: Optional[str] = None,
    persons: int = 1,
    departure_time: Optional[str] = None,
) -> dict[str, Any]:
    """Return cab options. ``persons`` is keyword-safe now."""
    persons = max(1, int(persons or 1))

    live: Optional[str] = None
    if settings.tavily_api_key:
        try:
            live = await _tavily_quote(origin, destination, travel_date, persons)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily cab lookup failed: %s", exc)

    distance = await estimate_distance_km(origin, destination)
    options: list[dict[str, Any]] = []
    if distance:
        for name, spec in _RATE_CARD.items():
            if spec["capacity"] < persons:
                continue
            fare = spec["base"] + spec["per_km"] * distance
            options.append(
                {
                    "vehicle": name.replace("_", " ").title(),
                    "capacity": spec["capacity"],
                    "estimated_fare_inr": round(fare, -1),
                    "per_person_inr": round(fare / persons, -1),
                    "basis": f"₹{spec['per_km']}/km over ~{distance:.0f} km + "
                             f"₹{spec['base']} base",
                }
            )

    return {
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "persons": persons,
        "departure_time": departure_time,
        "distance_km": distance,
        "options": options,
        "live_quotes": live,
        "is_estimate": True,
        "note": None if options else "Could not estimate cab fares for this route.",
    }


async def _tavily_quote(
    origin: str, destination: str, travel_date: Optional[str], persons: int
) -> Optional[str]:
    """Search live outstation cab fares via Tavily's REST API.

    Called directly over httpx: the previous ``langchain_tavily`` import was
    never in requirements.txt, so setting TAVILY_API_KEY silently did nothing.
    """
    import httpx

    query = (
        f"outstation cab taxi fare from {origin} to {destination} India "
        f"for {persons} passengers"
    )
    if travel_date:
        query += f" on {travel_date}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"query": query, "max_results": 3, "include_answer": True},
        )
        response.raise_for_status()
        data = response.json()

    parts = [data.get("answer") or ""]
    for item in data.get("results") or []:
        parts.append(f"- {item.get('title', '')}: {item.get('content', '')} ({item.get('url', '')})")
    text = "\n".join(p for p in parts if p).strip()
    return text[:2000] or None


def render(result: dict[str, Any]) -> str:
    options = result.get("options") or []
    if not options:
        return f"🚖 {result.get('note', 'No cab estimate available.')}"
    distance = result.get("distance_km")
    lines = [
        f"🚖 **Cab options** {result['origin']} → {result['destination']}"
        + (f" (~{distance:.0f} km)" if distance else "")
        + f" for {result['persons']} passenger(s)\n"
    ]
    for option in options:
        lines.append(
            f"🔹 **{option['vehicle']}** (up to {option['capacity']}) · "
            f"~₹{option['estimated_fare_inr']:,.0f} total · "
            f"~₹{option['per_person_inr']:,.0f} per person"
        )
    lines.append("\n_Fares are indicative estimates, not live quotes._")
    return "\n".join(lines)
