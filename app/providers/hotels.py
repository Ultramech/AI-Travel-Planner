"""Hotel search via SerpAPI (Google Hotels).

Fixes versus the original:
  * ``float(price)`` on values like ``"₹5,000"`` raised ``ValueError`` on every
    single hotel, so the bare ``except`` made *every* total price read "N/A".
    Prices are now parsed properly.
  * Nightly rate x nights was labelled "for N persons" — it was neither a
    per-person figure nor a per-night one. Both are now reported explicitly.
  * ``requests.get`` with no timeout could hang a worker forever.
  * The API key was hardcoded in source; nights were taken from a free-text
    "duration" field that could be a string like "3 days".
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional

import httpx

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"


# Matches the first real number, allowing thousands separators and decimals.
# Anchored on a digit so the "." in "Rs." cannot be mistaken for a decimal point.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_price(value: Any) -> Optional[float]:
    """Turn ``"₹5,000"``, ``"$120"``, ``5000`` or ``"Rs. 5,000"`` into a float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def _nights(checkin: str, checkout: str) -> int:
    try:
        start = datetime.strptime(checkin, "%Y-%m-%d").date()
        end = datetime.strptime(checkout, "%Y-%m-%d").date()
        return max((end - start).days, 1)
    except (ValueError, TypeError):
        return 1


@cached("hotels", ttl=settings.ttl_hotels)
@resilient("serpapi_hotels", timeout=30.0, retries=2)
async def search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    adults: int = 1,
    preference: str = "",
    limit: int = 7,
) -> dict[str, Any]:
    """Return structured hotel options with correctly computed totals."""
    settings.require("serpapi_key")

    query = f"hotels in {city}"
    preference = (preference or "").strip()
    if preference:
        query = f"{query} {preference}"

    params = {
        "engine": "google_hotels",
        "q": query,
        "check_in_date": checkin,
        "check_out_date": checkout,
        "adults": max(1, int(adults or 1)),
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "api_key": settings.serpapi_key,
    }

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds * 2) as client:
        response = await client.get(SERPAPI_URL, params=params)

    if response.status_code != 200:
        raise ProviderError(
            "serpapi_hotels", f"HTTP {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    if data.get("error"):
        raise ProviderError("serpapi_hotels", str(data["error"]))

    properties = data.get("properties") or data.get("hotels") or []
    nights = _nights(checkin, checkout)

    hotels: list[dict[str, Any]] = []
    for item in properties[: max(1, limit)]:
        nightly = parse_price(
            (item.get("rate_per_night") or {}).get("extracted_lowest")
            or (item.get("rate_per_night") or {}).get("lowest")
        )
        total = parse_price(
            (item.get("total_rate") or {}).get("extracted_lowest")
            or (item.get("total_rate") or {}).get("lowest")
        )
        if total is None and nightly is not None:
            total = round(nightly * nights, 2)

        hotels.append(
            {
                "name": item.get("name", "Unknown"),
                "rating": item.get("overall_rating") or item.get("rating"),
                "reviews": item.get("reviews"),
                "hotel_class": item.get("hotel_class"),
                "nightly_price_inr": nightly,
                "total_price_inr": total,
                "nights": nights,
                "link": item.get("link", ""),
                "amenities": (item.get("amenities") or [])[:6],
                "location_rating": item.get("location_rating"),
                "coordinates": item.get("gps_coordinates"),
            }
        )

    priced = [h["total_price_inr"] for h in hotels if h["total_price_inr"]]
    return {
        "city": city,
        "checkin": checkin,
        "checkout": checkout,
        "nights": nights,
        "adults": adults,
        "preference": preference,
        "count": len(hotels),
        "hotels": hotels,
        "cheapest_total_inr": min(priced) if priced else None,
        "median_total_inr": (
            sorted(priced)[len(priced) // 2] if priced else None
        ),
    }


def render(result: dict[str, Any]) -> str:
    """Markdown rendering for the UI and for LLM context."""
    hotels = result.get("hotels") or []
    if not hotels:
        return f"🚫 No hotels found for {result.get('city', 'this city')}."

    nights = result.get("nights", 1)
    adults = result.get("adults", 1)
    lines = [
        f"🏨 **Top {len(hotels)} hotels in {result['city']}** "
        f"({result['checkin']} → {result['checkout']}, {nights} night(s), "
        f"{adults} adult(s))\n"
    ]
    for hotel in hotels:
        nightly = hotel.get("nightly_price_inr")
        total = hotel.get("total_price_inr")
        price_bits = []
        if nightly:
            price_bits.append(f"₹{nightly:,.0f}/night")
        if total:
            price_bits.append(f"₹{total:,.0f} total for {nights} night(s)")
        price = " · ".join(price_bits) if price_bits else "price unavailable"
        rating = hotel.get("rating")
        rating_text = f"⭐ {rating}" if rating else "unrated"
        lines.append(f"🔹 **{hotel['name']}** — {price} — {rating_text}")
        if hotel.get("link"):
            lines.append(f"   🔗 {hotel['link']}")
    return "\n".join(lines)
