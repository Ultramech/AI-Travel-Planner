"""Keyless flight search via Google Flights.

Why this exists
---------------
Amadeus needs credentials, and its free *test* tier returns sparse, often
unrealistic inventory for Indian domestic routes. This provider reads the same
data Google Flights shows a browser — no API key, no signup, real fares.

It is the **default** flight source. Amadeus (``providers/flights.py``) stays
available and is preferred automatically when credentials are configured,
because it returns bookable offers with fare rules rather than display prices.

How it works: ``fast-flights`` builds the Base64 protobuf query string Google
Flights uses internally and parses the returned markup. No browser, so it is
far cheaper than the Selenium scrapers — typically 1-3 s.

Caveats, stated plainly:
  * These are *display* prices, not bookable offers. Treat them as indicative.
  * It depends on Google's internal response shape and will need maintenance.
  * No booking/ticketing path — we link out to Google Flights.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient

logger = logging.getLogger(__name__)

_SEAT_MAP = {
    "economy": "economy",
    "premium": "premium-economy",
    "premium_economy": "premium-economy",
    "business": "business",
    "first": "first",
}


def _as_int(value: Any) -> Optional[int]:
    """Google's arrays are ragged: ``time=[11]``, ``time=[14, 45]``, and on some
    routes ``[None]`` or a missing minute entirely. Every element must be
    treated as optional."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time_str(value: Any) -> Optional[str]:
    """``SimpleDatetime(time=[14, 45])`` -> ``"14:45"`` (minutes may be absent)."""
    parts = getattr(value, "time", None)
    if not parts:
        return None
    hour = _as_int(parts[0])
    if hour is None:
        return None
    minute = _as_int(parts[1]) if len(parts) > 1 else 0
    return f"{hour:02d}:{minute or 0:02d}"


def _date_str(value: Any) -> Optional[str]:
    parts = getattr(value, "date", None)
    if not parts or len(parts) < 3:
        return None
    year, month, day = (_as_int(parts[0]), _as_int(parts[1]), _as_int(parts[2]))
    if None in (year, month, day):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _iso(value: Any) -> Optional[str]:
    date_part, time_part = _date_str(value), _time_str(value)
    if not date_part:
        return None
    return f"{date_part}T{time_part or '00:00'}"


def _duration_pretty(minutes: Optional[int]) -> str:
    if not minutes:
        return ""
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h {mins}m" if hours else f"{mins}m"


def _booking_link(src: str, dst: str, departure_date: str) -> str:
    return (
        f"https://www.google.com/travel/flights?q=Flights%20from%20{src}%20to%20"
        f"{dst}%20on%20{departure_date}"
    )


def _search_sync(
    src: str,
    dst: str,
    departure_date: str,
    return_date: Optional[str],
    adults: int,
    seat: str,
    currency: str,
) -> list[Any]:
    from fast_flights import FlightQuery, Passengers, create_query, get_flights

    legs = [FlightQuery(date=departure_date, from_airport=src, to_airport=dst)]
    trip = "one-way"
    if return_date and return_date != departure_date:
        legs.append(
            FlightQuery(date=return_date, from_airport=dst, to_airport=src)
        )
        trip = "round-trip"

    query = create_query(
        flights=legs,
        trip=trip,
        seat=_SEAT_MAP.get(seat.lower(), "economy"),
        passengers=Passengers(adults=max(1, adults)),
        currency=currency,
    )
    return list(get_flights(query))


@cached("flights:google", ttl=settings.ttl_flights)
@resilient("google_flights", timeout=45.0, retries=2)
async def search_flights(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    currency: str = "INR",
    max_results: int = 8,
    seat: str = "economy",
) -> dict[str, Any]:
    """Search Google Flights. Same return shape as the Amadeus provider."""
    src = (origin_iata or "").strip().upper()
    dst = (destination_iata or "").strip().upper()
    if len(src) != 3 or len(dst) != 3:
        raise ProviderError("google_flights", f"invalid IATA pair {src!r}->{dst!r}")
    if src == dst:
        raise ProviderError("google_flights", "origin and destination are the same")
    try:
        datetime.strptime(departure_date, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ProviderError(
            "google_flights", f"bad date {departure_date!r}, expected YYYY-MM-DD"
        ) from exc

    try:
        raw = await asyncio.to_thread(
            _search_sync, src, dst, departure_date, return_date,
            int(adults or 1), seat, currency,
        )
    except ImportError as exc:
        raise ProviderError(
            "google_flights", "fast-flights is not installed (pip install fast-flights)"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderError("google_flights", str(exc)[:300]) from exc

    results: list[dict[str, Any]] = []
    for offer in raw:
        legs = list(getattr(offer, "flights", []) or [])
        if not legs:
            continue
        outbound = legs[0]
        final = legs[-1]

        total_minutes = sum(
            _as_int(getattr(leg, "duration", None)) or 0 for leg in legs
        )
        layover_airports = [
            getattr(getattr(leg, "from_airport", None), "code", "???")
            for leg in legs[1:]
        ]
        price = getattr(offer, "price", None)

        results.append(
            {
                "route": f"{getattr(outbound.from_airport, 'code', src)} → "
                         f"{getattr(final.to_airport, 'code', dst)}",
                "origin_iata": getattr(outbound.from_airport, "code", src),
                "destination_iata": getattr(final.to_airport, "code", dst),
                "origin_airport": getattr(outbound.from_airport, "name", None),
                "destination_airport": getattr(final.to_airport, "name", None),
                "departure": _iso(getattr(outbound, "departure", None)),
                "arrival": _iso(getattr(final, "arrival", None)),
                "arrival_time": _time_str(getattr(final, "arrival", None)),
                "duration": _duration_pretty(total_minutes),
                "duration_minutes": total_minutes,
                "price_inr": float(price) if isinstance(price, (int, float)) else None,
                "currency": currency,
                "stops": max(len(legs) - 1, 0),
                "layover_airports": layover_airports,
                "layover_pretty": "Direct" if len(legs) == 1
                                  else f"{len(legs) - 1} stop(s)",
                "carriers": list(getattr(offer, "airlines", []) or []),
                "aircraft": getattr(outbound, "plane_type", None),
                "carbon_kg": (
                    round(getattr(offer.carbon, "emission", 0) / 1000, 1)
                    if getattr(offer, "carbon", None)
                    and getattr(offer.carbon, "emission", None)
                    else None
                ),
                "is_round_trip": bool(return_date and return_date != departure_date),
                "source": "google_flights",
                "is_bookable_offer": False,  # display price, not a locked fare
                "booking_link": _booking_link(src, dst, departure_date),
            }
        )

    results.sort(key=lambda r: (r["price_inr"] if r["price_inr"] else float("inf")))
    trimmed = results[: max(1, max_results)]

    return {
        "origin_iata": src,
        "destination_iata": dst,
        "departure_date": departure_date,
        "return_date": return_date,
        "adults": adults,
        "count": len(trimmed),
        "total_found": len(results),
        "flights": trimmed,
        "cheapest_inr": trimmed[0]["price_inr"] if trimmed else None,
        "source": "google_flights",
        "price_disclaimer": "Display prices from Google Flights, not bookable offers.",
    }


def render(result: dict[str, Any]) -> str:
    flights = result.get("flights") or []
    if not flights:
        return "✈️ No flights found for this route and date."
    lines = [
        f"✈️ **{len(flights)} flight option(s)** {result['origin_iata']} → "
        f"{result['destination_iata']} on {result['departure_date']}"
        + (f" _(of {result['total_found']} found)_" if result.get("total_found") else "")
        + "\n"
    ]
    for flight in flights:
        carriers = ", ".join(flight.get("carriers") or []) or "Unknown airline"
        stops = (
            "Direct" if flight["stops"] == 0
            else f"{flight['stops']} stop(s) via {', '.join(flight['layover_airports'])}"
        )
        price = (
            f"₹{flight['price_inr']:,.0f}" if flight.get("price_inr") else "price n/a"
        )
        departure = (flight.get("departure") or "")[11:16] or "?"
        arrival = (flight.get("arrival") or "")[11:16] or "?"
        lines.append(
            f"🔹 **{carriers}** · {departure} → {arrival} · "
            f"{flight.get('duration') or '?'} · {price} · {stops}"
        )
    lines.append(f"\n🔗 {flights[0]['booking_link']}")
    lines.append(f"\n_{result.get('price_disclaimer', '')}_")
    return "\n".join(lines)
