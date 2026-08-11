"""Flight search via Amadeus.

Fixes versus the original:
  * ``search_flights_tool`` returned ``{"error": ...}`` on failure, and the
    caller did ``if not flights: raise`` — a non-empty dict is truthy, so error
    payloads were rendered to users as if they were flight results.
  * ``returnDate`` was always sent, forcing a round trip even for one-way legs
    and silently returning nothing for many routes.
  * Layover maths summed *all* gaps including the flight legs themselves in
    some cases, and the airport-name lookup issued a blocking Amadeus call per
    IATA code with only a per-process dict as cache.
  * Credentials were hardcoded in the source file.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Optional

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient
from app.providers.geo import lookup_iata

logger = logging.getLogger(__name__)

AFFILIATE_MARKER = "641880"


@lru_cache(maxsize=1)
def _client() -> Any:
    settings.require("amadeus_client_id", "amadeus_client_secret")
    from amadeus import Client

    return Client(
        client_id=settings.amadeus_client_id,
        client_secret=settings.amadeus_client_secret,
        hostname=settings.amadeus_hostname,
    )


@cached("flights:airport_name", ttl=settings.ttl_iata)
@resilient("amadeus_locations", timeout=15.0, retries=1)
async def get_airport_name(iata_code: str) -> str:
    """Human-readable airport name, Redis-cached for 30 days."""
    code = (iata_code or "").strip().upper()
    if len(code) != 3:
        return code

    def _call() -> str:
        from amadeus import ResponseError

        try:
            response = _client().reference_data.locations.get(
                keyword=code, subType="AIRPORT"
            )
            for item in response.data or []:
                if item.get("iataCode") == code:
                    return item.get("name") or f"{code} Airport"
        except ResponseError:
            pass
        return f"{code} Airport"

    return await asyncio.to_thread(_call)


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _layovers(segments: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """Total layover minutes and the airports they happen at.

    Only the *gaps between* consecutive segments count — arrival of leg N to
    departure of leg N+1.
    """
    total = timedelta()
    airports: list[str] = []
    for i in range(1, len(segments)):
        arrival = _parse_iso(segments[i - 1].get("arrival", {}).get("at", ""))
        departure = _parse_iso(segments[i].get("departure", {}).get("at", ""))
        if arrival and departure and departure > arrival:
            total += departure - arrival
        airports.append(segments[i].get("departure", {}).get("iataCode", "???"))
    return int(total.total_seconds() // 60), airports


def _format_duration(minutes: int) -> str:
    if minutes <= 0:
        return "0m"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m" if hours else f"{mins}m"


@cached("flights:search", ttl=settings.ttl_flights)
@resilient("amadeus_flights", timeout=45.0, retries=2)
async def search_flights(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    currency: str = "INR",
    max_results: int = 6,
    non_stop: bool = False,
) -> dict[str, Any]:
    """Search flight offers. Raises ``ProviderError`` on failure — never lies."""
    src = (origin_iata or "").strip().upper()
    dst = (destination_iata or "").strip().upper()
    if len(src) != 3 or len(dst) != 3:
        raise ProviderError("amadeus_flights", f"invalid IATA pair {src!r}->{dst!r}")
    if src == dst:
        raise ProviderError("amadeus_flights", "origin and destination are the same")

    params: dict[str, Any] = {
        "originLocationCode": src,
        "destinationLocationCode": dst,
        "departureDate": departure_date,
        "adults": max(1, int(adults or 1)),
        "currencyCode": currency,
        "max": max(1, int(max_results)),
    }
    # Only request a round trip when one was actually asked for.
    if return_date and return_date != departure_date:
        params["returnDate"] = return_date
    if non_stop:
        params["nonStop"] = True

    def _call() -> list[dict[str, Any]]:
        from amadeus import ResponseError

        try:
            return _client().shopping.flight_offers_search.get(**params).data or []
        except ResponseError as error:
            body = getattr(getattr(error, "response", None), "body", "")
            raise ProviderError("amadeus_flights", f"{error} {body}"[:400]) from error

    offers = await asyncio.to_thread(_call)

    results: list[dict[str, Any]] = []
    for offer in offers:
        itineraries = offer.get("itineraries") or []
        if not itineraries:
            continue
        outbound = itineraries[0]
        segments = outbound.get("segments") or []
        if not segments:
            continue

        origin_code = segments[0]["departure"]["iataCode"]
        dest_code = segments[-1]["arrival"]["iataCode"]
        departure = segments[0]["departure"]["at"]
        arrival = segments[-1]["arrival"]["at"]
        layover_minutes, layover_airports = _layovers(segments)

        dep_dt = _parse_iso(departure)
        booking_link = ""
        if dep_dt:
            booking_link = (
                f"https://www.aviasales.com/search/"
                f"{origin_code}{dep_dt.strftime('%d%m')}{dest_code}1"
                f"?marker={AFFILIATE_MARKER}"
            )

        carriers = sorted({s.get("carrierCode", "") for s in segments if s.get("carrierCode")})

        _party = max(1, int(adults or 1))
        _total = float(offer.get("price", {}).get("total", 0) or 0)

        results.append(
            {
                "route": f"{origin_code} → {dest_code}",
                "origin_iata": origin_code,
                "destination_iata": dest_code,
                "departure": departure,
                "arrival": arrival,
                "arrival_time": dep_dt and _parse_iso(arrival)
                and _parse_iso(arrival).strftime("%H:%M"),
                "duration": outbound.get("duration", "").replace("PT", "").lower(),
                # Amadeus ``price.total`` covers the whole party, exactly like
                # Google's. Named explicitly so nothing downstream multiplies it
                # by the traveller count a second time.
                "price_inr": _total,
                "price_total_inr": _total,
                "price_per_person_inr": round(_total / _party, 2) if _total else None,
                "party_size": _party,
                "currency": offer.get("price", {}).get("currency", currency),
                "stops": len(segments) - 1,
                "layover_minutes": layover_minutes,
                "layover_pretty": (
                    _format_duration(layover_minutes) if layover_minutes else "Direct"
                ),
                "layover_airports": layover_airports,
                "carriers": carriers,
                "seats_available": offer.get("numberOfBookableSeats"),
                "is_round_trip": len(itineraries) > 1,
                "booking_link": booking_link,
            }
        )

    results.sort(key=lambda r: (r["price_inr"] or float("inf")))

    return {
        "origin_iata": src,
        "destination_iata": dst,
        "departure_date": departure_date,
        "return_date": return_date,
        "adults": adults,
        "party_size": max(1, int(adults or 1)),
        "count": len(results),
        "flights": results,
        "cheapest_inr": results[0]["price_inr"] if results else None,
        "cheapest_total_inr": results[0]["price_total_inr"] if results else None,
        "cheapest_per_person_inr": (
            results[0]["price_per_person_inr"] if results else None
        ),
    }


async def search_flights_for_cities(
    origin_city: str,
    destination_city: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """City-name convenience wrapper backed by the local IATA dataset."""
    src = lookup_iata(origin_city)
    dst = lookup_iata(destination_city)
    if not src or not dst:
        missing = origin_city if not src else destination_city
        raise ProviderError("amadeus_flights", f"no airport known for {missing!r}")
    return await search_flights(
        src, dst, departure_date, return_date, adults=adults, **kwargs
    )


def render(result: dict[str, Any]) -> str:
    flights = result.get("flights") or []
    if not flights:
        return "✈️ No flights found for this route and date."
    lines = [f"✈️ **{len(flights)} flight option(s)** {result['origin_iata']} → "
             f"{result['destination_iata']} on {result['departure_date']}\n"]
    for f in flights:
        stops = "Direct" if f["stops"] == 0 else (
            f"{f['stops']} stop(s) via {', '.join(f['layover_airports'])} "
            f"({f['layover_pretty']})"
        )
        lines.append(
            f"🔹 {f['route']} · {f['departure'][11:16]} → {f['arrival'][11:16]} · "
            f"{f['duration']} · ₹{f['price_inr']:,.0f} · {stops}"
        )
        if f.get("booking_link"):
            lines.append(f"   🔗 {f['booking_link']}")
    return "\n".join(lines)
