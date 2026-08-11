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
import json
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


def _empty(
    src: str,
    dst: str,
    departure_date: str,
    return_date: Optional[str],
    adults: int,
    *,
    note: str,
    no_inventory: bool = False,
) -> dict[str, Any]:
    """The zero-result shape, identical in every key to a populated one."""
    return {
        "origin_iata": src,
        "destination_iata": dst,
        "departure_date": departure_date,
        "return_date": return_date,
        "adults": adults,
        "count": 0,
        "total_found": 0,
        "flights": [],
        "cheapest_inr": None,
        "cheapest_total_inr": None,
        "cheapest_per_person_inr": None,
        "source": "google_flights",
        "no_inventory": no_inventory,
        "note": note,
    }


class _NoInventory(Exception):
    """Google's response genuinely contains no flights for this route/date.

    Raised only after we have read the flight slot ourselves and found it empty
    — never merely because a parse blew up. That distinction was learnt the hard
    way: catching every ``TypeError``/``IndexError`` from ``fast_flights`` and
    calling it "no inventory" reported *"No airline flies DEL→COK"*, a route
    with dozens of daily departures, because the library indexes alliance
    metadata (``payload[7][1][1]``) **before** it reads the flights and Google
    does not always send that block.
    """


def _parse_payload(html: str) -> list[Any]:
    """Read flights out of Google's response, tolerating missing metadata.

    ``fast_flights.parser.parse_js`` builds its alliance/airline tables first
    and only then looks at ``payload[3][0]``, so a response without the metadata
    block raises before it ever reaches the flight data that is sitting right
    there. We only want the flights, so we read that slot directly and treat
    everything else as optional.
    """
    from selectolax.lexbor import LexborHTMLParser

    script = LexborHTMLParser(html).css_first(r"script.ds\:1")
    if script is None:
        raise ProviderError("google_flights", "response contained no flight script")

    body = script.text()
    raw = body.split("data:", 1)[1].rsplit(",", 1)[0]
    if raw.rstrip().endswith("errorHasStatus: true"):
        raise _NoInventory("google reported an error for this query")

    payload = json.loads(raw)

    if not isinstance(payload, list) or len(payload) < 4:
        raise ProviderError(
            "google_flights",
            f"unexpected response shape: {type(payload).__name__} "
            f"len={len(payload) if isinstance(payload, list) else 'n/a'}",
        )

    # ``payload[3]`` itself is None for a city pair nobody flies — that is a real
    # answer, not a broken response. Only a payload that is too short to *have* a
    # flight slot counts as a failure.
    section = payload[3][0] if payload[3] else None
    if not section:
        raise _NoInventory("no flights in the response")

    results = _offers_from_entries(section)
    if not results:
        raise _NoInventory("no parsable flights in the response")
    return results


def _offers_from_entries(entries: list[Any]) -> list[Any]:
    """Raw payload itineraries -> ``fast_flights`` model objects.

    Split out from the fetch so recorded fixtures can be replayed through the
    exact code path a live response takes. See tests/test_provider_contracts.py.
    """
    from fast_flights.model import (
        Airport, CarbonEmission, Flights, SimpleDatetime, SingleFlight,
    )

    results: list[Any] = []
    for entry in entries or []:
        try:
            flight = entry[0]
            price = entry[1][0][1]
            legs = []
            for single in flight[2]:
                legs.append(
                    SingleFlight(
                        from_airport=Airport(code=single[3], name=single[4]),
                        to_airport=Airport(code=single[6], name=single[5]),
                        departure=SimpleDatetime(date=single[20], time=single[8]),
                        arrival=SimpleDatetime(date=single[21], time=single[10]),
                        duration=single[11],
                        plane_type=single[17],
                    )
                )
            extras = flight[22] if len(flight) > 22 else None
            results.append(
                Flights(
                    type=flight[0],
                    price=price,
                    airlines=flight[1],
                    flights=legs,
                    carbon=CarbonEmission(
                        typical_on_route=extras[8] if extras else None,
                        emission=extras[7] if extras else None,
                    ),
                )
            )
        except (IndexError, TypeError, KeyError):
            # One malformed itinerary must not discard the other sixty-six.
            continue
    return results


def _build_results(
    entries: list[Any],
    src: str = "",
    dst: str = "",
    departure_date: str = "1970-01-01",
    party: int = 2,
    currency: str = "INR",
    return_date: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Raw payload itineraries -> the flight dicts the rest of the app uses.

    The whole chain in one call, so a contract test can assert on exactly what
    downstream code receives rather than on an intermediate shape.
    """
    return _to_dicts(
        _offers_from_entries(entries), src, dst, departure_date, party,
        currency, return_date,
    )


def _to_dicts(
    raw: list[Any],
    src: str,
    dst: str,
    departure_date: str,
    party: int,
    currency: str,
    return_date: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Map parsed offers onto the flight dicts the rest of the app consumes."""
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
        # Google quotes the price for the WHOLE party the query asked about:
        # DEL->BOM measured live at ₹6,074 / ₹12,148 / ₹18,222 for 1 / 2 / 3
        # adults — exact multiples. The old field name ``price_inr`` said
        # nothing about which it was, and the recommendation prompt then asked
        # the LLM for "total cost for all N travellers", so it multiplied an
        # already-multiplied number. Both figures are now named explicitly.
        raw_price = getattr(offer, "price", None)
        total_price = float(raw_price) if isinstance(raw_price, (int, float)) else None
        per_person = round(total_price / party, 2) if total_price is not None else None

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
                # Retained, and still the group total, so existing readers keep
                # working; the two explicit fields below are what new code uses.
                "price_inr": total_price,
                "price_total_inr": total_price,
                "price_per_person_inr": per_person,
                "party_size": party,
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
    return results


def _search_sync(
    src: str,
    dst: str,
    departure_date: str,
    return_date: Optional[str],
    adults: int,
    seat: str,
    currency: str,
) -> list[Any]:
    from fast_flights import FlightQuery, Passengers, create_query
    from fast_flights.fetcher import fetch_flights_html

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
    # Fetch and parse separately, rather than calling ``get_flights``, so a
    # missing metadata block cannot hide the flights.
    return _parse_payload(fetch_flights_html(query))


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

    party = max(1, int(adults or 1))
    try:
        raw = await asyncio.to_thread(
            _search_sync, src, dst, departure_date, return_date,
            party, seat, currency,
        )
    except _NoInventory as exc:
        # An honest empty answer, not a failure. Returned rather than raised so
        # the caller can say "nothing flies this route" instead of "search
        # unavailable", and so the circuit breaker stays closed.
        logger.info("No flight inventory for %s->%s on %s (%s)",
                    src, dst, departure_date, exc)
        return _empty(src, dst, departure_date, return_date, party,
                      note=f"No flights are offered on {src}→{dst} for this date.",
                      no_inventory=True)
    except ImportError as exc:
        raise ProviderError(
            "google_flights", "fast-flights is not installed (pip install fast-flights)"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderError("google_flights", str(exc)[:300]) from exc

    results = _to_dicts(raw, src, dst, departure_date, party, currency, return_date)

    results.sort(key=lambda r: (r["price_inr"] if r["price_inr"] else float("inf")))
    trimmed = results[: max(1, max_results)]

    if not trimmed:
        return _empty(
            src, dst, departure_date, return_date, party,
            note=f"No flights are offered on {src}→{dst} for this date.",
            no_inventory=True,
        )

    return {
        "origin_iata": src,
        "destination_iata": dst,
        "departure_date": departure_date,
        "return_date": return_date,
        "adults": party,
        "count": len(trimmed),
        "total_found": len(results),
        "flights": trimmed,
        "cheapest_inr": trimmed[0]["price_inr"],
        "cheapest_total_inr": trimmed[0]["price_total_inr"],
        "cheapest_per_person_inr": trimmed[0]["price_per_person_inr"],
        "party_size": party,
        "source": "google_flights",
        "price_disclaimer": (
            f"Display prices from Google Flights for all {party} traveller(s), "
            "not bookable offers."
        ),
    }


def render(result: dict[str, Any]) -> str:
    flights = result.get("flights") or []
    if not flights:
        # The router puts the reason in ``note``; dropping it made a crashed
        # provider indistinguishable from a route nobody flies.
        return f"✈️ {result.get('note') or 'No flights found for this route and date.'}"
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
        party = flight.get("party_size") or result.get("party_size") or 1
        if flight.get("price_total_inr"):
            price = f"₹{flight['price_total_inr']:,.0f}"
            if party > 1:
                price += f" total (₹{flight['price_per_person_inr']:,.0f} pp)"
        else:
            price = "price n/a"
        departure = (flight.get("departure") or "")[11:16] or "?"
        arrival = (flight.get("arrival") or "")[11:16] or "?"
        lines.append(
            f"🔹 **{carriers}** · {departure} → {arrival} · "
            f"{flight.get('duration') or '?'} · {price} · {stops}"
        )
    lines.append(f"\n🔗 {flights[0]['booking_link']}")
    lines.append(f"\n_{result.get('price_disclaimer', '')}_")
    return "\n".join(lines)
