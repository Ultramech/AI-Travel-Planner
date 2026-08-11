"""Flight-source selection with automatic failover.

Two backends, same contract:

  ``google_flights``  keyless, real display prices, wide inventory. Default.
  ``amadeus``         bookable offers with fare rules — better data, but needs
                      credentials, and its free *test* tier has thin inventory.

Policy:
  * ``FLIGHT_PROVIDER=auto`` (default) — prefer Amadeus when credentials exist
    **and** its circuit is closed; otherwise Google Flights. If the preferred
    source fails or returns nothing, fall back to the other one rather than
    telling the user "no flights".
  * ``FLIGHT_PROVIDER=google`` / ``amadeus`` — force one.

The point is that a missing API key degrades the *quality* of the flight data,
never its availability.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from app.config import settings
from app.core.resilience import ProviderError, get_breaker
from app.providers import flights as amadeus_provider
from app.providers import flights_google as google_provider

logger = logging.getLogger(__name__)

GOOGLE = "google"
AMADEUS = "amadeus"


def amadeus_available() -> bool:
    return bool(settings.amadeus_client_id and settings.amadeus_client_secret)


def _order() -> list[str]:
    """Sources to try, best first."""
    configured = (settings.flight_provider or "auto").strip().lower()

    if configured == GOOGLE:
        return [GOOGLE]
    if configured == AMADEUS:
        return [AMADEUS]

    # auto
    if amadeus_available() and get_breaker("amadeus_flights").state == "closed":
        return [AMADEUS, GOOGLE]
    return [GOOGLE, AMADEUS] if amadeus_available() else [GOOGLE]


async def search_flights(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    currency: str = "INR",
    max_results: int = 8,
    skip_cache: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search flights using the best available source, with failover."""
    result = await _search(
        origin_iata, destination_iata, departure_date, return_date,
        adults, currency, max_results, skip_cache,
    )

    # Thin routes often have no *round-trip* inventory even when the outbound
    # flies: IXC→BOM on 2026-08-25 returns nothing as a round trip and three
    # flights one-way. Everything downstream only ever uses the outbound leg, so
    # a return date must not be able to hide the flight the traveller needs.
    if return_date and return_date != departure_date and not result.get("flights"):
        logger.info(
            "No round-trip inventory %s->%s; retrying one-way.",
            origin_iata, destination_iata,
        )
        one_way = await _search(
            origin_iata, destination_iata, departure_date, None,
            adults, currency, max_results, skip_cache,
        )
        if one_way.get("flights"):
            one_way["outbound_only"] = True
            one_way["note"] = (
                "Outbound fares only — this route has no round-trip inventory "
                f"for a {departure_date} → {return_date} pairing, so the return "
                "leg needs booking separately."
            )
            return one_way

    # Nothing direct, and nothing one-way either. Before giving up, try the
    # thing a human would: connect through a big airport. Gangtok→Kochi
    # (IXB→COK) and Manali→Munnar (KUU→COK) both reported "no bookable
    # transport" while IXB→DEL→COK and KUU→DEL→COK were perfectly ordinary.
    if settings.flight_hub_routing and not result.get("flights"):
        via = await _search_via_hub(
            origin_iata, destination_iata, departure_date,
            adults, currency, max_results, skip_cache,
        )
        if via:
            return via

    return result


#: Airports with enough domestic reach to connect almost anything to anything.
#: Ordered by network size, and tried in that order.
HUBS = ("DEL", "BOM", "BLR", "HYD", "MAA", "CCU")


async def _search_via_hub(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    adults: int,
    currency: str,
    max_results: int,
    skip_cache: bool,
) -> Optional[dict[str, Any]]:
    """Find a same-day two-leg itinerary through a hub.

    Deliberately modest: same calendar day, a real connection window, and the
    result is clearly marked ``self_transfer`` because these are two separate
    bookings — a missed connection is the traveller's problem, not an airline's.
    Saying that plainly matters more than the saving.
    """
    src = (origin_iata or "").strip().upper()
    dst = (destination_iata or "").strip().upper()
    candidates = [h for h in HUBS if h not in (src, dst)][:settings.flight_hub_attempts]
    if not candidates:
        return None

    async def leg(a: str, b: str) -> dict[str, Any]:
        try:
            return await _search(a, b, departure_date, None, adults, currency,
                                 max_results, skip_cache)
        except Exception as exc:  # noqa: BLE001
            logger.info("Hub leg %s->%s failed: %s", a, b, exc)
            return {"flights": []}

    for hub in candidates:
        first, second = await asyncio.gather(leg(src, hub), leg(hub, dst))
        pairs = _pair_legs(first.get("flights") or [], second.get("flights") or [])
        if not pairs:
            continue
        logger.info("Routing %s->%s via %s (%d viable pairing(s))",
                    src, dst, hub, len(pairs))
        return {
            "origin_iata": src,
            "destination_iata": dst,
            "departure_date": departure_date,
            "return_date": None,
            "adults": adults,
            "party_size": max(1, int(adults or 1)),
            "count": len(pairs),
            "total_found": len(pairs),
            "flights": pairs,
            "cheapest_inr": pairs[0]["price_total_inr"],
            "cheapest_total_inr": pairs[0]["price_total_inr"],
            "cheapest_per_person_inr": pairs[0]["price_per_person_inr"],
            "source": "hub_routing",
            "via_hub": hub,
            "self_transfer": True,
            "note": (
                f"No direct {src}→{dst} service. These connect through {hub} as "
                f"two separate bookings — you clear and re-check bags yourself, "
                f"and a delay on the first leg is not covered by the second."
            ),
            "price_disclaimer": (
                f"Sum of two one-way display prices for all {max(1, adults)} "
                f"traveller(s), not a through-fare."
            ),
        }
    return None


#: Minimum and maximum gap at the hub. Under the floor you will not make it
#: with a self-transfer; over the ceiling it is a day trip, not a connection.
MIN_LAYOVER_MIN = 120
MAX_LAYOVER_MIN = 480


def _pair_legs(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Combine inbound and outbound legs into workable itineraries."""
    out: list[dict[str, Any]] = []
    for a in first[:6]:
        arrival = _parse_dt(a.get("arrival"))
        if not arrival:
            continue
        for b in second[:6]:
            departure = _parse_dt(b.get("departure"))
            if not departure:
                continue
            gap = (departure - arrival).total_seconds() / 60
            if not (MIN_LAYOVER_MIN <= gap <= MAX_LAYOVER_MIN):
                continue
            total = (a.get("price_total_inr") or 0) + (b.get("price_total_inr") or 0)
            if not total:
                continue
            party = a.get("party_size") or 1
            out.append({
                "route": f"{a['origin_iata']} → {a['destination_iata']} → "
                         f"{b['destination_iata']}",
                "origin_iata": a["origin_iata"],
                "destination_iata": b["destination_iata"],
                "departure": a.get("departure"),
                "arrival": b.get("arrival"),
                "arrival_time": b.get("arrival_time"),
                "duration": _pretty((departure - arrival).total_seconds() / 60
                                    + (a.get("duration_minutes") or 0)
                                    + (b.get("duration_minutes") or 0)),
                "layover_pretty": f"{_pretty(gap)} at {a['destination_iata']}",
                "layover_airports": [a["destination_iata"]],
                "stops": 1,
                "price_total_inr": total,
                "price_inr": total,
                "price_per_person_inr": round(total / party, 2),
                "party_size": party,
                "currency": a.get("currency", "INR"),
                "carriers": list(dict.fromkeys(
                    (a.get("carriers") or []) + (b.get("carriers") or [])
                )),
                "source": "hub_routing",
                "self_transfer": True,
                "is_bookable_offer": False,
                "booking_link": a.get("booking_link"),
                "second_leg_link": b.get("booking_link"),
            })
    out.sort(key=lambda r: r["price_total_inr"])
    return out[:5]


def _parse_dt(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _pretty(minutes: float) -> str:
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h {mins}m" if hours else f"{mins}m"


async def _search(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    return_date: Optional[str],
    adults: int,
    currency: str,
    max_results: int,
    skip_cache: bool,
) -> dict[str, Any]:
    """One pass over the configured sources, best first."""
    sources = _order()
    errors: list[str] = []
    # Distinguish "every source blew up" from "the route genuinely has no
    # service". Both used to produce the same empty dict, so a crashed provider
    # was reported to the user *and* to provider_health as a healthy no-result.
    crashed = 0
    no_inventory = False

    for name in sources:
        provider = google_provider if name == GOOGLE else amadeus_provider
        try:
            result = await provider.search_flights(
                origin_iata,
                destination_iata,
                departure_date,
                return_date=return_date,
                adults=adults,
                currency=currency,
                max_results=max_results,
                skip_cache=skip_cache,
            )
        except ProviderError as exc:
            logger.warning("Flight source %s failed: %s", name, exc)
            errors.append(f"{name}: {exc.message}")
            crashed += 1
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Flight source %s errored: %s", name, exc)
            errors.append(f"{name}: {exc}")
            crashed += 1
            continue

        if result.get("flights"):
            result.setdefault("source", name)
            result["sources_tried"] = sources[: sources.index(name) + 1]
            if errors:
                result["fallback_from"] = errors
            return result

        if result.get("no_inventory"):
            no_inventory = True
        errors.append(f"{name}: {result.get('note') or 'no flights returned'}")

    all_failed = crashed == len(sources) and not no_inventory
    if no_inventory:
        note = (
            f"No airline flies {(origin_iata or '').upper()}→"
            f"{(destination_iata or '').upper()} on {departure_date}."
        )
    elif all_failed:
        note = "Flight search is currently unavailable. " + "; ".join(errors)
    else:
        note = "No flights found. " + "; ".join(errors)

    return {
        "origin_iata": (origin_iata or "").upper(),
        "destination_iata": (destination_iata or "").upper(),
        "departure_date": departure_date,
        "return_date": return_date,
        "adults": adults,
        "party_size": max(1, int(adults or 1)),
        "count": 0,
        "flights": [],
        "cheapest_inr": None,
        "cheapest_total_inr": None,
        "cheapest_per_person_inr": None,
        "sources_tried": sources,
        "no_inventory": no_inventory,
        # Read by nodes.search_transport: an empty result that came from every
        # source failing must not be reported as a healthy provider.
        "degraded": all_failed,
        "errors": errors,
        "note": note,
    }


def render(result: dict[str, Any]) -> str:
    if result.get("source") == AMADEUS:
        return amadeus_provider.render(result)
    return google_provider.render(result)
