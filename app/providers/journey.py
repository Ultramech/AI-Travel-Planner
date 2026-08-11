"""Multi-leg journeys: getting to the gateway, and away from it.

The problem this solves
-----------------------
Every provider in this project is point-to-point. ``search_transport`` called
each one once with ``origin -> destination`` and presented whatever came back as
*the* journey. For any city without its own airport or railway station that is
quietly wrong, because ``geo.resolve_airport`` substitutes a neighbour's airport
and nothing downstream ever mentioned the substitution:

    request:  Shimla -> Mumbai
    answer:   "FLIGHT IXC -> BOM, depart 13:55, ₹16,302"

IXC is Chandigarh, a four-hour drive from Shimla. The traveller was told to be
at an airport in a different state with no plan and no budget for reaching it.
The same happens for Aligarh (AGR, 85 km), Gangtok (IXB, 125 km) and Munnar
(COK, 130 km) — all verified live.

The fix is not a general routing engine. It is to make the leg that was always
implicit *explicit*: resolve the gateway city, price the surface hop to it with
the providers we already have, and carry it through the recommendation, the
budget and the itinerary's arrival time.

A journey therefore has up to three parts:

    access leg   origin -> gateway city        (surface: train or cab)
    main leg     gateway -> arrival gateway    (flight/train/bus, as before)
    egress leg   arrival gateway -> destination (surface: train or cab)

Most trips have no access or egress leg at all, and for those nothing about the
old behaviour changes.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from app.providers import cabs as cabs_provider
from app.providers import geo
from app.providers import trains as trains_provider

logger = logging.getLogger(__name__)

# Below this, the "gateway" is effectively the city itself — a local airport
# transfer, not a leg worth planning or pricing separately.
MIN_LEG_KM = 25.0

# Above this a surface transfer is not a transfer any more; if the gateway is
# this far the substitution is not credible and we say so rather than quietly
# pricing an eight-hour taxi.
MAX_SENSIBLE_LEG_KM = 400.0

# Leg planning runs after the main provider fan-out, so its cost is added
# straight onto the user's wait. The train scraper alone can take 90 s; a leg is
# supporting detail, not the answer, and must not double the response time.
LEG_TIMEOUT = 45.0
ALL_LEGS_TIMEOUT = 75.0


# How early you must physically be at the gateway before departure.
CHECKIN_BUFFER_MIN = {"flight": 120, "train": 30, "bus": 20, "cab": 0}

# Leaving before this is a night departure, not an early start. Below it we say
# so rather than printing "leave at 02:40" as though it were a normal plan.
UNREASONABLE_DEPARTURE_HOUR = 5


def _pretty_minutes(minutes: Optional[int]) -> str:
    if not minutes:
        return ""
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    return f"{hours}h" if hours else f"{mins}m"


def check_connection(
    leg: dict[str, Any],
    departure_clock: Optional[str],
    departure_date: Optional[str] = None,
) -> dict[str, Any]:
    """Can the traveller actually make a service leaving at ``departure_clock``?

    The plan already said "road to Chandigarh, then the 13:55 flight" — but
    nothing checked that a four-hour drive plus two hours of airport buffer
    leaves in time. This returns the latest moment they can set off, and says
    plainly when that moment is the middle of the night or the day before.
    """
    out: dict[str, Any] = {"feasible": None}
    travel = leg.get("travel_minutes")
    if not travel or not departure_clock:
        return out

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", str(departure_clock))
    if not match:
        return out

    depart_min = int(match.group(1)) * 60 + int(match.group(2))
    buffer_min = CHECKIN_BUFFER_MIN.get(leg.get("for_mode", ""), 60)
    leave_min = depart_min - travel - buffer_min

    out["buffer_minutes"] = buffer_min
    out["travel_minutes"] = travel
    out["required_slack_minutes"] = travel + buffer_min

    if leave_min >= UNREASONABLE_DEPARTURE_HOUR * 60:
        out.update({
            "feasible": True,
            "leave_by": f"{leave_min // 60:02d}:{leave_min % 60:02d}",
            "leave_day_offset": 0,
        })
        return out

    if leave_min >= 0:
        out.update({
            "feasible": True,
            "tight": True,
            "leave_by": f"{leave_min // 60:02d}:{leave_min % 60:02d}",
            "leave_day_offset": 0,
            "warning": (
                f"means setting off from {leg['from']} at "
                f"{leave_min // 60:02d}:{leave_min % 60:02d} — a night departure"
            ),
        })
        return out

    # Would have to leave the previous day.
    prev = leave_min + 24 * 60
    out.update({
        "feasible": False,
        "leave_by": f"{prev // 60:02d}:{prev % 60:02d}",
        "leave_day_offset": -1,
        "warning": (
            f"not reachable on the day: {_pretty_minutes(travel)} from "
            f"{leg['from']} plus {buffer_min} min at the gateway means leaving "
            f"the previous day (~{prev // 60:02d}:{prev % 60:02d}), or staying "
            f"overnight in {leg['to']}"
        ),
    })
    if departure_date:
        try:
            prior = datetime.strptime(departure_date, "%Y-%m-%d") - timedelta(days=1)
            out["leave_date"] = prior.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return out


def needs_leg(city: str, gateway: Optional[str]) -> bool:
    """True when ``gateway`` is a genuinely different place from ``city``."""
    if not gateway or not city:
        return False
    return geo.normalise_city(city) != geo.normalise_city(gateway)


async def describe_leg(
    origin: str,
    gateway: str,
    *,
    travel_date: str,
    persons: int,
    role: str,
    for_mode: str,
) -> dict[str, Any]:
    """Cost and describe one surface leg between a city and its gateway.

    Uses the existing cab estimator (always available — it is distance-based)
    and the train scraper (often has something on these short hops). Failures
    degrade to "distance unknown" rather than removing the leg from the plan:
    an unpriced leg the user can see beats an invisible one.
    """
    leg: dict[str, Any] = {
        "role": role,               # "access" | "egress"
        "for_mode": for_mode,       # the main mode this leg exists to serve
        "from": origin,
        "to": gateway,
        "date": travel_date,
        "persons": persons,
        "distance_km": None,
        "cab": None,
        "trains": [],
        "cheapest_inr": None,
        "note": None,
    }

    # Both lookups at once, and never for longer than LEG_TIMEOUT: this runs
    # after the main provider fan-out, so anything spent here lands directly on
    # the user's wait. The cab estimate is the one that must not be lost — it is
    # what puts a number on the leg — so the train scrape is the one that gets
    # dropped if the budget runs out.
    road_task = asyncio.create_task(
        cabs_provider.estimate_road_journey(origin, gateway)
    )
    cab_task = asyncio.create_task(
        cabs_provider.search_cabs(
            origin, gateway, travel_date=travel_date, persons=persons
        )
    )
    rail_task = asyncio.create_task(
        trains_provider.search_trains(origin, gateway, travel_date)
    )
    done, pending = await asyncio.wait(
        {cab_task, rail_task, road_task}, timeout=LEG_TIMEOUT
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if road_task in done and not road_task.cancelled() and not road_task.exception():
        road = road_task.result() or {}
        if road.get("hours"):
            leg["travel_minutes"] = int(round(road["hours"] * 60))
            leg["travel_pretty"] = _pretty_minutes(leg["travel_minutes"])

    cab = None
    if cab_task in done and not cab_task.cancelled():
        error = cab_task.exception()
        if error:
            logger.warning("Access-leg cab estimate failed %s->%s: %s",
                           origin, gateway, error)
        else:
            cab = cab_task.result()
    elif cab_task in pending:
        leg["note"] = "Road-fare estimate timed out for this leg."

    if cab:
        leg["distance_km"] = cab.get("distance_km")
        options = cab.get("options") or []
        if options:
            cheapest = min(options, key=lambda o: o["estimated_fare_inr"])
            leg["cab"] = {
                "vehicle": cheapest["vehicle"],
                "fare_inr": cheapest["estimated_fare_inr"],
                "per_person_inr": cheapest["per_person_inr"],
                "is_estimate": True,
            }
            leg["cheapest_inr"] = cheapest["estimated_fare_inr"]

    distance = leg["distance_km"]
    if distance and distance < MIN_LEG_KM:
        leg["note"] = (
            f"{gateway} is only ~{distance:.0f} km from {origin} — a local transfer."
        )
        leg["trivial"] = True
        return leg

    if distance and distance > MAX_SENSIBLE_LEG_KM:
        # Not a transfer, and quite possibly not reachable by road at all: the
        # railhead resolver offered Kolkata for Port Blair, and this code
        # cheerfully quoted ₹18,310 to drive 1,370 km across the Bay of Bengal.
        # A fare here is worse than no fare, so the estimate is dropped and the
        # gateway is marked unusable rather than merely expensive.
        leg["note"] = (
            f"{gateway} is ~{distance:.0f} km from {origin} — too far to be a "
            f"transfer, and possibly not connected by road at all. Treat "
            f"{leg['for_mode']} via {gateway} as unconfirmed."
        )
        leg["implausible"] = True
        leg["cab"] = None
        leg["cheapest_inr"] = None

    # A train on the access hop is usually cheaper and more reliable than a taxi.
    if rail_task in done and not rail_task.cancelled():
        error = rail_task.exception()
        if error:
            logger.info("Access-leg train search failed %s->%s: %s",
                        origin, gateway, error)
            leg["rail_note"] = f"Train search unavailable for this leg: {error}"
        else:
            rail = rail_task.result()
            leg["trains"] = (rail.get("trains") or [])[:3]
            if not leg["trains"] and rail.get("note"):
                leg["rail_note"] = rail["note"]
    else:
        leg["rail_note"] = "Train search for this leg timed out; road fare shown."

    return leg


async def plan_connections(
    brief: dict[str, Any], modes: list[str]
) -> list[dict[str, Any]]:
    """Work out which access/egress legs this trip actually needs.

    Only builds a leg when the mode that requires it is being searched — there
    is no point pricing a drive to the airport for a trip that will go by train.
    """
    origin = brief.get("origin") or ""
    destination = brief.get("destination") or ""
    travel_date = brief.get("start_date") or ""
    persons = max(1, int(brief.get("persons", 1) or 1))

    wanted: list[tuple[str, str, str, str]] = []  # (role, from, to, for_mode)

    if "flight" in modes:
        origin_gateway = brief.get("origin_airport_city")
        if needs_leg(origin, origin_gateway):
            wanted.append(("access", origin, origin_gateway, "flight"))
        dest_gateway = brief.get("destination_airport_city")
        if needs_leg(destination, dest_gateway):
            wanted.append(("egress", dest_gateway, destination, "flight"))

    if "train" in modes:
        origin_railhead = brief.get("origin_railhead_city")
        if needs_leg(origin, origin_railhead):
            wanted.append(("access", origin, origin_railhead, "train"))
        dest_railhead = brief.get("destination_railhead_city")
        if needs_leg(destination, dest_railhead):
            wanted.append(("egress", dest_railhead, destination, "train"))

    if not wanted:
        return []

    # Legs are independent — an access leg and an egress leg have nothing to say
    # to each other — so they run together under one shared budget.
    tasks = [
        asyncio.ensure_future(
            describe_leg(
                src, dst, travel_date=travel_date, persons=persons,
                role=role, for_mode=mode,
            )
        )
        for role, src, dst, mode in wanted
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=ALL_LEGS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Connecting-leg planning exceeded its %.0fs budget.",
                       ALL_LEGS_TIMEOUT)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    legs: list[dict[str, Any]] = []
    for (role, src, dst, mode), task in zip(wanted, tasks):
        if task.cancelled() or task.exception() is not None:
            # Never drop the leg just because pricing it failed. An unpriced leg
            # the traveller can see beats an invisible one they discover at the
            # airport.
            legs.append({
                "role": role, "for_mode": mode, "from": src, "to": dst,
                "date": travel_date, "persons": persons,
                "distance_km": None, "cab": None, "trains": [],
                "cheapest_inr": None,
                "note": "Could not price this leg, but you still have to travel it.",
            })
        else:
            legs.append(task.result())
    return legs


def leg_cost(legs: list[dict[str, Any]], mode: str) -> float:
    """Total surface cost that using ``mode`` implies, in INR.

    Implausible legs contribute nothing: they carry no fare precisely because
    quoting one would be inventing a road that may not exist.
    """
    return sum(
        leg.get("cheapest_inr") or 0
        for leg in legs
        if leg.get("for_mode") == mode
        and not leg.get("trivial")
        and not leg.get("implausible")
    )


def render(legs: list[dict[str, Any]]) -> str:
    """Human-readable summary of the connecting legs."""
    real = [leg for leg in legs if not leg.get("trivial")]
    if not real:
        return ""

    lines = ["🔗 **Connecting legs you also need to travel**\n"]
    for leg in real:
        distance = leg.get("distance_km")
        head = (
            f"🔸 **{leg['from']} → {leg['to']}**"
            + (f" · ~{distance:.0f} km" if distance else "")
            + f" · needed to reach the {leg['for_mode']}"
        )
        if leg.get("implausible"):
            lines.append(f"{head} — ⚠️ **not a realistic transfer**")
            lines.append(f"   {leg['note']}")
            continue
        lines.append(head)
        if leg.get("cab"):
            cab = leg["cab"]
            lines.append(
                f"   🚖 {cab['vehicle']} ~₹{cab['fare_inr']:,.0f} "
                f"(~₹{cab['per_person_inr']:,.0f} per person, estimate)"
            )
        for train in leg.get("trains") or []:
            lines.append(
                f"   🚆 {train['train_no']} {train['train_name']} · "
                f"{train['from']} {train['departs']} → {train['to']} {train['arrives']}"
            )
        if leg.get("note"):
            lines.append(f"   ⚠️ {leg['note']}")
    lines.append(
        "\n_These legs are part of your journey and are included in the "
        "transport total below._"
    )
    return "\n".join(lines)


def facts(legs: list[dict[str, Any]]) -> list[str]:
    """Compact lines describing the legs for the recommendation prompt."""
    out: list[str] = []
    for leg in legs:
        if leg.get("trivial"):
            continue
        distance = leg.get("distance_km")
        cost = leg.get("cheapest_inr")
        if leg.get("implausible"):
            out.append(
                f"UNUSABLE_GATEWAY ({leg['for_mode']}) {leg['from']} -> {leg['to']}"
                + (f" ~{distance:.0f}km" if distance else "")
                + f" | {leg['note']}"
            )
            continue
        out.append(
            f"CONNECTING_LEG ({leg['for_mode']}) {leg['from']} -> {leg['to']}"
            + (f" ~{distance:.0f}km" if distance else "")
            + (f" ~₹{cost:,.0f} for the group by road" if cost else " cost unknown")
            + (f" | {len(leg.get('trains') or [])} train(s) also run this leg"
               if leg.get("trains") else "")
        )
    return out
