"""Graph nodes.

Every node here is:
  * **async** — the old nodes made blocking network calls inside a sync graph,
    so one Selenium scrape froze the entire process;
  * **UI-free** — the old nodes called ``st.subheader`` / ``st.markdown`` /
    ``st.write`` from inside business logic and one called ``input()``, which
    made them unusable outside a Streamlit script and unusable in a worker;
  * **partial-update returning** — only the keys a node actually changed, so
    the state reducers can do their job;
  * **honest about failure** — errors are recorded in ``errors`` rather than
    stringified into an LLM prompt as if they were data.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.config import settings
from app.core.bus import emit
from app.core.decision import Candidate
from app.core import decision
from app.core.events import EventType
from app.core.resilience import gather_providers
from app.graph.state import TripState
from app.providers import buses as buses_provider
from app.providers import cabs as cabs_provider
from app.providers import extractor
from app.providers import flex_dates
from app.providers import flights_router as flights_provider
from app.providers import hotels as hotels_provider
from app.providers import itinerary as itinerary_provider
from app.providers import journey as journey_provider
from app.providers import llm as llm_provider
from app.providers import trains as trains_provider
from app.providers import weather as weather_provider

logger = logging.getLogger(__name__)

MAX_REVISIONS = 3


def _error(state: TripState, node: str, exc: Exception) -> dict[str, Any]:
    logger.exception("Node %s failed", node)
    return {
        "errors": [
            {
                "node": node,
                "error": str(exc),
                "type": type(exc).__name__,
                "at": datetime.utcnow().isoformat(),
            }
        ],
        "messages": [AIMessage(content=f"⚠️ {node} could not complete: {exc}")],
    }


async def _emit(state: TripState, type_: EventType, message: str, node: str, **payload: Any) -> None:
    trip_id = state.get("trip_id")
    if trip_id:
        await emit(trip_id, type_, message, node=node, **payload)


# ---------------------------------------------------------------------------
# 1. Understand the request
# ---------------------------------------------------------------------------

async def extract_brief(state: TripState) -> dict[str, Any]:
    """Normalise the request into a validated ``TripBrief``."""
    await _emit(state, EventType.NODE_STARTED, "Understanding your request…", "extract_brief")
    try:
        existing = state.get("brief") or {}
        brief = await extractor.build_brief(
            user_input=state.get("raw_request", ""),
            origin=existing.get("origin"),
            destination=existing.get("destination"),
            start_date=existing.get("start_date"),
            end_date=existing.get("end_date"),
            persons=existing.get("persons"),
            preferences=existing.get("preferences"),
            budget_level=existing.get("budget_level"),
            travel_class=existing.get("travel_class"),
            special_requirements=existing.get("special_requirements"),
        )
        data = brief.to_dict()

        note = ""
        if brief.is_past:
            note = " (note: this start date is in the past, so live availability may be empty)"

        await _emit(
            state, EventType.NODE_COMPLETED,
            f"Planning {brief.origin} → {brief.destination}, {brief.duration} day(s)",
            "extract_brief", brief=data,
        )
        return {
            "brief": data,
            "arrival_date": brief.start_date,
            "messages": [
                AIMessage(
                    content=(
                        f"📋 {brief.origin} → {brief.destination} · "
                        f"{brief.start_date} to {brief.end_date} · "
                        f"{brief.duration} day(s) · {brief.persons} traveller(s){note}"
                    )
                )
            ],
        }
    except Exception as exc:  # noqa: BLE001
        await _emit(state, EventType.NODE_FAILED, str(exc), "extract_brief")
        return _error(state, "extract_brief", exc)


# ---------------------------------------------------------------------------
# 2. Decide which transport modes are worth querying
# ---------------------------------------------------------------------------

# A mode is only ever skipped for a reason that can be written down. Anything
# the LLM merely *thinks* is a bad idea still gets searched — its opinion ranks
# results, it does not decide what the user is allowed to see.
#
# Flight and cab are cheap to search (a keyless HTTP call, and a hard-cached
# distance estimate). Train and bus each cost a Selenium scrape, so they are the
# only ones where a pre-filter buys anything real.
MIN_FLIGHT_KM = 250.0
MAX_CAB_KM = 900.0
MAX_BUS_KM = 1500.0
MIN_INTERCITY_KM = 20.0


async def plan_transport_strategy(state: TripState) -> dict[str, Any]:
    """Pick which modes to search — then we search them all in parallel.

    The original asked the LLM to commit to exactly ONE mode *before seeing any
    data*, then searched only that. If it guessed "bus_or_cab_only" for
    Delhi→Chennai the user simply never saw the flights.

    Narrowing that to a "cheap feasibility filter" was not enough: on
    Aligarh→Patiala the model dropped both flight and cab before anything had
    been looked up, and the recommendation then stated "no flights are
    available" — a claim about AGR→IXC that nothing had checked. The filter is
    now deterministic and distance-based, the LLM can only *add* modes back, and
    whatever is skipped is recorded with a reason so a gap in the search can
    never be reported as a gap in the world.
    """
    await _emit(state, EventType.NODE_STARTED, "Deciding which transport to search…",
                "plan_transport_strategy")
    brief = state.get("brief") or {}
    feedback = (state.get("review_feedback") or "").strip()

    modes = {"flight", "train", "bus", "cab"}
    skipped: dict[str, str] = {}

    if not (brief.get("origin_iata") and brief.get("destination_iata")):
        modes.discard("flight")
        unresolved = (
            brief.get("origin") if not brief.get("origin_iata")
            else brief.get("destination")
        )
        # "No airport could be resolved" reads as "there is no airport", which
        # may be false — resolution also fails when the LLM behind it is down.
        # Seen for real: a rate-limited model turned Gangtok into a city with
        # apparently no air service at all.
        skipped["flight"] = (
            f"could not determine which airport serves {unresolved} "
            f"(it may have none, or airport lookup was unavailable)"
        )
    elif brief["origin_iata"] == brief["destination_iata"]:
        modes.discard("flight")
        skipped["flight"] = (
            f"both cities use the same airport ({brief['origin_iata']})"
        )

    # One hard-cached LLM call, shared with the cab provider — this is the same
    # lookup ``search_cabs`` makes, so it costs nothing extra in practice.
    distance: Optional[float] = None
    try:
        distance = await cabs_provider.estimate_distance_km(
            brief.get("origin", ""), brief.get("destination", "")
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("Distance unknown for strategy (%s); searching everything.", exc)

    if distance:
        if "flight" in modes and distance < MIN_FLIGHT_KM:
            modes.discard("flight")
            skipped["flight"] = f"only ~{distance:.0f} km — road or rail is faster door to door"
        if distance > MAX_CAB_KM:
            modes.discard("cab")
            skipped["cab"] = f"~{distance:.0f} km is too far to drive"
        if distance > MAX_BUS_KM:
            modes.discard("bus")
            skipped["bus"] = f"~{distance:.0f} km is beyond any scheduled bus"
        if distance < MIN_INTERCITY_KM:
            for mode in ("train", "bus", "flight"):
                modes.discard(mode)
                skipped.setdefault(mode, f"only ~{distance:.0f} km — a local transfer")

    # The user's own words outrank every rule above.
    if feedback:
        for mode in ("flight", "train", "bus", "cab"):
            if mode in feedback.lower() and mode not in modes:
                modes.add(mode)
                skipped.pop(mode, None)

    if not modes:  # never leave the traveller with nothing
        modes = {"train", "cab"}
        for mode in modes:
            skipped.pop(mode, None)

    # The skips are carried structurally in ``skipped`` and rendered separately;
    # repeating them here printed the same sentence twice in the UI.
    reason = (
        f"~{distance:.0f} km by road" if distance else "road distance unknown"
    ) + f" — searching {', '.join(sorted(modes))}"

    strategy = {"modes": sorted(modes), "reason": reason[:400],
                "skipped": skipped, "distance_km": distance}
    await _emit(state, EventType.NODE_COMPLETED, f"Searching: {', '.join(sorted(modes))}",
                "plan_transport_strategy", **strategy)
    return {
        "transport_strategy": strategy,
        "messages": [AIMessage(content=f"🧭 {reason}")],
    }


# ---------------------------------------------------------------------------
# 3. Search every chosen mode concurrently
# ---------------------------------------------------------------------------

async def search_transport(state: TripState) -> dict[str, Any]:
    """Fan out to all chosen providers at once, isolating failures.

    The original ran each leg sequentially and let a single slow scrape
    (etrain could take 40s) dominate the request. Concurrent execution turns
    "sum of all providers" into "slowest single provider", and one dead upstream
    no longer removes the others from the result.
    """
    await _emit(state, EventType.NODE_STARTED, "Searching transport options…", "search_transport")
    brief = state.get("brief") or {}
    strategy = state.get("transport_strategy") or {}
    modes = strategy.get("modes") or ["train", "bus"]
    force_refresh = bool(state.get("review_feedback"))

    origin = brief.get("origin", "")
    destination = brief.get("destination", "")
    start_date = brief.get("start_date", "")
    end_date = brief.get("end_date")
    persons = int(brief.get("persons", 1) or 1)

    # The return leg is now searched explicitly below, so the outbound must be
    # priced ONE WAY. Asking for a round trip here and then adding a separate
    # return fare counted the journey home twice: Shimla→Mumbai came out at
    # ₹34,656 where the outbound alone is ₹16,302 + ₹1,970 of road.
    separate_return = bool(
        settings.plan_return_journey and end_date and end_date != start_date
    )

    tasks: dict[str, Any] = {}
    if "flight" in modes and brief.get("origin_iata") and brief.get("destination_iata"):
        tasks["flight"] = flights_provider.search_flights(
            brief["origin_iata"], brief["destination_iata"], start_date,
            return_date=None if separate_return else end_date,
            adults=persons, skip_cache=force_refresh,
        )
    if "train" in modes:
        # Search from the railhead when the city has no station of its own.
        # Querying "Gangtok" directly can only ever answer "not in the dataset",
        # which is true and useless — the usable answer is New Jalpaiguri, and
        # the leg to it is priced separately as a connecting leg.
        train_origin = brief.get("origin_railhead_city") or origin
        train_dest = brief.get("destination_railhead_city") or destination
        tasks["train"] = trains_provider.search_trains(
            train_origin, train_dest, start_date, persons=persons,
            skip_cache=force_refresh,
        )
    if "bus" in modes:
        tasks["bus"] = buses_provider.search_buses(
            origin, destination, start_date, skip_cache=force_refresh
        )
    if "cab" in modes:
        tasks["cab"] = cabs_provider.search_cabs(
            origin, destination, travel_date=start_date, persons=persons,
            skip_cache=force_refresh,
        )

    # The return leg. Nothing planned it before, so every trip quietly assumed
    # the traveller stayed. Flights are searched because they are cheap (one
    # HTTP call) and are usually the largest line in the budget; the scrapers
    # are left out of the return so the first review is not held up behind four
    # more browser sessions.
    return_date = end_date
    if (
        separate_return
        and "flight" in modes
        and brief.get("origin_iata")
        and brief.get("destination_iata")
    ):
        tasks["return_flight"] = flights_provider.search_flights(
            brief["destination_iata"], brief["origin_iata"], return_date,
            adults=persons, skip_cache=force_refresh,
        )

    if not tasks:
        return {
            "errors": [{"node": "search_transport", "error": "no searchable transport modes"}],
            "messages": [AIMessage(content="⚠️ No transport modes could be searched.")],
        }

    results = await gather_providers(tasks, overall_timeout=120.0)

    options: dict[str, Any] = {}
    # Seeded with the deliberate skips so the health map covers all four modes:
    # a mode missing from this dict used to be indistinguishable from one that
    # was searched and found nothing.
    health: dict[str, Any] = {
        mode: f"not searched: {why}"
        for mode, why in (strategy.get("skipped") or {}).items()
    }
    errors: list[dict[str, Any]] = []
    lines: list[str] = []

    # Split the return leg back out before the outbound bookkeeping runs, so it
    # cannot be mistaken for an outbound option or counted twice in the budget.
    return_result = results.pop("return_flight", None)
    return_options: dict[str, Any] = {}
    if return_result and return_result["ok"]:
        return_options["flight"] = return_result["data"]
        health["return_flight"] = (
            "ok" if return_result["data"].get("flights")
            else f"none: {return_result['data'].get('note', 'no return flights')}"
        )
    elif return_result:
        health["return_flight"] = f"failed: {return_result['error']}"
        errors.append({"node": "search_transport", "provider": "return_flight",
                       "error": return_result["error"]})

    renderers = {
        "flight": flights_provider.render,
        "train": trains_provider.render,
        "bus": buses_provider.render,
        "cab": cabs_provider.render,
    }

    for mode, outcome in results.items():
        if outcome["ok"]:
            data = outcome["data"]
            options[mode] = data
            # A provider can return successfully and still be broken: the flight
            # router catches every source failing and hands back an empty result.
            # Reporting that as "ok" made a total outage invisible in both the
            # UI and the provider audit log.
            if isinstance(data, dict) and data.get("degraded"):
                health[mode] = f"degraded: {data.get('note', 'all sources failed')}"
                errors.append({"node": "search_transport", "provider": mode,
                               "error": data.get("note", "all sources failed")})
            else:
                health[mode] = "ok"
            lines.append(renderers[mode](data))
            await _emit(state, EventType.PROVIDER_CALLED, f"{mode}: found options",
                        "search_transport", mode=mode, ok=True)
        else:
            health[mode] = f"failed: {outcome['error']}"
            errors.append({"node": "search_transport", "provider": mode,
                           "error": outcome["error"]})
            lines.append(f"⚠️ {mode.title()} search unavailable: {outcome['error']}")
            await _emit(state, EventType.PROVIDER_CALLED, f"{mode}: {outcome['error']}",
                        "search_transport", mode=mode, ok=False)

    # Any leg the traveller must cover to reach the airport/railhead this trip
    # actually departs from. Runs after the fan-out because it only matters for
    # the modes that came back with something.
    usable = [m for m in modes if (options.get(m) or {}).get(
        {"flight": "flights", "train": "trains", "bus": "buses", "cab": "options"}[m]
    )]
    legs: list[dict[str, Any]] = []
    try:
        legs = await journey_provider.plan_connections(brief, usable or list(modes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Connecting-leg planning failed: %s", exc)
        errors.append({"node": "search_transport", "provider": "journey",
                       "error": str(exc)})

    if legs:
        rendered = journey_provider.render(legs)
        if rendered:
            lines.append(rendered)
        await _emit(state, EventType.NODE_COMPLETED,
                    f"{len(legs)} connecting leg(s) needed", "search_transport")

    # Counts only what was actually queried — the skipped entries seeded above
    # would otherwise inflate the denominator with providers nobody called.
    searched = {m: v for m, v in health.items() if not str(v).startswith("not searched")}
    await _emit(state, EventType.NODE_COMPLETED,
                f"{sum(1 for v in searched.values() if v == 'ok')}/{len(searched)} "
                "providers responded",
                "search_transport")

    # Cheap enough to always answer "would leaving a day earlier help?" — seven
    # keyless probes, all cached, ~3 s. Runs after the fan-out so it can be
    # skipped when there are no flights to compare.
    flexibility: dict[str, Any] = {}
    if (
        settings.flexible_date_window
        and (options.get("flight") or {}).get("flights")
        and brief.get("origin_iata") and brief.get("destination_iata")
    ):
        try:
            flexibility = await flex_dates.scan(
                brief["origin_iata"], brief["destination_iata"], start_date,
                adults=persons, window=settings.flexible_date_window,
                earliest=datetime.utcnow().strftime("%Y-%m-%d"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Flexible-date scan unavailable: %s", exc)
        rendered_flex = flex_dates.render(flexibility)
        if rendered_flex:
            lines.append(rendered_flex)

    if return_options.get("flight", {}).get("flights"):
        lines.append(
            "🔙 **Return journey** " + brief.get("destination", "")
            + " → " + brief.get("origin", "") + f" on {return_date}\n"
            + flights_provider.render(return_options["flight"])
        )

    return {
        "transport_options": options,
        "return_options": return_options,
        "date_flexibility": flexibility,
        "connecting_legs": legs,
        "provider_health": health,
        "errors": errors,
        "messages": [AIMessage(content="\n\n".join(lines))],
    }


# ---------------------------------------------------------------------------
# 4. Recommend, from real data
# ---------------------------------------------------------------------------

_ARRIVAL_RE = re.compile(
    r"ARRIVAL_TIME\s*[:=]?\s*([01]?\d|2[0-3])[:.]([0-5]\d)", re.IGNORECASE
)
_ARRIVAL_DATE_RE = re.compile(
    r"ARRIVAL_DATE\s*[:=]?\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
)

# "24:30H" (etrain), "3h 0m" / "45m" (flights), "18 hrs" (free text).
_DUR_COLON_RE = re.compile(r"^\s*(\d{1,3})\s*[:.]\s*([0-5]\d)\s*h?\s*$", re.IGNORECASE)
_DUR_HM_RE = re.compile(
    r"(?:(\d{1,3})\s*(?:h|hr|hrs|hour|hours))?\s*(?:(\d{1,2})\s*(?:m|min|mins|minutes))?",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def parse_duration_minutes(text: Any) -> Optional[int]:
    """Minutes from the several duration formats the providers hand back.

    etrain writes a 24-and-a-half-hour journey as ``"24:30H"``. Read as a clock
    time that is nonsense, which is one reason the value was simply dropped
    before reaching the comparison prompt.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text) or None
    raw = str(text).strip()
    if not raw:
        return None

    match = _DUR_COLON_RE.match(raw)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))

    match = _DUR_HM_RE.fullmatch(raw.strip())
    if match and (match.group(1) or match.group(2)):
        return int(match.group(1) or 0) * 60 + int(match.group(2) or 0)
    return None


def pretty_duration(minutes: Optional[int]) -> str:
    if not minutes:
        return ""
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    return f"{hours}h" if hours else f"{mins}m"


def _clock_minutes(text: Any) -> Optional[int]:
    match = _CLOCK_RE.search(str(text or ""))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def derive_arrival(
    start_date: str,
    *,
    departs: Any = None,
    arrives: Any = None,
    duration_minutes: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(arrival_date, arrival_time)`` for a leg leaving on ``start_date``.

    Duration is authoritative when we have it, because the overnight heuristic
    ("arrival clock < departure clock means +1 day") silently truncates anything
    longer than 24 hours — and the Aligarh→Mumbai train is 24 h 30 m, departing
    15:45 and arriving 16:15, which that heuristic reads as *thirty minutes*.
    """
    try:
        start = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None, None

    departure_minutes = _clock_minutes(departs)

    if duration_minutes and departure_minutes is not None:
        arrival = start + timedelta(minutes=departure_minutes + duration_minutes)
        return arrival.strftime("%Y-%m-%d"), arrival.strftime("%H:%M")

    arrival_minutes = _clock_minutes(arrives)
    if arrival_minutes is None:
        return None, None

    if duration_minutes:
        arrival = start + timedelta(minutes=duration_minutes)
        return arrival.strftime("%Y-%m-%d"), f"{arrival_minutes // 60:02d}:{arrival_minutes % 60:02d}"

    days = 0
    if departure_minutes is not None and arrival_minutes < departure_minutes:
        days = 1  # crossed midnight
    arrival = start + timedelta(days=days)
    return arrival.strftime("%Y-%m-%d"), f"{arrival_minutes // 60:02d}:{arrival_minutes % 60:02d}"


def _fallback_arrival(options: dict[str, Any]) -> Optional[str]:
    """Best-effort arrival time straight from structured provider data."""
    flights = (options.get("flight") or {}).get("flights") or []
    if flights and flights[0].get("arrival"):
        try:
            return datetime.fromisoformat(flights[0]["arrival"]).strftime("%H:%M")
        except (ValueError, TypeError):
            pass
    trains = (options.get("train") or {}).get("trains") or []
    for train in trains:
        match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", str(train.get("arrives", "")))
        if match:
            return f"{match.group(1).zfill(2)}:{match.group(2)}"
    buses = (options.get("bus") or {}).get("buses") or []
    for bus in buses:
        if bus.get("arrival"):
            return bus["arrival"]
    return None


def _gateway_note(brief: dict[str, Any]) -> str:
    """Tell the model, in words, that the codes are not the user's cities.

    Without this the model sees ``FLIGHT IXC → BOM`` under a heading that says
    "Shimla to Mumbai" and reports it as a Shimla departure. Verified: that is
    exactly what it did.
    """
    notes: list[str] = []
    if not brief.get("origin_airport_exact", True) and brief.get("origin_airport_city"):
        notes.append(
            f"{brief.get('origin')} has no airport of its own — flights below "
            f"depart from {brief['origin_airport_city']} "
            f"({brief.get('origin_iata')})."
        )
    if not brief.get("destination_airport_exact", True) and brief.get(
        "destination_airport_city"
    ):
        notes.append(
            f"{brief.get('destination')} has no airport of its own — flights below "
            f"land at {brief['destination_airport_city']} "
            f"({brief.get('destination_iata')})."
        )
    if not brief.get("origin_railhead_exact", True) and brief.get("origin_railhead_city"):
        notes.append(
            f"{brief.get('origin')} has no railway station — trains below start "
            f"from {brief['origin_railhead_city']}."
        )
    if not brief.get("destination_railhead_exact", True) and brief.get(
        "destination_railhead_city"
    ):
        notes.append(
            f"{brief.get('destination')} has no railway station — trains below "
            f"terminate at {brief['destination_railhead_city']}."
        )
    if not notes:
        return ""
    return (
        "\nIMPORTANT — THIS IS NOT A DIRECT JOURNEY:\n"
        + "\n".join(f"- {n}" for n in notes)
        + "\nSay this plainly in your answer. Never present a gateway departure "
        "as though the traveller boards in their own city.\n"
    )


def _fallback_arrival_date(
    options: dict[str, Any], start_date: str
) -> Optional[str]:
    """Structured-data arrival date, used when the model does not give one."""
    flights = (options.get("flight") or {}).get("flights") or []
    if flights and flights[0].get("arrival"):
        return str(flights[0]["arrival"])[:10] or None
    trains = (options.get("train") or {}).get("trains") or []
    for train in trains:
        arrival_date, _ = derive_arrival(
            start_date,
            departs=train.get("departs"),
            arrives=train.get("arrives"),
            duration_minutes=parse_duration_minutes(train.get("duration")),
        )
        if arrival_date:
            return arrival_date
    buses = (options.get("bus") or {}).get("buses") or []
    for bus in buses:
        arrival_date, _ = derive_arrival(
            start_date, departs=bus.get("departure"), arrives=bus.get("arrival")
        )
        if arrival_date:
            return arrival_date
    return None


def _connection_facts(
    options: dict[str, Any],
    legs: list[dict[str, Any]],
    served: set[str],
    start_date: str,
) -> list[str]:
    """Per-option "you must leave X by HH:MM" lines, and what is unmakeable.

    Naming a connecting leg was only half the job. A four-hour drive to
    Chandigarh plus two hours of airport buffer makes an 06:30 departure
    impossible without travelling the day before, and nothing said so.
    """
    access = [
        leg for leg in legs
        if leg.get("role") == "access"
        and leg["for_mode"] in served
        and not leg.get("trivial")
        and not leg.get("implausible")
        and leg.get("travel_minutes")
    ]
    if not access:
        return []

    out: list[str] = []
    for leg in access:
        mode = leg["for_mode"]
        departures: list[tuple[str, str]] = []
        if mode == "flight":
            for f in ((options.get("flight") or {}).get("flights") or [])[:4]:
                clock = (f.get("departure") or "")[11:16]
                if clock:
                    departures.append((f"{f.get('route', 'flight')} {clock}", clock))
        elif mode == "train":
            for t in ((options.get("train") or {}).get("trains") or [])[:4]:
                if t.get("departs"):
                    departures.append(
                        (f"train {t['train_no']} {t['departs']}", str(t["departs"]))
                    )

        out.append(
            f"CONNECTION_BASIS {leg['from']} → {leg['to']}: "
            f"{leg.get('travel_pretty') or '?'} on the road plus "
            f"{journey_provider.CHECKIN_BUFFER_MIN.get(mode, 60)} min at the "
            f"{'airport' if mode == 'flight' else 'station'}."
        )
        for label, clock in departures:
            check = journey_provider.check_connection(leg, clock, start_date)
            if check.get("feasible") is True:
                out.append(
                    f"CONNECTION OK for {label}: leave {leg['from']} by "
                    f"{check['leave_by']}"
                    + (f" — {check['warning']}" if check.get("warning") else "")
                )
            elif check.get("feasible") is False:
                out.append(
                    f"CONNECTION IMPOSSIBLE for {label}: {check['warning']}"
                )
    return out


def _compose_summary(
    brief: dict[str, Any],
    options: dict[str, Any],
    legs: list[dict[str, Any]],
    persons: int,
    start_date: str,
) -> str:
    """Write the recommendation from structured data, with no LLM.

    Plainer prose than the model's, and honest about being a fallback — but it
    carries the same facts, so a quota exhaustion degrades the wording rather
    than deleting the answer.
    """
    lines = [
        f"**Transport {brief.get('origin')} → {brief.get('destination')}** "
        f"for {persons} traveller(s) on {start_date}",
        "",
        "_Composed directly from the search results — the summarising model was "
        "unavailable, so this lists the cheapest option per mode rather than "
        "making a judgement between them._",
        "",
    ]

    surface = journey_provider.render(legs)
    if surface:
        lines += [surface, ""]

    flights = (options.get("flight") or {}).get("flights") or []
    if flights:
        best = flights[0]
        lines.append(
            f"✈️ **Flight** {best.get('route')} · dep {best.get('departure')} · "
            f"arr {best.get('arrival')} · {best.get('duration') or '?'} · "
            f"₹{(best.get('price_total_inr') or 0):,.0f} total for {persons}"
        )

    train_data = options.get("train") or {}
    trains = train_data.get("trains") or []
    if trains:
        best = trains[0]
        minutes = parse_duration_minutes(best.get("duration"))
        arr_date, _ = derive_arrival(
            start_date, departs=best.get("departs"), arrives=best.get("arrives"),
            duration_minutes=minutes,
        )
        fares = train_data.get("fares") or {}
        cheapest = min(fares.values(), key=lambda f: f["total_inr"]) if fares else None
        lines.append(
            f"🚆 **Train** {best['train_no']} {best['train_name']} · "
            f"{best['from']} {best['departs']} → {best['to']} {best['arrives']}"
            + (f" ({arr_date})" if arr_date and arr_date != start_date else "")
            + f" · {pretty_duration(minutes) or '?'}"
            + (f" · from ₹{cheapest['total_inr']:,.0f} total in {cheapest['class']}"
               if cheapest else "")
        )

    buses = (options.get("bus") or {}).get("buses") or []
    if buses:
        best = buses[0]
        fare = best.get("fare_inr")
        lines.append(
            f"🚌 **Bus** {best.get('operator')} · dep {best.get('departure')} → "
            f"arr {best.get('arrival')}"
            + (f" · ₹{fare * persons:,.0f} total for {persons}"
               if isinstance(fare, (int, float)) else "")
        )

    cabs = (options.get("cab") or {}).get("options") or []
    if cabs:
        best = min(cabs, key=lambda c: c["estimated_fare_inr"])
        lines.append(
            f"🚖 **Cab** {best['vehicle']} · ~₹{best['estimated_fare_inr']:,.0f} "
            f"total (estimate)"
        )

    cost = _cheapest_transport_inr(options, legs, persons)
    if cost:
        lines += ["", f"**Cheapest complete option: ₹{cost:,.0f}** for the group, "
                      "connecting legs included."]
    return "\n".join(lines)


def build_candidates(
    options: dict[str, Any],
    legs: list[dict[str, Any]],
    persons: int,
    start_date: str,
) -> list[Candidate]:
    """Turn raw provider output into comparable door-to-door candidates.

    "Comparable" is the whole point: a flight fare excludes the drive to the
    airport, a train fare is per class, and a bus fare is per person. Until all
    of that is folded into one group cost and one door-to-door duration, ranking
    them against each other is meaningless.
    """
    out: list[Candidate] = []

    def access_for(mode: str) -> tuple[float, int]:
        cost = journey_provider.leg_cost(legs, mode)
        minutes = sum(
            (leg.get("travel_minutes") or 0)
            + journey_provider.CHECKIN_BUFFER_MIN.get(mode, 0)
            for leg in legs
            if leg.get("for_mode") == mode
            and not leg.get("trivial") and not leg.get("implausible")
        )
        return cost, minutes

    flight_cost, flight_minutes = access_for("flight")
    for f in ((options.get("flight") or {}).get("flights") or [])[:4]:
        price = f.get("price_total_inr") or f.get("price_inr")
        if not price:
            continue
        depart = (f.get("departure") or "")[11:16] or None
        air_minutes = parse_duration_minutes(f.get("duration")) or 0
        out.append(Candidate(
            mode="flight",
            label=f"{f.get('route', 'flight')} · {depart or '?'}",
            total_inr=price + flight_cost,
            total_minutes=(air_minutes + flight_minutes) or None,
            depart_clock=depart,
            arrive_date=(f.get("arrival") or "")[:10] or None,
            arrive_clock=(f.get("arrival") or "")[11:16] or None,
            leg_cost_inr=flight_cost,
            stops=int(f.get("stops") or 0),
            self_transfer=bool(f.get("self_transfer")),
            arrives_next_day=bool(
                (f.get("arrival") or "")[:10] and (f.get("arrival") or "")[:10] != start_date
            ),
            notes=[f["note"]] if f.get("note") else [],
        ))

    train_data = options.get("train") or {}
    train_cost, train_minutes = access_for("train")
    fares = train_data.get("fares") or {}
    cheapest_fare = min(fares.values(), key=lambda x: x["total_inr"]) if fares else None
    for t in (train_data.get("trains") or [])[:4]:
        if not cheapest_fare:
            continue
        minutes = parse_duration_minutes(t.get("duration"))
        arr_date, arr_clock = derive_arrival(
            start_date, departs=t.get("departs"), arrives=t.get("arrives"),
            duration_minutes=minutes,
        )
        out.append(Candidate(
            mode="train",
            label=f"{t['train_no']} {t['train_name']} · {t.get('departs')}",
            total_inr=cheapest_fare["total_inr"] + train_cost,
            total_minutes=((minutes or 0) + train_minutes) or None,
            depart_clock=str(t.get("departs")),
            arrive_date=arr_date,
            arrive_clock=arr_clock,
            leg_cost_inr=train_cost,
            price_is_estimate=True,
            starts_at_night=_is_night(t.get("departs")),
            arrives_next_day=bool(arr_date and arr_date != start_date),
            notes=[f"fare shown is {cheapest_fare['class']} class, estimated"],
        ))

    for b in ((options.get("bus") or {}).get("buses") or [])[:4]:
        fare = b.get("fare_inr")
        if not isinstance(fare, (int, float)):
            continue
        arr_date, _ = derive_arrival(
            start_date, departs=b.get("departure"), arrives=b.get("arrival")
        )
        out.append(Candidate(
            mode="bus",
            label=f"{b.get('operator', 'bus')} · {b.get('departure')}",
            total_inr=fare * persons,   # bus fares are quoted per seat
            total_minutes=None,
            depart_clock=b.get("departure"),
            arrive_date=arr_date,
            arrive_clock=b.get("arrival"),
            starts_at_night=_is_night(b.get("departure")),
            arrives_next_day=bool(arr_date and arr_date != start_date),
        ))

    cab_data = options.get("cab") or {}
    for c in (cab_data.get("options") or [])[:2]:
        out.append(Candidate(
            mode="cab",
            label=f"{c['vehicle']} cab",
            total_inr=c["estimated_fare_inr"],
            total_minutes=None,
            price_is_estimate=True,
            notes=[f"~{cab_data.get('distance_km')} km, distance-based estimate"],
        ))

    return out


def _is_night(clock: Any) -> bool:
    minutes = _clock_minutes(clock)
    return minutes is not None and minutes < 5 * 60


def _cheapest_transport_inr(
    options: dict[str, Any], legs: list[dict[str, Any]], persons: int
) -> Optional[float]:
    """Cheapest complete group cost across modes, connecting legs included.

    A deterministic floor under the model's own arithmetic: if it declines to
    give a number, the budget still gets one rather than reporting a total that
    omits transport entirely.
    """
    candidates: list[float] = []

    flights = (options.get("flight") or {}).get("flights") or []
    if flights:
        cheapest = min(
            (f.get("price_total_inr") or f.get("price_inr") or 0) for f in flights
        )
        if cheapest:
            candidates.append(cheapest + journey_provider.leg_cost(legs, "flight"))

    train_fares = (options.get("train") or {}).get("fares") or {}
    if train_fares and (options.get("train") or {}).get("trains"):
        cheapest = min(f["total_inr"] for f in train_fares.values())
        candidates.append(cheapest + journey_provider.leg_cost(legs, "train"))

    buses = (options.get("bus") or {}).get("buses") or []
    fares = [b["fare_inr"] for b in buses if isinstance(b.get("fare_inr"), (int, float))]
    if fares:
        candidates.append(min(fares) * persons)  # bus fares are per person

    cabs = (options.get("cab") or {}).get("options") or []
    if cabs:
        candidates.append(min(c["estimated_fare_inr"] for c in cabs))

    return round(min(candidates), 2) if candidates else None


async def recommend_transport(state: TripState) -> dict[str, Any]:
    """Compare the real options and recommend one, with an arrival time."""
    await _emit(state, EventType.NODE_STARTED, "Comparing options…", "recommend_transport")
    brief = state.get("brief") or {}
    options = state.get("transport_options") or {}
    feedback = (state.get("review_feedback") or "").strip()

    if not options:
        return {
            "transport_summary": "No transport options could be retrieved.",
            "messages": [AIMessage(content="⚠️ No transport options available.")],
        }

    persons = max(1, int(brief.get("persons", 1) or 1))
    start_date = brief.get("start_date", "")
    legs = state.get("connecting_legs") or []

    # Feed the LLM compact structured facts, not a wall of prior chat messages.
    # Every line carries an explicit journey time and says whether its price is
    # per person or for the group — omitting either is how a 24 h 30 m train
    # came out looking like a 30-minute hop, and how group fares got doubled.
    facts: list[str] = []
    for mode, data in options.items():
        if mode == "flight":
            for f in (data.get("flights") or [])[:4]:
                total = f.get("price_total_inr") or f.get("price_inr")
                facts.append(
                    f"FLIGHT {f['route']} dep {f['departure']} arr {f['arrival']} "
                    f"journey {f.get('duration') or '?'} stops={f['stops']} "
                    + (f"₹{total:,.0f} TOTAL for all {persons} traveller(s)"
                       if total else "price unavailable")
                )
        elif mode == "train":
            fares = data.get("fares") or {}
            fare_text = (
                " | ".join(
                    f"{c} ₹{f['total_inr']:,.0f} TOTAL for all {persons}"
                    for c, f in fares.items()
                )
                if fares else "fare estimate unavailable"
            )
            for t in (data.get("trains") or [])[:5]:
                minutes = parse_duration_minutes(t.get("duration"))
                arr_date, arr_time = derive_arrival(
                    start_date, departs=t.get("departs"), arrives=t.get("arrives"),
                    duration_minutes=minutes,
                )
                facts.append(
                    f"TRAIN {t['train_no']} {t['train_name']} dep {t['departs']} "
                    f"({t['from']}) arr {t['arrives']} ({t['to']}) "
                    f"journey {pretty_duration(minutes) or t.get('duration') or 'unknown'}"
                    + (f" arriving {arr_date}" if arr_date and arr_date != start_date
                       else "")
                    + f" | estimated fares: {fare_text}"
                )
        elif mode == "bus":
            for b in (data.get("buses") or [])[:5]:
                arr_date, _ = derive_arrival(
                    start_date, departs=b.get("departure"), arrives=b.get("arrival")
                )
                fare = b.get("fare_inr")
                facts.append(
                    f"BUS {b['operator']} dep {b.get('departure')} arr {b.get('arrival')}"
                    + (f" (next day {arr_date})" if arr_date and arr_date != start_date
                       else "")
                    + (f" ₹{fare:,.0f} PER PERSON" if isinstance(fare, (int, float))
                       else " fare unavailable")
                    + f" {b.get('bus_type') or ''}"
                )
        elif mode == "cab":
            for c in (data.get("options") or [])[:3]:
                facts.append(
                    f"CAB {c['vehicle']} ₹{c['estimated_fare_inr']:,.0f} TOTAL for the "
                    f"group (~{data.get('distance_km')} km, estimate)"
                )

    # Only legs whose main mode actually produced something. A CONNECTING_LEG
    # line for a mode with no services reads to the model as evidence that the
    # journey exists: Gangtok→Kochi found zero IXB→COK flights, and the leg fact
    # alone was enough for it to recommend "fly Bagdogra to Kochi".
    served = {
        mode for mode, data in options.items()
        if (data or {}).get({"flight": "flights", "train": "trains",
                             "bus": "buses", "cab": "options"}[mode])
    }
    facts.extend(
        journey_provider.facts([leg for leg in legs if leg["for_mode"] in served])
    )
    facts.extend(_connection_facts(options, legs, served, start_date))

    # Rank on the arithmetic before the model sees anything. Its own ordering is
    # then checked against this rather than taken on trust.
    ranked = decision.rank(
        build_candidates(options, legs, persons, start_date),
        brief.get("budget_level", "medium"),
    )
    if ranked and ranked[0].score is not None:
        facts.append(
            "SCORED_BEST (computed, cost+time+convenience for this budget "
            f"level): {ranked[0].label} at ₹{ranked[0].total_inr:,.0f} total"
        )

    # State the absences explicitly rather than leaving them to be inferred.
    for mode in sorted(set(options) - served):
        note = (options.get(mode) or {}).get("note") or "no services found"
        facts.append(f"NO_OPTIONS {mode.upper()}: {note}")

    # "We did not look" and "there is nothing" are different claims, and only
    # one of them is ours to make.
    skipped = (state.get("transport_strategy") or {}).get("skipped") or {}
    for mode, why in sorted(skipped.items()):
        facts.append(f"NOT_SEARCHED {mode.upper()}: {why}")

    unpriced = [leg for leg in legs if leg["for_mode"] not in served]
    for leg in unpriced:
        facts.append(
            f"NOTE: {leg['from']} → {leg['to']} would only be needed for "
            f"{leg['for_mode']}, and no {leg['for_mode']} services were found."
        )

    # Checked against ``served``, not ``facts``: the NO_OPTIONS and NOTE lines
    # above are facts too, and none of them is something to recommend.
    if not served:
        reasons = [
            f"- {mode}: {(data or {}).get('note') or 'no services found'}"
            for mode, data in sorted(options.items())
        ]
        reasons += [
            f"- {mode}: not checked — {why}"
            for mode, why in sorted(
                ((state.get("transport_strategy") or {}).get("skipped") or {}).items()
            )
        ]
        summary = (
            f"No bookable transport was found from {brief.get('origin')} to "
            f"{brief.get('destination')} on {start_date}.\n\n"
            + "\n".join(reasons)
            + "\n\nTry a different date, or a nearby larger city as your "
            "starting point."
        )
        return {
            "transport_summary": summary,
            "messages": [AIMessage(content=f"⚠️ {summary}")],
        }

    gateway_note = _gateway_note(brief)

    prompt = f"""Compare these real transport options for {persons} traveller(s)
from {brief.get('origin')} to {brief.get('destination')}, departing {start_date}.
Budget level: {brief.get('budget_level')}.
{f'The user asked for: "{feedback}" — weight this heavily.' if feedback else ''}
{gateway_note}
OPTIONS FOUND:
{chr(10).join(facts)}

HOW TO READ THE DATA:
- A price marked TOTAL already covers all {persons} traveller(s). Do NOT multiply
  it again. A price marked PER PERSON must be multiplied by {persons}.
- "journey" is the real door-to-door time for that leg. Use it — do not infer
  duration by subtracting the arrival clock time from the departure clock time,
  because journeys longer than a day wrap around.
- A CONNECTING_LEG is not optional. It is a leg the traveller must physically
  cover, and it belongs in both the plan and the total.
- A NO_OPTIONS line means that mode has nothing bookable. Never build a
  recommendation on it, and never infer that a service exists because a
  connecting leg to its gateway was priced.
- A NOT_SEARCHED line means that mode was never queried. Say "not checked
  (<reason>)". You must NOT say it is unavailable, that none exist, or that
  none were found — nothing looked.
- An UNUSABLE_GATEWAY line means that mode's departure point cannot realistically
  be reached from the traveller's city. Do not recommend that mode; if you
  mention it, say plainly that the connection is unconfirmed.
- CONNECTION IMPOSSIBLE means the traveller cannot reach that departure in time
  on the day. Never recommend such an option; if the only options are
  impossible, say so and suggest travelling to the gateway the evening before.
- CONNECTION OK gives the latest time they can set off. Put that "leave
  <city> by <time>" in your recommendation — it is the single most useful
  sentence in the whole plan.

Write:
1. **Recommendation** — one option, with why (time, cost, convenience). If it
   requires a connecting leg, present the journey as numbered legs
   ("Leg 1: {brief.get('origin')} → gateway by …, Leg 2: flight …") and state
   the connecting time needed between them.
2. **Runner-up** — the best fallback.
3. **Comparison table** — mode, departure, arrival (with date if it is not
   {start_date}), total journey time, and total cost for all {persons}
   traveller(s) including any connecting leg.
4. **Total transport cost** for the group, as a number, connecting legs included.

Then, on the last three lines, exactly:
ARRIVAL_DATE: YYYY-MM-DD
ARRIVAL_TIME: HH:MM
TRANSPORT_COST_INR: <number>

ARRIVAL_DATE and ARRIVAL_TIME are when the travellers actually reach
{brief.get('destination')} itself — after any connecting leg, not when the
flight or train lands at the gateway.

Only use options listed above. Never invent a service. If costs are estimates,
say so."""

    degraded = False
    try:
        summary = await llm_provider.ask(prompt, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        # The LLM writes the prose; it does not hold the data. Losing it used to
        # lose the entire recommendation — no arrival, no budget, no options —
        # even though every fare and timing was already in hand. Hit for real
        # when the Groq free tier's daily token cap ran out mid-run.
        logger.warning("Recommendation LLM unavailable (%s); composing from data.", exc)
        await _emit(state, EventType.NODE_FAILED, str(exc), "recommend_transport")
        summary = _compose_summary(brief, options, legs, persons, start_date)
        degraded = True

    match = _ARRIVAL_RE.search(summary)
    arrival = f"{match.group(1).zfill(2)}:{match.group(2)}" if match else _fallback_arrival(options)

    date_match = _ARRIVAL_DATE_RE.search(summary)
    arrival_date = date_match.group(1) if date_match else None
    if not arrival_date:
        arrival_date = _fallback_arrival_date(options, start_date)
    # Never let a model hallucinate an arrival before departure.
    if arrival_date and start_date and arrival_date < start_date:
        logger.warning("Ignoring arrival_date %s before departure %s",
                       arrival_date, start_date)
        arrival_date = start_date

    cost = None
    cost_match = re.search(
        r"TRANSPORT_COST_INR\s*[:=]?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)",
        summary, re.IGNORECASE,
    )
    if cost_match:
        try:
            cost = float(cost_match.group(1).replace(",", ""))
        except ValueError:
            cost = None

    if cost is None:
        # The model routinely writes "not available" here when the cheapest
        # option has no hard price, and the trip total then silently lost its
        # largest line. Fall back to the cheapest concrete figure we hold.
        cost = _cheapest_transport_inr(options, legs, persons)
        if cost:
            logger.info("Model gave no transport cost; using computed ₹%.0f", cost)

    return_flights = (state.get("return_options") or {}).get("flight") or {}
    return_cost = return_flights.get("cheapest_total_inr")

    await _emit(state, EventType.NODE_COMPLETED, "Transport recommendation ready",
                "recommend_transport", arrival_time=arrival, arrival_date=arrival_date)

    # The audit trail. Everything a reviewer needs to check the recommendation
    # without re-running the search: what was considered, what was skipped and
    # why, how each option scored, and whether the prose matches the maths.
    rationale = {
        "budget_level": brief.get("budget_level", "medium"),
        "weights": decision._WEIGHTS.get(
            brief.get("budget_level", "medium"), decision._WEIGHTS["medium"]
        ),
        "considered": [c.to_dict() for c in ranked],
        "skipped_modes": (state.get("transport_strategy") or {}).get("skipped") or {},
        "modes_with_no_service": sorted(set(options) - served),
        "connecting_legs": [
            {"from": leg["from"], "to": leg["to"], "for_mode": leg["for_mode"],
             "cost_inr": leg.get("cheapest_inr"),
             "travel_minutes": leg.get("travel_minutes"),
             "implausible": bool(leg.get("implausible"))}
            for leg in legs
        ],
        "model_agreement": decision.compare_with_model(ranked, summary),
        "summary_source": "composed_from_data" if degraded else "llm",
    }

    update: dict[str, Any] = {
        "transport_summary": summary,
        "arrival_time": arrival or "14:00",
        "arrival_date": arrival_date or start_date or None,
        "budget": {"transport_inr": cost, "return_transport_inr": return_cost},
        "decision_rationale": rationale,
        "messages": [AIMessage(content=summary)],
    }
    if degraded:
        # Recorded, not hidden: the plan is usable but it was not reasoned over.
        update["provider_health"] = {"recommendation": "degraded: LLM unavailable"}
        update["errors"] = [{
            "node": "recommend_transport",
            "error": "LLM unavailable; recommendation composed from search data",
        }]
    return update


# ---------------------------------------------------------------------------
# 5. Human review — a real LangGraph interrupt
# ---------------------------------------------------------------------------

async def human_review_transport(state: TripState) -> dict[str, Any]:
    """Pause the graph and wait for a human decision.

    The original ``human_node`` rendered Streamlit widgets *inside* a graph node
    during ``graph.invoke()``. ``st.button`` returns ``False`` on first render,
    so ``human_review_result`` was never set, the conditional edge always took
    its "approved" default, and the human review had literally no effect on any
    run. Streamlit also reruns the whole script on interaction, by which point
    the graph invocation is long finished.

    ``interrupt()`` is the mechanism that actually does this: it checkpoints the
    state, returns control to the caller, and resumes from exactly this point
    when the client sends ``Command(resume=...)`` — even in a different process,
    minutes later.
    """
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        logger.info("Revision limit reached; auto-approving.")
        return {
            "review_decision": "approve",
            "review_feedback": None,
            "messages": [AIMessage(content="↪️ Revision limit reached — proceeding.")],
        }

    await _emit(state, EventType.AWAITING_REVIEW, "Waiting for your review of transport",
                "human_review_transport")

    decision = interrupt(
        {
            "stage": "transport_review",
            "question": "Do these transport options work for you?",
            "options": ["approve", "revise_transport"],
            "transport_summary": state.get("transport_summary"),
            "provider_health": state.get("provider_health"),
            "revision_count": state.get("revision_count", 0),
            "revisions_remaining": MAX_REVISIONS - state.get("revision_count", 0),
        }
    )

    action, feedback = _parse_resume(decision, default="approve")
    is_revision = action != "approve"

    await _emit(state, EventType.TRIP_RESUMED, f"Review: {action}", "human_review_transport")
    return {
        "review_decision": action,
        "review_feedback": feedback,
        "revision_count": 1 if is_revision else 0,
        "messages": [
            HumanMessage(content=feedback) if feedback
            else AIMessage(content="✅ Transport approved.")
        ],
    }


def _parse_resume(value: Any, default: str) -> tuple[str, Optional[str]]:
    """Accept a dict, a bare string, or ``None`` from the resume payload."""
    if value is None:
        return default, None
    if isinstance(value, str):
        return (value.strip().lower() or default), None
    if isinstance(value, dict):
        action = str(value.get("action") or value.get("decision") or default).strip().lower()
        feedback = value.get("feedback") or value.get("comment")
        return action, (str(feedback).strip() or None) if feedback else None
    return default, None


# ---------------------------------------------------------------------------
# 6. Enrich: weather + hotels, concurrently
# ---------------------------------------------------------------------------

async def enrich(state: TripState) -> dict[str, Any]:
    """Fetch weather and hotels at the same time — they are independent."""
    await _emit(state, EventType.NODE_STARTED, "Fetching weather and hotels…", "enrich")
    brief = state.get("brief") or {}
    feedback = (state.get("review_feedback") or "").strip()
    force_refresh = state.get("review_decision") == "revise_hotels"

    # Check in when they actually arrive, not when they set off. On an overnight
    # journey these differ by a day, and booking from the departure date buys a
    # night nobody sleeps in.
    checkin = state.get("arrival_date") or brief.get("start_date", "")
    checkout = brief.get("end_date")
    if checkin and checkout and checkin > checkout:
        checkin = brief.get("start_date", "")
    # Google Hotels needs checkout > checkin; a same-day trip still needs 1 night.
    if checkout == checkin:
        checkout = (
            datetime.strptime(checkin, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    nights_lost = 0
    if checkin and checkin != brief.get("start_date"):
        nights_lost = (
            datetime.strptime(checkin, "%Y-%m-%d")
            - datetime.strptime(brief["start_date"], "%Y-%m-%d")
        ).days

    results = await gather_providers(
        {
            "weather": weather_provider.get_forecast(
                brief.get("destination", ""), checkin,
                int(brief.get("duration", 3) or 3),
            ),
            "hotels": hotels_provider.search_hotels(
                brief.get("destination", ""), checkin, checkout,
                adults=int(brief.get("persons", 1) or 1),
                preference=feedback if force_refresh else "",
                skip_cache=force_refresh,
            ),
        }
    )

    update: dict[str, Any] = {"provider_health": {}, "errors": [], "messages": []}
    lines: list[str] = []
    if nights_lost > 0:
        lines.append(
            f"🛏️ Check-in set to **{checkin}**, not {brief['start_date']} — the "
            f"journey takes {nights_lost} night(s), so those are spent travelling."
        )

    if results["weather"]["ok"]:
        update["weather"] = results["weather"]["data"]
        update["provider_health"]["weather"] = "ok"
        lines.append("🌤️ **Weather forecast**\n" + weather_provider.summarise(results["weather"]["data"]))
    else:
        update["weather"] = []
        update["provider_health"]["weather"] = f"failed: {results['weather']['error']}"
        update["errors"].append({"node": "enrich", "provider": "weather",
                                 "error": results["weather"]["error"]})
        lines.append(f"⚠️ Weather unavailable: {results['weather']['error']}")

    if results["hotels"]["ok"]:
        data = results["hotels"]["data"]
        rendered = hotels_provider.render(data)
        update["hotels"] = data
        update["hotels_summary"] = rendered
        update["provider_health"]["hotels"] = "ok"
        update["budget"] = {"hotel_inr": data.get("cheapest_total_inr")}
        lines.append(rendered)
    else:
        update["provider_health"]["hotels"] = f"failed: {results['hotels']['error']}"
        update["errors"].append({"node": "enrich", "provider": "hotels",
                                 "error": results["hotels"]["error"]})
        update["hotels_summary"] = "Hotel search unavailable."
        lines.append(f"⚠️ Hotels unavailable: {results['hotels']['error']}")

    update["messages"] = [AIMessage(content="\n\n".join(lines))]
    await _emit(state, EventType.NODE_COMPLETED, "Weather and hotels ready", "enrich")
    return update


# ---------------------------------------------------------------------------
# 7. Itinerary
# ---------------------------------------------------------------------------

def _skeleton_itinerary(
    brief: dict[str, Any],
    duration: int,
    arrival_date: str,
    arrival_time: Optional[str],
    weather: list[dict[str, Any]],
    hotel_summary: str,
) -> str:
    """A dated day-by-day outline with no LLM involved.

    Deliberately does not invent attractions — that is exactly what the model is
    for. It gives the traveller the real scaffolding: correct dates, the day
    their journey actually ends, and the forecast against each one.
    """
    destination = brief.get("destination", "your destination")
    by_date = {str(day.get("date")): day for day in weather if day.get("date")}

    lines = [
        f"## {duration}-day outline for {destination}",
        "",
        "_The planning model was unavailable, so this is a dated skeleton "
        "rather than a written itinerary. Dates, arrival and forecast are real; "
        "the activities are for you to fill in._",
        "",
    ]
    try:
        start = datetime.strptime(arrival_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return "\n".join(lines + ["_Arrival date unavailable._"])

    for day in range(duration):
        current = start + timedelta(days=day)
        key = current.strftime("%Y-%m-%d")
        forecast = by_date.get(key)
        heading = f"### Day {day + 1} — {current.strftime('%A, %Y-%m-%d')}"
        # Only when there is a real reading. Days past the forecast horizon
        # carry a placeholder row, which rendered as "?°/?° · rain ?%".
        if forecast and forecast.get("max_temp_c") is not None:
            heading += (
                f" · {forecast.get('condition', '')} "
                f"{forecast['max_temp_c']}°/{forecast.get('min_temp_c', '?')}°"
                f" · rain {forecast.get('chance_of_rain', '?')}%"
            )
        elif forecast and forecast.get("condition"):
            heading += f" · {forecast['condition']}"
        lines.append(heading)
        if day == 0:
            lines.append(
                f"- Arrive in {destination}"
                + (f" around {arrival_time}" if arrival_time else "")
                + ". Check in, stay close to the hotel, keep the day light."
            )
        elif day == duration - 1:
            lines.append("- Check out and travel onward. Leave buffer for the journey.")
        else:
            lines.append("- Full day free.")
        lines.append("")

    if hotel_summary:
        lines += ["### Accommodation", hotel_summary]
    return "\n".join(lines)


async def build_itinerary(state: TripState) -> dict[str, Any]:
    await _emit(state, EventType.NODE_STARTED, "Writing your day-by-day plan…", "build_itinerary")
    brief = state.get("brief") or {}

    # Days actually spent at the destination. A 4-day trip whose outbound
    # journey eats a full day is a 3-day itinerary, and planning Day 1 sightseeing
    # while the traveller is still on the train is the single most visible
    # symptom of the old fixed arrival_date.
    arrival_date = state.get("arrival_date") or brief.get("start_date", "")
    duration = int(brief.get("duration", 3) or 3)
    try:
        on_ground = (
            datetime.strptime(brief["end_date"], "%Y-%m-%d")
            - datetime.strptime(arrival_date, "%Y-%m-%d")
        ).days + 1
        duration = max(1, min(duration, on_ground))
    except (KeyError, ValueError, TypeError):
        pass

    try:
        result = await itinerary_provider.generate_itinerary(
            destination=brief.get("destination", ""),
            duration=duration,
            interests=brief.get("preferences") or [],
            weather=state.get("weather") or [],
            arrival_date=arrival_date,
            arrival_time=state.get("arrival_time"),
            hotel_summary=state.get("hotels_summary") or "",
            transport_summary=(state.get("transport_summary") or "")[:1500],
            persons=int(brief.get("persons", 1) or 1),
            budget_level=brief.get("budget_level", "medium"),
            special_requirements=brief.get("special_requirements", ""),
            feedback=(state.get("review_feedback") or "")
            if state.get("review_decision") == "revise_plan" else "",
        )
    except Exception as exc:  # noqa: BLE001
        # A failed plan used to return errors and nothing else, and the trip
        # still reported "completed" — with an empty itinerary. Hit for real
        # when the Groq daily cap ran out at this node. A skeleton built from
        # the weather, the arrival time and the hotel is worth far more than a
        # blank page, and it keeps the trip honestly usable.
        logger.warning("Itinerary LLM unavailable (%s); building a skeleton.", exc)
        await _emit(state, EventType.NODE_FAILED, str(exc), "build_itinerary")
        return {
            "itinerary": _skeleton_itinerary(
                brief, duration, arrival_date, state.get("arrival_time"),
                state.get("weather") or [], state.get("hotels_summary") or "",
            ),
            "itinerary_meta": {"grounded": False, "degraded": True},
            "provider_health": {"itinerary": "degraded: LLM unavailable"},
            "errors": [{
                "node": "build_itinerary",
                "error": f"LLM unavailable ({exc}); outline generated instead",
            }],
            "messages": [AIMessage(content="⚠️ Day-by-day plan is an outline only.")],
        }

    await _emit(state, EventType.NODE_COMPLETED, "Itinerary ready", "build_itinerary")
    return {
        "itinerary": result["plan"],
        "itinerary_meta": {
            "grounded": result.get("grounded"),
            "source_url": result.get("source_url"),
        },
        "budget": {"onground_inr": result.get("estimated_onground_budget_inr")},
        "messages": [AIMessage(content=result["plan"])],
    }


# ---------------------------------------------------------------------------
# 8. Final review + totals
# ---------------------------------------------------------------------------

async def human_review_plan(state: TripState) -> dict[str, Any]:
    """Second interrupt: accept the plan or ask for a targeted revision.

    This replaces ``llm_feedback_router``, which called ``input()`` — a blocking
    stdin read — from inside a graph node. In Streamlit that hangs the server
    thread; in a worker it raises ``EOFError``.
    """
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return {"review_decision": "approve", "review_feedback": None}

    await _emit(state, EventType.AWAITING_REVIEW, "Waiting for your review of the plan",
                "human_review_plan")

    decision = interrupt(
        {
            "stage": "plan_review",
            "question": "Happy with this itinerary?",
            "options": ["approve", "revise_plan", "revise_hotels", "revise_transport"],
            "itinerary": state.get("itinerary"),
            "budget": state.get("budget"),
            "revisions_remaining": MAX_REVISIONS - state.get("revision_count", 0),
        }
    )

    action, feedback = _parse_resume(decision, default="approve")
    await _emit(state, EventType.TRIP_RESUMED, f"Review: {action}", "human_review_plan")
    return {
        "review_decision": action,
        "review_feedback": feedback,
        "revision_count": 0 if action == "approve" else 1,
        "messages": [HumanMessage(content=feedback)] if feedback else [],
    }


async def finalise(state: TripState) -> dict[str, Any]:
    """Compute the real total instead of the hardcoded table the UI showed."""
    budget = state.get("budget") or {}
    brief = state.get("brief") or {}
    persons = max(1, int(brief.get("persons", 1) or 1))

    transport = budget.get("transport_inr") or 0
    return_transport = budget.get("return_transport_inr") or 0
    hotel = budget.get("hotel_inr") or 0
    onground = budget.get("onground_inr") or 0
    total = transport + return_transport + hotel + onground

    totals = {
        "transport_inr": transport or None,
        "return_transport_inr": return_transport or None,
        "hotel_inr": hotel or None,
        "onground_inr": onground or None,
        "total_inr": total or None,
        "per_person_inr": round(total / persons, 2) if total else None,
        "currency": "INR",
        "is_estimate": True,
        "components_missing": [
            k for k, v in (
                ("transport", transport), ("return transport", return_transport),
                ("hotel", hotel), ("onground", onground)
            ) if not v
        ],
    }

    await _emit(state, EventType.TRIP_COMPLETED, "Plan complete", "finalise", budget=totals)
    return {
        "budget": totals,
        "messages": [
            AIMessage(
                content=(
                    f"✅ Plan complete. Estimated total ₹{total:,.0f} "
                    f"(₹{totals['per_person_inr']:,.0f} per person)."
                    if total else "✅ Plan complete (cost estimate unavailable)."
                )
            )
        ],
    }
