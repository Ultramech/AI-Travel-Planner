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

from app.core.bus import emit
from app.core.events import EventType
from app.core.resilience import gather_providers
from app.graph.state import TripState
from app.providers import buses as buses_provider
from app.providers import cabs as cabs_provider
from app.providers import extractor
from app.providers import flights_router as flights_provider
from app.providers import hotels as hotels_provider
from app.providers import itinerary as itinerary_provider
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

async def plan_transport_strategy(state: TripState) -> dict[str, Any]:
    """Pick which modes to search — then we search them all in parallel.

    The original asked the LLM to commit to exactly ONE mode *before seeing any
    data*, then searched only that. If it guessed "bus_or_cab_only" for
    Delhi→Chennai the user simply never saw the flights. Here the LLM's job is
    narrowed to a cheap feasibility filter, and the actual recommendation is
    made later from real prices and real timings.
    """
    await _emit(state, EventType.NODE_STARTED, "Deciding which transport to search…",
                "plan_transport_strategy")
    brief = state.get("brief") or {}
    feedback = (state.get("review_feedback") or "").strip()

    modes = {"flight", "train", "bus", "cab"}
    if not (brief.get("origin_iata") and brief.get("destination_iata")):
        modes.discard("flight")
    elif brief["origin_iata"] == brief["destination_iata"]:
        modes.discard("flight")

    try:
        prompt = f"""Trip: {brief.get('origin')} to {brief.get('destination')}, India.
Travellers: {brief.get('persons')}. Budget level: {brief.get('budget_level')}.
{f'The user asked for: "{feedback}"' if feedback else ''}

Which of these modes are sensible to search? Consider distance and cost:
flight, train, bus, cab

Rules:
- Drop "flight" for journeys under ~300 km.
- Drop "cab" for journeys over ~700 km unless the user explicitly asked for one.
- Always keep at least one mode.
- If the user expressed a preference, keep that mode.

Reply with JSON only: {{"modes": ["..."], "reason": "<one sentence>"}}"""

        data = await llm_provider.ask_json(prompt)
        suggested = {str(m).strip().lower() for m in (data.get("modes") or [])}
        chosen = (suggested & modes) or modes
        reason = str(data.get("reason", ""))[:300]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Strategy LLM failed (%s); searching all feasible modes.", exc)
        chosen, reason = modes, "Fallback: searching all feasible modes."

    strategy = {"modes": sorted(chosen), "reason": reason}
    await _emit(state, EventType.NODE_COMPLETED, f"Searching: {', '.join(sorted(chosen))}",
                "plan_transport_strategy", **strategy)
    return {
        "transport_strategy": strategy,
        "messages": [AIMessage(content=f"🧭 Searching {', '.join(sorted(chosen))}. {reason}")],
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

    tasks: dict[str, Any] = {}
    if "flight" in modes and brief.get("origin_iata") and brief.get("destination_iata"):
        tasks["flight"] = flights_provider.search_flights(
            brief["origin_iata"], brief["destination_iata"], start_date,
            return_date=end_date, adults=persons, skip_cache=force_refresh,
        )
    if "train" in modes:
        tasks["train"] = trains_provider.search_trains(
            origin, destination, start_date, skip_cache=force_refresh
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

    if not tasks:
        return {
            "errors": [{"node": "search_transport", "error": "no searchable transport modes"}],
            "messages": [AIMessage(content="⚠️ No transport modes could be searched.")],
        }

    results = await gather_providers(tasks, overall_timeout=120.0)

    options: dict[str, Any] = {}
    health: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    lines: list[str] = []

    renderers = {
        "flight": flights_provider.render,
        "train": trains_provider.render,
        "bus": buses_provider.render,
        "cab": cabs_provider.render,
    }

    for mode, outcome in results.items():
        if outcome["ok"]:
            options[mode] = outcome["data"]
            health[mode] = "ok"
            lines.append(renderers[mode](outcome["data"]))
            await _emit(state, EventType.PROVIDER_CALLED, f"{mode}: found options",
                        "search_transport", mode=mode, ok=True)
        else:
            health[mode] = f"failed: {outcome['error']}"
            errors.append({"node": "search_transport", "provider": mode,
                           "error": outcome["error"]})
            lines.append(f"⚠️ {mode.title()} search unavailable: {outcome['error']}")
            await _emit(state, EventType.PROVIDER_CALLED, f"{mode}: {outcome['error']}",
                        "search_transport", mode=mode, ok=False)

    await _emit(state, EventType.NODE_COMPLETED,
                f"{sum(1 for v in health.values() if v == 'ok')}/{len(health)} providers responded",
                "search_transport")

    return {
        "transport_options": options,
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

    # Feed the LLM compact structured facts, not a wall of prior chat messages.
    facts: list[str] = []
    for mode, data in options.items():
        if mode == "flight":
            for f in (data.get("flights") or [])[:4]:
                facts.append(
                    f"FLIGHT {f['route']} dep {f['departure']} arr {f['arrival']} "
                    f"₹{f['price_inr']:,.0f} stops={f['stops']} {f['duration']}"
                )
        elif mode == "train":
            for t in (data.get("trains") or [])[:5]:
                facts.append(
                    f"TRAIN {t['train_no']} {t['train_name']} dep {t['departs']} "
                    f"arr {t['arrives']} ({t['from']}→{t['to']})"
                )
        elif mode == "bus":
            for b in (data.get("buses") or [])[:5]:
                facts.append(
                    f"BUS {b['operator']} dep {b.get('departure')} arr {b.get('arrival')} "
                    f"₹{b.get('fare_inr')} {b.get('bus_type') or ''}"
                )
        elif mode == "cab":
            for c in (data.get("options") or [])[:3]:
                facts.append(
                    f"CAB {c['vehicle']} ~₹{c['estimated_fare_inr']:,.0f} total "
                    f"(~{data.get('distance_km')} km, estimate)"
                )

    if not facts:
        summary = "No concrete transport options were returned by any provider."
        return {
            "transport_summary": summary,
            "messages": [AIMessage(content=f"⚠️ {summary}")],
        }

    prompt = f"""Compare these real transport options for {brief.get('persons')} traveller(s)
from {brief.get('origin')} to {brief.get('destination')} on {brief.get('start_date')}.
Budget level: {brief.get('budget_level')}.
{f'The user asked for: "{feedback}" — weight this heavily.' if feedback else ''}

OPTIONS FOUND:
{chr(10).join(facts)}

Write:
1. **Recommendation** — one option, with why (time, cost, convenience).
2. **Runner-up** — the best fallback.
3. **Comparison table** — mode, departure, arrival, duration, total cost for all
   {brief.get('persons')} traveller(s).
4. **Total transport cost** for the group, as a number.

Then, on the last two lines, exactly:
ARRIVAL_TIME: HH:MM
TRANSPORT_COST_INR: <number>

Only use options listed above. Never invent a service. If costs are estimates,
say so."""

    try:
        summary = await llm_provider.ask(prompt, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        await _emit(state, EventType.NODE_FAILED, str(exc), "recommend_transport")
        return _error(state, "recommend_transport", exc)

    match = _ARRIVAL_RE.search(summary)
    arrival = f"{match.group(1).zfill(2)}:{match.group(2)}" if match else _fallback_arrival(options)

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

    await _emit(state, EventType.NODE_COMPLETED, "Transport recommendation ready",
                "recommend_transport", arrival_time=arrival)

    return {
        "transport_summary": summary,
        "arrival_time": arrival or "14:00",
        "budget": {"transport_inr": cost},
        "messages": [AIMessage(content=summary)],
    }


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

    checkout = brief.get("end_date")
    # Google Hotels needs checkout > checkin; a same-day trip still needs 1 night.
    if checkout == brief.get("start_date"):
        checkout = (
            datetime.strptime(brief["start_date"], "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    results = await gather_providers(
        {
            "weather": weather_provider.get_forecast(
                brief.get("destination", ""), brief.get("start_date", ""),
                int(brief.get("duration", 3) or 3),
            ),
            "hotels": hotels_provider.search_hotels(
                brief.get("destination", ""), brief.get("start_date", ""), checkout,
                adults=int(brief.get("persons", 1) or 1),
                preference=feedback if force_refresh else "",
                skip_cache=force_refresh,
            ),
        }
    )

    update: dict[str, Any] = {"provider_health": {}, "errors": [], "messages": []}
    lines: list[str] = []

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

async def build_itinerary(state: TripState) -> dict[str, Any]:
    await _emit(state, EventType.NODE_STARTED, "Writing your day-by-day plan…", "build_itinerary")
    brief = state.get("brief") or {}
    try:
        result = await itinerary_provider.generate_itinerary(
            destination=brief.get("destination", ""),
            duration=int(brief.get("duration", 3) or 3),
            interests=brief.get("preferences") or [],
            weather=state.get("weather") or [],
            arrival_date=state.get("arrival_date") or brief.get("start_date", ""),
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
        await _emit(state, EventType.NODE_FAILED, str(exc), "build_itinerary")
        return _error(state, "build_itinerary", exc)

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
    hotel = budget.get("hotel_inr") or 0
    onground = budget.get("onground_inr") or 0
    total = transport + hotel + onground

    totals = {
        "transport_inr": transport or None,
        "hotel_inr": hotel or None,
        "onground_inr": onground or None,
        "total_inr": total or None,
        "per_person_inr": round(total / persons, 2) if total else None,
        "currency": "INR",
        "is_estimate": True,
        "components_missing": [
            k for k, v in (
                ("transport", transport), ("hotel", hotel), ("onground", onground)
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
