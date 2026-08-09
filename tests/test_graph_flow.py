"""End-to-end graph tests with every external provider stubbed.

These lock in the behaviours that were broken before:
  * human-in-the-loop actually pauses and actually resumes,
  * revision loops terminate,
  * messages do not duplicate,
  * routers never raise on unexpected LLM output.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.graph import nodes
from app.graph.builder import (
    build_graph_definition,
    route_after_brief,
    route_after_plan_review,
    route_after_transport_review,
)

START = (date.today() + timedelta(days=20)).isoformat()
END = (date.today() + timedelta(days=22)).isoformat()


@pytest.fixture
def compiled_graph():
    from langgraph.checkpoint.memory import MemorySaver

    return build_graph_definition().compile(checkpointer=MemorySaver())


@pytest.fixture(autouse=True)
def stub_providers(monkeypatch):
    """Replace every network-touching provider with a deterministic stub."""
    from app.providers import extractor

    async def fake_build_brief(**kwargs):
        return extractor.TripBrief(
            origin=kwargs.get("origin") or "Delhi",
            destination=kwargs.get("destination") or "Goa",
            start_date=kwargs.get("start_date") or START,
            end_date=kwargs.get("end_date") or END,
            duration=3,
            persons=kwargs.get("persons") or 2,
            preferences=kwargs.get("preferences") or ["beaches"],
            origin_iata="DEL",
            destination_iata="GOI",
            origin_airport_city="Delhi",
            destination_airport_city="Goa",
        )

    async def fake_ask_json(prompt, **kwargs):
        return {"modes": ["flight", "train"], "reason": "long distance"}

    async def fake_ask(prompt, **kwargs):
        if "Compare these real transport options" in prompt:
            return "Recommendation: fly.\nARRIVAL_TIME: 11:45\nTRANSPORT_COST_INR: 18000"
        return "### Day 1\nBeach.\n\nESTIMATED_ONGROUND_BUDGET_INR: 12000"

    async def fake_flights(*args, **kwargs):
        return {
            "origin_iata": "DEL", "destination_iata": "GOI",
            "departure_date": START, "count": 1,
            "flights": [{
                "route": "DEL → GOI", "departure": f"{START}T09:00:00",
                "arrival": f"{START}T11:15:00", "duration": "2h15m",
                "price_inr": 9000.0, "stops": 0, "layover_pretty": "Direct",
                "layover_airports": [], "booking_link": "https://example.test",
            }],
            "cheapest_inr": 9000.0,
        }

    async def fake_trains(*args, **kwargs):
        return {"origin": "Delhi", "destination": "Goa", "date": START,
                "count": 0, "trains": [], "note": "No direct trains found."}

    async def fake_weather(*args, **kwargs):
        return [{"date": START, "day_name": "Monday", "available": True,
                 "condition": "Sunny", "max_temp": 32, "min_temp": 24,
                 "avg_temp": 28, "rain_chance": 5, "humidity": 60}]

    async def fake_hotels(*args, **kwargs):
        return {"city": "Goa", "checkin": START, "checkout": END, "nights": 2,
                "adults": 2, "count": 1, "cheapest_total_inr": 8000.0,
                "hotels": [{"name": "Test Resort", "rating": 4.4,
                            "nightly_price_inr": 4000.0, "total_price_inr": 8000.0,
                            "nights": 2, "link": "https://example.test"}]}

    async def fake_itinerary(**kwargs):
        return {"destination": "Goa", "duration": 3, "plan": "### Day 1\nBeach day.",
                "grounded": True, "source_url": "https://en.wikivoyage.org/wiki/Goa",
                "estimated_onground_budget_inr": 12000.0}

    monkeypatch.setattr(nodes.extractor, "build_brief", fake_build_brief)
    monkeypatch.setattr(nodes.llm_provider, "ask_json", fake_ask_json)
    monkeypatch.setattr(nodes.llm_provider, "ask", fake_ask)
    monkeypatch.setattr(nodes.flights_provider, "search_flights", fake_flights)
    monkeypatch.setattr(nodes.trains_provider, "search_trains", fake_trains)
    monkeypatch.setattr(nodes.weather_provider, "get_forecast", fake_weather)
    monkeypatch.setattr(nodes.hotels_provider, "search_hotels", fake_hotels)
    monkeypatch.setattr(nodes.itinerary_provider, "generate_itinerary", fake_itinerary)


def _config(trip_id="trip_test"):
    return {"configurable": {"thread_id": trip_id}, "recursion_limit": 40}


async def pending(graph, config, result=None):
    """Version-agnostic 'is the graph waiting on a human?' check."""
    from app.services.planner import pending_interrupt

    return await pending_interrupt(graph, config, result)


# ---------------------------------------------------------------------------
# Human in the loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_pauses_at_transport_review(compiled_graph):
    """The old human_node never paused anything — st.button always returned
    False and the graph sailed past it."""
    from app.graph.state import new_state

    config = _config()
    result = await compiled_graph.ainvoke(
        new_state("trip_test", "Delhi to Goa"), config=config
    )

    payload = await pending(compiled_graph, config, result)
    assert payload is not None, "graph should pause for human review"
    assert payload["stage"] == "transport_review"
    assert "approve" in payload["options"]
    assert result["transport_summary"]
    assert result["arrival_time"] == "11:45"
    # Nothing downstream of the interrupt should have run yet.
    assert not result.get("itinerary")


@pytest.mark.asyncio
async def test_resume_completes_the_plan(compiled_graph):
    from langgraph.types import Command
    from app.graph.state import new_state

    config = _config("trip_resume")
    await compiled_graph.ainvoke(new_state("trip_resume", "Delhi to Goa"), config=config)

    # Approve transport -> pauses again at the plan review.
    second = await compiled_graph.ainvoke(
        Command(resume={"action": "approve"}), config=config
    )
    second_pending = await pending(compiled_graph, config, second)
    assert second_pending is not None
    assert second_pending["stage"] == "plan_review"
    assert second["itinerary"], "itinerary should exist by the second interrupt"
    assert second["weather"], "weather should have been fetched"
    assert second["hotels"], "hotels should have been fetched"

    # Approve the plan -> runs to completion.
    final = await compiled_graph.ainvoke(
        Command(resume={"action": "approve"}), config=config
    )
    assert await pending(compiled_graph, config, final) is None
    budget = final["budget"]
    assert budget["transport_inr"] == 18000
    assert budget["hotel_inr"] == 8000
    assert budget["onground_inr"] == 12000
    assert budget["total_inr"] == 38000
    assert budget["per_person_inr"] == 19000  # 2 travellers


@pytest.mark.asyncio
async def test_revision_loop_terminates(compiled_graph):
    """The original had no revision cap: Human_Review -> llm_decider -> ... ->
    Human_Review could spin until the recursion limit blew up."""
    from langgraph.types import Command
    from app.graph.state import new_state

    config = _config("trip_revise")
    await compiled_graph.ainvoke(new_state("trip_revise", "Delhi to Goa"), config=config)

    for _ in range(nodes.MAX_REVISIONS + 2):
        result = await compiled_graph.ainvoke(
            Command(resume={"action": "revise_transport", "feedback": "cheaper please"}),
            config=config,
        )
        if await pending(compiled_graph, config, result) is None:
            break
    else:
        pytest.fail("revision loop never terminated")

    assert result["revision_count"] >= nodes.MAX_REVISIONS


@pytest.mark.asyncio
async def test_state_survives_a_new_graph_instance(compiled_graph):
    """Checkpointed state must be resumable — the old code made a fresh
    MemorySaver per call and a fresh uuid4 thread_id, so nothing ever was."""
    from langgraph.types import Command
    from app.graph.state import new_state

    config = _config("trip_persist")
    await compiled_graph.ainvoke(new_state("trip_persist", "Delhi to Goa"), config=config)

    # Same checkpointer, newly compiled graph object.
    checkpointer = compiled_graph.checkpointer
    reloaded = build_graph_definition().compile(checkpointer=checkpointer)

    snapshot = await reloaded.aget_state(config)
    assert snapshot.next, "a paused graph must have pending work"
    assert snapshot.values["transport_summary"]

    resumed = await reloaded.ainvoke(Command(resume={"action": "approve"}), config=config)
    assert resumed["itinerary"]


# ---------------------------------------------------------------------------
# Message accumulation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_messages_do_not_duplicate(compiled_graph):
    """Nodes used to return [*state["messages"], new] into an add_messages
    reducer, re-appending the whole history at every step."""
    from langgraph.types import Command
    from app.graph.state import new_state

    config = _config("trip_msgs")
    await compiled_graph.ainvoke(
        new_state("trip_msgs", "Delhi to Goa"), config=config
    )
    await compiled_graph.ainvoke(Command(resume={"action": "approve"}), config=config)
    final = await compiled_graph.ainvoke(Command(resume={"action": "approve"}), config=config)

    contents = [m.content for m in final["messages"]]
    assert len(contents) == len(set(contents)), "duplicate messages in transcript"
    # Nine nodes, at most one message each, plus review acknowledgements.
    assert len(contents) < 15, f"message list grew to {len(contents)}"


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def test_routers_handle_garbage_decisions():
    """route_from_llm_decision used to feed raw LLM text into the edge map."""
    assert route_after_plan_review({"review_decision": "Hotels."}) == "finalise"
    assert route_after_plan_review({"review_decision": None}) == "finalise"
    assert route_after_plan_review({}) == "finalise"
    assert route_after_plan_review({"review_decision": "revise_plan"}) == "build_itinerary"
    assert route_after_plan_review(
        {"review_decision": "revise_plan", "revision_count": 99}
    ) == "finalise"

    assert route_after_transport_review({"review_decision": "approve"}) == "enrich"
    assert route_after_transport_review({}) == "enrich"
    assert route_after_transport_review(
        {"review_decision": "revise_transport", "revision_count": 1}
    ) == "plan_transport_strategy"


def test_route_after_brief_fails_closed():
    assert route_after_brief({"brief": {}}) == "failed"
    assert route_after_brief({"brief": {"origin": "Delhi"}}) == "failed"
    assert route_after_brief(
        {"brief": {"origin": "Delhi", "destination": "Goa"}}
    ) == "plan_transport_strategy"


@pytest.mark.asyncio
async def test_failed_brief_short_circuits(compiled_graph, monkeypatch):
    from app.graph.state import new_state

    async def broken(**kwargs):
        raise ValueError("Could not determine origin and destination")

    monkeypatch.setattr(nodes.extractor, "build_brief", broken)
    config = _config("trip_bad")
    result = await compiled_graph.ainvoke(new_state("trip_bad", "hello"), config=config)
    assert await pending(compiled_graph, config, result) is None
    assert result["errors"]
    assert not result.get("itinerary")
