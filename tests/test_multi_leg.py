"""Multi-leg journeys, journey time, and price semantics.

Every case here is a bug that shipped and was found by running the real
providers, not by reading the code:

  * Shimla→Mumbai was answered with "FLIGHT IXC → BOM" and no mention that IXC
    is Chandigarh, 113 km away, nor any cost for reaching it.
  * A 24 h 30 m Aligarh→Mumbai train reached the comparison prompt as
    "dep 15:45 arr 16:15" — its duration was dropped — and the itinerary was
    told the travellers arrived on the departure date.
  * Google Flights quotes the whole party's fare; the prompt then asked for a
    group total, so two travellers were billed twice over.
  * A route with no service crashed the parser, was retried three times, and
    was reported as a healthy provider returning nothing.
"""

from __future__ import annotations

import pytest

from app.graph import nodes
from app.providers import journey


# ---------------------------------------------------------------------------
# Journey time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("24:30H", 1470),   # etrain's format for a day-and-a-half journey
        ("18:05", 1085),
        ("3h 0m", 180),
        ("1h 30m", 90),
        ("45m", 45),
        ("2h", 120),
        ("", None),
        (None, None),
        ("unknown", None),
    ],
)
def test_parse_duration_minutes(text, expected):
    assert nodes.parse_duration_minutes(text) == expected


def test_overnight_train_arrives_the_next_day():
    """The exact Aligarh→Mumbai service that exposed this."""
    arrival_date, arrival_time = nodes.derive_arrival(
        "2026-08-25",
        departs="15:45",
        arrives="16:15",
        duration_minutes=nodes.parse_duration_minutes("24:30H"),
    )
    assert arrival_date == "2026-08-26"
    assert arrival_time == "16:15"


def test_duration_beats_the_midnight_heuristic():
    """"arrival clock < departure clock means +1 day" truncates >24 h journeys.

    Without the duration this journey reads as thirty minutes long.
    """
    without = nodes.derive_arrival("2026-08-25", departs="15:45", arrives="16:15")
    assert without[0] == "2026-08-25"  # the old, wrong answer

    with_duration = nodes.derive_arrival(
        "2026-08-25", departs="15:45", arrives="16:15", duration_minutes=1470
    )
    assert with_duration[0] == "2026-08-26"


def test_midnight_crossing_without_duration():
    assert nodes.derive_arrival(
        "2026-08-25", departs="22:00", arrives="06:30"
    ) == ("2026-08-26", "06:30")


def test_same_day_journey_keeps_its_date():
    assert nodes.derive_arrival(
        "2026-08-25", departs="09:00", arrives="12:30"
    ) == ("2026-08-25", "12:30")


def test_derive_arrival_survives_junk():
    assert nodes.derive_arrival("not-a-date", departs="09:00") == (None, None)
    assert nodes.derive_arrival("2026-08-25") == (None, None)


# ---------------------------------------------------------------------------
# Connecting legs
# ---------------------------------------------------------------------------

def test_needs_leg_only_when_the_city_differs():
    assert journey.needs_leg("Shimla", "Chandigarh") is True
    assert journey.needs_leg("Delhi", "Delhi") is False
    assert journey.needs_leg("Bangalore", "Bengaluru") is False  # alias, same place
    assert journey.needs_leg("Delhi", None) is False
    assert journey.needs_leg("", "Delhi") is False


@pytest.mark.asyncio
async def test_plan_connections_builds_an_access_leg(monkeypatch):
    async def fake_leg(origin, gateway, **kwargs):
        return {
            "role": kwargs["role"], "for_mode": kwargs["for_mode"],
            "from": origin, "to": gateway, "distance_km": 113.0,
            "cheapest_inr": 1970.0, "trains": [],
        }

    monkeypatch.setattr(journey, "describe_leg", fake_leg)

    legs = await journey.plan_connections(
        {
            "origin": "Shimla", "destination": "Mumbai",
            "origin_airport_city": "Chandigarh",
            "destination_airport_city": "Mumbai",
            "start_date": "2026-08-25", "persons": 2,
        },
        ["flight"],
    )
    assert len(legs) == 1
    assert (legs[0]["from"], legs[0]["to"]) == ("Shimla", "Chandigarh")
    assert legs[0]["role"] == "access"


@pytest.mark.asyncio
async def test_no_legs_for_a_city_with_its_own_airport(monkeypatch):
    monkeypatch.setattr(
        journey, "describe_leg",
        lambda *a, **k: pytest.fail("should not price a leg that does not exist"),
    )
    legs = await journey.plan_connections(
        {
            "origin": "Delhi", "destination": "Mumbai",
            "origin_airport_city": "Delhi", "destination_airport_city": "Mumbai",
            "origin_railhead_city": "Delhi", "destination_railhead_city": "Mumbai",
            "start_date": "2026-08-25", "persons": 2,
        },
        ["flight", "train"],
    )
    assert legs == []


@pytest.mark.asyncio
async def test_legs_are_not_priced_for_unsearched_modes(monkeypatch):
    """No point costing a drive to the airport for a trip going by train."""
    calls: list[str] = []

    async def fake_leg(origin, gateway, **kwargs):
        calls.append(kwargs["for_mode"])
        return {"role": kwargs["role"], "for_mode": kwargs["for_mode"],
                "from": origin, "to": gateway, "cheapest_inr": 100.0}

    monkeypatch.setattr(journey, "describe_leg", fake_leg)
    await journey.plan_connections(
        {
            "origin": "Shimla", "destination": "Mumbai",
            "origin_airport_city": "Chandigarh",
            "origin_railhead_city": "Shimla",
            "start_date": "2026-08-25", "persons": 1,
        },
        ["train"],   # flight not searched
    )
    assert calls == []


def test_leg_cost_excludes_trivial_transfers():
    legs = [
        {"for_mode": "flight", "cheapest_inr": 1970.0},
        {"for_mode": "flight", "cheapest_inr": 500.0, "trivial": True},
        {"for_mode": "train", "cheapest_inr": 800.0},
    ]
    assert journey.leg_cost(legs, "flight") == 1970.0
    assert journey.leg_cost(legs, "train") == 800.0
    assert journey.leg_cost(legs, "bus") == 0


def test_an_impossible_transfer_is_never_priced():
    """Port Blair's "railhead" resolved to Kolkata, 1,370 km across the sea.

    The leg was quoted at ₹18,310 as a road transfer. A fare here is worse than
    no fare — it invents a road.
    """
    legs = [{"for_mode": "train", "from": "Kolkata", "to": "Port Blair",
             "distance_km": 1370.0, "cheapest_inr": None, "implausible": True,
             "note": "too far to be a transfer"}]
    assert journey.leg_cost(legs, "train") == 0
    assert "UNUSABLE_GATEWAY" in journey.facts(legs)[0]
    assert "CONNECTING_LEG" not in journey.facts(legs)[0]
    assert "not a realistic transfer" in journey.render(legs)


@pytest.mark.asyncio
async def test_describe_leg_drops_the_fare_beyond_the_sensible_limit(monkeypatch):
    from app.providers import cabs as cabs_provider
    from app.providers import trains as trains_provider

    async def fake_cabs(origin, destination, **kwargs):
        return {"distance_km": 1370.0, "options": [
            {"vehicle": "Sedan", "estimated_fare_inr": 18310.0,
             "per_person_inr": 9155.0},
        ]}

    async def fake_trains(*a, **k):
        return {"trains": [], "note": "none"}

    monkeypatch.setattr(cabs_provider, "search_cabs", fake_cabs)
    monkeypatch.setattr(trains_provider, "search_trains", fake_trains)

    leg = await journey.describe_leg(
        "Kolkata", "Port Blair", travel_date="2026-08-25", persons=2,
        role="egress", for_mode="train",
    )
    assert leg["implausible"] is True
    assert leg["cheapest_inr"] is None
    assert leg["cab"] is None


def test_render_and_facts_skip_trivial_legs():
    assert journey.render([{"for_mode": "flight", "trivial": True, "from": "A",
                            "to": "B"}]) == ""
    assert journey.facts([{"for_mode": "flight", "trivial": True, "from": "A",
                           "to": "B"}]) == []


def test_facts_mention_the_leg_and_its_cost():
    line = journey.facts([
        {"for_mode": "flight", "from": "Shimla", "to": "Chandigarh",
         "distance_km": 113.0, "cheapest_inr": 1970.0, "trains": []},
    ])[0]
    assert "Shimla -> Chandigarh" in line
    assert "113km" in line
    assert "1,970" in line


# ---------------------------------------------------------------------------
# The prompt must not hide the substitution
# ---------------------------------------------------------------------------

def test_gateway_note_names_the_real_departure_city():
    note = nodes._gateway_note(
        {
            "origin": "Shimla", "destination": "Mumbai",
            "origin_airport_city": "Chandigarh", "origin_iata": "IXC",
            "origin_airport_exact": False,
            "destination_airport_exact": True,
        }
    )
    assert "Shimla has no airport" in note
    assert "Chandigarh" in note and "IXC" in note


def test_gateway_note_is_empty_for_a_direct_journey():
    assert nodes._gateway_note(
        {
            "origin": "Delhi", "destination": "Mumbai",
            "origin_airport_exact": True, "destination_airport_exact": True,
            "origin_railhead_exact": True, "destination_railhead_exact": True,
        }
    ) == ""


# ---------------------------------------------------------------------------
# Price semantics
# ---------------------------------------------------------------------------

def test_flight_prices_are_labelled_as_group_totals():
    """Measured live: Google quotes ₹6,074 / ₹12,148 / ₹18,222 for 1 / 2 / 3."""
    facts = _flight_facts(
        {
            "flight": {
                "flights": [{
                    "route": "IXC → BOM", "departure": "2026-08-25T13:55",
                    "arrival": "2026-08-25T18:25", "duration": "3h 0m",
                    "stops": 1, "price_total_inr": 16302.0,
                    "price_per_person_inr": 8151.0,
                }],
            }
        },
        persons=2,
    )
    assert "TOTAL for all 2 traveller(s)" in facts[0]
    assert "16,302" in facts[0]
    assert "32,604" not in facts[0]   # the doubling this test exists to prevent


def test_bus_fares_are_labelled_per_person():
    facts = _flight_facts(
        {"bus": {"buses": [{"operator": "X", "departure": "22:00",
                            "arrival": "06:30", "fare_inr": 900}]}},
        persons=2,
    )
    assert "PER PERSON" in facts[0]


def test_train_facts_carry_the_journey_time():
    facts = _flight_facts(
        {"train": {"trains": [{
            "train_no": "14314", "train_name": "BE LTT EXPRESS",
            "from": "ALJN", "to": "LTT", "departs": "15:45",
            "arrives": "16:15", "duration": "24:30H",
        }]}},
        persons=1,
    )
    assert "24h 30m" in facts[0]
    assert "2026-08-26" in facts[0]   # and says it lands the next day


# ---------------------------------------------------------------------------
# Mode selection: deterministic, and never a silent omission
# ---------------------------------------------------------------------------

async def _strategy(monkeypatch, distance, brief=None, feedback=None):
    from app.providers import cabs as cabs_provider

    async def fake_distance(origin, destination):
        return distance

    monkeypatch.setattr(cabs_provider, "estimate_distance_km", fake_distance)
    state = {
        "brief": brief or {
            "origin": "Aligarh", "destination": "Patiala",
            "origin_iata": "AGR", "destination_iata": "IXC", "persons": 2,
        },
    }
    if feedback:
        state["review_feedback"] = feedback
    result = await nodes.plan_transport_strategy(state)
    return result["transport_strategy"]


@pytest.mark.asyncio
async def test_medium_distance_searches_everything(monkeypatch):
    """Aligarh→Patiala, 540 km.

    The LLM filter used to drop flight *and* cab here before anything had been
    looked up, and the summary then asserted "no flights are available" — a
    claim about AGR→IXC that nothing had checked.
    """
    strategy = await _strategy(monkeypatch, 540.0)
    assert strategy["modes"] == ["bus", "cab", "flight", "train"]
    assert strategy["skipped"] == {}


@pytest.mark.asyncio
async def test_short_hop_skips_flight_with_a_reason(monkeypatch):
    strategy = await _strategy(monkeypatch, 233.0)
    assert "flight" not in strategy["modes"]
    assert "233 km" in strategy["skipped"]["flight"]


@pytest.mark.asyncio
async def test_very_long_route_skips_road_modes(monkeypatch):
    strategy = await _strategy(monkeypatch, 2325.0)
    assert set(strategy["skipped"]) == {"cab", "bus"}
    assert "flight" in strategy["modes"]


@pytest.mark.asyncio
async def test_same_airport_skips_flight(monkeypatch):
    strategy = await _strategy(
        monkeypatch, 25.0,
        brief={"origin": "Delhi", "destination": "Noida",
               "origin_iata": "DEL", "destination_iata": "DEL", "persons": 1},
    )
    assert "flight" not in strategy["modes"]
    assert "same airport" in strategy["skipped"]["flight"]


@pytest.mark.asyncio
async def test_user_request_overrides_a_skip(monkeypatch):
    """The traveller's own words outrank every distance rule."""
    strategy = await _strategy(
        monkeypatch, 233.0, feedback="I would rather take a flight"
    )
    assert "flight" in strategy["modes"]
    assert "flight" not in strategy["skipped"]


@pytest.mark.asyncio
async def test_unknown_distance_searches_everything(monkeypatch):
    strategy = await _strategy(monkeypatch, None)
    assert strategy["modes"] == ["bus", "cab", "flight", "train"]


@pytest.mark.asyncio
async def test_never_leaves_the_traveller_with_no_modes(monkeypatch):
    strategy = await _strategy(
        monkeypatch, 5.0,
        brief={"origin": "Delhi", "destination": "Delhi Cantt",
               "origin_iata": "DEL", "destination_iata": "DEL", "persons": 1},
    )
    assert strategy["modes"]


def test_a_skipped_mode_is_never_reported_as_unavailable():
    facts = _flight_facts(
        {"train": {"trains": [{
            "train_no": "12002", "train_name": "SHATABDI", "from": "NDLS",
            "to": "AGC", "departs": "06:00", "arrives": "07:50",
            "duration": "1:50H",
        }]}},
        persons=2,
        strategy={"modes": ["train"],
                  "skipped": {"flight": "only ~233 km"}},
    )
    joined = "\n".join(facts)
    assert "NOT_SEARCHED FLIGHT" in joined
    assert "NO_OPTIONS FLIGHT" not in joined


# ---------------------------------------------------------------------------
# The train search must use the railhead, not the station-less city
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_train_search_departs_from_the_railhead(monkeypatch):
    """Gangtok has no railway station; New Jalpaiguri is its railhead.

    Querying "Gangtok" can only ever answer "not in the dataset". Live, this
    change turned zero results into ten real NJP→DLI trains.
    """
    seen: dict[str, str] = {}

    async def fake_trains(origin, destination, date, **kwargs):
        seen["origin"], seen["destination"] = origin, destination
        return {"trains": [], "count": 0, "note": "stub"}

    from app.providers import buses as buses_provider
    from app.providers import trains as trains_provider

    monkeypatch.setattr(trains_provider, "search_trains", fake_trains)
    monkeypatch.setattr(buses_provider, "search_buses", fake_trains)
    monkeypatch.setattr(journey, "plan_connections",
                        lambda *a, **k: _async_value([]))

    await nodes.search_transport({
        "brief": {
            "origin": "Gangtok", "destination": "Delhi",
            "origin_railhead_city": "New Jalpaiguri",
            "destination_railhead_city": "Delhi",
            "start_date": "2026-08-25", "persons": 1,
        },
        "transport_strategy": {"modes": ["train"]},
    })
    assert seen["origin"] == "New Jalpaiguri"
    assert seen["destination"] == "Delhi"


def _async_value(value):
    async def _inner():
        return value
    return _inner()


# ---------------------------------------------------------------------------
# Absences must be stated, not inferred
# ---------------------------------------------------------------------------

def test_a_leg_alone_does_not_imply_a_service_exists():
    """Gangtok→Kochi found zero IXB→COK flights.

    The connecting-leg fact on its own was enough for the model to answer "fly
    Bagdogra to Kochi", inventing the flight the leg was supposed to feed.
    """
    facts = _flight_facts(
        {"flight": {"flights": [], "note": "No airline flies IXB→COK."}},
        persons=2,
        legs=[{"for_mode": "flight", "from": "Gangtok", "to": "Bagdogra",
               "distance_km": 125.0, "cheapest_inr": 2120.0, "trains": []}],
        expect_summary=True,
    )
    assert facts is None  # short-circuits before ever reaching the LLM


def test_no_options_is_spelled_out_when_another_mode_works():
    facts = _flight_facts(
        {
            "flight": {"flights": [], "note": "No airline flies IXB→COK."},
            "train": {"trains": [{
                "train_no": "1", "train_name": "T", "from": "NJP", "to": "DLI",
                "departs": "10:00", "arrives": "12:00", "duration": "26:00H",
            }]},
        },
        persons=1,
        legs=[{"for_mode": "flight", "from": "Gangtok", "to": "Bagdogra",
               "distance_km": 125.0, "cheapest_inr": 2120.0, "trains": []}],
    )
    joined = "\n".join(facts)
    assert "NO_OPTIONS FLIGHT" in joined
    # the flight's leg must not be presented as part of a bookable journey
    assert "CONNECTING_LEG" not in joined


def test_every_mode_empty_reports_each_reason():
    import asyncio

    result = asyncio.run(nodes.recommend_transport({
        "brief": {"origin": "Gangtok", "destination": "Kochi", "persons": 2,
                  "start_date": "2026-08-25", "budget_level": "medium"},
        "transport_options": {
            "flight": {"flights": [], "note": "No airline flies IXB→COK."},
            "train": {"trains": [], "note": "No station in the dataset."},
        },
    }))
    summary = result["transport_summary"]
    assert "No bookable transport" in summary
    assert "No airline flies" in summary
    assert "No station in the dataset" in summary


def _flight_facts(
    options: dict, persons: int, legs: list | None = None,
    expect_summary: bool = False, strategy: dict | None = None,
) -> list[str] | None:
    """Rebuild the fact lines the recommendation prompt is given.

    Returns ``None`` when the node short-circuits without calling the LLM.
    """
    import asyncio

    captured: list[str] = []

    async def fake_ask(prompt, **kwargs):
        captured.append(prompt)
        return "ARRIVAL_TIME: 10:00\nTRANSPORT_COST_INR: 1000"

    from app.providers import llm as llm_provider

    original = llm_provider.ask
    llm_provider.ask = fake_ask
    try:
        asyncio.run(nodes.recommend_transport({
            "brief": {"origin": "A", "destination": "B", "persons": persons,
                      "start_date": "2026-08-25", "budget_level": "medium"},
            "transport_options": options,
            "connecting_legs": legs or [],
            "transport_strategy": strategy or {},
        }))
    finally:
        llm_provider.ask = original

    if expect_summary or not captured:
        return None

    body = captured[0].split("OPTIONS FOUND:")[1].split("HOW TO READ")[0]
    return [line for line in body.strip().splitlines() if line.strip()]
