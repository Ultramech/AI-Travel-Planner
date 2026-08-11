"""Fares, connection times, hub routing and the return leg.

Every case here comes from something that was wrong when the code was run:

  * a train recommendation left ``budget.transport_inr`` empty, so the trip
    total silently omitted its largest line;
  * the plan named a road leg to the gateway but never checked it landed before
    the flight departed;
  * "No airline flies DEL→COK" — a route with dozens of daily departures —
    because the upstream parser reads alliance metadata before the flights;
  * nothing ever planned the journey home.
"""

from __future__ import annotations

import pytest

from app.providers import flights_router, journey, trains


# ---------------------------------------------------------------------------
# Train fares — the budget's missing line
# ---------------------------------------------------------------------------

def test_fares_scale_with_distance_and_class():
    short = trains.estimate_fares(200, persons=1)
    long = trains.estimate_fares(1500, persons=1)

    assert short["SL"]["per_person_inr"] < short["3A"]["per_person_inr"]
    assert short["3A"]["per_person_inr"] < short["2A"]["per_person_inr"]
    assert short["2A"]["per_person_inr"] < short["1A"]["per_person_inr"]
    assert long["SL"]["per_person_inr"] > short["SL"]["per_person_inr"]


def test_long_journeys_taper_rather_than_scaling_flat():
    """IR fares per km fall with distance; a flat rate doubles a long trip."""
    per_km_500 = trains.estimate_fares(500, 1)["3A"]["per_person_inr"] / 500
    per_km_2000 = trains.estimate_fares(2000, 1)["3A"]["per_person_inr"] / 2000
    assert per_km_2000 < per_km_500


def test_fare_totals_multiply_by_party():
    fares = trains.estimate_fares(800, persons=3)
    assert fares["SL"]["total_inr"] == fares["SL"]["per_person_inr"] * 3


def test_no_distance_means_no_invented_fare():
    assert trains.estimate_fares(None, 2) == {}
    assert trains.estimate_fares(0, 2) == {}


def test_estimates_land_in_the_right_band():
    """Sanity-checked against real IRCTC fares, ±25%."""
    # Delhi–Kolkata 3A is roughly ₹1,900.
    assert 1400 <= trains.estimate_fares(1472, 1)["3A"]["per_person_inr"] <= 2400
    # Delhi–Agra sleeper is a couple of hundred rupees.
    assert 120 <= trains.estimate_fares(233, 1)["SL"]["per_person_inr"] <= 400


# ---------------------------------------------------------------------------
# Connection time
# ---------------------------------------------------------------------------

def _leg(minutes: int = 240, mode: str = "flight") -> dict:
    return {"from": "Shimla", "to": "Chandigarh", "for_mode": mode,
            "travel_minutes": minutes}


def test_comfortable_connection_reports_when_to_leave():
    check = journey.check_connection(_leg(), "13:55", "2026-08-25")
    assert check["feasible"] is True
    assert check["leave_by"] == "07:55"          # 13:55 − 4h road − 2h airport
    assert not check.get("tight")


def test_night_departure_is_flagged_not_hidden():
    check = journey.check_connection(_leg(), "06:30", "2026-08-25")
    assert check["feasible"] is True
    assert check["tight"] is True
    assert "night departure" in check["warning"]


def test_unreachable_departure_is_refused():
    """A 04:45 flight needs a 22:45 start the day before."""
    check = journey.check_connection(_leg(), "04:45", "2026-08-25")
    assert check["feasible"] is False
    assert check["leave_day_offset"] == -1
    assert check["leave_date"] == "2026-08-24"
    assert "overnight" in check["warning"]


def test_station_buffer_is_smaller_than_airport_buffer():
    by_air = journey.check_connection(_leg(mode="flight"), "12:00")
    by_rail = journey.check_connection(_leg(mode="train"), "12:00")
    assert by_rail["leave_by"] > by_air["leave_by"]


def test_connection_check_declines_to_guess():
    assert journey.check_connection({"from": "A", "to": "B"}, "12:00")["feasible"] is None
    assert journey.check_connection(_leg(), None)["feasible"] is None
    assert journey.check_connection(_leg(), "not a time")["feasible"] is None


# ---------------------------------------------------------------------------
# Hub routing
# ---------------------------------------------------------------------------

def _flight(dep: str, arr: str, price: float, src: str, dst: str) -> dict:
    return {
        "origin_iata": src, "destination_iata": dst,
        "departure": dep, "arrival": arr, "arrival_time": arr[11:16],
        "price_total_inr": price, "party_size": 2, "duration_minutes": 120,
        "carriers": ["IndiGo"], "booking_link": "http://example.test",
    }


def test_legs_pair_only_with_a_workable_layover():
    first = [_flight("2026-08-25T06:00", "2026-08-25T08:00", 5000, "IXB", "DEL")]
    second = [
        _flight("2026-08-25T08:30", "2026-08-25T11:00", 6000, "DEL", "COK"),  # 30m
        _flight("2026-08-25T11:00", "2026-08-25T14:00", 6000, "DEL", "COK"),  # 3h
        _flight("2026-08-25T23:00", "2026-08-26T02:00", 4000, "DEL", "COK"),  # 15h
    ]
    pairs = flights_router._pair_legs(first, second)
    assert len(pairs) == 1
    assert pairs[0]["layover_pretty"].startswith("3h")
    assert pairs[0]["price_total_inr"] == 11000
    assert pairs[0]["stops"] == 1


def test_paired_itineraries_are_marked_self_transfer():
    """Two bookings, not a through-fare — the traveller carries that risk."""
    first = [_flight("2026-08-25T06:00", "2026-08-25T08:00", 5000, "IXB", "DEL")]
    second = [_flight("2026-08-25T11:00", "2026-08-25T14:00", 6000, "DEL", "COK")]
    pair = flights_router._pair_legs(first, second)[0]
    assert pair["self_transfer"] is True
    assert pair["is_bookable_offer"] is False
    assert pair["route"] == "IXB → DEL → COK"
    assert pair["second_leg_link"]


def test_pairs_are_cheapest_first():
    first = [_flight("2026-08-25T06:00", "2026-08-25T08:00", 5000, "IXB", "DEL")]
    second = [
        _flight("2026-08-25T11:00", "2026-08-25T14:00", 9000, "DEL", "COK"),
        _flight("2026-08-25T12:00", "2026-08-25T15:00", 6000, "DEL", "COK"),
    ]
    pairs = flights_router._pair_legs(first, second)
    assert [p["price_total_inr"] for p in pairs] == [11000, 14000]


def test_unparsable_times_never_produce_a_pairing():
    first = [_flight("nonsense", "also nonsense", 5000, "IXB", "DEL")]
    second = [_flight("2026-08-25T11:00", "2026-08-25T14:00", 6000, "DEL", "COK")]
    assert flights_router._pair_legs(first, second) == []


# ---------------------------------------------------------------------------
# Geo resolution must not depend on the LLM being up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gateways_resolve_with_the_llm_down(monkeypatch):
    """A rate-limited model made Gangtok look like a city with no air service.

    Flights, connecting legs and the whole budget vanished from the trip. The
    curated table keeps the common cases deterministic.
    """
    from app.providers import geo, llm as llm_provider

    async def down(*args, **kwargs):
        raise RuntimeError("429 quota exhausted")

    monkeypatch.setattr(llm_provider, "ask_json", down)

    air = await geo.resolve_airport("Gangtok")
    rail = await geo.resolve_railhead("Gangtok")

    assert air["iata"] == "IXB"
    assert air["airport_city"] == "Bagdogra"
    assert air["exact"] is False
    assert air["source"] == "gateway_table"
    assert rail["station_city"] == "New Jalpaiguri"


@pytest.mark.asyncio
async def test_gateway_prefers_a_served_airport_over_a_nearer_empty_one(monkeypatch):
    """Agra is closer to Aligarh, but almost nothing flies there.

    The LLM offered AGR; the honest answer for someone actually travelling is
    Delhi.
    """
    from app.providers import geo, llm as llm_provider

    async def down(*args, **kwargs):
        raise RuntimeError("should not be consulted")

    monkeypatch.setattr(llm_provider, "ask_json", down)
    assert (await geo.resolve_airport("Aligarh"))["iata"] == "DEL"


def test_cities_with_their_own_airport_bypass_the_table():
    from app.providers import geo

    assert geo.gateway_for("Delhi") is None
    assert geo.gateway_for("Mumbai") is None
    assert geo.gateway_for("Shimla") is not None


def test_dataset_iata_corrections_are_applied():
    """The bundled CSV had Coimbatore as CIB (Iowa) and Kozhikode as CCI (Kansas).

    Both would have searched flights on the wrong continent; Pune was missing.
    """
    from app.providers import geo

    assert geo.lookup_iata("Coimbatore") == "CJB"
    assert geo.lookup_iata("Kozhikode") == "CCJ"
    assert geo.lookup_iata("Pune") == "PNQ"
    assert geo.lookup_iata("Jaipur") == "JAI"


# ---------------------------------------------------------------------------
# Losing the LLM must not lose the plan
# ---------------------------------------------------------------------------

_STATE = {
    "brief": {"origin": "Aligarh", "destination": "Patiala", "persons": 2,
              "start_date": "2026-08-25", "budget_level": "medium",
              "origin_airport_exact": True, "destination_airport_exact": True},
    "transport_strategy": {"modes": ["train"], "skipped": {}},
    "connecting_legs": [{"role": "access", "for_mode": "train",
                         "from": "Aligarh", "to": "Aligarh Jn",
                         "distance_km": 60.0, "cheapest_inr": 900.0,
                         "trains": []}],
    "transport_options": {"train": {
        "trains": [{"train_no": "15707", "train_name": "KIR ASR EXPRESS",
                    "from": "ALJN", "to": "PTA", "departs": "00:20",
                    "arrives": "07:14", "duration": "6:54H"}],
        "fares": {"SL": {"class": "SL", "total_inr": 670, "per_person_inr": 335}},
    }},
}


@pytest.mark.asyncio
async def test_quota_exhaustion_degrades_the_wording_not_the_answer(monkeypatch):
    """Groq's free tier ran out mid-run and the whole node returned only errors.

    The model writes the prose; it does not hold the data. Every fare, timing
    and leg was already in hand.
    """
    from app.graph import nodes
    from app.providers import llm as llm_provider

    async def rate_limited(*args, **kwargs):
        raise RuntimeError("429 rate limit reached")

    monkeypatch.setattr(llm_provider, "ask", rate_limited)
    result = await nodes.recommend_transport(dict(_STATE))

    assert result["arrival_date"] == "2026-08-25"
    assert result["arrival_time"] == "07:14"
    # ₹670 train + ₹900 access leg, still added up correctly.
    assert result["budget"]["transport_inr"] == 1570.0
    assert "15707" in result["transport_summary"]
    assert "6h 54m" in result["transport_summary"]


@pytest.mark.asyncio
async def test_itinerary_falls_back_to_a_dated_skeleton(monkeypatch):
    """A failed plan returned nothing while the trip still said "completed".

    An empty itinerary is the one output a traveller cannot use at all.
    """
    from app.graph import nodes
    from app.providers import itinerary as itinerary_provider

    async def down(**kwargs):
        raise RuntimeError("429 quota exhausted")

    monkeypatch.setattr(itinerary_provider, "generate_itinerary", down)

    result = await nodes.build_itinerary({
        "brief": {"destination": "Kochi", "origin": "Gangtok", "duration": 4,
                  "start_date": "2026-08-25", "end_date": "2026-08-29",
                  "persons": 2},
        "arrival_date": "2026-08-26", "arrival_time": "22:15",
        "hotels_summary": "Hotel Xandari",
        "weather": [{"date": "2026-08-26", "condition": "Light rain",
                     "max_temp_c": 30, "min_temp_c": 24, "chance_of_rain": 78}],
    })

    plan = result["itinerary"]
    assert plan, "an outline beats a blank page"
    # Dates run from the real arrival, not the departure date.
    assert "2026-08-26" in plan
    assert "22:15" in plan
    assert "Light rain" in plan          # the forecast we already had
    assert result["itinerary_meta"]["degraded"] is True
    assert result["provider_health"]["itinerary"].startswith("degraded")


def test_skeleton_survives_an_unusable_arrival_date():
    from app.graph import nodes

    plan = nodes._skeleton_itinerary({"destination": "Goa"}, 3, "", None, [], "")
    assert "Arrival date unavailable" in plan


@pytest.mark.asyncio
async def test_degradation_is_recorded_not_hidden(monkeypatch):
    from app.graph import nodes
    from app.providers import llm as llm_provider

    async def rate_limited(*args, **kwargs):
        raise RuntimeError("429")

    monkeypatch.setattr(llm_provider, "ask", rate_limited)
    result = await nodes.recommend_transport(dict(_STATE))

    assert result["provider_health"]["recommendation"].startswith("degraded")
    assert any("LLM unavailable" in e["error"] for e in result["errors"])
    assert "model was unavailable" in result["transport_summary"]


# ---------------------------------------------------------------------------
# The return leg must be counted once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_outbound_is_priced_one_way_when_a_return_is_searched(monkeypatch):
    """Otherwise the journey home is paid for twice.

    Asking the outbound search for a round-trip fare *and* adding a separate
    return fare put Shimla→Mumbai at ₹34,656 when the outbound is ₹16,302 plus
    ₹1,970 of road.
    """
    from app.graph import nodes
    from app.providers import journey as journey_provider

    seen: list[tuple[str, str, object]] = []

    async def fake_flights(src, dst, date, return_date=None, **kwargs):
        seen.append((src, dst, return_date))
        return {"flights": [], "count": 0, "note": "stub"}

    monkeypatch.setattr(nodes.flights_provider, "search_flights", fake_flights)
    monkeypatch.setattr(journey_provider, "plan_connections",
                        lambda *a, **k: _async([]))

    await nodes.search_transport({
        "brief": {
            "origin": "Shimla", "destination": "Mumbai",
            "origin_iata": "IXC", "destination_iata": "BOM",
            "start_date": "2026-08-25", "end_date": "2026-08-28", "persons": 2,
        },
        "transport_strategy": {"modes": ["flight"], "skipped": {}},
    })

    outbound = [s for s in seen if s[0] == "IXC"]
    inbound = [s for s in seen if s[0] == "BOM"]
    assert outbound and outbound[0][2] is None, "outbound must be one-way"
    assert inbound, "the journey home must be searched"


def _async(value):
    async def _inner():
        return value
    return _inner()


@pytest.mark.asyncio
async def test_hub_routing_skips_hubs_that_are_the_endpoints(monkeypatch):
    tried: list[tuple[str, str]] = []

    async def fake_search(a, b, *args, **kwargs):
        tried.append((a, b))
        return {"flights": []}

    monkeypatch.setattr(flights_router, "_search", fake_search)
    await flights_router._search_via_hub("DEL", "BOM", "2026-08-25", 1, "INR", 8, False)
    # Neither endpoint may be used as its own connecting hub.
    assert all(hub not in ("DEL", "BOM") for _, hub in tried[:1])
    assert ("DEL", "DEL") not in tried and ("BOM", "BOM") not in tried
