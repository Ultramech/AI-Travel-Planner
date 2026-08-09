"""Unit tests pinning the specific provider bugs that were fixed."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.providers import geo, hotels, itinerary
from app.providers.extractor import TripBrief

FUTURE = (date.today() + timedelta(days=10)).isoformat()
FUTURE_END = (date.today() + timedelta(days=12)).isoformat()


# ---------------------------------------------------------------------------
# Hotel price parsing — the old code's float("₹5,000") threw on every hotel,
# so a bare `except` made every total read "N/A".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("₹5,000", 5000.0),
        ("Rs. 4,500", 4500.0),
        ("$120", 120.0),
        ("1,23,456", 123456.0),
        (7999, 7999.0),
        (2500.50, 2500.50),
        ("N/A", None),
        ("", None),
        (None, None),
    ],
)
def test_price_parsing(raw, expected):
    assert hotels.parse_price(raw) == expected


def test_hotel_total_is_nights_not_days():
    result = {
        "city": "Goa", "checkin": FUTURE, "checkout": FUTURE_END, "nights": 2,
        "adults": 2,
        "hotels": [{"name": "X", "rating": 4.0, "nightly_price_inr": 3000.0,
                    "total_price_inr": 6000.0, "nights": 2, "link": ""}],
    }
    rendered = hotels.render(result)
    assert "₹3,000/night" in rendered
    assert "₹6,000 total for 2 night(s)" in rendered


def test_nights_calculation():
    assert hotels._nights("2026-03-01", "2026-03-04") == 3
    assert hotels._nights("2026-03-01", "2026-03-01") == 1  # never zero
    assert hotels._nights("garbage", "2026-03-04") == 1


# ---------------------------------------------------------------------------
# Cab signature — every caller passed `persons` positionally into `time`.
# ---------------------------------------------------------------------------

def test_cab_signature_has_no_positional_trap():
    import inspect

    from app.providers.cabs import search_cabs

    params = list(inspect.signature(search_cabs).parameters)
    # persons must come before any optional time-like argument
    assert params.index("persons") < params.index("departure_time")
    assert params[:3] == ["origin", "destination", "travel_date"]


# ---------------------------------------------------------------------------
# Geo — deterministic instead of LLM-invented
# ---------------------------------------------------------------------------

def test_iata_lookup_from_dataset():
    assert geo.lookup_iata("Delhi") == "DEL"
    assert geo.lookup_iata("goa") == "GOI"
    assert geo.lookup_iata("  Mumbai  ") == "BOM"
    assert geo.lookup_iata("Bangalore") == geo.lookup_iata("Bengaluru")


def test_normalise_handles_unicode_hyphen():
    # The station dataset uses U+2010, which broke slug building.
    assert geo.normalise_city("DELHI‐SARAI ROHILLA") == "delhi-sarai rohilla"
    assert geo.normalise_city("Delhi Cantt.") == "delhi cantt"


def test_station_lookup_prefers_major_terminals():
    stations = geo.lookup_stations("Delhi", limit=3)
    assert stations, "Delhi must resolve to at least one station"
    assert stations[0]["category"] == "NSG1"
    assert all(len(s["code"]) >= 2 for s in stations)


def test_station_slug_is_url_safe():
    slug = geo.station_slug("DELHI‐SARAI ROHILLA", "DEE")
    assert slug == "DELHI-SARAI-ROHILLA-DEE"
    assert " " not in slug and "." not in slug


def test_unknown_city_returns_nothing_rather_than_wrong_station():
    assert geo.lookup_stations("Zzzznotacity") == []
    assert geo.lookup_iata("Zzzznotacity") is None


# ---------------------------------------------------------------------------
# Trip brief validation — the old parsed dict had none
# ---------------------------------------------------------------------------

def test_brief_rejects_end_before_start():
    with pytest.raises(ValueError, match="before start_date"):
        TripBrief(origin="Delhi", destination="Goa", start_date=FUTURE_END,
                  end_date=FUTURE, duration=3)


def test_brief_rejects_same_city():
    with pytest.raises(ValueError, match="must differ"):
        TripBrief(origin="Delhi", destination="delhi", start_date=FUTURE,
                  end_date=FUTURE_END, duration=3)


def test_brief_derives_duration_from_dates():
    brief = TripBrief(origin="Delhi", destination="Goa", start_date=FUTURE,
                      end_date=FUTURE_END, duration=99)
    assert brief.duration == 3  # dates win over a bogus duration


def test_brief_clamps_persons():
    with pytest.raises(ValueError):
        TripBrief(origin="Delhi", destination="Goa", start_date=FUTURE,
                  end_date=FUTURE_END, duration=3, persons=500)


# ---------------------------------------------------------------------------
# Budget extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("ESTIMATED_ONGROUND_BUDGET_INR: 45,000", 45000.0),
        ("estimated_onground_budget_inr = ₹12000", 12000.0),
        ("ESTIMATED_ONGROUND_BUDGET_INR: Rs. 8,500.50", 8500.50),
        ("no budget line here", None),
    ],
)
def test_budget_extraction(text, expected):
    assert itinerary.extract_budget(text) == expected


# ---------------------------------------------------------------------------
# LLM output handling — the old code compared raw content to a set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"modes": ["train"]}', {"modes": ["train"]}),
        ('```json\n{"modes": ["bus"]}\n```', {"modes": ["bus"]}),
        ('Sure! Here is the JSON:\n{"modes": ["cab"]}\nHope that helps.',
         {"modes": ["cab"]}),
        ("[1, 2, 3]", [1, 2, 3]),
        ("no json at all", None),
    ],
)
def test_json_extraction_tolerates_llm_chatter(raw, expected):
    from app.providers.llm import extract_json

    assert extract_json(raw) == expected


@pytest.mark.asyncio
async def test_choose_normalises_messy_answers(monkeypatch):
    from app.providers import llm as llm_provider

    options = {"direct_flights", "direct_trains", "bus_or_cab_only"}

    for raw, expected in [
        ("direct_trains", "direct_trains"),
        ("**direct_flights**", "direct_flights"),
        ('"bus_or_cab_only"', "bus_or_cab_only"),
        ("I recommend direct_trains for this route.", "direct_trains"),
        ("something unrelated entirely", "bus_or_cab_only"),  # falls back
    ]:
        async def fake_ask(*args, _raw=raw, **kwargs):
            return _raw

        monkeypatch.setattr(llm_provider, "ask", fake_ask)
        result = await llm_provider.choose("pick", options, default="bus_or_cab_only")
        assert result == expected, f"{raw!r} -> {result!r}, wanted {expected!r}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_returns_hit_without_recalling(monkeypatch):
    from app.core.cache import cache, cached

    cache._local.clear()
    calls = {"n": 0}

    @cached("test:ns", ttl=60, single_flight=False)
    async def expensive(city: str) -> dict:
        calls["n"] += 1
        return {"city": city}

    assert await expensive("Delhi") == {"city": "Delhi"}
    assert await expensive("Delhi") == {"city": "Delhi"}
    assert calls["n"] == 1, "second call should be served from cache"

    # Normalised keys: whitespace/case must not create a new entry.
    await expensive("  delhi ")
    assert calls["n"] == 1

    # skip_cache forces a refresh — used when a user asks for different options.
    await expensive("Delhi", skip_cache=True)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_cache_key_is_stable_and_namespaced():
    from app.core.cache import build_key

    a = build_key("hotels", "Goa", checkin="2026-01-01")
    b = build_key("hotels", " goa ", checkin="2026-01-01")
    c = build_key("hotels", "Goa", checkin="2026-01-02")
    assert a == b
    assert a != c
    assert a.startswith("tp:v1:hotels:")


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_opens_and_stops_hammering_a_dead_provider():
    from app.core.resilience import CircuitOpenError, ProviderError, resilient

    calls = {"n": 0}

    @resilient("test_dead_provider", timeout=2.0, retries=0, failure_threshold=2)
    async def always_fails() -> None:
        calls["n"] += 1
        raise RuntimeError("upstream is down")

    for _ in range(2):
        with pytest.raises(ProviderError):
            await always_fails()
    assert calls["n"] == 2

    # Third call must be rejected by the breaker, not sent upstream.
    with pytest.raises(CircuitOpenError):
        await always_fails()
    assert calls["n"] == 2, "circuit should have blocked the third call"


@pytest.mark.asyncio
async def test_gather_providers_isolates_one_failure():
    from app.core.resilience import gather_providers

    async def good():
        return {"ok": 1}

    async def bad():
        raise RuntimeError("boom")

    results = await gather_providers({"good": good(), "bad": bad()})
    assert results["good"]["ok"] is True
    assert results["good"]["data"] == {"ok": 1}
    assert results["bad"]["ok"] is False
    assert "boom" in results["bad"]["error"]


@pytest.mark.asyncio
async def test_gather_providers_enforces_an_overall_budget():
    """The outer budget stops one request running unboundedly long even when
    every individual provider is still inside its own timeout."""
    import asyncio

    from app.core.resilience import gather_providers

    async def quick():
        return "done"

    async def slow():
        await asyncio.sleep(10)
        return "never"

    results = await gather_providers(
        {"quick": quick(), "slow": slow()}, overall_timeout=0.4
    )
    assert results["quick"] == {"ok": True, "data": "done"}
    assert results["slow"]["ok"] is False
    assert "overall budget" in results["slow"]["error"]


# ---------------------------------------------------------------------------
# Bus card parsing — all three bugs below were found by running the scraper
# against the live AbhiBus site, not by reading the code.
# ---------------------------------------------------------------------------

REAL_CARD_LINES = [
    "Upto ₹40 off. Code:RETURN500W",       # promo banner, NOT the operator
    "GOLDLINE SUPERDELUXE",                 # the actual operator
    "VOLVO 9600 AC Seater/Sleeper (2 + 1)",
    "Starts from Delhi",
    "21:55", "Delhi", "05h.35m", "03:30", "Jaipur",
    "Save ₹40",                             # discount, NOT the fare
    "₹400",                                 # struck-through original
    "From",
    "₹360",                                 # the payable fare
    "Select Seats", "42 Seats Left",
]


def test_fare_is_payable_price_not_the_discount():
    """min() over every ₹ figure returned 40 — the 'Save ₹40' discount."""
    from app.providers.buses import _extract_fare

    assert _extract_fare(REAL_CARD_LINES) == 360


def test_fare_falls_back_to_cheapest_non_promo_line():
    from app.providers.buses import _extract_fare

    assert _extract_fare(["Save ₹99", "₹550", "₹480"]) == 480
    assert _extract_fare(["10% off for Women", "Flash Offer : ₹36 off"]) is None


def test_promo_lines_are_not_mistaken_for_operators():
    from app.providers.buses import _is_promo

    assert _is_promo("Flash Offer : ₹36 off per seat!")
    assert _is_promo("Upto ₹40 off. Code:RETURN500W")
    assert _is_promo("10% off for Women")
    assert _is_promo("Save ₹40")
    assert not _is_promo("GOLDLINE SUPERDELUXE")
    assert not _is_promo("SHAKTI TRAVELS")


def test_abhibus_url_requires_numeric_city_ids():
    """A name-only URL 302s to the homepage; ids are mandatory."""
    from datetime import datetime

    from app.providers.buses import _SEARCH_URL_RE, _build_url

    route = {"from_name": "Delhi", "from_id": "344",
             "to_name": "Jaipur", "to_id": "241"}
    url = _build_url(route, datetime(2026, 8, 22))
    assert url == (
        "https://www.abhibus.com/bus_search/Delhi/344/Jaipur/241/22-08-2026/O"
    )
    match = _SEARCH_URL_RE.search(url)
    assert match and match.group("from_id") == "344"


# ---------------------------------------------------------------------------
# Train row filtering — etrain renders headers as <td> rows inside tbody
# ---------------------------------------------------------------------------

def test_train_number_regex_rejects_header_rows():
    from app.providers.trains import _TRAIN_NO_RE

    assert _TRAIN_NO_RE.fullmatch("19702")
    assert _TRAIN_NO_RE.fullmatch("12414")
    assert not _TRAIN_NO_RE.fullmatch("Train Number")   # column heading
    assert not _TRAIN_NO_RE.fullmatch("Su")             # day-of-week strip
    assert not _TRAIN_NO_RE.fullmatch("")


# ---------------------------------------------------------------------------
# Keyless Google Flights source. The ragged-array bug below was found by
# running real queries: DEL->GOI parsed fine while DEL->BOM and BLR->DEL threw
# TypeError, because Google returns partial date/time arrays on some routes.
# ---------------------------------------------------------------------------

class _DT:
    def __init__(self, date=None, time=None):
        self.date, self.time = date, time


def test_time_parsing_handles_ragged_arrays():
    from app.providers.flights_google import _time_str

    assert _time_str(_DT(time=[14, 45])) == "14:45"
    assert _time_str(_DT(time=[11])) == "11:00"      # minute omitted
    assert _time_str(_DT(time=[9, 5])) == "09:05"
    assert _time_str(_DT(time=[None])) is None       # used to raise TypeError
    assert _time_str(_DT(time=[14, None])) == "14:00"
    assert _time_str(_DT(time=[])) is None
    assert _time_str(_DT()) is None
    assert _time_str(None) is None


def test_date_parsing_handles_missing_parts():
    from app.providers.flights_google import _date_str

    assert _date_str(_DT(date=[2026, 8, 29])) == "2026-08-29"
    assert _date_str(_DT(date=[2026, 8])) is None    # incomplete
    assert _date_str(_DT(date=[2026, None, 29])) is None
    assert _date_str(_DT(date=[])) is None
    assert _date_str(None) is None


def test_duration_formatting():
    from app.providers.flights_google import _duration_pretty

    assert _duration_pretty(140) == "2h 20m"
    assert _duration_pretty(45) == "45m"
    assert _duration_pretty(120) == "2h 0m"
    assert _duration_pretty(0) == ""
    assert _duration_pretty(None) == ""


def test_flight_source_order_prefers_keyless_without_credentials(monkeypatch):
    """A missing Amadeus key must degrade data quality, never availability."""
    from app.config import settings
    from app.providers import flights_router as router

    monkeypatch.setattr(settings, "amadeus_client_id", "")
    monkeypatch.setattr(settings, "amadeus_client_secret", "")
    monkeypatch.setattr(settings, "flight_provider", "auto")
    assert router._order() == ["google"]
    assert router.amadeus_available() is False

    # With credentials, Amadeus leads but Google stays as fallback.
    monkeypatch.setattr(settings, "amadeus_client_id", "id")
    monkeypatch.setattr(settings, "amadeus_client_secret", "secret")
    assert router._order() == ["amadeus", "google"]

    # Explicit override wins.
    monkeypatch.setattr(settings, "flight_provider", "google")
    assert router._order() == ["google"]


@pytest.mark.asyncio
async def test_flight_router_falls_back_when_first_source_is_empty(monkeypatch):
    from app.config import settings
    from app.core.resilience import ProviderError
    from app.providers import flights_router as router

    monkeypatch.setattr(settings, "amadeus_client_id", "id")
    monkeypatch.setattr(settings, "amadeus_client_secret", "secret")
    monkeypatch.setattr(settings, "flight_provider", "auto")

    async def amadeus_dies(*a, **k):
        raise ProviderError("amadeus_flights", "test tier has no inventory")

    async def google_works(*a, **k):
        return {"flights": [{"price_inr": 5000.0}], "count": 1}

    monkeypatch.setattr(router.amadeus_provider, "search_flights", amadeus_dies)
    monkeypatch.setattr(router.google_provider, "search_flights", google_works)

    result = await router.search_flights("DEL", "BOM", "2026-09-01")
    assert result["count"] == 1
    assert result["source"] == "google"
    assert any("amadeus" in e for e in result["fallback_from"])


@pytest.mark.asyncio
async def test_flight_router_reports_cleanly_when_all_sources_fail(monkeypatch):
    from app.config import settings
    from app.core.resilience import ProviderError
    from app.providers import flights_router as router

    monkeypatch.setattr(settings, "flight_provider", "google")

    async def dead(*a, **k):
        raise ProviderError("google_flights", "upstream changed")

    monkeypatch.setattr(router.google_provider, "search_flights", dead)
    result = await router.search_flights("DEL", "BOM", "2026-09-01")
    assert result["count"] == 0
    assert result["flights"] == []
    assert "upstream changed" in result["note"]
