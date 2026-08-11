"""Fault injection, deterministic ranking, and date flexibility.

These three exist because of the same realisation: the interesting failures in
this system are the quiet ones. A provider that answers successfully with
nothing, a model whose prose disagrees with the arithmetic, a date one day away
that costs a third less — none of them raise, and none of them show up in a log.
"""

from __future__ import annotations

import pytest

from app.core import chaos, decision
from app.core.decision import Candidate
from app.core.resilience import ProviderError, resilient


# ---------------------------------------------------------------------------
# Chaos injection
# ---------------------------------------------------------------------------

@pytest.fixture
def arm(monkeypatch):
    """Arm the injector for one provider."""
    def _arm(providers="etrain", mode="error", probability=1.0, env="development"):
        from app.config import settings
        monkeypatch.setattr(settings, "chaos_providers", providers)
        monkeypatch.setattr(settings, "chaos_mode", mode)
        monkeypatch.setattr(settings, "chaos_probability", probability)
        monkeypatch.setattr(settings, "app_env", env)
    return _arm


def test_disarmed_by_default():
    assert chaos.armed() is False
    assert chaos.should_fail("etrain") is False


def test_refuses_to_arm_in_production(arm):
    arm(providers="all", env="production")
    assert chaos.armed() is False
    assert chaos.should_fail("etrain") is False


def test_only_the_named_provider_fails(arm):
    arm(providers="etrain")
    assert chaos.should_fail("etrain") is True
    assert chaos.should_fail("google_flights") is False


def test_all_targets_everything(arm):
    arm(providers="all")
    assert chaos.should_fail("anything") is True


def test_status_is_reportable(arm):
    arm(providers="etrain,abhibus")
    assert chaos.status() == {
        "armed": True,
        "providers": ["abhibus", "etrain"],
        "mode": "error",
        "probability": 1.0,
    }


@pytest.mark.asyncio
async def test_injected_failure_surfaces_as_a_provider_error(arm):
    """The point: a chaos failure is handled by the *existing* machinery."""
    arm(providers="fakeprov")

    @resilient("fakeprov", timeout=5.0, retries=0)
    async def provider():
        return {"results": [1, 2, 3]}

    with pytest.raises(ProviderError) as caught:
        await provider()
    assert "simulated" in str(caught.value)


@pytest.mark.asyncio
async def test_empty_mode_simulates_the_quiet_failure(arm):
    """A provider that succeeds and returns nothing is the dangerous one.

    It is the shape that used to be cached under a full TTL and reported to the
    user as though it were real data.
    """
    arm(providers="fakeprov2", mode="empty")

    @resilient("fakeprov2", timeout=5.0, retries=0)
    async def provider():
        return {"results": [1, 2, 3]}

    result = await provider()
    assert result["results"] == []
    assert result["chaos"] is True


@pytest.mark.asyncio
async def test_untargeted_providers_are_untouched(arm):
    arm(providers="somethingelse")

    @resilient("realprov", timeout=5.0, retries=0)
    async def provider():
        return {"results": [1]}

    assert (await provider())["results"] == [1]


# ---------------------------------------------------------------------------
# Deterministic ranking
# ---------------------------------------------------------------------------

def _c(mode, label, cost, minutes, **kw) -> Candidate:
    return Candidate(mode=mode, label=label, total_inr=cost,
                     total_minutes=minutes, **kw)


def test_cheaper_and_faster_wins_outright():
    ranked = decision.rank([
        _c("train", "slow train", 5000, 1800),
        _c("flight", "good flight", 4000, 180),
    ], "medium")
    assert ranked[0].label == "good flight"


def test_budget_level_changes_the_answer():
    """The same two options, ranked differently for different travellers."""
    options = lambda: [                                    # noqa: E731
        _c("flight", "fast pricey", 20000, 180),
        _c("train", "slow cheap", 2000, 1800),
    ]
    assert decision.rank(options(), "low")[0].label == "slow cheap"
    assert decision.rank(options(), "high")[0].label == "fast pricey"


def test_self_transfer_is_penalised():
    """Two bookings means a missed connection is the traveller's problem."""
    clean, risky = decision.rank([
        _c("flight", "through fare", 10000, 300),
        _c("flight", "self transfer", 10000, 300, self_transfer=True),
    ], "medium")
    assert clean.label == "through fare"
    assert risky.comfort_penalty > clean.comfort_penalty


def test_night_start_and_overnight_arrival_cost_points():
    plain = _c("train", "day train", 1000, 600)
    awkward = _c("train", "night train", 1000, 600,
                 starts_at_night=True, arrives_next_day=True)
    assert awkward.comfort_penalty > plain.comfort_penalty


def test_unknown_duration_is_scored_as_the_slowest():
    """A missing timing must never flatter an option into first place."""
    ranked = decision.rank([
        _c("bus", "no duration known", 1000, None),
        _c("train", "known and quick", 1200, 200),
    ], "high")
    assert ranked[0].label == "known and quick"


def test_unpriced_options_sink_but_are_not_dropped():
    ranked = decision.rank([
        Candidate(mode="cab", label="no price"),
        _c("train", "priced", 1000, 600),
    ], "medium")
    assert ranked[0].label == "priced"
    assert ranked[-1].label == "no price"      # still shown, never silently lost


def test_identical_options_do_not_divide_by_zero():
    ranked = decision.rank([
        _c("flight", "a", 5000, 300),
        _c("flight", "b", 5000, 300),
    ], "medium")
    assert all(c.score is not None for c in ranked)


# ---------------------------------------------------------------------------
# Auditing the model against the maths
# ---------------------------------------------------------------------------

def test_agreement_is_detected():
    ranked = decision.rank([
        _c("train", "12450 GOA EXPRESS", 1800, 1800),
        _c("flight", "DEL → GOI", 15000, 200),
    ], "low")
    verdict = decision.compare_with_model(
        ranked, "**Recommendation**: 12450 GOA EXPRESS is the best value."
    )
    assert verdict["checked"] is True
    assert verdict["agrees"] is True


def test_disagreement_is_surfaced_with_the_gap():
    ranked = decision.rank([
        _c("train", "12450 GOA EXPRESS", 1800, 1800),
        _c("flight", "DEL → GOI", 15000, 200),
    ], "low")
    verdict = decision.compare_with_model(
        ranked, "**Recommendation**: DEL → GOI, the fastest option."
    )
    assert verdict["agrees"] is False
    assert verdict["model_appears_to_pick"] == "DEL → GOI"
    assert verdict["cost_gap_inr"] == 13200.0
    assert "worth a look" in verdict["note"]


def test_comparison_declines_when_there_is_nothing_to_compare():
    assert decision.compare_with_model([], "anything")["checked"] is False


# ---------------------------------------------------------------------------
# Flexible dates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flexible_dates_finds_a_worthwhile_saving(monkeypatch):
    from app.providers import flex_dates, flights_router

    prices = {
        "2026-08-24": 13516.0, "2026-08-25": 14846.0, "2026-08-26": 16996.0,
    }

    async def fake_search(src, dst, day, **kwargs):
        return {"cheapest_total_inr": prices.get(day)}

    monkeypatch.setattr(flights_router, "search_flights", fake_search)
    result = await flex_dates.scan("DEL", "GOI", "2026-08-25", adults=2, window=1)

    assert result["available"] is True
    assert result["cheapest_date"] == "2026-08-24"
    assert result["saving_inr"] == 1330.0
    assert result["worth_moving"] is True


@pytest.mark.asyncio
async def test_trivial_savings_are_not_advertised(monkeypatch):
    """Nobody rearranges a trip for two hundred rupees."""
    from app.providers import flex_dates, flights_router

    prices = {"2026-08-24": 14700.0, "2026-08-25": 14846.0}

    async def fake_search(src, dst, day, **kwargs):
        return {"cheapest_total_inr": prices.get(day)}

    monkeypatch.setattr(flights_router, "search_flights", fake_search)
    result = await flex_dates.scan("DEL", "GOI", "2026-08-25", window=1)
    assert result["worth_moving"] is False


@pytest.mark.asyncio
async def test_the_scan_never_proposes_a_date_in_the_past(monkeypatch):
    from app.providers import flex_dates, flights_router

    asked: list[str] = []

    async def fake_search(src, dst, day, **kwargs):
        asked.append(day)
        return {"cheapest_total_inr": 1000.0}

    monkeypatch.setattr(flights_router, "search_flights", fake_search)
    await flex_dates.scan("DEL", "GOI", "2026-08-25", window=3,
                          earliest="2026-08-24")
    assert min(asked) == "2026-08-24"


@pytest.mark.asyncio
async def test_a_failed_probe_does_not_fail_the_scan(monkeypatch):
    from app.providers import flex_dates, flights_router

    async def flaky(src, dst, day, **kwargs):
        if day == "2026-08-24":
            raise RuntimeError("provider down")
        return {"cheapest_total_inr": 9000.0}

    monkeypatch.setattr(flights_router, "search_flights", flaky)
    result = await flex_dates.scan("DEL", "GOI", "2026-08-25", window=1)
    assert result["available"] is True
    assert "2026-08-24" not in result["prices_by_date"]
