"""Caching behaviour that the scrapers depend on.

Two properties matter here and neither was covered before:

1. **A "found nothing" answer must not be cached like a real one.** The
   scrapers return a populated dict when they find nothing
   (``{"count": 0, "buses": [], "note": …}``), which is truthy and non-empty,
   so it was stored under the full TTL — pinning "no trains available" for six
   hours after a single transient failure.

2. **Stale entries refresh in the background.** Otherwise every entry expires
   under some unlucky request, which then pays the full 30-second scrape.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.cache import _is_empty, _is_stale, build_key, cache, cached


# ---------------------------------------------------------------------------
# "Nothing found" detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected,reason",
    [
        ({"count": 0, "buses": [], "note": "no route"}, True, "empty bus payload"),
        ({"count": 0, "trains": [], "note": "none"}, True, "empty train payload"),
        ({"flights": [], "origin": "DEL"}, True, "empty flight list with metadata"),
        ({"error": "upstream exploded"}, True, "error payload"),
        ({}, True, "empty dict"),
        ([], True, "empty list"),
        ("", True, "empty string"),
        (None, True, "none"),
        ({"count": 10, "trains": [{"no": "19702"}]}, False, "real results"),
        ({"buses": [{"fare_inr": 240}], "count": 1}, False, "one real bus"),
    ],
)
def test_is_empty_understands_provider_payloads(value, expected, reason):
    assert _is_empty(value) is expected, reason


def test_metadata_only_payload_is_not_mistaken_for_results():
    """A dict with no result keys at all should not be called empty."""
    assert _is_empty({"note": "informational", "source": "cache"}) is False


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_entry_is_fresh_before_the_refresh_point():
    import time

    entry = {"v": 1, "t": time.time()}
    assert _is_stale(entry, ttl=100, refresh_ratio=0.75) is False


def test_entry_is_stale_past_the_refresh_point():
    import time

    entry = {"v": 1, "t": time.time() - 80}
    assert _is_stale(entry, ttl=100, refresh_ratio=0.75) is True


def test_unstamped_legacy_entry_is_treated_as_stale():
    """Entries written before stamping existed must refresh once."""
    assert _is_stale({"v": 1}, ttl=100, refresh_ratio=0.75) is True


def test_refresh_can_be_disabled():
    assert _is_stale({"v": 1, "t": 0}, ttl=100, refresh_ratio=0) is False


# ---------------------------------------------------------------------------
# Stale-while-revalidate end to end
# ---------------------------------------------------------------------------


async def test_stale_entry_is_served_immediately_and_refreshed_behind_it():
    await cache.connect()
    await cache.invalidate_prefix("tp:v1:swr_unit")

    calls = {"n": 0}

    @cached("swr_unit", ttl=4, refresh_ratio=0.25)  # stale after 1s
    async def fetch(city: str):
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return {"count": 1, "results": [f"v{calls['n']}"]}

    first = await fetch("delhi")
    assert first["results"] == ["v1"] and calls["n"] == 1

    # Still fresh: no refresh triggered.
    await fetch("delhi")
    assert calls["n"] == 1

    await asyncio.sleep(1.2)  # now past ttl * refresh_ratio

    stale = await fetch("delhi")
    assert stale["results"] == ["v1"], "the cached value must be served, not awaited"

    await asyncio.sleep(0.8)  # let the background refresh land
    assert calls["n"] == 2, "a background refresh should have run"

    refreshed = await fetch("delhi")
    assert refreshed["results"] == ["v2"], "the refreshed value should now be served"


async def test_empty_result_is_stored_under_the_negative_ttl():
    """A zero-result scrape must not occupy the cache for the full TTL.

    Asserted on the TTL actually written to Redis rather than by sleeping past
    it: a one-second expiry raced against wall-clock sleeps makes for a test
    that passes alone and fails under load, which is worse than no test.
    """
    await cache.connect()
    await cache.invalidate_prefix("tp:v1:neg_unit")

    calls = {"n": 0}

    @cached("neg_unit", ttl=3600, negative_ttl=120, refresh_ratio=0)
    async def scrape(route: str):
        calls["n"] += 1
        # A *transient* empty: zero results and no explanation, which is what a
        # broken selector or a blipped scrape looks like. Note the deliberate
        # absence of a "note" — with one this would count as a settled "no
        # service" answer and earn the much longer retry window instead.
        return {"count": 0, "buses": []}

    await scrape("mumbai-pune")
    await scrape("mumbai-pune")
    assert calls["n"] == 1, "the second call should have been served from cache"

    key = build_key("neg_unit", "mumbai-pune")
    ttl = await _ttl_of(key)
    assert ttl is not None, "the empty result should still have been cached"
    assert ttl <= 120, f"expected the 120s negative TTL, got {ttl}s"


async def _ttl_of(key: str):
    """Remaining TTL for a key, whichever cache backend is in play."""
    client = getattr(cache, "_redis", None)
    if client is not None:
        ttl = await client.ttl(key)
        return None if ttl is None or ttl < 0 else ttl

    # In-process LRU fallback: entries are (expires_at, payload).
    item = cache._local._data.get(key)  # type: ignore[attr-defined]
    if item is None:
        return None
    import time as _time

    return max(0, int(item[0] - _time.time()))


# ---------------------------------------------------------------------------
# Settled vs transient emptiness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,settled,reason",
    [
        ({"count": 0, "buses": [], "note": "No buses on this route."}, True,
         "provider explicitly answered 'none'"),
        ({"count": 0, "trains": [], "note": "no service"}, True, "same for trains"),
        ({"count": 0, "buses": []}, False, "zero but no explanation — likely a failure"),
        ({"error": "timeout", "count": 0, "note": "x"}, False, "errors are transient"),
        ([], False, "a bare empty list says nothing about why"),
        ({"count": 5, "buses": [1]}, False, "not empty at all"),
    ],
)
def test_settled_empty_detection(value, settled, reason):
    from app.core.cache import _is_settled_empty

    assert _is_settled_empty(value) is settled, reason


def test_settled_empty_gets_the_long_retry_window():
    """A definitive 'no service' must not re-scrape every 5 minutes.

    AbhiBus takes ~30s to answer, and bus operators only publish schedules
    about a month ahead — so a far-future search is reliably empty and
    re-asking constantly is pure waste.
    """
    from app.core.cache import _SETTLED_EMPTY_TTL, _effective_ttl

    settled = {"count": 0, "buses": [], "note": "No buses found for this route."}
    assert _effective_ttl(settled, ttl=1800, negative_ttl=300) == _SETTLED_EMPTY_TTL

    transient = {"error": "scrape failed"}
    assert _effective_ttl(transient, ttl=1800, negative_ttl=300) == 300

    real = {"count": 3, "buses": [1, 2, 3]}
    assert _effective_ttl(real, ttl=1800, negative_ttl=300) == 1800
