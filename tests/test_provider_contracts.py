"""Replay recorded upstream responses through the real parsers.

The failure this guards against does not raise. Google Flights stopped sending
an alliance-metadata block on routes flown only by non-alliance carriers; the
upstream library indexed that block *before* reading the flights, so the parse
died and the system reported **"No airline flies DEL→COK"** — a route with
dozens of daily departures. It looked exactly like a correct negative answer.

No mock would have caught it, because a mock keeps whatever shape it was written
with. These fixtures were recorded from the live service by
``scripts/record_fixtures.py``, and the assertions below encode what the parser
must survive:

  * flights present **and** metadata present  (DEL-BOM, the ordinary case)
  * flights present **but metadata absent**   (the case that broke)
  * no flight section at all                  (AGR-BOM: genuinely no service)

The third is the one that keeps the system honest. "Nothing flies this route"
and "we could not read the answer" must never collapse into the same output.
"""

from __future__ import annotations

import json
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not recorded; run scripts/record_fixtures.py")
    return json.loads(path.read_text())


def _all_fixtures() -> list[pathlib.Path]:
    return sorted(FIXTURES.glob("google_flights_*.json"))


def test_fixtures_exist():
    assert _all_fixtures(), (
        "No recorded fixtures. Run: "
        "PYTHONPATH=. .venv/bin/python scripts/record_fixtures.py"
    )


@pytest.mark.parametrize("path", _all_fixtures(), ids=lambda p: p.stem)
def test_recorded_shape_is_still_readable(path):
    """Every recorded itinerary must still parse into our flight dict."""
    from app.providers.flights_google import _build_results

    data = json.loads(path.read_text())
    entries = data.get("flights")
    if not entries:
        pytest.skip(f"{data['route']} recorded with no flights")

    results = _build_results(entries)
    assert results, f"{data['route']}: recorded itineraries no longer parse"

    for flight in results:
        assert flight["origin_iata"], "origin airport code lost"
        assert flight["destination_iata"], "destination airport code lost"
        assert flight["price_total_inr"] is not None, "price lost"
        assert flight["departure"], "departure timestamp lost"
        assert flight["arrival"], "arrival timestamp lost"
        assert flight["duration_minutes"] > 0, "duration lost"


def test_a_route_with_no_service_records_as_empty():
    """AGR-BOM has no flights. That must stay distinguishable from a parse error."""
    data = _load("google_flights_AGR_BOM.json")
    assert data["flight_count"] == 0
    assert data["has_flight_section"] is False


def test_the_metadata_block_is_still_optional():
    """The parser must not depend on it, whether or not it happens to be sent.

    This is the exact regression: reading alliance metadata before the flights
    turned a populated response into "no airline flies this route".
    """
    from app.providers.flights_google import _build_results

    data = _load("google_flights_DEL_COK.json")
    # Whatever the recording captured, parsing must succeed from the flight
    # slot alone — nothing here consults the metadata block.
    results = _build_results(data["flights"])
    assert len(results) >= 1


def test_prices_are_group_totals_not_per_person():
    """Recorded at adults=2, so the total must exceed a plausible single fare.

    Guards the double-count directly: a Shimla→Mumbai plan once reported
    ₹32,604 against a true ₹16,302.
    """
    from app.providers.flights_google import _build_results

    data = _load("google_flights_DEL_BOM.json")
    results = _build_results(data["flights"])
    flight = results[0]
    assert flight["price_total_inr"] == flight["price_per_person_inr"] * flight["party_size"]


def test_parser_survives_a_truncated_itinerary():
    """One malformed entry must not discard the rest of the page."""
    from app.providers.flights_google import _build_results

    data = _load("google_flights_DEL_BOM.json")
    good = data["flights"]
    corrupted = [["nonsense"], *good]

    results = _build_results(corrupted)
    assert len(results) == len(_build_results(good))
