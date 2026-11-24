"""Open-Meteo geocoding: the destination must resolve to the place travellers mean."""

from __future__ import annotations

import pytest

from app.providers import weather


def _r(name, admin1, country="IN", population=None, feature="PPL", lat=0.0, lon=0.0):
    return {"name": name, "admin1": admin1, "country_code": country,
            "population": population, "feature_code": feature,
            "latitude": lat, "longitude": lon}


@pytest.fixture
def fake_search(monkeypatch):
    """Route _search to canned results keyed by (query, country)."""
    table: dict[tuple[str, str | None], list[dict]] = {}
    calls: list[tuple[str, str | None]] = []

    async def search(client, query, country):
        calls.append((query, country))
        return table.get((query, country), [])

    monkeypatch.setattr(weather, "_search", search)
    return table, calls


@pytest.mark.asyncio
async def test_known_places_skip_the_geocoder(fake_search):
    _, calls = fake_search
    for name, label in [("Goa", "Panaji, Goa"), ("Manali", "Manali, Himachal Pradesh"),
                        ("Kerala", "Kochi, Kerala"), ("Ranthambore", "Sawai Madhopur (Ranthambore), Rajasthan")]:
        place = await weather._geocode(None, name)
        assert place["label"] == label
    assert calls == []


@pytest.mark.asyncio
async def test_population_beats_administrative_rank(fake_search):
    table, _ = fake_search
    table[("Udaipur", "IN")] = [
        _r("Udaipur", "Tripura", population=32_758, feature="PPLA2"),
        _r("Udaipur", "Rajasthan", population=451_100, feature="PPL"),
    ]
    place = await weather._geocode(None, "Udaipur")
    assert place["label"] == "Udaipur, Rajasthan"


@pytest.mark.asyncio
async def test_colloquial_names_are_searched_by_canonical_name(fake_search):
    table, calls = fake_search
    table[("Bengaluru", "IN")] = [_r("Bengaluru", "Karnataka", population=8_495_492, feature="PPLA")]
    place = await weather._geocode(None, "Bangalore")
    assert place["label"] == "Bengaluru, Karnataka"
    assert calls[0] == ("Bengaluru", "IN")


@pytest.mark.asyncio
async def test_indian_hamlet_does_not_shadow_a_foreign_city(fake_search):
    table, _ = fake_search
    table[("Dubai", "IN")] = [_r("Dubai", "Uttar Pradesh")]
    table[("Dubai", None)] = [_r("Dubai", "Dubai", country="AE", population=1_137_347, feature="PPLA"),
                              _r("Dubai", "Uttar Pradesh")]
    place = await weather._geocode(None, "Dubai")
    assert place["label"] == "Dubai, Dubai"


@pytest.mark.asyncio
async def test_airports_are_never_a_fallback(fake_search):
    table, _ = fake_search
    table[("Nowhere", "IN")] = [_r("Nowhere Airport", "Tamil Nadu", feature="AIRP")]
    with pytest.raises(Exception, match="could not locate"):
        await weather._geocode(None, "Nowhere")
