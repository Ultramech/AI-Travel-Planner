"""Weather forecast provider — Open-Meteo.

Open-Meteo needs no API key and forecasts 16 days ahead. It replaced
WeatherAPI, whose free plan only returns 3 days (today + 2) even when 14 are
requested, so any trip starting later than that silently got no weather.

  * Two calls per city: geocode the name, then one forecast request covering
    the whole horizon, sliced to the trip days.
  * Days beyond the horizon are reported as such, never invented.
  * Async + timeout + retry + Redis cache; returns structured data in the same
    shape as before, so the graph and itinerary code are unchanged.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient
from app.providers.geo import normalise_city

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_FORECAST_DAYS = 16

_DAILY_FIELDS = ",".join([
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_probability_max",
    "snowfall_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "sunrise",
    "sunset",
])

# WMO weather interpretation codes, as documented by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _clock(iso: Optional[str]) -> Optional[str]:
    """'2026-11-24T06:53' -> '06:53 AM'."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).strftime("%I:%M %p")
    except ValueError:
        return iso


# Destinations a name search gets wrong: states/regions that are not towns
# ("Goa" -> Genoa, Italy; "Kerala" -> a village in Finland) and famous places
# outranked by a bigger namesake ("Manali" -> a Chennai suburb). Coordinates
# are the place travellers actually mean (for a state: its main tourist hub).
_KNOWN_PLACES: dict[str, tuple[float, float, str]] = {
    "goa":          (15.4909, 73.8278, "Panaji, Goa"),
    "north goa":    (15.5439, 73.7553, "Calangute, Goa"),
    "south goa":    (15.2832, 73.9862, "Margao, Goa"),
    "kerala":       (9.9312, 76.2673, "Kochi, Kerala"),
    "manali":       (32.2432, 77.1892, "Manali, Himachal Pradesh"),
    "kasol":        (32.0100, 77.3150, "Kasol, Himachal Pradesh"),
    "spiti":        (32.2276, 78.0710, "Kaza, Himachal Pradesh"),
    "himachal":     (31.1048, 77.1734, "Shimla, Himachal Pradesh"),
    "himachal pradesh": (31.1048, 77.1734, "Shimla, Himachal Pradesh"),
    "coorg":        (12.4244, 75.7382, "Madikeri, Karnataka"),
    "leh":          (34.1526, 77.5771, "Leh, Ladakh"),
    "ladakh":       (34.1526, 77.5771, "Leh, Ladakh"),
    "kashmir":      (34.0837, 74.7973, "Srinagar, Jammu and Kashmir"),
    "andaman":      (11.6234, 92.7265, "Port Blair, Andaman and Nicobar"),
    "andamans":     (11.6234, 92.7265, "Port Blair, Andaman and Nicobar"),
    "andaman and nicobar": (11.6234, 92.7265, "Port Blair, Andaman and Nicobar"),
    "lakshadweep":  (10.5669, 72.6420, "Kavaratti, Lakshadweep"),
    "sikkim":       (27.3389, 88.6065, "Gangtok, Sikkim"),
    "meghalaya":    (25.5788, 91.8933, "Shillong, Meghalaya"),
    "uttarakhand":  (30.3165, 78.0322, "Dehradun, Uttarakhand"),
    "rajasthan":    (26.9124, 75.7873, "Jaipur, Rajasthan"),
    "munnar":       (10.0889, 77.0595, "Munnar, Kerala"),
    "alleppey":     (9.4981, 76.3388, "Alappuzha, Kerala"),
    "varkala":      (8.7379, 76.7163, "Varkala, Kerala"),
    "ooty":         (11.4102, 76.6950, "Ooty, Tamil Nadu"),
    "hampi":        (15.3350, 76.4600, "Hampi, Karnataka"),
    "gokarna":      (14.5479, 74.3188, "Gokarna, Karnataka"),
    "auli":         (30.5280, 79.5660, "Auli, Uttarakhand"),
    # National parks: not towns, so a name search finds nothing.
    "ranthambore":  (26.0173, 76.3532, "Sawai Madhopur (Ranthambore), Rajasthan"),
    "jim corbett":  (29.5300, 78.7747, "Ramnagar (Jim Corbett), Uttarakhand"),
    "corbett":      (29.5300, 78.7747, "Ramnagar (Jim Corbett), Uttarakhand"),
    "kaziranga":    (26.5775, 93.1711, "Kaziranga, Assam"),
    "sundarbans":   (22.1500, 88.8000, "Sundarbans, West Bengal"),
    # Islands/countries people name as a destination.
    "bali":         (-8.5069, 115.2625, "Ubud, Bali"),
    "maldives":     (4.1755, 73.5093, "Male, Maldives"),
}

# Indian matches smaller than this are checked against same-named places abroad.
_SMALL_TOWN = 100_000

# Lower is better: national capital, state capitals, district seats, towns.
_FEATURE_RANK = {"PPLC": 0, "PPLA": 1, "PPLA2": 2, "PPLA3": 3, "PPLA4": 4, "PPL": 5}


def _rank(results: list[dict[str, Any]], wanted: str) -> Optional[dict[str, Any]]:
    """Exact name matches only; the most populous wins, administrative rank breaks ties.

    Population first: Udaipur, Rajasthan (pop. 451k, plain PPL) must beat
    Udaipur, Tripura (32k, but a district seat). Famous small places that a
    bigger namesake outranks (Manali) are pinned in ``_KNOWN_PLACES`` instead.
    """
    exact = [r for r in results if normalise_city(r.get("name", "")) == wanted]
    if not exact:
        return None
    return min(
        exact,
        key=lambda r: (-(r.get("population") or 0),
                       _FEATURE_RANK.get(r.get("feature_code", ""), 9)),
    )


async def _search(client: httpx.AsyncClient, city: str, country: Optional[str]) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"name": city, "count": 20, "language": "en", "format": "json"}
    if country:
        params["countryCode"] = country
    response = await client.get(GEOCODE_URL, params=params)
    if response.status_code != 200:
        raise ProviderError("open_meteo", f"geocoding HTTP {response.status_code}")
    return response.json().get("results") or []


async def _geocode(client: httpx.AsyncClient, city: str) -> dict[str, Any]:
    """Resolve a destination to coordinates, India first (this planner's market)."""
    wanted = normalise_city(city)
    if wanted in _KNOWN_PLACES:
        lat, lon, label = _KNOWN_PLACES[wanted]
        return {"latitude": lat, "longitude": lon, "label": label}

    # Search by the canonical name: the geocoder knows "Bengaluru" and
    # "Puducherry", but not "Bangalore" or "Pondicherry".
    query = wanted.title()
    indian = await _search(client, query, "IN")
    place = _rank(indian, wanted)
    if place is None or (place.get("population") or 0) < _SMALL_TOWN:
        # A village-sized Indian match must not shadow a real city abroad
        # ("Dubai" is also a hamlet in Uttar Pradesh).
        abroad = _rank(await _search(client, query, None), wanted)
        if abroad is not None and (
            place is None or (abroad.get("population") or 0) > (place.get("population") or 0)
        ):
            place = abroad
    if place is None:
        # No exact spelling match anywhere: an Indian populated place with a
        # near-match name beats a foreign one (airports etc. are skipped).
        towns = [r for r in indian if str(r.get("feature_code", "")).startswith("PPL")]
        if towns:
            place = max(towns, key=lambda r: r.get("population") or 0)
    if place is None:
        raise ProviderError("open_meteo", f"could not locate {city!r}")
    place["label"] = ", ".join(
        p for p in (place.get("name"), place.get("admin1") or place.get("country")) if p
    )
    return place


@cached("weather", ttl=settings.ttl_weather)
@resilient("open_meteo", timeout=20.0, retries=2)
async def get_forecast(city: str, start_date: str, days: int) -> list[dict[str, Any]]:
    """Return one structured record per trip day."""
    start = _parse_date(start_date)
    days = max(1, int(days or 1))
    horizon_end = date.today() + timedelta(days=MAX_FORECAST_DAYS - 1)

    if start + timedelta(days=days - 1) < date.today():
        logger.info("Trip for %s is entirely in the past; no forecast available.", city)
        return []

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        place = await _geocode(client, city)
        response = await client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": _DAILY_FIELDS,
                "timezone": "auto",
                "forecast_days": MAX_FORECAST_DAYS,
            },
        )

    data = response.json() if response.content else {}
    if response.status_code != 200 or data.get("error"):
        raise ProviderError(
            "open_meteo", f"HTTP {response.status_code}: {data.get('reason', response.text[:200])}"
        )

    daily = data.get("daily") or {}
    times = daily.get("time") or []
    index = {t: i for i, t in enumerate(times)}

    def field(name: str, i: int) -> Any:
        values = daily.get(name) or []
        return values[i] if i < len(values) else None

    out: list[dict[str, Any]] = []
    for offset in range(days):
        target = start + timedelta(days=offset)
        key = target.strftime("%Y-%m-%d")
        i = index.get(key)

        if i is None:
            out.append(
                {
                    "date": key,
                    "day_name": target.strftime("%A"),
                    "available": False,
                    "condition": "Beyond forecast horizon",
                    "note": (
                        f"Open-Meteo forecasts {MAX_FORECAST_DAYS} days ahead "
                        f"(through {horizon_end.isoformat()})."
                    ),
                }
            )
            continue

        max_t = field("temperature_2m_max", i)
        min_t = field("temperature_2m_min", i)
        avg_t = field("temperature_2m_mean", i)
        if avg_t is None and max_t is not None and min_t is not None:
            avg_t = round((max_t + min_t) / 2, 1)
        snowfall = field("snowfall_sum", i) or 0

        out.append(
            {
                "date": key,
                "day_name": target.strftime("%A"),
                "available": True,
                "condition": WMO_CODES.get(field("weather_code", i), "Unknown"),
                "max_temp": max_t,
                "min_temp": min_t,
                "avg_temp": avg_t,
                "rain_chance": field("precipitation_probability_max", i) or 0,
                # Open-Meteo gives snowfall amount, not a probability.
                "snow_chance": 100 if snowfall > 0 else 0,
                "humidity": field("relative_humidity_2m_mean", i) or 0,
                "wind_kph": field("wind_speed_10m_max", i),
                "sunrise": _clock(field("sunrise", i)),
                "sunset": _clock(field("sunset", i)),
                "location": place.get("label"),
            }
        )

    return out


def summarise(forecast: list[dict[str, Any]]) -> str:
    """Compact text form for LLM prompts."""
    if not forecast:
        return "No weather data available."
    lines = []
    for day in forecast:
        if not day.get("available", True):
            lines.append(f"{day['day_name']} ({day['date']}): forecast not available")
            continue
        lines.append(
            f"{day['day_name']} ({day['date']}): {day['condition']}, "
            f"{day.get('min_temp')}-{day.get('max_temp')}°C, "
            f"rain {day.get('rain_chance', 0)}%, humidity {day.get('humidity', 0)}%"
        )
    return "\n".join(lines)
