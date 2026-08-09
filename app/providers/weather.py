"""WeatherAPI forecast provider.

Fixes versus the original:
  * One API call for the whole trip instead of one call **per day**. The old
    loop issued up to 14 requests to fetch data a single ``days=N`` request
    already returns — 14x the cost and latency for identical data.
  * Dates beyond the plan's forecast horizon are reported as such rather than
    silently falling back to ``history.json`` (a paid endpoint that returns an
    error on the free tier, which the old code then rendered as the whole
    forecast).
  * Async + timeout + retry + Redis cache, and it returns structured data
    instead of writing Streamlit widgets from inside a graph node.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient

logger = logging.getLogger(__name__)

BASE_URL = "https://api.weatherapi.com/v1/forecast.json"
MAX_FORECAST_DAYS = 14


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


@cached("weather", ttl=settings.ttl_weather)
@resilient("weatherapi", timeout=20.0, retries=2)
async def get_forecast(city: str, start_date: str, days: int) -> list[dict[str, Any]]:
    """Return one structured record per trip day."""
    settings.require("weather_api_key")

    start = _parse_date(start_date)
    today = date.today()
    days = max(1, min(int(days or 1), MAX_FORECAST_DAYS))

    # WeatherAPI counts forecast days from *today*, so to cover a trip starting
    # later we must request through the last trip day and then slice.
    span = (start - today).days + days
    if span <= 0:
        logger.info("Trip for %s is entirely in the past; no forecast available.", city)
        return []
    requested = max(1, min(span, MAX_FORECAST_DAYS))

    params = {
        "key": settings.weather_api_key,
        "q": city,
        "days": requested,
        "aqi": "no",
        "alerts": "no",
    }

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        response = await client.get(BASE_URL, params=params)

    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:  # noqa: BLE001
            detail = response.text[:200]
        raise ProviderError("weatherapi", f"HTTP {response.status_code}: {detail}")

    data = response.json()
    if "error" in data:
        raise ProviderError("weatherapi", data["error"].get("message", "unknown error"))

    by_date = {
        entry["date"]: entry
        for entry in data.get("forecast", {}).get("forecastday", [])
    }

    out: list[dict[str, Any]] = []
    for offset in range(days):
        target = start + timedelta(days=offset)
        key = target.strftime("%Y-%m-%d")
        entry = by_date.get(key)

        if entry is None:
            out.append(
                {
                    "date": key,
                    "day_name": target.strftime("%A"),
                    "available": False,
                    "condition": "Beyond forecast horizon",
                    "note": "WeatherAPI provides up to 14 days ahead.",
                }
            )
            continue

        day = entry.get("day", {})
        max_t = day.get("maxtemp_c")
        min_t = day.get("mintemp_c")
        avg_t = day.get("avgtemp_c")
        if avg_t is None and max_t is not None and min_t is not None:
            avg_t = round((max_t + min_t) / 2, 1)

        out.append(
            {
                "date": key,
                "day_name": target.strftime("%A"),
                "available": True,
                "condition": day.get("condition", {}).get("text", "Unknown"),
                "max_temp": max_t,
                "min_temp": min_t,
                "avg_temp": avg_t,
                "rain_chance": day.get("daily_chance_of_rain", 0),
                "snow_chance": day.get("daily_chance_of_snow", 0),
                "humidity": day.get("avghumidity", 0),
                "wind_kph": day.get("maxwind_kph"),
                "sunrise": entry.get("astro", {}).get("sunrise"),
                "sunset": entry.get("astro", {}).get("sunset"),
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
