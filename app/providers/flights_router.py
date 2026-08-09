"""Flight-source selection with automatic failover.

Two backends, same contract:

  ``google_flights``  keyless, real display prices, wide inventory. Default.
  ``amadeus``         bookable offers with fare rules — better data, but needs
                      credentials, and its free *test* tier has thin inventory.

Policy:
  * ``FLIGHT_PROVIDER=auto`` (default) — prefer Amadeus when credentials exist
    **and** its circuit is closed; otherwise Google Flights. If the preferred
    source fails or returns nothing, fall back to the other one rather than
    telling the user "no flights".
  * ``FLIGHT_PROVIDER=google`` / ``amadeus`` — force one.

The point is that a missing API key degrades the *quality* of the flight data,
never its availability.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import settings
from app.core.resilience import ProviderError, get_breaker
from app.providers import flights as amadeus_provider
from app.providers import flights_google as google_provider

logger = logging.getLogger(__name__)

GOOGLE = "google"
AMADEUS = "amadeus"


def amadeus_available() -> bool:
    return bool(settings.amadeus_client_id and settings.amadeus_client_secret)


def _order() -> list[str]:
    """Sources to try, best first."""
    configured = (settings.flight_provider or "auto").strip().lower()

    if configured == GOOGLE:
        return [GOOGLE]
    if configured == AMADEUS:
        return [AMADEUS]

    # auto
    if amadeus_available() and get_breaker("amadeus_flights").state == "closed":
        return [AMADEUS, GOOGLE]
    return [GOOGLE, AMADEUS] if amadeus_available() else [GOOGLE]


async def search_flights(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    currency: str = "INR",
    max_results: int = 8,
    skip_cache: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search flights using the best available source, with failover."""
    sources = _order()
    errors: list[str] = []

    for name in sources:
        provider = google_provider if name == GOOGLE else amadeus_provider
        try:
            result = await provider.search_flights(
                origin_iata,
                destination_iata,
                departure_date,
                return_date=return_date,
                adults=adults,
                currency=currency,
                max_results=max_results,
                skip_cache=skip_cache,
            )
        except ProviderError as exc:
            logger.warning("Flight source %s failed: %s", name, exc)
            errors.append(f"{name}: {exc.message}")
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Flight source %s errored: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue

        if result.get("flights"):
            result.setdefault("source", name)
            result["sources_tried"] = sources[: sources.index(name) + 1]
            if errors:
                result["fallback_from"] = errors
            return result

        errors.append(f"{name}: no flights returned")

    return {
        "origin_iata": (origin_iata or "").upper(),
        "destination_iata": (destination_iata or "").upper(),
        "departure_date": departure_date,
        "return_date": return_date,
        "adults": adults,
        "count": 0,
        "flights": [],
        "cheapest_inr": None,
        "sources_tried": sources,
        "note": "No flights found. " + "; ".join(errors),
    }


def render(result: dict[str, Any]) -> str:
    if result.get("source") == AMADEUS:
        return amadeus_provider.render(result)
    return google_provider.render(result)
