"""What the same trip costs on the days either side of the one you asked for.

Most travellers have a date in mind and some slack around it, and Indian
domestic fares move a long way over a couple of days — a Tuesday departure can
be a third cheaper than the Friday next to it. The planner had no way to say so:
it priced exactly the date it was given and never mentioned that moving by 48
hours halved the bill.

Design notes
------------
* Flights only. They are the volatile, expensive line and cost one keyless HTTP
  call to check; probing a week of train and bus scrapes would mean fourteen
  browser sessions to answer a "what if".
* Bounded fan-out. The whole scan runs under one budget with a semaphore, so a
  slow day cannot hold up the rest, and a scan that overruns degrades to
  "no comparison available" rather than delaying the plan.
* Every probe goes through the normal cache, so the common case of nudging a
  date around costs nothing after the first look.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from app.config import settings
from app.providers import flights_router

logger = logging.getLogger(__name__)

#: Probes run at once. Keeps a week-wide scan from opening fourteen sockets.
_CONCURRENCY = 4

#: The whole scan's budget. It is a nicety, not the plan.
SCAN_TIMEOUT = 30.0

#: Below this a "saving" is noise, not a reason to rearrange a trip.
MIN_MEANINGFUL_SAVING_INR = 500.0


def _shift(date_str: str, days: int) -> Optional[str]:
    try:
        return (
            datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
        ).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


async def scan(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    *,
    adults: int = 1,
    window: int = 3,
    earliest: Optional[str] = None,
) -> dict[str, Any]:
    """Price ``departure_date`` and the days around it.

    ``earliest`` (normally today) stops the scan proposing dates in the past,
    which would be a cheaper flight nobody can board.
    """
    dates: list[str] = []
    for offset in range(-window, window + 1):
        shifted = _shift(departure_date, offset)
        if not shifted:
            continue
        if earliest and shifted < earliest:
            continue
        dates.append(shifted)

    if departure_date not in dates:
        dates.append(departure_date)
    if len(dates) < 2:
        return {"available": False, "reason": "no alternative dates in range"}

    limiter = asyncio.Semaphore(_CONCURRENCY)

    async def probe(day: str) -> tuple[str, Optional[float]]:
        async with limiter:
            try:
                result = await flights_router.search_flights(
                    origin_iata, destination_iata, day, adults=adults,
                )
                return day, result.get("cheapest_total_inr")
            except Exception as exc:  # noqa: BLE001
                logger.info("Flexible-date probe failed for %s: %s", day, exc)
                return day, None

    try:
        pairs = await asyncio.wait_for(
            asyncio.gather(*(probe(day) for day in sorted(dates))),
            timeout=SCAN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Flexible-date scan exceeded %.0fs.", SCAN_TIMEOUT)
        return {"available": False, "reason": "scan timed out"}

    prices = {day: price for day, price in pairs if price}
    if len(prices) < 2:
        return {"available": False, "reason": "not enough priced dates to compare"}

    asked = prices.get(departure_date)
    cheapest_day = min(prices, key=lambda d: prices[d])
    cheapest_price = prices[cheapest_day]

    saving = (asked - cheapest_price) if asked else None
    worthwhile = bool(
        saving
        and saving >= MIN_MEANINGFUL_SAVING_INR
        and cheapest_day != departure_date
    )

    return {
        "available": True,
        "requested_date": departure_date,
        "requested_price_inr": asked,
        "prices_by_date": dict(sorted(prices.items())),
        "cheapest_date": cheapest_day,
        "cheapest_price_inr": cheapest_price,
        "saving_inr": round(saving, 2) if saving else None,
        "worth_moving": worthwhile,
        "advice": (
            f"Leaving on {cheapest_day} instead of {departure_date} would save "
            f"about ₹{saving:,.0f} on flights for the group."
            if worthwhile else
            f"{departure_date} is already the best-priced day in this window."
            if asked and cheapest_day == departure_date else
            "No meaningful saving from shifting these dates."
        ),
    }


def render(result: dict[str, Any]) -> str:
    if not result.get("available"):
        return ""
    lines = [f"📅 **Date flexibility** — {result['advice']}", ""]
    cheapest = result.get("cheapest_date")
    requested = result.get("requested_date")
    for day, price in (result.get("prices_by_date") or {}).items():
        marker = ""
        if day == cheapest:
            marker = "  ← cheapest"
        if day == requested:
            marker += "  (your date)"
        lines.append(f"- {day} · ₹{price:,.0f}{marker}")
    return "\n".join(lines)


def enabled() -> bool:
    return bool(settings.flexible_date_window)
