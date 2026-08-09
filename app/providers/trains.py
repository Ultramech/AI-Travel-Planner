"""Train search by scraping etrain.info.

Fixes versus the original ``train_data.py``:
  * **The driver was never quit** — every call leaked a Chrome process.
  * The 414 KB station spreadsheet was loaded at *module import* time, so
    importing the module cost seconds and a missing file was an import error.
    It is now lazily loaded and cached in ``providers.geo``.
  * The filename case (``Station_name_...`` vs ``Station_Name_...``) broke on
    Linux.
  * Only ``src.iloc[0]`` was ever tried; if the top station had no direct
    service the answer was "no trains" even when a sibling terminal in the same
    city had plenty. We now try the top few station pairs.
  * The date was chosen by clicking through a JS calendar up to 12 times with
    ``sleep(0.5)`` between clicks. etrain accepts the date in the URL, so we
    try that first and only fall back to clicking.
  * Runs off the event loop in a worker thread so it cannot block the API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient
from app.providers.geo import lookup_stations, station_slug
from app.scraping.driver import ScraperUnavailable, browser, find_all_first

logger = logging.getLogger(__name__)

BASE = "https://etrain.info/trains"

# Indian train numbers are 4-5 digits. Used to reject header/day-strip rows.
_TRAIN_NO_RE = re.compile(r"\d{4,5}")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _row_selectors() -> list[tuple[str, str]]:
    from selenium.webdriver.common.by import By

    return [
        (By.XPATH, '//div[contains(@class,"trnlstcont")]//table//tbody//tr'),
        (By.CSS_SELECTOR, "div.trnlstcont table tbody tr"),
        (By.CSS_SELECTOR, "table.trainlist tbody tr"),
        (By.XPATH, '//table[contains(@class,"tbl")]//tbody//tr'),
    ]


def _scrape_pair(
    src: dict[str, str], dst: dict[str, str], date_obj: datetime
) -> list[dict[str, Any]]:
    """Scrape one station pair. Runs in a worker thread."""
    from selenium.webdriver.common.by import By

    url = (
        f"{BASE}/{station_slug(src['name'], src['code'])}"
        f"-to-{station_slug(dst['name'], dst['code'])}"
        f"?date={date_obj.strftime('%d-%m-%Y')}"
    )
    logger.info("etrain scrape: %s", url)

    trains: list[dict[str, Any]] = []
    with browser() as driver:
        driver.get(url)

        # Give the train table a chance to render; no blind sleeps.
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//table//tbody//tr | //div[contains(@class,"trnlst")]')
                )
            )
        except Exception:  # noqa: BLE001
            logger.info("No train table rendered for %s", url)
            return []

        rows = find_all_first(driver, _row_selectors())
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 6:
                    continue
                values = [(c.text or "").strip() for c in cells[:7]]
                if not values[0] or not any(values[1:]):
                    continue
                # etrain renders its column headings and a day-of-week strip
                # ("Su Mo Tu We Th Fr Sa") as ordinary <td> rows inside tbody,
                # so a naive scrape returns them as if they were trains.
                # Indian train numbers are 4-5 digits; that is the reliable
                # discriminator.
                if not _TRAIN_NO_RE.fullmatch(values[0].replace("\n", " ").strip()):
                    continue
                if not _TIME_RE.search(values[3]):
                    continue
                trains.append(
                    {
                        "train_no": values[0],
                        "train_name": values[1],
                        "from": values[2],
                        "departs": values[3],
                        "to": values[4],
                        "arrives": values[5],
                        "duration": values[6] if len(values) > 6 else None,
                        "source_station": f"{src['name']} ({src['code']})",
                        "dest_station": f"{dst['name']} ({dst['code']})",
                        "booking_link": url,
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    return trains


@cached("trains", ttl=settings.ttl_trains)
@resilient("etrain", timeout=90.0, retries=1, recovery_seconds=180)
async def search_trains(
    origin_city: str, destination_city: str, travel_date: str, limit: int = 10
) -> dict[str, Any]:
    """Find direct trains. ``travel_date`` is ISO ``YYYY-MM-DD``."""
    try:
        date_obj = datetime.strptime(travel_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ProviderError("etrain", f"bad date {travel_date!r}, expected YYYY-MM-DD") from exc

    sources = lookup_stations(origin_city, limit=2)
    destinations = lookup_stations(destination_city, limit=2)

    if not sources or not destinations:
        missing = origin_city if not sources else destination_city
        return {
            "origin": origin_city,
            "destination": destination_city,
            "date": travel_date,
            "count": 0,
            "trains": [],
            "note": f"No railway station found in the dataset for {missing!r}.",
        }

    attempted: list[str] = []
    # Try the most important station pairs, best first, and stop at first hit.
    for src in sources:
        for dst in destinations:
            attempted.append(f"{src['code']}->{dst['code']}")
            try:
                trains = await asyncio.to_thread(_scrape_pair, src, dst, date_obj)
            except ScraperUnavailable as exc:
                raise ProviderError("etrain", str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                logger.warning("etrain pair %s failed: %s", attempted[-1], exc)
                continue
            if trains:
                return {
                    "origin": origin_city,
                    "destination": destination_city,
                    "date": travel_date,
                    "count": len(trains[:limit]),
                    "trains": trains[:limit],
                    "station_pair": attempted[-1],
                }

    return {
        "origin": origin_city,
        "destination": destination_city,
        "date": travel_date,
        "count": 0,
        "trains": [],
        "note": f"No direct trains found (tried {', '.join(attempted)}).",
    }


def render(result: dict[str, Any]) -> str:
    trains = result.get("trains") or []
    if not trains:
        return f"🚆 {result.get('note', 'No direct trains found.')}"
    lines = [
        f"🚆 **{len(trains)} direct train(s)** {result['origin']} → "
        f"{result['destination']} on {result['date']}\n"
    ]
    for t in trains:
        lines.append(
            f"🔹 **{t['train_no']} {t['train_name']}** · "
            f"{t['from']} {t['departs']} → {t['to']} {t['arrives']}"
            + (f" · {t['duration']}" if t.get("duration") else "")
        )
    if trains and trains[0].get("booking_link"):
        lines.append(f"\n🔗 {trains[0]['booking_link']}")
    return "\n".join(lines)
