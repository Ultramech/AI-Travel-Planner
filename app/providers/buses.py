"""Bus search by scraping AbhiBus.

Fixes versus the original ``bus_data.py``:
  * It drove the homepage autocomplete on *every* search, pressed DOWN+ENTER
    blindly, then rewrote the resulting URL by string-splitting it
    (``url.split("/")[-2]``). Two problems: DOWN+ENTER selects whatever the
    dropdown shows first (verified live: asking for Delhi->Jaipur searched
    Hyderabad->Bengaluru), and the URL surgery broke on any redirect change.
    Search URLs embed permanent numeric city ids, so we drive the form **once**
    per route, click the option that actually matches the typed city, cache the
    ids, and build the URL directly from then on.
  * Fares: a card shows "Save Rs40", a struck "Rs400" and the payable "Rs360".
    Taking the minimum returned 40 -- the discount -- as the fare.
  * Operator names were read from the first line of the card, which is often a
    promo banner ("Flash Offer : Rs36 off per seat!").
  * Departure/arrival times were derived by slicing one flat list of ``span``
    texts with ``times[::2]`` / ``times[1::2]``. If any row rendered a third
    span (common — "Rest Stop", "Live Tracking" badges) every subsequent row's
    times were shifted by one and silently wrong. We now parse per bus card.
  * The seat-availability XPath pinned six exact utility classes.
  * Non-headless, and no exception handling at all — only ``finally``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from app.config import settings
from app.core.cache import cached
from app.core.resilience import ProviderError, resilient
from app.providers.geo import normalise_city
from app.scraping.driver import ScraperUnavailable, browser, find_all_first

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_FARE_RE = re.compile(r"(?:₹|Rs\.?|INR)\s*([\d,]+)", re.IGNORECASE)


def _slug(city: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalise_city(city)).strip("-")


# AbhiBus search URLs embed numeric city ids:
#   /bus_search/Delhi/344/Jaipur/241/08-08-2026/O
# You cannot construct one from city names alone — which is why the original
# code had to drive the homepage autocomplete on *every* search. We drive the
# form once per city pair, extract the ids, and cache them: the ids never
# change, so every later search for that pair is a direct URL fetch.
_SEARCH_URL_RE = re.compile(
    r"/bus_search/(?P<from_name>[^/]+)/(?P<from_id>\d+)/"
    r"(?P<to_name>[^/]+)/(?P<to_id>\d+)/",
    re.IGNORECASE,
)


def _discover_route(origin: str, destination: str) -> Optional[dict[str, str]]:
    """Drive the homepage form once to learn a route's city ids."""
    from selenium.webdriver.common.by import By

    with browser() as driver:
        driver.get("https://www.abhibus.com")

        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//input[@placeholder="Leaving From"]')
                )
            )
        except Exception:  # noqa: BLE001
            return None

        def fill(placeholder: str, value: str) -> bool:
            """Type a city and click the option that actually matches it.

            Blindly sending DOWN+ENTER (what the original code did) selects
            whatever the dropdown happens to show first — including the
            'popular cities' list rendered before any keystrokes are processed.
            That silently searched a completely different route: asking for
            Delhi→Jaipur returned Hyderabad→Bengaluru. We wait for an option
            whose text contains the city we typed, and click that one.
            """
            try:
                field = driver.find_element(
                    By.XPATH, f'//input[@placeholder="{placeholder}"]'
                )
            except Exception:  # noqa: BLE001
                return False

            field.click()
            time.sleep(0.4)
            field.clear()
            field.send_keys(value)

            target = value.strip().lower()
            deadline = time.monotonic() + 15
            seen = 0

            # Hand-rolled poll rather than WebDriverWait: the dropdown re-renders
            # on every keystroke, so the matcher routinely hits
            # StaleElementReferenceException. WebDriverWait only swallows
            # NoSuchElementException by default and would abort on the first
            # stale hit — which is exactly why this silently found nothing.
            while time.monotonic() < deadline:
                try:
                    options = driver.find_elements(
                        By.CSS_SELECTOR, "ul li, li, [role='option']"
                    )
                except Exception:  # noqa: BLE001
                    options = []
                seen = max(seen, len(options))

                for option in options:
                    try:
                        if not option.is_displayed():
                            continue
                        text = (option.text or "").strip().lower()
                        # Prefer the city entry ("Delhi (All boarding points)")
                        # over an individual boarding point ("Narela (Delhi)").
                        if text.startswith(target):
                            # Must be a *native* click. A JS
                            # ``arguments[0].click()`` sets the visible input
                            # text but never registers with the site's React
                            # state, so the search silently runs with no route
                            # selected and bounces back to the homepage.
                            # Verified live: JS click -> "/", native -> the
                            # real "/bus_search/Delhi/344/Jaipur/241/..." URL.
                            option.click()
                            time.sleep(0.6)
                            return True
                    except Exception:  # noqa: BLE001
                        continue  # stale element: re-poll
                time.sleep(0.4)

            logger.warning(
                "No autocomplete match for %r in %r (saw %d options)",
                value, placeholder, seen,
            )
            return False

        if not fill("Leaving From", origin) or not fill("Going To", destination):
            return None

        for selector in ("#search-button a", "#search-button",
                         "button[type='submit']", "a[class*='search']"):
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if not elements:
                continue
            try:
                elements[0].click()  # native click, for the same reason
                break
            except Exception:  # noqa: BLE001
                try:  # obscured by an overlay: fall back to JS
                    driver.execute_script("arguments[0].click();", elements[0])
                    break
                except Exception:  # noqa: BLE001
                    continue

        try:
            WebDriverWait(driver, 25).until(
                lambda d: "/bus_search/" in d.current_url
            )
        except Exception:  # noqa: BLE001
            return None

        match = _SEARCH_URL_RE.search(driver.current_url)
        if not match:
            return None
        return {
            "from_name": match.group("from_name"),
            "from_id": match.group("from_id"),
            "to_name": match.group("to_name"),
            "to_id": match.group("to_id"),
        }


#: Discovery drives the homepage autocomplete — two typed fields, each polling
#: a React dropdown for up to 15 s, then a 25 s wait for the search redirect. On
#: a route AbhiBus does not serve, every one of those waits runs to the end and
#: the whole call could exceed the 120 s fan-out budget, taking the bus result
#: with it (seen live on Delhi→Agra). Discovery gets its own, smaller budget so
#: a slow lookup costs the bus result at worst, never the whole search.
DISCOVERY_TIMEOUT = 45.0


@cached("buses:route_ids", ttl=settings.ttl_iata)
async def resolve_route(origin: str, destination: str) -> Optional[dict[str, str]]:
    """Cached city-id lookup for an AbhiBus route (ids are permanent)."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_discover_route, origin, destination),
            timeout=DISCOVERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "AbhiBus route discovery for %s->%s exceeded %.0fs; treating as "
            "unserved.", origin, destination, DISCOVERY_TIMEOUT,
        )
        return None
    except ScraperUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("AbhiBus route discovery failed %s->%s: %s",
                       origin, destination, exc)
        return None


def _build_url(route: dict[str, str], date_obj: datetime) -> str:
    return (
        f"https://www.abhibus.com/bus_search/"
        f"{route['from_name']}/{route['from_id']}/"
        f"{route['to_name']}/{route['to_id']}/"
        f"{date_obj.strftime('%d-%m-%Y')}/O"
    )


def _card_selectors() -> list[tuple[str, str]]:
    from selenium.webdriver.common.by import By

    return [
        (By.CSS_SELECTOR, "div.service-info"),
        (By.CSS_SELECTOR, "div[class*='bus-card']"),
        (By.CSS_SELECTOR, "div[class*='service-list'] > div"),
        (By.XPATH, '//h5[@class="title"]/ancestor::div[contains(@class,"row")][1]'),
        (By.XPATH, '//div[contains(@class,"search-result")]//div[contains(@class,"item")]'),
    ]


def _parse_card(card: Any) -> dict[str, Any] | None:
    """Extract one bus from a card element, tolerating missing fields."""
    from selenium.webdriver.common.by import By

    try:
        text = (card.text or "").strip()
    except Exception:  # noqa: BLE001
        return None
    if not text or len(text) < 10:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # A card reads:
    #   [optional offer banner]        e.g. "Upto ₹40 off. Code:RETURN500W"
    #   OPERATOR                       e.g. "GOLDLINE SUPERDELUXE"
    #   BUS TYPE                       e.g. "VOLVO 9600 AC Seater/Sleeper (2 + 1)"
    #   Starts from <city>
    #   ...
    #   Save ₹40 / ₹400 / From / ₹360 / Select Seats / 42 Seats Left
    name, bus_type = "", None
    anchor = next(
        (i for i, ln in enumerate(lines) if ln.lower().startswith("starts from")), None
    )
    if anchor is not None:
        if anchor >= 1:
            bus_type = lines[anchor - 1]
        if anchor >= 2:
            name = lines[anchor - 2]

    if not name:  # structural fallback
        for by, sel in (
            (By.CSS_SELECTOR, "[class*='travel-name']"),
            (By.CSS_SELECTOR, "h5.title"),
            (By.CSS_SELECTOR, "h5"),
        ):
            try:
                name = (card.find_element(by, sel).text or "").strip()
                if name:
                    break
            except Exception:  # noqa: BLE001
                continue
    # Never let a promo banner masquerade as the operator.
    if not name or _is_promo(name):
        name = next((ln for ln in lines if not _is_promo(ln) and len(ln) > 3), "Unknown")

    times = [f"{h.zfill(2)}:{m}" for h, m in _TIME_RE.findall(text)]
    fare = _extract_fare(lines)

    seats = None
    seat_match = re.search(r"(\d+)\s*Seats?\s*(?:available|left)", text, re.IGNORECASE)
    if seat_match:
        seats = int(seat_match.group(1))

    return {
        "operator": name,
        "departure": times[0] if len(times) > 0 else None,
        "arrival": times[1] if len(times) > 1 else None,
        "fare_inr": fare,
        "seats_available": seats,
        "bus_type": bus_type,
    }


_PROMO_RE = re.compile(
    r"(off|offer|code:|save|cashback|discount|%\s*off|auto-applied)", re.IGNORECASE
)


def _is_promo(line: str) -> bool:
    return bool(_PROMO_RE.search(line))


def _extract_fare(lines: list[str]) -> Optional[int]:
    """Return the payable fare, not the discount.

    A card shows "Save ₹40", then the struck-through "₹400", then "From", then
    the actual payable "₹360". Taking ``min()`` of every rupee figure on the
    card — as the first version did — returned 40, the *discount*, as the fare.
    """
    # Preferred: the amount immediately after the "From" label.
    for index, line in enumerate(lines):
        if line.strip().lower() == "from" and index + 1 < len(lines):
            match = _FARE_RE.search(lines[index + 1])
            if match:
                return int(match.group(1).replace(",", ""))

    # Otherwise the smallest amount on a line that is not a promo line.
    candidates: list[int] = []
    for line in lines:
        if _is_promo(line):
            continue
        for raw in _FARE_RE.findall(line):
            candidates.append(int(raw.replace(",", "")))
    return min(candidates) if candidates else None


def _scrape(url: str, max_cards: int = 40) -> list[dict[str, Any]]:
    """Scrape a prepared search URL. Runs in a worker thread.

    ``max_cards`` exists for a measured reason: every ``element.text`` is a
    round trip to the browser, and a busy corridor lists well over a hundred
    services. Parsing all of them to return the cheapest ten took Delhi→Agra
    past its 85-second budget and lost the result entirely, while Delhi→Jaipur
    (a shorter list) finished in 66. We sort by fare afterwards, so reading a
    generous prefix costs nothing real.
    """
    logger.info("abhibus scrape: %s", url)

    buses: list[dict[str, Any]] = []
    with browser() as driver:
        driver.get(url)

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        try:
            # "Showing N Buses on this route" is the real completion signal.
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        '//*[contains(text(),"Buses on this route")] '
                        '| //div[contains(@class,"service-info")] '
                        '| //*[contains(translate(text(),"NOBUSE","nobuse"),"no buses")]',
                    )
                )
            )
        except Exception:  # noqa: BLE001
            logger.info("abhibus: results never rendered for %s", url)
            return []

        page_text = (driver.find_element(By.TAG_NAME, "body").text or "").lower()
        if "no buses" in page_text or "no services" in page_text:
            return []

        cards = find_all_first(driver, _card_selectors())
        if len(cards) > max_cards:
            logger.info("abhibus: %d services listed, parsing the first %d.",
                        len(cards), max_cards)
        for card in cards[:max_cards]:
            parsed = _parse_card(card)
            if parsed and (parsed["departure"] or parsed["fare_inr"]):
                parsed["booking_link"] = url
                buses.append(parsed)

    # De-duplicate: nested selectors can match the same card twice.
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for bus in buses:
        key = (bus["operator"], bus["departure"], bus["fare_inr"])
        if key not in seen:
            seen.add(key)
            unique.append(bus)
    return unique


@cached("buses", ttl=settings.ttl_buses)
# retries=0 deliberately: the failure mode here is a slow page, and a second
# 90-second attempt could only ever be cancelled by the caller's own budget.
# Two attempts made the bus the one provider that could not finish in time.
@resilient("abhibus", timeout=85.0, retries=0, recovery_seconds=180)
async def search_buses(
    origin: str, destination: str, travel_date: str, limit: int = 10
) -> dict[str, Any]:
    """Find buses. ``travel_date`` is ISO ``YYYY-MM-DD``."""
    try:
        date_obj = datetime.strptime(travel_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ProviderError("abhibus", f"bad date {travel_date!r}, expected YYYY-MM-DD") from exc

    try:
        route = await resolve_route(origin, destination)
        if not route:
            return {
                "origin": origin, "destination": destination, "date": travel_date,
                "count": 0, "buses": [], "cheapest_inr": None,
                "note": f"AbhiBus does not list a bus route from {origin} to {destination}.",
            }
        buses = await asyncio.to_thread(_scrape, _build_url(route, date_obj))
    except ScraperUnavailable as exc:
        raise ProviderError("abhibus", str(exc)) from exc

    buses.sort(key=lambda b: (b.get("fare_inr") or float("inf")))
    return {
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "count": len(buses[:limit]),
        "buses": buses[:limit],
        "cheapest_inr": buses[0]["fare_inr"] if buses and buses[0].get("fare_inr") else None,
        "note": None if buses else "No buses found for this route and date.",
    }


def render(result: dict[str, Any]) -> str:
    buses = result.get("buses") or []
    if not buses:
        return f"🚌 {result.get('note', 'No buses found.')}"
    lines = [
        f"🚌 **{len(buses)} bus option(s)** {result['origin']} → "
        f"{result['destination']} on {result['date']}\n"
    ]
    for b in buses:
        bits = [f"**{b['operator']}**"]
        if b.get("departure"):
            bits.append(f"{b['departure']} → {b.get('arrival') or '?'}")
        if b.get("bus_type"):
            bits.append(b["bus_type"])
        if b.get("fare_inr"):
            bits.append(f"₹{b['fare_inr']:,}")
        if b.get("seats_available") is not None:
            bits.append(f"{b['seats_available']} seats")
        lines.append("🔹 " + " · ".join(bits))
    if buses[0].get("booking_link"):
        lines.append(f"\n🔗 {buses[0]['booking_link']}")
    return "\n".join(lines)
