"""Selenium driver lifecycle.

The two scrapers in the original project were the most dangerous code in it:

  * ``train_data.get_etrain_trains`` **never called ``driver.quit()``** on the
    success path or on most error paths. Every train lookup leaked a Chrome
    process and its profile directory. A handful of requests exhausted memory.
  * Both launched a *visible, maximised* browser — impossible to run on a
    server, and it steals focus on a laptop.
  * Waits were ``sleep(5)`` rather than conditions, so they were simultaneously
    too slow on fast networks and too flaky on slow ones.
  * No overall time budget: a hung page blocked the caller indefinitely.

This module gives every scrape a context manager that guarantees teardown, a
headless-by-default configuration, and hard page/script timeouts.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from typing import Any, Iterator, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class ScraperUnavailable(RuntimeError):
    """Raised when no usable browser can be started."""


def _chrome_options() -> Any:
    from selenium.webdriver.chrome.options import Options

    options = Options()
    if settings.scraper_headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")  # avoids /dev/shm crashes in Docker
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--blink-settings=imagesEnabled=false")  # big speed win
    options.add_argument("--lang=en-IN")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    options.set_capability("pageLoadStrategy", "eager")
    return options


@contextlib.contextmanager
def browser(timeout: Optional[int] = None) -> Iterator[Any]:
    """Yield a configured WebDriver and *always* tear it down."""
    if not settings.scraper_enabled:
        raise ScraperUnavailable("scraping is disabled (SCRAPER_ENABLED=false)")

    from selenium import webdriver

    timeout = timeout or settings.scraper_timeout_seconds
    profile_dir = tempfile.mkdtemp(prefix="tp-chrome-")
    options = _chrome_options()
    options.add_argument(f"--user-data-dir={profile_dir}")

    driver: Any = None
    try:
        try:
            if settings.selenium_remote_url:
                driver = webdriver.Remote(
                    command_executor=settings.selenium_remote_url, options=options
                )
            else:
                driver = webdriver.Chrome(options=options)
        except Exception as exc:  # noqa: BLE001
            raise ScraperUnavailable(f"could not start Chrome: {exc}") from exc

        driver.set_page_load_timeout(timeout)
        driver.set_script_timeout(timeout)
        yield driver
    finally:
        if driver is not None:
            with contextlib.suppress(Exception):
                driver.quit()
        shutil.rmtree(profile_dir, ignore_errors=True)


def find_first(driver: Any, selectors: list[tuple[str, str]], timeout: int = 10) -> Any:
    """Try several locators in order and return the first that resolves.

    Site markup drifts constantly. The original code pinned single brittle
    XPaths like ``//div[@class="row seat-info bd-success-400 text-success-600
    bg-success-50"]`` — one Tailwind class rename and the scraper silently
    returned nothing. Multiple fallbacks make it survive small changes.
    """
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    per_selector = max(2, timeout // max(1, len(selectors)))
    for by, value in selectors:
        try:
            return WebDriverWait(driver, per_selector).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            continue
    return None


def find_all_first(driver: Any, selectors: list[tuple[str, str]]) -> list[Any]:
    """Return elements from the first selector that yields any."""
    for by, value in selectors:
        with contextlib.suppress(Exception):
            elements = driver.find_elements(by, value)
            if elements:
                return elements
    return []


def texts(elements: list[Any]) -> list[str]:
    out = []
    for element in elements:
        with contextlib.suppress(Exception):
            text = (element.text or "").strip()
            if text:
                out.append(text)
    return out
