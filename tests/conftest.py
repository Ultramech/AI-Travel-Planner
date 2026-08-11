"""Test isolation.

This file exists because of a real bug: the suite had no conftest, so it read
whatever ``.env`` happened to contain. Once ``DATABASE_URL`` was set, running
``make test`` wrote trips and provider-call rows into the developer's database —
and would have written them into Supabase or production just as happily.

Environment variables are set **before** ``app.config`` is imported anywhere,
and pydantic-settings ranks real environment variables above ``.env``, so these
win. The default posture is "no external infrastructure": no database, no
Kafka, in-memory checkpoints.

To run the persistence tests against a real Postgres, set TEST_DATABASE_URL:

    TEST_DATABASE_URL=postgresql://localhost:5433/travel_planner_test make test
"""

from __future__ import annotations

import os

import pytest

# --- must happen before any `app.*` import -------------------------------
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

os.environ["DATABASE_URL"] = ""  # never touch a real database by default
os.environ["KAFKA_ENABLED"] = "false"
os.environ["CHECKPOINTER_BACKEND"] = "sqlite"
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
# Forced off. Leaving this to the developer was a mistake of the same family as
# the missing conftest itself: with KAFKA_ENABLED=false the API degrades to
# running the trip in an in-process background task, so `POST /api/v1/trips`
# in the suite launched real Chrome, scraped live sites and called a paid LLM.
# It showed up as one test taking 122 seconds. A unit suite must not touch the
# network; the live checks are run deliberately, not as a side effect.
os.environ["SCRAPER_ENABLED"] = "false"
# Selenium is off above, so nothing should ever queue for a browser slot.
os.environ["SCRAPER_QUEUE_SECONDS"] = "1"
# Point the cache at a Redis that is not there, so it falls back to the
# in-process LRU. Otherwise the suite reads and writes the developer's real
# Redis: bringing the Docker stack up made a geo test that passes alone fail in
# the suite, because a live run had already cached an answer for that city.
# Test results must not depend on what happens to be in a cache.
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/15"


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """The opt-in database for persistence tests; empty means skip them."""
    return TEST_DATABASE_URL


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
