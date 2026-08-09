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
# SCRAPER_ENABLED is deliberately NOT forced here: the suite's behaviour with
# real Chrome is what it was before this file existed. Set it to false yourself
# for a faster run.


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """The opt-in database for persistence tests; empty means skip them."""
    return TEST_DATABASE_URL


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
