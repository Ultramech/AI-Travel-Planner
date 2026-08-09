"""Persistence layer tests.

Split in two:

* **URL normalisation and degradation** — pure logic, always run. These cover
  the connection strings a user actually pastes (Supabase pooler URIs with
  ``sslmode``) and the guarantee that a missing database degrades instead of
  breaking planning.
* **Round-trip against real Postgres** — skipped unless ``TEST_DATABASE_URL``
  is set, because asserting on SQL behaviour against a mock proves nothing.
"""

from __future__ import annotations

import os
import ssl
import uuid

import pytest

from app.db.session import checkpointer_dsn, normalise_url

# ---------------------------------------------------------------------------
# URL normalisation — the Supabase footguns
# ---------------------------------------------------------------------------


def test_bare_postgres_scheme_is_rewritten_to_asyncpg():
    url, args = normalise_url("postgresql://u:p@host:5432/db")
    assert url.startswith("postgresql+asyncpg://")
    assert args == {}


def test_heroku_style_postgres_scheme_is_accepted():
    url, _ = normalise_url("postgres://u:p@host:5432/db")
    assert url.startswith("postgresql+asyncpg://")


def test_sslmode_is_stripped_and_translated():
    """asyncpg raises TypeError on an unexpected 'sslmode' kwarg.

    Supabase hands out URIs ending in ?sslmode=require, so passing the query
    string through untouched breaks on the first connection attempt.
    """
    url, args = normalise_url("postgresql://u:p@db.supabase.co:5432/postgres?sslmode=require")
    assert "sslmode" not in url
    assert isinstance(args["ssl"], ssl.SSLContext)


def test_sslmode_disable_turns_ssl_off():
    _, args = normalise_url("postgresql://u:p@localhost:5432/db?sslmode=disable")
    assert args["ssl"] is False


def test_transaction_pooler_port_disables_prepared_statements():
    """Port 6543 is Supabase's transaction pooler.

    Prepared statements cached by asyncpg land on a different backend after
    pgbouncer re-multiplexes, failing with 'prepared statement does not exist'.
    """
    _, args = normalise_url("postgresql://u:p@aws-0-ap-south-1.pooler.supabase.com:6543/postgres")
    assert args["statement_cache_size"] == 0
    assert callable(args["prepared_statement_name_func"])
    # Names must be unique per statement, not a constant.
    assert args["prepared_statement_name_func"]() != args["prepared_statement_name_func"]()


def test_pgbouncer_flag_is_removed_from_query():
    url, args = normalise_url("postgresql://u:p@host:5432/db?pgbouncer=true")
    assert "pgbouncer" not in url
    assert args["statement_cache_size"] == 0


def test_empty_url_yields_nothing():
    assert normalise_url("") == ("", {})


def test_checkpointer_dsn_keeps_libpq_form(monkeypatch):
    """The LangGraph saver uses psycopg, which wants the opposite of asyncpg."""
    from app.config import settings

    monkeypatch.setattr(
        settings, "database_url", "postgresql://u:p@db.abc.supabase.co:5432/postgres"
    )
    dsn = checkpointer_dsn()
    assert dsn.startswith("postgresql://")
    assert "+asyncpg" not in dsn
    assert "sslmode=require" in dsn  # added for Supabase hosts


# ---------------------------------------------------------------------------
# Degradation — no database configured
# ---------------------------------------------------------------------------


async def test_writes_are_silent_noops_without_a_database():
    """A planning run must not fail because persistence is unavailable."""
    from app.db import repository
    from app.db.session import db_enabled

    assert not db_enabled()
    # None of these should raise.
    await repository.record_requested("trip_x", {"origin": "Delhi"})
    await repository.record_status("trip_x", "running")
    await repository.persist_result("trip_x", {"status": "completed"})
    await repository.record_provider_call("trip_x", "trains", ok=True)


async def test_reads_raise_a_clear_error_without_a_database():
    """Reads are different: silently returning [] would look like 'no history'."""
    from app.db import repository

    with pytest.raises(repository.PersistenceUnavailable):
        await repository.list_trips()
    with pytest.raises(repository.PersistenceUnavailable):
        await repository.get_trip("trip_x")


# ---------------------------------------------------------------------------
# Round trip against real Postgres
# ---------------------------------------------------------------------------

pytestmark_db = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run persistence integration tests",
)


@pytestmark_db
class TestRealPostgres:
    @pytest.fixture(autouse=True)
    async def _db(self, test_database_url, monkeypatch):
        from app.config import settings
        from app.db import session as db_session

        monkeypatch.setattr(settings, "database_url", test_database_url)
        db_session._engine = None
        db_session._sessionmaker = None
        assert await db_session.init_db(), "TEST_DATABASE_URL is set but unreachable"
        yield
        await db_session.shutdown_db()

    async def test_request_then_result_round_trip(self):
        from app.db import repository

        trip_id = f"trip_test_{uuid.uuid4().hex[:12]}"
        await repository.record_requested(
            trip_id,
            {
                "origin": "Delhi",
                "destination": "Goa",
                "start_date": "2026-09-01",
                "end_date": "2026-09-05",
                "persons": 2,
                "preferences": ["beaches"],
            },
        )
        stored = await repository.get_trip(trip_id)
        assert stored["status"] == "queued"
        assert stored["destination"] == "Goa"

        await repository.persist_result(
            trip_id,
            {
                "status": "completed",
                "brief": {"origin": "Delhi", "destination": "Goa", "persons": 2},
                "itinerary": "# Day 1\nBaga Beach",
                "budget": {"total_inr": 42000, "per_person_inr": 21000, "currency": "INR"},
                "revision_count": 0,
            },
        )
        stored = await repository.get_trip(trip_id)
        assert stored["status"] == "completed"
        assert stored["total_inr"] == 42000
        assert "Baga Beach" in stored["itinerary"]

    async def test_revisions_are_kept_not_overwritten(self):
        """A revised plan must not destroy the one it replaced."""
        from app.db import repository

        trip_id = f"trip_test_{uuid.uuid4().hex[:12]}"
        await repository.record_requested(trip_id, {"origin": "A", "destination": "B"})
        for revision, text, total in ((0, "first plan", 30000), (1, "cheaper plan", 21000)):
            await repository.persist_result(
                trip_id,
                {
                    "status": "completed",
                    "brief": {},
                    "itinerary": text,
                    "revision_count": revision,
                    "budget": {"total_inr": total, "currency": "INR"},
                },
            )

        stored = await repository.get_trip(trip_id)
        assert [r["revision"] for r in stored["itinerary_revisions"]] == [0, 1]
        assert [c["total_inr"] for c in stored["cost_history"]] == [30000.0, 21000.0]

    async def test_persist_result_is_idempotent_per_revision(self):
        """At-least-once Kafka delivery replays a job; that must not duplicate."""
        from app.db import repository

        trip_id = f"trip_test_{uuid.uuid4().hex[:12]}"
        await repository.record_requested(trip_id, {"origin": "A", "destination": "B"})
        payload = {
            "status": "completed",
            "brief": {},
            "itinerary": "same plan",
            "revision_count": 0,
            "budget": {"total_inr": 1000, "currency": "INR"},
        }
        await repository.persist_result(trip_id, payload)
        await repository.persist_result(trip_id, payload)

        stored = await repository.get_trip(trip_id)
        assert len(stored["itinerary_revisions"]) == 1
        assert len(stored["cost_history"]) == 1

    async def test_search_filters(self):
        from app.db import repository

        marker = uuid.uuid4().hex[:8]
        trip_id = f"trip_test_{marker}"
        await repository.record_requested(
            trip_id,
            {"origin": f"Origin{marker}", "destination": f"Dest{marker}", "persons": 1},
        )

        rows, total = await repository.list_trips(destination=f"Dest{marker}")
        assert total == 1 and rows[0]["trip_id"] == trip_id

        # case-insensitive
        rows, total = await repository.list_trips(destination=f"dest{marker}".lower())
        assert total == 1

        rows, total = await repository.list_trips(q=marker)
        assert total >= 1

        rows, total = await repository.list_trips(destination="definitely-not-here")
        assert total == 0 and rows == []

    async def test_provider_calls_and_stats(self):
        from app.db import repository

        trip_id = f"trip_test_{uuid.uuid4().hex[:12]}"
        await repository.record_requested(trip_id, {"origin": "A", "destination": "B"})
        provider = f"prov_{uuid.uuid4().hex[:8]}"
        await repository.record_provider_call(trip_id, provider, ok=True, duration_ms=100)
        await repository.record_provider_call(trip_id, provider, ok=False, duration_ms=300,
                                              error="boom")

        calls = await repository.get_provider_calls(trip_id)
        assert len(calls) == 2

        stats = {s["provider"]: s for s in await repository.provider_stats()}
        assert stats[provider]["calls"] == 2
        assert stats[provider]["failed"] == 1
        assert stats[provider]["success_rate"] == 0.5

    async def test_provider_call_survives_unknown_trip(self):
        """An audit row must never be lost to a missing foreign key."""
        from app.db import repository

        provider = f"prov_{uuid.uuid4().hex[:8]}"
        await repository.record_provider_call("trip_does_not_exist", provider, ok=True)
        stats = {s["provider"]: s for s in await repository.provider_stats()}
        assert stats[provider]["calls"] == 1


# ---------------------------------------------------------------------------
# Fail-fast must not be recorded as success
# ---------------------------------------------------------------------------


def test_fatal_error_is_detected():
    """Regression: an unparseable request was stored as a completed trip.

    The fail-fast node routes to END, which is also how a successful run ends,
    so 'no pending interrupt' was being read as 'finished successfully'.
    """
    from app.services.planner import has_fatal_error

    assert has_fatal_error({"errors": [{"node": "failed", "fatal": True}]})
    assert not has_fatal_error({"errors": []})
    assert not has_fatal_error({})
    # A non-fatal provider error must not condemn the whole trip.
    assert not has_fatal_error({"errors": [{"node": "weather", "message": "no key"}]})


async def test_fail_fast_node_marks_the_trip_fatal():
    from app.graph.builder import _fail_fast
    from app.services.planner import has_fatal_error

    result = await _fail_fast({})
    assert has_fatal_error(result)


@pytestmark_db
class TestProviderStatsWindow:
    """`since_hours` must actually bound the window, not just be accepted."""

    @pytest.fixture(autouse=True)
    async def _db(self, test_database_url, monkeypatch):
        from app.config import settings
        from app.db import session as db_session

        monkeypatch.setattr(settings, "database_url", test_database_url)
        db_session._engine = None
        db_session._sessionmaker = None
        assert await db_session.init_db()
        yield
        await db_session.shutdown_db()

    async def test_old_calls_fall_outside_the_window(self):
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import update

        from app.db import repository
        from app.db.models import ProviderCall
        from app.db.session import get_session

        provider = f"prov_{uuid.uuid4().hex[:8]}"
        await repository.record_provider_call(None, provider, ok=True, duration_ms=10)

        # Backdate it well beyond the window.
        async with get_session() as session:
            await session.execute(
                update(ProviderCall)
                .where(ProviderCall.provider == provider)
                .values(created_at=datetime.now(timezone.utc) - timedelta(hours=48))
            )

        names_24h = {s["provider"] for s in await repository.provider_stats(since_hours=24)}
        assert provider not in names_24h, "a 48h-old call leaked into the 24h window"

        names_72h = {s["provider"] for s in await repository.provider_stats(since_hours=72)}
        assert provider in names_72h


# ---------------------------------------------------------------------------
# Checkpointer setup race
# ---------------------------------------------------------------------------


async def test_concurrent_checkpoint_setup_is_tolerated():
    """Regression: the API and every worker call setup() at startup.

    On a shared database they race to insert the same migration row and all but
    one crash with a duplicate-key error, silently falling back to local SQLite
    — so 'postgres' in the config produced per-process state in practice.
    """
    from app.graph.checkpointer import _setup_shared

    class RacingSaver:
        def __init__(self) -> None:
            self.probed = False

        async def setup(self):
            raise RuntimeError(
                'duplicate key value violates unique constraint '
                '"checkpoint_migrations_pkey"'
            )

        async def aget_tuple(self, config):
            self.probed = True
            return None

    saver = RacingSaver()
    await _setup_shared(saver)  # must not raise
    assert saver.probed, "the race path must verify the schema, not just assume it"


async def test_genuine_setup_failure_still_propagates():
    """Only the race is forgiven — a real failure must still fall back."""
    from app.graph.checkpointer import _setup_shared

    class BrokenSaver:
        async def setup(self):
            raise RuntimeError("connection refused")

        async def aget_tuple(self, config):  # pragma: no cover
            raise AssertionError("should not probe after a non-race failure")

    with pytest.raises(RuntimeError, match="connection refused"):
        await _setup_shared(BrokenSaver())


# ---------------------------------------------------------------------------
# Status inference
# ---------------------------------------------------------------------------


async def test_in_flight_checkpoint_is_not_reported_as_completed(monkeypatch):
    """Regression: polling a just-submitted trip returned `completed`.

    Between two graph steps the checkpoint holds state but has no scheduled
    next task. Reading that as "finished" told clients a trip was done — with
    no itinerary — seconds after submission and before any work had happened.
    """
    from app.core.events import TripStatus
    from app.services import planner as planner_module

    class Snapshot:
        values = {"trip_id": "trip_x", "brief": {"origin": "Delhi"}}
        next = ()          # nothing scheduled *right now*
        tasks = ()         # and no pending interrupt
        created_at = "2026-08-09T00:00:00Z"

    class FakeGraph:
        async def aget_state(self, config):
            return Snapshot()

    async def fake_graph():
        return FakeGraph()

    monkeypatch.setattr(planner_module, "get_graph", fake_graph)

    async def running_status(trip_id):
        return {"status": TripStatus.RUNNING.value}

    monkeypatch.setattr(planner_module, "get_status", running_status)
    view = await planner_module.get_trip("trip_x")
    assert view["status"] == TripStatus.RUNNING.value

    # Once nothing is recorded any more (status key expired), an idle
    # checkpoint legitimately means the run is over.
    async def no_status(trip_id):
        return None

    monkeypatch.setattr(planner_module, "get_status", no_status)
    view = await planner_module.get_trip("trip_x")
    assert view["status"] == TripStatus.COMPLETED.value
