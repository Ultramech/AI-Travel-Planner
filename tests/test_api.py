"""API contract tests against the real ASGI app (no network)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.main import create_app

FUTURE = (date.today() + timedelta(days=15)).isoformat()
FUTURE_END = (date.today() + timedelta(days=18)).isoformat()


@pytest.fixture
async def client(monkeypatch):
    """App with infrastructure stubbed out — no Redis, Kafka or LLM needed."""
    from app.core import bus as bus_module
    from app.core.cache import cache

    async def noop(*args, **kwargs):
        return None

    async def no_kafka(*args, **kwargs):
        return False

    monkeypatch.setattr(cache, "connect", noop)
    monkeypatch.setattr(bus_module.bus, "start", noop)
    monkeypatch.setattr(bus_module.bus, "stop", noop)
    monkeypatch.setattr(bus_module.bus, "send", no_kafka)

    # Each test runs in its own event loop; the checkpointer holds a connection
    # bound to the loop that opened it, so start from a clean slate.
    from app.graph.builder import shutdown_graph

    await shutdown_graph()

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Trigger lifespan so the graph compiles.
        async with app.router.lifespan_context(app):
            yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "redis" in body["dependencies"]
    assert "kafka" in body["dependencies"]


@pytest.mark.asyncio
async def test_liveness_is_always_cheap(client):
    assert (await client.get("/health/live")).json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_graph_topology_endpoint(client):
    response = await client.get("/api/v1/graph")
    assert response.status_code == 200
    diagram = response.json()["diagram"]
    assert "human_review_transport" in diagram
    assert "build_itinerary" in diagram


# ---------------------------------------------------------------------------
# Validation happens at the edge — the old app had none at all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,reason",
    [
        ({}, "neither query nor origin/destination"),
        ({"origin": "Delhi"}, "destination missing"),
        ({"origin": "Delhi", "destination": "Delhi"}, "same city"),
        (
            {"origin": "Delhi", "destination": "Goa",
             "start_date": FUTURE_END, "end_date": FUTURE},
            "end before start",
        ),
        (
            {"origin": "Delhi", "destination": "Goa",
             "start_date": "2020-01-01", "end_date": "2020-01-05"},
            "start date in the past",
        ),
        (
            {"origin": "Delhi", "destination": "Goa", "persons": 0},
            "zero travellers",
        ),
        (
            {"origin": "Delhi", "destination": "Goa", "persons": 999},
            "too many travellers",
        ),
        (
            {"origin": "Delhi", "destination": "Goa", "start_date": FUTURE,
             "end_date": (date.today() + timedelta(days=90)).isoformat()},
            "trip too long",
        ),
    ],
)
async def test_invalid_requests_are_rejected(client, payload, reason):
    response = await client.post("/api/v1/trips", json=payload)
    assert response.status_code == 422, f"should reject: {reason}"


@pytest.mark.asyncio
async def test_valid_request_is_accepted_and_queued(client, monkeypatch):
    from app.services import planner

    async def fake_set_queued(trip_id, request):
        return None

    monkeypatch.setattr(planner, "set_queued", fake_set_queued)

    response = await client.post(
        "/api/v1/trips",
        json={
            "origin": "Delhi", "destination": "Goa",
            "start_date": FUTURE, "end_date": FUTURE_END,
            "persons": 2, "preferences": ["beaches"],
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["trip_id"].startswith("trip_")
    assert body["status"] == "queued"
    assert body["stream_url"].endswith("/stream")


@pytest.mark.asyncio
async def test_unknown_trip_returns_404(client):
    response = await client.get("/api/v1/trips/trip_doesnotexist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_resume_requires_feedback_for_revisions(client):
    response = await client.post(
        "/api/v1/trips/trip_x/resume", json={"action": "revise_transport"}
    )
    assert response.status_code == 422

    # approve needs no feedback — it should get past validation to a 404/409.
    response = await client.post(
        "/api/v1/trips/trip_x/resume", json={"action": "approve"}
    )
    assert response.status_code in {404, 409}


@pytest.mark.asyncio
async def test_resume_rejects_unknown_actions(client):
    response = await client.post(
        "/api/v1/trips/trip_x/resume",
        json={"action": "delete_everything", "feedback": "hi"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_request_id_header_is_returned(client):
    response = await client.get("/health/live")
    assert response.headers.get("x-request-id")
    assert response.headers.get("x-response-time-ms")
