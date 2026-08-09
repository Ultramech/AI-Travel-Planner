"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import VERSION, router
from app.config import settings
from app.core.bus import bus
from app.core.cache import cache
from app.db.session import init_db, shutdown_db
from app.graph.builder import get_graph, shutdown_graph
from app.logging_conf import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect infrastructure once at startup, not per request.

    The original code constructed a new ChatGroq client, a new MemorySaver and a
    freshly compiled graph on every Streamlit rerun.
    """
    configure_logging()
    logger.info("Starting AI Travel Planner API v%s", VERSION)

    await cache.connect()
    await init_db()  # before the graph: the Postgres checkpointer needs it
    await bus.start()
    await get_graph()  # compile + open the checkpointer up front

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set — planning requests will fail.")

    try:
        yield
    finally:
        logger.info("Shutting down…")
        await bus.stop()
        await cache.close()
        await shutdown_graph()
        await shutdown_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Travel Planner API",
        description=(
            "LangGraph-powered trip planning with durable checkpointing, "
            "human-in-the-loop review, Redis caching and Kafka-backed async "
            "execution."
        ),
        version=VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten before any real deployment
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id and log latency — basic observability."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error [%s] %s %s", request_id,
                             request.method, request.url.path)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
        if not request.url.path.startswith("/health"):
            logger.info("[%s] %s %s -> %s (%.1f ms)", request_id, request.method,
                        request.url.path, response.status_code, elapsed_ms)
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "ai-travel-planner",
            "version": VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
