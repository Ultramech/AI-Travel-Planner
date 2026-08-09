"""Central configuration.

Every secret and tunable lives here, sourced from the environment. Nothing is
hardcoded — the previous version embedded live API keys for WeatherAPI, SerpAPI,
Amadeus and Groq directly in source files that were pushed to a public repo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---------------------------------------------------------------
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_extractor_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_retries: int = 3

    # --- Providers ---------------------------------------------------------
    weather_api_key: str = ""
    serpapi_key: str = ""
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
    amadeus_hostname: Literal["test", "production"] = "test"
    # Flight source: "auto" prefers Amadeus when credentials exist and falls
    # back to keyless Google Flights; "google"/"amadeus" force one.
    flight_provider: Literal["auto", "google", "amadeus"] = "auto"
    tavily_api_key: str = ""

    # --- Redis -------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True

    # --- Kafka -------------------------------------------------------------
    kafka_enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_requests: str = "trip.requests"
    kafka_topic_events: str = "trip.events"
    kafka_topic_dlq: str = "trip.dlq"
    kafka_consumer_group: str = "trip-planner-workers"

    # --- Database ----------------------------------------------------------
    # Optional, like Redis and Kafka: empty means "run without durable history".
    # Accepts any Postgres URI — local, Supabase pooler, RDS. The driver is
    # normalised to asyncpg in app/db/session.py, so postgres://, postgresql://
    # and ?sslmode=require all work as pasted.
    database_url: str = ""

    # Supabase REST/auth credentials. Deliberately separate from DATABASE_URL:
    # these reach PostgREST and GoTrue over HTTPS, they do NOT open a Postgres
    # connection. The persistence layer needs DATABASE_URL (project ref + the
    # database password), which is a different secret entirely.
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""
    supabase_jwks_url: str = ""

    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # --- Checkpointing -----------------------------------------------------
    # "postgres" shares one durable store across every worker, which is what
    # makes horizontal scaling real: any worker can resume any paused trip.
    checkpointer_backend: Literal["memory", "sqlite", "redis", "postgres"] = "sqlite"
    checkpoint_sqlite_path: str = "data/checkpoints/trips.sqlite"

    # --- Scraping ----------------------------------------------------------
    scraper_enabled: bool = True
    scraper_headless: bool = True
    scraper_timeout_seconds: int = 35
    selenium_remote_url: str = ""

    # --- Service -----------------------------------------------------------
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    graph_recursion_limit: int = 40
    http_timeout_seconds: float = 15.0

    # --- Cache TTLs (seconds) ---------------------------------------------
    # Tuned to how fast each upstream actually changes. Weather moves hourly,
    # a railway timetable barely moves at all.
    ttl_weather: int = Field(default=3600)
    ttl_flights: int = Field(default=1800)
    ttl_hotels: int = Field(default=21600)
    ttl_trains: int = Field(default=21600)
    ttl_buses: int = Field(default=1800)
    ttl_cabs: int = Field(default=3600)
    ttl_wikivoyage: int = Field(default=604800)
    ttl_iata: int = Field(default=2592000)
    ttl_llm_extract: int = Field(default=86400)

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def checkpoint_path(self) -> Path:
        p = PROJECT_ROOT / self.checkpoint_sqlite_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def require(self, *names: str) -> None:
        """Fail loudly and early when a needed credential is absent."""
        missing = [n for n in names if not getattr(self, n, "")]
        if missing:
            raise RuntimeError(
                f"Missing required settings: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
