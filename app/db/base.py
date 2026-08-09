"""Declarative base and shared column conventions.

The naming convention matters more than it looks: without it, Alembic
autogenerate invents names for indexes and constraints that differ between the
model metadata and what Postgres actually reports, so every autogenerate run
produces spurious drop/create churn. Naming them deterministically makes
migrations diff cleanly.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Server-side timestamps.

    ``server_default=func.now()`` rather than a Python default so the database
    clock is the single source of truth — two workers on machines with drifting
    clocks would otherwise write inconsistent orderings.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
