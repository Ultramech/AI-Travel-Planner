"""Durable persistence.

Everything else in this project treats storage as ephemeral: Redis is a cache,
the LangGraph checkpointer is an execution log. Neither is a system of record.
This package is.
"""

from app.db.session import db_enabled, get_session, init_db, shutdown_db

__all__ = ["db_enabled", "get_session", "init_db", "shutdown_db"]
