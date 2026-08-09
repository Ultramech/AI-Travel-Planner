"""Logging setup. The original project used bare ``print()`` and ``st.write``
for diagnostics, which is invisible in a worker and unfilterable in production.
"""

from __future__ import annotations

import logging
import sys

from app.config import settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, (level or settings.log_level).upper(), logging.INFO))

    # These are chatty and rarely useful at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "selenium", "aiokafka",
                  "WDM", "asyncio", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
