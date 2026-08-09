"""Diagnose DATABASE_URL before running migrations.

    PYTHONPATH=. .venv/bin/python scripts/check_db.py

Connection problems against a hosted Postgres are slow to diagnose from a
stack trace thrown three layers down inside Alembic. This says, in one run,
exactly which part is wrong: the host, the credentials, the TLS mode, or the
schema.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from urllib.parse import unquote, urlsplit

from app.config import settings
from app.db.session import checkpointer_dsn, normalise_url


def _fail(message: str, hint: str = "") -> None:
    print(f"  ✗ {message}")
    if hint:
        print(f"    → {hint}")


async def main() -> int:
    url = settings.database_url
    print("DATABASE_URL check")
    print("-" * 60)

    if not url:
        _fail(
            "DATABASE_URL is empty.",
            "The app still runs, but with no durable history: writes become "
            "no-ops and the history endpoints answer 503.",
        )
        return 1

    parts = urlsplit(url)
    host, port = parts.hostname, parts.port or 5432
    # urlsplit leaves percent-encoding in place. A password like "Gaurang79@@"
    # must appear as %40%40 in the URI, and passing the raw form straight to
    # asyncpg authenticates with the literal "%40%40" and fails — looking
    # exactly like a wrong password. SQLAlchemy and psycopg unquote for us;
    # this script has to do it itself.
    user = unquote(parts.username or "")
    password = unquote(parts.password) if parts.password else parts.password
    safe = url.replace(parts.password, "***") if parts.password else url
    print(f"  url    {safe}")
    print(f"  host   {host}:{port}")
    print(f"  user   {user}")

    is_local = host in ("localhost", "127.0.0.1", "::1")
    if password == "<DB-PASSWORD>":
        _fail(
            "The password placeholder was never filled in.",
            "Supabase → Project Settings → Database → Connection string → URI. "
            "Reset the password there if you do not have it.",
        )
        return 1
    if not password and not is_local:
        _fail(
            "No database password in the URL for a remote host.",
            "A hosted Postgres will not accept trust auth.",
        )
        return 1

    # --- DNS ------------------------------------------------------------
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        families = {("IPv6" if i[0] == socket.AF_INET6 else "IPv4") for i in infos}
        print(f"  ✓ resolves ({', '.join(sorted(families))})")
        if families == {"IPv6"}:
            print(
                "    ! IPv6-only. Supabase direct connections are IPv6-only; "
                "use the pooler hostname if this machine has no IPv6 route."
            )
    except socket.gaierror as exc:
        _fail(f"DNS lookup failed: {exc}", "Check the hostname for typos.")
        return 1

    # --- Connect ---------------------------------------------------------
    import asyncpg

    normalised, connect_args = normalise_url(url)
    asyncpg_args = {k: v for k, v in connect_args.items() if k != "ssl"}
    ssl_mode = connect_args.get("ssl", "prefer")

    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=(parts.path or "/postgres").lstrip("/"),
            ssl=ssl_mode if ssl_mode is not False else False,
            timeout=20,
            **asyncpg_args,
        )
    except asyncpg.InvalidPasswordError:
        _fail(
            "Password rejected.",
            "This is the DATABASE password, not the anon/service API key.",
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        _fail(f"{type(exc).__name__}: {detail[:160]}")
        if "not found" in detail.lower():
            print(
                "    → 'tenant/user not found' means the pooler region is wrong. "
                "Copy the exact host from the Supabase dashboard."
            )
        return 1

    version = await conn.fetchval("select version()")
    print(f"  ✓ connected: {version.split(',')[0]}")

    tables = await conn.fetch(
        "select table_name from information_schema.tables "
        "where table_schema='public' order by 1"
    )
    names = [r["table_name"] for r in tables]
    expected = {"trips", "itineraries", "provider_calls", "cost_snapshots"}
    missing = expected - set(names)
    if missing:
        _fail(
            f"missing tables: {', '.join(sorted(missing))}",
            "Run:  make migrate",
        )
    else:
        print(f"  ✓ schema present ({len(names)} tables)")
        counts = {
            t: await conn.fetchval(f"select count(*) from {t}")  # noqa: S608
            for t in sorted(expected)
        }
        print(f"    rows: {counts}")

    if "checkpoints" in names:
        print("  ✓ LangGraph checkpoint tables present")
    elif settings.checkpointer_backend == "postgres":
        print("    ! checkpoint tables absent; they are created on first API start")

    await conn.close()

    if settings.checkpointer_backend == "postgres":
        dsn = checkpointer_dsn()
        if parts.password:
            dsn = dsn.replace(parts.password, "***")
        print(f"  checkpointer DSN: {dsn}")
        if port == 6543:
            _fail(
                "Checkpointer is set to postgres but this is the TRANSACTION "
                "pooler (6543), which does not support session-mode connections.",
                "Use the session pooler (5432) or set CHECKPOINTER_BACKEND=sqlite.",
            )
            return 1

    print("-" * 60)
    print("OK" if not missing else "Connected, but run `make migrate`.")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
