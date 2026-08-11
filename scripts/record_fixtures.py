"""Record live provider responses as replay fixtures.

Why
---
This project reads two websites that owe it nothing. When Google Flights changes
its internal response shape, the parser does not raise something obvious — it
produced *"No airline flies DEL→COK"* for a route with dozens of daily
departures, because the upstream library indexed a metadata block that had
stopped being sent. That is a silent wrong answer, and no unit test written
against a mock would have caught it: the mock would have kept the old shape
forever.

So the fixtures are recorded from the real thing, and
``tests/test_provider_contracts.py`` replays them through the real parsers. The
tests then answer a specific question: *if the upstream response looked like it
did on the day we recorded it, does today's code still read it correctly?*

Re-record after an upstream change, and the diff on the fixture is the change.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/record_fixtures.py
    PYTHONPATH=. .venv/bin/python scripts/record_fixtures.py --route DEL:COK
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date, timedelta

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

#: Recorded deliberately: a trunk route, a route that exposed the metadata bug,
#: and a pair with no service at all. The last one matters most — "nothing
#: flies here" and "we failed to read the answer" must stay distinguishable.
DEFAULT_ROUTES = [("DEL", "BOM"), ("DEL", "COK"), ("AGR", "BOM")]


def record_google_flights(src: str, dst: str, when: str) -> dict:
    from selectolax.lexbor import LexborHTMLParser

    from fast_flights import FlightQuery, Passengers, create_query
    from fast_flights.fetcher import fetch_flights_html

    query = create_query(
        flights=[FlightQuery(date=when, from_airport=src, to_airport=dst)],
        trip="one-way", seat="economy",
        passengers=Passengers(adults=2), currency="INR",
    )
    html = fetch_flights_html(query)
    script = LexborHTMLParser(html).css_first(r"script.ds\:1")
    if script is None:
        return {"route": f"{src}-{dst}", "recorded": when, "payload": None,
                "note": "no flight script in response"}

    raw = script.text().split("data:", 1)[1].rsplit(",", 1)[0]
    payload = json.loads(raw)

    # Store only the slots the parser actually reads. The full payload is
    # megabytes of unrelated page state, and a fixture nobody can read is a
    # fixture nobody maintains.
    flights = payload[3][0] if payload[3] else None
    return {
        "route": f"{src}-{dst}",
        "recorded": when,
        "has_flight_section": payload[3] is not None,
        "flight_count": len(flights) if flights else 0,
        # The metadata block whose absence broke the upstream parser.
        "metadata_present": bool(
            len(payload) > 7 and payload[7] and len(payload[7]) > 1 and payload[7][1]
        ),
        "flights": flights[:3] if flights else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", action="append", default=None,
                        help="SRC:DST, repeatable")
    parser.add_argument("--days-ahead", type=int, default=14)
    args = parser.parse_args()

    routes = (
        [tuple(r.split(":", 1)) for r in args.route] if args.route
        else DEFAULT_ROUTES
    )
    when = (date.today() + timedelta(days=args.days_ahead)).isoformat()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for src, dst in routes:
        print(f"recording {src}->{dst} on {when} …", end=" ", flush=True)
        try:
            data = record_google_flights(src, dst, when)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        path = FIXTURES / f"google_flights_{src}_{dst}.json"
        path.write_text(json.dumps(data, indent=1))
        print(f"{data['flight_count']} flight(s) → {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
