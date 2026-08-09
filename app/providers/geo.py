"""Deterministic city -> IATA and city -> railway-station resolution.

The original design asked an LLM to produce IATA codes inside the same call
that extracted dates and preferences ("Also convert the source and destination
into IATA codes"). LLMs hallucinate airport codes confidently, and a wrong code
silently produces flights between the wrong cities.

``full_iata_codes.csv`` and ``Station_Name_8477.xlsx`` were already sitting in
the repo, unused. This module makes them the source of truth and uses the LLM
only as a last-resort fallback for cities absent from the dataset.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

from app.config import PROJECT_ROOT, settings
from app.core.cache import cached
from app.providers import llm as llm_provider

logger = logging.getLogger(__name__)

IATA_CSV = PROJECT_ROOT / "full_iata_codes.csv"
STATIONS_XLSX = PROJECT_ROOT / "Station_Name_8477.xlsx"

# Railway station category ranking: NSG1 are the big terminals, HG/SG are
# suburban and halt stations we never want to route long-distance travel through.
_STATION_PRIORITY = {
    "NSG1": 1, "NSG2": 2, "NSG3": 3, "NSG4": 4, "NSG5": 5, "NSG6": 6,
    "SG1": 7, "SG2": 8, "SG3": 9,
    "HG1": 10, "HG2": 11, "HG3": 12,
}

# Cities whose railway/airport naming does not match the colloquial city name.
_CITY_ALIASES = {
    "bangalore": "bengaluru",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "poona": "pune",
    "trivandrum": "thiruvananthapuram",
    "cochin": "kochi",
    "mysore": "mysuru",
    "baroda": "vadodara",
    "gurgaon": "gurugram",
    "pondicherry": "puducherry",
    "benares": "varanasi",
    "banaras": "varanasi",
}


def normalise_city(name: str) -> str:
    """Fold accents, unify unicode dashes, collapse whitespace, lowercase."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # The station dataset uses U+2010 HYPHEN, not ASCII '-'. Left unhandled this
    # breaks both matching and URL slug construction.
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[^A-Za-z0-9\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return _CITY_ALIASES.get(text, text)


# ---------------------------------------------------------------------------
# IATA
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _iata_table() -> dict[str, str]:
    try:
        df = pd.read_csv(IATA_CSV)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", IATA_CSV, exc)
        return {}
    table: dict[str, str] = {}
    for _, row in df.iterrows():
        city = normalise_city(row.get("City", ""))
        code = str(row.get("IATA Code", "")).strip().upper()
        if city and len(code) == 3:
            table[city] = code
    for alias, target in _CITY_ALIASES.items():
        if target in table and alias not in table:
            table[alias] = table[target]
    return table


def lookup_iata(city: str) -> Optional[str]:
    """Exact/prefix lookup against the bundled airport dataset."""
    key = normalise_city(city)
    if not key:
        return None
    table = _iata_table()
    if key in table:
        return table[key]
    for name, code in table.items():
        if name.startswith(key) or key.startswith(name):
            return code
    return None


@cached("geo:iata", ttl=settings.ttl_iata)
async def resolve_airport(city: str) -> dict[str, Any]:
    """Resolve a city to a usable airport.

    Returns ``{city, iata, airport_city, exact}`` where ``airport_city`` is the
    city whose airport will actually be used — the original code called this
    ``nearby_src_city``/``nearby_dest_city`` and relied entirely on the LLM.
    """
    code = lookup_iata(city)
    if code:
        return {
            "city": city,
            "iata": code,
            "airport_city": city,
            "exact": True,
            "source": "dataset",
        }

    # Not in the dataset — ask the LLM for the nearest served airport, then
    # validate its answer back against the dataset so a hallucinated code
    # cannot reach the flight search.
    prompt = (
        f"The Indian city '{city}' may not have its own airport.\n"
        "Reply with JSON only: "
        '{"airport_city": "<nearest city with a commercial airport>", '
        '"iata": "<3-letter IATA code>"}'
    )
    try:
        data = await llm_provider.ask_json(prompt)
        airport_city = str(data.get("airport_city", "")).strip()
        iata = str(data.get("iata", "")).strip().upper()
        verified = lookup_iata(airport_city)
        if verified:
            iata, source = verified, "llm+dataset"
        elif re.fullmatch(r"[A-Z]{3}", iata):
            source = "llm"
        else:
            raise ValueError(f"unusable IATA {iata!r}")
        return {
            "city": city,
            "iata": iata,
            "airport_city": airport_city or city,
            "exact": False,
            "source": source,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Airport resolution failed for %s: %s", city, exc)
        return {
            "city": city,
            "iata": None,
            "airport_city": city,
            "exact": False,
            "source": "unresolved",
        }


# ---------------------------------------------------------------------------
# Railway stations
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _stations() -> pd.DataFrame:
    """Load the station master once, with normalised column names.

    The original read ``'Station_name_8477.xlsx'`` (lowercase n) at module
    import time. That works on case-insensitive macOS and fails on Linux/Docker,
    where it took the whole module down at import.
    """
    path = STATIONS_XLSX
    if not path.exists():
        # Case-insensitive rescue so a rename cannot break the build again.
        for candidate in PROJECT_ROOT.glob("*.xlsx"):
            if candidate.name.lower() == path.name.lower():
                path = candidate
                break
    try:
        df = pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Station dataset unavailable (%s): train search disabled.", exc)
        return pd.DataFrame(
            columns=["Station Name", "Station Code", "New Station Category", "District"]
        )

    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    df["Station Name"] = df["Station Name"].astype(str)
    df["Station Code"] = df["Station Code"].astype(str).str.strip().str.upper()
    df["_name_norm"] = df["Station Name"].map(normalise_city)
    df["_district_norm"] = df["District"].astype(str).map(normalise_city)
    df["_priority"] = (
        df["New Station Category"].astype(str).str.strip().map(_STATION_PRIORITY)
    ).fillna(99)
    return df


def lookup_stations(city: str, limit: int = 3) -> list[dict[str, str]]:
    """Return the most important stations for a city, best first."""
    key = normalise_city(city)
    if not key:
        return []
    df = _stations()
    if df.empty:
        return []

    pattern = r"\b" + re.escape(key) + r"\b"
    matches = df[
        df["_name_norm"].str.contains(pattern, na=False, regex=True)
        | df["_district_norm"].str.contains(pattern, na=False, regex=True)
    ]
    if matches.empty:
        matches = df[df["_district_norm"] == key]
    if matches.empty:
        matches = df[df["_name_norm"].str.startswith(key, na=False)]
    if matches.empty:
        return []

    ordered = matches.sort_values(["_priority", "Station Name"])
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for _, row in ordered.iterrows():
        code = row["Station Code"]
        if code in seen or not code or code == "NAN":
            continue
        seen.add(code)
        out.append(
            {
                "name": row["Station Name"].strip(),
                "code": code,
                "category": str(row.get("New Station Category", "")).strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


def station_slug(name: str, code: str) -> str:
    """Build the etrain.info URL fragment for a station."""
    cleaned = unicodedata.normalize("NFKD", name)
    cleaned = re.sub(r"[‐-―−]", "-", cleaned)
    cleaned = cleaned.replace(".", "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9\- ]+", "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned).upper().strip("-")
    return f"{cleaned}-{code.upper()}"
