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
STATIONS_SUPPLEMENT = PROJECT_ROOT / "data" / "stations_supplement.csv"
IATA_SUPPLEMENT = PROJECT_ROOT / "data" / "iata_supplement.csv"

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
    # The bundled CSV holds 98 airports and is missing several obvious ones —
    # Jaipur among them, so a very ordinary destination fell through to an LLM
    # guess in a module whose whole purpose is not guessing.
    if IATA_SUPPLEMENT.exists():
        try:
            df = pd.concat([df, pd.read_csv(IATA_SUPPLEMENT)], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IATA supplement unreadable (%s); continuing.", exc)

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


# Well-known Indian destinations with no usable airport of their own, and the
# gateway people actually use. This exists for three reasons:
#
#   1. It removes the LLM from the common path. When Groq's daily token cap ran
#      out mid-run, ``resolve_airport`` returned "unresolved", the strategy node
#      skipped flights entirely, and a Gangtok→Kochi trip lost every flight,
#      every connecting leg and its whole budget. A quota should degrade the
#      prose, not delete the itinerary.
#   2. The LLM's answer is sometimes technically right and practically useless:
#      it offered Agra (AGR) for Aligarh, and AGR has almost no scheduled
#      service. Delhi is the honest answer.
#   3. It is faster and free.
#
# ``railhead`` is the long-distance station people actually board at.
_GATEWAYS: dict[str, dict[str, str]] = {
    "shimla":      {"airport_city": "Chandigarh",  "railhead": "Kalka"},
    "manali":      {"airport_city": "Bhuntar",     "railhead": "Chandigarh"},
    "kasol":       {"airport_city": "Bhuntar",     "railhead": "Chandigarh"},
    "dharamshala": {"airport_city": "Dharamshala", "railhead": "Pathankot"},
    "gangtok":     {"airport_city": "Bagdogra",    "railhead": "New Jalpaiguri"},
    "darjeeling":  {"airport_city": "Bagdogra",    "railhead": "New Jalpaiguri"},
    "munnar":      {"airport_city": "Kochi",       "railhead": "Aluva"},
    "alleppey":    {"airport_city": "Kochi",       "railhead": "Alappuzha"},
    "kumarakom":   {"airport_city": "Kochi",       "railhead": "Kottayam"},
    "varkala":     {"airport_city": "Thiruvananthapuram", "railhead": "Varkala"},
    "wayanad":     {"airport_city": "Kozhikode",   "railhead": "Kozhikode"},
    "coorg":       {"airport_city": "Mangalore",   "railhead": "Mysuru"},
    "madikeri":    {"airport_city": "Mangalore",   "railhead": "Mysuru"},
    "chikmagalur": {"airport_city": "Mangalore",   "railhead": "Hubballi"},
    "hampi":       {"airport_city": "Hubli",       "railhead": "Hubballi"},
    "gokarna":     {"airport_city": "Goa",         "railhead": "Madgaon"},
    "ooty":        {"airport_city": "Coimbatore",  "railhead": "Coimbatore"},
    "udhagamandalam": {"airport_city": "Coimbatore", "railhead": "Coimbatore"},
    "kodaikanal":  {"airport_city": "Madurai",     "railhead": "Madurai"},
    "mussoorie":   {"airport_city": "Dehradun",    "railhead": "Dehradun"},
    "rishikesh":   {"airport_city": "Dehradun",    "railhead": "Haridwar"},
    "haridwar":    {"airport_city": "Dehradun",    "railhead": "Haridwar"},
    "auli":        {"airport_city": "Dehradun",    "railhead": "Dehradun"},
    "joshimath":   {"airport_city": "Dehradun",    "railhead": "Dehradun"},
    "nainital":    {"airport_city": "Pantnagar",   "railhead": "Kathgodam"},
    "almora":      {"airport_city": "Pantnagar",   "railhead": "Kathgodam"},
    "mount abu":   {"airport_city": "Udaipur",     "railhead": "Abu Road"},
    "pushkar":     {"airport_city": "Jaipur",      "railhead": "Ajmer"},
    "ajmer":       {"airport_city": "Jaipur",      "railhead": "Ajmer"},
    "jaisalmer":   {"airport_city": "Jodhpur",     "railhead": "Jaisalmer"},
    # Agra's airport carries almost no scheduled traffic; Delhi is the real one.
    "aligarh":     {"airport_city": "Delhi",       "railhead": "Aligarh"},
    "mathura":     {"airport_city": "Delhi",       "railhead": "Mathura"},
    "vrindavan":   {"airport_city": "Delhi",       "railhead": "Mathura"},
    "shirdi":      {"airport_city": "Shirdi",      "railhead": "Ahmednagar"},
    "mahabaleshwar": {"airport_city": "Pune",      "railhead": "Pune"},
    "lonavala":    {"airport_city": "Pune",        "railhead": "Lonavala"},
    "panchgani":   {"airport_city": "Pune",        "railhead": "Pune"},
    "ranthambore": {"airport_city": "Jaipur",      "railhead": "Sawai Madhopur"},
    "khajuraho":   {"airport_city": "Khajuraho",   "railhead": "Khajuraho"},
    "spiti":       {"airport_city": "Bhuntar",     "railhead": "Shimla"},
    "leh":         {"airport_city": "Leh",         "railhead": "Jammu"},
    "patiala":     {"airport_city": "Chandigarh",  "railhead": "Patiala"},
}


def gateway_for(city: str) -> Optional[dict[str, str]]:
    return _GATEWAYS.get(normalise_city(city))


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

    # A curated gateway beats an LLM guess: deterministic, instant, and it
    # still works when the model is rate-limited.
    gateway = gateway_for(city)
    if gateway:
        code = lookup_iata(gateway["airport_city"])
        if code:
            return {
                "city": city,
                "iata": code,
                "airport_city": gateway["airport_city"],
                "exact": False,
                "source": "gateway_table",
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

    # The bundled spreadsheet has 8,477 rows and yet almost no Kerala coverage —
    # "Kozhikkode" is the only entry for the entire state, so Kochi, Thrissur,
    # Alappuzha and Thiruvananthapuram all resolved to "no railway station in the
    # dataset". Also missing: Madgaon (the Goa railhead), Hubballi, Belagavi,
    # Udupi and Tiruchirappalli. This supplement fills those in rather than
    # reporting a data gap as though no trains ran there.
    if STATIONS_SUPPLEMENT.exists():
        try:
            extra = pd.read_csv(STATIONS_SUPPLEMENT)
            missing = [c for c in df.columns if c not in extra.columns]
            for column in missing:
                extra[column] = None
            df = pd.concat([df, extra[df.columns]], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Station supplement unreadable (%s); continuing.", exc)

    df["Station Name"] = df["Station Name"].astype(str)
    df["Station Code"] = df["Station Code"].astype(str).str.strip().str.upper()
    df["_name_norm"] = df["Station Name"].map(normalise_city)
    df["_district_norm"] = df["District"].astype(str).map(normalise_city)
    df["_priority"] = (
        df["New Station Category"].astype(str).str.strip().map(_STATION_PRIORITY)
    ).fillna(99)
    return df


# Cities whose main long-distance terminus sits in a *neighbouring* town and so
# never matches on name or district. Kolkata is the costly one: its top matches
# are Sealdah and Kolkata station, while most western- and northern-bound
# expresses — including the only direct Jaisalmer service — leave from Howrah,
# which the dataset files under its own district. Without this, a perfectly
# ordinary Kolkata→Jaisalmer request answered "no direct trains".
# Kept deliberately short: only cities whose principal long-distance terminus is
# genuinely filed under another name. Anything else is left to
# ``resolve_railhead``, because padding this map dilutes the category ranking
# and pushes a city's own terminals down the list.
_CITY_EXTRA_STATION_CITIES = {
    "kolkata": ["howrah"],
    "hyderabad": ["secunderabad"],
    "delhi": ["hazrat nizamuddin"],
    "kochi": ["ernakulam"],
}


# Colloquial spellings that differ from the railway's own. Deliberately kept
# separate from ``_CITY_ALIASES``: that map feeds airport lookup too, and the
# IATA dataset spells several of these the *other* way round — "hubli" has an
# airport entry while "hubballi" does not, so folding them globally would fix
# the station and break the flight.
_STATION_CITY_ALIASES = {
    "trichy": "tiruchchirappalli",
    "tiruchirappalli": "tiruchchirappalli",
    "hubli": "hubballi",
    "belgaum": "belagavi",
    "calicut": "kozhikode",
    "alleppey": "alappuzha",
    "quilon": "kollam",
    "ernakulam": "ernakulam",
}


def _matching_rows(df: Any, key: str) -> Any:
    pattern = r"\b" + re.escape(key) + r"\b"
    return df[
        df["_name_norm"].str.contains(pattern, na=False, regex=True)
        | df["_district_norm"].str.contains(pattern, na=False, regex=True)
    ]


def lookup_stations(city: str, limit: int = 3) -> list[dict[str, str]]:
    """Return the most important stations for a city, best first."""
    key = normalise_city(city)
    if not key:
        return []
    key = _STATION_CITY_ALIASES.get(key, key)
    df = _stations()
    if df.empty:
        return []

    matches = _matching_rows(df, key)

    # Fold in the neighbouring terminus, then let the category ranking below
    # decide the order — Howrah is NSG1 and correctly outranks Sealdah's
    # suburban siblings on its own merits.
    for extra in _CITY_EXTRA_STATION_CITIES.get(key, []):
        extra_rows = _matching_rows(df, normalise_city(extra))
        if not extra_rows.empty:
            matches = pd.concat([matches, extra_rows])
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


@cached("geo:railhead", ttl=settings.ttl_iata)
async def resolve_railhead(city: str) -> dict[str, Any]:
    """Resolve a city to a usable railway station, borrowing a neighbour's if needed.

    ``lookup_stations`` returns ``[]`` for places with no railway at all —
    Gangtok, Coorg, Munnar, most of the hills. The train provider then answered
    "no railway station found in the dataset", which is true but useless: the
    real answer is "the railhead is New Jalpaiguri, 125 km away". This mirrors
    ``resolve_airport``: dataset first, LLM only for what the dataset cannot
    know, and the LLM's answer validated back against the dataset.
    """
    own = lookup_stations(city, limit=3)
    if own:
        return {
            "city": city,
            "station_city": city,
            "stations": own,
            "exact": True,
            "source": "dataset",
        }

    gateway = gateway_for(city)
    if gateway:
        stations = lookup_stations(gateway["railhead"], limit=3)
        if stations:
            return {
                "city": city,
                "station_city": gateway["railhead"],
                "stations": stations,
                "exact": False,
                "source": "gateway_table",
            }

    prompt = (
        f"The Indian town '{city}' has no railway station of its own.\n"
        "Which nearby city is its usual railhead for long-distance trains?\n"
        'Reply with JSON only: {"railhead_city": "<city name>"}'
    )
    try:
        data = await llm_provider.ask_json(prompt)
        railhead = str(data.get("railhead_city", "")).strip()
        stations = lookup_stations(railhead, limit=3) if railhead else []
        if not stations:
            raise ValueError(f"no station found for suggested railhead {railhead!r}")
        return {
            "city": city,
            "station_city": railhead,
            "stations": stations,
            "exact": False,
            "source": "llm+dataset",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Railhead resolution failed for %s: %s", city, exc)
        return {
            "city": city,
            "station_city": None,
            "stations": [],
            "exact": False,
            "source": "unresolved",
        }


def station_slug(name: str, code: str) -> str:
    """Build the etrain.info URL fragment for a station."""
    cleaned = unicodedata.normalize("NFKD", name)
    cleaned = re.sub(r"[‐-―−]", "-", cleaned)
    cleaned = cleaned.replace(".", "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9\- ]+", "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned).upper().strip("-")
    return f"{cleaned}-{code.upper()}"
