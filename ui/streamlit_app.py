"""Streamlit UI — a thin client over the FastAPI backend.

What changed and why
--------------------
The old ``app.py`` imported the graph, the LLM and every provider directly, then
ran the whole 60-120 s pipeline inside a Streamlit script run. That meant:

  * business logic lived in the UI process and could not be reused by anything
    else (no API, no worker, no tests);
  * ``st.*`` calls were made from inside graph nodes, so the graph could not run
    anywhere except Streamlit;
  * every widget interaction re-ran the script from the top, rebuilding the
    graph and the checkpointer;
  * the human-review step could never work, because Streamlit reruns and
    synchronous graph invocation are fundamentally incompatible;
  * the budget table was hardcoded fiction and the "Download PDF" button
    returned the literal string "This would be the PDF content in a real app".

Now the UI only does HTTP. It submits a trip, polls status, renders results and
posts review decisions. All of which a mobile app or a CLI could do identically.

On the presentation layer
-------------------------
Streamlit's stock widgets are used for everything *interactive* (forms, tabs,
buttons) because they carry the state handling. Everything *read-only* —
transport options, weather, hotels, the cost strip — is rendered as our own HTML
so the visual design does not depend on Streamlit's internal class names, which
change between releases. Provider strings are escaped before they reach that
HTML; they come from scraped pages and an LLM, so they are untrusted.
"""

from __future__ import annotations

import html
import os
import time
from datetime import date, timedelta
from typing import Any, Iterable, Optional

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 300

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# One place for every colour, radius and shadow. Custom components below use
# these variables exclusively, so a palette change is a single edit here.

THEME = """
<style>
  :root {
    --bg:          #0B0F17;
    --surface:     #131A26;
    --surface-2:   #1A2333;
    --surface-3:   #212C40;
    --border:      rgba(255,255,255,.09);
    --border-firm: rgba(255,255,255,.16);
    --text:        #E8EDF5;
    --muted:       #8B98AD;
    --faint:       #5C6880;

    --accent:      #FF6B4A;
    --accent-soft: rgba(255,107,74,.14);
    --indigo:      #7C6BFF;
    --success:     #34D399;
    --warning:     #FBBF24;
    --danger:      #F87171;

    --r-sm: 8px;
    --r-md: 14px;
    --r-lg: 20px;
    --shadow: 0 8px 30px rgba(0,0,0,.35);
  }

  /* ---- page chrome ---------------------------------------------------- */
  .stApp { background: var(--bg); }
  #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
  .block-container { padding-top: 2.2rem; max-width: 1280px; }

  [data-testid="stSidebar"] {
    min-width: 372px;
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

  /* ---- hero ----------------------------------------------------------- */
  .tp-hero {
    position: relative;
    border-radius: var(--r-lg);
    padding: 2.1rem 2.2rem 1.9rem;
    margin-bottom: 1.5rem;
    overflow: hidden;
    background:
      radial-gradient(1100px 380px at 12% -30%, rgba(124,107,255,.42), transparent 62%),
      radial-gradient(900px 420px at 88% 130%, rgba(255,107,74,.40), transparent 60%),
      linear-gradient(135deg, #161C2E 0%, #1D1B36 48%, #2A1B33 100%);
    border: 1px solid var(--border-firm);
    box-shadow: var(--shadow);
  }
  .tp-hero-brand {
    font-size: .74rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: rgba(255,255,255,.62);
    margin-bottom: .7rem;
  }
  .tp-hero-route {
    font-size: 2.6rem; font-weight: 750; line-height: 1.08;
    letter-spacing: -.02em; color: #fff; margin: 0 0 .55rem;
  }
  .tp-hero-route .arr { color: var(--accent); margin: 0 .5rem; }
  .tp-hero-meta { color: rgba(255,255,255,.74); font-size: .96rem; }
  .tp-hero-right { text-align: right; }
  .tp-hero-total {
    font-size: 2.05rem; font-weight: 750; color: #fff;
    letter-spacing: -.02em; line-height: 1.1;
  }
  .tp-hero-pp { color: rgba(255,255,255,.66); font-size: .87rem; }

  /* ---- status pill ---------------------------------------------------- */
  .tp-pill {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .3rem .78rem; border-radius: 999px;
    font-size: .77rem; font-weight: 650; letter-spacing: .02em;
    border: 1px solid var(--border-firm); background: rgba(255,255,255,.07);
    color: rgba(255,255,255,.9);
  }
  .tp-pill .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--muted);
  }
  .tp-pill.ok      .dot { background: var(--success); box-shadow: 0 0 9px var(--success); }
  .tp-pill.wait    .dot { background: var(--warning); box-shadow: 0 0 9px var(--warning); }
  .tp-pill.err     .dot { background: var(--danger);  box-shadow: 0 0 9px var(--danger); }
  .tp-pill.run     .dot { background: var(--indigo);  box-shadow: 0 0 9px var(--indigo); }

  /* ---- stat strip ----------------------------------------------------- */
  .tp-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem;
              margin: 0 0 1.4rem; }
  .tp-stat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r-md); padding: .95rem 1.05rem;
  }
  .tp-stat.accent {
    background: linear-gradient(160deg, var(--accent-soft), rgba(255,107,74,.04));
    border-color: rgba(255,107,74,.35);
  }
  .tp-stat-label {
    font-size: .69rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: var(--faint); margin-bottom: .38rem;
  }
  .tp-stat-value { font-size: 1.5rem; font-weight: 700; letter-spacing: -.02em;
                   color: var(--text); line-height: 1.15; }
  .tp-stat-sub { font-size: .78rem; color: var(--muted); margin-top: .2rem; }

  /* ---- generic card --------------------------------------------------- */
  .tp-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r-md); padding: 1rem 1.15rem; margin-bottom: .7rem;
    transition: border-color .16s ease, transform .16s ease;
  }
  .tp-card:hover { border-color: var(--border-firm); }
  .tp-card.best {
    border-color: rgba(255,107,74,.42);
    background: linear-gradient(155deg, var(--accent-soft), var(--surface) 62%);
  }

  .tp-row { display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; flex-wrap: wrap; }
  .tp-badge {
    display: inline-block; font-size: .64rem; font-weight: 750;
    letter-spacing: .11em; text-transform: uppercase;
    padding: .2rem .5rem; border-radius: 5px;
    background: var(--accent); color: #150A06; margin-bottom: .5rem;
  }
  .tp-badge.muted { background: var(--surface-3); color: var(--muted); }

  .tp-title-row { font-size: 1.06rem; font-weight: 680; color: var(--text); }
  .tp-sub-row  { font-size: .84rem; color: var(--muted); margin-top: .22rem; }
  .tp-price    { font-size: 1.32rem; font-weight: 720; color: var(--text);
                 letter-spacing: -.01em; white-space: nowrap; }
  .tp-price-sub{ font-size: .74rem; color: var(--faint); text-align: right; }

  /* timeline: dep ——— arr */
  .tp-leg { display: flex; align-items: center; gap: .7rem; margin-top: .6rem; }
  .tp-leg-time { font-size: 1.02rem; font-weight: 660; color: var(--text); }
  .tp-leg-code { font-size: .72rem; color: var(--faint); letter-spacing: .06em; }
  .tp-leg-line {
    flex: 1; height: 1px; background: var(--border-firm); position: relative;
    min-width: 46px;
  }
  .tp-leg-line::after {
    content: ""; position: absolute; right: -1px; top: -2.5px;
    width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  }
  .tp-leg-mid { font-size: .72rem; color: var(--muted); text-align: center;
                white-space: nowrap; }

  /* chips */
  .tp-chips { display: flex; flex-wrap: wrap; gap: .32rem; margin-top: .6rem; }
  .tp-chip {
    font-size: .72rem; color: var(--muted); background: var(--surface-2);
    border: 1px solid var(--border); border-radius: 999px;
    padding: .18rem .55rem;
  }

  /* ---- weather -------------------------------------------------------- */
  .tp-wx { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px,1fr));
           gap: .7rem; }
  .tp-wx-card {
    background: linear-gradient(165deg, var(--surface-2), var(--surface));
    border: 1px solid var(--border); border-radius: var(--r-md);
    padding: 1rem .9rem; text-align: center;
  }
  .tp-wx-day { font-size: .73rem; font-weight: 700; letter-spacing: .1em;
               text-transform: uppercase; color: var(--faint); }
  .tp-wx-icon { font-size: 1.9rem; margin: .45rem 0 .3rem; }
  .tp-wx-temp { font-size: 1.55rem; font-weight: 700; color: var(--text); }
  .tp-wx-min  { font-size: .8rem; color: var(--muted); }
  .tp-wx-cond { font-size: .78rem; color: var(--muted); margin-top: .35rem;
                min-height: 1.1em; }
  .tp-wx-rain { font-size: .72rem; color: #6FB6FF; margin-top: .3rem; }
  .tp-wx-card.na { opacity: .6; }

  /* ---- review panel --------------------------------------------------- */
  .tp-review-head {
    border-radius: var(--r-md) var(--r-md) 0 0;
    border: 1px solid rgba(255,107,74,.35); border-bottom: none;
    background: linear-gradient(135deg, rgba(255,107,74,.18), rgba(124,107,255,.12));
    padding: 1rem 1.2rem .9rem;
  }
  .tp-review-title { font-size: 1.12rem; font-weight: 700; color: #fff; }
  .tp-review-q { color: rgba(255,255,255,.8); font-size: .93rem; margin-top: .25rem; }

  /* ---- section heading ------------------------------------------------ */
  .tp-sec {
    font-size: .72rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: var(--faint);
    margin: 1.3rem 0 .6rem;
  }
  .tp-note {
    background: var(--surface); border: 1px dashed var(--border-firm);
    border-radius: var(--r-md); padding: 1.1rem 1.2rem;
    color: var(--muted); font-size: .9rem;
  }

  /* ---- itinerary ------------------------------------------------------ */
  .tp-itin h2 {
    font-size: 1.16rem; font-weight: 700; color: var(--text);
    margin: 1.5rem 0 .55rem; padding-left: .8rem;
    border-left: 3px solid var(--accent);
  }
  .tp-itin h2:first-child { margin-top: .2rem; }
  .tp-itin ul { margin: .3rem 0 .9rem 1.1rem; }
  .tp-itin li { margin-bottom: .34rem; color: var(--muted); }
  .tp-itin li strong, .tp-itin p strong { color: var(--text); }
  .tp-itin p { color: var(--muted); }

  /* ---- streamlit widget restyling ------------------------------------- */
  .stButton > button, .stFormSubmitButton > button,
  .stDownloadButton > button {
    border-radius: 10px; font-weight: 620; border: 1px solid var(--border-firm);
    transition: transform .12s ease, filter .12s ease;
  }
  .stButton > button:hover, .stFormSubmitButton > button:hover,
  .stDownloadButton > button:hover { transform: translateY(-1px); filter: brightness(1.08); }
  .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), #FF8A63);
    border: none; color: #150A06;
  }

  .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--border); }
  .stTabs [data-baseweb="tab"] {
    border-radius: 9px 9px 0 0; padding: .5rem .95rem; font-weight: 600;
    color: var(--muted);
  }
  .stTabs [aria-selected="true"] { background: var(--surface); color: var(--text); }

  [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: 9px !important;
  }

  /* backend status dots in the sidebar */
  .tp-dots { display: flex; flex-wrap: wrap; gap: .3rem; margin: .1rem 0 .3rem; }
  .tp-dot {
    display: inline-flex; align-items: center; gap: .32rem;
    font-size: .7rem; color: var(--muted);
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: .17rem .5rem;
  }
  .tp-dot i { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
  .tp-dot.up i   { background: var(--success); }
  .tp-dot.down i { background: var(--danger); }
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def esc(value: Any, fallback: str = "—") -> str:
    """Escape anything before it goes near our HTML.

    Provider data is scraped or LLM-generated, so it is untrusted input.
    """
    if value is None or value == "":
        return fallback
    return html.escape(str(value))


def money(value: Any, prefix: str = "₹") -> Optional[str]:
    try:
        return f"{prefix}{float(value):,.0f}"
    except (TypeError, ValueError):
        return None


def chips(values: Iterable[Any], limit: int = 5) -> str:
    items = [v for v in values if v][:limit]
    if not items:
        return ""
    inner = "".join(f'<span class="tp-chip">{esc(v)}</span>' for v in items)
    return f'<div class="tp-chips">{inner}</div>'


def leg(dep: Any, dep_code: Any, arr: Any, arr_code: Any, middle: str) -> str:
    """The departure ——— arrival timeline used by every transport card."""
    return (
        '<div class="tp-leg">'
        f'<div><div class="tp-leg-time">{esc(dep)}</div>'
        f'<div class="tp-leg-code">{esc(dep_code, "")}</div></div>'
        '<div class="tp-leg-line"></div>'
        f'<div class="tp-leg-mid">{esc(middle, "")}</div>'
        '<div class="tp-leg-line"></div>'
        f'<div style="text-align:right"><div class="tp-leg-time">{esc(arr)}</div>'
        f'<div class="tp-leg-code">{esc(arr_code, "")}</div></div>'
        "</div>"
    )


STATUS_PILL = {
    "completed":       ("ok",   "Completed"),
    "awaiting_review": ("wait", "Needs your input"),
    "running":         ("run",  "Planning"),
    "queued":          ("run",  "Queued"),
    "failed":          ("err",  "Failed"),
}


def status_pill(status: str) -> str:
    css, label = STATUS_PILL.get(status, ("", esc(status)))
    return f'<span class="tp-pill {css}"><span class="dot"></span>{label}</span>'


def weather_icon(condition: str) -> str:
    text = (condition or "").lower()
    if "thunder" in text or "storm" in text:
        return "⛈️"
    if "rain" in text or "drizzle" in text or "shower" in text:
        return "🌧️"
    if "snow" in text or "sleet" in text or "ice" in text:
        return "❄️"
    if "fog" in text or "mist" in text or "haze" in text:
        return "🌫️"
    if "overcast" in text:
        return "☁️"
    if "cloud" in text:
        return "⛅"
    if "clear" in text or "sunny" in text:
        return "☀️"
    return "🌤️"


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def api(method: str, path: str, **kwargs: Any) -> tuple[bool, Any]:
    """Every call returns (ok, payload) — no exceptions leak into the UI."""
    try:
        response = requests.request(
            method, f"{API_BASE}{path}", timeout=kwargs.pop("timeout", 30), **kwargs
        )
    except requests.exceptions.ConnectionError:
        return False, f"Cannot reach the API at {API_BASE}. Is it running?"
    except requests.exceptions.Timeout:
        return False, "The API did not respond in time."
    except requests.exceptions.RequestException as exc:
        return False, str(exc)

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:  # noqa: BLE001
            detail = response.text[:300]
        return False, f"HTTP {response.status_code}: {detail}"

    try:
        return True, response.json()
    except ValueError:
        return True, response.text


def backend_online() -> tuple[bool, dict]:
    ok, body = api("GET", "/health", timeout=5)
    return ok, (body if isinstance(body, dict) else {})


# ---------------------------------------------------------------------------
# Rendering — transport
# ---------------------------------------------------------------------------

def _flight_card(item: dict, best: bool) -> str:
    price = money(item.get("price_inr"))
    carriers = ", ".join(item.get("carriers") or []) or "Flight"
    stops = item.get("stops")
    stop_text = "non-stop" if stops == 0 else (item.get("layover_pretty") or f"{stops} stop(s)")

    extras = []
    if item.get("aircraft"):
        extras.append(item["aircraft"])
    if item.get("carbon_kg"):
        extras.append(f"{item['carbon_kg']:.0f} kg CO₂")
    if item.get("source"):
        extras.append(str(item["source"]).replace("_", " "))

    right = (
        f'<div><div class="tp-price">{price}</div>'
        '<div class="tp-price-sub">per person</div></div>'
        if price else '<div class="tp-price-sub">price unavailable</div>'
    )

    return (
        f'<div class="tp-card {"best" if best else ""}">'
        + ('<span class="tp-badge">Best price</span>' if best else "")
        + '<div class="tp-row">'
        f'<div><div class="tp-title-row">{esc(carriers)}</div>'
        f'<div class="tp-sub-row">{esc(item.get("origin_airport"), "")} → '
        f'{esc(item.get("destination_airport"), "")}</div></div>'
        f"{right}</div>"
        + leg(
            item.get("departure", "")[-5:] or "—", item.get("origin_iata"),
            item.get("arrival_time") or (item.get("arrival", "")[-5:] or "—"),
            item.get("destination_iata"),
            f'{item.get("duration") or ""} · {stop_text}',
        )
        + chips(extras)
        + "</div>"
    )


def _train_card(item: dict, best: bool) -> str:
    return (
        f'<div class="tp-card {"best" if best else ""}">'
        '<div class="tp-row">'
        f'<div><div class="tp-title-row">{esc(item.get("train_name"), "Train")}</div>'
        f'<div class="tp-sub-row">#{esc(item.get("train_no"), "")} · '
        f'{esc(item.get("source_station"), "")} → {esc(item.get("dest_station"), "")}</div></div>'
        '<div class="tp-price-sub">fare at booking</div>'
        "</div>"
        + leg(
            item.get("departs"), item.get("from"),
            item.get("arrives"), item.get("to"),
            item.get("duration") or "",
        )
        + "</div>"
    )


def _bus_card(item: dict, best: bool) -> str:
    price = money(item.get("fare_inr"))
    seats = item.get("seats_available")
    extras = []
    if item.get("bus_type"):
        extras.append(item["bus_type"])
    if seats is not None:
        extras.append(f"{seats} seats left")

    right = (
        f'<div><div class="tp-price">{price}</div>'
        '<div class="tp-price-sub">per seat</div></div>'
        if price else '<div class="tp-price-sub">price unavailable</div>'
    )
    return (
        f'<div class="tp-card {"best" if best else ""}">'
        + ('<span class="tp-badge">Best price</span>' if best else "")
        + '<div class="tp-row">'
        f'<div><div class="tp-title-row">{esc(item.get("operator"), "Bus")}</div></div>'
        f"{right}</div>"
        + leg(item.get("departure"), "", item.get("arrival"), "", "")
        + chips(extras)
        + "</div>"
    )


#: mode -> (list key, card renderer, key to minimise for the "best price" badge)
MODE_SPEC = {
    "flight": ("flights", _flight_card, "price_inr"),
    "train":  ("trains",  _train_card,  None),
    "bus":    ("buses",   _bus_card,    "fare_inr"),
}


def render_transport(trip: dict) -> None:
    strategy = trip.get("transport_strategy") or {}
    if strategy.get("modes"):
        st.caption(
            f"Searched **{', '.join(strategy['modes'])}** — {strategy.get('reason', '')}"
        )

    health = trip.get("provider_health") or {}
    failed = {k: v for k, v in health.items() if v != "ok"}
    if failed:
        st.warning(
            "Some providers were unavailable: "
            + "; ".join(f"**{k}** ({v})" for k, v in failed.items())
        )

    if trip.get("transport_summary"):
        st.markdown('<div class="tp-sec">Recommendation</div>', unsafe_allow_html=True)
        st.markdown(trip["transport_summary"])
    else:
        st.markdown(
            '<div class="tp-note">No transport summary yet.</div>',
            unsafe_allow_html=True,
        )

    options = trip.get("transport_options") or {}
    if not options:
        return

    st.markdown('<div class="tp-sec">All options found</div>', unsafe_allow_html=True)
    labels = {"flight": "✈️ Flights", "train": "🚆 Trains", "bus": "🚌 Buses"}
    tabs = st.tabs([labels.get(m, m.title()) for m in options])

    for tab, (mode, data) in zip(tabs, options.items()):
        with tab:
            list_key, renderer, price_key = MODE_SPEC.get(
                mode, ("options", None, None)
            )
            items = data.get(list_key) or data.get("options") or []
            if not items:
                st.markdown(
                    f'<div class="tp-note">{esc(data.get("note"), "No options found.")}</div>',
                    unsafe_allow_html=True,
                )
                continue

            # Which row earns the "Best price" badge. Only meaningful where the
            # provider actually returns a fare — trains come back without one.
            best_index = -1
            if price_key:
                priced = [
                    (i, item.get(price_key)) for i, item in enumerate(items)
                    if isinstance(item.get(price_key), (int, float))
                ]
                if priced:
                    best_index = min(priced, key=lambda pair: pair[1])[0]

            if renderer is None:
                st.dataframe(items, width="stretch", hide_index=True)
                continue

            st.markdown(
                "".join(
                    renderer(item, index == best_index)
                    for index, item in enumerate(items)
                ),
                unsafe_allow_html=True,
            )

            if data.get("price_disclaimer"):
                st.caption(data["price_disclaimer"])


# ---------------------------------------------------------------------------
# Rendering — weather, hotels, cost
# ---------------------------------------------------------------------------

def render_weather(days: list[dict]) -> None:
    if not days:
        st.markdown(
            '<div class="tp-note">Weather data was not available for this trip. '
            "Forecasts only reach ~14 days ahead; further-out days are reported "
            "as beyond the horizon rather than guessed.</div>",
            unsafe_allow_html=True,
        )
        return

    cards = []
    for day in days:
        date_label = esc((day.get("date") or "")[5:], "")
        if not day.get("available", True):
            cards.append(
                '<div class="tp-wx-card na">'
                f'<div class="tp-wx-day">{esc(day.get("day_name"), "")}</div>'
                '<div class="tp-wx-icon">🗓️</div>'
                '<div class="tp-wx-temp">—</div>'
                f'<div class="tp-wx-cond">{esc(day.get("condition"), "")}</div>'
                f'<div class="tp-wx-min">{date_label}</div>'
                "</div>"
            )
            continue

        condition = day.get("condition") or ""
        rain = day.get("rain_chance") or 0
        cards.append(
            '<div class="tp-wx-card">'
            f'<div class="tp-wx-day">{esc(day.get("day_name"), "")} · {date_label}</div>'
            f'<div class="tp-wx-icon">{weather_icon(condition)}</div>'
            f'<div class="tp-wx-temp">{esc(day.get("max_temp"))}°</div>'
            f'<div class="tp-wx-min">min {esc(day.get("min_temp"))}°</div>'
            f'<div class="tp-wx-cond">{esc(condition, "")}</div>'
            + (f'<div class="tp-wx-rain">💧 {esc(rain)}% rain</div>' if rain else "")
            + "</div>"
        )

    st.markdown(f'<div class="tp-wx">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_hotels(hotels: Optional[dict]) -> None:
    if not hotels or not hotels.get("hotels"):
        st.markdown(
            '<div class="tp-note">No hotel options were retrieved. '
            "Set <code>SERPAPI_KEY</code> to price real stays.</div>",
            unsafe_allow_html=True,
        )
        return

    st.caption(
        f"{hotels.get('nights')} night(s) · {hotels.get('adults')} adult(s) · "
        f"{hotels.get('checkin')} → {hotels.get('checkout')}"
    )

    rows = hotels["hotels"]
    cheapest = hotels.get("cheapest_total_inr")

    cards = []
    for hotel in rows:
        total = hotel.get("total_price_inr")
        nightly = hotel.get("nightly_price_inr")
        is_best = (
            cheapest is not None and total is not None and float(total) == float(cheapest)
        )

        meta = []
        if hotel.get("rating"):
            reviews = hotel.get("reviews")
            meta.append(
                f"⭐ {hotel['rating']}" + (f" ({reviews:,} reviews)" if reviews else "")
            )
        if hotel.get("hotel_class"):
            meta.append(str(hotel["hotel_class"]))

        price_block = (
            f'<div><div class="tp-price">{money(total)}</div>'
            f'<div class="tp-price-sub">{money(nightly) or "—"}/night</div></div>'
            if total else '<div class="tp-price-sub">price unavailable</div>'
        )

        cards.append(
            f'<div class="tp-card {"best" if is_best else ""}">'
            + ('<span class="tp-badge">Cheapest</span>' if is_best else "")
            + '<div class="tp-row">'
            f'<div style="flex:1;min-width:200px">'
            f'<div class="tp-title-row">{esc(hotel.get("name"))}</div>'
            f'<div class="tp-sub-row">{esc(" · ".join(meta), "")}</div></div>'
            f"{price_block}</div>"
            + chips(hotel.get("amenities") or [], limit=6)
            + "</div>"
        )

    st.markdown("".join(cards), unsafe_allow_html=True)

    # Booking links have to be real anchors, so they go through Streamlit's own
    # markdown rather than the card HTML above.
    links = [
        f"[{hotel['name']}]({hotel['link']})"
        for hotel in rows if hotel.get("link") and hotel.get("name")
    ]
    if links:
        with st.expander("Booking links"):
            st.markdown("  \n".join(f"• {link}" for link in links))


def render_budget(budget: dict) -> None:
    """Real numbers from the graph — the old UI showed a hardcoded table."""
    if not budget or not budget.get("total_inr"):
        st.markdown(
            '<div class="tp-note">Cost estimate not available yet — it is '
            "computed once the itinerary is approved.</div>",
            unsafe_allow_html=True,
        )
        if budget.get("components_missing"):
            st.caption(f"Missing: {', '.join(budget['components_missing'])}")
        return

    per_person = money(budget.get("per_person_inr"))
    st.markdown(
        '<div class="tp-stats">'
        + _stat("Transport", money(budget.get("transport_inr")) or "—")
        + _stat("Accommodation", money(budget.get("hotel_inr")) or "—")
        + _stat("On-ground", money(budget.get("onground_inr")) or "—")
        + _stat(
            "Total",
            money(budget["total_inr"]) or "—",
            f"{per_person} per person" if per_person else "",
            accent=True,
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if budget.get("components_missing"):
        st.warning("Estimate excludes: " + ", ".join(budget["components_missing"]))
    st.caption("All figures are estimates, computed from the options actually found.")


def _stat(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    return (
        f'<div class="tp-stat {"accent" if accent else ""}">'
        f'<div class="tp-stat-label">{esc(label)}</div>'
        f'<div class="tp-stat-value">{esc(value)}</div>'
        + (f'<div class="tp-stat-sub">{esc(sub)}</div>' if sub else "")
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Rendering — human review
# ---------------------------------------------------------------------------

def render_review(trip: dict, trip_id: str) -> None:
    """The human-in-the-loop control. This is a real graph interrupt."""
    review = trip.get("pending_review")
    if not review:
        return

    remaining = review.get("revisions_remaining")
    st.markdown(
        '<div class="tp-review-head">'
        '<div class="tp-review-title">Your input is needed</div>'
        f'<div class="tp-review-q">{esc(review.get("question"), "Please review.")}'
        + (f" · {esc(remaining)} revision(s) left" if remaining is not None else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    options = review.get("options") or ["approve"]
    labels = {
        "approve": "✅ Approve and continue",
        "revise_transport": "🔁 Different transport",
        "revise_hotels": "🏨 Different hotels",
        "revise_plan": "📅 Change the itinerary",
    }

    with st.form(key=f"review_{trip_id}_{review.get('stage')}"):
        action = st.radio(
            "What would you like to do?",
            options,
            format_func=lambda o: labels.get(o, o),
            horizontal=len(options) <= 2,
        )
        feedback = st.text_area(
            "What should change? (required unless approving)",
            placeholder="e.g. prefer an overnight train, or hotels near the beach",
        )
        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        if action != "approve" and not feedback.strip():
            st.error("Please describe what you would like changed.")
            return
        with st.spinner("Applying your feedback…"):
            ok, result = api(
                "POST", f"/api/v1/trips/{trip_id}/resume",
                json={"action": action, "feedback": feedback.strip() or None},
                timeout=300,
            )
        if ok:
            st.session_state["trip"] = result
            st.rerun()
        else:
            st.error(result)


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def poll_until_settled(trip_id: str) -> Optional[dict]:
    """Poll until the trip completes, fails or needs a human.

    The API is async, so the UI never holds a long request open. A production
    client would consume ``/stream`` (SSE) instead; polling keeps the Streamlit
    code simple and single-threaded.
    """
    placeholder = st.empty()
    progress = st.progress(0.0)
    deadline = time.time() + POLL_TIMEOUT

    stage_weights = {
        "queued": 0.05, "running": 0.4,
        "awaiting_review": 0.75, "completed": 1.0, "failed": 1.0,
    }
    friendly = {
        "extract_brief": "understanding your request",
        "plan_transport_strategy": "deciding which modes to search",
        "search_transport": "searching flights, trains and buses",
        "recommend_transport": "comparing the real options",
        "enrich": "fetching weather and hotels",
        "build_itinerary": "writing your day-by-day plan",
        "finalise": "totalling the budget",
    }

    while time.time() < deadline:
        ok, trip = api("GET", f"/api/v1/trips/{trip_id}")
        if not ok:
            placeholder.info("Waiting for the planner to start…")
            time.sleep(POLL_INTERVAL)
            continue

        status = trip.get("status", "running")
        progress.progress(stage_weights.get(status, 0.4))

        if status in {"awaiting_review", "completed", "failed"}:
            placeholder.empty()
            progress.empty()
            return trip

        nodes = trip.get("next_nodes") or []
        label = " · ".join(friendly.get(n, n) for n in nodes) if nodes else status
        placeholder.info(f"Planning — {label}")
        time.sleep(POLL_INTERVAL)

    placeholder.empty()
    progress.empty()
    st.warning("Still working — reload the page to check again.")
    return None


# ---------------------------------------------------------------------------
# Sidebar pieces
# ---------------------------------------------------------------------------

def render_backend_dots(health: dict) -> None:
    deps = health.get("dependencies", {}) or {}
    watched = [
        ("redis", deps.get("redis")),
        ("kafka", deps.get("kafka")),
        ("database", deps.get("database")),
        ("checkpointer", deps.get("checkpointer")),
    ]
    dots = "".join(
        f'<span class="tp-dot {"up" if value in {"ok", "postgres", "redis", "sqlite"} else "down"}">'
        f"<i></i>{esc(name)}</span>"
        for name, value in watched if value is not None
    )
    st.markdown(f'<div class="tp-dots">{dots}</div>', unsafe_allow_html=True)
    with st.expander("Backend detail"):
        st.json(deps)


def render_history_browser() -> None:
    """Browse trips planned in the past.

    Backed by the database, not the cache — so trips whose Redis entry expired
    long ago are still here. When no database is configured the API answers 503
    and this collapses to a short explanation rather than an error.
    """
    with st.expander("📚 Past trips"):
        query = st.text_input(
            "Search", placeholder="city or request text…", key="history_q"
        )
        params: dict[str, Any] = {"limit": 15}
        if query.strip():
            params["q"] = query.strip()

        ok, payload = api("GET", "/api/v1/trips", params=params)
        if not ok:
            st.caption(
                "No trip history available. Set `DATABASE_URL` and run "
                "`make migrate` to keep trips permanently."
            )
            return

        trips = payload.get("trips") or []
        if not trips:
            st.caption("No matching trips yet.")
            return

        st.caption(f"{payload.get('total', len(trips))} trip(s) stored")
        for row in trips:
            label = (
                f"{row.get('origin') or '?'} → {row.get('destination') or '?'}"
                f" · {row.get('start_date') or 'no date'}"
            )
            total = row.get("total_inr")
            if total:
                label += f" · ₹{total:,.0f}"
            icon = {
                "completed": "✅", "failed": "❌", "awaiting_review": "⏸",
                "running": "⏳", "queued": "•",
            }.get(row.get("status"), "•")

            if st.button(f"{icon} {label}", key=f"hist_{row['trip_id']}", width="stretch"):
                loaded, trip_data = api("GET", f"/api/v1/trips/{row['trip_id']}")
                if loaded:
                    st.session_state["trip_id"] = row["trip_id"]
                    st.session_state["trip"] = trip_data
                    st.rerun()
                else:
                    st.error(trip_data)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

online, health = backend_online()
if not online:
    st.markdown(
        '<div class="tp-hero"><div class="tp-hero-brand">AI Travel Planner</div>'
        '<div class="tp-hero-route">Backend offline</div></div>',
        unsafe_allow_html=True,
    )
    st.error(
        f"Backend unreachable at `{API_BASE}`.\n\n"
        "Start it with `make api` (or `docker compose up -d`)."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Plan a trip")
    if health.get("status") == "degraded":
        st.warning("Backend is degraded — some providers may be unavailable.")
    render_backend_dots(health)

    origin = st.text_input("From", "Delhi")
    destination = st.text_input("To", "Goa")

    today = date.today()
    start_date = st.date_input("Start date", today + timedelta(days=10), min_value=today)
    end_date = st.date_input("End date", today + timedelta(days=13), min_value=today)

    persons = st.number_input("Travellers", min_value=1, max_value=20, value=2)
    budget_level = st.selectbox("Budget", ["low", "medium", "high"], index=1)
    travel_class = st.selectbox("Travel class", ["economy", "business", "first"])

    all_interests = [
        "Sightseeing", "Adventure", "Beaches", "Shopping",
        "Food & Dining", "Culture & History", "Nature", "Nightlife",
    ]
    preferences = st.multiselect("Interests", all_interests, default=["Sightseeing"])
    special = st.text_area(
        "Special requirements", placeholder="wheelchair access, pet-friendly…"
    )

    submit = st.button("Plan my trip", type="primary", width="stretch")
    st.divider()
    render_history_browser()

if submit:
    if end_date < start_date:
        st.sidebar.error("End date must be on or after the start date.")
    elif origin.strip().lower() == destination.strip().lower():
        st.sidebar.error("Origin and destination must differ.")
    else:
        ok, accepted = api(
            "POST", "/api/v1/trips",
            json={
                "origin": origin.strip(),
                "destination": destination.strip(),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "persons": int(persons),
                "preferences": preferences,
                "budget_level": budget_level,
                "travel_class": travel_class,
                "special_requirements": special.strip(),
                "async_mode": True,
            },
        )
        if not ok:
            st.error(accepted)
        else:
            st.session_state["trip_id"] = accepted["trip_id"]
            st.session_state["trip"] = None
            trip = poll_until_settled(accepted["trip_id"])
            if trip:
                st.session_state["trip"] = trip
                st.rerun()

trip = st.session_state.get("trip")
trip_id = st.session_state.get("trip_id")

# --- empty state -----------------------------------------------------------
if not trip:
    st.markdown(
        '<div class="tp-hero">'
        '<div class="tp-hero-brand">✈️ AI Travel Planner</div>'
        '<div class="tp-hero-route">Where to next?</div>'
        '<div class="tp-hero-meta">Transport, stays and a day-by-day plan — '
        "grounded in live prices, not guesses.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="tp-card"><div class="tp-title-row">How it works</div>'
            '<div class="tp-sub-row">Your request is validated and queued on Kafka, '
            "so you get a trip id back in milliseconds. A worker then runs a "
            "LangGraph pipeline: understand → search every transport mode in "
            "parallel → recommend from real fares → <b>pause for your review</b> → "
            "weather and hotels → itinerary → <b>pause again</b> → totals."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="tp-card"><div class="tp-title-row">Why the pauses are real</div>'
            '<div class="tp-sub-row">Each one is a genuine LangGraph '
            "<code>interrupt()</code> with the state durably checkpointed in "
            "Postgres. You can close the tab, come back later, and it resumes "
            "exactly where it stopped — on any worker, not just the one that "
            "started it.</div></div>",
            unsafe_allow_html=True,
        )
    st.info("Fill in your trip details on the left and hit **Plan my trip**.")
    st.stop()

# --- failed ----------------------------------------------------------------
if trip.get("status") == "failed":
    st.markdown(
        '<div class="tp-hero"><div class="tp-hero-brand">AI Travel Planner</div>'
        '<div class="tp-hero-route">Planning failed</div></div>',
        unsafe_allow_html=True,
    )
    for error in trip.get("errors", []):
        st.error(f"**{error.get('node')}** — {error.get('error')}")
    st.stop()

# --- hero ------------------------------------------------------------------
brief = trip.get("brief") or {}
budget = trip.get("budget") or {}
total = money(budget.get("total_inr"))
per_person = money(budget.get("per_person_inr"))

meta_bits = [f"{esc(brief.get('duration'))} days", f"{esc(brief.get('persons'))} travellers"]
if brief.get("start_date"):
    meta_bits.insert(0, esc(brief["start_date"]))
if trip.get("arrival_time"):
    meta_bits.append(f"arrives {esc(trip['arrival_time'])}")

right_block = ""
if total:
    right_block = (
        f'<div class="tp-hero-total">{total}</div>'
        + (f'<div class="tp-hero-pp">{per_person} per person</div>' if per_person else "")
    )

st.markdown(
    '<div class="tp-hero">'
    '<div class="tp-hero-brand">✈️ AI Travel Planner</div>'
    '<div style="display:flex;justify-content:space-between;align-items:flex-end;'
    'gap:1.5rem;flex-wrap:wrap">'
    "<div>"
    f'<div class="tp-hero-route">{esc(brief.get("origin"))}'
    f'<span class="arr">→</span>{esc(brief.get("destination"))}</div>'
    f'<div class="tp-hero-meta">{" · ".join(meta_bits)}</div>'
    f'<div style="margin-top:.85rem">{status_pill(trip.get("status", ""))}</div>'
    "</div>"
    f'<div class="tp-hero-right">{right_block}</div>'
    "</div></div>",
    unsafe_allow_html=True,
)

render_review(trip, trip_id)

transport_tab, weather_tab, hotels_tab, plan_tab, cost_tab = st.tabs(
    ["🚗 Transport", "🌤️ Weather", "🏨 Hotels", "📅 Itinerary", "💰 Cost"]
)

with transport_tab:
    render_transport(trip)

with weather_tab:
    render_weather(trip.get("weather") or [])

with hotels_tab:
    render_hotels(trip.get("hotels"))

with plan_tab:
    if trip.get("itinerary"):
        meta = trip.get("itinerary_meta") or {}
        if meta.get("source_url"):
            st.caption(f"Grounded in [WikiVoyage]({meta['source_url']})")
        elif meta.get("grounded") is False:
            st.caption("⚠️ No reference guide was found — this plan is less specific.")

        # One call, not three: Streamlit wraps every st.markdown in its own
        # container, so an opening <div> in a separate call is closed before the
        # content arrives and the styling never applies. Blank lines around the
        # itinerary keep the inner markdown parsed as markdown.
        st.markdown(
            f'<div class="tp-itin">\n\n{trip["itinerary"]}\n\n</div>',
            unsafe_allow_html=True,
        )

        ok, markdown = api("GET", f"/api/v1/trips/{trip_id}/itinerary")
        if ok:
            st.download_button(
                "⬇︎  Download itinerary (Markdown)",
                data=markdown,
                file_name=f"{brief.get('origin')}-{brief.get('destination')}.md",
                mime="text/markdown",
            )
    else:
        st.markdown(
            '<div class="tp-note">The itinerary is generated after you approve '
            "the transport options.</div>",
            unsafe_allow_html=True,
        )

with cost_tab:
    render_budget(budget)

if trip.get("errors"):
    with st.expander(f"⚠️ {len(trip['errors'])} issue(s) during planning"):
        for error in trip["errors"]:
            st.caption(f"**{error.get('node')}** — {error.get('error')}")
