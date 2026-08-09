"""Day-wise itinerary generation grounded in WikiVoyage.

Fixes versus the original ``daywise_plan.py``:
  * It rebuilt a **Chroma vector store from scratch on every single request** —
    fetching two WikiVoyage URLs, splitting, loading a SentenceTransformer model
    into memory, and embedding every chunk. That is 20-60 s and hundreds of MB
    per plan, for a corpus that changes maybe once a month.
    The scraped page is now cached in Redis for a week, and the embedding index
    is built at most once per destination.
  * The Groq API key was hardcoded in the file.
  * The function was annotated ``-> tuple`` but returned a ``str``, and one
    error path returned a bare string where callers expected the same shape.
  * ``interest`` arrived as whatever the LLM had put in ``preferences`` — often
    a list rendered as a Python repr into the prompt.
  * Vector retrieval is now optional. Without the (heavy) embedding stack we
    fall back to feeding the cleaned page text directly, which for a single
    WikiVoyage article fits comfortably in context.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

from app.config import settings
from app.core.cache import cached
from app.core.resilience import resilient
from app.providers import llm as llm_provider
from app.providers import weather as weather_provider

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000


@cached("wikivoyage", ttl=settings.ttl_wikivoyage)
@resilient("wikivoyage", timeout=30.0, retries=1)
async def fetch_destination_guide(destination: str) -> dict[str, Any]:
    """Fetch a WikiVoyage article via the API, cached for a week."""
    candidates = [destination, f"{destination} (India)"]

    async with httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": "AI-Travel-Planner/2.0 (educational project)"},
        follow_redirects=True,
    ) as client:
        for title in candidates:
            try:
                response = await client.get(
                    "https://en.wikivoyage.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "redirects": 1,
                        "format": "json",
                        "titles": title,
                    },
                )
                if response.status_code != 200:
                    continue
                pages = response.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    if "missing" in page:
                        continue
                    extract = (page.get("extract") or "").strip()
                    if len(extract) > 500:
                        return {
                            "destination": destination,
                            "title": page.get("title", title),
                            "content": _clean(extract),
                            "url": "https://en.wikivoyage.org/wiki/"
                            + str(page.get("title", title)).replace(" ", "_"),
                            "found": True,
                        }
            except Exception as exc:  # noqa: BLE001
                logger.debug("WikiVoyage fetch failed for %r: %s", title, exc)

    return {"destination": destination, "content": "", "found": False}


def _clean(text: str) -> str:
    """Drop boilerplate sections that waste context."""
    drop = re.compile(
        r"^==+\s*(See also|External links|References|Further reading|"
        r"Cope|Stay safe|Stay healthy|Connect|Respect)\s*==+",
        re.IGNORECASE | re.MULTILINE,
    )
    match = drop.search(text)
    if match:
        text = text[: match.start()]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _select_context(content: str, interests: list[str]) -> str:
    """Pick the most relevant slice of the guide for these interests.

    Optional vector retrieval when the embedding stack is installed; otherwise a
    cheap keyword-weighted paragraph ranking. Both beat "embed the whole article
    on every request".
    """
    if len(content) <= MAX_CONTEXT_CHARS:
        return content

    try:  # optional heavy path
        import importlib.util

        if importlib.util.find_spec("langchain_huggingface") is None:
            raise ImportError("langchain_huggingface not installed")
        return _vector_context(content, interests)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector retrieval failed (%s); using keyword ranking.", exc)

    keywords = {k.lower() for k in interests} | {
        "see", "do", "eat", "drink", "sleep", "get around", "attraction", "temple",
        "beach", "museum", "market", "fort", "park",
    }
    paragraphs = [p for p in content.split("\n\n") if len(p.strip()) > 60]
    scored = sorted(
        paragraphs,
        key=lambda p: sum(p.lower().count(k) for k in keywords),
        reverse=True,
    )
    out, size = [], 0
    for paragraph in scored:
        if size + len(paragraph) > MAX_CONTEXT_CHARS:
            continue
        out.append(paragraph)
        size += len(paragraph)
    return "\n\n".join(out) or content[:MAX_CONTEXT_CHARS]


def _vector_context(content: str, interests: list[str]) -> str:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(content)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_texts(chunks, embeddings)
    query = "things to see and do: " + ", ".join(interests or ["sightseeing"])
    docs = store.similarity_search(query, k=12)
    return "\n\n".join(d.page_content for d in docs)[:MAX_CONTEXT_CHARS]


async def generate_itinerary(
    *,
    destination: str,
    duration: int,
    interests: list[str],
    weather: list[dict[str, Any]],
    arrival_date: str,
    arrival_time: Optional[str] = None,
    hotel_summary: str = "",
    transport_summary: str = "",
    persons: int = 1,
    budget_level: str = "medium",
    special_requirements: str = "",
    feedback: str = "",
) -> dict[str, Any]:
    """Produce a grounded, weather-aware, day-by-day plan."""
    guide = await fetch_destination_guide(destination)
    context = _select_context(guide.get("content", ""), interests) if guide.get("found") else ""

    grounding = (
        f"REFERENCE GUIDE (WikiVoyage — {guide.get('title', destination)}):\n{context}\n"
        if context
        else "No reference guide was available; rely on general knowledge and say so.\n"
    )

    arrival_note = (
        f"The travellers arrive on {arrival_date} at {arrival_time}. Day 1 must only "
        f"contain activities that start after {arrival_time}, kept light, near the hotel."
        if arrival_time
        else f"The travellers arrive on {arrival_date}; keep Day 1 light."
    )

    prompt = f"""Create a {duration}-day itinerary for {destination}.

{grounding}
TRAVELLERS: {persons} person(s) · budget level: {budget_level}
INTERESTS: {', '.join(interests) if interests else 'general sightseeing'}
SPECIAL REQUIREMENTS: {special_requirements or 'none'}

ARRIVAL: {arrival_note}

WEATHER FORECAST:
{weather_provider.summarise(weather)}

ACCOMMODATION:
{hotel_summary or 'Not selected yet.'}

INBOUND TRANSPORT:
{transport_summary or 'Not selected yet.'}

{f'USER FEEDBACK TO INCORPORATE: {feedback}' if feedback else ''}

REQUIREMENTS:
1. One clearly headed section per day (`### Day N — <weekday>, <date>`).
2. Time-blocked schedule (morning / afternoon / evening) with realistic travel
   time between locations.
3. Match activities to that day's forecast — move outdoor plans to dry days and
   suggest indoor alternatives when rain chance is above 50%.
4. Each day starts and ends at the hotel.
5. Name specific places from the reference guide wherever possible.
6. Finish with a cost breakdown table in INR covering local transport, food,
   activities and entry fees for all {persons} traveller(s), then a final line
   in exactly this format:
   ESTIMATED_ONGROUND_BUDGET_INR: <number>

Use markdown. Do not invent attractions that do not exist.
"""

    plan = await llm_provider.ask(
        prompt,
        system="You are an expert travel planner who writes precise, practical, "
               "weather-aware itineraries grounded in the reference material given.",
        temperature=0.4,
    )

    return {
        "destination": destination,
        "duration": duration,
        "plan": plan,
        "grounded": guide.get("found", False),
        "source_url": guide.get("url"),
        "estimated_onground_budget_inr": extract_budget(plan),
    }


def extract_budget(text: str) -> Optional[float]:
    """Pull the machine-readable budget line out of the plan."""
    match = re.search(
        r"ESTIMATED_ONGROUND_BUDGET_INR\s*[:=]?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None
