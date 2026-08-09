"""Shared LLM access.

Previously every module constructed its own ``ChatGroq`` — some with a
hardcoded key, some at import time, one referencing a module-level ``llm`` that
was never defined (a live ``NameError`` on three graph nodes). One factory,
one config, cached per model.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_llm(model: Optional[str] = None, temperature: Optional[float] = None) -> Any:
    """Return a cached ChatGroq client."""
    settings.require("groq_api_key")
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_retries=settings.llm_max_retries,
        timeout=60,
    )


async def ainvoke(
    messages: list[BaseMessage],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Invoke the LLM and return plain text."""
    llm = get_llm(model, temperature)
    response = await llm.ainvoke(messages)
    return (response.content or "").strip()


async def ask(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    return await ainvoke(messages, model=model, temperature=temperature)


async def choose(
    prompt: str,
    options: set[str] | list[str],
    *,
    default: str,
    system: Optional[str] = None,
) -> str:
    """Ask the LLM to pick one of ``options``, robustly.

    The original code did ``response.content.strip().lower()`` and compared it
    to a set — any punctuation, quoting or explanatory sentence silently fell
    through to the fallback branch. Here we normalise and substring-match, and
    only then give up.
    """
    valid = {str(o).strip().lower(): str(o) for o in options}
    try:
        raw = await ask(prompt, system=system, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM choice failed (%s); using default %r.", exc, default)
        return default

    cleaned = re.sub(r"[^a-z0-9_ ]+", " ", raw.lower()).strip()

    if cleaned in valid:
        return valid[cleaned]
    for token in cleaned.split():
        if token in valid:
            return valid[token]
    # Longest option that appears anywhere in the response wins.
    matches = [k for k in valid if k in cleaned]
    if matches:
        return valid[max(matches, key=len)]

    logger.warning("LLM returned unmappable choice %r; using default %r.", raw, default)
    return default


async def ask_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    retries: int = 2,
) -> Any:
    """Ask for JSON and parse it, tolerating fences and surrounding prose."""
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        text = await ask(
            prompt if attempt == 0 else f"{prompt}\n\nYour last reply was not valid "
            f"JSON ({last_error}). Reply with JSON only.",
            system=system,
            model=model,
            temperature=0.0,
        )
        parsed = extract_json(text)
        if parsed is not None:
            return parsed
        last_error = "no JSON object found"
        await asyncio.sleep(0.3)
    raise ValueError(f"LLM did not return parseable JSON: {last_error}")


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM response."""
    import json

    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        start = min(
            [i for i in (candidate.find("{"), candidate.find("[")) if i != -1],
            default=-1,
        )
        if start == -1:
            continue
        closing = "}" if candidate[start] == "{" else "]"
        end = candidate.rfind(closing)
        if end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None
