"""Graph state.

The original ``State`` had two structural problems:

1. **Message duplication.** ``messages`` used the ``add_messages`` reducer, but
   every node returned ``[*state["messages"], AIMessage(...)]``. ``add_messages``
   *appends* what you return, so each node re-appended the entire history. By
   the sixth node the transcript held hundreds of duplicated messages, and each
   one was being sent to the LLM. That is the single most expensive bug in the
   project. Nodes here return **only the new messages**.

2. **No reducers on anything else**, plus ``{**state, ...}`` returns that fought
   the reducers. Each field below now has an explicit merge policy.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

ReviewDecision = Literal["approve", "revise_transport", "revise_hotels", "revise_plan"]


def merge_dict(left: Optional[dict], right: Optional[dict]) -> dict:
    """Shallow-merge partial updates instead of clobbering."""
    return {**(left or {}), **(right or {})}


def take_last(left: Any, right: Any) -> Any:
    """Last writer wins, but ``None`` never erases a previous value."""
    return left if right is None else right


class TripState(TypedDict, total=False):
    # --- identity ----------------------------------------------------------
    trip_id: str

    # --- conversation ------------------------------------------------------
    # Nodes append ONLY their new messages here.
    messages: Annotated[list[BaseMessage], add_messages]

    # --- request -----------------------------------------------------------
    raw_request: str
    brief: Annotated[dict[str, Any], merge_dict]

    # --- transport ---------------------------------------------------------
    transport_strategy: Annotated[Optional[dict[str, Any]], take_last]
    transport_options: Annotated[dict[str, Any], merge_dict]
    transport_summary: Annotated[Optional[str], take_last]
    recommended_transport: Annotated[Optional[dict[str, Any]], take_last]
    arrival_time: Annotated[Optional[str], take_last]
    arrival_date: Annotated[Optional[str], take_last]

    # --- enrichment --------------------------------------------------------
    weather: Annotated[Optional[list[dict[str, Any]]], take_last]
    hotels: Annotated[Optional[dict[str, Any]], take_last]
    hotels_summary: Annotated[Optional[str], take_last]

    # --- itinerary ---------------------------------------------------------
    itinerary: Annotated[Optional[str], take_last]
    itinerary_meta: Annotated[dict[str, Any], merge_dict]
    budget: Annotated[dict[str, Any], merge_dict]

    # --- human in the loop -------------------------------------------------
    review_decision: Annotated[Optional[str], take_last]
    review_feedback: Annotated[Optional[str], take_last]
    revision_count: Annotated[int, operator.add]

    # --- diagnostics -------------------------------------------------------
    # Append-only so a later partial failure cannot hide an earlier one.
    errors: Annotated[list[dict[str, Any]], operator.add]
    provider_health: Annotated[dict[str, Any], merge_dict]


def new_state(trip_id: str, raw_request: str = "", **kwargs: Any) -> TripState:
    state: TripState = {
        "trip_id": trip_id,
        "raw_request": raw_request,
        "messages": [],
        "brief": {},
        "transport_options": {},
        "itinerary_meta": {},
        "budget": {},
        "revision_count": 0,
        "errors": [],
        "provider_health": {},
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state
