"""Graph assembly.

Topology
--------
    extract_brief
        └─> plan_transport_strategy ─> search_transport ─> recommend_transport
                ▲                                                │
                │                                                ▼
                └──────────── revise_transport ──── human_review_transport
                                                                 │ approve
                                                                 ▼
                                                              enrich
                                                                 │
                                                                 ▼
                                                          build_itinerary
                                                                 │
                                                                 ▼
                                     ┌──────────── human_review_plan
                                     │  revise_plan │ revise_hotels │ approve
                                     ▼              ▼               ▼
                             build_itinerary     enrich          finalise ─> END

Structural problems in the original graph that this fixes
---------------------------------------------------------
1. ``should_continue`` was registered as the router for ``llm_decider`` but read
   ``state["decision"]`` — which ``llm_decider`` had *just* set from an
   unvalidated LLM string. Any unmapped value silently routed to
   ``bus_or_cab_only``.
2. ``route_from_llm_decision`` returned raw LLM text (``"Hotels."``, ``"end"``,
   a whole sentence) straight into ``add_conditional_edges``. Anything not
   matching a key exactly raised at runtime. Routers here return a closed set of
   literals and are unit-testable.
3. The ``Human_Review -> llm_decider -> ... -> Human_Review`` loop and the
   ``User_Feedback -> DayWise_Plan`` loop had **no revision cap**, so a stubborn
   LLM could spin until LangGraph's recursion limit blew up mid-plan.
4. Six near-identical transport "tool" nodes each re-implemented the same
   fan-out by hand. One search node now covers all modes concurrently.
5. Nodes were registered by ``tool.__name__``, coupling graph topology to Python
   function names.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.checkpointer import get_checkpointer
from app.graph.state import TripState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routers — pure functions over state, returning a closed set of literals
# ---------------------------------------------------------------------------

def route_after_brief(state: TripState) -> Literal["plan_transport_strategy", "failed"]:
    """Stop early if we never understood the request."""
    brief = state.get("brief") or {}
    if not brief.get("origin") or not brief.get("destination"):
        return "failed"
    return "plan_transport_strategy"


def route_after_transport_review(
    state: TripState,
) -> Literal["enrich", "plan_transport_strategy"]:
    decision = (state.get("review_decision") or "approve").lower()
    if decision.startswith("revise") and state.get("revision_count", 0) <= nodes.MAX_REVISIONS:
        return "plan_transport_strategy"
    return "enrich"


def route_after_plan_review(
    state: TripState,
) -> Literal["finalise", "build_itinerary", "enrich", "plan_transport_strategy"]:
    decision = (state.get("review_decision") or "approve").lower()
    if state.get("revision_count", 0) > nodes.MAX_REVISIONS:
        return "finalise"
    return {
        "revise_plan": "build_itinerary",
        "revise_hotels": "enrich",
        "revise_transport": "plan_transport_strategy",
    }.get(decision, "finalise")


async def _fail_fast(state: TripState) -> dict[str, Any]:
    from langchain_core.messages import AIMessage

    return {
        # Marked fatal so the planner reports this trip as `failed`. Reaching
        # END is how the graph stops, not evidence that anything succeeded —
        # without this flag an unparseable request was recorded as a completed
        # trip with no itinerary.
        "errors": [
            {
                "node": "failed",
                "fatal": True,
                "message": "Could not determine origin and destination.",
            }
        ],
        "messages": [
            AIMessage(
                content="❌ Could not determine the origin and destination for this "
                        "trip. Please state both cities explicitly."
            )
        ],
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_graph_definition() -> StateGraph:
    """The topology, independent of any checkpointer — handy for tests."""
    graph = StateGraph(TripState)

    graph.add_node("extract_brief", nodes.extract_brief)
    graph.add_node("plan_transport_strategy", nodes.plan_transport_strategy)
    graph.add_node("search_transport", nodes.search_transport)
    graph.add_node("recommend_transport", nodes.recommend_transport)
    graph.add_node("human_review_transport", nodes.human_review_transport)
    graph.add_node("enrich", nodes.enrich)
    graph.add_node("build_itinerary", nodes.build_itinerary)
    graph.add_node("human_review_plan", nodes.human_review_plan)
    graph.add_node("finalise", nodes.finalise)
    graph.add_node("failed", _fail_fast)

    graph.add_edge(START, "extract_brief")
    graph.add_conditional_edges(
        "extract_brief",
        route_after_brief,
        {"plan_transport_strategy": "plan_transport_strategy", "failed": "failed"},
    )
    graph.add_edge("failed", END)

    graph.add_edge("plan_transport_strategy", "search_transport")
    graph.add_edge("search_transport", "recommend_transport")
    graph.add_edge("recommend_transport", "human_review_transport")

    graph.add_conditional_edges(
        "human_review_transport",
        route_after_transport_review,
        {"enrich": "enrich", "plan_transport_strategy": "plan_transport_strategy"},
    )

    graph.add_edge("enrich", "build_itinerary")
    graph.add_edge("build_itinerary", "human_review_plan")

    graph.add_conditional_edges(
        "human_review_plan",
        route_after_plan_review,
        {
            "finalise": "finalise",
            "build_itinerary": "build_itinerary",
            "enrich": "enrich",
            "plan_transport_strategy": "plan_transport_strategy",
        },
    )

    graph.add_edge("finalise", END)
    return graph


_compiled: Any = None


async def get_graph(force_rebuild: bool = False) -> Any:
    """Compile once per process, with the durable checkpointer attached."""
    global _compiled
    if _compiled is not None and not force_rebuild:
        return _compiled
    checkpointer = await get_checkpointer()
    _compiled = build_graph_definition().compile(checkpointer=checkpointer)
    logger.info("Trip graph compiled with %s.", type(checkpointer).__name__)
    return _compiled


async def shutdown_graph() -> None:
    """Drop the compiled graph and close the checkpointer.

    The checkpointer owns a connection bound to the event loop that opened it
    (aiosqlite runs a background thread per connection), so the compiled graph
    must be discarded together with it. Leaving ``_compiled`` pointing at a
    closed checkpointer produces "threads can only be started once" on the next
    use.
    """
    global _compiled
    _compiled = None
    from app.graph.checkpointer import close_checkpointer

    await close_checkpointer()


async def export_mermaid() -> str:
    graph = await get_graph()
    return graph.get_graph().draw_mermaid()
