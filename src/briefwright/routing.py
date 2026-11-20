"""Control-flow decisions. These functions are ordinary code, not model output."""

from __future__ import annotations

from briefwright.state import ResearchState

_ANALYSIS_TERMS = (
    "precision",
    "precision@k",
    "p@k",
    "recall",
    "calculate",
    "compute",
    "metric",
    "percent",
    "average",
    "how many",
)


def needs_analysis(query: str) -> bool:
    folded = query.lower()
    return any(term in folded for term in _ANALYSIS_TERMS)


def usable_sources(state: ResearchState) -> list[dict]:
    return [
        source
        for source in state.get("sources", [])
        if not source.get("quarantined") and (source.get("text") or "").strip()
    ]


def decide_next(state: ResearchState) -> str:
    """Pick the next worker from what the state already contains.

    The planner model may suggest a search string. It does not choose this branch.
    """

    rounds = int(state.get("research_rounds") or 0)
    max_rounds = int(state.get("max_research_rounds") or 0)
    if not usable_sources(state) and rounds < max_rounds:
        return "research"
    wants_metric = bool(state.get("needs_metric")) or needs_analysis(state.get("query", ""))
    if wants_metric and not state.get("analysis_attempted"):
        return "analyze"
    if not (state.get("draft") or "").strip():
        return "write"
    return "finish"


def route_planner(state: ResearchState) -> str:
    return {
        "research": "researcher",
        "analyze": "analyst",
        "write": "writer",
        "finish": "finalize",
    }.get(state.get("next_step", "write"), "writer")


def route_after_critic(state: ResearchState) -> str:
    if state.get("send_back"):
        return "writer"
    return "human_review"


def route_after_human(state: ResearchState) -> str:
    if state.get("status") == "revising":
        return "writer"
    return "finalize"
