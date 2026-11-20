"""Graph state. Annotated lists append across nodes; everything else is overwritten."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from briefwright.config import Settings


class ResearchState(TypedDict, total=False):
    query: str
    thread_id: str
    plan: list[str]
    plan_notes: str
    next_step: str
    search_query: str
    sources: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    research_notes: list[str]
    research_rounds: int
    max_research_rounds: int
    analysis: str
    analysis_attempted: bool
    draft: str
    critique: dict[str, Any]
    send_back: bool
    revision_count: int
    max_revisions: int
    score_threshold: float
    auto_approve: bool
    approved: bool
    human_notes: str
    report: str
    status: str
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    usage: Annotated[list[dict[str, Any]], operator.add]


def initial_state(
    query: str,
    settings: Settings,
    *,
    auto_approve: bool | None = None,
    thread_id: str = "",
    max_revisions: int | None = None,
) -> ResearchState:
    return {
        "query": query.strip(),
        "thread_id": thread_id,
        "plan": [],
        "plan_notes": "",
        "next_step": "research",
        "search_query": query.strip(),
        "sources": [],
        "memory_hits": [],
        "research_notes": [],
        "research_rounds": 0,
        "max_research_rounds": settings.max_research_rounds,
        "analysis": "",
        "analysis_attempted": False,
        "draft": "",
        "critique": {},
        "send_back": False,
        "revision_count": 0,
        "max_revisions": settings.max_revisions if max_revisions is None else max_revisions,
        "score_threshold": settings.score_threshold,
        "auto_approve": settings.auto_approve if auto_approve is None else auto_approve,
        "approved": False,
        "human_notes": "",
        "report": "",
        "status": "planning",
        "errors": [],
        "trace": [],
        "usage": [],
    }
