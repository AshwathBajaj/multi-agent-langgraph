"""Run and resume the crew. The CLI, API, UI, and eval harness all enter here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from briefwright.config import Settings
from briefwright.deps import Deps, build_deps
from briefwright.graph import build_graph
from briefwright.observability.cost import summarize_usage
from briefwright.observability.logging import configure_logging
from briefwright.state import ResearchState, initial_state


@dataclass
class Session:
    deps: Deps
    graph: Any
    checkpointer: MemorySaver | None


@dataclass
class RunResult:
    thread_id: str
    status: str
    interrupted: bool
    approved: bool
    revision_count: int
    report: str
    draft: str
    query: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    critique: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_nodes: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    analysis: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "status": self.status,
            "interrupted": self.interrupted,
            "approved": self.approved,
            "revision_count": self.revision_count,
            "report": self.report,
            "draft": self.draft,
            "query": self.query,
            "sources": self.sources,
            "memory_hits": self.memory_hits,
            "critique": self.critique,
            "trace": self.trace,
            "usage": self.usage,
            "errors": self.errors,
            "next_nodes": self.next_nodes,
            "plan": self.plan,
            "analysis": self.analysis,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def open_session(settings: Settings | None = None, *, checkpointer: bool = True) -> Session:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    deps = build_deps(settings)
    saver = MemorySaver() if checkpointer else None
    graph = build_graph(deps, checkpointer=saver)
    return Session(deps=deps, graph=graph, checkpointer=saver)


def run_research(
    query: str,
    *,
    settings: Settings | None = None,
    auto_approve: bool | None = None,
    thread_id: str | None = None,
    max_revisions: int | None = None,
    session: Session | None = None,
) -> RunResult:
    settings = settings or (session.deps.settings if session else Settings())
    if session is None:
        session = open_session(settings)
    thread_id = thread_id or uuid4().hex[:12]
    state = initial_state(
        query,
        settings,
        auto_approve=auto_approve,
        thread_id=thread_id,
        max_revisions=max_revisions,
    )
    config = {"configurable": {"thread_id": thread_id}}
    session.graph.invoke(state, config)
    return result_from_session(session, thread_id)


def resume_research(
    session: Session,
    thread_id: str,
    action: str,
    notes: str = "",
) -> RunResult:
    config = {"configurable": {"thread_id": thread_id}}
    session.graph.invoke(Command(resume={"action": action, "notes": notes}), config)
    return result_from_session(session, thread_id)


def result_from_session(session: Session, thread_id: str) -> RunResult:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = session.graph.get_state(config)
    values: ResearchState = dict(snapshot.values or {})
    usage = summarize_usage(list(values.get("usage") or []))
    next_nodes = [str(node) for node in (snapshot.next or ())]
    return RunResult(
        thread_id=thread_id,
        status=str(values.get("status") or ""),
        interrupted=bool(next_nodes),
        approved=bool(values.get("approved")),
        revision_count=int(values.get("revision_count") or 0),
        report=str(values.get("report") or ""),
        draft=str(values.get("draft") or ""),
        query=str(values.get("query") or ""),
        sources=list(values.get("sources") or []),
        memory_hits=list(values.get("memory_hits") or []),
        critique=dict(values.get("critique") or {}),
        trace=list(values.get("trace") or []),
        usage=list(values.get("usage") or []),
        errors=list(values.get("errors") or []),
        next_nodes=next_nodes,
        plan=list(values.get("plan") or []),
        analysis=str(values.get("analysis") or ""),
        cost_usd=float(usage["cost_usd"]),
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
    )
