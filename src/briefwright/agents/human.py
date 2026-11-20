"""Human review. Auto-approve skips the interrupt; otherwise the graph checkpoints here."""

from __future__ import annotations

from langgraph.types import interrupt

from briefwright.deps import Deps
from briefwright.observability.tracing import finish_node, now
from briefwright.state import ResearchState


def run_human(state: ResearchState, deps: Deps) -> dict:
    started = now()
    critique = state.get("critique") or {}
    if state.get("auto_approve"):
        decision = {"action": "approve", "notes": "Auto-approved for an unattended run."}
    else:
        decision = interrupt(
            {
                "query": state.get("query", ""),
                "draft": state.get("draft", ""),
                "score": critique.get("score"),
                "passed": critique.get("passed"),
                "hard_failures": critique.get("hard_failures") or [],
            }
        )
    if not isinstance(decision, dict):
        decision = {"action": "approve", "notes": str(decision)}
    action = str(decision.get("action") or "approve").lower()
    notes = str(decision.get("notes") or "")
    revision_count = int(state.get("revision_count") or 0)
    max_revisions = int(state.get("max_revisions") or 0)
    if action == "revise" and revision_count < max_revisions:
        update = {
            "human_notes": notes,
            "approved": False,
            "status": "revising",
            "revision_count": revision_count + 1,
            "send_back": True,
        }
        return finish_node("human_review", started, update, "revise", None)

    approved = action == "approve"
    status = "approved" if approved else "rejected"
    update = {
        "human_notes": notes,
        "approved": approved,
        "status": status,
        "send_back": False,
    }
    return finish_node("human_review", started, update, status, None)
