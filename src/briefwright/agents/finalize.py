"""Assemble the published brief and, only if it passed, write long-term memory."""

from __future__ import annotations

from briefwright.deps import Deps
from briefwright.observability.cost import summarize_usage
from briefwright.observability.tracing import finish_node, now
from briefwright.state import ResearchState


def run_finalize(state: ResearchState, deps: Deps) -> dict:
    started = now()
    errors: list[str] = []
    critique = state.get("critique") or {}
    usage = summarize_usage(list(state.get("usage") or []))
    parts = [state.get("draft") or ""]
    if not critique.get("passed"):
        failures = critique.get("hard_failures") or ["The guardrail did not pass."]
        bullet_list = "\n".join(f"- {item}" for item in failures)
        parts.append("## Review note\n\nUnresolved guardrail findings:\n\n" + bullet_list)
    if not state.get("approved"):
        parts.append("## Reviewer note\n\nA reviewer did not approve this brief.")
    elif state.get("human_notes"):
        parts.append("## Reviewer note\n\n" + str(state["human_notes"]))
    parts.append(
        "## Run metadata\n\n"
        f"- Revisions sent back: {int(state.get('revision_count') or 0)}\n"
        f"- Approved: {bool(state.get('approved'))}\n"
        f"- Prompt tokens: {usage['prompt_tokens']}\n"
        f"- Completion tokens: {usage['completion_tokens']}\n"
        f"- Estimated cost USD: {usage['cost_usd']:.6f}"
    )
    report = "\n\n".join(part for part in parts if part)
    if state.get("approved") and critique.get("passed"):
        try:
            deps.memory.add(
                text=f"Query: {state.get('query', '')}\n\n{report[:1200]}",
                metadata={
                    "kind": "brief",
                    "query": state.get("query", ""),
                    "thread_id": state.get("thread_id", ""),
                },
            )
            deps.memory.save()
        except (OSError, ValueError) as exc:
            errors.append(f"memory save failed: {exc}")
    update: dict = {"report": report, "status": "complete"}
    if errors:
        update["errors"] = errors
    return finish_node("finalize", started, update, f"chars={len(report)}", None)
