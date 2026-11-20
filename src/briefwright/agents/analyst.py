"""Analyst: ask the model for Python, then execute it on real SciFact relevance labels."""

from __future__ import annotations

from briefwright.deps import Deps
from briefwright.llm.base import ChatMessage, LLMError
from briefwright.llm.parsing import dump_payload, parse_json_object, system_prompt
from briefwright.observability.tracing import finish_node, now
from briefwright.state import ResearchState
from briefwright.tools.python_exec import run_python

ANALYST_SYSTEM = system_prompt(
    "analyst",
    """
You write Python that computes a metric from the preloaded dict DATA.
math, statistics, json, and collections are already available. Do not import anything.
Do not open files or touch the network. Print the figures you compute.
Return JSON with keys intent (string) and code (string).
DATA["judgments"] is a list of {"query_id": str, "relevant": [0 or 1, ...]}.
Each list is the top retrieved abstracts for one SciFact claim, in rank order.
1 means SciFact lists that abstract as evidence. Compute mean precision at 1, 3, and 5.
If DATA["focus_query_id"] is set, also print focus_precision_at_k for that query.
Print n_queries as well.
""",
)


def run_analyst(state: ResearchState, deps: Deps) -> dict:
    started = now()
    errors: list[str] = []
    sources = list(state.get("sources", []))
    matched = deps.corpus.match_query(state.get("query", ""))
    data = {
        "judgments": deps.corpus.judgment_rows(k=5),
        "focus_query_id": matched.id if matched else "",
        "dataset": "scifact-dev",
        "embedder": deps.corpus.embedder_name,
        "n_documents": len(deps.corpus.documents),
    }

    if not data["judgments"]:
        update = {
            "sources": sources,
            "analysis": "No labeled judgments were available, so no metric was computed.",
            "analysis_attempted": True,
            "status": "analyzing",
        }
        return finish_node("analyst", started, update, "no judgments", None)

    code = ""
    intent = "Compute precision at k."
    llm_result = None
    try:
        llm_result = deps.llm.complete(
            [
                ChatMessage(role="system", content=ANALYST_SYSTEM),
                ChatMessage(
                    role="user",
                    content=dump_payload(
                        {
                            "query": state.get("query", ""),
                            "n_judgments": len(data.get("judgments", [])),
                        }
                    ),
                ),
            ]
        )
        parsed = parse_json_object(llm_result.content)
        code = str(parsed.get("code") or "")
        intent = str(parsed.get("intent") or intent)
    except (LLMError, ValueError) as exc:
        errors.append(f"analyst model failed: {exc}")

    if not code.strip():
        analysis = "The analyst did not return executable code."
        detail = "no code"
    else:
        result = run_python(code, data=data, timeout=deps.settings.python_timeout_s)
        if result.ok:
            stdout = result.stdout.strip() or "(no output)"
            analysis = deps.corpus.describe() + "\n\n```\n" + stdout + "\n```\n\nIntent: " + intent
            detail = "python ok"
        else:
            analysis = f"The Python tool failed: {result.error}"
            detail = "python failed"
            errors.append(analysis)

    update = {
        "sources": sources,
        "analysis": analysis,
        "analysis_attempted": True,
        "status": "analyzing",
    }
    if errors:
        update["errors"] = errors
    return finish_node("analyst", started, update, detail, llm_result)
