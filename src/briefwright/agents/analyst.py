"""Analyst: ask the model for Python, then execute that Python against retrieved data."""

from __future__ import annotations

import json
import re

from briefwright.deps import Deps
from briefwright.llm.base import ChatMessage, LLMError
from briefwright.llm.parsing import dump_payload, parse_json_object, system_prompt
from briefwright.observability.tracing import finish_node, now
from briefwright.state import ResearchState
from briefwright.tools.python_exec import run_python
from briefwright.tools.search import search_corpus

ANALYST_SYSTEM = system_prompt(
    "analyst",
    """
You write Python that computes a metric from the preloaded dict DATA.
math, statistics, json, and collections are already available. Do not import anything.
Do not open files or touch the network. Print the figures you compute.
Return JSON with keys intent (string) and code (string).
DATA looks like {"judgments": [{"query": "...", "relevant": [1, 0, 1]}]}.
Compute mean precision at 1, 3, and 5. Print lines like precision_at_3=0.778 and n_queries=3.
""",
)

_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def run_analyst(state: ResearchState, deps: Deps) -> dict:
    started = now()
    errors: list[str] = []
    sources = list(state.get("sources", []))
    data = extract_judgments(sources)
    if data is None:
        extra = _sources_from_hits(search_corpus(deps.corpus, "precision at k retrieval judgments", k=3))
        sources = _merge(sources, extra)
        data = extract_judgments(sources)

    if data is None:
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
            analysis = (
                "The analyst ran the following calculation on the labeled retrieval log.\n\n"
                f"```\n{stdout}\n```\n\nIntent: {intent}"
            )
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


def extract_judgments(sources: list[dict]) -> dict | None:
    for source in sources:
        if source.get("quarantined"):
            continue
        match = _FENCE.search(source.get("text") or "")
        if match is None:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("judgments"), list):
            return payload
    return None


def _sources_from_hits(hits: list) -> list[dict]:
    return [
        {
            "id": hit.id,
            "title": hit.title,
            "text": hit.text,
            "score": round(hit.score, 4),
            "flags": [],
            "quarantined": False,
        }
        for hit in hits
    ]


def _merge(existing: list[dict], fresh: list[dict]) -> list[dict]:
    merged = {str(source["id"]): source for source in existing}
    for source in fresh:
        merged.setdefault(str(source["id"]), source)
    return list(merged.values())
