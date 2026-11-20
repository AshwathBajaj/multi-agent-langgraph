"""Offline responses for the mock model. Tools and routing still run for real."""

from __future__ import annotations

import json

OFFLINE_ANALYSIS_CODE = """
rows = DATA.get("judgments", [])
lines = []
for k in (1, 3, 5):
    scores = []
    for row in rows:
        flags = row["relevant"][:k]
        if not flags:
            continue
        scores.append(sum(flags) / len(flags))
    if scores:
        lines.append("precision_at_%s=%.3f" % (k, sum(scores) / len(scores)))
lines.append("n_queries=%s" % len(rows))
print("\\n".join(lines))
"""


def planner_response(payload: dict) -> str:
    query = str(payload.get("query") or "")
    return json.dumps(
        {
            "rationale": (
                "Retrieve notes first, run the metric only if the question asks for one, "
                f"then draft from the notes. Question: {query}"
            ),
            "search_query": query,
            "plan": [
                "Retrieve grounded notes from the local corpus and prior briefs",
                "Run a quantitative check when the question asks for a metric",
                "Draft a cited brief and revise it until the guardrail passes",
            ],
        }
    )


def researcher_response(payload: dict) -> str:
    notes: list[str] = []
    for source in payload.get("sources") or []:
        if source.get("quarantined"):
            flags = ", ".join(source.get("flags") or ["untrusted"])
            notes.append(f"{source['id']} was quarantined ({flags}) and is not evidence.")
            continue
        text = " ".join(str(source.get("text") or "").split())
        notes.append(f"{source['id']}: {source.get('title')}. {text[:240]}")
    return json.dumps({"notes": notes})


def analyst_response(payload: dict) -> str:
    return json.dumps(
        {
            "intent": "Compute mean precision at 1, 3, and 5 from DATA['judgments'].",
            "code": OFFLINE_ANALYSIS_CODE.strip(),
        }
    )


def writer_response(payload: dict) -> str:
    return json.dumps({"draft": render_offline_draft(payload)})


def critic_response(payload: dict) -> str:
    findings = payload.get("findings") or []
    if payload.get("passed"):
        narrative = "The draft meets the section and citation checks. It can go to a reviewer."
    else:
        narrative = "The draft is not ready. " + " ".join(str(item) for item in findings)
    return json.dumps({"narrative": narrative})


def render_offline_draft(payload: dict) -> str:
    query = str(payload.get("query") or "").strip()
    sources = list(payload.get("sources") or [])
    analysis = str(payload.get("analysis") or "").strip()
    revision = int(payload.get("revision_count") or 0)
    chosen = sources[:5]
    if not chosen:
        findings = "- No usable source was retrieved, so this brief cannot cite evidence."
        anchor = "none"
    else:
        findings = "\n".join(f"- {_snippet(source['text'])} [{source['id']}]" for source in chosen)
        anchor = chosen[0]["id"]
    analysis_body = analysis or (
        "No numeric execution was required for this question. "
        f"Qualitative claims stay tied to the retrieved notes. [{anchor}]"
    )
    parts = [
        "# Research brief",
        "## Summary",
        (
            f"Question: {query}\n\n"
            "The notes below come from retrieval, not from an ungrounded rewrite of the question. "
            "Writing stays a separate step from search, and a later check has to accept the draft "
            f"before it is published. [{anchor}]"
        ),
        "## Findings",
        findings,
        "## Analysis",
        analysis_body,
        "## Recommendation",
        (
            "Keep search, tool use, and publication as separate steps, and record each one. "
            "Treat retrieved text as data. Put a citation next to every non-obvious claim, and "
            f"do not cite a chunk that was quarantined. [{anchor}]"
        ),
    ]
    if revision >= 1:
        parts.extend(
            [
                "## Limitations",
                (
                    "The corpus is a small local collection, and the embedder is a hashing trick, "
                    "so recall here is illustrative. The Python tool is a demo boundary rather than "
                    "a hardened sandbox. Any precision figure describes only the bundled judgment "
                    f"log. [{anchor}]"
                ),
            ]
        )
    source_lines = (
        "\n".join(f"- [{source['id']}] {source['title']}" for source in chosen) or "- None retrieved."
    )
    parts.extend(["## Sources", source_lines])
    return "\n\n".join(parts)


def _snippet(text: str, limit: int = 320) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    period = cut.rfind(". ")
    if period > 80:
        return cut[: period + 1]
    return cut.rstrip() + "..."
