"""Scores a finished run against one task. Every number is computed from the output."""

from __future__ import annotations

import re

from briefwright.service import RunResult

_CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")


def score_run(task: dict, result: RunResult) -> dict:
    report = result.report or ""
    folded = report.lower()
    terms = [term.lower() for term in task.get("required_terms", [])]
    term_hits = [term for term in terms if term in folded]
    sections = list(task.get("required_sections", []))
    section_hits = [name for name in sections if _has_section(report, name)]
    valid_ids = {
        source["id"] for source in result.sources if not source.get("quarantined") and source.get("text")
    }
    citations = [token for token in _CITATION.findall(report) if token in valid_ids]
    unique_citations = list(dict.fromkeys(citations))
    min_citations = int(task.get("min_citations") or 0)
    forbidden = [phrase for phrase in task.get("forbidden", []) if phrase.lower() in folded]
    term_coverage = _ratio(len(term_hits), len(terms))
    section_coverage = _ratio(len(section_hits), len(sections))
    citation_coverage = 1.0 if min_citations == 0 else min(1.0, len(unique_citations) / min_citations)
    forbidden_clear = 1.0 if not forbidden else 0.0
    guardrail_passed = 1.0 if result.critique.get("passed") else 0.0
    parts = [
        term_coverage,
        section_coverage,
        citation_coverage,
        forbidden_clear,
        guardrail_passed,
    ]
    return {
        "id": task["id"],
        "composite": round(sum(parts) / len(parts), 4),
        "term_coverage": round(term_coverage, 4),
        "terms_hit": term_hits,
        "terms_missed": [term for term in terms if term not in term_hits],
        "section_coverage": round(section_coverage, 4),
        "sections_hit": section_hits,
        "citation_coverage": round(citation_coverage, 4),
        "valid_citations": unique_citations,
        "forbidden_hits": forbidden,
        "guardrail_passed": bool(result.critique.get("passed")),
        "revision_count": result.revision_count,
        "quarantined": [source["id"] for source in result.sources if source.get("quarantined")],
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
    }


def _has_section(report: str, name: str) -> bool:
    return re.search(rf"^##\s+{re.escape(name)}\s*$", report, flags=re.MULTILINE | re.IGNORECASE) is not None


def _ratio(hit: int, total: int) -> float:
    if total == 0:
        return 1.0
    return hit / total
