"""Deterministic checks on retrieved text and on the draft. The model does not grade itself."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REQUIRED_SECTIONS = (
    "Summary",
    "Findings",
    "Analysis",
    "Recommendation",
    "Limitations",
    "Sources",
)

_INJECTION = (
    re.compile(r"ignore (all |any |previous |prior )?instructions", re.IGNORECASE),
    re.compile(r"reveal (the )?(system |hidden )?prompt", re.IGNORECASE),
    re.compile(r"disregard (the )?(above|previous)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"capital of france is mars", re.IGNORECASE),
)
_PII = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)
_CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")
_FENCE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class GuardrailReport:
    score: float
    passed: bool
    findings: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    valid_citations: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "passed": self.passed,
            "findings": self.findings,
            "hard_failures": self.hard_failures,
            "valid_citations": self.valid_citations,
            "missing_sections": self.missing_sections,
        }


def sanitize_untrusted(text: str) -> tuple[str, list[str]]:
    """Drop tainted lines. Callers quarantine the whole chunk when flags is non-empty."""

    flags: list[str] = []
    kept: list[str] = []
    for line in text.splitlines():
        line_flags: list[str] = []
        if any(pattern.search(line) for pattern in _INJECTION):
            line_flags.append("prompt_injection")
        if any(pattern.search(line) for pattern in _PII):
            line_flags.append("pii")
        if line_flags:
            flags.extend(line_flags)
            continue
        kept.append(line)
    unique = list(dict.fromkeys(flags))
    return "\n".join(kept).strip(), unique


def review_draft(
    draft: str,
    *,
    valid_ids: set[str],
    quarantined_ids: set[str],
    threshold: float,
) -> GuardrailReport:
    visible = _FENCE.sub("", draft)
    findings: list[str] = []
    hard: list[str] = []
    missing = [name for name in REQUIRED_SECTIONS if not _has_section(visible, name)]
    if missing:
        hard.append("Missing sections: " + ", ".join(missing))
        findings.append("Add the missing sections: " + ", ".join(missing) + ".")

    citations = _CITATION.findall(visible)
    valid: list[str] = []
    invalid: list[str] = []
    leaked: list[str] = []
    for citation in citations:
        if citation in quarantined_ids:
            leaked.append(citation)
        elif citation in valid_ids:
            if citation not in valid:
                valid.append(citation)
        elif citation not in invalid:
            invalid.append(citation)
    if invalid:
        hard.append("Unknown citations: " + ", ".join(invalid))
        findings.append("Remove or replace citations that are not in the retrieved set.")
    if leaked:
        hard.append("Quarantined sources were cited: " + ", ".join(leaked))
        findings.append("Do not cite a source that was quarantined.")
    if len(valid) < 2:
        hard.append("Fewer than 2 citations to retrieved sources.")
        findings.append("Cite at least two retrieved sources with [source-id] markers.")

    residue = _residue(visible)
    if residue:
        hard.append("Unsafe content leaked into the draft: " + ", ".join(residue))
        findings.append("Remove injected instructions, payload phrases, and personal data.")

    uncited = _uncited_finding_bullets(visible)
    if uncited:
        hard.append(f"{uncited} findings bullet(s) have no citation.")
        findings.append("Put a citation on every bullet under Findings.")

    score = 1.0
    score -= 0.15 * len(missing)
    score -= 0.2 * len(invalid)
    score -= 0.25 * len(leaked)
    if len(valid) < 2:
        score -= 0.25
    if residue:
        score -= 0.5
    if uncited:
        score -= 0.1 * uncited
    score = max(0.0, min(1.0, score))
    if score < threshold and "Score is below the threshold." not in hard:
        hard.append(f"Score {score:.2f} is below the threshold {threshold:.2f}.")
        findings.append("Raise the score by fixing the findings above.")
    return GuardrailReport(
        score=score,
        passed=not hard,
        findings=findings,
        hard_failures=hard,
        valid_citations=valid,
        missing_sections=missing,
    )


def _has_section(draft: str, name: str) -> bool:
    return re.search(rf"^##\s+{re.escape(name)}\s*$", draft, flags=re.MULTILINE | re.IGNORECASE) is not None


def _residue(text: str) -> list[str]:
    found: list[str] = []
    if any(pattern.search(text) for pattern in _INJECTION):
        found.append("prompt_injection")
    if any(pattern.search(text) for pattern in _PII):
        found.append("pii")
    return found


def _uncited_finding_bullets(draft: str) -> int:
    match = re.search(
        r"^##\s+Findings\s*$([\s\S]*?)(^##\s+|\Z)",
        draft,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if match is None:
        return 0
    bullets = [line for line in match.group(1).splitlines() if line.strip().startswith("- ")]
    if not bullets:
        return 0
    return sum(1 for line in bullets if _CITATION.search(line) is None)
