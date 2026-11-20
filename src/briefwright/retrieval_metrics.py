"""Precision at k for a ranked list of binary relevance labels."""

from __future__ import annotations


def precision_at_k(flags: list[int], k: int) -> float | None:
    window = flags[:k]
    if not window:
        return None
    return sum(window) / len(window)


def mean_precision(rows: list[list[int]], ks: tuple[int, ...] = (1, 3, 5)) -> dict[int, float]:
    """Macro-average precision. A query with no retrieved docs is skipped for that k."""

    scores: dict[int, float] = {}
    for k in ks:
        values = [value for value in (precision_at_k(row, k) for row in rows) if value is not None]
        if not values:
            continue
        scores[k] = sum(values) / len(values)
    return scores


def format_precision_lines(scores: dict[int, float], n_queries: int, prefix: str = "") -> list[str]:
    lines = [f"{prefix}precision_at_{k}={score:.3f}" for k, score in scores.items()]
    lines.append(f"{prefix}n_queries={n_queries}")
    return lines
