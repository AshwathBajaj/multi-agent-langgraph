"""Token price table. Figures are public list-price estimates, not an invoice."""

from __future__ import annotations

# USD per token. Update these when a provider changes list price.
_PRICES: dict[str, tuple[float, float]] = {
    "mock-research-v1": (0.0, 0.0),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
}
_DEFAULT_PRICE = (0.15 / 1_000_000, 0.60 / 1_000_000)


def price_usage(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = _PRICES.get(model, _DEFAULT_PRICE)
    cost = prompt_tokens * prompt_price + completion_tokens * completion_price
    return round(cost, 8)


def summarize_usage(records: list[dict]) -> dict[str, float | int]:
    prompt = sum(int(record.get("prompt_tokens") or 0) for record in records)
    completion = sum(int(record.get("completion_tokens") or 0) for record in records)
    cost = round(sum(float(record.get("cost_usd") or 0.0) for record in records), 8)
    return {
        "calls": len(records),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost_usd": cost,
    }
