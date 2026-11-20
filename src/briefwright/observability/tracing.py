"""Attach one trace span and any model usage to a node's state update."""

from __future__ import annotations

import logging
import time
from typing import Any

from briefwright.llm.base import LLMResult

logger = logging.getLogger("briefwright")


def finish_node(
    name: str,
    started: float,
    update: dict[str, Any],
    detail: str,
    llm_result: LLMResult | None = None,
) -> dict[str, Any]:
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    event: dict[str, Any] = {
        "node": name,
        "latency_ms": latency_ms,
        "detail": detail,
    }
    merged = dict(update)
    if llm_result is not None:
        record = llm_result.to_record()
        event.update(record)
        merged["usage"] = [record]
    merged["trace"] = [event]
    logger.info(
        detail,
        extra={"node": name, "latency_ms": latency_ms, "model": event.get("model")},
    )
    return merged


def now() -> float:
    return time.perf_counter()
