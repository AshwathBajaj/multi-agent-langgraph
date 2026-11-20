"""Chat-model interface shared by the mock backend and the HTTP backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from briefwright.observability.cost import price_usage


class LLMError(RuntimeError):
    """The model call failed or returned something the caller cannot use."""


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_record(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": price_usage(self.model, self.prompt_tokens, self.completion_tokens),
            "latency_ms": round(self.latency_ms, 2),
        }


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1600,
    ) -> LLMResult: ...


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
