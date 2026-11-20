"""Deterministic model used by tests, CI, and the offline demo."""

from __future__ import annotations

import time

from briefwright.llm.base import ChatMessage, LLMResult
from briefwright.llm.mock_content import (
    analyst_response,
    critic_response,
    planner_response,
    researcher_response,
    writer_response,
)
from briefwright.llm.parsing import estimate_tokens, parse_json_object

_ROLES = {
    "planner": planner_response,
    "researcher": researcher_response,
    "analyst": analyst_response,
    "writer": writer_response,
    "critic": critic_response,
}


class MockLLM:
    model = "mock-research-v1"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1600,
    ) -> LLMResult:
        started = time.perf_counter()
        system = next((message.content for message in messages if message.role == "system"), "")
        user = next((message.content for message in messages if message.role == "user"), "")
        role = _role_name(system)
        builder = _ROLES.get(role)
        if builder is None:
            content = '{"note": "unsupported role"}'
        else:
            content = builder(parse_json_object(user))
        prompt = "\n".join(message.content for message in messages)
        return LLMResult(
            content=content,
            model=self.model,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(content),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def _role_name(system: str) -> str:
    first = system.splitlines()[0] if system else ""
    if first.lower().startswith("role:"):
        return first.split(":", 1)[1].strip().lower()
    return ""
