"""Pick a model client from settings."""

from __future__ import annotations

from briefwright.config import Settings
from briefwright.llm.base import LLMClient, LLMError
from briefwright.llm.mock import MockLLM
from briefwright.llm.openai_compat import OpenAICompatibleClient


def create_llm(settings: Settings) -> LLMClient:
    backend = settings.llm_backend.strip().lower()
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        if not settings.openai_api_key:
            raise LLMError("BRIEFWRIGHT_OPENAI_API_KEY is required when BRIEFWRIGHT_LLM_BACKEND=openai.")
        return OpenAICompatibleClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout=settings.request_timeout_s,
        )
    raise LLMError(f"Unknown BRIEFWRIGHT_LLM_BACKEND: {settings.llm_backend}")
