"""Minimal client for any server that speaks the OpenAI chat completions API."""

from __future__ import annotations

import time

import httpx

from briefwright.llm.base import ChatMessage, LLMError, LLMResult
from briefwright.llm.parsing import estimate_tokens


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("BRIEFWRIGHT_OPENAI_API_KEY is empty.")
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1600,
    ) -> LLMResult:
        started = time.perf_counter()
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
        }
        response: httpx.Response | None = None
        last_error = ""
        for attempt in range(2):
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
            if response.status_code in {429, 500, 502, 503} and attempt == 0:
                last_error = response.text[:300]
                continue
            break
        if response is None:
            raise LLMError(last_error or "No response from the model endpoint.")
        if response.status_code >= 400:
            raise LLMError(f"Model HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Model response did not include a message.") from exc
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(_join(messages)))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content or ""))
        return LLMResult(
            content=content or "",
            model=str(payload.get("model") or self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def _join(messages: list[ChatMessage]) -> str:
    return "\n".join(message.content for message in messages)
