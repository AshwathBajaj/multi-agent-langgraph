import json

import httpx

from briefwright.config import Settings
from briefwright.llm.base import ChatMessage
from briefwright.llm.factory import create_llm
from briefwright.llm.openai_compat import OpenAICompatibleClient
from briefwright.llm.parsing import system_prompt
from briefwright.observability.cost import price_usage, summarize_usage


def test_mock_planner_returns_json_and_token_counts() -> None:
    model = create_llm(Settings(llm_backend="mock"))
    result = model.complete(
        [
            ChatMessage(role="system", content=system_prompt("planner", "Return JSON.")),
            ChatMessage(role="user", content=json.dumps({"query": "How does retrieval work?"})),
        ]
    )
    payload = json.loads(result.content)
    assert payload["search_query"]
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.to_record()["cost_usd"] == 0.0


def test_price_table_for_a_known_model() -> None:
    cost = price_usage("gpt-4o-mini", 1000, 1000)
    assert cost == round(1000 * (0.15 / 1_000_000) + 1000 * (0.60 / 1_000_000), 8)
    summary = summarize_usage([{"prompt_tokens": 1000, "completion_tokens": 1000, "cost_usd": cost}])
    assert summary["total_tokens"] == 2000
    assert summary["calls"] == 1


def test_openai_compatible_client_reads_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"].startswith("Bearer ")
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            },
        )

    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="gpt-4o-mini",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = client.complete([ChatMessage(role="user", content="hello")])
    assert result.content == '{"ok": true}'
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 5
    assert result.to_record()["cost_usd"] > 0
