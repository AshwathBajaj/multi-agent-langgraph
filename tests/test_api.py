from fastapi.testclient import TestClient

from briefwright.api.app import create_app
from briefwright.config import DEFAULT_QUERY


def test_health_and_sync_run(settings) -> None:
    client = TestClient(create_app(settings))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "mock-research-v1"

    response = client.post("/v1/research/sync", json={"query": DEFAULT_QUERY, "auto_approve": True})
    assert response.status_code == 200
    body = response.json()
    assert body["interrupted"] is False
    assert "Limitations" in body["report"]
    assert body["cost_usd"] == 0.0


def test_stream_emits_done(settings) -> None:
    client = TestClient(create_app(settings))
    with client.stream(
        "POST",
        "/v1/research",
        json={"query": DEFAULT_QUERY, "auto_approve": True},
    ) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())
    assert "event: meta" in payload
    assert "event: update" in payload
    assert "event: done" in payload
    assert "Limitations" in payload


def test_resume_after_interrupt(settings) -> None:
    client = TestClient(create_app(settings))
    started = client.post(
        "/v1/research/sync",
        json={"query": DEFAULT_QUERY, "auto_approve": False, "thread_id": "thread-a"},
    )
    assert started.status_code == 200
    assert started.json()["interrupted"] is True
    resumed = client.post(
        "/v1/research/thread-a/resume",
        json={"action": "approve", "notes": "ship it"},
    )
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["approved"] is True
    assert "ship it" in body["report"]
    missing = client.post("/v1/research/missing/resume", json={"action": "approve"})
    assert missing.status_code == 404
