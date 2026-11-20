from briefwright.memory.embeddings import HashingEmbedder, cosine
from briefwright.memory.store import VectorStore


def test_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    first = embedder.embed("retrieval guardrail citation")
    second = embedder.embed("retrieval guardrail citation")
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_similar_text_outranks_unrelated_text() -> None:
    embedder = HashingEmbedder()
    topic = embedder.embed("hybrid retrieval for a research brief")
    near = embedder.embed("hybrid retrieval for a research assistant")
    far = embedder.embed("sourdough hydration and oven temperature")
    assert cosine(topic, near) > cosine(topic, far)


def test_store_round_trip(tmp_path) -> None:
    path = tmp_path / "memory.json"
    store = VectorStore(path, HashingEmbedder())
    store.add("Query: guardrails for retrieval\n\nA guardrail checks citations.", {"query": "guardrails"})
    store.save()
    reloaded = VectorStore(path, HashingEmbedder())
    hits = reloaded.search("What did we say about guardrails for retrieval?", k=2)
    assert hits
    assert "guardrail" in hits[0].text.lower()
