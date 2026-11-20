import pytest

from briefwright.corpus import get_corpus
from briefwright.tools.python_exec import run_python
from briefwright.tools.search import search_corpus


def test_sample_search_ranks_a_real_evidence_abstract_first() -> None:
    corpus = get_corpus()
    hits = search_corpus(corpus, "Birth-weight is positively associated with breast cancer.", k=5)
    assert hits[0].id == "27123743"
    assert hits[0].id in corpus.qrels["179"]
    assert corpus.embedder_name == "sentence-transformers/all-MiniLM-L6-v2"
    scores = corpus.mean_precision()
    assert round(scores[1], 3) == 0.667


def test_stored_query_vector_matches_a_live_minilm() -> None:
    fastembed = pytest.importorskip("fastembed")
    del fastembed
    from briefwright.memory.minilm import MiniLMEmbedder

    corpus = get_corpus()
    claim = "Birth-weight is positively associated with breast cancer."
    live = MiniLMEmbedder()
    stored = corpus.embed_query(claim)
    assert stored is not None
    fresh = live.embed(claim)
    from briefwright.memory.embeddings import cosine

    assert cosine(stored, fresh) > 0.999


def test_python_tool_runs_safe_code_and_blocks_the_rest() -> None:
    result = run_python("print(sum(DATA['nums']))", data={"nums": [1, 2, 3]})
    assert result.ok
    assert result.stdout.strip() == "6"

    blocked = run_python("open('secret.txt')")
    assert not blocked.ok
    assert "blocked" in blocked.error.lower()

    imported = run_python("import os\nprint(os.getcwd())")
    assert not imported.ok
    assert "import" in imported.error.lower()


def test_python_tool_times_out() -> None:
    result = run_python("while True:\n    pass", timeout=0.5)
    assert not result.ok
    assert "timed out" in result.error.lower()
