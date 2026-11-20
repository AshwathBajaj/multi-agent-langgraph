from briefwright.corpus import get_corpus
from briefwright.guardrails.checks import sanitize_untrusted
from briefwright.tools.python_exec import run_python
from briefwright.tools.search import search_corpus


def test_search_finds_precision_log_and_quarantines_paste() -> None:
    corpus = get_corpus()
    metric_hits = search_corpus(corpus, "precision at k retrieval judgments", k=3)
    assert any(hit.id == "retrieval-log" for hit in metric_hits)

    poison_hits = search_corpus(corpus, "prompt injection guardrails", k=4)
    poison = next(hit for hit in poison_hits if hit.id == "untrusted-paste")
    _cleaned, flags = sanitize_untrusted(poison.text)
    assert "prompt_injection" in flags
    assert "pii" in flags


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
