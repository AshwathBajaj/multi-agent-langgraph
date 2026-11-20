from briefwright.eval.runner import run_eval


def test_eval_scores_are_computed_not_hardcoded(settings) -> None:
    report = run_eval(settings)
    assert report["n_tasks"] == 3
    assert report["backend"] == "mock"
    assert 0.0 < report["macro_score"] <= 1.0
    by_id = {row["id"]: row for row in report["tasks"]}
    assert by_id["architecture"]["guardrail_passed"]
    assert "precision_at_" in by_id["architecture"]["terms_hit"] or any(
        term.startswith("precision_at_") for term in by_id["architecture"]["terms_hit"]
    )
    assert by_id["architecture"]["forbidden_hits"] == []
    assert by_id["architecture"]["composite"] >= 0.8
    assert report["macro_score"] >= 0.8
