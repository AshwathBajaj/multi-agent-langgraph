from briefwright.config import DEFAULT_QUERY
from briefwright.graph import build_graph
from briefwright.service import open_session, resume_research, run_research


def test_unattended_run_revises_once_and_cites_real_numbers(settings) -> None:
    result = run_research(DEFAULT_QUERY, settings=settings, auto_approve=True)
    assert result.status == "complete"
    assert result.approved
    assert result.critique["passed"]
    assert result.revision_count == 1
    assert "## Limitations" in result.report
    from briefwright.corpus import get_corpus

    scores = get_corpus().mean_precision()
    for k, score in scores.items():
        assert f"precision_at_{k}={score:.3f}" in result.report
    assert "focus_precision_at_1=1.000" in result.report
    assert "SciFact" in result.report
    assert "Mars" not in result.report
    assert "admin@example.com" not in result.report
    nodes = [event["node"] for event in result.trace]
    for name in ("planner", "researcher", "analyst", "writer", "critic", "human_review", "finalize"):
        assert name in nodes
    assert nodes.count("writer") == 2
    assert nodes.count("critic") == 2
    assert result.prompt_tokens > 0
    assert result.cost_usd == 0.0


def test_revision_cap_publishes_the_failure(settings) -> None:
    result = run_research(DEFAULT_QUERY, settings=settings, auto_approve=True, max_revisions=0)
    assert result.revision_count == 0
    assert result.critique["passed"] is False
    assert "Unresolved guardrail findings" in result.report
    assert "## Limitations" not in result.draft


def test_memory_is_recalled_on_the_next_run(settings) -> None:
    first = run_research(DEFAULT_QUERY, settings=settings, auto_approve=True)
    assert first.approved and first.critique["passed"]
    second = run_research(
        "What did we already record about birth-weight and breast cancer?",
        settings=settings,
        auto_approve=True,
    )
    assert second.memory_hits
    assert any("breast" in hit["text"].lower() for hit in second.memory_hits)


def test_graph_runs_without_a_checkpointer_when_it_does_not_pause(settings) -> None:
    session = open_session(settings, checkpointer=False)
    from briefwright.state import initial_state

    final = session.graph.invoke(
        initial_state(DEFAULT_QUERY, settings, auto_approve=True, thread_id="nocheck")
    )
    assert final["status"] == "complete"
    assert "Limitations" in final["report"]


def test_human_review_pauses_and_can_reject_a_further_revision(settings) -> None:
    tight = settings.model_copy(update={"max_revisions": 1})
    session = open_session(tight)
    paused = run_research(
        DEFAULT_QUERY,
        settings=tight,
        auto_approve=False,
        max_revisions=1,
        session=session,
        thread_id="review-1",
    )
    assert paused.interrupted
    assert paused.next_nodes == ["human_review"]
    assert paused.critique["passed"]
    assert paused.revision_count == 1

    finished = resume_research(session, "review-1", "revise", "one more pass")
    assert not finished.interrupted
    assert finished.approved is False
    assert "did not approve" in finished.report


def test_compiled_graph_has_the_crew_nodes(settings) -> None:
    session = open_session(settings, checkpointer=False)
    names = set(build_graph(session.deps).get_graph().nodes)
    assert {
        "planner",
        "researcher",
        "analyst",
        "writer",
        "critic",
        "human_review",
        "finalize",
    } <= names
