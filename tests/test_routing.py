from briefwright.routing import decide_next, route_after_critic, route_after_human


def test_decide_next_follows_evidence_then_metric_then_draft() -> None:
    empty = {"query": "explain retrieval", "research_rounds": 0, "max_research_rounds": 2}
    assert decide_next(empty) == "research"
    assert (
        decide_next(
            {
                "query": "compute precision@k",
                "research_rounds": 1,
                "max_research_rounds": 2,
                "sources": [{"id": "a", "text": "notes", "quarantined": False}],
            }
        )
        == "analyze"
    )
    assert (
        decide_next(
            {
                "query": "explain memory",
                "research_rounds": 1,
                "max_research_rounds": 2,
                "sources": [{"id": "a", "text": "notes", "quarantined": False}],
            }
        )
        == "write"
    )
    assert (
        decide_next(
            {
                "query": "explain memory",
                "research_rounds": 1,
                "max_research_rounds": 2,
                "sources": [{"id": "a", "text": "notes", "quarantined": False}],
                "draft": "already written",
            }
        )
        == "finish"
    )


def test_critic_and_human_routes_use_explicit_flags() -> None:
    revising = {"send_back": True, "revision_count": 2, "max_revisions": 2}
    assert route_after_critic(revising) == "writer"
    assert route_after_critic({"send_back": False}) == "human_review"
    assert route_after_human({"status": "revising"}) == "writer"
    assert route_after_human({"status": "approved"}) == "finalize"
