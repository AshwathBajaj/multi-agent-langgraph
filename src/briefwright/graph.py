"""The research crew as an explicit LangGraph state graph."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from briefwright.agents.analyst import run_analyst
from briefwright.agents.critic import run_critic
from briefwright.agents.finalize import run_finalize
from briefwright.agents.human import run_human
from briefwright.agents.planner import run_planner
from briefwright.agents.researcher import run_researcher
from briefwright.agents.writer import run_writer
from briefwright.deps import Deps
from briefwright.routing import route_after_critic, route_after_human, route_planner
from briefwright.state import ResearchState


def build_graph(
    deps: Deps,
    checkpointer: MemorySaver | None = None,
) -> CompiledStateGraph:
    """Compile the crew.

    The checkpointer is what makes the human-review interrupt resumable.
    Unattended runs can pass None when auto_approve is set, because that
    path never calls interrupt.
    """

    builder: StateGraph = StateGraph(ResearchState)
    builder.add_node("planner", lambda state: run_planner(state, deps))
    builder.add_node("researcher", lambda state: run_researcher(state, deps))
    builder.add_node("analyst", lambda state: run_analyst(state, deps))
    builder.add_node("writer", lambda state: run_writer(state, deps))
    builder.add_node("critic", lambda state: run_critic(state, deps))
    builder.add_node("human_review", lambda state: run_human(state, deps))
    builder.add_node("finalize", lambda state: run_finalize(state, deps))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_planner,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "finalize": "finalize",
        },
    )
    builder.add_edge("researcher", "planner")
    builder.add_edge("analyst", "planner")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "human_review": "human_review"},
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {"writer": "writer", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
