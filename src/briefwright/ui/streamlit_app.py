"""Local demo UI. Talks to the crew in-process so it runs without the API."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if (_SRC / "briefwright").exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from briefwright import __version__
from briefwright.config import DEFAULT_QUERY, Settings
from briefwright.service import Session, open_session, resume_research, run_research

st.set_page_config(page_title="Briefwright", page_icon="B", layout="wide")

st.title("Briefwright")
st.caption(
    f"A LangGraph research crew. Planner, researcher, analyst, writer, and critic. Version {__version__}."
)

with st.sidebar:
    st.header("Run")
    backend = st.selectbox("Model backend", ["mock", "openai"], index=0)
    auto_approve = st.checkbox("Auto-approve the brief", value=True)
    max_revisions = st.slider("Max critique revisions", min_value=0, max_value=4, value=2)
    st.markdown(
        "Mock mode needs no API key. The retriever, Python tool, guardrail, "
        "and revision cap still run. Choose openai only when `BRIEFWRIGHT_OPENAI_API_KEY` is set."
    )
    st.markdown("API, if you started `briefwright serve`: `POST /v1/research`.")

query = st.text_area("Question", value=DEFAULT_QUERY, height=120)

if "session" not in st.session_state:
    st.session_state.session = None
    st.session_state.result = None


def _settings() -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "llm_backend": backend,
            "max_revisions": max_revisions,
            "auto_approve": auto_approve,
        }
    )


def _guard_backend() -> bool:
    if backend == "openai" and not Settings().openai_api_key:
        st.error("Set BRIEFWRIGHT_OPENAI_API_KEY or switch the backend to mock.")
        return False
    return True


if st.button("Run crew", type="primary"):
    if _guard_backend():
        try:
            session: Session = open_session(_settings())
            result = run_research(
                query,
                settings=session.deps.settings,
                auto_approve=auto_approve,
                max_revisions=max_revisions,
                session=session,
            )
        except Exception as exc:  # noqa: BLE001 - show the failure in the demo UI
            st.exception(exc)
        else:
            st.session_state.session = session
            st.session_state.result = result

result = st.session_state.result
if result is None:
    st.info("Run the crew to draft a cited brief from the local corpus.")
else:
    if result.interrupted:
        st.warning("Paused at human review. Approve the draft or send it back.")
        notes = st.text_input("Reviewer notes", value="")
        left, right = st.columns(2)
        if left.button("Approve") and st.session_state.session is not None:
            st.session_state.result = resume_research(
                st.session_state.session, result.thread_id, "approve", notes
            )
            st.rerun()
        if right.button("Request revision") and st.session_state.session is not None:
            st.session_state.result = resume_research(
                st.session_state.session, result.thread_id, "revise", notes
            )
            st.rerun()

    report_tab, source_tab, critique_tab, trace_tab = st.tabs(
        ["Brief", "Sources", "Critique", "Trace and cost"]
    )
    with report_tab:
        st.markdown(result.report or result.draft or "_No draft yet._")
    with source_tab:
        if result.memory_hits:
            st.markdown("**Recalled from long-term memory**")
            st.json(result.memory_hits)
        for source in result.sources:
            label = f"{source['id']}  ·  score {source.get('score')}"
            if source.get("quarantined"):
                label += "  ·  quarantined"
            with st.expander(label, expanded=not source.get("quarantined")):
                if source.get("flags"):
                    st.write("Flags:", ", ".join(source["flags"]))
                st.write(source.get("text") or "_Body withheld._")
    with critique_tab:
        st.json(result.critique)
        st.write(
            f"Revisions sent back: {result.revision_count}. "
            f"Approved: {result.approved}. Status: {result.status}."
        )
    with trace_tab:
        st.json(result.trace)
        st.write(
            f"Prompt tokens {result.prompt_tokens}, "
            f"completion tokens {result.completion_tokens}, "
            f"estimated cost USD {result.cost_usd:.6f}."
        )
        if result.errors:
            st.error("\n".join(result.errors))
