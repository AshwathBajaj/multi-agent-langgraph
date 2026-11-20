"""FastAPI surface. Streaming uses server-sent events."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from briefwright import __version__
from briefwright.api.schemas import ResearchRequest, ResumeRequest, RunResponse
from briefwright.config import Settings
from briefwright.service import Session, open_session, result_from_session, resume_research, run_research
from briefwright.state import initial_state


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    session = open_session(settings)
    app = FastAPI(
        title="Briefwright",
        version=__version__,
        summary="Multi-agent research crew.",
    )
    app.state.session = session
    app.state.lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "backend": settings.llm_backend,
            "model": str(getattr(session.deps.llm, "model", "")),
        }

    @app.post("/v1/research/sync", response_model=RunResponse)
    def research_sync(body: ResearchRequest) -> RunResponse:
        with app.state.lock:
            result = run_research(
                body.query,
                settings=settings,
                auto_approve=body.auto_approve,
                thread_id=_thread_id(body.thread_id),
                max_revisions=body.max_revisions,
                session=session,
            )
        return RunResponse(**result.to_dict())

    @app.post("/v1/research")
    def research_stream(body: ResearchRequest) -> StreamingResponse:
        thread_id = _thread_id(body.thread_id)

        def events() -> Iterator[str]:
            with app.state.lock:
                yield _sse(
                    "meta",
                    {"thread_id": thread_id, "model": str(getattr(session.deps.llm, "model", ""))},
                )
                state = initial_state(
                    body.query,
                    settings,
                    auto_approve=body.auto_approve,
                    thread_id=thread_id,
                    max_revisions=body.max_revisions,
                )
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    for chunk in session.graph.stream(state, config, stream_mode="updates"):
                        public = _public_chunk(chunk)
                        if public:
                            yield _sse("update", public)
                except Exception as exc:
                    yield _sse("error", {"message": str(exc)})
                    return
                result = result_from_session(session, thread_id)
                event = "interrupt" if result.interrupted else "done"
                yield _sse(event, result.to_dict())

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/research/{thread_id}/resume", response_model=RunResponse)
    def resume(thread_id: str, body: ResumeRequest) -> RunResponse:
        _require_thread(session, thread_id)
        snapshot = session.graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="Thread is not waiting for review.")
        with app.state.lock:
            result = resume_research(session, thread_id, body.action, body.notes)
        return RunResponse(**result.to_dict())

    @app.get("/v1/threads/{thread_id}", response_model=RunResponse)
    def get_thread(thread_id: str) -> RunResponse:
        _require_thread(session, thread_id)
        return RunResponse(**result_from_session(session, thread_id).to_dict())

    return app


def _thread_id(requested: str | None) -> str:
    if requested:
        if not requested.replace("-", "").replace("_", "").isalnum() or len(requested) > 64:
            raise HTTPException(status_code=422, detail="thread_id must be 1-64 letters, digits, _ or -.")
        return requested
    return uuid4().hex[:12]


def _require_thread(session: Session, thread_id: str) -> None:
    snapshot = session.graph.get_state({"configurable": {"thread_id": thread_id}})
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread.")


def _public_chunk(chunk: dict) -> dict:
    public: dict = {}
    for name, update in chunk.items():
        if name == "__interrupt__" or not isinstance(update, dict):
            continue
        public[name] = {
            "status": update.get("status"),
            "next_step": update.get("next_step"),
            "send_back": update.get("send_back"),
            "revision_count": update.get("revision_count"),
            "trace": update.get("trace") or [],
        }
    return public


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


app = create_app()
