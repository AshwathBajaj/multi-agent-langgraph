"""HTTP request and response bodies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = None
    auto_approve: bool = True
    max_revisions: int = Field(default=2, ge=0, le=4)


class ResumeRequest(BaseModel):
    action: str = Field(pattern="^(approve|revise)$")
    notes: str = ""


class RunResponse(BaseModel):
    thread_id: str
    status: str
    interrupted: bool
    approved: bool
    revision_count: int
    report: str
    draft: str
    query: str
    sources: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    critique: dict[str, Any]
    trace: list[dict[str, Any]]
    usage: list[dict[str, Any]]
    errors: list[str]
    next_nodes: list[str]
    plan: list[str]
    analysis: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
