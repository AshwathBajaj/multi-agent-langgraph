"""Shared dependencies closed over by graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from briefwright.config import Settings
from briefwright.corpus import Corpus, get_corpus
from briefwright.llm.base import LLMClient
from briefwright.llm.factory import create_llm
from briefwright.memory.embeddings import HashingEmbedder
from briefwright.memory.store import VectorStore


@dataclass
class Deps:
    settings: Settings
    llm: LLMClient
    corpus: Corpus
    memory: VectorStore


def build_deps(settings: Settings | None = None) -> Deps:
    settings = settings or Settings()
    return Deps(
        settings=settings,
        llm=create_llm(settings),
        corpus=get_corpus(),
        memory=VectorStore(Path(settings.memory_path), HashingEmbedder()),
    )
