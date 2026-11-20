"""Retrieval tools. Nodes call these instead of touching indexes inline."""

from __future__ import annotations

from briefwright.corpus import Corpus, SearchHit
from briefwright.memory.store import MemoryHit, VectorStore


def search_corpus(corpus: Corpus, query: str, k: int = 5) -> list[SearchHit]:
    return corpus.search(query, k=k)


def recall_memory(memory: VectorStore, query: str, k: int = 3) -> list[MemoryHit]:
    return memory.search(query, k=k)
