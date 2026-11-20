"""Local document corpus used as the retrieval tool's collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from briefwright.memory.embeddings import HashingEmbedder, cosine, tokenize


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    id: str
    title: str
    text: str
    score: float
    topics: tuple[str, ...]


class Corpus:
    def __init__(self, documents: list[Document], embedder: HashingEmbedder) -> None:
        self.documents = documents
        self.embedder = embedder
        self._vectors = [
            embedder.embed(f"{doc.title}\n{' '.join(doc.topics)}\n{doc.text}") for doc in documents
        ]

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query_vector = self.embedder.embed(query)
        query_terms = set(tokenize(query))
        ranked: list[SearchHit] = []
        for document, vector in zip(self.documents, self._vectors, strict=True):
            lexical = _keyword_score(query_terms, document)
            semantic = cosine(query_vector, vector)
            score = 0.65 * lexical + 0.35 * semantic
            if score <= 0:
                continue
            ranked.append(
                SearchHit(
                    id=document.id,
                    title=document.title,
                    text=document.text,
                    score=score,
                    topics=document.topics,
                )
            )
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:k]


def _keyword_score(query_terms: set[str], document: Document) -> float:
    if not query_terms:
        return 0.0
    title_terms = set(tokenize(document.title))
    body_terms = set(tokenize(f"{document.text} {' '.join(document.topics)}"))
    title_hits = len(query_terms & title_terms)
    body_hits = len(query_terms & body_terms)
    raw = title_hits * 2 + body_hits
    return min(1.0, raw / min(8, len(query_terms)))


def load_documents() -> list[Document]:
    path = Path(__file__).with_name("documents.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            id=item["id"],
            title=item["title"],
            text=item["text"],
            topics=tuple(item.get("topics") or ()),
        )
        for item in payload
    ]


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    return Corpus(load_documents(), HashingEmbedder())
