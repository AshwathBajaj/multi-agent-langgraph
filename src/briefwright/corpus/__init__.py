"""SciFact retrieval index. The committed sample stores real MiniLM vectors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from briefwright.memory.embeddings import cosine, tokenize
from briefwright.retrieval_metrics import mean_precision

_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    id: str
    title: str
    text: str
    score: float
    topics: tuple[str, ...] = ()


class Corpus:
    """Dense index over SciFact abstracts.

    Document vectors and the vectors for the bundled claims are stored next to
    the sample, so search works without downloading the model. A query that is
    not in that cache is embedded with MiniLM when the optional dependency is
    installed, and otherwise falls back to keyword overlap.
    """

    def __init__(
        self,
        documents: list[Document],
        queries: list[LabeledQuery],
        qrels: dict[str, set[str]],
        doc_vectors: dict[str, list[float]],
        query_vectors: dict[str, list[float]],
        meta: dict,
    ) -> None:
        self.documents = documents
        self.queries = queries
        self.qrels = qrels
        self.meta = meta
        self.embedder_name = str(meta.get("embedder") or "sentence-transformers/all-MiniLM-L6-v2")
        self._doc_vectors = doc_vectors
        self._query_vectors = query_vectors
        self._queries_by_text = {_norm(query.text): query for query in queries}
        self._live = None
        self._live_checked = False

    def match_query(self, text: str) -> LabeledQuery | None:
        folded = _norm(text)
        exact = self._queries_by_text.get(folded)
        if exact is not None:
            return exact
        contained = [query for query in self.queries if _norm(query.text) in folded]
        if not contained:
            return None
        return max(contained, key=lambda query: len(query.text))

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        vector = self.embed_query(query)
        query_terms = set(tokenize(query))
        scored: list[SearchHit] = []
        for document in self.documents:
            lexical = _keyword_score(query_terms, document)
            if vector is None:
                score = lexical
            else:
                score = cosine(vector, self._doc_vectors[document.id])
            if score <= 0:
                continue
            scored.append(
                SearchHit(
                    id=document.id,
                    title=document.title,
                    text=document.text,
                    score=score,
                    topics=document.topics,
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def embed_query(self, text: str) -> list[float] | None:
        matched = self.match_query(text)
        if matched is not None and matched.id in self._query_vectors:
            return self._query_vectors[matched.id]
        live = self._live_embedder()
        if live is None:
            return None
        return live.embed(text)

    def judgment_rows(self, k: int = 5) -> list[dict]:
        rows = []
        for query in self.queries:
            hits = self.search(query.text, k=k)
            relevant = self.qrels.get(query.id, set())
            rows.append(
                {
                    "query_id": query.id,
                    "relevant": [1 if hit.id in relevant else 0 for hit in hits],
                }
            )
        return rows

    def mean_precision(self, ks: tuple[int, ...] = (1, 3, 5)) -> dict[int, float]:
        rows = [row["relevant"] for row in self.judgment_rows(k=max(ks))]
        return mean_precision(rows, ks)

    def describe(self) -> str:
        kind = "committed SciFact sample" if self.meta.get("sample") else "SciFact index"
        return (
            f"Dense retrieval with {self.embedder_name} over the {kind} "
            f"({len(self.documents)} abstracts, {len(self.queries)} labeled SciFact dev claims). "
            "A document is relevant when the SciFact dev set lists it as evidence for the claim. "
            "Those labels are not from a language model. The Python tool computed the means below."
        )

    def _live_embedder(self):
        if self._live_checked:
            return self._live
        self._live_checked = True
        try:
            from briefwright.memory.minilm import MiniLMEmbedder
        except ImportError:
            self._live = None
            return None
        try:
            self._live = MiniLMEmbedder()
        except (OSError, RuntimeError, ValueError):
            self._live = None
        return self._live


def packaged_sample_dir() -> Path:
    return Path(__file__).resolve().parent / "sample"


def load_corpus(path: Path | None = None) -> Corpus:
    directory = Path(path) if path else packaged_sample_dir()
    documents = [
        Document(id=str(item["id"]), title=item["title"], text=item["text"])
        for item in json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    ]
    queries = [
        LabeledQuery(id=str(item["id"]), text=item["text"])
        for item in json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    ]
    raw_qrels = json.loads((directory / "qrels.json").read_text(encoding="utf-8"))
    qrels = {str(key): {str(doc_id) for doc_id in value} for key, value in raw_qrels.items()}
    vectors = json.loads((directory / "vectors.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    doc_vectors = {str(key): list(value) for key, value in vectors["documents"].items()}
    query_vectors = {str(key): list(value) for key, value in vectors["queries"].items()}
    missing = [doc.id for doc in documents if doc.id not in doc_vectors]
    if missing:
        raise ValueError(f"Sample is missing vectors for {missing[:3]}")
    return Corpus(documents, queries, qrels, doc_vectors, query_vectors, meta)


@lru_cache(maxsize=4)
def get_corpus(path: str = "") -> Corpus:
    return load_corpus(Path(path) if path else None)


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def _keyword_score(query_terms: set[str], document: Document) -> float:
    if not query_terms:
        return 0.0
    title_terms = set(tokenize(document.title))
    body_terms = set(tokenize(document.text))
    raw = 2 * len(query_terms & title_terms) + len(query_terms & body_terms)
    return min(1.0, raw / min(8, len(query_terms)))
