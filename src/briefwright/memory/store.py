"""JSON-backed vector memory that survives across runs on one machine."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from briefwright.memory.embeddings import HashingEmbedder, cosine, tokenize

_MAX_RECORDS = 200


@dataclass
class MemoryHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


@dataclass
class _Record:
    id: str
    text: str
    metadata: dict[str, Any]
    vector: list[float]


@dataclass
class VectorStore:
    path: Path
    embedder: HashingEmbedder
    records: list[_Record] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        record_id = uuid.uuid4().hex[:12]
        meta = dict(metadata or {})
        meta.setdefault("created_at", datetime.now(UTC).isoformat(timespec="seconds"))
        self.records.append(
            _Record(
                id=record_id,
                text=text,
                metadata=meta,
                vector=self.embedder.embed(text),
            )
        )
        if len(self.records) > _MAX_RECORDS:
            self.records = self.records[-_MAX_RECORDS:]
        return record_id

    def search(self, query: str, k: int = 3) -> list[MemoryHit]:
        if not self.records or k <= 0:
            return []
        query_vector = self.embedder.embed(query)
        query_terms = set(tokenize(query))
        ranked: list[MemoryHit] = []
        for record in self.records:
            lexical = _keyword_score(query_terms, record.text)
            semantic = cosine(query_vector, record.vector)
            if lexical <= 0.0 and semantic < 0.35:
                continue
            score = 0.6 * lexical + 0.4 * semantic
            ranked.append(MemoryHit(record.id, record.text, score, record.metadata))
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.embedder.dim,
            "records": [
                {
                    "id": record.id,
                    "text": record.text,
                    "metadata": record.metadata,
                    "vector": [round(value, 6) for value in record.vector],
                }
                for record in self.records
            ],
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if int(payload.get("dim", self.embedder.dim)) != self.embedder.dim:
            raise ValueError(
                f"Memory file dim {payload.get('dim')} does not match embedder dim {self.embedder.dim}."
            )
        self.records = [
            _Record(
                id=item["id"],
                text=item["text"],
                metadata=item.get("metadata") or {},
                vector=list(item["vector"]),
            )
            for item in payload.get("records", [])
        ]


def _keyword_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    hits = len(query_terms & set(tokenize(text)))
    return min(1.0, hits / min(6, len(query_terms)))
