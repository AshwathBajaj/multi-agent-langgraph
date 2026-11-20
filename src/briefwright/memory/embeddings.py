"""Deterministic hashing embedder. Same text always maps to the same vector."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "with",
        "from",
        "by",
        "how",
        "should",
        "what",
        "when",
        "is",
        "are",
        "be",
        "this",
        "that",
        "it",
        "as",
        "at",
        "into",
        "over",
        "than",
        "not",
        "can",
        "could",
        "include",
        "using",
        "use",
        "about",
        "your",
        "you",
        "we",
        "our",
    }
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9@]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class HashingEmbedder:
    """Bag-of-tokens hashing trick. No model download and no network."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
