"""Optional local MiniLM embedder. Imported only when fastembed is installed."""

from __future__ import annotations

import math
from pathlib import Path

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE = Path("data/cache/fastembed")


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via ONNX, on CPU. Apache-2.0 weights."""

    name = MODEL_NAME
    dim = 384

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._model = TextEmbedding(MODEL_NAME, cache_dir=str(cache_dir or _CACHE))

    def embed(self, text: str) -> list[float]:
        vector = [float(value) for value in next(iter(self._model.embed([text])))]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
