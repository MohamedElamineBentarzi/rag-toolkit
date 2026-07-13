"""MemoryVectorStore: an in-process, zero-dependency vector index.

Pure-Python cosine search over dicts. Not built for scale — built to be the
honest, dependency-free store that the test suite and the auto-tuner lean on for
small corpora, and the reference implementation of the `store` contract. Pairs
naturally with `HashingEmbedder` for a fully hermetic index→search loop.

Idempotency falls out for free: everything is keyed by `chunk.id` (deterministic
`doc_id:index`), so re-upserting overwrites. Ephemeral by design — `persist` is
a no-op; reach for `qdrant` when you need durability or scale.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from ..core.contracts import Chunk, ScoredChunk
from ..core.errors import StorageError
from ..core.registry import registry
from .vector_store import VectorStore

__all__ = ["MemoryVectorStore"]


@registry.register
class MemoryVectorStore(VectorStore):
    name = "memory"
    version = "0.1.0"

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise StorageError(
                f"upsert got {len(chunks)} chunks but {len(vectors)} vectors"
            )
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = list(vector)

    def search(
        self, vector: list[float], k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        scored: list[ScoredChunk] = []
        for chunk_id, stored in self._vectors.items():
            chunk = self._chunks[chunk_id]
            if filters and not _matches(chunk, filters):
                continue
            scored.append(ScoredChunk(chunk=chunk, score=_cosine(vector, stored)))
        # Highest score first; ties broken by chunk id for a stable order.
        scored.sort(key=lambda sc: (sc.score, sc.chunk.id), reverse=True)
        return scored[:k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _matches(chunk: Chunk, filters: dict) -> bool:
    """Payload-equality filter: each key must equal a Chunk field (doc_id,
    index, page_start, …) or a `chunk.metadata` entry."""
    for key, expected in filters.items():
        actual = getattr(chunk, key, None)
        if actual is None:
            actual = chunk.metadata.get(key)
        if actual != expected:
            return False
    return True
