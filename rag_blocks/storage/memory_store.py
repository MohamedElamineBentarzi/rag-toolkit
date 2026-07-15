"""MemoryVectorStore: an in-process, zero-dependency, multi-vector index.

Pure-Python search over dicts — cosine for dense spaces, dot product for
sparse ones. Not built for scale — built to be the honest, dependency-free
store that the test suite and the auto-tuner lean on for small corpora, and the
*reference implementation* of the `vector_store` contract (v2): named+typed
multi-vector spaces, `fetch` without a query vector, and `update_vectors`.
Pairs naturally with `HashingEmbedder` for a fully hermetic index→search loop.

Idempotency falls out for free: everything is keyed by `chunk.id` (deterministic
`doc_id:index`), so re-upserting overwrites. Ephemeral by design — `persist` is
a no-op; reach for `qdrant` when you need durability or scale.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from ..core.contracts import Chunk, ScoredChunk, SparseVector, VectorSpec, VectorValue
from ..core.errors import ConfigError, StorageError
from ..core.registry import registry
from .filters import matches
from .vector_store import VectorStore

__all__ = ["MemoryVectorStore"]


@registry.register
class MemoryVectorStore(VectorStore):
    name = "memory"
    version = "0.2.0"

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._chunks: dict[str, Chunk] = {}
        # name -> {chunk_id -> vector}
        self._spaces: dict[str, dict[str, VectorValue]] = {}
        self._specs: dict[str, VectorSpec] = {}

    # -- schema --------------------------------------------------------------

    def ensure_schema(self, specs: Sequence[VectorSpec]) -> None:
        for spec in specs:
            existing = self._specs.get(spec.name)
            if existing is None:
                self._specs[spec.name] = spec
                self._spaces.setdefault(spec.name, {})
            elif not _specs_compatible(existing, spec):
                raise ConfigError(
                    f"MemoryVectorStore: schema mismatch for space "
                    f"{spec.name!r}: have {existing}, asked for {spec}"
                )

    # -- writes --------------------------------------------------------------

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Mapping[str, Sequence[VectorValue]],
    ) -> None:
        for name, seq in vectors.items():
            if name not in self._specs:
                raise StorageError(
                    f"upsert to undeclared space {name!r}; call ensure_schema "
                    f"first (have: {sorted(self._specs) or '<none>'})"
                )
            if len(seq) != len(chunks):
                raise StorageError(
                    f"upsert space {name!r}: {len(chunks)} chunks but "
                    f"{len(seq)} vectors"
                )
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        for name, seq in vectors.items():
            space = self._spaces[name]
            for chunk, vector in zip(chunks, seq):
                space[chunk.id] = _copy_vector(vector)

    def update_vectors(
        self,
        name: str,
        chunk_ids: Sequence[str],
        vectors: Sequence[VectorValue],
    ) -> None:
        if name not in self._specs:
            raise StorageError(f"update_vectors: undeclared space {name!r}")
        if len(chunk_ids) != len(vectors):
            raise StorageError(
                f"update_vectors: {len(chunk_ids)} ids but {len(vectors)} vectors"
            )
        space = self._spaces[name]
        for chunk_id, vector in zip(chunk_ids, vectors):
            if chunk_id not in self._chunks:
                raise StorageError(
                    f"update_vectors: no such point {chunk_id!r}", key=chunk_id
                )
            space[chunk_id] = _copy_vector(vector)

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        name: str,
        vector: VectorValue,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        space = self._spaces.get(name)
        if not space:
            return []
        is_sparse = self._specs[name].kind == "sparse"
        score = _sparse_dot if is_sparse else _cosine
        scored: list[ScoredChunk] = []
        for chunk_id, stored in space.items():
            chunk = self._chunks[chunk_id]
            if filters and not matches(chunk, filters):
                continue
            scored.append(ScoredChunk(chunk=chunk, score=score(vector, stored)))
        # Highest score first; ties broken by chunk id for a stable order.
        scored.sort(key=lambda sc: (sc.score, sc.chunk.id), reverse=True)
        return scored[:k]

    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        out: list[Chunk] = []
        for chunk in self._chunks.values():
            if matches(chunk, filters):
                out.append(chunk)
                if len(out) >= limit:
                    break
        return out


# -- scoring & filtering ---------------------------------------------------


def _copy_vector(vector: VectorValue) -> VectorValue:
    # SparseVector is frozen/immutable; only dense lists need defensive copy.
    return vector if isinstance(vector, SparseVector) else list(vector)


def _cosine(a: Any, b: Any) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _sparse_dot(a: Any, b: Any) -> float:
    """Dot product of two sparse vectors over their shared term indices.

    Static-sparse relevance is a plain inner product (idf is baked into the
    encoder's weights), so no length normalization here."""
    bmap = dict(zip(b.indices, b.values))
    return sum(w * bmap.get(i, 0.0) for i, w in zip(a.indices, a.values))


def _specs_compatible(a: VectorSpec, b: VectorSpec) -> bool:
    if a.kind != b.kind:
        return False
    if a.kind == "dense":
        return a.dimensions == b.dimensions and a.distance == b.distance
    return True
