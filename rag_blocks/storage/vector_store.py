"""VectorStore: the named, typed, multi-vector index Strategy interface (v2).

Where it sits: chunks get encoded into one or more *named* representations
("dense", "splade", …), then land here; at query time a query vector for one
named space comes in and the k nearest chunks come out as `ScoredChunk`s. This
is the `kind = "vector_store"` sibling of the `blob_store` kind — same
subsystem, very different job (the blob store is the durable *truth*; the
vector store is a *derived, rebuildable* index).

The load-bearing design decision (AGENTS.md §7.2): **the store duplicates each
chunk's text and provenance (`doc_id`, `index`, `char/page` offsets) alongside
its vectors**, so `search`/`fetch` return fully-formed `Chunk`s and query time
NEVER touches the blob store. Re-embedding with a new model rebuilds this index
from the blob store's markdown; losing it is cheap.

Why multi-vector (DR-0001 v2, D3). One corpus is often searchable several ways
at once — a dense embedding *and* a static sparse (SPLADE) vector, or two dense
models being A/B'd. Rather than one store per representation kept consistent by
convention, a store holds N *named, typed* vector spaces per point. The owning
`ChunkIndex` declares those spaces up front; the store's job is to store, search
and fetch them honestly.

Contract:
- `ensure_schema(specs)` — declare the named vector spaces. Create them, or
  *validate* that an existing collection already matches; on mismatch raise
  `ConfigError`. NEVER silently coerce (fail fast beats lossy).
- `upsert(chunks, vectors)` — `vectors` maps each declared name to a sequence
  parallel to `chunks`. One payload write, N named vectors per point. Keyed by
  `chunk.id` (deterministic `doc_id:index`), so re-upserting OVERWRITES rather
  than duplicates — indexing is idempotent.
- `search(name, vector, k, filters=None)` — up to `k` `ScoredChunk`s from ONE
  named space, highest score first. An empty/absent space yields `[]`.
- `fetch(filters, limit)` — point retrieval WITHOUT a query vector (neighbor
  expansion, get-by-(doc_id, index), dedup, staleness scans). List filter
  values mean membership: `{"doc_id": d, "index": [3, 5]}`.
- `update_vectors(name, chunk_ids, vectors)` — replace ONE representation on
  existing points, payload and sibling vectors untouched. Default raises
  (`StorageError`); backends that can, override.
- `persist()` — flush to durable storage; default no-op.

`filters` semantics are shared across stores, lexical indexes and every fused
sub-search: scalar value ⇒ equality; list value ⇒ membership.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Optional, Sequence

from ..core.component import Component
from ..core.contracts import Chunk, ScoredChunk, VectorSpec, VectorValue
from ..core.errors import StorageError

__all__ = ["VectorStore"]


class VectorStore(Component):
    """Strategy interface: a named, typed, multi-vector upsert-and-search index."""

    kind = "vector_store"

    @abstractmethod
    def ensure_schema(self, specs: Sequence[VectorSpec]) -> None:
        """Declare the named vector spaces this store holds.

        Create them if absent, or validate that an existing collection already
        matches `specs`; on any mismatch raise `ConfigError`. Never coerce."""

    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Mapping[str, Sequence[VectorValue]],
    ) -> None:
        """Insert/overwrite `chunks` with their named `vectors`.

        Every mapping value is a sequence parallel to `chunks` (same order,
        same length); every key must be a declared space. Idempotent by
        `chunk.id`."""

    @abstractmethod
    def search(
        self,
        name: str,
        vector: VectorValue,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        """Return up to `k` nearest chunks in the named space, highest first."""

    @abstractmethod
    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        """Point retrieval without a query vector. List filter values mean
        membership: ``{"doc_id": d, "index": [3, 5]}``."""

    def update_vectors(
        self,
        name: str,
        chunk_ids: Sequence[str],
        vectors: Sequence[VectorValue],
    ) -> None:
        """Replace one representation on existing points. Optional capability;
        the default declares it unsupported rather than faking it."""
        raise StorageError(
            f"{type(self).__name__}: partial vector updates not supported"
        )

    def persist(self) -> None:
        """Flush to durable storage. Default: nothing to do."""
