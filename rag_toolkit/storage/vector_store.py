"""VectorStore: the searchable-index Strategy interface.

Where it sits: chunks get embedded, then land here; at query time a query
vector comes in and the k nearest chunks come out as `ScoredChunk`s. This is
the `kind = "store"` sibling of the `blob_store` kind — same subsystem, very
different job (the blob store is the durable *truth*; the vector store is a
*derived, rebuildable* index).

The load-bearing design decision (AGENTS.md §7.2): **the store duplicates each
chunk's text and provenance (`doc_id`, `index`, `char/page` offsets) alongside
its vector**, so `search` returns fully-formed `Chunk`s and query time NEVER
touches the blob store. Re-embedding with a new model rebuilds this index from
the blob store's markdown; losing it is cheap.

Contract:
- `upsert(chunks, vectors)` — parallel sequences, one vector per chunk. Keyed
  by `chunk.id` (which is deterministic: `doc_id:index`), so re-upserting the
  same chunks OVERWRITES rather than duplicates — indexing is idempotent.
- `search(vector, k, filters=None)` — up to `k` `ScoredChunk`s, highest score
  first. `filters` is payload-equality (e.g. `{"doc_id": x}`) — the seam
  neighbor expansion and scoped retrieval use. An empty store yields `[]`.
- `persist()` — flush to durable storage; default no-op (in-memory and
  server-managed stores have nothing to do).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Sequence

from ..core.component import Component
from ..core.contracts import Chunk, ScoredChunk

__all__ = ["VectorStore"]


class VectorStore(Component):
    """Strategy interface: an upsert-and-search vector index."""

    kind = "store"

    @abstractmethod
    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        """Insert/overwrite `chunks` with their `vectors` (same order, same
        length). Idempotent by `chunk.id`."""

    @abstractmethod
    def search(
        self, vector: list[float], k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        """Return up to `k` nearest chunks, highest score first."""

    def persist(self) -> None:
        """Flush to durable storage. Default: nothing to do."""
