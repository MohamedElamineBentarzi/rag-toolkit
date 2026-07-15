"""LexicalIndex: the term-based (sparse) sibling of VectorStore.

Dense vectors miss exact-term matches (product codes, names, rare jargon); a
lexical index catches them. It is a SEPARATE component kind from VectorStore on
purpose (Interface Segregation): keeping "search by vector" and "search by
terms" as two narrow interfaces is what lets a `hybrid` retriever compose them
cleanly instead of one store growing a fat dual API.

Symmetry with VectorStore: `add(chunks)` is idempotent by `chunk.id`, and
`search` returns fully-formed `ScoredChunk`s (the index keeps the chunk text +
provenance), so the query path never touches the blob store.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Sequence

from ..core.component import Component
from ..core.contracts import Chunk, ScoredChunk

__all__ = ["LexicalIndex"]


class LexicalIndex(Component):
    """Strategy interface: a term-based search index."""

    kind = "lexical_index"

    @abstractmethod
    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index `chunks`. Idempotent by `chunk.id` (re-adding overwrites)."""

    @abstractmethod
    def search(
        self, text: str, k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        """Return up to `k` chunks most relevant to `text`, highest first."""

    def persist(self) -> None:
        """Flush the index to durable storage. Default: nothing to do (an
        in-memory index with no backing store is ephemeral)."""

    def load(self) -> None:
        """Rehydrate the index from durable storage. Default: nothing to do."""
