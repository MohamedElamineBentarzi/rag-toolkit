"""Retriever: the query-time Strategy interface.

A retriever answers one question: given a `Query`, which chunks are relevant?
It is deliberately thin over the stores — a `DenseRetriever` wraps a vector
store + embedder, a `Bm25Retriever` wraps a lexical index, and a future
`HybridRetriever` *contains* two retrievers and fuses them (Composition over
inheritance: no `HybridDenseBm25Retriever` class explosion).

Note on construction: unlike stateless components (parsers, chunkers), a
retriever is composed from *live, populated* backends (the store already holds
the corpus). So retrievers are built with their backend instances, not by
`registry.create(name)` alone — the same way the pipeline is wired from
components. They remain `Component`s for identity/fingerprint so the tuner can
treat retrieval as a search-space dimension; `describe()` folds in the backend
fingerprints so swapping the embedder or store changes the retriever's identity.
"""

from __future__ import annotations

from abc import abstractmethod

from ..core.component import Component
from ..core.contracts import Query, ScoredChunk

__all__ = ["Retriever"]


class Retriever(Component):
    """Strategy interface: Query → ranked ScoredChunks."""

    kind = "retriever"

    @abstractmethod
    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        """Return up to `k` chunks for `query`, highest score first, each
        stamped with `retriever_name` so fused results stay attributable."""
