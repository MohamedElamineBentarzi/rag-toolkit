"""Reranker: the second-pass relevance Strategy.

Retrieval optimizes for recall cheaply (embed + ANN, or BM25) and returns a
generous candidate list. A reranker then spends more compute to reorder that
short list for precision — classically a cross-encoder that reads the query and
each candidate *together* (unlike the bi-encoder embedder, which encodes them
apart). Interface: `rerank(query, candidates, top_k)` → the best `top_k`,
highest score first.

Keeping this a distinct stage (rather than folding it into the retriever) is
what lets the pipeline run "retrieve 50 → rerank to 8" as swappable config, and
what makes `NoOpReranker` a first-class option — the honest baseline the tuner
compares every real reranker against.
"""

from __future__ import annotations

from abc import abstractmethod

from ..core.component import Component
from ..core.contracts import Query, ScoredChunk

__all__ = ["Reranker"]


class Reranker(Component):
    """Strategy interface: reorder retrieved candidates for precision."""

    kind = "reranker"

    @abstractmethod
    def rerank(
        self, query: Query, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Return the best `top_k` candidates, highest score first. May return
        fewer than `top_k` (never more), and only chunks drawn from
        `candidates`."""
