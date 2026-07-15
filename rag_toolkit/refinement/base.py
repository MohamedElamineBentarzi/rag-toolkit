"""Refiner: the uniform post-retrieval chain stage (DR-0001 v2, D9).

Everything after retrieval — cross-encoder reranking, sentence-window / parent
expansion, MMR diversity, score floors, near-dup collapse, context compression —
is the *same shape*: `list[ScoredChunk] -> list[ScoredChunk]`. This is the
MongoDB-aggregation half of the composition algebra: a chain of uniform stages
over one data shape, the `$match`/`$sort`/`$limit` of this library. The old
`reranker` kind dissolved into it (a cross-encoder is just one refiner).

Because they share a shape, refiners compose in a list and reorder freely:

    QueryPipeline(retriever, refine=[NeighborExpander(index),
                                     CrossEncoderReranker(...)])

`k` is the caller's final budget — a *hint*. Budget-aware refiners (rerankers)
use it to decide how hard to work; others ignore it. A refiner MAY return more
or fewer than `k` candidates (expansion adds, thresholds drop); the pipeline
enforces the final truncation to `k`, so refiners never have to.
"""

from __future__ import annotations

from abc import abstractmethod

from ..core.component import Component
from ..core.contracts import Query, ScoredChunk

__all__ = ["Refiner"]


class Refiner(Component):
    """Strategy interface: one stage of the post-retrieval chain."""

    kind = "refiner"

    @abstractmethod
    def refine(
        self, query: Query, candidates: list[ScoredChunk], k: int
    ) -> list[ScoredChunk]:
        """Transform the candidate list. `k` is the caller's final budget (a
        hint); the pipeline enforces final truncation to `k`, so a refiner may
        return more or fewer. Only chunks derived from `candidates` (or their
        neighbors, for expansion) should appear."""
