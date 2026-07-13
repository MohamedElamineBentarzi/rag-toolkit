"""NoOpReranker: the Null Object reranker.

Does nothing but honor the contract: keep the retriever's order and scores, just
cap to `top_k`. Two reasons it earns a name rather than being `reranker=None`:

1. Pipeline code stays branch-free — the query pipeline always calls
   `reranker.rerank(...)`, never `if reranker is not None`. That is the whole
   point of the Null Object pattern.
2. It is the honest baseline. "Does reranking actually help on this dataset?"
   is answered by putting `noop` in the search space next to the real rerankers
   and reading the leaderboard — no special-casing.
"""

from __future__ import annotations

from ..core.contracts import Query, ScoredChunk
from ..core.registry import registry
from .base import Reranker

__all__ = ["NoOpReranker"]


@registry.register
class NoOpReranker(Reranker):
    name = "noop"
    version = "0.1.0"

    def rerank(
        self, query: Query, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        # Candidates already arrive ranked from the retriever; preserve that.
        return list(candidates[:top_k])
