"""HybridRetriever: progressive-disclosure sugar over FusionRetriever.

The common case of fusion: search several representations of *one* index and
blend them. `HybridRetriever(index)` builds an `IndexRetriever` per
representation (all of them by default) and delegates to the same RRF fusion
node — the common case reads like English; the power case (fusing across
indexes or paradigms) is `FusionRetriever` directly (DR-0001 v2, D5/F2b).

There is no `HybridDenseBm25Retriever` and never will be: hybridization is
composition, not a class.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.chunk_index import ChunkIndex
from .base import Retriever
from .fusion_retriever import FusionRetriever
from .index_retriever import IndexRetriever

__all__ = ["HybridRetriever"]


@registry.register
class HybridRetriever(Retriever):
    name = "hybrid"
    version = "0.2.0"

    @dataclass
    class Config:
        rrf_k: int = 60
        fetch_k: int = 60
        weights: Optional[list[float]] = field(default=None)

    def __init__(
        self,
        index: ChunkIndex | None = None,
        representations: Optional[Sequence[str]] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if index is None:
            raise ConfigError(
                "HybridRetriever must be built with index= (a ChunkIndex), "
                "not by name alone"
            )
        self.index = index
        reps = list(representations) if representations else index.representations()
        if not reps:
            raise ConfigError("HybridRetriever: the index declares no representations")
        self.representations = reps
        # Sugar: one IndexRetriever per representation, fused by RRF. The fusion
        # node owns all the mechanics — this class only picks the sub-retrievers.
        self._fusion = FusionRetriever(
            [IndexRetriever(index, rep) for rep in reps],
            rrf_k=self.config.rrf_k,
            fetch_k=self.config.fetch_k,
            weights=self.config.weights,
        )

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        # Re-stamp as "hybrid" so this node is attributable as one unit while
        # per-representation attribution survives in metadata["sources"].
        return [
            replace(r, retriever_name=self.name)
            for r in self._fusion.retrieve(query, k)
        ]

    def describe(self) -> dict:
        info = super().describe()
        info["index_fingerprint"] = self.index.fingerprint()
        info["representations"] = self.representations
        return info
