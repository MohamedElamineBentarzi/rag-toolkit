"""FusionRetriever: the general composition node — fuse any retrievers.

The one node that generalizes hybrid retrieval (DR-0001 v2, F2b). Where the old
`HybridRetriever` fused representations of one index, `FusionRetriever` fuses
*any* retrievers: across representations, across indexes (multi-corpus
federation), across paradigms (a vector `IndexRetriever` beside a graph
retriever). A retriever wrapping retrievers, like `nn.Sequential` wraps modules.

Each sub-retriever is over-fetched to `fetch_k` depth so fusion sees enough of
each ranking, then their rankings are blended by RRF (see `fusion.py`). Filters
fan out for free: the same `Query` (filters included) goes to every
sub-retriever, which forwards them to its backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from .base import Retriever
from .fusion import fuse, source_labels

__all__ = ["FusionRetriever"]


@registry.register
class FusionRetriever(Retriever):
    name = "fusion"
    version = "0.1.0"

    @dataclass
    class Config:
        fusion: str = "rrf"     # only "rrf" today; the seam for future methods
        rrf_k: int = 60         # RRF damping constant (from the original paper)
        fetch_k: int = 60       # depth pulled from each sub-retriever before fusing
        #: Optional per-retriever weights (same order as `retrievers`); None ⇒
        #: equal weight. This is the "weighted" fusion variant.
        weights: Optional[list[float]] = field(default=None)

    def __init__(
        self,
        retrievers: Optional[Sequence[Retriever]] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if not retrievers:
            raise ConfigError(
                "FusionRetriever must be built with retrievers=[...] (>= 1)"
            )
        if self.config.fusion != "rrf":
            raise ConfigError(
                f"FusionRetriever: unknown fusion {self.config.fusion!r} "
                "(only 'rrf' is supported)"
            )
        self.retrievers = list(retrievers)
        weights = self.config.weights
        if weights is not None and len(weights) != len(self.retrievers):
            raise ConfigError(
                f"weights has {len(weights)} entries but there are "
                f"{len(self.retrievers)} retrievers"
            )

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        depth = max(k, self.config.fetch_k)
        labels = source_labels([r.label for r in self.retrievers])
        rankings = [
            (label, retriever.retrieve(query, depth))
            for label, retriever in zip(labels, self.retrievers)
        ]
        return fuse(
            rankings, k=k, rrf_k=self.config.rrf_k,
            weights=self.config.weights, name=self.name,
        )

    def describe(self) -> dict:
        info = super().describe()
        info["retriever_fingerprints"] = [r.fingerprint() for r in self.retrievers]
        return info
