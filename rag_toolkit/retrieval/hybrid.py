"""HybridRetriever: fuse several retrievers with Reciprocal Rank Fusion.

The payoff of keeping dense and lexical retrieval as separate narrow interfaces:
a hybrid retriever simply *contains* other retrievers and blends their rankings
(Composition over inheritance — there is no `HybridDenseBm25Retriever` class,
and adding a third retriever is a constructor argument, not a new type).

Why RRF (Reciprocal Rank Fusion) and not score averaging: a dense retriever's
scores are cosine similarities, a BM25 retriever's are unbounded term scores —
averaging them is meaningless (different scales, different distributions). RRF
throws the raw scores away and fuses on *rank* alone:

    fused(d) = Σ_r  weight_r · 1 / (k_rrf + rank_r(d))

where `rank_r(d)` is d's 1-based position in retriever r's list (a document
missing from r contributes nothing). `k_rrf` (default 60, the value from the
original RRF paper) damps the influence of top ranks so a single retriever can't
dominate. Optional per-retriever `weights` give the "weighted" fusion variant.

Each sub-retriever is over-fetched to `candidates` depth so fusion sees enough
of each ranking; the fused top-`k` is returned, stamped `retriever_name="hybrid"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..core.contracts import Chunk, Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from .base import Retriever

__all__ = ["HybridRetriever"]


@registry.register
class HybridRetriever(Retriever):
    name = "hybrid"
    version = "0.1.0"

    @dataclass
    class Config:
        k_rrf: int = 60        # RRF damping constant (from the original paper)
        candidates: int = 60   # depth pulled from each sub-retriever before fusing
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
                "HybridRetriever must be built with retrievers=[...] (>= 1)"
            )
        self.retrievers = list(retrievers)
        weights = self.config.weights
        if weights is not None and len(weights) != len(self.retrievers):
            raise ConfigError(
                f"weights has {len(weights)} entries but there are "
                f"{len(self.retrievers)} retrievers"
            )

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        weights = self.config.weights or [1.0] * len(self.retrievers)
        depth = max(k, self.config.candidates)
        k_rrf = self.config.k_rrf

        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever, weight in zip(self.retrievers, weights):
            for rank, scored in enumerate(retriever.retrieve(query, depth), start=1):
                cid = scored.chunk.id
                fused[cid] = fused.get(cid, 0.0) + weight / (k_rrf + rank)
                chunks.setdefault(cid, scored.chunk)

        ranked = sorted(fused.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return [
            ScoredChunk(chunk=chunks[cid], score=score, retriever_name=self.name)
            for cid, score in ranked[:k]
        ]

    def describe(self) -> dict:
        info = super().describe()
        info["retriever_fingerprints"] = [r.fingerprint() for r in self.retrievers]
        return info
