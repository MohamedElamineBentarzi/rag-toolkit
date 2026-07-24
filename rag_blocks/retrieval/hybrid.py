"""HybridRetriever: progressive-disclosure sugar over FusionRetriever.

The common case of fusion: search several representations of *one* corpus and
blend them. `HybridRetriever(corpus)` builds an `IndexRetriever` per
representation (all of them by default) and delegates to the same RRF fusion
node — the common case reads like English; the power case (fusing across
corpora or paradigms) is `FusionRetriever` directly (DR-0001 v2, D5/F2b).

There is no `HybridDenseBm25Retriever` and never will be: hybridization is
composition, not a class.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.corpus import Corpus
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
        corpus: Corpus | None = None,
        representations: Optional[Sequence[str]] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if corpus is None:
            raise ConfigError(
                "HybridRetriever must be built with corpus= (a Corpus), "
                "not by name alone"
            )
        self.corpus = corpus
        reps = list(representations) if representations else corpus.representations()
        if not reps:
            raise ConfigError("HybridRetriever: the corpus declares no representations")
        self.representations = reps
        # Sugar: one IndexRetriever per representation, fused by RRF. The fusion
        # node owns all the mechanics — this class only picks the sub-retrievers.
        self._fusion = FusionRetriever(
            [IndexRetriever(corpus, rep) for rep in reps],
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
        info["corpus_fingerprint"] = self.corpus.fingerprint()
        info["representations"] = self.representations
        return info
