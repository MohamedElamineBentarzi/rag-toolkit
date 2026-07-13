"""Bm25Retriever: term-based retrieval over a LexicalIndex.

The sparse counterpart to `DenseRetriever` — the thin adapter that fits a
`LexicalIndex` (BM25 and friends) to the `Query` contract. All the term math
lives in the index; this class just forwards the query text and filters and
stamps `retriever_name` so its hits stay attributable when a hybrid retriever
fuses them with dense results.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..storage.lexical_index import LexicalIndex
from .base import Retriever

__all__ = ["Bm25Retriever"]


@registry.register
class Bm25Retriever(Retriever):
    name = "bm25"
    version = "0.1.0"

    def __init__(
        self,
        index: LexicalIndex | None = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if index is None:
            raise ConfigError(
                "Bm25Retriever must be built with index= (a populated "
                "LexicalIndex), not by name alone"
            )
        self.index = index

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        results = self.index.search(query.text, k, query.filters)
        return [replace(r, retriever_name=self.name) for r in results]

    def describe(self) -> dict:
        info = super().describe()
        info["index_fingerprint"] = self.index.fingerprint()
        return info
