"""DenseRetriever: embed the query, search the vector store.

The dense half of retrieval — the thin adapter that ties an `Embedder` and a
`VectorStore` together behind the `Query` contract. It embeds the query with
`embed_query` (NOT `embed_texts` — the query/passage asymmetry matters) and
hands the vector to the store, passing `query.filters` straight through as the
scoping seam.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..embedding.base import Embedder
from ..storage.vector_store import VectorStore
from .base import Retriever

__all__ = ["DenseRetriever"]


@registry.register
class DenseRetriever(Retriever):
    name = "dense"
    version = "0.1.0"

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if embedder is None or store is None:
            raise ConfigError(
                "DenseRetriever must be built with embedder= and store= "
                "(a populated vector store), not by name alone"
            )
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        vector = self.embedder.embed_query(query.text)
        results = self.store.search(vector, k, query.filters)
        return [replace(r, retriever_name=self.name) for r in results]

    def describe(self) -> dict:
        """Fold in backend identities so the retriever's fingerprint changes
        when its embedder or store does — correct cache invalidation."""
        info = super().describe()
        info["embedder_fingerprint"] = self.embedder.fingerprint()
        info["store_fingerprint"] = self.store.fingerprint()
        return info
