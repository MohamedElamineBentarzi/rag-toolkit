"""IndexRetriever: a read-only view over one representation of a ChunkIndex.

The collapse point of the old retriever zoo (DR-0001 v2, D5): `DenseRetriever`
and `Bm25Retriever` were two thin adapters doing the same thing over different
backends. Because `ChunkIndex.search(representation, text, k)` is uniform — it
owns query encoding for *every* representation — one retriever covers them all:
dense, static-sparse, or lexical, selected by name.

Read-only by design: the retriever never writes. *What representations exist* is
decided once, expensively, at index time; *how to query them* is decided many
times, cheaply, here. That split is what lets the tuner index once and enumerate
retrieval strategies for free (the G10 acceptance test).

Progressive disclosure: `representation` is optional when the index has exactly
one — the common case reads `IndexRetriever(index)`. Ambiguity (several
representations, none named) fails fast, listing the options.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.chunk_index import ChunkIndex
from .base import Retriever

__all__ = ["IndexRetriever"]


@registry.register
class IndexRetriever(Retriever):
    name = "index"
    version = "0.1.0"

    def __init__(
        self,
        index: ChunkIndex | None = None,
        representation: Optional[str] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if index is None:
            raise ConfigError(
                "IndexRetriever must be built with index= (a ChunkIndex), "
                "not by name alone"
            )
        self.index = index
        reps = index.representations()
        if representation is None:
            if len(reps) != 1:
                raise ConfigError(
                    f"IndexRetriever: index has {len(reps)} representations "
                    f"{reps}; pass representation= to pick one"
                )
            representation = reps[0]
        elif representation not in reps:
            raise ConfigError(
                f"IndexRetriever: no representation {representation!r}; "
                f"available: {reps}"
            )
        self.representation = representation

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        results = self.index.search(self.representation, query.text, k, query.filters)
        return [replace(r, retriever_name=self.name) for r in results]

    @property
    def label(self) -> str:
        # Distinguish sibling views (index:dense vs index:lexical) under fusion.
        return f"{self.name}:{self.representation}"

    def describe(self) -> dict:
        info = super().describe()
        info["index_fingerprint"] = self.index.fingerprint()
        info["representation"] = self.representation
        return info
