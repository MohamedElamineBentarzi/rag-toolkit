"""IndexRetriever: a read-only view over one representation of a Corpus.

The collapse point of the old retriever zoo (DR-0001 v2, D5): `DenseRetriever`
and `Bm25Retriever` were two thin adapters doing the same thing over different
backends. Because `Corpus.search(representation, text, k)` is uniform — the
corpus owns query encoding for *every* representation — one retriever covers them
all: dense, static-sparse, or lexical, selected by name (DR-0004 D4: the
retriever addresses a corpus by space name, never a bare representation).

Read-only by design: the retriever never writes. *What representations exist* is
decided once, expensively, at index time; *how to query them* is decided many
times, cheaply, here. That split is what lets the tuner index once and enumerate
retrieval strategies for free (the G10 acceptance test).

Progressive disclosure: `representation` is optional when the corpus has exactly
one — the common case reads `IndexRetriever(corpus)`. Ambiguity (several
representations, none named) fails fast, listing the options.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.corpus import Corpus
from .base import Retriever

__all__ = ["IndexRetriever"]


@registry.register
class IndexRetriever(Retriever):
    name = "index"
    version = "0.1.0"

    def __init__(
        self,
        corpus: Corpus | None = None,
        representation: Optional[str] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if corpus is None:
            raise ConfigError(
                "IndexRetriever must be built with corpus= (a Corpus), "
                "not by name alone"
            )
        self.corpus = corpus
        reps = corpus.representations()
        if representation is None:
            if len(reps) != 1:
                raise ConfigError(
                    f"IndexRetriever: corpus has {len(reps)} representations "
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
        results = self.corpus.search(self.representation, query.text, k, query.filters)
        return [replace(r, retriever_name=self.name) for r in results]

    @property
    def label(self) -> str:
        # Distinguish sibling views (index:dense vs index:lexical) under fusion.
        return f"{self.name}:{self.representation}"

    def describe(self) -> dict:
        info = super().describe()
        info["corpus_fingerprint"] = self.corpus.fingerprint()
        info["representation"] = self.representation
        return info
