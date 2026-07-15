"""Query-shaping retrievers: pre-retrieval variation *as composition* (F2).

Multi-query expansion and HyDE reshape the query before it hits the backend.
The v2 decision (DR-0001 v2, D5/F2): these are not new pipeline slots — you
don't add a "pre-layer" to `nn.Sequential`, you wrap modules in modules. So each
is a `Retriever` wrapping a `Retriever`: expand/hypothesize with an LLM, retrieve
through the wrapped retriever once per variant, and fuse the results by RRF (the
same `fusion.py` mechanics).

The `complete` seam (F5). These need bare text completion — `(prompt) -> str` —
a shape `Generator` deliberately doesn't expose (its contract is
`(query, context) -> Answer`). Rather than bend the generator or invent a
`completer` kind before a third consumer demands it, they take
`complete: Callable[[str], str]`; `AnthropicGenerator.complete` supplies it. A
fake `complete` makes them fully hermetic to test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from ..core.contracts import Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from .base import Retriever
from .fusion import fuse, source_labels

__all__ = ["MultiQueryRetriever", "HydeRetriever"]


_MULTI_QUERY_PROMPT = (
    "You are helping improve search recall. Generate {n} alternative phrasings "
    "of the following question, each on its own line, no numbering, no extra "
    "text. Vary vocabulary and specificity while preserving intent.\n\n"
    "Question: {query}"
)

_HYDE_PROMPT = (
    "Write a short, factual passage that would directly answer the following "
    "question, as if it were an excerpt from a relevant document. Do not say "
    "you are unsure; write the passage confidently.\n\nQuestion: {query}"
)


@registry.register
class MultiQueryRetriever(Retriever):
    """Expand the query into `n` phrasings, retrieve each, fuse (RAG-fusion)."""

    name = "multi-query"
    version = "0.1.0"

    @dataclass
    class Config:
        n: int = 4
        rrf_k: int = 60

    def __init__(
        self,
        inner: Retriever | None = None,
        complete: Optional[Callable[[str], str]] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if inner is None or complete is None:
            raise ConfigError(
                "MultiQueryRetriever must be built with inner= (a Retriever) "
                "and complete= (a text-completion callable)"
            )
        self.inner = inner
        self._complete = complete

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        variants = self._expansions(query.text)
        # Always include the original query so expansion only ever adds recall.
        queries = [query] + [
            Query(text=v, filters=query.filters, metadata=query.metadata)
            for v in variants
        ]
        labels = source_labels([f"q{i}" for i in range(len(queries))])
        rankings = [
            (label, self.inner.retrieve(q, k)) for label, q in zip(labels, queries)
        ]
        return fuse(rankings, k=k, rrf_k=self.config.rrf_k, name=self.name)

    def _expansions(self, text: str) -> list[str]:
        prompt = _MULTI_QUERY_PROMPT.format(n=self.config.n, query=text)
        raw = self._complete(prompt)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return lines[: self.config.n]

    def describe(self) -> dict:
        info = super().describe()
        info["inner_fingerprint"] = self.inner.fingerprint()
        return info


@registry.register
class HydeRetriever(Retriever):
    """Hypothetical Document Embeddings: retrieve on an LLM-drafted answer."""

    name = "hyde"
    version = "0.1.0"

    def __init__(
        self,
        inner: Retriever | None = None,
        complete: Optional[Callable[[str], str]] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if inner is None or complete is None:
            raise ConfigError(
                "HydeRetriever must be built with inner= (a Retriever) and "
                "complete= (a text-completion callable)"
            )
        self.inner = inner
        self._complete = complete

    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]:
        hypothetical = self._complete(_HYDE_PROMPT.format(query=query.text)).strip()
        # Retrieve on the hypothetical passage but keep the caller's filters.
        shaped = Query(
            text=hypothetical or query.text,
            filters=query.filters,
            metadata=query.metadata,
        )
        results = self.inner.retrieve(shaped, k)
        return [replace(r, retriever_name=self.name) for r in results]

    def describe(self) -> dict:
        info = super().describe()
        info["inner_fingerprint"] = self.inner.fingerprint()
        return info
