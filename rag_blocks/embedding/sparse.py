"""SparseEncoder: the static-sparse vectorization Strategy interface.

The sparse sibling of `Embedder`. Where a dense `Embedder` turns text into a
`list[float]` in a learned continuous space, a `SparseEncoder` turns text into
a `SparseVector` — a handful of (term-index, weight) pairs, SPLADE-style — that
a `VectorStore` stores under a named *sparse* space (DR-0001 v2, D4).

Two scoring models, both first-class (D4): classic corpus-relative BM25 is a
`LexicalIndex` (query-time idf/avgdl, not a per-chunk vector), mounted inside a
`ChunkIndex`; *static* sparse is this — a genuine per-chunk `SparseVector`
frozen at encode time, with engine-side idf if the store offers it (Qdrant's
IDF modifier). Different mechanism, same read API through the index.

Same passage/query asymmetry as `Embedder`, and for the same reason:
instruction-tuned sparse models weight a query differently from a passage, so
`encode_texts` (passages) and `encode_query` (one query) are separate methods —
symmetric encoders just implement both the same way.

Concrete encoders ship as lazy-import extras and have not landed yet (the
contract landed in v0.6; encoders are fast-follow); the memory store + this
contract stay the zero-dep reference.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from ..core.component import Component
from ..core.contracts import SparseVector

__all__ = ["SparseEncoder"]


class SparseEncoder(Component):
    """Strategy interface: text → static sparse vector."""

    kind = "sparse_encoder"

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> list[SparseVector]:
        """Encode a batch of *passages*, one `SparseVector` per input, order
        preserved. `encode_texts([])` returns `[]`. Raise `EmbeddingError` on
        failure."""

    @abstractmethod
    def encode_query(self, text: str) -> SparseVector:
        """Encode a single *query*. Separate from `encode_texts` so any
        query-side weighting asymmetry lives here and nowhere else."""
