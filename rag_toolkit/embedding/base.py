"""Embedder: the vectorization Strategy interface.

Turns text into dense vectors so chunks become searchable by meaning. The one
interface decision that earns its keep: **`embed_query` is separate from
`embed_texts`.** Instruction-tuned models (BGE, E5, …) encode a *query*
differently from a *passage* — typically a query gets a prefixed instruction
("Represent this sentence for searching relevant passages: …") while passages
do not. Collapse the two into one method and every caller silently embeds
queries as passages, quietly wrecking retrieval quality with no error. Splitting
them makes the asymmetry impossible to get wrong (and a no-op for symmetric
models, which just implement both the same way).

`dimensions` is on the interface because the downstream vector store must know
the vector width up front (to allocate an index/collection) — and it is a
property, not a constant, because a model adapter only knows its width after the
model loads.

Batch in, list out: `embed_texts` takes a whole batch so adapters with a real
batched encoder (every GPU model) use it; the pipeline feeds chunk batches
through here. Streaming stays the caller's job (embed one batch at a time).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from ..core.component import Component

__all__ = ["Embedder"]


class Embedder(Component):
    """Strategy interface: text → dense vector."""

    kind = "embedder"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Width of the vectors this embedder produces (the vector store needs
        it to size its index). Stable for the life of the instance."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of *passages*, one vector per input, order preserved.
        `embed_texts([])` returns `[]`. Raise `EmbeddingError` on failure."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single *query*. Separate from `embed_texts` so instruction
        prefixes / pooling differences for queries live here and nowhere else."""
