"""Representation: how a chunk is made searchable under one named space.

The Strategy half of DR-0004's Coordinator+Strategy split. A `Representation` is
a **pure projection**: it declares the storage it needs and encodes corpus + a
query with one encoder — and it holds **no store and does no I/O**. That is what
keeps it a Strategy (registry-instantiable from a flat `{name, params}` spec,
stateless with respect to the backend) instead of the stateful infrastructure a
store reference would make it. The `Corpus` (see `corpus.py`) is the single
owner of the `VectorStore` and performs every read/write; a representation never
touches a database. Design B, chosen over "representation owns search against a
store it is bound to" precisely to avoid that store reference and its temporal
coupling (DR-0004 D1).

Two families live under one interface, because two scoring models do (DR-0001
D4, carried forward):

- **Vector-backed** (dense, static sparse): `declare_schema()` returns the named
  vector space(s) it needs, and `encode_corpus`/`encode_query` turn text into the
  `VectorValue`s the Corpus stores and searches. The Corpus owns the I/O.
- **Self-managed** (classic corpus-relative BM25): declares *no* vector space
  (BM25 is not a static per-chunk vector), and instead owns its own backend
  through `ingest` (write) and `search` (read). The Corpus just delegates to it.

The Corpus discriminates the two by whether `declare_schema()` is non-empty, so
a concrete class overrides only the half it uses; the base supplies safe
defaults for the other half. Like retrievers and the old `ChunkIndex`, a
representation is a `Component` for identity/fingerprint but is *wired from a
live encoder* — built with its encoder instance (or from a nested sub-spec the
builder resolves), never by `registry.create` alone.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..core.component import Component
from ..core.contracts import Chunk, ScoredChunk, VectorSpec, VectorValue
from ..core.errors import ConfigError
from ..core.registry import registry
from ..embedding.base import Embedder
from ..embedding.sparse import SparseEncoder
from ..storage.lexical_index import LexicalIndex

__all__ = [
    "Representation",
    "DenseRepresentation",
    "SparseRepresentation",
    "LexicalRepresentation",
]


class Representation(Component):
    """Strategy: how a chunk is made searchable under one named space.

    Pure projection — owns encoding + schema declaration, never storage. See the
    module docstring for the two families (vector-backed vs self-managed) and
    why a representation holds no store.
    """

    kind = "representation"

    @property
    def space(self) -> str:
        """The name this representation mounts under in a `Corpus` — the address
        a retriever queries. Defaults to the registry `name`; a concrete class
        lets it be overridden to A/B two encoders of the same kind."""
        return self.name

    @property
    def encoder(self) -> Optional[Component]:
        """The underlying encoder/backend component this representation wraps
        (an `Embedder`, `SparseEncoder`, or `LexicalIndex`) — for trial-log
        introspection, which records *which* encoder ran under a space. `None`
        for a representation that wraps no single component."""
        return None

    # -- vector-backed family (default: not vector-backed) -------------------

    def declare_schema(self) -> Sequence[VectorSpec]:
        """The named vector spaces this representation needs in the shared store.
        Non-empty ⇒ vector-backed (the Corpus stores/searches for it); empty
        (default) ⇒ self-managed."""
        return ()

    def encode_corpus(
        self, chunks: Sequence[Chunk]
    ) -> Mapping[str, Sequence[VectorValue]]:
        """Encode a batch into named vectors for the Corpus's single upsert.
        Keys are declared space names; each value is parallel to `chunks`.
        Default empty — self-managed reps contribute through `ingest` instead,
        and this method NEVER performs I/O."""
        return {}

    def encode_query(self, text: str) -> Mapping[str, VectorValue]:
        """Encode a query into its named vector(s). MUST use the same encoder as
        `encode_corpus` — this method *is* the query/corpus parity guarantee.
        Not called for self-managed reps."""
        return {}

    # -- self-managed family (default: no-op / unsupported) ------------------

    def ingest(self, chunks: Sequence[Chunk]) -> None:
        """Write a batch to a self-owned backend (BM25's `LexicalIndex`).
        Idempotent by `chunk.id`. Default no-op: vector reps contribute through
        `encode_corpus`, not here."""

    def search(
        self, text: str, k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        """Self-managed query path (BM25 against its own index). Vector reps
        never implement this — the Corpus searches the shared store for them.
        Default raises: a vector rep reaching here is a routing bug."""
        raise NotImplementedError(
            f"{type(self).__name__} is vector-backed; the Corpus searches the "
            f"store for it. Only self-managed representations implement search()."
        )

    def persist(self) -> None:
        """Flush a self-owned backend. Default no-op: a vector rep's data lives
        in the shared store, which the Corpus persists."""


@registry.register
class DenseRepresentation(Representation):
    """A dense-embedding space: an `Embedder` mounted as one searchable space."""

    name = "dense"
    version = "0.1.0"

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        space: Optional[str] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if embedder is None:
            raise ConfigError(
                "DenseRepresentation must be built with embedder= (an Embedder), "
                "not by name alone"
            )
        self._embedder = embedder
        self._space = space or self.name

    @property
    def space(self) -> str:
        return self._space

    @property
    def encoder(self) -> Optional[Component]:
        return self._embedder

    def declare_schema(self) -> Sequence[VectorSpec]:
        return [VectorSpec(self._space, "dense",
                           dimensions=self._embedder.dimensions,
                           distance=self._embedder.distance)]

    def encode_corpus(
        self, chunks: Sequence[Chunk]
    ) -> Mapping[str, Sequence[VectorValue]]:
        return {self._space: self._embedder.embed_texts([c.text for c in chunks])}

    def encode_query(self, text: str) -> Mapping[str, VectorValue]:
        return {self._space: self._embedder.embed_query(text)}

    def describe(self) -> dict:
        info = super().describe()
        info["space"] = self._space
        info["encoder"] = self._embedder.fingerprint()
        return info


@registry.register
class SparseRepresentation(Representation):
    """A static-sparse (SPLADE-style) space: a `SparseEncoder` mounted as one
    searchable space."""

    name = "sparse"
    version = "0.1.0"

    def __init__(
        self,
        encoder: Optional[SparseEncoder] = None,
        space: Optional[str] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if encoder is None:
            raise ConfigError(
                "SparseRepresentation must be built with encoder= (a "
                "SparseEncoder), not by name alone"
            )
        self._encoder = encoder
        self._space = space or self.name

    @property
    def space(self) -> str:
        return self._space

    @property
    def encoder(self) -> Optional[Component]:
        return self._encoder

    def declare_schema(self) -> Sequence[VectorSpec]:
        return [VectorSpec(self._space, "sparse")]

    def encode_corpus(
        self, chunks: Sequence[Chunk]
    ) -> Mapping[str, Sequence[VectorValue]]:
        return {self._space: self._encoder.encode_texts([c.text for c in chunks])}

    def encode_query(self, text: str) -> Mapping[str, VectorValue]:
        return {self._space: self._encoder.encode_query(text)}

    def describe(self) -> dict:
        info = super().describe()
        info["space"] = self._space
        info["encoder"] = self._encoder.fingerprint()
        return info


@registry.register
class LexicalRepresentation(Representation):
    """Classic corpus-relative BM25: a self-managed `LexicalIndex`, mounted as a
    space the Corpus searches without ever touching the vector store."""

    name = "lexical"
    version = "0.1.0"

    def __init__(
        self,
        index: Optional[LexicalIndex] = None,
        space: Optional[str] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if index is None:
            raise ConfigError(
                "LexicalRepresentation must be built with index= (a "
                "LexicalIndex), not by name alone"
            )
        self._index = index
        self._space = space or self.name

    @property
    def space(self) -> str:
        return self._space

    @property
    def encoder(self) -> Optional[Component]:
        return self._index

    # declare_schema inherited -> ()  (self-managed: no vector space)

    def ingest(self, chunks: Sequence[Chunk]) -> None:
        self._index.add(chunks)

    def search(
        self, text: str, k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        return self._index.search(text, k, filters)

    def persist(self) -> None:
        self._index.persist()

    def describe(self) -> dict:
        info = super().describe()
        info["space"] = self._space
        info["backend"] = self._index.fingerprint()
        return info
