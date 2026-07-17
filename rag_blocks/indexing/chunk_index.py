"""ChunkIndex: the aggregate that owns every retrieval representation of a corpus.

`ChunkIndex` is to representations what `Document` is to pages: a consistency
boundary. It guards one invariant — *every representation of every chunk in this
corpus was produced by the encoders this index declares, and queries are encoded
the same way* — closing the old split between write ingredients (embedder+store,
once hardcoded in the pipeline) and read ingredients (a retriever) kept
consistent only by convention (DR-0001 v2, D1).

One corpus can be searchable several ways at once — a dense embedding *and* a
static-sparse (SPLADE) vector *and* classic BM25. Each is a named
*representation*. `add(chunks)` writes them all in one pass;
`search(representation, TEXT, k)` encodes the query with the *same* encoder that
encoded the corpus — that single line is the query/corpus compatibility
guarantee (P6). Note the shape: **text in, not a vector** — the index owns query
encoding so no caller can reimplement it inconsistently.

Two scoring models live side by side under one read API (D4): dense/static-sparse
vectors are stored in the multi-vector `VectorStore`; classic BM25 is a
corpus-relative `LexicalIndex` (query-time idf/avgdl — not a per-chunk vector),
mounted here as representation ``"lexical"``. The index dispatches to whichever
owns the named representation; callers never learn the difference.

Progressive disclosure (A1). The common case reads like English —
`ChunkIndex(store, dense=embedder, lexical=Bm25Index())` auto-names the
representations "dense"/"lexical"; the rare multi-representation case
(A/B-ing two dense models) passes a `{name: encoder}` mapping. The rare case's
ceremony never leaks into the common case.

Not a god object: one responsibility — own the representations of one corpus
(schema, writes, reads). Encoding delegates to encoders, storage to the store,
term scoring to the lexical index; no ranking strategy, no parsing/chunking, no
generation. It is a `Component` (for identity/fingerprint, so the tuner can
treat "which representations exist" as cache-key input) but is wired from live,
stateful backends — never built by `registry.create` alone (the retriever
precedent, `retrieval/base.py`).
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from typing import Mapping, Optional, Sequence

from ..core.component import Component
from ..core.contracts import Chunk, ScoredChunk, VectorSpec
from ..core.errors import ConfigError
from ..embedding.base import Embedder
from ..embedding.sparse import SparseEncoder
from ..storage.lexical_index import LexicalIndex
from ..storage.vector_store import VectorStore

__all__ = ["ChunkIndex"]

#: The fixed name classic BM25 mounts under (D4). Static-sparse spaces are user
#: named; corpus-stats lexical is singular, so it needs no naming ceremony.
LEXICAL_NAME = "lexical"


class ChunkIndex(Component):
    """The aggregate owning all representations of one corpus, write + read."""

    kind = "index"
    name = "chunk-index"
    version = "0.1.0"

    def __init__(
        self,
        store: VectorStore,
        dense: Embedder | Mapping[str, Embedder] | None = None,
        sparse: SparseEncoder | Mapping[str, SparseEncoder] | None = None,
        lexical: Optional[LexicalIndex] = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._dense: dict[str, Embedder] = _normalize(dense, "dense")
        self._sparse: dict[str, SparseEncoder] = _normalize(sparse, "sparse")
        self._lexical = lexical

        names = list(self._dense) + list(self._sparse)
        if lexical is not None:
            names.append(LEXICAL_NAME)
        if not names:
            raise ConfigError(
                "ChunkIndex needs at least one representation "
                "(dense=, sparse=, and/or lexical=)."
            )
        dupes = [n for n in set(names) if names.count(n) > 1]
        if dupes:
            raise ConfigError(
                f"ChunkIndex representation names must be unique; "
                f"duplicated: {sorted(dupes)}"
            )
        self._names = names

        # Eager create-or-validate: fail fast on a schema mismatch, never
        # coerce (§8.1). Lexical is not a stored vector space, so it never
        # appears in the store schema.
        self._specs = self._build_specs()
        if self._specs:
            self._store.ensure_schema(self._specs)

    # -- introspection -------------------------------------------------------

    def representations(self) -> list[str]:
        """The named ways a chunk in this corpus is searchable, stable order."""
        return list(self._names)

    def encoders(self) -> dict[str, Component]:
        """Representation name → the component that encodes it, stable order.

        `describe()` reports these as *fingerprints*, which is right for
        identity and useless for reading: a hash cannot tell a trial log which
        embedder ran, or how it was configured. This is the read accessor for
        that question — group by `component.kind` to tell a dense embedder from
        a lexical index. It exists so the evaluation suite never has to reach
        into this object's internals to record what it ran.
        """
        found: dict[str, Component] = {**self._dense, **self._sparse}
        if self._lexical is not None:
            found[LEXICAL_NAME] = self._lexical
        return {name: found[name] for name in self._names}

    def _build_specs(self) -> list[VectorSpec]:
        specs: list[VectorSpec] = []
        for name, emb in self._dense.items():
            specs.append(
                VectorSpec(name, "dense", dimensions=emb.dimensions,
                           distance=emb.distance)
            )
        for name in self._sparse:
            specs.append(VectorSpec(name, "sparse"))
        return specs

    # -- writes --------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Encode every representation and write them all — one store upsert
        (N named vectors per point) plus the lexical index. Batch-scoped:
        O(batch × representations). Best-effort across representations; keyed by
        `chunk.id`, so recovery is idempotent retry (§8.3)."""
        chunks = list(chunks)
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors: dict[str, Sequence] = {}
        for name, emb in self._dense.items():
            vectors[name] = emb.embed_texts(texts)
        for name, enc in self._sparse.items():
            vectors[name] = enc.encode_texts(texts)
        if vectors:
            self._store.upsert(chunks, vectors)
        if self._lexical is not None:
            self._lexical.add(chunks)

    def update_representation(
        self, name: str, chunks: Sequence[Chunk]
    ) -> None:
        """Refresh ONE representation over existing chunks (P9 partial refresh),
        leaving sibling representations and payload untouched."""
        chunks = list(chunks)
        if not chunks:
            return
        texts = [c.text for c in chunks]
        if name in self._dense:
            self._store.update_vectors(
                name, [c.id for c in chunks], self._dense[name].embed_texts(texts)
            )
        elif name in self._sparse:
            self._store.update_vectors(
                name, [c.id for c in chunks], self._sparse[name].encode_texts(texts)
            )
        elif self._is_lexical(name):
            self._lexical.add(chunks)  # type: ignore[union-attr]
        else:
            raise self._unknown(name)

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        representation: str,
        text: str,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        """Search one named representation. Encodes the query with the same
        encoder that encoded the corpus (P6) — text in, not a vector."""
        if representation in self._dense:
            dense_q = self._dense[representation].embed_query(text)
            return self._store.search(representation, dense_q, k, filters)
        if representation in self._sparse:
            sparse_q = self._sparse[representation].encode_query(text)
            return self._store.search(representation, sparse_q, k, filters)
        if self._is_lexical(representation):
            return self._lexical.search(text, k, filters)  # type: ignore[union-attr]
        raise self._unknown(representation)

    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        """Point retrieval without a query vector (neighbor/parent expansion,
        get-by-(doc_id, index), dedup). Reads the vector store's payloads."""
        return self._store.fetch(filters, limit)

    # -- lifecycle -----------------------------------------------------------

    def persist(self) -> None:
        self._store.persist()
        if self._lexical is not None:
            self._lexical.persist()

    def describe(self) -> dict:
        """Fold store + every encoder + lexical into identity, so adding a
        representation (or swapping an encoder) changes the fingerprint and its
        cache keyspace (P8, D8)."""
        info = super().describe()
        info["store_fingerprint"] = self._store.fingerprint()
        info["representations"] = {
            **{name: emb.fingerprint() for name, emb in self._dense.items()},
            **{name: enc.fingerprint() for name, enc in self._sparse.items()},
        }
        if self._lexical is not None:
            info["representations"][LEXICAL_NAME] = self._lexical.fingerprint()
        return info

    # -- helpers -------------------------------------------------------------

    def _is_lexical(self, name: str) -> bool:
        return self._lexical is not None and name == LEXICAL_NAME

    def _unknown(self, name: str) -> ConfigError:
        return ConfigError(
            f"ChunkIndex has no representation {name!r}; "
            f"available: {self.representations()}"
        )


def _normalize(value, default_name: str) -> dict:
    """Progressive disclosure: a bare encoder auto-names to `default_name`; a
    mapping is used verbatim; None means no representation of this kind."""
    if value is None:
        return {}
    if isinstance(value, ABCMapping):
        return dict(value)
    return {default_name: value}
