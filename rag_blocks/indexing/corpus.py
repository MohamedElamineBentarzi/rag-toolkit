"""Corpus: the single storage owner that coordinates a set of representations.

The Coordinator half of DR-0004's split, and the replacement for `ChunkIndex`.
Where DR-0001's `ChunkIndex` hardcoded three representation kinds
(`dense=`/`sparse=`/`lexical=`) into its constructor and re-stated that trio
across six files, a `Corpus` takes a *list* of first-class `Representation`
objects and knows nothing about their kinds — adding a new kind is a new
registered `Representation` class, nothing here changes (the Open/Closed fix).

The one invariant it guards, unchanged in spirit from `ChunkIndex`: *every
representation of every chunk in this corpus was produced by the encoders this
corpus declares, and queries are encoded the same way.* Concretely, the Corpus:

- is the **single owner of the `VectorStore`** — the ONLY thing that upserts,
  searches, or fetches. Representations are storeless strategies (DR-0004 D1);
- aggregates every representation's `declare_schema()` into **one** eager
  `ensure_schema` (create-or-validate, fail fast in `__init__`);
- drives a **single-pass write** (Invariant 1): gather every vector-backed rep's
  `encode_corpus` into one bundle → **one** `store.upsert`; then each
  self-managed rep's `ingest` (BM25's own side-write). Never N vector upserts;
- owns **all search** (Invariant 2): `search(space, TEXT, k)` — text in, never a
  vector — routes to the store for a vector-backed space, or to the rep's own
  backend for a self-managed one; the retriever never learns the difference.

It is a `Component` for identity/fingerprint (folding store + each rep, so
changing one rep changes only its slice — Invariant 4) but is *wired from live
backends*, never built by `registry.create` alone. It satisfies the `ChunkSink`
protocol (`add` + `persist`) so the write path can still fan out to it (G9).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.component import Component
from ..core.contracts import Chunk, ScoredChunk, VectorSpec, VectorValue
from ..core.errors import ConfigError
from ..storage.vector_store import VectorStore
from .representation import Representation

__all__ = ["Corpus"]


class Corpus(Component):
    """The single storage owner coordinating a corpus's representations."""

    kind = "corpus"
    name = "corpus"
    version = "0.1.0"

    def __init__(
        self,
        store: VectorStore,
        representations: Sequence[Representation],
    ) -> None:
        super().__init__()
        self._store = store
        reps = list(representations)
        if not reps:
            raise ConfigError(
                "Corpus needs at least one representation (a list of "
                "Representation objects, e.g. [DenseRepresentation(embedder)])."
            )
        spaces = [r.space for r in reps]
        dupes = sorted({s for s in spaces if spaces.count(s) > 1})
        if dupes:
            raise ConfigError(
                f"Corpus representation spaces must be unique; duplicated: {dupes}"
            )
        self._reps = reps
        self._by_space: dict[str, Representation] = {r.space: r for r in reps}

        # Partition into the two families by whether a rep declares vector
        # spaces (DR-0004 D3). Aggregate every declared spec into ONE schema.
        self._vector_backed: list[Representation] = []
        self._self_managed: list[Representation] = []
        self._vector_spaces: set[str] = set()
        specs: list[VectorSpec] = []
        for rep in reps:
            rep_specs = list(rep.declare_schema())
            if rep_specs:
                self._vector_backed.append(rep)
                self._vector_spaces.add(rep.space)
                specs.extend(rep_specs)
            else:
                self._self_managed.append(rep)

        # Eager create-or-validate: fail fast on a schema mismatch (never
        # coerce). Self-managed reps declare no vector space, so an all-lexical
        # corpus touches no store schema.
        if specs:
            self._store.ensure_schema(specs)

    # -- introspection -------------------------------------------------------

    def representations(self) -> list[str]:
        """The named ways a chunk in this corpus is searchable (its "spaces"),
        in construction order."""
        return [r.space for r in self._reps]

    #: Alias reading better where the space *name* is what matters.
    spaces = representations

    def encoders(self) -> dict[str, Component]:
        """Space name → the underlying encoder component that produces it (an
        `Embedder`/`SparseEncoder`/`LexicalIndex`), in stable order. The read
        accessor the evaluation suite uses to record what actually ran — grouped
        by `component.kind`, so it never has to know how a Corpus stores its
        representations. Reps wrapping no single component are omitted."""
        return {
            r.space: r.encoder for r in self._reps if r.encoder is not None
        }

    # -- writes --------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Single-pass write (Invariant 1). Gather every vector-backed rep's
        vectors into ONE `store.upsert`, then each self-managed rep's own
        `ingest`. Idempotent by `chunk.id`, so recovery is idempotent retry."""
        chunks = list(chunks)
        if not chunks:
            return
        vectors: dict[str, Sequence[VectorValue]] = {}
        for rep in self._vector_backed:
            vectors.update(rep.encode_corpus(chunks))
        if vectors:
            self._store.upsert(chunks, vectors)   # exactly ONE upsert
        for rep in self._self_managed:
            rep.ingest(chunks)                    # BM25's own side-write

    def update_representation(
        self, space: str, chunks: Sequence[Chunk]
    ) -> None:
        """Refresh ONE space over existing chunks (partial refresh), leaving
        sibling spaces and payload untouched."""
        rep = self._require(space)
        chunks = list(chunks)
        if not chunks:
            return
        if space in self._vector_spaces:
            vecs = rep.encode_corpus(chunks)[space]
            self._store.update_vectors(space, [c.id for c in chunks], vecs)
        else:
            rep.ingest(chunks)

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        space: str,
        text: str,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        """Search one named space (Invariant 2). Text in, not a vector — the
        Corpus encodes the query with the same encoder that encoded the corpus.
        Vector-backed: encode → store.search. Self-managed: delegate to the
        rep's own backend. Callers never learn which."""
        rep = self._require(space)
        if space in self._vector_spaces:
            query_vector = rep.encode_query(text)[space]
            return self._store.search(space, query_vector, k, filters)
        return rep.search(text, k, filters)

    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        """Point retrieval without a query vector (neighbor/parent expansion,
        get-by-(doc_id, index), dedup). Reads the vector store's payloads."""
        return self._store.fetch(filters, limit)

    # -- lifecycle -----------------------------------------------------------

    def persist(self) -> None:
        self._store.persist()
        for rep in self._self_managed:
            rep.persist()

    def describe(self) -> dict:
        """Fold store + every representation into identity, so adding a
        representation (or swapping an encoder) changes the fingerprint and its
        cache keyspace — and changing ONE rep changes only its slice."""
        info = super().describe()
        info["store_fingerprint"] = self._store.fingerprint()
        info["representations"] = {r.space: r.fingerprint() for r in self._reps}
        return info

    # -- helpers -------------------------------------------------------------

    def _require(self, space: str) -> Representation:
        rep = self._by_space.get(space)
        if rep is None:
            raise ConfigError(
                f"Corpus has no representation {space!r}; "
                f"available: {self.representations()}"
            )
        return rep
