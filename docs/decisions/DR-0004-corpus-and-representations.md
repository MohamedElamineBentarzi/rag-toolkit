# DR-0004 — Corpus + Representation: making representation kind Open/Closed

**Status:** accepted · **Supersedes:** DR-0001 v2 (the `ChunkIndex` aggregate and
its hardcoded dense/sparse/lexical trio) · **Keeps:** everything else in DR-0001
v2 (the composition algebra, the refiner chain, the retriever composition axis,
`VectorStore`/`LexicalIndex` contracts, the `ChunkSink` write fan-out) ·
**Breaking:** yes (pre-1.0) · **Audience:** maintainer + coding agents ·
**Method:** resolves the three gaps found reviewing the initial
Coordinator+Strategy blueprint against `tump_docs/INDEXING-CONTRACTS-FOR-REVIEW.md`.

---

## 0. TL;DR

DR-0001 gave us one aggregate, `ChunkIndex`, that owns every representation of a
corpus. It works, but it **hardcodes the three representation kinds**
(`dense=`/`sparse=`/`lexical=`) into its constructor and re-states that trio
across six files. Adding a fourth kind means editing all six. That violates the
Open/Closed rule the rest of the library lives by (the registry).

We split the aggregate into **two roles**:

> **`Representation` (the Strategy)** — a *pure*, registrable component that
> knows how to turn text into a searchable form and nothing else: declare the
> storage it needs, encode a batch of chunks, encode a query. **It holds no
> store and does no I/O.** `DenseRepresentation`, `SparseRepresentation`,
> `LexicalRepresentation` are the first three; a new kind is a new registered
> class.
>
> **`Corpus` (the Coordinator)** — the *single* owner of the physical
> `VectorStore`. It aggregates every representation's schema into **one**
> `ensure_schema`, drives a **single-pass** write (gather all vectors → **one**
> `upsert`, plus any self-managed side-writes), and owns **all** search and
> fetch I/O. It replaces `ChunkIndex`.

Retrievers address a corpus by **space name**, exactly as they addressed a
`ChunkIndex` by representation name — the read API barely changes; the
extensibility all lands underneath.

The flagship script, after this change:

```python
rag = RagPipeline(
    corpus=Corpus(
        store=QdrantVectorStore(url="http://localhost:6333", collection="siia_docs"),
        representations=[
            DenseRepresentation(CachingEmbedder(
                SentenceTransformerEmbedder(model="BAAI/bge-m3"),
                cache=MinioBlobStore())),
            LexicalRepresentation(Bm25Index()),
        ],
    ),
    generator=AnthropicGenerator(model="claude-sonnet-5"),
    chunker=MarkdownChunker(),
    enrich=[HeadingEnricher()],
    blob_store=MinioBlobStore(),
)
```

Same shape as DR-0001 §0, but the representations are now a **list of
first-class objects**, not three fixed keywords. Add `ColbertRepresentation(...)`
to that list and nothing in `Corpus` changes (see Gap 3 for the honest caveat).

---

## 1. Root cause (the OCP violation, precisely)

`ChunkIndex` is not just a container — it is a *consistency boundary* (DR-0001
D1), and that part is correct and preserved. The defect is narrower: **the set
of representation kinds is a hand-maintained list of three, hardcoded in six
places** instead of a registrable, discoverable set. The blast radius of "add
one kind":

1. `indexing/chunk_index.py` — 3 constructor params; 3 branches each in
   `_build_specs`, `add`, `search`, `update_representation`, `describe`.
2. `core/contracts.py` — `VectorSpec.kind = Literal["dense", "sparse"]` (closed).
3. `evaluation/space.py` — `STAGE_KINDS` names `embedder`/`sparse`/`lexical` as
   three fixed stages.
4. `evaluation/builder.py` — three `if "…" in spec` lines + a fixed
   `ChunkIndex(store, dense=…, sparse=…, lexical=…)` call.
5. `studio/manifest.py` — three fixed `STAGE_IO` entries feeding one synthetic
   `INDEX_NODE`.
6. Docs/tests restating the trio.

The registry makes chunkers, retrievers, generators Open/Closed — *register a
class, it appears everywhere with zero edits*. Representation kind is the one
concept that never got that treatment. DR-0004 gives it that treatment.

---

## 2. Decisions

### D1 — `Representation`: a pure, registrable Strategy (Design B)

**The load-bearing choice.** A `Representation` owns *projection*, never
*storage*. It takes text, returns vectors (or declares itself self-managed); it
holds no `VectorStore` reference and performs no database I/O. This is what
keeps it a Strategy: registry-instantiable from a flat `{name, params}` spec,
stateless with respect to the backend, and free of the temporal coupling a
`rep.bind(store)` step would introduce.

The alternative considered — *Representation owns `search` against a store it is
attached to* — was **rejected**. It would force a store reference into every
representation (breaking dynamic registry instantiation, the "two construction
worlds" debt DR-0001 §6.7 already names) or a post-construction `bind` step
(temporal coupling: `search()` fails until bound). "The `Corpus` is the single
storage owner" must be *literally* true, not aspirational.

```python
# indexing/representation.py
class Representation(Component):
    """Strategy: how a chunk is made searchable under one named space.

    Pure projection — owns encoding + schema declaration, NEVER storage.
    The Corpus owns all I/O. Two families exist (D3): vector-backed
    representations (dense, sparse) declare vector spaces and encode into
    them; self-managed representations (BM25) declare no vector space and own
    their own backend, exposed through `ingest` + `search`.
    """
    kind = "representation"

    @property
    def space(self) -> str:
        """The name this representation mounts under in a Corpus — the address
        a retriever queries. Defaults to the registry `name`; override to A/B
        two encoders of the same kind (space="dense_a" / "dense_b")."""
        return self.name

    # -- vector-backed family (default: not vector-backed) ------------------

    def declare_schema(self) -> Sequence[VectorSpec]:
        """The named vector spaces this representation needs in the shared
        store. Non-empty ⇒ vector-backed (the Corpus stores/searches for it).
        Empty (default) ⇒ self-managed (see below)."""
        return ()

    def encode_corpus(
        self, chunks: Sequence[Chunk]
    ) -> Mapping[str, Sequence[VectorValue]]:
        """Encode a batch into named vectors for the Corpus's ONE upsert.
        Keys are declared space names; each value is parallel to `chunks`.
        Default empty (self-managed reps contribute through `ingest`)."""
        return {}

    def encode_query(self, text: str) -> Mapping[str, VectorValue]:
        """Encode a query into its named vector(s). MUST use the same encoder
        as `encode_corpus` — this method IS the query/corpus parity guarantee
        (Invariant 2). Not called for self-managed reps."""
        return {}

    # -- self-managed family (default: no-op / unsupported) -----------------

    def ingest(self, chunks: Sequence[Chunk]) -> None:
        """Write a batch to a self-owned backend (BM25's LexicalIndex).
        Idempotent by chunk.id. Default no-op: vector reps contribute through
        `encode_corpus`, not here (Gap 2)."""

    def search(
        self, text: str, k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        """Self-managed query path (BM25 against its own LexicalIndex). Vector
        reps never implement this — the Corpus searches the shared store for
        them. Default raises (a vector rep reaching here is a routing bug)."""
        raise NotImplementedError(
            f"{type(self).__name__} is vector-backed; the Corpus searches the "
            f"store for it. Only self-managed representations implement search()."
        )
```

A representation is one of two families, discriminated by whether
`declare_schema()` is non-empty. It need not be an explicit subclass hierarchy —
concrete classes just override the half they use — but the DR treats them as two
families so the Corpus routing (D3/D4) is unambiguous.

### D2 — The three concrete representations, registered

```python
@registry.register
class DenseRepresentation(Representation):   # vector-backed
    name = "dense"
    def __init__(self, embedder: Embedder, space: Optional[str] = None): ...
    def declare_schema(self):
        return [VectorSpec(self.space, "dense",
                           dimensions=self._embedder.dimensions,
                           distance=self._embedder.distance)]
    def encode_corpus(self, chunks):
        return {self.space: self._embedder.embed_texts([c.text for c in chunks])}
    def encode_query(self, text):
        return {self.space: self._embedder.embed_query(text)}

@registry.register
class SparseRepresentation(Representation):   # vector-backed (static SPLADE)
    name = "sparse"
    def __init__(self, encoder: SparseEncoder, space: Optional[str] = None): ...
    def declare_schema(self):
        return [VectorSpec(self.space, "sparse")]
    def encode_corpus(self, chunks):
        return {self.space: self._encoder.encode_texts([c.text for c in chunks])}
    def encode_query(self, text):
        return {self.space: self._encoder.encode_query(text)}

@registry.register
class LexicalRepresentation(Representation):   # self-managed (corpus-relative BM25)
    name = "lexical"
    def __init__(self, index: LexicalIndex, space: Optional[str] = None): ...
    # declare_schema() -> ()  (inherited: no vector space)
    def ingest(self, chunks):
        self._index.add(chunks)
    def search(self, text, k, filters=None):
        return self._index.search(text, k, filters)
```

`space` defaults to the class `name` ("dense"/"sparse"/"lexical"). The
progressive-disclosure ergonomics of DR-0001 A1 survive: the common case names
nothing; A/B'ing two dense models sets distinct `space=` values.

**Construction note (the nested sub-spec seam).** A `DenseRepresentation` wraps
an `Embedder` — itself a registrable component. From a flat spec it is built
like a composite retriever's `inner`: the encoder is a **nested sub-spec** the
builder resolves first, then injects. So a representation *is* registry-buildable
(unlike `Corpus`, which wraps live backends), as long as its encoder is
expressed as a nested spec. See D6/D7.

### D3 — `Corpus`: the single storage owner (replaces `ChunkIndex`)

```python
# indexing/corpus.py
class Corpus(Component):
    kind = "corpus"                       # renamed concept (was index="index")

    def __init__(self, store: VectorStore, representations: Sequence[Representation]):
        # validate: >= 1 representation; space names unique.
        # aggregate declare_schema() across ALL reps -> ONE ensure_schema.
        # remember which spaces are vector-backed vs self-managed (routing).
        ...

    def representations(self) -> list[str]: ...     # the space names, stable order
    def spaces(self) -> list[str]: ...              # alias; the queryable addresses

    def add(self, chunks: Sequence[Chunk]) -> None:
        """SINGLE PASS (Invariant 1). Gather every vector-backed rep's
        encode_corpus() into one bundle -> ONE store.upsert(). Then each
        self-managed rep's ingest() (BM25 side-write). Never N vector upserts."""

    def search(self, space: str, text: str, k: int,
               filters: Optional[dict] = None) -> list[ScoredChunk]:
        """Route by space. Vector-backed: qv = rep.encode_query(text)[space];
        return store.search(space, qv, k, filters). Self-managed: return
        rep.search(text, k, filters). The Corpus is the ONLY store toucher."""

    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        """Preserved from ChunkIndex (DR-0001 G6): point retrieval for
        neighbor/parent expansion; reads the store payloads."""

    def update_representation(self, space: str, chunks: Sequence[Chunk]) -> None:
        """Preserved (DR-0001 P9 partial refresh): re-encode ONE space,
        siblings untouched. Vector-backed -> store.update_vectors; self-managed
        -> rep.ingest."""

    def persist(self) -> None: ...
    def describe(self) -> dict:
        """Folds store fingerprint + {space: rep.fingerprint()} — so changing
        ONE rep changes only its slice of identity (Invariant 4)."""
```

Reference write/search bodies (the invariants made concrete):

```python
def add(self, chunks):
    chunks = list(chunks)
    if not chunks:
        return
    vectors: dict[str, Sequence[VectorValue]] = {}
    for rep in self._vector_backed:            # dense, sparse, ...
        vectors.update(rep.encode_corpus(chunks))
    if vectors:
        self._store.upsert(chunks, vectors)    # <-- exactly ONE upsert
    for rep in self._self_managed:             # bm25, ...
        rep.ingest(chunks)                      # <-- side-write, its own backend

def search(self, space, text, k, filters=None):
    rep = self._by_space[space]
    if space in self._vector_spaces:
        qv = rep.encode_query(text)[space]
        return self._store.search(space, qv, k, filters)
    return rep.search(text, k, filters)
```

`Corpus` is a `Component` for identity/fingerprint but is **wired from live
backends, never built by `registry.create` alone** — the same precedent as
`ChunkIndex` and retrievers. It satisfies the **`ChunkSink` protocol** (`add` +
`persist`) unchanged, so the GraphRAG write fan-out (DR-0001 G9/F4) still works
with `Corpus` as one sink among others.

### D4 — Retrievers address a corpus by space name (reconciles Step 4)

Because search I/O lives on the `Corpus` (Design B), a retriever cannot hold a
*bare* `Representation` — a bare representation cannot reach the store. So the
retriever holds `(corpus, space)`, which is what `IndexRetriever(index,
representation)` already was. The original blueprint's "retriever takes a
Representation object" is **superseded by Design B**; the pluggability the user
wanted lands in the *spec* and *Studio* (D6/D7), not in the retriever's arms.

```python
IndexRetriever(corpus, space=None)        # one space; optional when corpus has one
HybridRetriever(corpus, spaces=None)      # fuse several spaces of ONE corpus (RRF)
FusionRetriever(retrievers=[...])         # unchanged: fuse ANY retrievers
```

`fusion.py` is reused verbatim (dedup by `chunk.id`, RRF, filter fan-out,
`metadata["sources"]` attribution). `IndexRetriever.label` stays
`f"{name}:{space}"` so sibling views remain distinguishable under fusion.

**RagPipeline wiring.** The keyword becomes `corpus=` (was `chunk_index=`); the
wiring guard becomes *"a retriever exposing `.corpus` must satisfy
`retriever.corpus is corpus`"* — the last way to recreate the query/corpus
mismatch (DR-0001 P6) stays a construction-time explosion.

### D5 — `VectorSpec.kind` opens to a string; stores still validate (Gap 3)

```python
@dataclass(frozen=True)
class VectorSpec:
    name: str
    kind: str            # was Literal["dense", "sparse"]; now open for plugins
    dimensions: Optional[int] = None
    distance: str = "cosine"
```

Opening `kind` lets a new representation declare its own storage kind without a
core edit. **But a store is not obligated to accept every kind.** Each backend
validates the kinds it physically supports and raises `ConfigError` at
`ensure_schema` on an unknown one (loud beats lossy — the DR-0001 D3 rule).
`MemoryVectorStore` and `QdrantVectorStore` support `"dense"` and `"sparse"`;
anything else is a backend feature request.

**The honest ColBERT caveat.** This refactor makes representations whose data
*fits the existing physical model* — one `VectorValue` (a dense vector or a
`SparseVector`) per named space per point — a genuine one-file addition (another
dense model, a different sparse encoder, an IDF-style static-sparse). A
representation with a *new data shape* — ColBERT/late-interaction stores **many
vectors per chunk**, which `VectorValue = list[float] | SparseVector` cannot
express — still requires extending the `VectorStore` contract and its backends.
DR-0004 makes the **logic** Open/Closed; it does not rewrite database schema
constraints. Do not sell "add ColBERT = one file." Adding ColBERT is: one
representation class **plus** a store capability. The architecture makes that
easier and localized; it does not make it free.

### D6 — Spec & builder: representations are a nested list under the corpus

The spec drops the three fixed keys (`embedder`/`sparse`/`lexical`) for one
`corpus` entry carrying a list of representation sub-specs, each with its encoder
nested inside it:

```json
{
  "corpus": {
    "vector_store": {"name": "qdrant", "params": {"collection": "docs"}},
    "representations": [
      {"name": "dense", "params": {
          "space": "dense",
          "embedder": {"name": "sentence-transformer", "params": {"model": "BAAI/bge-m3"}}}},
      {"name": "lexical", "params": {
          "index": {"name": "bm25"}}}
    ]
  }
}
```

`PipelineBuilder` gains one recursive resolver: build each representation from
its sub-spec (resolving the nested encoder first, exactly like it resolves a
composite retriever's `inner`/`retrievers`), then construct
`Corpus(store, representations)`. The three hardcoded `if "embedder"/"sparse"/
"lexical" in spec` lines are **deleted**. `space.py` replaces the three
`STAGE_KINDS` entries with a single representation kind for the search space to
enumerate.

`validate_spec` learns the `corpus` shape (a store sub-spec + a
`representations` list of `{name, params}`), validated recursively — structure
only, semantics still deferred to build time (the DR-0001 rule).

### D7 — Studio: a `Corpus` node fed by draggable representation nodes

The synthetic `INDEX_NODE` and its three fixed `STAGE_IO` inputs are **deleted**.
In their place:

- **`Corpus`** is a real node: one input port for a `vector_store`, one
  **multi-input** `representations` port, one `Corpus` output that feeds
  retrievers. (The `representations` port is the first "many-in" port; the
  manifest gains a port cardinality flag, `single` vs `many`, reused later.)
- **Representation nodes** (`dense`, `sparse`, `lexical`, and any future
  registered kind — automatically, via the registry) are draggable and wire
  their output into the `Corpus.representations` port.
- **The encoder is a nested param inside the representation node**, not a
  separate wired block. A `DenseRepresentation` *is* "an embedder mounted as a
  searchable space"; an encoder with no representation is meaningless. The
  inspector renders the encoder as a sub-form (nested spec), the same machinery
  `_composite_shape` uses for a composite retriever's `inner`. This keeps the
  canvas coarse and legible: `[dense][lexical] → [Corpus] → [retriever]`, encoder
  config living inside each representation block.

Exportability: a representation is exportable iff its encoder is expressible as a
nested sub-spec (it is, for the three built-ins). The manifest treats the
`embedder`/`encoder`/`index` param as **composition** (nested), not a flat field
or an export blocker — the `_HANDLED_COMPOSITION` set grows to include it.

### D8 — Identity & tuner caching (Invariant 4, made cleaner)

`Corpus.fingerprint()` folds the store fingerprint and `{space:
rep.fingerprint()}`. Because each representation is its own component with its
own fingerprint, **changing one representation's config changes only that
representation's slice** — the tuner re-encodes only the changed space and reuses
cached encodings for the rest. This is strictly cleaner than the monolith, where
identity was a single `describe()` blob. The per-encoder `CachingEmbedder`
namespacing (DR-0001 D7) is unchanged and composes per representation.

The **acceptance test carries over verbatim** (DR-0001 §6.7, now DR-0004's
tripwire): the tuner must index a corpus **once** and enumerate
retrieval/refinement strategies (dense-only, lexical-only, every hybrid, with and
without reranking/expansion) with **zero** tuner-motivated parameters appearing
on `Corpus` or `Representation`. If one appears, the abstraction leaked — stop
and write DR-0005.

---

## 3. The three gaps, resolved (for the record)

- **Gap 1 (store ownership / temporal coupling) → D1 + D4.** Design B chosen:
  representations are pure strategies with no store reference; the `Corpus` owns
  all I/O and search; retrievers address `(corpus, space)`. No `bind` step, no
  stateful strategy, registry-instantiable.
- **Gap 2 (BM25 has no slot in a vector-shaped write) → D2 + D3.** Two families
  under one interface: vector-backed reps contribute named vectors to the one
  shared `upsert` via `encode_corpus`; self-managed reps (BM25) write to their
  own backend via `ingest` and are searched via their own `search`. Invariant 1
  holds precisely: **one** vector `upsert` per batch, plus self-managed
  side-writes. `encode_corpus` never does I/O.
- **Gap 3 (ColBERT / "one file" overclaim) → D5.** `VectorSpec.kind` opens to a
  string and the logic is Open/Closed, but a representation with a new *data
  shape* still needs `VectorStore` support. Stated plainly so ColBERT is a
  known two-part change, not a broken promise.

---

## 4. Invariants preserved (the audit)

1. **Single write pass** — `Corpus.add` gathers all vector-backed reps into one
   `store.upsert`; never N sequential vector upserts (D3).
2. **Query/corpus parity** — `Representation.encode_query` uses the same encoder
   as `encode_corpus`; `Corpus.search` takes text, never a vector; retrievers
   can only address a corpus they are wired to (D1/D4).
3. **BM25 seamlessness** — `LexicalRepresentation` never touches the
   `VectorStore`; the retriever sees a uniform `search` and cannot tell a
   term-scored space from a vector one (D2/D3).
4. **Per-representation caching** — compositional fingerprint; one rep changing
   invalidates only its keyspace (D8).

Carried over from DR-0001 and explicitly kept: `fetch` (G6 neighbor expansion),
`update_representation` (P9 partial refresh), the `ChunkSink` fan-out (G9), eager
create-or-validate schema (fail fast in `__init__`), the composition axis
(`Fusion`/`Hybrid`/`MultiQuery`/`Hyde`), the refiner chain, chunks-stay-vector-free.

---

## 5. Migration plan (ordered; each step lands green)

`Corpus` is introduced **alongside** `ChunkIndex`; callers migrate; `ChunkIndex`
is deleted **last**. The tree compiles and tests pass at every step.

1. **`core/contracts.py`** — open `VectorSpec.kind` to `str` (backends still
   validate). No behavior change; existing `"dense"`/`"sparse"` still valid.
2. **`indexing/representation.py`** — `Representation` base + the three concrete
   classes, registered under kind `"representation"`. Contract tests: a
   vector-backed rep round-trips encode_corpus/encode_query; a self-managed rep
   ingests + searches; parity (query encoder == corpus encoder).
3. **`indexing/corpus.py`** — `Corpus` (aggregate schema, one-pass add, routed
   search, fetch, update_representation, ChunkSink). Contract test mirrors the
   old `assert_index_contract`, hermetic on memory store + HashingEmbedder +
   Bm25Index. `ChunkIndex` still present and untouched.
4. **`retrieval/`** — `IndexRetriever`/`HybridRetriever` accept a `Corpus`
   (keep a temporary `ChunkIndex` path if needed for green). Fusion reused.
5. **`pipeline.py`** — `RagPipeline(corpus=…)` + `.corpus` wiring guard;
   retriever derivation (one space → IndexRetriever, several → HybridRetriever)
   reads `corpus.representations()`.
6. **`evaluation/space.py` + `builder.py`** — single representation kind in the
   search space; builder resolves the nested `corpus`/`representations` spec;
   delete the three hardcoded stage lines.
7. **`studio/manifest.py`** — delete `INDEX_NODE` + the three `STAGE_IO`
   entries; add the `Corpus` node (with a `many` `representations` port) and
   auto-emit representation nodes from the registry; nested-encoder inspector.
8. **Delete `indexing/chunk_index.py`** and the last references; update
   `tuning.py`, tests, examples, README, AGENTS.md. Migration note in CHANGELOG
   (breaking): `chunk_index=` → `corpus=`, `ChunkIndex(dense=…)` →
   `Corpus(store, [DenseRepresentation(…)])`.

---

## 6. Non-goals (YAGNI fences)

No multi-vector-per-chunk storage yet (ColBERT waits on a `VectorStore`
extension — D5); no server-side fusion; no representation that is *both*
vector-backed and self-managed (a rep is one family); no `bind(store)` on
representations, ever (that is the rejected Design A); no async; no reintroducing
per-kind constructor keywords on `Corpus` (the acceptance-test tripwire, D8).

---

## 7. AGENTS.md §7.6 addendum (paste-ready, supersedes the DR-0001 v2 block on
this topic)

> **§7.6 Corpus, Representation & multi-representation retrieval (DR-0004,
> supersedes DR-0001 v2's ChunkIndex).** A corpus is searchable in several named
> ways ("spaces"); each way is a **`Representation`** — a *pure, registrable
> Strategy* that declares the storage it needs and encodes corpus + query with
> one encoder, and **holds no store and does no I/O**. Two families, one
> interface: *vector-backed* reps (dense, sparse) return named vectors from
> `encode_corpus`/`encode_query`; *self-managed* reps (BM25 `LexicalIndex`)
> declare no vector space and own their backend through `ingest`/`search`.
> **`Corpus`** is the single owner of the `VectorStore`: it aggregates every
> rep's `declare_schema` into ONE `ensure_schema`, drives a single-pass `add`
> (all vectors → ONE `upsert`, then self-managed `ingest`s), and owns ALL search
> and `fetch` I/O — retrievers address a corpus by `space`, never a bare
> representation, and never a vector. `Corpus` is wired from live backends (never
> `registry.create` alone) and satisfies `ChunkSink`. `VectorSpec.kind` is an
> open string, but stores validate the kinds they support (fail fast). Adding a
> representation whose data fits one `VectorValue` per space per point is a new
> registered class and nothing else; a new *data shape* (ColBERT) also needs a
> `VectorStore` extension — say so. Fingerprint folds store + `{space:
> rep.fingerprint()}` so one rep changing invalidates only its keyspace. Do not
> re-litigate: no store reference on a representation, no `bind` step, no per-kind
> keyword on `Corpus`, no multi-vector storage until the store contract grows it.
> Acceptance test: the tuner indexes once and enumerates strategies with ZERO
> tuner-motivated parameters on `Corpus`/`Representation` — if one appears, stop
> and write DR-0005.

---

## 8. Vocabulary

| Term | Meaning |
|---|---|
| **Space** | A named searchable way into a corpus: `"dense"`, `"splade"`, `"lexical"`. The address a retriever queries. |
| **`Representation`** | Pure Strategy: declares schema + encodes corpus/query for one space. No store, no I/O. |
| **Vector-backed rep** | Declares vector spaces; its vectors go into the Corpus's shared `upsert` (dense, sparse). |
| **Self-managed rep** | Declares no vector space; owns its own backend via `ingest`/`search` (BM25). |
| **`Corpus`** | Coordinator: the single owner of the `VectorStore`; one schema, one-pass write, all search/fetch I/O. Replaces `ChunkIndex`. |
| **Design B** | Corpus owns search; representations are storeless strategies (the chosen resolution of Gap 1). |
