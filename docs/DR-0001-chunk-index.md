# DR-0001 v2 — The ChunkIndex, the composition algebra, and what a stress test changed

**Status:** v2, supersedes v1 · **Resolves:** ARCHITECTURE-REVIEW.md (P1–P9,
Q1–Q5) · **Method for v2:** the design was stress-tested by writing ~10 real
RAG architectures against it on paper (§5) and amending wherever the pen
dragged (§6) · **Breaking:** yes (pre-1.0) · **Audience:** maintainer + coding
agents (AGENTS.md §7.6 paste block in §11).

---

## 0. TL;DR

One new aggregate, two uniform chains, one composition axis, one root:

> **`ChunkIndex`** owns every retrieval representation of a corpus
> (dense/sparse/lexical) on both paths: `add(chunks)` writes them all;
> `search(representation, TEXT, k)` encodes the query with the same encoder
> that encoded the corpus. Query/corpus compatibility is structural.
>
> **Retrievers are read-only views that compose like `nn.Module`**
> (`IndexRetriever`, `HybridRetriever`, `FusionRetriever`,
> `MultiQueryRetriever`, `HydeRetriever` — retrievers wrapping retrievers).
>
> **Refiners are a uniform post-retrieval chain, MongoDB-aggregation style**
> (`(query, candidates, k) → candidates`): cross-encoder reranking, MMR,
> neighbor/parent expansion, score floors — all one stage shape. The
> `reranker` kind dissolves into it.
>
> **`RagPipeline` is a composition root**: live backends are created once at
> the edge and shared by reference; the write path fans out to any
> `ChunkSink` (ChunkIndex is the flagship sink; a GraphRAG index is just
> another one).

The flagship script (hybrid dense+BM25, cached embeddings, cited answers):

```python
rag = RagPipeline(
    chunk_index=ChunkIndex(
        store=QdrantVectorStore(url="http://localhost:6333", collection="siia_docs"),
        dense=CachingEmbedder(SentenceTransformerEmbedder(model="BAAI/bge-m3"),
                              cache=MinioBlobStore()),
        lexical=Bm25Index(),
    ),
    generator=AnthropicGenerator(model="claude-sonnet-5"),
    chunker=MarkdownChunker(),
    enrich=[HeadingEnricher()],
    blob_store=MinioBlobStore(),
)
rag.index(Source.from_path(PDF_PATH))
answer = rag.ask("Quels sont les objectifs du parcours SIIA ?", k=5)
print(answer.text, answer.citations)
```

No stutter, no dict, no name you didn't choose; every layer still overridable.
Retriever not passed ⇒ derived: one representation → `IndexRetriever`, several
→ `HybridRetriever` over all of them with RRF.

---

## 1. Root cause (unchanged from v1, condensed)

Write ingredients (embedder+store, hardcoded in the old `RagPipeline._flush`)
and read ingredients (retriever) were separate object graphs kept consistent
by convention. That is P1 (facade forces dense), P2 (accidental sharing), P5
(split write path), P6 (compatibility by convention); P3/P4/P7–P9 are the same
gap seen from the representation side. The fix is a consistency boundary — an
aggregate root that guards: *"every representation of every chunk in this
corpus was produced by the encoders this index declares, and queries are
encoded the same way."* `ChunkIndex` is to representations what `Document` is
to pages.

---

## 2. The elegance bar (and how v2 was judged)

The maintainer's benchmarks: **PyTorch** (uniform composable units — a Module
contains Modules; `Sequential` is just composition) and **MongoDB aggregation**
(a pipeline of uniform stages over one data shape). v2's claim is that the
architecture now maps onto *both*, on purpose:

- **Pre-retrieval variation = composition** (PyTorch-style). Query shaping —
  multi-query, HyDE, routing, federation — is retrievers wrapping retrievers,
  not new pipeline slots. One axis, already proven by `HybridRetriever`.
- **Post-retrieval variation = a chain of uniform stages** (Mongo-style).
  Everything after retrieval is `list[ScoredChunk] → list[ScoredChunk]`; the
  `refine=[...]` chain is the `$match`/`$sort`/`$limit` of this library.
- **Write-side variation = the same two moves**: a chain (`enrich=[...]` over
  the chunk stream) and fan-out to sinks.

The **composition algebra**, stated once: *nouns* (the contracts), *two
chains* (enrich on the write path, refine on the read path), *one composition
axis* (retrievers of retrievers), *one aggregate* (ChunkIndex), *one root*
(RagPipeline). Everything in §5's gallery is spelled with only these.

Elegance was also measured in **deletions** (§6.6): four classes and one
registered kind disappear in v2.

---

## 3. Decisions

### D1 — `ChunkIndex`: the aggregate owning representations, write and read

`Component` (`kind="index"`), constructed with live instances (like
retrievers — it wraps stateful backends; `registry.create` alone cannot build
it, per the precedent already documented in `retrieval/base.py`).

**Constructor (Amendment A1 — progressive disclosure).** Union-typed keywords
with auto-naming; dicts only for the genuine multi-representation power case:

```python
ChunkIndex(
    store: VectorStore,
    dense:  Embedder      | Mapping[str, Embedder]      | None = None,
    sparse: SparseEncoder | Mapping[str, SparseEncoder] | None = None,
    lexical: LexicalIndex | None = None,        # mounted as "lexical"
)
# single encoder → auto-named "dense" / "sparse" / "lexical"
# mapping       → explicit names (A/B-ing two dense models, etc.)
```

Rationale for A1: the old `dense={"dense": embedder}` stuttered, made the
common case pay ceremony for the rare case, and asked the caller to restate
what the objects' types already say (an `Embedder` *is* dense). The design
rule this instantiates — now a standing AGENTS.md rule — is **progressive
disclosure**: the common case reads like English; the rare case is possible;
the rare case's ceremony never leaks into the common case.

Surface: `add(chunks)` (encode every representation, upsert once,
batch-scoped: O(batch × representations)); `search(representation, text, k,
filters=None)` (**text in, not vector** — encodes with the representation's
own query encoder; this line is the P6 guarantee);
`fetch(filters, limit)` (point retrieval, see D3);
`representations()`; `update_representation(name, chunks)` (P9 partial
refresh); `persist()`; `describe()/fingerprint()` folding store + every
encoder (P8). `ensure_schema` runs **eagerly in `__init__`** — create or
*validate*, never coerce (fail fast; see §8.1).

Not a god object: one responsibility — "own the representations of one
corpus: schema, writes, reads." Encoding delegates to encoders, storage to the
store, term scoring to the lexical index; no ranking strategy, no
parsing/chunking, no generation. `Embedder` gains a `distance: str` property
beside `dimensions` (the adapter knows its metric), so per-representation
config needs no wrapper classes.

### D2 — Chunks stay vector-free (fact vs interpretation, again)

`Chunk` keeps text + provenance only; representations are derived, keyed data
in the store and the embedding cache — never on the chunk. Same principle
that kept chunks off `Document`: *a Document is a fact; a chunking is an
interpretation* ⇒ *a Chunk is a fact; its representations are interpretations
under particular encoders.* `chunk.vectors` would couple a cacheable stage
output to encoder choices, bloat the blob cache, and break identity layering
(re-embedding must not change chunk identity — that is P9). Inside
`ChunkIndex.add`, vectors exist only transiently, per batch.

### D3 — Multi-vector `VectorStore` contract (named + typed + fetchable)

- `VectorSpec(name, kind: "dense"|"sparse", dimensions?, distance?)`;
  `SparseVector(indices, values)`; `VectorValue = list[float] | SparseVector`.
- `ensure_schema(specs)` — create, or validate an existing collection matches;
  mismatch → `ConfigError`. Never silently coerce.
- `upsert(chunks, vectors: Mapping[name, Sequence[VectorValue]])` — one
  payload write, N named vectors per point; idempotent by `chunk.id`.
- `search(name, vector, k, filters)` — one named space.
- **`fetch(filters, limit) -> list[Chunk]` (v2 addition, Finding F3)** —
  point retrieval *without* a query vector. The stress test showed this hole
  immediately: neighbor/parent expansion, "get chunk by (doc_id, index)",
  dedup checks, and staleness scans all need it, and the library had already
  promised it rhetorically ("get chunk by index lives in the store"). Qdrant:
  scroll API; memory: trivial. **Filter semantics extended**: a list value
  means membership (`{"doc_id": d, "index": [i-1, i+1]}`), applied uniformly
  by stores, lexical indexes, and every fused sub-search.
- `update_vectors(name, chunk_ids, vectors)` — replace one representation on
  existing points, payload and siblings untouched (default raises
  `StorageError("not supported")`; Qdrant and memory implement it).

Backends: **Qdrant** native (named dense + native sparse; scroll;
point-vectors update). **Memory** is the reference implementation and what
contract tests + the tuner run on. Degradation policy for simpler backends:
emulate named vectors (one collection per name) only if cheap and
transparent; otherwise raise at `ensure_schema` listing what is supported —
loud beats lossy. Kind renamed `"store"` → `"vector_store"` (§8.4).

### D4 — Sparse/lexical: two scoring models, both first-class, one read API

The review's subtlety is real; the architecture refuses to pretend otherwise:

1. **Classic BM25** is corpus-relative (query-time idf, avgdl) — *not* a
   static per-chunk vector, so it is not stored as one. `Bm25Index`
   (`LexicalIndex` kind) survives, but **mounted inside `ChunkIndex`** as
   representation `"lexical"` — unifying *lifecycle and ownership* (the actual
   P4 defect; physical duplication into index structures is what every real
   engine does, Elasticsearch included).
2. **Static sparse (SPLADE-style)** is a genuine per-chunk `SparseVector`
   from a `SparseEncoder` (new kind; `encode_texts`/`encode_query` mirroring
   the Embedder's passage/query asymmetry), stored under a named sparse spec.
   Qdrant footnote kept from v1: its sparse **IDF modifier** makes a
   BM25-*style* static encoding (tf + length-norm frozen at encode time,
   engine-side idf) a later drop-in `SparseEncoder` — zero new architecture.

Scope: v0.6 lands contracts + lexical mounting; `SparseEncoder`
implementations are fast-follow.

### D5 — Retrievers: read-only views + a full composition axis

Retrievers stay **read-only** (the tuner is the decider — see §7 Q1). Because
`index.search(name, text)` is uniform, the zoo collapses and then *composes*:

- `IndexRetriever(index, representation=None)` — replaces `DenseRetriever`
  and `Bm25Retriever`. With A1: representation optional when the index has
  exactly one (ambiguity → fail fast listing options).
- `FusionRetriever(retrievers, fusion="rrf", rrf_k=60, weights=None)` —
  **the general composition node (v2, Finding F2b)**: fuses *any* retrievers —
  across representations, across indexes (federation), across paradigms
  (vector + graph).
- `HybridRetriever(index, representations=None)` — progressive-disclosure
  sugar: builds `IndexRetriever` per representation (default: all of them) and
  delegates to the same fusion. The common case reads like English; the power
  case is `FusionRetriever`.
- `MultiQueryRetriever(inner, complete, n=4)` and `HydeRetriever(inner,
  complete)` — query shaping **as composition, not as a pipeline slot**
  (Finding F2): expand/hypothesize with an LLM, retrieve through the wrapped
  retriever, fuse. `RouterRetriever(routes, select)` follows the same shape
  when needed.

Shared mechanics extracted once to `retrieval/fusion.py` (used by Fusion,
Hybrid, MultiQuery): **fuse keyed by `chunk.id`** (same chunk from two sources
merges, never duplicates), RRF `score = Σ w_r / (rrf_k + rank_r)`,
`metadata["sources"] = {source: rank}` for eval attribution, **filters fan out
to every sub-search**, `fetch_k` applied per source.

**The `complete` seam (Finding F5).** MultiQuery, HyDE, contextual enrichment
(and later, LLM judges) need bare text completion — a shape `Generator`
deliberately doesn't expose ((query, context) → Answer is the wrong contract).
v2 decision, progressive-disclosure applied to the architecture itself: these
components take `complete: Callable[[str], str]`; `AnthropicGenerator` (and
peers) expose a public `.complete(prompt) -> str`. No new kind today; the
documented promotion path is a `completer` kind the day a third independent
consumer or vendor adapter demands registry-level swappability.

### D6 — Composition root, write fan-out, and the two chains

**`IndexingPipeline`** becomes the complete write path (P5 closed):

- `enrich: Sequence[Enricher] = ()` — enrichers are a **chain** over the chunk
  stream (Iterator → Iterator compose trivially). `NoOpEnricher` is deleted:
  the empty chain *is* the null object.
- `sinks: Sequence[ChunkSink] = ()` — **write fan-out (Finding F4)**. A
  `ChunkSink` is a structural protocol (`add(chunks)`, `persist()`); a
  `ChunkIndex` satisfies it natively, and so does a GraphRAG index, a
  keyword-alert index, or a bare `LexicalIndex`. This is the one place the
  codebase uses `typing.Protocol`, deliberately: AGENTS.md's "ABCs, not
  Protocols" rule governs *stage contracts*, which carry inherited plumbing
  (config, fingerprint); `ChunkSink` is a *capability seam* spanning worlds a
  common base cannot — shape is exactly what's meant, so structural typing is
  the honest tool. (Rule refined, not broken; the AGENTS block in §11 says so.)
- Batching lives here: parse → chunk → enrich chain → batch → every sink.

**`RagPipeline`** is the composition root and only that:

```python
RagPipeline(chunk_index=None, retriever=None, generator=None, parser=None,
            chunker=None, enrich: Sequence[Enricher] = (),
            refine: Sequence[Refiner] = (), extra_sinks: Sequence[ChunkSink] = (),
            blob_store=None, fetch_k=50, batch_size=32, trace=_noop_trace)
```

Defaults preserve zero-config (`MemoryVectorStore` + `HashingEmbedder` index;
derived retriever per A1). Wiring guard: a retriever exposing `.index` must
satisfy `retriever.index is chunk_index` — the last way to recreate P6 becomes
a constructor-time explosion. Direct answer to the maintainer's own question
("is it clean if I instantiate components before RagPipeline and it gives the
same instance to both?"): **yes — that is the pattern** (dependency injection
at the edge), refined so the shared thing is one `ChunkIndex`, not loose
pieces. `_flush`, `_embed`, `_EmbeddingCache` are deleted from the facade.
`RagPipeline.dense(embedder, store, **kw)` remains the one-call convenience.
Naming: method `index(sources)` (verb), attribute `chunk_index` (noun);
pipelines stay non-Components (wiring, not algorithms).

### D7 — Caching is a decorator, and identity-transparent

`CachingEmbedder(inner: Embedder, cache: BlobStore)`: implements `Embedder`,
memoizes by `sha256(text)` under `embeddings/{inner.fingerprint()}/…`. Two
correctness rules: **`fingerprint()` returns `inner.fingerprint()`** (the
wrapper changes cost, not output — it must be invisible to cache keys and
trial identity; the override is documented and deliberate), and **passage vs
query caches use separate namespaces** (`…/passages/`, `…/queries/`) because
asymmetric models encode the same string differently through
`embed_texts` vs `embed_query`. Composes per representation; a
`CachingSparseEncoder` mirrors it when sparse lands. P8's "N representations ⇒
N cache keyspaces" falls out for free.

### D8 — Identity across representations

`ChunkIndex.fingerprint()` folds store + `{name: encoder.fingerprint()}` +
lexical. Tuner cache keys: chunks = (sources × parser × chunker × enrich
chain) fingerprints; representations = chunks-key × per-encoder fingerprint
(adding SPLADE re-encodes SPLADE only); retrieval and refinement trials are
read-only ⇒ free. `chunk.id` stable; representation values keyed
`(chunk.id × name)`; partial refresh via `update_representation` →
`store.update_vectors`. Stores MAY record encoder fingerprints in payload for
skip-if-current incremental refresh — enabled, not required.

### D9 — The refiner chain (Finding F1; the `reranker` kind dissolves)

The stress test's biggest hole: sentence-window/parent expansion, MMR
diversity, score floors, near-dup collapse, context compression — the most
common post-retrieval patterns in real systems — **had no home**. Yet every
one of them, including cross-encoder reranking, is the same shape:

```python
class Refiner(Component):
    kind = "refiner"
    @abstractmethod
    def refine(self, query: Query, candidates: list[ScoredChunk],
               k: int) -> list[ScoredChunk]:
        """k is the caller's final budget (a hint — budget-aware refiners
        like rerankers use it; others ignore it). May return more or fewer;
        the pipeline enforces final truncation to k."""
```

`QueryPipeline(retriever, refine: Sequence[Refiner] = (), fetch_k=50)` runs:
retrieve `fetch_k` → each refiner in order → truncate to `k`. Uniform stages
over one data shape — this is the MongoDB-aggregation half of the algebra.
Consequences: the `reranker` kind is **retired**; `CrossEncoderReranker`
re-registers as `refiner:cross-encoder`; `NoOpReranker` is deleted (the empty
chain is the null object). Planned refiners: `CrossEncoderReranker`,
`NeighborExpander(index, window=1)` (uses D3 `fetch` + char-offset-aware
merge — overlap-safe *because* chunks carry `char_start/char_end`; the
provenance chain pays again), `MMRDiversifier`, `ScoreThreshold`,
`NearDupCollapser`; `ContextCompressor` later.

---

## 4. Contracts (exact shapes)

```python
# core/contracts.py — additions
@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

VectorValue = Union[list[float], "SparseVector"]

@dataclass(frozen=True)
class VectorSpec:
    name: str
    kind: Literal["dense", "sparse"]
    dimensions: Optional[int] = None      # dense only
    distance: str = "cosine"              # dense only
```

```python
# storage/vector_store.py — v2
class VectorStore(Component):
    kind = "vector_store"                 # renamed from "store" (§8.4)

    @abstractmethod
    def ensure_schema(self, specs: Sequence[VectorSpec]) -> None: ...
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk],
               vectors: Mapping[str, Sequence[VectorValue]]) -> None: ...
    @abstractmethod
    def search(self, name: str, vector: VectorValue, k: int,
               filters: Optional[dict] = None) -> list[ScoredChunk]: ...
    @abstractmethod
    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        """Point retrieval without a query vector. List filter values mean
        membership: {"doc_id": d, "index": [3, 5]}."""
    def update_vectors(self, name: str, chunk_ids: Sequence[str],
                       vectors: Sequence[VectorValue]) -> None:
        raise StorageError(f"{type(self).__name__}: partial vector updates "
                           "not supported")
    def persist(self) -> None: ...
```

```python
# embedding/ — additions
class Embedder(Component):
    ...existing...
    @property
    def distance(self) -> str: return "cosine"   # adapters override

class SparseEncoder(Component):
    kind = "sparse_encoder"
    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> list[SparseVector]: ...
    @abstractmethod
    def encode_query(self, text: str) -> SparseVector: ...
```

```python
# indexing/chunk_index.py
class ChunkIndex(Component):
    kind = "index"; name = "chunk-index"
    def __init__(self, store: VectorStore,
                 dense: Embedder | Mapping[str, Embedder] | None = None,
                 sparse: SparseEncoder | Mapping[str, SparseEncoder] | None = None,
                 lexical: Optional[LexicalIndex] = None) -> None:
        # normalize to name→encoder maps (auto-names "dense"/"sparse"/"lexical");
        # fail fast: ≥1 representation, unique names; store.ensure_schema NOW.
        ...
    def representations(self) -> list[str]: ...
    def add(self, chunks: Sequence[Chunk]) -> None: ...
    def search(self, representation: str, text: str, k: int,
               filters: Optional[dict] = None) -> list[ScoredChunk]: ...
    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]: ...
    def update_representation(self, name: str, chunks: Sequence[Chunk]) -> None: ...
    def persist(self) -> None: ...
```

```python
# indexing/sink.py — the one deliberate Protocol (rationale in D6)
@runtime_checkable
class ChunkSink(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...
    def persist(self) -> None: ...
```

```python
# retrieval/ — the composition axis
@registry.register
class IndexRetriever(Retriever):
    name = "index"
    def __init__(self, index: ChunkIndex,
                 representation: Optional[str] = None, ...): ...

@registry.register
class FusionRetriever(Retriever):
    name = "fusion"
    def __init__(self, retrievers: Sequence[Retriever],
                 fusion: str = "rrf", rrf_k: int = 60,
                 weights: Optional[Sequence[float]] = None): ...

@registry.register
class HybridRetriever(Retriever):        # sugar over FusionRetriever
    name = "hybrid"
    def __init__(self, index: ChunkIndex,
                 representations: Optional[Sequence[str]] = None, ...): ...

@registry.register
class MultiQueryRetriever(Retriever):
    name = "multi-query"
    def __init__(self, inner: Retriever,
                 complete: Callable[[str], str], n: int = 4, ...): ...

@registry.register
class HydeRetriever(Retriever):
    name = "hyde"
    def __init__(self, inner: Retriever,
                 complete: Callable[[str], str], ...): ...
```

```python
# refinement/base.py — replaces reranking/
class Refiner(Component):
    kind = "refiner"
    @abstractmethod
    def refine(self, query: Query, candidates: list[ScoredChunk],
               k: int) -> list[ScoredChunk]: ...
```

```python
# pipeline.py — deltas
class IndexingPipeline:
    def __init__(..., enrich: Sequence[Enricher] = (),
                 sinks: Sequence[ChunkSink] = (), batch_size: int = 32, ...): ...

class QueryPipeline:
    def __init__(self, retriever: Retriever,
                 refine: Sequence[Refiner] = (), fetch_k: int = 50, ...): ...

class RagPipeline:
    def __init__(self, chunk_index=None, retriever=None, generator=None,
                 parser=None, chunker=None, enrich: Sequence[Enricher] = (),
                 refine: Sequence[Refiner] = (),
                 extra_sinks: Sequence[ChunkSink] = (),
                 blob_store=None, fetch_k=50, batch_size=32, trace=...): ...
    @classmethod
    def dense(cls, embedder=None, store=None, **kw) -> "RagPipeline": ...
```

---

## 5. The gallery — ten architectures, written against v2

The evidence. Each entry: the code as a user would write it, and the mechanism
carrying it. (1)–(3) exercise the base; (4)–(10) are the patterns that forced
v2's findings.

**G1 · Zero-config smoke test** — defaults all the way down:
```python
rag = RagPipeline()
rag.index(Source.from_path("notes.md")); print(rag.ask("what is this?").text)
```
*Mechanism: A1 defaults (memory store, hashing embedder, derived retriever).*

**G2 · Production dense + rerank** — the 80% deployment:
```python
rag = RagPipeline.dense(
    embedder=SentenceTransformerEmbedder(model="BAAI/bge-m3"),
    store=QdrantVectorStore(url=..., collection="docs"),
    generator=AnthropicGenerator(model="claude-sonnet-5"),
    refine=[CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")],
)
```
*Mechanism: convenience constructor + one-stage refiner chain.*

**G3 · Hybrid dense+BM25** — the flagship script in §0. *Mechanism:
ChunkIndex multi-representation + derived HybridRetriever.*

**G4 · RAG-fusion (multi-query + RRF)**:
```python
gen = AnthropicGenerator(model="claude-sonnet-5")
retriever = MultiQueryRetriever(HybridRetriever(index), complete=gen.complete, n=4)
rag = RagPipeline(chunk_index=index, retriever=retriever, generator=gen)
```
*Mechanism: composition axis — a retriever wrapping a retriever wrapping an
index. No pipeline slots were added (F2).*

**G5 · HyDE**:
```python
retriever = HydeRetriever(IndexRetriever(index), complete=gen.complete)
```
*Mechanism: same axis; the `complete` seam (F5).*

**G6 · Sentence-window / small-to-big** — index small, answer big:
```python
rag = RagPipeline(
    chunk_index=index, chunker=SentenceChunker(),
    refine=[NeighborExpander(index, window=2),
            CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")],
    generator=gen,
)
```
*Mechanism: refiner chain (F1) + `fetch` by (doc_id, index-membership) (F3) +
char-offset overlap-safe merging — the provenance chain doing retrieval work,
not just citation work.*

**G7 · Multi-corpus federation** — legal + HR, one question:
```python
retriever = FusionRetriever([IndexRetriever(legal_index),
                             IndexRetriever(hr_index)], fusion="rrf")
```
*Mechanism: FusionRetriever generalization (F2b). Two indexes, two schemas,
one read API.*

**G8 · Contextual retrieval (Anthropic-style)**:
```python
rag = RagPipeline(chunk_index=index,
                  enrich=[HeadingEnricher(),
                          ContextualEnricher(complete=gen.complete)],
                  generator=gen)
```
*Mechanism: enricher chain + `complete` seam. Order matters and is visible.*

**G9 · Vector + GraphRAG, side by side** — extension without core edits:
```python
graph = MyGraphIndex(neo4j_url=...)          # implements add() + persist()
rag = RagPipeline(
    chunk_index=index, extra_sinks=[graph],  # write path fans out (F4)
    retriever=FusionRetriever([HybridRetriever(index),
                               MyGraphRetriever(graph)]),
    generator=gen,
)
```
*Mechanism: ChunkSink protocol + FusionRetriever across paradigms. The
library never heard of graphs; nothing in core changed.*

**G10 · The tuner loop (the falsifiable prediction, on paper)**:
```python
for _ in IndexingPipeline(chunker=chunker, enrich=enrichers,
                          sinks=[index]).index(corpus): pass   # index ONCE
for retriever in [IndexRetriever(index, "dense"),
                  IndexRetriever(index, "lexical"),
                  HybridRetriever(index)]:
    for chain in ([], [CrossEncoderReranker(...)],
                  [NeighborExpander(index), CrossEncoderReranker(...)]):
        evaluate(QueryPipeline(retriever, refine=chain))
```
*Nine strategies, zero re-indexing, zero new ChunkIndex parameters. The
prediction holds on paper; §6.7 keeps the verdict gated on it holding in code.*

Also verified: **attach-to-existing-corpus** (construct `ChunkIndex` over an
already-populated collection; eager `ensure_schema` *validates*, `add` is
never called, retrieval just works) and **retrieval-only eval** (a bare
`QueryPipeline` is the whole system — no facade needed).

---

## 6. Findings ledger — what the stress test changed, and the honest audit

- **F1 → D9.** Post-retrieval patterns (expansion, MMR, thresholds,
  compression) were homeless; rerank was a hardcoded slot. Fixed by the
  refiner chain; `reranker` kind dissolved into it.
- **F2 → D5.** Pre-retrieval patterns (multi-query, HyDE, routing) were also
  homeless — but the fix is *not* a new stage kind: they are composite
  retrievers on the existing axis. A `QueryTransform` kind was considered and
  **rejected** (you don't add a "pre-layer slot" to `nn.Sequential`; you wrap
  modules in modules). **F2b:** fusion generalized from
  representation-fusion to retriever-fusion (`FusionRetriever`;
  `HybridRetriever` becomes sugar); mechanics extracted once to `fusion.py`.
- **F3 → D3.** `VectorStore` had no point retrieval; `fetch(filters, limit)` +
  list-membership filter semantics added. The library had already promised
  this capability rhetorically; now the contract keeps the promise.
- **F4 → D6.** The write path was closed to non-ChunkIndex indexes (GraphRAG,
  alert indexes). `ChunkSink` protocol + `sinks` fan-out opens it — with an
  explicit, narrow justification for using a Protocol where AGENTS.md
  otherwise mandates ABCs.
- **F5 → D5.** Three gallery entries needed bare LLM completion and
  `Generator` is deliberately the wrong shape for it. Resolved with a
  `Callable[[str], str]` seam (+ public `.complete` on generator adapters);
  promotion to a registered `completer` kind is documented, not preempted —
  progressive disclosure applied to the architecture itself.

**6.6 The deletion ledger** (elegance measured in deletions):
`DenseRetriever`, `Bm25Retriever`, `NoOpReranker`, `NoOpEnricher` — deleted
(the first two collapse into `IndexRetriever`; the null objects become empty
chains). `RagPipeline._flush/_embed/_EmbeddingCache` — deleted (D6/D7).
`reranker` kind — retired into `refiner`. Net registered kinds: **unchanged**
(refiner replaces reranker; `ChunkSink` is a protocol, `FusionRetriever` is a
retriever). Expressiveness: §5 went from ~3 expressible architectures to 10.

**6.7 The audit, kept honest.** Defended without hesitation: the contracts
spine and provenance chain (now doing *retrieval* work in G6, not just
citations); the fingerprint thread from `Component` to cache keys; streaming
ingestion; ChunkIndex as a consistency boundary. Named tradeoffs, still open:
**two construction worlds** (registry-by-name for stateless components,
instance-wiring for composites) — config-as-data does not yet cross that seam;
the tuner will need a `ChunkIndex.from_config`/pipeline-spec resolver, and
that is acknowledged debt, not a solved problem. **Text lives in three
places** (blob truth, vector payload, lexical index) — deliberate, buys
query-time independence; storage cost, not elegance. **Stringly-typed
representation names** at the edges — mitigated by auto-naming and
`representations()`, accepted for the same reason the registry accepts
strings (serializability). **The verdict stays gated** on the falsifiable
test: the tuner must run G10 against real code without a single
tuner-motivated parameter appearing on `ChunkIndex`; if one appears, the
abstraction leaked and we redesign. Meta-rule retained: nine-ish kinds is a
lot for a solo pre-1.0 library and is defensible only because *the seams are
the product* — enforced by "≥2 implementations per kind, or delete the kind."

---

## 7. The review's five questions (final answers)

**Q1** — Composition root (D6); retrievers stay read-only. Retriever-owned
indexing fuses *what representations exist* (expensive, once) with *how to
query them* (cheap, many); G10 is the counterexample that kills it — the tuner
must index once and enumerate strategies freely. **Q2** — Yes: named + typed
multi-vector with eager create-or-validate schema, `fetch`, and
`update_vectors` (D3). **Q3** — Both models, first-class, one read API (D4):
corpus-stats BM25 mounted; static sparse as `SparseEncoder` representations.
**Q4** — Created once at the application edge, owned by `ChunkIndex`
(encoders, store) and shared by reference through the root; mis-wiring
explodes at construction (D6). **Q5** — Inside the write path:
`IndexingPipeline` batches into sinks; embedding stops being a facade
afterthought (D6).

---

## 8. Operational problems surfaced earlier (unchanged in substance)

**8.1 Schema evolution cliff.** Named-vector sets are fixed at Qdrant
collection creation — you cannot add `"splade"` to an existing collection.
`ensure_schema` validates loudly; operational rule: *representation-set change
⇒ new collection*; opt-in convention: suffix collection names with a short
hash of the representation-set fingerprint so incompatible schemas cannot
collide. **8.2 Synthetic-chunk identity.** Enrichers that *add* chunks must
use parent-derived ids (`f"{parent.id}#aug{n}"`), set
`metadata["synthetic"]=True`, carry the parent's `index`; `NeighborExpander`
and any index-based lookup MUST exclude synthetic chunks. Latent today
(HeadingEnricher only augments) — close it in the enricher contract + checks.
**8.3 `add` is not transactional.** Best-effort across representations;
recovery model is idempotent retry (all writes keyed by `chunk.id`);
`StorageError` names the failed representation. **8.4 Kind rename** `"store"`
→ `"vector_store"` (one grep; stale configs get a clear
`ComponentNotFoundError`). **8.5 App-script hygiene** (unchanged):
`load_dotenv()` in the app is correct per the secrets policy; print
`answer.citations` to watch provenance work.

---

## 9. Invariants audit

**Contracts, not coupling** — new flows speak dataclasses (`SparseVector`,
`VectorSpec`); ChunkIndex depends on abstractions; refiners/retrievers depend
on `ChunkIndex`, never vendors. **Swappability** — kinds keep ≥2 interchangeable
implementations (memory+qdrant stores; hashing+ST embedders; index/hybrid/
fusion retrievers; ≥2 refiners at launch: cross-encoder + neighbor-expander).
**Config-as-data** — serializable specs extend to indexes and refiner chains;
the composite-wiring seam is the named debt (§6.7). **Fingerprints** —
compositional (D8); caching decorators are identity-transparent by rule.
**Provenance** — strengthened: char offsets now power retrieval-time expansion
(G6), payloads carry it end to end, fusion preserves attribution.
**Streaming** — write path O(batch × representations × sinks); chains are
iterator/list transforms; nothing materializes the corpus. **Batteries-
optional** — SPLADE/vendor encoders arrive as lazy-import extras; the memory
store remains the zero-dep reference.

---

## 10. Non-goals (YAGNI fences)

No `QueryTransform` stage kind (rejected deliberately — composition covers
it, F2); no generic DAG-pipeline framework (two chains + one composition axis
+ one root suffice; the day they don't, write DR-0002, don't grow slots); no
store capability-negotiation framework (server-side fusion is a later
optimization behind `HybridRetriever`); no transactional multi-backend writes;
no `completer` kind yet (promotion path documented); no `chunk.vectors`,
ever; no async interfaces.

---

## 11. Migration plan (ordered; each step lands green)

1. **Contracts:** `SparseVector`, `VectorValue`, `VectorSpec`.
2. **Store v2:** contract + kind rename; migrate `MemoryVectorStore`
   (reference impl incl. `fetch` + `update_vectors`), then
   `QdrantVectorStore` (named+sparse, scroll-backed `fetch`,
   create-or-validate). Contract tests: multi-name round-trip, sparse
   round-trip, schema-mismatch raises, membership filters, partial update,
   idempotency, fetch-without-vector.
3. **`SparseEncoder`** interface (+ `Embedder.distance`).
4. **`ChunkIndex`** (A1 constructor, eager validate, fetch) +
   `assert_index_contract` — hermetic on memory store + HashingEmbedder +
   Bm25Index.
5. **Retrieval:** `fusion.py` (dedup-by-id, RRF, attribution, filter fan-out
   — property-tested once, reused everywhere); `IndexRetriever`,
   `FusionRetriever`, `HybridRetriever`; then `MultiQueryRetriever`/
   `HydeRetriever` with a fake `complete`. Delete `DenseRetriever`/
   `Bm25Retriever` after parity.
6. **Refinement:** `Refiner` kind; port cross-encoder as
   `refiner:cross-encoder`; add `NeighborExpander` (+ overlap-merge tests
   built on char offsets); delete `NoOpReranker`; retire `reranking/`.
7. **Pipelines:** `IndexingPipeline(enrich, sinks, batch_size)`;
   `QueryPipeline(retriever, refine, fetch_k)`; `RagPipeline` composition
   root + `.dense()` + wiring guard + retriever derivation; delete
   `_flush`/`_embed`/`_EmbeddingCache`, `NoOpEnricher`.
8. **`CachingEmbedder`** (transparency test:
   `fingerprint() == inner.fingerprint()`; namespace-split test; hit/miss via
   fake blob store).
9. **Enricher identity rule** (8.2) into contract + checks. `.complete` on
   generator adapters.
10. Docs: ARCHITECTURE.md flows; AGENTS.md §7.6 below; minor version bump;
    CHANGELOG the breaking changes.

**AGENTS.md §7.6 (paste-ready, v2):**

> **§7.6 ChunkIndex, composition algebra & multi-representation retrieval
> (DR-0001 v2).** All retrieval representations of a corpus are owned by one
> `ChunkIndex`: `add(chunks)` writes every representation;
> `search(name, TEXT, k, filters)` encodes the query with the same encoder
> that encoded the corpus — never reimplement query encoding elsewhere.
> Constructor uses progressive disclosure: `dense=embedder` auto-names;
> mappings only for multiple representations. **Standing design rule
> (progressive disclosure):** the common case reads like English; the rare
> case is possible; the rare case's ceremony never leaks into the common
> case. Chunks NEVER carry vectors. The composition algebra: pre-retrieval
> variation = composite retrievers (`Fusion`/`Hybrid`/`MultiQuery`/`Hyde` —
> never new pipeline slots); post-retrieval variation = the `refine` chain
> (`Refiner.refine(query, candidates, k)`; `reranker` kind is retired into
> it); write-side = `enrich` chain + `sinks` fan-out (`ChunkSink` — the one
> sanctioned `typing.Protocol`: a capability seam, not a stage contract;
> stage contracts remain ABCs). Fusion always: dedup by `chunk.id`, filters
> fan out to every sub-search, per-source rank attribution in
> `metadata["sources"]`. `VectorStore` is named+typed multi-vector with
> `ensure_schema` create-or-validate, `fetch(filters, limit)` (list values =
> membership), `update_vectors`. Classic BM25 stays a mounted corpus-stats
> `LexicalIndex`; SPLADE-style sparse is a `SparseEncoder` representation.
> `CachingEmbedder` is fingerprint-transparent with separate passage/query
> namespaces. Bare LLM completion is a `Callable[[str], str]` seam
> (`generator.complete`); do not invent a `completer` kind until a third
> independent consumer demands it. Empty chains are the null objects
> (`NoOpReranker`/`NoOpEnricher` are deleted — do not recreate them). Do not
> re-litigate: no retriever write-side, no `chunk.vectors`, no
> `QueryTransform` kind, no DAG framework, no capability negotiation. The
> architecture's acceptance test: the tuner must index once and enumerate
> retrieval/refinement strategies with ZERO tuner-motivated parameters on
> `ChunkIndex` — if one appears, stop and write DR-0002.

---

## 12. Vocabulary

| Term | Meaning |
|---|---|
| Representation | A named way a chunk is searchable: `"dense"`, `"splade"`, `"lexical"` |
| `ChunkIndex` | The aggregate owning all representations of one corpus, write + read |
| Encoder | An `Embedder` (dense) or `SparseEncoder` (static sparse) |
| `VectorSpec` | Declared schema of one named vector space in the store |
| View retriever | Read-only strategy over an index (`IndexRetriever`, `HybridRetriever`) |
| Composition axis | Retrievers wrapping retrievers (`Fusion`, `MultiQuery`, `Hyde`, `Router`) |
| Refiner chain | Ordered `refine=[…]` stages over candidates (Mongo-aggregation style) |
| `ChunkSink` | Structural protocol for anything that consumes chunks at write time |
| Composition root | Where live instances are created once and shared (`RagPipeline` / app) |
| Progressive disclosure | Common case reads like English; rare-case ceremony never leaks in |
