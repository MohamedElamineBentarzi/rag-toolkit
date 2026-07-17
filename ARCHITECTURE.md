# rag-blocks — Architecture

This document is the blueprint of the whole library: the pipeline map, the
data contracts, every stage interface, the design-pattern rationale, and the
design of the evaluation / auto-tuning suite. As of v0.7.0 every subsystem
through generation is implemented; the evaluation & auto-tuning suite (§6)
is the committed v0.8 milestone and remains design-only.

---

## 0. Design principles

These eight rules decide every API question in the codebase. When in doubt,
come back here.

1. **Contracts, not coupling.** Stages never import each other. They agree
   only on typed data shapes (`Source → Page → Document → Chunk → ScoredChunk
   → Answer`). A stage can be replaced by anything that speaks the same
   contract — this is the Strategy pattern applied at architecture scale.
2. **Composition over inheritance.** A hybrid retriever *contains* two
   retrievers; an auto parser *contains* format-specific parsers. Deep
   inheritance trees are forbidden; the only mandatory base is `Component`.
3. **Streaming-first.** The primitive operation of data-producing stages is
   a generator (`iter_pages`, `iter_spans`). Materialization (`parse()`)
   is a convenience layered on top, never the foundation. Memory must not
   scale with document size.
4. **Open/Closed via the registry.** New capability = new registered class.
   Core code is never edited to add a parser, engine, chunker, or store.
5. **Config-as-data.** A pipeline is a serializable dict/YAML spec
   (`{"parser": "docling", "chunker": {"name": "recursive", "size": 512}}`).
   This is what makes auto-tuning possible: the tuner enumerates *data*, not
   code.
6. **Provenance from day one.** Every artifact can answer "where did this
   come from?" (`Chunk.page_start/end` → `PageSpan` → `Source.uri`).
   Citations and error forensics are impossible to bolt on later.
7. **Batteries optional.** Zero-dependency core; every vendor SDK
   (docling, mistralai, qdrant-client, …) is an optional extra, imported
   lazily inside the adapter that needs it.
8. **Everything measurable.** Every component exposes a deterministic
   `fingerprint()` (kind + name + version + config). Fingerprints are the
   cache keys and the identity system of the evaluation suite.

---

## 1. Pipeline map

Two runtime flows plus one offline loop:

```
INDEXING FLOW (offline, per corpus)
┌────────┐   ┌────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌───────┐
│ Source │──▶│ Parser │──▶│ Chunker │──▶│ Enricher │──▶│ Embedder │──▶│ Store │
└────────┘   └────────┘   └─────────┘   └──────────┘   └──────────┘   └───────┘
  file        Iterator      Iterator      Iterator       vectors        upsert
              [Page]        [Chunk]       [Chunk]        attached
              (markdown)                  (optional,
                                          e.g. contextual
                                          retrieval)

QUERY FLOW (online, per question)
┌───────┐   ┌───────────┐   ┌──────────────┐   ┌───────────┐
│ Query │──▶│ Retriever │──▶│ Refiner chain │──▶│ Generator │──▶ Answer + citations
└───────┘   └───────────┘   └──────────────┘   └───────────┘
              index/hybrid/    rerank / expand /   LLM with
              fusion/multi-q    threshold …         provenance-aware
              (composable)      (uniform chain)     context packing

TUNING LOOP (offline, per dataset)
┌─────────────┐    ┌───────┐    ┌───────────────────┐    ┌─────────────┐
│ SearchSpace │──▶│ Tuner │──▶│ Trials (pipelines) │──▶│ Leaderboard │
└─────────────┘    └───────┘    └───────────────────┘    └─────────────┘
  declarative       grid /        stage-output cache       best config +
  choices per       random /      (fingerprint-keyed)      per-stage
  stage             bayesian      + metric evaluation      marginal insights
```

Orchestrators: `IndexingPipeline`, `QueryPipeline`, and a `RagPipeline`
facade that owns both and exposes `index(sources)` / `ask(question)`.
Pipelines are thin — a for-loop over generators plus tracing hooks. All
intelligence lives in components; all wiring lives in config.

---

## 2. Data contracts (`rag_blocks.core.contracts`)

| Artifact      | Produced by | Consumed by          | Key fields                                            |
|---------------|-------------|----------------------|-------------------------------------------------------|
| `Source`      | user        | Parser               | lazy `uri`/`data`, `open()`, `head()`, `content_hash()` |
| `Page`        | Parser      | Chunker / assembly   | `number`, `markdown`, `ocr_applied`                    |
| `Document`    | assembly    | Chunker / Enricher   | `markdown`, `pages: [PageSpan]`, `pages_for_span()`    |
| `Chunk`       | Chunker     | Encoder / ChunkIndex | `text`, `doc_id`, `index`, `page_start/end`            |
| `Query`       | user        | Retriever            | `text`, optional `filters`                             |
| `ScoredChunk` | Retriever   | Refiner / Generator  | `chunk`, `score`, `retriever_name`                     |
| `Citation`    | Generator   | user                 | `marker`, `chunk_id`, `doc_id`, `page_start/end`       |
| `Answer`      | Generator   | user / Evaluator     | `text`, `citations: [Citation]`, `usage`               |
| `SparseVector`| SparseEncoder | VectorStore        | `indices: tuple[int]`, `values: tuple[float]`          |
| `VectorSpec`  | ChunkIndex  | VectorStore          | `name`, `kind: dense\|sparse`, `dimensions?`, `distance` |
| `VectorValue` | encoders    | VectorStore          | alias `list[float] \| SparseVector`                    |

Rules: contracts are plain dataclasses (stdlib only), immutable where they
cross cache boundaries (`Source`, `PageSpan`, `SparseVector`, `VectorSpec`), and
every contract carries `metadata: dict` as a pressure valve so extensions never
require schema changes. The three vector contracts (DR-0001 v2) are the
vocabulary of multi-representation storage; a `Chunk` never carries vectors.

---

## 3. Stage catalog

Each stage is a `Component` subclass with a `kind`, registered under a name.
Signatures below are the committed interfaces; every stage through Generator
is implemented — Evaluator (§3.9) is design-only until v0.8.

### 3.1 Parser — implemented (v0.1)

```python
class Parser(Component):
    kind = "parser"
    def iter_pages(self, source: Source) -> Iterator[Page]: ...   # primitive
    def parse(self, source: Source) -> Document: ...              # convenience
```

Built-ins: `auto` (router/facade), `docling` (PDF/office/HTML/images, hybrid
OCR routing), `plaintext` (txt/md, streaming reference implementation).

### 3.2 Chunker

```python
class Chunker(Component):
    kind = "chunker"
    def chunk(self, document: Document) -> Iterator[Chunk]: ...
    def chunk_stream(self, pages: Iterator[Page]) -> Iterator[Chunk]: ...
```

`chunk_stream` exists so indexing can run Source→Store without ever holding
a whole document (pairs with `Parser.iter_pages`). Planned built-ins:
`recursive` (size/overlap), `markdown-aware` (respects headings from the
parser — this is *why* the ingestion contract is markdown), `semantic`, and a
`chonkie` adapter wrapping the Chonkie library.

### 3.3 Enricher (optional stage)

```python
class Enricher(Component):
    kind = "enricher"
    def enrich(self, chunks: Iterator[Chunk], document: Document) -> Iterator[Chunk]: ...
```

For contextual retrieval (prepend an LLM-generated situating sentence per
chunk), metadata extraction, summaries. It receives the parent `Document`
because context is exactly what a lone chunk lacks. Enrichers compose as a chain
(`enrich=[...]`); the *empty* chain is the null object, so there is no
`NoOpEnricher` (DR-0001 v2, D6).

### 3.4 Embedder

```python
class Embedder(Component):
    kind = "embedder"
    dimensions: int
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

`embed_query` is separate because instruction-tuned models (BGE, E5) prefix
queries differently from passages — hiding that asymmetry in the interface
prevents a whole class of silent retrieval bugs. Planned: `bge-m3`
(dense+sparse+colbert), `sentence-transformers`, `openai`, `voyage`.

### 3.5 VectorStore

```python
class VectorStore(Component):
    kind = "vector_store"                 # renamed from "store" in DR-0001 v2
    def ensure_schema(self, specs: Sequence[VectorSpec]) -> None: ...
    def upsert(self, chunks: Sequence[Chunk],
               vectors: Mapping[str, Sequence[VectorValue]]) -> None: ...
    def search(self, name: str, vector: VectorValue, k: int,
               filters: dict | None = None) -> list[ScoredChunk]: ...
    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]: ...
    def update_vectors(self, name: str, chunk_ids: Sequence[str],
                       vectors: Sequence[VectorValue]) -> None: ...
    def persist(self) -> None: ...
```

A store is a **named, typed, multi-vector** index (DR-0001 v2, D3): each corpus
representation ("dense", "splade", …) is one named space, declared up front via
`ensure_schema` (create-or-validate — a mismatch raises, never coerces).
`fetch` is point retrieval without a query vector (neighbor expansion,
get-by-index); list filter values mean membership (`{"index": [3, 5]}`).
Planned: `memory` (reference impl, tests + tuner), `qdrant` (native named +
sparse vectors), `lancedb`. Classic BM25 is a sibling `LexicalIndex` kind
(corpus-relative, query-time idf — not a stored vector) mounted inside a
`ChunkIndex`; static SPLADE-style sparse is a `SparseEncoder` producing a stored
`SparseVector`. Two scoring models, one read API (Interface Segregation).

### 3.5a ChunkIndex (the aggregate over representations)

```python
class ChunkIndex(Component):              # kind = "index"; wired from instances
    def __init__(self, store, dense=None, sparse=None, lexical=None): ...
    def add(self, chunks): ...            # writes every representation
    def search(self, representation, text, k, filters=None): ...  # TEXT in, not a vector
    def fetch(self, filters, limit=100): ...
```

`ChunkIndex` owns all retrieval representations of one corpus on both paths — the
consistency boundary that guarantees queries are encoded by the same encoder that
encoded the corpus (DR-0001 v2, D1). It is the composition root's shared object:
the write path's flagship `ChunkSink` and the read path's backend.

**BlobStore** (`kind = "blob_store"`, `put`/`get`/`exists` over opaque keys) is
the companion *truth store* — raw ingested bytes and the parse cache live here
under content-addressed keys (`raw/{sha256}/original{ext}`), while the vector
store is derived and rebuildable. Shipped ahead of the v0.3 milestone:
`LocalBlobStore` (on-disk, zero-dep, default, atomic writes) and `MinioBlobStore`
(Adapter over the S3-compatible `minio` client — covers MinIO, AWS S3, R2, B2;
extra `[minio]`). The store attaches no meaning to keys: the content-addressing
convention lives in the caller, keeping the two implementations interchangeable.
A BlobStore is a `Component` for identity/config/redaction, but it sits
*underneath* the fingerprint chain — it is a side-effecting service, not a
stage-output cache key.

### 3.6 Retriever

```python
class Retriever(Component):
    kind = "retriever"
    def retrieve(self, query: Query, k: int = 20) -> list[ScoredChunk]: ...
```

Retrievers are read-only *views* over a `ChunkIndex` and compose like
`nn.Module` (DR-0001 v2, D5): `IndexRetriever` (one representation — replaces the
old `dense`/`bm25`), `HybridRetriever` (sugar fusing all representations of one
index), `FusionRetriever` (the general node — fuses any retrievers: across
representations, across indexes for federation, across paradigms),
`MultiQueryRetriever` / `HydeRetriever` (query shaping via the `complete` seam).
Composing retrievers out of retrievers is the clearest payoff of "composition
over inheritance": no `HybridDenseBM25RRFRetriever` class explosion, and query
shaping is composition, not a new pipeline slot. Fusion mechanics (dedup by
`chunk.id`, RRF, per-source attribution, filter fan-out) live once in
`retrieval/fusion.py`.

### 3.7 Refiner (the post-retrieval chain)

```python
class Refiner(Component):
    kind = "refiner"                      # replaces "reranker" (DR-0001 v2, D9)
    def refine(self, query: Query, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...
```

Everything after retrieval — cross-encoder reranking, sentence-window / parent
expansion, MMR diversity, score floors — is the same shape
(`list[ScoredChunk] → list[ScoredChunk]`), so it is one uniform chain
(`refine=[...]`), the MongoDB-aggregation half of the composition algebra. `k` is
a budget hint; the pipeline enforces the final truncation. Shipped:
`cross-encoder` (the old `bge-reranker`, ported), `keyword`, `neighbor-expander`
(char-offset overlap-safe small-to-big expansion), `score-threshold`. The empty
chain is the null object — there is no `NoOpReranker`.

### 3.8 Generator

```python
class Generator(Component):
    kind = "generator"
    def generate(self, query: Query, context: list[ScoredChunk]) -> Answer: ...
```

Owns prompt template + context packing (token budget, ordering, citation
markers). Returns `Answer` with `citations` resolved through chunk → page
provenance.

### 3.9 Evaluator — implemented (v0.8)

```python
class Evaluator(Component):
    kind = "evaluator"
    stage: ClassVar[Literal["retrieval", "generation"]]
    def evaluate(self, outcomes: Sequence[EvalOutcome]) -> MetricReport: ...
```

Scores outcomes the pipeline **already produced**; it never runs the pipeline
(DR-0002 — this signature was amended from `evaluate(dataset, pipeline)`,
which would have made every evaluator reimplement the run loop and depend
backward on the composition root). The run loop lives once, in the tuner.

Two families with very different costs (see §6): retrieval metrics
(recall@k, MRR, nDCG — pure math, milliseconds) and generation metrics
(faithfulness, answer relevancy — LLM-as-judge, cents per sample). `stage`
records which, and is what the two-phase screening in §6.3 keys off.

Implementations: `ir` (recall@k/MRR/nDCG, binary relevance), `answer-match`
(token-F1 + exact match — vendor-free, so the hermetic suite and the tuner can
score generation with no key), `ragas` (LLM judge, extra `[ragas]`).

**Unlabeled samples are skipped, never scored 0.0** — an aggregate is the mean
over the labeled subset. `0.0` would read as "the pipeline failed this
question" when the truth is "we never asked" (DR-0002 §4, the `ocr_applied`
family of honesty rules).

---

## 4. Pattern glossary

| Pattern | Where | Why it earns its place |
|---|---|---|
| Strategy | every stage interface | swap algorithms without touching callers; the whole premise of the toolkit |
| Adapter | `DoclingParser`, `MistralOcrEngine`, `GoogleDocAiOcrEngine`, `QdrantVectorStore`, `MinioBlobStore`, `SentenceTransformerEmbedder` | vendor APIs normalized behind our contracts; vendor churn stays inside one file |
| Registry + Factory Method | `core.registry` | string → instance; makes pipelines pure data and enables plugins via entry points |
| Facade | `rk.ingest()`, `AutoParser`, `RagPipeline` | one obvious call for the 90% case, full machinery still reachable underneath |
| Template Method | `Parser.parse()` over abstract `iter_pages()` | assembly + provenance implemented once, correctly, for every parser |
| Iterator / generator pipeline | `iter_pages`, `iter_spans`, `recognize_batch` | O(batch) memory, backpressure for free, no queues or threads |
| Composite | `AutoParser`, `HybridRetriever`, `FusionRetriever` | components made of components, uniform to callers (Liskov) |
| Null Object (as empty chain) | empty `refine=[]` / `enrich=[]` | optional stages without `if x is not None`; the empty chain *is* the null object (no `NoOp*` classes) |
| Immutable value objects | `Source`, `PageSpan`, `PageImage` | safe to share across caches and (later) threads |
| Lazy initialization | vendor SDK imports, converter cache, OCR clients | zero-dep core; heavy models loaded once, reused everywhere |

Anti-patterns deliberately avoided: god-object "PipelineManager", inheritance
for configuration (config is data, not subclasses), and eager I/O anywhere.

---

## 5. Ingestion subsystem (implemented)

```
Source ──▶ detect_format (magic bytes, ext tiebreak)
              │
        AutoParser routes  ──▶  txt/md ──▶ PlainTextParser (streamed pages)
              │
              ▼
        DoclingParser
              │
     PDF? ────┼──── no ──▶ whole-doc docling convert ──▶ Pages
              ▼ yes
     OCR policy × engine matrix:
        no external engine → docling windows (do_ocr per policy)
        engine + NEVER     → docling windows, no OCR
        engine + FORCE     → every page rendered → OcrEngine
        engine + AUTO      → per-page text-layer probe (pdfium char count)
                              digital segments → docling windows (no OCR)
                              scanned segments → render → OcrEngine
              │
              ▼
        Iterator[Page] ──▶ Document.from_pages (offsets → PageSpan provenance)
```

Key decisions, condensed (full rationale in module docstrings):

- **Bytes over extensions** for detection; ZIP family disambiguated by member
  paths (`word/` vs `ppt/` vs `xl/`).
- **Policy ≠ engine.** *When* to OCR (`OcrPolicy`) and *how* (`OcrEngine`)
  are orthogonal axes; conflating them is why other libraries make "Mistral
  OCR only on scanned pages" impossible to express.
- **Per-page routing** because real corpora contain mixed digital/scanned
  PDFs; the probe costs microseconds per page (text-layer char count, no
  rendering).
- **Windowed conversion** (`page_batch_size`) bounds memory on huge PDFs;
  non-PDF formats convert whole because they offer no sub-file random access
  and are rarely large.
- **Converter/client caching**: docling's layout models and OCR HTTP clients
  are constructed once per configuration and reused for every window of every
  document — the dominant throughput factor.
- **Provenance spans** recorded at assembly, enabling chunk→page citations
  downstream at zero extra cost.

Failure policy: errors raise `ParseError`/`OcrError` with source + page
context. A `on_error="skip"` mode (log and continue) is a planned config
addition for large batch runs.

---

## 6. Evaluation & auto-tuning suite — implemented (v0.8)

The differentiating feature: **given a labeled dataset, find the best
pipeline configuration, with every trial logged and explainable.**

Built per this section, with five refinements recorded in
[DR-0002](docs/decisions/DR-0002-evaluator-contract.md) and
[DR-0003](docs/decisions/DR-0003-tuning-and-caching.md): evaluators score data
instead of driving pipelines (§3.9 was amended); `Tuner.run()` is a Template
Method over `iter_candidates`; **no stage-output cache was built** because §6.2's
formula is already materialized by the blob parse cache and `CachingEmbedder`
(the tuner contributes *enumeration order* — measured: 12 combinations, 1
parse); `cost` splits cache-confounded `index_ms` from clean `query_ms`, and
`api_usd` is never guessed; and `EvalSample` grew `relevant_doc_ids`, without
which §6.4's own headline example (tuning chunk size) is unmeasurable — a
chunk-level label silently denotes a different passage under a different
chunker.

User-facing guide: [`docs/guide/11-evaluation-and-tuning.md`](docs/guide/11-evaluation-and-tuning.md).
The committed regression baseline lives in `benchmarks/baseline/`.

### 6.1 Vocabulary

```python
@dataclass(frozen=True)
class EvalSample:      # one row of the user's dataset — implemented (v0.8)
    question: str
    relevant_chunk_ids: tuple[str, ...] | None = None   # retrieval metrics
    reference_answer: str | None = None                 # generation metrics
    filters: dict | None = None
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class EvalOutcome:     # what one pipeline produced for one sample (DR-0002)
    sample: EvalSample
    retrieved: tuple[ScoredChunk, ...] = ()
    answer: Answer | None = None           # None after phase 1: normal, not an error

class SearchSpace:
    """Declarative choices per stage — the tuner's input."""
    # space = SearchSpace(
    #     chunker=[choice("recursive", size=[256, 512, 1024], overlap=[0, 64]),
    #              choice("markdown-aware")],
    #     retriever=[choice("index", representation="dense"), choice("hybrid")],
    #     refine=[[], [choice("cross-encoder", top_k=[5, 10])],
    #             [choice("neighbor-expander"), choice("cross-encoder")]],
    # )

class Tuner(Component):            # Strategy, again
    kind = "tuner"                  # "grid" | "random" | "bayesian" | "halving"
    def run(self, space: SearchSpace, dataset: EvalDataset) -> Leaderboard: ...

@dataclass
class Trial:                        # one tested combination — fully reproducible
    trial_id: str
    pipeline_spec: dict             # component describe()s — secrets redacted
    fingerprints: dict[str, str]    # per stage
    metrics: dict[str, float]       # recall@10, ndcg@10, faithfulness, ...
    cost: dict[str, float]          # latency_ms, tokens, api_usd
    cache_hits: dict[str, bool]     # which stages were reused
    started_at / finished_at: str
```

Trials append to a JSONL log (plus SQLite index for querying); the
`Leaderboard` is a view over trials.

### 6.2 The caching insight (what makes tuning tractable)

Pipelines form a DAG of stage outputs. Two combinations that share a prefix
share work:

```
cache_key(stage N) = sha256(
    dataset/source content hashes
    + fingerprint(stage 1) + ... + fingerprint(stage N)
)
```

Example — 2 chunkers × 2 embedders × 3 retrievers × 2 rerankers = 24
pipelines. Naively: 24 parses, 24 chunk runs, 24 embedding runs. With
fingerprint-keyed caching: **1 parse, 2 chunk runs, 4 embedding runs**, and
only retrieval/rerank/eval vary per trial. Parsing and embedding are
typically >90% of wall-clock cost, so this is the difference between an
overnight grid search and a coffee-break one. This is exactly why
`Component.fingerprint()` exists in the very first commit.

### 6.3 Two-phase evaluation (cost control)

Retrieval metrics (recall@k, MRR, nDCG) are free; LLM-judged generation
metrics (faithfulness, relevancy) are not. So:

1. **Phase 1 — screen**: run *all* combinations, score retrieval metrics
   only. Rank.
2. **Phase 2 — finals**: run generation + LLM-judge on the top-N (configurable,
   default 5) combinations only.

Optionally combine with successive halving: phase 1 on a dataset subset,
survivors graduate to the full set. Judge verdicts are themselves cached by
(question, answer, judge-model) hash so re-runs cost nothing.

### 6.4 Insights, not just a winner

The leaderboard computes **per-stage marginal analysis** by grouping trials:
"averaged over all other choices, the `cross-encoder` refiner adds +0.07
nDCG@10 for +180 ms/query; chunk size 512→1024 costs −0.04 recall@10." That per-dimension
attribution — quality *and* cost — is the "deep insights" deliverable, and it
falls out of the trial log structure for free.

---

## 7. Performance notes

- Reuse converters/models/clients (done in v0.1); never construct per file.
- `OcrEngine.recognize_batch` is the hook for HTTP parallelism; the default
  stays sequential and O(1)-memory.
- Embedding stage will batch by token count, not item count.
- Async is a v0.5+ concern: interfaces stay sync (simple to implement); an
  async executor can wrap components without changing them.
- Rule of thumb enforced by review: no stage may hold more than one window /
  batch of data at a time.

---

## 8. Extending the toolkit

A custom component is a class + a decorator (see README for a full OCR
example). Contract tests to satisfy:

1. Declares `kind`, `name`, optionally `Config` (a dataclass).
2. Pure function of (config, inputs) — no hidden global state, so
   fingerprint-based caching stays sound.
3. Streaming stages yield lazily and never materialize the whole input.
4. Vendor imports happen lazily inside methods.
5. Bump `version` on behavioral change (cache invalidation).

Third-party packages publish components via the `rag_blocks.components`
entry-point group; they appear in the registry automatically.

---

## 9. Repository layout & roadmap

```
rag_blocks/
  core/          contracts (+ SparseVector/VectorSpec), component, registry [v0.1 ✓]
  ingestion/     detection, parsers/, ocr/                     [v0.1 ✓]
  chunking/      fixed, markdown-aware ✓; chonkie adapter      [v0.2]
  enrichment/    heading ✓, contextual ✓ (LLM)                  [v0.2]
  embedding/     hashing ✓, sentence-transformers ✓, caching ✓; sparse_encoder (iface) [v0.3/0.6]
  storage/       blob (local,minio) + vector_store (memory,qdrant, multi-vector) ✓ [v0.3/0.6]
  indexing/      ChunkIndex ✓ (aggregate over representations), ChunkSink [v0.6]
  retrieval/     index ✓, hybrid ✓, fusion ✓, multi-query ✓, hyde ✓ (composition axis) [v0.4/0.6]
  refinement/    keyword ✓, cross-encoder ✓, neighbor-expander ✓, score-threshold ✓ (was reranking/) [v0.4/0.6]
  generation/    extractive ✓, anthropic ✓ (+.complete); packing+citations ✓ [v0.5]
  evaluation/    retrieval + LLM-judge metrics                 [v0.8]
  tuning/        search space, tuners, trial log, leaderboard  [v0.8]
  pipeline.py    IndexingPipeline ✓ QueryPipeline ✓ RagPipeline ✓ (composition root) [v0.2+]
```

Each milestone ships with contract tests, a cookbook example, and at least
two interchangeable implementations per stage — the point of the library is
proven by swapping.
