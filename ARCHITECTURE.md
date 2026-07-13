# rag-toolkit — Architecture

This document is the blueprint of the whole library: the pipeline map, the
data contracts, every stage interface (including the ones not yet coded), the
design-pattern rationale, and the design of the evaluation / auto-tuning
suite. The ingestion subsystem (v0.1) is fully implemented; everything else
is specified here so each future stage drops into a slot that already exists.

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
   a generator (`iter_pages`, `chunk_stream`). Materialization (`parse()`)
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
┌───────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐
│ Query │──▶│ Retriever │──▶│ Reranker │──▶│ Generator │──▶ Answer + citations
└───────┘   └───────────┘   └──────────┘   └───────────┘
              dense/sparse/    cross-        LLM with
              hybrid/fusion    encoder       provenance-aware
              (composable)     (or NoOp)     context packing

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

## 2. Data contracts (`rag_toolkit.core.contracts`)

| Artifact      | Produced by | Consumed by          | Key fields                                            |
|---------------|-------------|----------------------|-------------------------------------------------------|
| `Source`      | user        | Parser               | lazy `uri`/`data`, `open()`, `head()`, `content_hash()` |
| `Page`        | Parser      | Chunker / assembly   | `number`, `markdown`, `ocr_applied`                    |
| `Document`    | assembly    | Chunker / Enricher   | `markdown`, `pages: [PageSpan]`, `pages_for_span()`    |
| `Chunk`       | Chunker     | Embedder / Store     | `text`, `doc_id`, `index`, `page_start/end`            |
| `Query`       | user        | Retriever            | `text`, optional `filters`                             |
| `ScoredChunk` | Retriever   | Reranker / Generator | `chunk`, `score`, `retriever_name`                     |
| `Answer`      | Generator   | user / Evaluator     | `text`, `citations: [ChunkRef]`, `usage`               |

Rules: contracts are plain dataclasses (stdlib only), immutable where they
cross cache boundaries (`Source`, `PageSpan`), and every contract carries
`metadata: dict` as a pressure valve so extensions never require schema
changes.

---

## 3. Stage catalog

Each stage is a `Component` subclass with a `kind`, registered under a name.
Signatures below are the committed interfaces for future versions.

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
because context is exactly what a lone chunk lacks. `NoOpEnricher` is the
default (Null Object pattern: pipeline code has zero `if enricher:` branches).

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
    kind = "store"
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None: ...
    def search(self, vector: list[float], k: int, filters: dict | None = None) -> list[ScoredChunk]: ...
    def persist(self) -> None: ...
```

Planned: `memory` (numpy, for tests and tuning on small corpora), `qdrant`,
`lancedb`. Lexical (BM25) indexes implement a sibling `LexicalIndex` kind so
hybrid retrieval composes two narrow interfaces instead of one fat one
(Interface Segregation).

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

Planned: `dense`, `bm25`, `hybrid` (Composite: owns a dense and a sparse
retriever plus a fusion strategy — RRF or weighted), `multi-query`. Composing
retrievers out of retrievers is the clearest payoff of "composition over
inheritance": no `HybridDenseBM25RRFRetriever` class explosion.

### 3.7 Reranker

```python
class Reranker(Component):
    kind = "reranker"
    def rerank(self, query: Query, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...
```

Planned: `bge-reranker` (cross-encoder), `cohere`, `noop` (Null Object —
also the honest baseline the tuner compares against).

### 3.8 Generator

```python
class Generator(Component):
    kind = "generator"
    def generate(self, query: Query, context: list[ScoredChunk]) -> Answer: ...
```

Owns prompt template + context packing (token budget, ordering, citation
markers). Returns `Answer` with `citations` resolved through chunk → page
provenance.

### 3.9 Evaluator

```python
class Evaluator(Component):
    kind = "evaluator"
    stage: Literal["retrieval", "generation"]
    def evaluate(self, dataset: EvalDataset, pipeline: RagPipeline) -> MetricReport: ...
```

Two families with very different costs (see §6): retrieval metrics
(recall@k, MRR, nDCG — pure math, milliseconds) and generation metrics
(faithfulness, answer relevancy — LLM-as-judge, cents per sample).

---

## 4. Pattern glossary

| Pattern | Where | Why it earns its place |
|---|---|---|
| Strategy | every stage interface | swap algorithms without touching callers; the whole premise of the toolkit |
| Adapter | `DoclingParser`, `MistralOcrEngine`, `GoogleDocAiOcrEngine`, future store/embedder wrappers | vendor APIs normalized behind our contracts; vendor churn stays inside one file |
| Registry + Factory Method | `core.registry` | string → instance; makes pipelines pure data and enables plugins via entry points |
| Facade | `rk.ingest()`, `AutoParser`, future `RagPipeline` | one obvious call for the 90% case, full machinery still reachable underneath |
| Template Method | `Parser.parse()` over abstract `iter_pages()` | assembly + provenance implemented once, correctly, for every parser |
| Iterator / generator pipeline | `iter_pages`, `chunk_stream`, `recognize_batch` | O(batch) memory, backpressure for free, no queues or threads |
| Composite | `AutoParser`, future `HybridRetriever` | components made of components, uniform to callers (Liskov) |
| Null Object | `NoOpReranker`, `NoOpEnricher` | optional stages without `if x is not None` scattered through pipelines |
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

## 6. Evaluation & auto-tuning suite (design)

The differentiating feature: **given a labeled dataset, find the best
pipeline configuration, with every trial logged and explainable.**

### 6.1 Vocabulary

```python
@dataclass
class EvalSample:      # one row of the user's dataset
    question: str
    relevant_chunk_ids: list[str] | None   # for retrieval metrics
    reference_answer: str | None           # for generation metrics

class SearchSpace:
    """Declarative choices per stage — the tuner's input."""
    # space = SearchSpace(
    #     chunker=[choice("recursive", size=[256, 512, 1024], overlap=[0, 64]),
    #              choice("markdown-aware")],
    #     retriever=[choice("dense"), choice("hybrid", fusion=["rrf"])],
    #     reranker=[choice("noop"), choice("bge-reranker", top_k=[5, 10])],
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
"averaged over all other choices, `bge-reranker` adds +0.07 nDCG@10 for
+180 ms/query; chunk size 512→1024 costs −0.04 recall@10." That per-dimension
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

Third-party packages publish components via the `rag_toolkit.components`
entry-point group; they appear in the registry automatically.

---

## 9. Repository layout & roadmap

```
rag_toolkit/
  core/          contracts, component, registry, errors        [v0.1 ✓]
  ingestion/     detection, parsers/, ocr/                     [v0.1 ✓]
  chunking/      fixed, markdown-aware ✓; chonkie adapter      [v0.2]
  enrichment/    noop ✓, heading ✓, contextual ✓ (LLM)          [v0.2]
  embedding/     hashing ✓, sentence-transformers ✓; API adapters [v0.3]
  storage/       blob (local,minio) + vector (memory,qdrant) ✓  [v0.3]
  retrieval/     dense ✓, bm25 ✓, hybrid RRF ✓                 [v0.4]
  reranking/     noop ✓, keyword ✓, bge-reranker ✓             [v0.4]
  generation/    extractive ✓, anthropic ✓; packing+citations ✓ [v0.5]
  evaluation/    retrieval + LLM-judge metrics                 [v0.6]
  tuning/        search space, tuners, trial log, leaderboard  [v0.7]
  pipeline.py    IndexingPipeline ✓ QueryPipeline ✓ RagPipeline ✓ [v0.2+]
```

Each milestone ships with contract tests, a cookbook example, and at least
two interchangeable implementations per stage — the point of the library is
proven by swapping.
