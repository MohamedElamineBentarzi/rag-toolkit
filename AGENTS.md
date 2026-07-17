# AGENTS.md — rag-blocks agent context

**Read this file completely before writing or modifying any code.** It is the
canonical knowledge transfer from the project's design phase. It contains the
philosophy, every design decision and its rationale, the semantics of the data
contracts, specs for components that are designed but NOT yet coded, and the
rules any contribution must follow. When this file conflicts with your general
habits, this file wins. When code conflicts with this file, flag it — do not
silently "fix" either side.

Recommended reading order before your first change:
`AGENTS.md` (this file) → `ARCHITECTURE.md` → `rag_blocks/core/contracts.py`
→ `rag_blocks/core/component.py` → `rag_blocks/ingestion/parsers/base.py`
→ one concrete parser (`plaintext.py`, then `docling_parser.py`) →
`tests/contract_checks.py`. Module docstrings are load-bearing documentation,
not decoration — they explain *why*, and you are expected to write in the
same style.

---

## 1. What this project is

`rag-blocks` is an open-source Python library of **composable building blocks
for production RAG pipelines**: every stage (parsing, chunking, embedding,
storage, retrieval, reranking, generation, evaluation) is a swappable
component behind a stable contract, every pipeline is a serializable config,
and an auto-tuning evaluation suite finds the best component combination for a
given dataset — with full trial logs and per-stage insights.

The differentiators over neighbors (AutoRAG is the closest competitor;
Haystack/LlamaIndex are the frameworks):

1. **Swappability as the product.** Not a framework you live inside — blocks
   you compose. "SWAPPABLE" is the owner's one-word summary of the project.
2. **Streaming-first ingestion** with per-page OCR routing to *any* engine
   (Mistral, Google Document AI, custom) — memory never scales with document
   size.
3. **Fingerprint-keyed cross-pipeline caching** that makes tuning tractable
   (shared stage prefixes are computed once across all trial combinations).
4. **Provenance end to end**: every chunk can answer "which pages of which
   file", enabling citations.

Owner context that shapes decisions: this is the maintainer's **first
open-source project** and is explicitly a learning vehicle for clean design.
Code quality, pattern discipline, and explanatory docstrings are requirements,
not nice-to-haves. He is reading *Clean Code*; honor its spirit (small
functions, intention-revealing names, no clever tricks — there is a comment in
`plaintext.py` where a "clever" one-liner was deliberately rewritten as two
clear lines; that is the house style).

Current state: **v0.7.0 — eight subsystems implemented and tested** (core,
ingestion, chunking, enrichment, embedding, storage, retrieval + refinement,
generation; 237 hermetic tests + 13 opt-in integration tests). The evaluation
& auto-tuning suite (§7.3) is the committed v0.8 milestone and the one major
spec in this file not yet coded.

---

## 2. The prime directive: design principles are hard requirements

The owner's explicit instruction: *use design patterns and principles so the
code and architecture stay as clean as possible.* Every PR is judged against
these eight principles. They are not aspirational.

1. **Contracts, not coupling.** Stages never import each other. They agree
   only on the typed dataclasses in `core/contracts.py`
   (`Source → Page → Document → Chunk → ScoredChunk → Answer`). A Chunker
   must not know what a Parser is.
2. **Composition over inheritance.** The only mandatory base is `Component`
   plus the stage ABC. Never create deep hierarchies; a hybrid retriever
   *contains* two retrievers.
3. **Streaming-first.** Data-producing primitives are generators
   (`iter_pages`). Materializing conveniences (`parse()`) are layered on top
   via Template Method. No stage may hold more than one window/batch of data
   at a time. Memory must not scale with input size.
4. **Open/Closed via the registry.** New capability = new registered class.
   Adding a parser/engine/chunker must require ZERO edits to existing files
   (except an import in the subsystem `__init__.py` for built-ins).
5. **Config-as-data.** Pipelines are serializable dicts/YAML. Behavior
   differences come from config, never from subclassing-for-configuration.
6. **Provenance from day one.** Every artifact must answer "where did this
   come from". Never drop offsets, page numbers, or source references —
   they cannot be reconstructed later.
7. **Batteries optional.** `rag_blocks.core` has ZERO third-party
   dependencies (stdlib dataclasses, not pydantic — deliberate). Every vendor
   SDK is a pip extra, imported lazily *inside the method that uses it*,
   with an actionable ImportError message naming the extra.
8. **Everything measurable.** Every component has a deterministic
   `fingerprint()` = sha256(kind, name, version, redacted-config)[:16].
   Fingerprints are cache keys and trial identity. **If you change a
   component's behavior, bump its `version`** — that is how caches
   invalidate. Never change fingerprint semantics casually.

Corollary the owner has internalized and expects you to apply:
**"Testability is the first consumer of the architecture. If a change is hard
to test, the design is wrong"** — fix the design (extract a pure function,
inject through a seam), don't write a heroic test.

---

## 3. Architecture core: the two-layer class hierarchy

This confused the owner once; it is now settled and must not be redesigned.

```
Component                      layer 1 — shared PLUMBING, no domain logic:
 │                                (kind, name, version) identity, config
 │                                dataclass merging, describe()/fingerprint()
 │                                with secret redaction
 ├── Parser        (abstract iter_pages)      ┐
 ├── OcrEngine     (abstract recognize)       │ layer 2 — one ABC per stage,
 ├── Chunker       (abstract iter_spans)      │ carries the stage CONTRACT
 ├── Embedder, VectorStore, Retriever, ...    ┘ via @abstractmethod
 └────── concrete implementations (DoclingParser, MistralOcrEngine, ...)
```

Why one `Component` grandparent: the plumbing is identical for all stages
(DRY), and the registry/pipelines/tuner need a **common type** to hold
heterogeneous collections ("give me the fingerprints of all 6 components of
trial #14"). Why per-stage ABCs on top: each stage has its own contract.
In Java terms: stage ABC = `interface`, `Component` = shared `abstract class`.

Enforcement is three layers — implement all three for any new stage:
1. `@abstractmethod` → `TypeError` at instantiation (runtime).
2. Type hints + mypy in CI (Python's "compile time"; ABCs don't check
   signatures, mypy does).
3. **Behavioral contract tests** (`tests/contract_checks.py`) for what
   neither can check (ordering, span validity, determinism). Every new stage
   kind gets an `assert_<stage>_contract()` helper; every implementation's
   tests must call it.

`typing.Protocol` was considered and rejected: we want inherited *behavior*
(config, fingerprint) and registration, not just structural shape. Do not
migrate to Protocols.

Registry mechanics (`core/registry.py`): `@registry.register` class decorator
reads `kind`/`name` from the class (single source of truth, decorator takes no
args). Re-registering the *same* class is idempotent; a *different* class
under an existing key raises. Third-party plugins load lazily via the
`rag_blocks.components` entry-point group; a broken plugin must never crash
core (exceptions are swallowed per entry point). `registry.create(kind, name,
**overrides)` is the Factory Method everything uses.

Config mechanics (`core/component.py`): each component optionally declares a
nested `@dataclass class Config`. `__init__(config=None, **overrides)` accepts
a ready Config, keyword overrides, or both (overrides win via
`dataclasses.replace`); unknown keys → `ConfigError` (fail fast).
`describe()` redacts any config field whose lowercase name contains one of
`("key", "token", "secret", "password", "credential")` and normalizes enums to
`.value`. `fingerprint()` hashes the *redacted* describe — consequence:
rotating an API key never invalidates caches, and secrets can never appear in
logs or trial records.

---

## 4. Data contract semantics (the subtle parts)

The dataclasses are in `core/contracts.py`; what follows is the *meaning* an
agent must preserve.

**`Source`** is a lazy pointer (path or small in-memory bytes), frozen.
Never eagerly read content; access via `open()` (streams), `head(n)` (sniff),
`content_hash()` (streaming sha256 — a future cache key). Derive variants
with `with_format()` / `dataclasses.replace`, never mutate.

**`Page`** is the streaming unit of ingestion. 1-based `number`.
`ocr_applied=True` only when we *know* OCR produced the text (external engine
or FORCE); docling AUTO OCRs bitmap regions selectively and doesn't tell us,
so we don't lie.

**`Document` is a fact; a chunking is an interpretation of it.** A Document is
parsed content + provenance, cached under (source hash × parser fingerprint).
It must NEVER hold chunks, know about chunkers, or grow a `get_chunk(i)`
method — the same document legitimately has many simultaneous chunkings (the
tuner depends on this). **Arrows point backward, like database foreign keys:**
`Chunk` carries `doc_id`; `Document` has no forward references. "Get chunk by
index" lives where chunks live — the vector store, via payload filter
(`doc_id == X AND index IN (...)`).

**`PageSpan`** records char offsets `[start, end)` of each page inside
`Document.markdown` (assembled with `PAGE_SEPARATOR = "\n\n"`). The invariant
tests enforce: `doc.markdown[span.start:span.end] == page.markdown`, spans
ordered and non-overlapping. **`Document.pages_for_span(start, end)` is the
designed bridge between parsing and chunking** — chunkers resolve page
provenance through it and through nothing else.

**`Chunk` field semantics:**
- `id`: deterministic, `f"{doc_id}:{index}"` → idempotent re-indexing
  (re-running upserts overwrites instead of duplicating).
- `index`: **reading-order position within the document, contiguous 0-based,
  NO holes** — even when whitespace-only spans are skipped, the counter must
  not skip (use a manual counter, not `enumerate` over raw spans). Reason:
  neighbor expansion at query time fetches `index ± 1` from the store to give
  the generator surrounding context; relevance order (retrieval) and reading
  order (index) are different orderings and both are needed.
- `page_start` / `page_end`: a *range* because chunks legitimately cross page
  boundaries. They are `Optional` meaning **"not always applicable"** — for
  any chunk sliced from a parsed document the base chunker ALWAYS fills them;
  `None` is reserved for synthetic chunks (enricher-generated summaries,
  synthesized Q/A) that never came from a document's markdown. A doc-derived
  chunk with `None` pages is a bug.
- Committed contract change for v0.2: **promote `char_start`/`char_end` to
  first-class `Chunk` fields** (currently they'd sit in metadata). Char
  offsets are the primary provenance; pages are derived from them.
- `metadata: dict` exists on every contract as a pressure valve so extensions
  never force schema changes.

---

## 5. Pattern glossary (use these names in docstrings and reviews)

| Pattern | Where it lives | Why |
|---|---|---|
| Strategy | every stage interface | swap algorithms without touching callers — the product thesis |
| Adapter | `DoclingParser`, `MistralOcrEngine`, `GoogleDocAiOcrEngine`; future `ChonkieChunker`, `RagasEvaluator`, `MinioBlobStore`, `QdrantStore` | vendor churn stays inside one file |
| Registry + Factory Method | `core/registry.py` | string → instance; pipelines become data; plugin ecosystem via entry points |
| Facade | `rk.ingest()`, `AutoParser`, future `RagPipeline` | one obvious call for the 90% case |
| Template Method | `Parser.parse()` over `iter_pages()`; committed for `Chunker.chunk()` over `iter_spans()` | bookkeeping written once, correctly; strategies implement ONE primitive |
| Iterator / generator pipeline | `iter_pages`, `recognize_batch`, future `iter_spans` | O(batch) memory, backpressure free |
| Composite | `AutoParser`, `HybridRetriever`, `FusionRetriever` | components made of components, uniform to callers |
| Null Object (as empty chain) | empty `refine=[]` / `enrich=[]` | optional stages without `if x is not None` litter; the empty chain *is* the null object (no `NoOp*` classes — DR-0001 v2) |
| Immutable value objects | `Source`, `PageSpan`, `PageImage` | safe across caches/threads |
| Lazy initialization | vendor imports, docling converter cache, OCR clients | zero-dep core; heavy models built once, reused |

**Forbidden anti-patterns:** god objects ("PipelineManager" that knows
everything), inheritance-for-configuration, eager whole-file reads, stages
importing sibling stages, `localStorage`-style hidden global state inside
components (components must be pure functions of (config, inputs) or
fingerprint caching becomes unsound), swallowing exceptions without context,
returning strings where offsets/spans are available.

---

## 6. Ingestion subsystem — operational knowledge

Flow: `detect_format` (magic bytes first — files lie about extensions; ZIP
family disambiguated by member paths `word/`→docx, `ppt/`→pptx, `xl/`→xlsx;
extension only as tiebreaker for signatureless text) → `AutoParser` routes via
its `routes` config dict (data, overridable) → delegate parser.

**OCR is two orthogonal axes — never merge them:**
- `OcrPolicy` = WHEN (a decision): `AUTO` probe each page's embedded text
  layer, OCR only pages below `min_chars_digital` (32 chars — above stray
  page-number noise, below real content); `FORCE` = OCR everything (rescues
  scanner-generated garbage text layers); `NEVER` = text layer only.
- `OcrEngine` = HOW (a Strategy): tiny interface, `recognize(PageImage) →
  OcrResult`. Engines know nothing about PDFs/pages/documents.

Dispatch matrix in `DoclingParser._iter_pdf` (tested exhaustively in
`test_docling_routing.py::test_pdf_dispatch_matrix` — keep that test green):
no external engine → delegate policy to docling's own OCR options; external
engine + NEVER → docling no-OCR; external engine + AUTO → hybrid per-page
routing (pdfium char-count probe, consecutive same-kind pages grouped into
segments so docling keeps efficient windows); external + FORCE → every page
rendered (200 dpi PNG) → engine.

Memory strategy: PDFs are random-access → processed in windows of
`page_batch_size` (8) pages via docling `page_range`; office formats have no
sub-file random access → converted whole (deliberate asymmetry — they're
rarely huge). The pdfium document opens ONCE per file for probe + rendering;
one page bitmap in memory at a time; images stream through
`recognize_batch()` (the parallelism hook engines may override).

Throughput rule: docling converters (layout models) and OCR HTTP clients are
expensive — **cache per option-set on the parser instance, reuse across all
windows and documents**. Never construct per file/page.

Known API-drift guards (do not remove; verify on dependency bumps):
docling `page_range` requires >= 2.15; per-page
`export_to_markdown(page_no=...)` is wrapped in `try/except TypeError` with a
window-level fallback (Page carries `metadata["page_span"]` — provenance
degrades honestly, never wrongly); pypdfium2 `count_chars()` falls back to
`len(get_text_bounded())`; the `mistralai` call signature
(`client.ocr.process(model=..., document={"type": "image_url", ...})`) was
written against SDK 1.x and must be re-verified against current docs before a
release. `GoogleDocAiOcrEngine` returns plain text (valid markdown);
reconstructing headings/tables from DocAI layout entities is a welcome
improvement that must stay entirely inside that adapter.

File-naming note: the docling module is `docling_parser.py`, not `docling.py`
— avoids tooling confusion with the real `docling` package. Follow the same
caution for future vendor adapters.

---

## 7. Committed designs and their rationale (status marked per subsection)

These were decided in design discussion with the owner. Most are now
implemented — their specs remain here as the rationale of record and the
standard any change is reviewed against. §7.3 is the one still-uncoded spec:
implement it to the letter. Do not re-litigate any of them; do refine details
that don't contradict them, and flag (don't silently fix) code/spec conflicts.

### 7.1 Chunker *(implemented in v0.2 — spec kept as rationale)*

```python
class Chunker(Component):
    kind = "chunker"

    @abstractmethod
    def iter_spans(self, document: Document) -> Iterator[tuple[int, int]]:
        """The ONLY strategy decision: WHERE to cut, as half-open char
        offsets [start, end) into document.markdown. Cut coordinates,
        not copies."""

    def chunk(self, document: Document) -> Iterator[Chunk]:
        # Template Method — ALL bookkeeping lives here, once:
        # - slice text = document.markdown[start:end]
        # - skip whitespace-only slices WITHOUT advancing index
        #   (manual counter; index stays contiguous 0-based, no holes)
        # - id = f"{document.id}:{index}" (deterministic)
        # - pages = document.pages_for_span(start, end);
        #   page_start/page_end ALWAYS filled here
        # - char_start/char_end stored as first-class Chunk fields
```

Rules: spans yielded in reading order of `start`; **overlapping spans are
legal** (overlap strategies express naturally in coordinates — this is *why*
strategies emit spans, not strings: return strings and provenance,
overlap, and neighbor merging all die). Strategies are config-only.

Implementations — `fixed` and `markdown-aware` shipped; `chonkie` and
`semantic` still wanted: `fixed` (chunk_chars=1600, overlap_chars=200; prefer
cutting at `\n\n`, refuse a soft cut that would leave < size/2 — mirror the
newline-preference logic in `PlainTextParser._cut_point`), `markdown-aware`
(cut at heading positions — this is the payoff of normalizing ingestion to
markdown: structure survives to the cutting decision), `chonkie` (Adapter —
Chonkie chunks already expose `start_index`/`end_index`, map them straight to
spans; Chonkie is a preferred dependency, extra `[chonkie]`), `semantic`
(later). **No `chunk_stream` in v0.2** — deliberate YAGNI: markdown of even a
2,000-page PDF is a few MB; keep the streaming hook documented for a future
version, don't build it now. Ship
`tests/contract_checks.py::assert_chunker_contract` (index contiguity, span
ordering/bounds, slices match text, page fields filled, determinism) alongside.

### 7.2 BlobStore *(implemented in v0.3)*

New component kind `"blob_store"`: `put(key: str, data: bytes)`,
`get(key) -> bytes`, `exists(key) -> bool` (streaming variants may come
later). Implementations: `LocalBlobStore` (filesystem, zero-dep, default) and
`MinioBlobStore` (Adapter over the `minio` SDK — which is Apache-2.0 even
though the MinIO *server* is AGPL; extra `[minio]`, S3-compatible so it covers
AWS too).

Content-addressed layout — note the second key IS the tuner's parse cache
materialized (this is why `Source.content_hash()` and `fingerprint()` exist
since v0.1):

```
raw/{sha256}/original{ext}                 immutable source of truth (dedup free)
parsed/{sha256}/{parser_fingerprint}.md    the parse cache
parsed/{sha256}/{parser_fingerprint}.meta.json   spans, ocr pages, doc metadata
```

Principle: **blob store = truth; Qdrant = derived and rebuildable.** Chunk
text + `{doc_id, index, page_start, page_end}` are duplicated into the Qdrant
payload so query time never touches the blob store; re-embedding with a new
model reads markdown from the blob store and never re-parses. Trial logs are
NOT blobs — they go to JSONL + SQLite.

### 7.3 Evaluation & tuning *(implemented in v0.8 — DR-0002, DR-0003)*

`Evaluator` kind with `stage: "retrieval" | "generation"`. Two families by
cost: classic IR metrics (recall@k, MRR, nDCG — pure math, no LLM) and
LLM-judged. **RAGAS integrates as `RagasEvaluator`, an Adapter** translating
our trial data (question, retrieved contexts, answer, ground truth) into a
RAGAS `EvaluationDataset` (faithfulness, answer_relevancy,
context_precision/recall) and mapping scores back to our `MetricReport`.
Two-phase evaluation: phase 1 screens ALL combinations with IR metrics;
phase 2 runs RAGAS/LLM-judge on the top-N (default 5) only. Judge verdicts
cached by (question, answer, judge-model) hash.

Tuner is a Strategy (`grid`, `random`; `bayesian`/successive-halving later).
Stage-output cache key = `sha256(dataset/source hashes + fingerprint chain of
stages 1..N)` — shared pipeline prefixes across trials are computed once
(e.g., 24 combos → 1 parse, 2 chunk runs, 4 embed runs). `Trial` records:
trial_id, full `describe()` per stage (secrets already redacted by design),
fingerprints, metrics, cost (latency_ms, tokens, api_usd), cache_hits,
timestamps → JSONL + SQLite. Leaderboard computes **per-stage marginal
analysis** ("averaged over all else, the cross-encoder refiner adds +0.07 nDCG
for +180 ms/query") — quality AND cost attribution is the "deep insights"
deliverable.

**As built** — five refinements, each with its reasoning in a DR; read those
before changing any of it:

1. **Evaluators score data, they never run pipelines** (DR-0002):
   `evaluate(outcomes) -> MetricReport`. This *contradicts* ARCHITECTURE §3.9's
   original `evaluate(dataset, pipeline)`, which was amended — a
   pipeline-driving evaluator reimplements the run loop per implementation and
   depends backward on the composition root. The loop lives in `Tuner.run`.
2. **`Tuner.run()` is a Template Method** over one primitive,
   `iter_candidates(space)` — the `parse/iter_pages` shape again. Two-phase
   screening, cost, logging and error isolation live once in the base.
3. **No stage-output cache was built** (DR-0003 §2). The key formula above is
   already materialized by the blob parse cache (§7.2) and `CachingEmbedder`;
   a second implementation of it would drift. The tuner's contribution is
   **enumeration order** — `SearchSpace` varies the earliest stage slowest, so
   prefix-sharing trials inherit a warm cache. Measured on the committed
   benchmark: **12 combinations, 1 parse.** `STAGE_KINDS`' declaration order is
   load-bearing; sorting it alphabetically re-parses on nearly every trial.
4. **`cost` splits `index_ms` from `query_ms`** (DR-0003 §3). Total latency is
   cache-confounded inside a run — the first trial pays for the parse, so
   "cheaper" can mean "ran second". `query_ms` is the clean number and what the
   leaderboard ranks on. **`api_usd` is never guessed**: no price table ships,
   and an unpriced run has *no* `api_usd` key rather than `0.0`.
5. **`EvalSample.relevant_doc_ids`** exists beside `relevant_chunk_ids` because
   `Chunk.id` is `{doc_id}:{index}` — a chunk-level label denotes a *different
   passage* under a different chunker, silently. Doc-level labels are what make
   chunk size tunable at all (§6.4's own headline example), and what
   `benchmarks/baseline/` uses.

`PipelineBuilder` is the one new abstraction (spec → live `RagPipeline`),
needed because `ChunkIndex` is wired from live backends and `IndexRetriever`
refuses to be built by name alone. It is wiring, not a Strategy: the tuner
depends on `PipelineFactory = Callable[[dict], RagPipeline]`, never on the
class.

**Known gap:** enricher LLM token usage is uncaptured — `Enricher.enrich`
returns an `Iterator[Chunk]` with no usage channel, so reporting it means
changing a shipped stage's contract and earns its own DR. Per-enricher
*latency* is captured.

### 7.4 Secrets policy (applies to every adapter you write)

- Credential resolution pattern, exactly:
  `self.config.api_key or os.environ.get("<VENDOR>_API_KEY")` — explicit
  config wins (users with their own secret managers), env var is the default.
  **Use the vendor-standard env name** (`MISTRAL_API_KEY`), never a
  toolkit-prefixed one — least surprise.
- The library **NEVER** calls `load_dotenv()`, never writes secrets, never
  logs them. Populating the environment is the application's job.
- **Pipeline specs / YAML / trial logs must never contain secrets.** Config
  names *which* engine; the environment supplies *its* credentials. A future
  config loader may support `${ENV_VAR}` interpolation resolved at load time.
- Custom-engine authors must name credential fields with a redaction marker
  substring (`api_key`, `auth_token`, ...) to get automatic redaction —
  document this in the extension guide.
- Google uses Application Default Credentials (no key field at all —
  `GOOGLE_APPLICATION_CREDENTIALS` or ambient identity). Vendor-specific
  credential mechanics belong inside the vendor's Adapter.
- Repo hygiene: `.env` gitignored; committed `.env.example` with placeholder
  keys (`MISTRAL_API_KEY=`, `GOOGLE_APPLICATION_CREDENTIALS=`,
  `rag_blocks_TEST_PDF=`); CI secrets via GitHub Actions secrets. Fork PRs
  don't receive secrets — which is fine BY DESIGN because the default test
  suite is hermetic; only `-m integration` needs keys (run on main/nightly).

### 7.5 License *(done)*

Apache-2.0, fully in place: verbatim `LICENSE` text (never edit it), one-line
`NOTICE` (`rag-blocks — Copyright 2026 Mohamed Elamine Bentarzi`), and the
PEP 639 form `license = "Apache-2.0"` + `license-files = ["LICENSE"]` in
pyproject. Rationale: explicit patent grant, contribution licensing (§5),
ecosystem norm for RAG infra.

### 7.6 ChunkIndex, composition algebra & multi-representation retrieval *(implemented in v0.6 — DR-0001 v2, see `docs/decisions/`)*

All retrieval representations of a corpus are owned by one `ChunkIndex`:
`add(chunks)` writes every representation; `search(name, TEXT, k, filters)`
encodes the query with the same encoder that encoded the corpus — never
reimplement query encoding elsewhere. Constructor uses progressive disclosure:
`dense=embedder` auto-names; mappings only for multiple representations.
**Standing design rule (progressive disclosure):** the common case reads like
English; the rare case is possible; the rare case's ceremony never leaks into
the common case. Chunks NEVER carry vectors. The composition algebra:
pre-retrieval variation = composite retrievers
(`Fusion`/`Hybrid`/`MultiQuery`/`Hyde` — never new pipeline slots);
post-retrieval variation = the `refine` chain
(`Refiner.refine(query, candidates, k)`; the `reranker` kind is retired into
it); write-side = `enrich` chain + `sinks` fan-out (`ChunkSink` — the one
sanctioned `typing.Protocol`: a capability seam, not a stage contract; stage
contracts remain ABCs). Fusion always: dedup by `chunk.id`, filters fan out to
every sub-search, per-source rank attribution in `metadata["sources"]`.
`VectorStore` is named+typed multi-vector with `ensure_schema` create-or-validate,
`fetch(filters, limit)` (list values = membership), `update_vectors`. Classic
BM25 stays a mounted corpus-stats `LexicalIndex`; SPLADE-style sparse is a
`SparseEncoder` representation. `CachingEmbedder` is fingerprint-transparent with
separate passage/query namespaces. Bare LLM completion is a `Callable[[str], str]`
seam (`generator.complete`); do not invent a `completer` kind until a third
independent consumer demands it. Empty chains are the null objects
(`NoOpReranker`/`NoOpEnricher` are deleted — do not recreate them). Do not
re-litigate: no retriever write-side, no `chunk.vectors`, no `QueryTransform`
kind, no DAG framework, no capability negotiation. The architecture's acceptance
test: the tuner must index once and enumerate retrieval/refinement strategies
with ZERO tuner-motivated parameters on `ChunkIndex` — if one appears, stop and
write DR-0002.

Kinds (v2): `vector_store` (renamed from `store`), `embedder`, `sparse_encoder`,
`lexical_index`, `index` (ChunkIndex — aggregate, wired from instances, not
registry-built), `retriever`, `refiner` (replaces `reranker`), `enricher`,
`generator`, `blob_store`, plus ingestion kinds. `ChunkSink` is a Protocol, not
a kind.

---

## 8. Coding conventions

- Python >= 3.10, `from __future__ import annotations` everywhere. Full type
  hints. Line length 88 (ruff configured).
- **Core stays stdlib-only.** Contracts and configs are plain `@dataclass`
  (pydantic was considered and rejected for the hot path; wrapping at app
  edges is fine for users, not for us).
- Lazy vendor imports, exact idiom: import inside the method that needs it,
  wrap `ImportError`, raise a toolkit error naming the extra:
  `"... requires 'docling'. Install with: pip install 'rag-blocks[docling]'"`.
- Errors: single root `RagBlocksError`; raise narrow subclasses with context
  (`ParseError(msg, source_uri=..., page_number=...)`) — "PDF failed" is
  useless in a 10k-document batch. Fail fast at construction (unknown engine
  name explodes in `__init__`, not on page 500). Never swallow exceptions
  without normalizing them into toolkit errors with `from exc`.
- Docstrings explain **WHY and the decision/tradeoff**, not what the code
  restates. Pattern names are cited explicitly. Every module opens with a
  design-rationale docstring — match `contracts.py`/`docling_parser.py` style.
- Naming: stages are agent nouns (`Parser`, `Chunker`, `Embedder`), artifacts
  plain nouns (`Source`, `Page`, `Chunk`), routers/facades prefixed `Auto`,
  policies are str-Enums, test doubles prefixed `Fake`. `kind` is the stage
  slot, `name` the implementation.
- Built-ins register via module import side effect; wire new modules into the
  subsystem `__init__.py` and export in `__all__` (top-level `rag_blocks/
  __init__.py` for user-facing names).
- Bump the component `version` on ANY behavioral change (cache invalidation).
- Public API discipline: pre-1.0, adding is cheap, removing is a breaking
  event — keep surface small; when in doubt, keep it private (`_helper`).

---

## 9. Testing rules (non-negotiable)

- Framework: pytest; layout mirrors the package under `tests/`. Default run
  is **fast and hermetic** — zero vendor deps, zero network, zero keys
  (`addopts = "-m 'not integration'"`). Real-stack tests live in
  `tests/integration/`, marked `integration`, opt-in
  (`rag_blocks_TEST_PDF=... pytest -m integration`).
- **Tests ship WITH the feature, same PR.** A component without tests does
  not exist.
- Test OUR logic, not vendors': extract pure functions (`_plan_segments`,
  `_windows`) and test them directly; verify dispatch with `monkeypatch`;
  inject `FakeOcrEngine` (`tests/helpers.py`) through the same registry seam
  production uses. Never mock what you can fake through a designed seam.
- Every stage gets contract checks in `tests/contract_checks.py`; every
  implementation calls them. Key existing invariants to never break:
  `doc.markdown[span.start:span.end] == page.markdown`; secret redaction +
  key-rotation keeps fingerprint stable; the PDF dispatch matrix; the UTF-8
  multibyte-across-block-boundary regression in the plaintext parser.
- Style: table-driven `parametrize` for many-cases-one-behavior;
  `pytest.raises(..., match=...)` for error paths; `tmp_path` for files;
  fresh `Registry()` instances in registry tests (never assert exact contents
  of the global registry — other test modules register fakes into it).
- `scripts/mini_pytest.py` is a fallback runner for offline environments
  (emulates the pytest subset this suite uses). It is intentionally frozen:
  **extend the tests, never the shim**; if a new pytest feature is needed and
  the shim can't run it, the shim loses.
- CI (`.github/workflows/ci.yml`): ruff + mypy + pytest with a coverage gate
  (fail under 80%) on Python 3.10–3.13, plus a Windows lane. Keep it green;
  run lint/type/tests locally before proposing changes.

---

## 10. Definition of Done — any new component

1. Class with `kind`, `name`, `version`, optional nested `Config` dataclass;
   registered with `@registry.register`; wired into subsystem `__init__.py`.
2. Implements exactly the stage ABC primitive(s); depends only on
   `core.contracts` + its own stage's abstractions (Dependency Inversion —
   e.g., parsers depend on `OcrEngine`, never on Mistral).
3. Vendor deps: lazy import + pyproject extra + actionable ImportError.
4. Credential fields named for auto-redaction; env-var fallback per §7.4.
5. Pure function of (config, inputs); heavy resources cached on the instance.
6. Streaming discipline if data-producing; provenance fields populated.
7. Hermetic tests incl. the stage contract check; integration test only if a
   real vendor is involved (marked, env-gated).
8. Docstrings state the pattern and the why; README/ARCHITECTURE touched if
   user-visible.
9. `ruff check` clean, `mypy` clean, full suite green.

## 11. Roadmap

**Shipped (v0.2–v0.7, in this order):** chunking (`fixed`, `markdown-aware`)
+ `Chunk` char-offset fields + chunker contract checks → thin
`IndexingPipeline`/`QueryPipeline`/`RagPipeline` (dumb for-loops over
generators + tracing hooks; intelligence in components, wiring in config) →
embedding (`bge-m3` via sentence-transformers, `embed_query` separate from
`embed_texts` — instruction-prefix asymmetry) + storage (`memory` store for
tests/tuning, `qdrant`, `LocalBlobStore`/`MinioBlobStore` per §7.2) →
retrieval + refinement → generation (context packing, token budget, citation
markers resolved through chunk→page provenance) → the DR-0001 v2 restructure
(§7.6): `ChunkIndex` aggregate + multi-vector `vector_store`; retrieval
collapsed into the composition axis (`index`/`hybrid`/`fusion`/`multi-query`/
`hyde`) and reranking dissolved into the `refiner` chain; `CachingEmbedder`;
`sparse_encoder` interface.

**Shipped (v0.8):** evaluation & tuning per §7.3 — `Evaluator` kind (`ir`,
`answer-match`, `ragas`), `SearchSpace`/`choice`, `Tuner` (`grid`, `random`),
`PipelineBuilder`, `Trial`/`TrialLog` (JSONL + SQLite), `CostCollector`,
`Leaderboard` with marginal analysis, and the committed regression baseline in
`benchmarks/baseline/`.

**Known gaps within the shipped surface:** no concrete `SparseEncoder`
implementation yet (the contract exists; encoders are fast-follow), no
`chonkie` chunker adapter yet (still wanted per §7.1), and no enricher token
usage (§7.3).

**Next: v0.9, the streaming generation seam** (`iter_complete` as an optional
`Generator` capability). The versioned unlock plan (streaming, late chunking,
agentic composites, multi-vector/MaxSim, ColPali projection) lives in
`tump_docs/RAG-SOTA-ROADMAP.md` and its companions; those documents graduate
into `docs/` as each milestone starts. Each milestone ships **at least two
interchangeable implementations per stage** — swapping is the proof the library
works — and, since v0.8 exists, **a benchmark rerun proving its marginal win**
(roadmap rule 2). `benchmarks/baseline/` is the artifact to rerun; the honest
number, favorable or not, goes in the release notes.

## 12. When uncertain, decide like this

Does it belong in `core`? Only if every stage needs it AND it's stdlib-only —
otherwise it's a component. New dependency? → optional extra + lazy import,
core stays clean. Two components share logic? → extract a helper module or a
value object; do NOT create an inheritance link between stages. Tempted to add
a parameter to a contract dataclass? → prefer `metadata` first; promote to a
field only when multiple stages rely on it (as done for char offsets).
Behavior change? → bump `version`, update tests, note it. Can't test it
cleanly? → the design is wrong; add a seam or extract a pure function.
Ambiguity between this file, ARCHITECTURE.md, and code? → flag it to the
owner; don't guess silently. And keep the owner's bar in mind: he is learning
from this codebase — every shortcut you take teaches him the wrong lesson.
