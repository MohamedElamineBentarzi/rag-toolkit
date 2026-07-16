# Changelog

All notable changes to rag-blocks are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/). Pre-1.0, breaking changes are
expected between minor versions.

## [Unreleased]

## [0.7.0] — 2026-07-16 — OSS readiness: rename, packaging, hardening

First public release. The library is renamed and the repository is brought up to
publishable standard; several correctness and hardening fixes land alongside.

### Changed (breaking)
- **Renamed `rag-toolkit` → `rag-blocks`** (distribution and import package
  `rag_blocks`) to resolve a PyPI name collision. The root exception is now
  `RagBlocksError`; the entry-point group is `rag_blocks.components`.
- **`BM25Index.add` now upserts** existing ids (recomputes term counts/length)
  instead of skipping them, matching `VectorStore.upsert`. Re-indexing a
  persisted lexical namespace after a chunker/enricher change now stays
  consistent with the vector side. `BM25Index.version` → `0.2.0`.

### Added
- `py.typed` marker (PEP 561) so consumers see the library's type hints.
- Export-integrity test covering every package `__all__`.
- `storage/filters.py`: single shared definition of metadata-filter semantics
  (memory store, BM25, and Qdrant's native translation all reference it).
- Community scaffolding: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR
  templates, `examples/.env.example`, CITATION.cff; CI mypy enforcement, a
  Windows lane, Python 3.13, a coverage gate, and release (Trusted Publishing)
  + nightly integration workflows.

### Changed
- `__version__` is now read from the installed distribution metadata
  (`importlib.metadata`) instead of being hardcoded, so it can never drift from
  `pyproject.toml`. Running from a source tree with no install reports
  `0.0.0+unknown`.

### Fixed
- `EnrichmentError` was listed in `__all__` but never imported, breaking
  `from rag_blocks import *`.
- Secret redaction now walks nested dicts (e.g. an `authorization` header inside
  a config field), not just top-level field names, and `authorization` joins the
  redacted-name markers.
- Entry-point plugin load failures are now logged (warning) instead of silently
  dropped.
- `NeighborExpander` over-fetches so a synthetic chunk can't silently displace a
  real neighbor.
- `CachingEmbedder` raises if an inner embedder returns a mismatched vector count
  instead of silently misaligning results.
- `pack_context` counts the block joiners against its character budget.
- Packaging metadata: PEP 639 license expression, full project URLs, correct
  repository homepage, Python 3.13 classifier.

## [0.6.0] — 2026-07-15 — DR-0001 v2: ChunkIndex, composition algebra, multi-representation

The retrieval architecture is unified around one aggregate and two uniform
chains (see `DR-0001-v2`). **This release is breaking** (pre-1.0).

### Added
- **`ChunkIndex`** (`kind="index"`): the aggregate owning every retrieval
  representation of a corpus on both paths. `add(chunks)` writes them all;
  `search(representation, text, k)` encodes the query with the same encoder that
  encoded the corpus. Progressive-disclosure constructor
  (`dense=`/`sparse=`/`lexical=`, auto-named; mappings for the power case).
- **Named, typed, multi-vector `VectorStore`**: `ensure_schema(specs)`
  (create-or-validate), `fetch(filters, limit)` (point retrieval without a query
  vector; list filter values mean membership), `update_vectors` (partial
  refresh). New contracts `SparseVector`, `VectorValue`, `VectorSpec`.
- **`SparseEncoder`** kind + `Embedder.distance` property.
- **Composition axis**: `IndexRetriever`, `FusionRetriever` (fuse any
  retrievers — federation, cross-paradigm), `MultiQueryRetriever`,
  `HydeRetriever`. Fusion mechanics extracted to `retrieval/fusion.py`.
- **Refiner chain** (`kind="refiner"`): `refine(query, candidates, k)` — one
  uniform post-retrieval stage shape. Ships `cross-encoder`, `keyword`,
  `neighbor-expander` (char-offset overlap-safe small-to-big expansion),
  `score-threshold`.
- **`ChunkSink`** protocol + write-path `sinks` fan-out (a `ChunkIndex`, a
  `LexicalIndex`, or a GraphRAG index are all sinks).
- **`CachingEmbedder`**: fingerprint-transparent memoizing decorator with
  separate passage/query cache namespaces.
- **`AnthropicGenerator.complete(prompt) -> str`**: the bare-completion seam for
  query shaping and contextual enrichment.
- **`DocumentCatalog`** + `docs/{doc_id}.json` manifest: resolve a citation's
  `doc_id` to its `source_uri` and a download link in one hop
  (`RagPipeline.source_uri(doc_id)` / `download_url(doc_id)`), no hashing or
  parser fingerprint needed. Requires a `blob_store`.
- **`BlobStore.url(key)`**: optional download-link capability — `LocalBlobStore`
  returns a `file://` URI, `MinioBlobStore` a presigned S3 GET URL.
- **`QdrantVectorStore(recreate_on_mismatch=True)`**: dev/test opt-in to drop and
  rebuild a collection whose schema no longer matches (default off); clearer,
  actionable schema-mismatch errors that name what the collection actually holds.

### Changed (breaking)
- `VectorStore.kind` renamed `"store"` → `"vector_store"`. Single-vector
  `upsert(chunks, vectors)` / `search(vector, k)` are now named+multi-vector:
  `upsert(chunks, {name: vectors})`, `search(name, vector, k)`. Stale configs
  raise `ComponentNotFoundError`.
- `RagPipeline` is now a composition root over a shared `ChunkIndex`:
  `RagPipeline(chunk_index=..., retriever=..., refine=[...], enrich=[...],
  extra_sinks=[...])`. Old `embedder=`/`store=`/`reranker=`/`embedding_cache=`
  removed; use `RagPipeline.dense(embedder=, store=)` for the 80% dense case.
  A retriever wired to a different index than `chunk_index` explodes at
  construction (P6 guard).
- `IndexingPipeline(enrich=[...], sinks=[...], batch_size=...)` — enrichers are a
  chain, writes fan out to sinks.
- `QueryPipeline(retriever, refine=[...], fetch_k=...)` — the reranker slot is
  replaced by the refine chain.
- `Enricher` synthetic-chunk identity rule (§8.2) is now enforced by the
  contract check: added chunks need parent-derived ids, `metadata["synthetic"]`,
  and the parent index.
- `doc_id` is now the **full** sha256 content hash (was `content_hash[:16]`),
  eliminating truncation-collision risk; it therefore equals the raw-blob
  address (`raw/{doc_id}/original{ext}`). Every `doc_id`/`chunk.id`/blob key
  changes ⇒ existing indexes and blob caches must be rebuilt.

### Removed
- `DenseRetriever`, `Bm25Retriever` (collapsed into `IndexRetriever`).
- The `reranker` kind and `reranking/` package (`BgeReranker`, `KeywordReranker`,
  `NoOpReranker`) — retired into `refiner`; `BgeReranker` ported to
  `refinement.CrossEncoderReranker`.
- `NoOpEnricher` (the empty `enrich` chain is the null object).
- `RagPipeline._flush/_embed/_EmbeddingCache` (superseded by `CachingEmbedder`).

## [0.5.0] — 2026-07-14 — persistence, caching & generation

- Persistence & caching: BM25 index persist/load through a `BlobStore`, an
  embedding cache, and parse-cache read-through.
- Generation stage: `Answer`, extractive generator, and `AnthropicGenerator`
  with context packing, token budget, and citation markers resolved through
  chunk→page provenance.

## [0.4.0] — 2026-07-13 — retrieval & reranking

- Retrieval: `dense` and `bm25` retrievers over `Query`/`LexicalIndex`, and a
  `HybridRetriever` (RRF fusion Composite).
- `QueryPipeline` with a reranker seam; keyword and BGE cross-encoder rerankers.

## [0.3.0] — 2026-07-13 — embedding & storage

- Embedding stage: hashing (dependency-free) and sentence-transformers embedders.
- Storage: in-memory and Qdrant vector stores, `ScoredChunk`, and
  `LocalBlobStore`/`MinioBlobStore` blob stores.

## [0.2.0] — 2026-07-13 — chunking & enrichment

- Chunking: `fixed` and `markdown-aware` chunkers with char-offset provenance.
- Enrichment stage wired into the pipelines.
- Thin `IndexingPipeline` (parse → chunk, with an optional truth store).

## [0.1.0] — 2026-07-12 — core & ingestion

- Core: contracts, component model, registry, errors — zero third-party deps.
- Streaming ingestion: any file → clean markdown with page provenance, per-page
  OCR routing (Mistral, Google Document AI, or a custom engine).
