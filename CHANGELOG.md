# Changelog

All notable changes to rag-blocks are documented here. Pre-1.0, breaking
changes are expected between minor versions.

## [0.6.0] — DR-0001 v2: ChunkIndex, composition algebra, multi-representation

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
