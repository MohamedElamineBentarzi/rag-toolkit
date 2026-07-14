# rag-toolkit — User Guide

Composable building blocks for production RAG pipelines. Every stage (parse,
chunk, enrich, embed, store, retrieve, rerank, generate) is a swappable
component behind a stable contract; every pipeline is wired from those
components; and the core runs on the Python standard library alone — heavy
vendor SDKs are optional extras you install only for the components you use.

This guide covers installation, the core concepts, every stage with examples,
the three pipelines, end-to-end recipes, and how to extend the library with your
own components.

> New here? Jump to [Quick start](#2-quick-start), then [Recipes](#8-recipes).
> Building a component? Go straight to [Extending the library](#9-extending-the-library).

---

## Table of contents

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Core concepts](#3-core-concepts)
4. [The data contracts](#4-the-data-contracts)
5. [The component model, registry & fingerprints](#5-the-component-model-registry--fingerprints)
6. [The stages (with built-ins)](#6-the-stages-with-built-ins)
7. [The pipelines](#7-the-pipelines)
8. [Recipes](#8-recipes)
9. [Extending the library](#9-extending-the-library)
10. [Configuration, secrets & environment](#10-configuration-secrets--environment)
11. [Errors](#11-errors)
12. [Testing your components](#12-testing-your-components)
13. [Cheat sheet](#13-cheat-sheet)

---

## 1. Install

```bash
pip install rag-toolkit                       # core only — stdlib, zero deps
```

The core (contracts, registry, pipelines, plaintext parsing, the hashing
embedder, the in-memory + BM25 stores, the extractive generator) works with **no
third-party dependencies**. Install extras only for the components you actually
route to:

| Extra | Enables | Components |
|---|---|---|
| `docling` | PDF/DOCX/PPTX/XLSX/HTML/image parsing | `DoclingParser` |
| `mistral` | Mistral cloud OCR | `MistralOcrEngine` |
| `google` | Google Document AI OCR | `GoogleDocAiOcrEngine` |
| `sentence-transformers` | real embeddings + cross-encoder rerank | `SentenceTransformerEmbedder`, `BgeReranker` |
| `qdrant` | Qdrant vector store | `QdrantVectorStore` |
| `minio` | S3-compatible blob storage | `MinioBlobStore` |
| `anthropic` | Claude generation + contextual enrichment | `AnthropicGenerator`, `ContextualEnricher` |

```bash
pip install "rag-toolkit[docling]"                    # local PDF parsing
pip install "rag-toolkit[sentence-transformers,qdrant,anthropic]"   # a prod stack
pip install "rag-toolkit[all]"                        # everything
```

If you call a component whose extra isn't installed, you get an actionable error
naming the exact `pip install` to run — nothing fails at import time.

---

## 2. Quick start

The whole loop in two calls, using the zero-dependency defaults (hashing
embedder, in-memory store, extractive generator) — no extras, no API key:

```python
from rag_toolkit import RagPipeline, Source

rag = RagPipeline()
rag.index(Source.from_path("report.pdf"))          # parse → chunk → embed → store
answer = rag.ask("What was Q3 revenue?", k=5)      # retrieve → rerank → generate

print(answer.text)
for c in answer.citations:                          # each resolves to doc + pages
    print(f"  [{c.marker}] {c.doc_id} p{c.page_start}–{c.page_end}")
```

> `Source.from_path("report.pdf")` needs the `[docling]` extra to parse a PDF.
> With core only, feed it text/markdown: `Source.from_path("notes.md")` or
> `Source.from_bytes(b"# Title\n...", name="notes.md")`.

Swap in production components without changing the wiring — see
[Recipe: a production stack](#82-a-production-stack).

---

## 3. Core concepts

**Contracts, not coupling.** Stages never import each other. They agree only on
the typed dataclasses in `rag_toolkit.core.contracts`
(`Source → Page → Document → Chunk → ScoredChunk → Answer`). A chunker doesn't
know what a parser is. This is what makes every stage swappable.

**Everything is a `Component`.** Every stage implementation subclasses
`Component`: it has a `kind` (the slot it fills), a `name` (the implementation),
a `version`, and an optional nested `Config` dataclass. Components are
registered under `(kind, name)` and built by name from the registry.

**Streaming-first.** Data-producing primitives are generators (`iter_pages`,
`index`), so memory stays O(one batch), never O(corpus) — a 2,000-page PDF
parses without loading all of it.

**Batteries optional.** The core is stdlib-only. Vendor SDKs are lazy-imported
inside the method that uses them, behind pip extras.

**Provenance end to end.** Every chunk can answer "which pages of which file",
so generated answers carry citations that resolve back to exact pages.

---

## 4. The data contracts

All in `rag_toolkit` (re-exported from `rag_toolkit.core.contracts`).

| Type | Produced by | Key fields |
|---|---|---|
| `Source` | you | `uri`, `data` (bytes, optional); `open()`, `head(n)`, `content_hash()` |
| `Page` | parser | `number` (1-based), `markdown`, `ocr_applied`, `metadata` |
| `Document` | `parser.parse()` | `id`, `markdown`, `pages: [PageSpan]`, `source_uri`; `pages_for_span(a,b)` |
| `Chunk` | chunker | `id`, `doc_id`, `text`, `index`, `char_start/end`, `page_start/end`, `metadata` |
| `Query` | you | `text`, `filters: dict \| None`, `metadata` |
| `ScoredChunk` | retriever/store | `chunk`, `score`, `retriever_name`, `metadata` |
| `Citation` | generator | `marker`, `chunk_id`, `doc_id`, `page_start/end` |
| `Answer` | generator | `text`, `citations: [Citation]`, `usage: dict`, `metadata` |

Constructing a `Source`:

```python
from rag_toolkit import Source

Source.from_path("report.pdf")                        # a file on disk (lazy)
Source.from_bytes(b"# Notes\n...", name="notes.md")   # in-memory; name aids detection
```

`Source` is a *lazy pointer* — it never eagerly reads a big file. `content_hash()`
is a streaming sha256 used as a cache/dedup key.

Provenance invariants worth knowing:
- `document.markdown[chunk.char_start:chunk.char_end] == chunk.text` for a
  freshly-chunked chunk (before enrichment may augment the text).
- `chunk.index` is contiguous 0-based with no holes (neighbour expansion relies
  on `index ± 1`).
- `page_start`/`page_end` are always filled for a doc-derived chunk; `None` is
  reserved for synthetic chunks (enricher summaries).

---

## 5. The component model, registry & fingerprints

Every component shares the same plumbing (from `Component`):

```python
comp.kind            # "embedder", "parser", ...
comp.name            # "hashing", "docling", ...
comp.version         # bump when behavior changes → caches invalidate
comp.config          # the resolved nested Config dataclass (or None)
comp.describe()      # loggable, secret-free dict of (kind, name, version, config)
comp.fingerprint()   # sha256(describe())[:16] — the cache key / trial identity
```

**Building by name (the Factory).** Pipelines are data: a component is a
`(kind, name)` string plus config overrides.

```python
from rag_toolkit import registry

emb = registry.create("embedder", "hashing", dimensions=512)
chunker = registry.create("chunker", "fixed", chunk_chars=800, overlap_chars=100)

registry.available("reranker")      # ['bge-reranker', 'keyword', 'noop']
registry.available()                # every (kind:name) registered
```

Equivalently, import and construct directly:

```python
from rag_toolkit import HashingEmbedder
emb = HashingEmbedder(dimensions=512)
```

**Config.** Each component may declare a nested `@dataclass class Config`. You
can pass a ready `Config`, keyword overrides, or both (overrides win); unknown
keys fail fast with `ConfigError`.

```python
from rag_toolkit import FixedChunker
FixedChunker(chunk_chars=800)                          # keyword override
FixedChunker(FixedChunker.Config(chunk_chars=800))     # explicit config object
```

**Fingerprints.** `fingerprint()` hashes the *redacted* `describe()`. Two
consequences you rely on: rotating an API key never changes the fingerprint (so
caches survive key rotation and secrets never leak into logs), and **if you
change a component's behavior you must bump its `version`** so downstream caches
invalidate.

---

## 6. The stages (with built-ins)

Each stage is a `Component` subclass with one contract method. Import the base
from `rag_toolkit` (e.g. `from rag_toolkit import Parser`) or its subsystem.

| Stage (`kind`) | Contract method | Built-ins | Extra |
|---|---|---|---|
| Parser (`parser`) | `iter_pages(source) → Iterator[Page]` | `auto`, `plaintext`, `docling` | `docling` for docling |
| OCR engine (`ocr`) | `recognize(PageImage) → OcrResult` | `mistral`, `google-docai` | `mistral` / `google` |
| Chunker (`chunker`) | `iter_spans(document) → Iterator[(int,int)]` | `fixed`, `markdown-aware` | — |
| Enricher (`enricher`) | `enrich(chunks, document) → Iterator[Chunk]` | `noop`, `heading`, `contextual` | `anthropic` for contextual |
| Embedder (`embedder`) | `embed_texts`, `embed_query`, `dimensions` | `hashing`, `sentence-transformers` | `sentence-transformers` |
| Vector store (`store`) | `upsert(chunks, vecs)`, `search(vec, k, filters)` | `memory`, `qdrant` | `qdrant` |
| Lexical index (`lexical_index`) | `add(chunks)`, `search(text, k, filters)` | `bm25` | — |
| Retriever (`retriever`) | `retrieve(query, k) → [ScoredChunk]` | `dense`, `bm25`, `hybrid` | — |
| Reranker (`reranker`) | `rerank(query, cands, top_k) → [ScoredChunk]` | `noop`, `keyword`, `bge-reranker` | `sentence-transformers` for bge |
| Generator (`generator`) | `generate(query, context) → Answer` | `extractive`, `anthropic` | `anthropic` |
| Blob store (`blob_store`) | `put`, `get`, `exists` | `local`, `minio` | `minio` |

### 6.1 Parser — any file → markdown pages

```python
from rag_toolkit import AutoParser, Source

parser = AutoParser()                                  # routes by detected format
doc = parser.parse(Source.from_path("report.pdf"))     # → Document
print(doc.markdown[:200], len(doc.pages), "pages")

for page in parser.iter_pages(Source.from_path("huge.pdf")):  # streaming
    process(page.markdown)                             # O(page batch) memory
```

`AutoParser` detects the format (magic bytes first) and delegates: PDF/office →
`docling`, txt/md → `plaintext`. Override routes or per-parser config:

```python
AutoParser(parser_configs={"docling": {"page_batch_size": 4, "ocr_policy": "force"}})
AutoParser(routes={"pdf": "my-parser"})                # send PDFs to your parser
```

`DoclingParser` details (PDF OCR routing) — see
[Recipe: scanned PDFs](#84-scanned-pdfs-with-cloud-ocr).

### 6.2 Chunker — document → chunks

```python
from rag_toolkit import FixedChunker, MarkdownChunker

chunks = list(FixedChunker(chunk_chars=1600, overlap_chars=200).chunk(doc))
chunks = list(MarkdownChunker().chunk(doc))            # cut at headings
```

`fixed` slices fixed-size windows with overlap, preferring paragraph/line
boundaries. `markdown-aware` cuts at ATX headings — each chunk is a coherent
section. Both fill `char_start/end` and `page_start/end`.

### 6.3 Enricher — augment chunks with context (optional)

```python
from rag_toolkit import NoOpEnricher, HeadingEnricher

# heading: prepend each chunk's section heading so it embeds with its context
enriched = list(HeadingEnricher().enrich(iter(chunks), doc))
```

`noop` (default) passes through; `heading` prepends the nearest markdown heading
(deterministic contextual retrieval); `contextual` uses Claude to write a
situating sentence per chunk (`[anthropic]`). Enrichers preserve provenance.

### 6.4 Embedder — text → vectors

```python
from rag_toolkit import HashingEmbedder

emb = HashingEmbedder(dimensions=256)                  # zero-dep, deterministic
vecs = emb.embed_texts(["passage one", "passage two"]) # list[list[float]]
qvec = emb.embed_query("a question")                   # query encoded separately
emb.dimensions                                         # 256
```

`embed_query` is **separate** from `embed_texts` on purpose — instruction-tuned
models prefix queries differently from passages. For real embeddings:

```python
from rag_toolkit import SentenceTransformerEmbedder
emb = SentenceTransformerEmbedder(model="BAAI/bge-m3")            # [sentence-transformers]
emb = SentenceTransformerEmbedder(model="intfloat/e5-large-v2",
                                  query_instruction="query: ")     # prefix queries only
```

### 6.5 Vector store & lexical index — the searchable indexes

```python
from rag_toolkit import MemoryVectorStore, BM25Index

store = MemoryVectorStore()                            # dense (cosine)
store.upsert(chunks, emb.embed_texts([c.text for c in chunks]))
hits = store.search(emb.embed_query("q"), k=5, filters={"doc_id": "abc"})

index = BM25Index()                                    # sparse (lexical)
index.add(chunks)
hits = index.search("exact terms", k=5)
```

Both are idempotent by `chunk.id` and return `ScoredChunk`s with the full chunk
(text + provenance) inline, so query time never touches the blob store. Swap in
`QdrantVectorStore(url="http://localhost:6333")` for a real server (`[qdrant]`).

### 6.6 Retriever — query → ranked chunks

Retrievers are composed from *populated* backends (a store/index that already
holds the corpus):

```python
from rag_toolkit import DenseRetriever, Bm25Retriever, HybridRetriever, Query

dense = DenseRetriever(embedder=emb, store=store)
bm25 = Bm25Retriever(index=index)
hybrid = HybridRetriever(retrievers=[dense, bm25], k_rrf=60)   # RRF fusion

hits = hybrid.retrieve(Query(text="what was revenue?"), k=10)
```

`hybrid` fuses sub-retrievers by **rank** (Reciprocal Rank Fusion), so it blends
dense and lexical results whose raw scores are on incompatible scales. Optional
per-retriever `weights=[...]` gives weighted fusion.

### 6.7 Reranker — precise second pass (optional)

```python
from rag_toolkit import NoOpReranker, KeywordReranker

reranked = KeywordReranker().rerank(query, candidates, top_k=8)
```

`noop` (default) preserves order; `keyword` reorders by query-term overlap;
`bge-reranker` is a cross-encoder (`[sentence-transformers]`) that reads
query+candidate together for the best accuracy.

### 6.8 Generator — context → cited answer

```python
from rag_toolkit import ExtractiveGenerator, AnthropicGenerator, Query

answer = ExtractiveGenerator().generate(Query(text="q"), scored_chunks)  # zero-dep
answer = AnthropicGenerator(model="claude-opus-4-8").generate(query, scored_chunks)
print(answer.text, answer.citations, answer.usage)
```

The base class numbers the context `[1]`, `[2]`, …, and resolves whichever
markers the answer uses back to source provenance — so `[2]` in the text maps to
exact pages of an exact document.

### 6.9 Blob store — the durable truth store

```python
from rag_toolkit import LocalBlobStore, Source

store = LocalBlobStore(root="./.rag_cache/blobs")      # or MinioBlobStore(...)
src = Source.from_path("report.pdf")
key = f"raw/{src.content_hash()}/original.pdf"         # content-addressed ⇒ dedup free
if not store.exists(key):
    store.put(key, src.open().read())
data = store.get(key)
```

Keys are opaque strings — the content-addressing convention lives in your code,
so `LocalBlobStore` and `MinioBlobStore` are interchangeable.

---

## 7. The pipelines

Pipelines are thin — a for-loop over generators plus a tracing hook. All
intelligence is in the components.

### 7.1 IndexingPipeline — `Source → Chunk` stream

```python
from rag_toolkit import IndexingPipeline, Source

pipe = IndexingPipeline()                              # AutoParser + FixedChunker + NoOpEnricher
for chunk in pipe.index(Source.from_path("report.pdf")):
    embed_and_store(chunk)
```

Optional truth-store capture (raw bytes + parse cache, content-addressed,
deduped) and a tracing hook:

```python
from rag_toolkit import IndexingPipeline, LocalBlobStore, HeadingEnricher, MarkdownChunker

pipe = IndexingPipeline(
    chunker=MarkdownChunker(),
    enricher=HeadingEnricher(),
    blob_store=LocalBlobStore(root="./.rag_cache/blobs"),
    trace=print,                                       # receives TraceEvent per stage
)
chunks = list(pipe.index([Source.from_path(p) for p in ("a.pdf", "b.pdf")]))
```

### 7.2 QueryPipeline — `Query → ScoredChunks`

```python
from rag_toolkit import QueryPipeline

qp = QueryPipeline(retriever=hybrid, reranker=KeywordReranker(), fetch_k=50)
hits = qp.query("what was Q3 revenue?", k=8)           # retrieve 50 → rerank to 8
```

### 7.3 RagPipeline — the facade (`index` + `ask`)

Owns the whole loop. Defaults are the zero-dependency stack, so it runs with no
extras; pass components to go to production without changing the wiring.

```python
from rag_toolkit import RagPipeline, Source

rag = RagPipeline()                                    # hashing + memory + extractive
rag.index(Source.from_path("report.pdf"))
answer = rag.ask("What was Q3 revenue?", k=5)
```

`RagPipeline(...)` accepts `embedder`, `store`, `generator`, `parser`, `chunker`,
`enricher`, `reranker`, `blob_store`, `fetch_k`, `batch_size`, `trace`. It builds
a `DenseRetriever` over your embedder+store internally; for hybrid/custom
retrieval, compose `QueryPipeline` + a `Generator` yourself.

### 7.4 Persistence & caching

**What survives a process restart** depends on the store you chose:

| Store | Durable across restart? |
|---|---|
| `QdrantVectorStore(url=...)` / `(path=...)` | ✅ vectors **and** chunk text/provenance live server-/disk-side |
| `QdrantVectorStore(location=":memory:")`, `MemoryVectorStore` | ❌ in-process, ephemeral |
| `BM25Index` (no store) | ❌ in-memory — **persist it explicitly** (below) |
| `LocalBlobStore` / `MinioBlobStore` | ✅ the durable truth (raw bytes + parse cache) |

With a **server-backed Qdrant**, re-instantiating a `RagPipeline` with the **same
store config and the same embedder** lets you query immediately — no re-index:

```python
rag = RagPipeline(embedder=SentenceTransformerEmbedder(model="BAAI/bge-m3"),
                  store=QdrantVectorStore(url="http://localhost:6333", collection="docs"))
rag.ask("...")           # works after a restart — Qdrant already holds everything
```

**Persist a BM25 index** by injecting a blob store (the index serializes itself;
*where* the bytes go is the store's job — memory / local / MinIO, all via the one
`BlobStore` abstraction):

```python
from rag_toolkit import BM25Index, MinioBlobStore

index = BM25Index(store=MinioBlobStore(bucket="rag"), namespace="my-corpus")
index.load()             # rehydrate if a saved index exists (no-op otherwise)
index.add(chunks)        # idempotent by chunk.id — re-adds are skipped
index.persist()          # flush to the store
```

**Skip re-embedding** unchanged chunks with an opt-in embedding cache (keyed by
`text × embedder.fingerprint()`, so swapping the model is a clean miss):

```python
rag = RagPipeline(embedder=SentenceTransformerEmbedder(model="BAAI/bge-m3"),
                  embedding_cache=MinioBlobStore(bucket="rag"))
rag.index(sources)       # first run embeds; a later run reuses cached vectors
```

**Skip re-parsing** happens automatically once a `blob_store` is set: the pipeline
writes `parsed/{sha256}/{parser_fingerprint}.md` on first parse and loads it back
instead of re-parsing next time (the parser never touches storage — the cache
lives in the pipeline). Watch `cache_hit` in the `parse`/`store_*` trace events.

> Design rule: compute components stay pure; persistence/caching is either
> orchestrated by the pipeline (which owns the stores) or delegated to an injected
> `BlobStore` — never baked into a component as a concrete vendor.

---

## 8. Recipes

### 8.1 Zero-dependency local demo

```python
from rag_toolkit import RagPipeline, MarkdownChunker, Source

rag = RagPipeline(chunker=MarkdownChunker())
rag.index(Source.from_bytes(
    b"# Q3\nRevenue was $4.2M, up 18%.\n\n# Team\nWe hired 12 engineers.\n",
    name="report.md",
))
print(rag.ask("What was revenue?", k=1).text)
```

### 8.2 A production stack

```python
from rag_toolkit import (
    RagPipeline, SentenceTransformerEmbedder, QdrantVectorStore,
    AnthropicGenerator, HeadingEnricher, MarkdownChunker,
)

rag = RagPipeline(
    embedder=SentenceTransformerEmbedder(model="BAAI/bge-m3"),
    store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
    generator=AnthropicGenerator(model="claude-opus-4-8"),
    enricher=HeadingEnricher(),
    chunker=MarkdownChunker(),
)
rag.index([Source.from_path(p) for p in my_files])     # needs [docling] for PDFs
print(rag.ask("Summarize the risk factors.", k=8).text)
```
Requires `pip install "rag-toolkit[docling,sentence-transformers,qdrant,anthropic]"`
and `ANTHROPIC_API_KEY` in the environment.

### 8.3 Hybrid retrieval + reranking, wired by hand

When you want dense + BM25 fusion (RagPipeline covers the dense case), compose
the pieces:

```python
from rag_toolkit import (
    IndexingPipeline, QueryPipeline, HashingEmbedder, MemoryVectorStore, BM25Index,
    DenseRetriever, Bm25Retriever, HybridRetriever, KeywordReranker,
    AnthropicGenerator, Query, Source,
)

emb, store, index = HashingEmbedder(), MemoryVectorStore(), BM25Index()

# Index once, into both a vector store and a lexical index.
for chunk in IndexingPipeline().index(Source.from_path("report.md")):
    store.upsert([chunk], emb.embed_texts([chunk.text]))
    index.add([chunk])

retriever = HybridRetriever(retrievers=[
    DenseRetriever(embedder=emb, store=store),
    Bm25Retriever(index=index),
])
qp = QueryPipeline(retriever=retriever, reranker=KeywordReranker(), fetch_k=50)

query = Query(text="what was revenue?")
context = qp.query(query, k=8)
answer = AnthropicGenerator().generate(query, context)   # or ExtractiveGenerator()
```

### 8.4 Scanned PDFs with cloud OCR

`DoclingParser` routes OCR per page (`AUTO` probes each page's text layer and
only OCRs the scanned ones; `FORCE` OCRs everything; `NEVER` uses the text layer
only), and can send scanned pages to any OCR engine:

```python
import rag_toolkit as rk

doc = rk.ingest("scan.pdf", ocr_engine="mistral", ocr_policy=rk.OcrPolicy.FORCE)
# or via a pipeline:
from rag_toolkit import AutoParser
parser = AutoParser(parser_configs={"docling": {
    "ocr_engine": "mistral", "ocr_policy": "auto", "page_batch_size": 4,
}})
```
`rk.ingest(path, **docling_overrides)` is a one-call facade returning a
`Document`. Needs `[docling]` and `[mistral]`; set `MISTRAL_API_KEY`.

### 8.5 Persist raw files + the parse cache

Hand the IndexingPipeline a blob store and it captures, per source:
`raw/{sha256}/original{ext}` (immutable source of truth) and
`parsed/{sha256}/{parser_fingerprint}.md` (+ `.meta.json`). Re-indexing the same
bytes is a deduped no-op.

```python
from rag_toolkit import RagPipeline, LocalBlobStore
rag = RagPipeline(blob_store=LocalBlobStore(root="./.rag_cache/blobs"))
rag.index(Source.from_path("report.pdf"))              # bytes + parse cached
```

### 8.6 Observing a run (tracing)

Every pipeline accepts a `trace` hook called with a `TraceEvent(stage,
source_uri, duration_ms, detail)` at each stage boundary:

```python
from rag_toolkit import RagPipeline, TraceEvent, Source

events: list[TraceEvent] = []
rag = RagPipeline(trace=events.append)
rag.index(Source.from_bytes(b"# T\nbody\n", name="t.md"))
rag.ask("body?", k=1)
for e in events:
    print(f"{e.stage:12} {e.duration_ms:6.1f}ms {e.detail}")
# parse / chunk / retrieve / rerank ...
```

---

## 9. Extending the library

A new capability is a new registered class — **zero edits to existing files**.
The recipe is always the same:

1. Subclass the stage's base class.
2. Set `name` (unique within the kind) and `version`.
3. Optionally declare a nested `@dataclass class Config`.
4. Implement the one contract method.
5. Decorate with `@registry.register`.

Then build it by name (`registry.create(kind, name, ...)`) or import it — and it
drops into any pipeline in that slot.

### 9.1 A custom embedder

```python
from dataclasses import dataclass
from typing import Sequence
from rag_toolkit import registry, Embedder

@registry.register
class MyEmbedder(Embedder):
    name = "my-embedder"
    version = "0.1.0"

    @dataclass
    class Config:
        dimensions: int = 384

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]          # your model here

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)                         # prefix queries if needed

    def _vector(self, text: str) -> list[float]:
        ...

# Use it anywhere an embedder goes:
rag = RagPipeline(embedder=MyEmbedder(dimensions=384))
emb = registry.create("embedder", "my-embedder", dimensions=384)
```

### 9.2 A custom reranker

```python
from rag_toolkit import registry, Reranker, Query, ScoredChunk

@registry.register
class LengthReranker(Reranker):
    """Toy: prefer longer passages."""
    name = "length"
    version = "0.1.0"

    def rerank(self, query: Query, candidates: list[ScoredChunk],
               top_k: int) -> list[ScoredChunk]:
        ranked = sorted(candidates, key=lambda sc: len(sc.chunk.text), reverse=True)
        return ranked[:top_k]        # return only chunks from `candidates`, highest first
```

### 9.3 A custom generator

The `Generator` base owns context packing + citation resolution; you implement
only `_complete(query, packed) → (text, usage)`. `packed.prompt_block` is the
numbered context, `packed.citations`/`packed.texts` are per-chunk.

```python
from rag_toolkit import registry, Generator

@registry.register
class EchoGenerator(Generator):
    name = "echo"
    version = "0.1.0"

    def _complete(self, query, packed):
        if not packed.citations:
            return ("No context found.", {})
        # Cite [1] so the base resolves it to that chunk's pages.
        return (f"Based on the context: {packed.texts[0]} [1]", {})
```

### 9.4 A custom chunker

Implement only *where to cut* (half-open char offsets); the base `chunk()`
Template Method does all the bookkeeping (slicing, ids, index contiguity, page
provenance).

```python
from typing import Iterator
from rag_toolkit import registry, Chunker, Document

@registry.register
class ParagraphChunker(Chunker):
    name = "paragraph"
    version = "0.1.0"

    def iter_spans(self, document: Document) -> Iterator[tuple[int, int]]:
        start = 0
        for para in document.markdown.split("\n\n"):
            end = start + len(para)
            yield (start, end)                            # coordinates, not copies
            start = end + 2                               # skip the "\n\n"
```

### 9.5 A custom OCR engine

The OCR interface is tiny — one page image in, markdown out. It knows nothing
about PDFs or pages.

```python
from rag_toolkit import registry
from rag_toolkit.ingestion.ocr.base import OcrEngine, OcrResult, PageImage

@registry.register
class MyOcrEngine(OcrEngine):
    name = "my-ocr"
    version = "0.1.0"

    def recognize(self, image: PageImage) -> OcrResult:
        markdown = my_model(image.data)                   # image.data is PNG bytes
        return OcrResult(markdown=markdown, confidence=0.9)

# Route DoclingParser's scanned pages to it:
doc = rk.ingest("scan.pdf", ocr_engine="my-ocr")
```

### 9.6 A custom parser

```python
from typing import Iterator
from rag_toolkit import registry, Parser, Page, Source, SourceFormat

@registry.register
class CsvParser(Parser):
    name = "csv"
    version = "0.1.0"
    supported_formats = (SourceFormat.TEXT,)              # for AutoParser routing

    def iter_pages(self, source: Source) -> Iterator[Page]:
        text = source.head(10_000_000).decode("utf-8")
        yield Page(number=1, markdown=csv_to_markdown(text))
```

`parse()` (materialize into a `Document`) comes for free from the base.

### 9.7 A vendor Adapter (lazy import + extra)

Follow the house pattern for any component wrapping a third-party SDK: import it
lazily *inside* the method, and raise an actionable error naming the extra.

```python
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from rag_toolkit import registry, Embedder
from rag_toolkit.core.errors import EmbeddingError

@registry.register
class VoyageEmbedder(Embedder):
    name = "voyage"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "voyage-3"
        api_key: Optional[str] = None                     # → auto-redacted (see §10)

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client = None                               # heavy: build once, reuse

    @property
    def dimensions(self) -> int:
        return 1024

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._client_or_build()
        try:
            return client.embed(list(texts), model=self.config.model).embeddings
        except Exception as exc:                          # normalize vendor errors
            raise EmbeddingError(f"Voyage embed failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _client_or_build(self):
        if self._client is None:
            try:
                import voyageai                            # lazy: optional dependency
            except ImportError as exc:
                raise EmbeddingError(
                    "VoyageEmbedder requires 'voyageai'. "
                    "Install with: pip install voyageai"
                ) from exc
            key = self.config.api_key or os.environ.get("VOYAGE_API_KEY")
            self._client = voyageai.Client(api_key=key)
        return self._client
```

### 9.8 Ship components as a plugin (entry points)

A third-party package's components register automatically when the toolkit
loads, via the `rag_toolkit.components` entry-point group — no import needed by
the user:

```toml
# your package's pyproject.toml
[project.entry-points."rag_toolkit.components"]
my_components = "my_pkg.components"   # importing this module runs its @registry.register
```

A broken plugin never crashes the core — its entry-point load failure is
isolated.

---

## 10. Configuration, secrets & environment

**Config-as-data.** A pipeline is components-by-name plus config. That means a
pipeline is serializable (a dict/YAML of `{kind, name, config}`), which is what
lets the tuner enumerate combinations.

**Secrets are auto-redacted.** Name any credential config field with a marker
substring — `key`, `token`, `secret`, `password`, or `credential` — and it is
redacted from `describe()`/`fingerprint()`/logs automatically. So `api_key`,
`access_key`, `secret_key` all redact; a field called `password` redacts.

**Credential resolution.** The house pattern is *explicit config wins, else the
vendor-standard env var*:

```python
key = self.config.api_key or os.environ.get("MISTRAL_API_KEY")
```

The library **never** calls `load_dotenv()`, never writes secrets, never logs
them — populating the environment is your app's job. Standard env vars:

| Component | Env var |
|---|---|
| `MistralOcrEngine` | `MISTRAL_API_KEY` |
| `GoogleDocAiOcrEngine` | `GOOGLE_APPLICATION_CREDENTIALS` (ADC) |
| `AnthropicGenerator`, `ContextualEnricher` | `ANTHROPIC_API_KEY` (or an `ant` login profile) |
| `MinioBlobStore` | `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` |
| `QdrantVectorStore` | `QDRANT_API_KEY` |

Because rotating a key never changes a fingerprint, cached parses/embeddings
survive key rotation, and pipeline specs / logs never contain secrets.

---

## 11. Errors

All exceptions descend from `RagToolkitError`, so a pipeline boundary can
`except RagToolkitError`. Narrow subclasses carry context:

| Error | Raised when |
|---|---|
| `ConfigError` | invalid/unknown config key |
| `ComponentNotFoundError` | `registry.create` with an unknown name |
| `DuplicateComponentError` | two classes register under one `(kind, name)` |
| `UnsupportedFormatError` | no parser handles a source's format |
| `ParseError` (`source_uri`, `page_number`) | parsing failed |
| `OcrError` | an OCR engine failed on a page |
| `StorageError` (`key`) | a blob/vector store op failed |
| `EmbeddingError` | embedder load/inference/missing-dep |
| `EnrichmentError` | enricher LLM/missing-dep |
| `GenerationError` | generator LLM/missing-dep |

```python
from rag_toolkit import RagToolkitError, ParseError
try:
    rag.index(Source.from_path("weird.pdf"))
except ParseError as e:
    print(e.source_uri, e.page_number)          # exactly which file/page failed
except RagToolkitError as e:
    ...
```

---

## 12. Testing your components

The library ships behavioral **contract tests** — reusable assertions that check
what ABCs and type-checkers can't (ordering, span validity, determinism). When
you write a component, run the matching contract against it. In this repo they
live in `tests/contract_checks.py`:

```python
from tests.contract_checks import assert_embedder_contract, assert_reranker_contract
assert_embedder_contract(MyEmbedder(dimensions=384))
assert_reranker_contract(LengthReranker())
```

Available: `assert_parser_contract`, `assert_chunker_contract`,
`assert_enricher_contract`, `assert_embedder_contract`,
`assert_vector_store_contract`, `assert_lexical_index_contract`,
`assert_retriever_contract`, `assert_reranker_contract`,
`assert_generator_contract`, `assert_blob_store_contract`.

Guarantees they enforce, for example — an embedder returns one equal-width
vector per input, `embed_texts([]) == []`, and is deterministic; a reranker
returns ≤ `top_k` chunks drawn only from its candidates, highest score first; a
vector store's `search` returns nearest-first with provenance intact and
`upsert` is idempotent by `chunk.id`.

Keep the default test run hermetic (no network, no keys); put real-vendor tests
behind an integration marker and gate them on an env var, mirroring the built-in
adapters.

---

## 13. Cheat sheet

```python
import rag_toolkit as rk
from rag_toolkit import (
    Source, Query,                                    # inputs
    RagPipeline, IndexingPipeline, QueryPipeline,     # pipelines
    registry,                                         # build by name
)

# One-call parse
doc = rk.ingest("report.pdf")                         # [docling]

# Full loop, zero deps
rag = rk.RagPipeline()
rag.index(Source.from_path("report.md"))
ans = rag.ask("question?", k=5)                       # ans.text, ans.citations

# Build any component by name
emb   = registry.create("embedder", "hashing", dimensions=512)
store = registry.create("store", "memory")
rr    = registry.create("reranker", "keyword")
registry.available("retriever")                       # discover what's registered
```

Discover everything currently registered:

```python
registry.available()                                  # every "kind:name" string
registry.available("retriever")                       # names within one kind
```

---

*This guide covers the implemented library (through v0.5: ingestion → chunking →
enrichment → embedding → storage → retrieval → reranking → generation, plus the
three pipelines). Evaluation and auto-tuning (v0.6–0.7) are on the roadmap; see
`ARCHITECTURE.md` and `AGENTS.md` for the design rationale and what's next.*
