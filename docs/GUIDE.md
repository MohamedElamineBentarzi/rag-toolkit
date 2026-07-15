# rag-blocks — one-page cheat sheet

A quick reference. For the friendly, worked-example walkthrough, read the
**[guide](guide/README.md)**. For the design reasoning and internals, see
**[ARCHITECTURE.md](../ARCHITECTURE.md)**.

`rag-blocks` reads your documents, searches them, and answers questions with
citations that point back to the exact file and page. Every stage is a part you
can swap; the core needs no dependencies beyond the Python standard library.

---

## Install

```bash
pip install rag-blocks                    # core: text/markdown, local search, cited answers
```

| Add-on | Unlocks |
|---|---|
| `[docling]` | Read PDF, Word, PowerPoint, Excel, HTML, scanned images |
| `[sentence-transformers]` | Real embeddings + cross-encoder reranking |
| `[qdrant]` | Qdrant vector database |
| `[anthropic]` | Claude answers + AI-written chunk context |
| `[minio]` | S3-compatible file storage |
| `[mistral]` / `[google]` | Cloud OCR for scanned pages |

```bash
pip install "rag-blocks[docling,sentence-transformers,anthropic]"
```

## Quick start

```python
from rag_blocks import RagPipeline, Source

rag = RagPipeline()                                # all-local defaults, no key
rag.index(Source.from_path("handbook.md"))         # read → cut → store
answer = rag.ask("How many vacation days do new hires get?")

print(answer.text)                                  # answer with [1], [2] markers
for c in answer.citations:
    print(f"[{c.marker}] {c.doc_id[:8]}  pages {c.page_start}-{c.page_end}")
```

Two methods carry everything: **`index`** (take in documents) and **`ask`**
(answer questions). Every answer traces back to its source.

## Going to production — swap parts, same code

```python
from rag_blocks import (RagPipeline, ChunkIndex, SentenceTransformerEmbedder,
                         QdrantVectorStore, BM25Index, AnthropicGenerator, CrossEncoderReranker)

rag = RagPipeline(
    chunk_index=ChunkIndex(
        store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
        dense=SentenceTransformerEmbedder(model="BAAI/bge-m3"),   # search by meaning
        lexical=BM25Index(),                                      # search by keyword → hybrid
    ),
    generator=AnthropicGenerator(model="claude-opus-4-8"),
    refine=[CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")],
)
```

`index`/`ask` don't change — you just named better parts.

---

## The parts (kind → built-ins)

| Kind | What it does | Built-ins |
|---|---|---|
| `parser` | file → text with page numbers | `AutoParser` (router), `PlainTextParser`, `DoclingParser` `[docling]` |
| `ocr` | read text off a scanned page | `MistralOcrEngine` `[mistral]`, `GoogleDocAiOcrEngine` `[google]` |
| `chunker` | cut text into pieces | `MarkdownChunker` (cut on headings), `FixedChunker` (by size + overlap) |
| `enricher` | add context / generate chunks | `HeadingEnricher`, `ContextualEnricher` `[anthropic]` |
| `embedder` | text → meaning-vector | `HashingEmbedder` (no deps), `SentenceTransformerEmbedder` `[sentence-transformers]`, `CachingEmbedder` (wraps any) |
| `sparse_encoder` | text → weighted terms | interface only — bring your own |
| `vector_store` | hold + search vectors | `MemoryVectorStore`, `QdrantVectorStore` `[qdrant]` |
| `lexical_index` | keyword (BM25) search | `BM25Index` |
| `blob_store` | durable file storage | `LocalBlobStore`, `MinioBlobStore` `[minio]` |
| `index` | owns all search-forms of a corpus | `ChunkIndex` |
| `retriever` | question → relevant chunks | `IndexRetriever`, `HybridRetriever`, `FusionRetriever`, `MultiQueryRetriever`, `HydeRetriever` |
| `refiner` | clean up results | `CrossEncoderReranker` `[sentence-transformers]`, `KeywordRefiner`, `ScoreThreshold`, `NeighborExpander` |
| `generator` | chunks → cited answer | `ExtractiveGenerator` (no model), `AnthropicGenerator` `[anthropic]` |

Build a simple part by name: `registry.create("embedder", "hashing", dimensions=512)`.
Build a live part (index, retriever, pipeline) by handing it its pieces.

## The data shapes

```
Source → Page → Document → Chunk → (search) → ScoredChunk → Answer (+ Citations)
```

- **`Source`** — a pointer to one input file (`Source.from_path` / `.from_bytes`).
- **`Document`** — the whole file as markdown; `id` is a content fingerprint (the `doc_id`).
- **`Chunk`** — a searchable piece; carries `char_start/end`, `page_start/end` (its provenance).
- **`ScoredChunk`** — a chunk + relevance score (higher = better).
- **`Citation`** — `marker` (`[n]`), `doc_id`, `page_start/end`.
- **`Answer`** — `text` + `citations` + `usage`.

## The three pipelines

```python
IndexingPipeline(parser=?, chunker=?, enrich=[...], sinks=[...], blob_store=?)   # write path
QueryPipeline(retriever, refine=[...], fetch_k=50)                               # read path
RagPipeline(chunk_index=?, generator=?, enrich=[...], refine=[...], blob_store=?) # both, shared index
```

`RagPipeline` picks a retriever for you: one search-form → plain; several → hybrid.
Add a `blob_store` to keep originals and enable `rag.source_uri(doc_id)` /
`rag.download_url(doc_id)`.

---

## Common recipes

```python
# Hybrid (meaning + keyword) — just declare both:
ChunkIndex(store, dense=embedder, lexical=BM25Index())        # → hybrid retrieval, automatic

# Ask the question several ways:
MultiQueryRetriever(HybridRetriever(index), complete=gen.complete, n=4)

# Search on a drafted answer (HyDE):
HydeRetriever(IndexRetriever(index), complete=gen.complete)

# Search small, answer big:
QueryPipeline(IndexRetriever(index), refine=[NeighborExpander(index, window=2)])

# Search across two collections:
FusionRetriever([IndexRetriever(legal_index), IndexRetriever(hr_index)])

# Cache embeddings across runs:
CachingEmbedder(SentenceTransformerEmbedder(...), cache=LocalBlobStore("./cache"))
```

Full runnable versions: **[guide/10-recipes.md](guide/10-recipes.md)**.

## Add your own part

```python
from rag_blocks import registry, Refiner

@registry.register
class LengthRefiner(Refiner):
    name = "length"
    version = "0.1.0"
    def refine(self, query, candidates, k):
        return sorted(candidates, key=lambda sc: len(sc.chunk.text), reverse=True)
```

1. Subclass the stage base · 2. set `name` + `version` · 3. optional `Config` ·
4. implement the one method · 5. `@registry.register`.

Prove it: `assert_refiner_contract(LengthRefiner())` (a contract test per stage
in `tests/contract_checks.py`). Bump `version` whenever behavior changes so caches
invalidate. Details: **[guide/09-extending-and-testing.md](guide/09-extending-and-testing.md)**.

## Testing

```bash
pytest                                    # fast, hermetic — no network, no keys (default)
pytest -m integration tests/integration   # real Qdrant, models, Claude, MinIO
ruff check && mypy                         # both must pass
```

---

Deeper: **[the guide](guide/README.md)** (worked examples) ·
**[ARCHITECTURE.md](../ARCHITECTURE.md)** (design reasoning).
