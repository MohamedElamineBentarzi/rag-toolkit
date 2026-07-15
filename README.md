# rag-toolkit

Composable building blocks for production RAG pipelines — every stage is a
swappable component, every pipeline is a serializable config, and an
auto-tuning evaluation suite finds the best combination for *your* dataset
with full trial logs.

**Status: v0.1 — ingestion subsystem.** Any file in → clean markdown out,
with per-page OCR routing (Mistral, Google Document AI, or your own engine)
and streaming that keeps memory flat on 2 000-page PDFs.

## Install

```bash
pip install "rag-toolkit[docling]"           # local parsing (default route)
pip install "rag-toolkit[docling,mistral]"   # + Mistral OCR
pip install "rag-toolkit[minio]"             # + MinIO / S3-compatible storage
```

The core has **zero dependencies**; vendor SDKs are optional extras.

### GPU acceleration

Only the local-model components use a GPU — `SentenceTransformerEmbedder`,
`CrossEncoderReranker`, and `DoclingParser`'s layout models. GPU-ness is a property of
*how PyTorch is installed*, not of a toolkit extra: the CUDA wheels live on
PyTorch's own index, not PyPI, so a pip extra can't pull them. What to do
depends on your OS:

```bash
# Linux (NVIDIA): CUDA torch already comes from PyPI — nothing special.
pip install "rag-toolkit[sentence-transformers]"

# Windows / macOS: PyPI's torch is CPU-only. Install a CUDA torch FIRST,
# then the extras (pip leaves the already-satisfied torch alone):
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install "rag-toolkit[sentence-transformers]"
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

**Install order matters.** Any later `pip install <torch-dependent-package>` on
its own can re-resolve `torch` and pull the CPU wheel back from PyPI, clobbering
your CUDA build — so install the CUDA torch **last**, or re-run it if a later
install flips `torch.cuda.is_available()` back to `False`.
[`requirements-gpu.txt`](requirements-gpu.txt) captures the CUDA-torch install
(edit the `cu126` tag to match your driver — see `nvidia-smi`).

There is deliberately **no `[all-gpu]` extra**: it would be redundant on Linux
and misleading on Windows/macOS (pip extras can't select PyTorch's CUDA index).

## Quick start

```python
import rag_toolkit as rk

# One call: any file → markdown Document with page provenance
doc = rk.ingest("report.pdf")
print(doc.markdown[:500])
print(doc.pages_for_span(1200, 1800))   # -> which pages a char range came from

# Scanned document through cloud OCR (needs MISTRAL_API_KEY)
doc = rk.ingest("scan.pdf", ocr_engine="mistral", ocr_policy=rk.OcrPolicy.FORCE)

# Streaming — memory stays O(page batch) on huge files
parser = rk.AutoParser()
for page in parser.iter_pages(rk.Source.from_path("huge.pdf")):
    process(page.markdown)
```

## Bring your own OCR

```python
from dataclasses import dataclass
from rag_toolkit import registry
from rag_toolkit.ingestion.ocr.base import OcrEngine, OcrResult, PageImage

@registry.register
class MyOcrEngine(OcrEngine):
    name = "my-ocr"

    @dataclass
    class Config:
        endpoint: str = "http://localhost:9000"

    def recognize(self, image: PageImage) -> OcrResult:
        markdown = my_model(image.data)          # your logic here
        return OcrResult(markdown=markdown)

doc = rk.ingest("scan.pdf", ocr_engine="my-ocr")   # that's it
```

## Ask a question (the whole loop)

`RagPipeline` is the facade over everything: index files, then ask. The defaults
are the zero-dependency stack (hashing embedder, in-memory store, extractive
generator), so this runs with no extras and no API key:

```python
from rag_toolkit import RagPipeline, Source

rag = RagPipeline()
rag.index(Source.from_path("report.pdf"))          # parse → chunk → embed → store

answer = rag.ask("What was Q3 revenue?", k=5)      # retrieve → refine → generate
print(answer.text)
for c in answer.citations:                          # each resolves to doc + pages
    print(f"  [{c.marker}] {c.doc_id} p{c.page_start}-{c.page_end}")
```

Swap in production components without changing the wiring. Backends live on a
`ChunkIndex` (the aggregate that owns a corpus's searchable representations),
created once and shared by the pipeline:

```python
from rag_toolkit import (
    RagPipeline, ChunkIndex, SentenceTransformerEmbedder, QdrantVectorStore,
    BM25Index, AnthropicGenerator,
)

rag = RagPipeline(
    chunk_index=ChunkIndex(
        store=QdrantVectorStore(url="http://localhost:6333"),
        dense=SentenceTransformerEmbedder(),                # bge-m3
        lexical=BM25Index(),                                # ⇒ hybrid retrieval, derived
    ),
    generator=AnthropicGenerator(),                         # claude-opus-4-8
)

# The 80% dense-only case has a convenience constructor:
rag = RagPipeline.dense(
    embedder=SentenceTransformerEmbedder(),
    store=QdrantVectorStore(url="http://localhost:6333"),
    generator=AnthropicGenerator(),
)
```

## Chunk a document

Chunkers turn a parsed `Document` into retrieval `Chunk`s. A strategy decides
only *where* to cut (character-offset spans); the base class owns id assignment,
contiguous indexing, and page provenance — so every chunk can still answer
"which pages did I come from":

```python
import rag_toolkit as rk
from rag_toolkit import FixedChunker, MarkdownChunker

doc = rk.ingest("report.pdf")

chunker = FixedChunker(chunk_chars=1600, overlap_chars=200)  # or by config:
chunker = rk.registry.create("chunker", "markdown-aware")     # cut at headings

for chunk in chunker.chunk(doc):
    print(chunk.index, chunk.page_start, chunk.page_end, chunk.text[:80])
```

`char_start`/`char_end` are the primary provenance; pages are derived from them.
Overlapping spans are legal — that is how overlap strategies express themselves.

## Index a corpus

`IndexingPipeline` is the thin wiring that runs `Source → parse → chunk` and,
when you hand it a blob store, captures the durable truth on the way (raw bytes
+ parse cache, content-addressed, deduped). All intelligence is in the
components; the pipeline is a dumb for-loop with a tracing hook:

```python
from rag_toolkit import IndexingPipeline, LocalBlobStore, Source

pipeline = IndexingPipeline(
    blob_store=LocalBlobStore(root="./.rag_cache/blobs"),  # optional truth store
    trace=print,                                           # optional TraceEvent hook
)

for chunk in pipeline.index(Source.from_path("report.pdf")):
    embed_and_store(chunk)        # your v0.3 embedder/vector store goes here
```

Swap the parser, chunker, or blob store by passing a different component — no
pipeline code changes. Re-indexing the same bytes is a no-op (same content →
same key).

## Embed and search

A `ChunkIndex` owns a corpus's searchable representations: `add(chunks)` writes
every one, `search(representation, TEXT, k)` encodes the query with the *same*
encoder that encoded the corpus. The `memory` store + `hashing` embedder give a
fully local, dependency-free loop; swap in `sentence-transformers` + `qdrant` for
production by changing two component names:

```python
from rag_toolkit import (ChunkIndex, HashingEmbedder, IndexingPipeline,
                         MemoryVectorStore, Source)

index = ChunkIndex(
    store=MemoryVectorStore(),         # or QdrantVectorStore(url="http://localhost:6333")
    dense=HashingEmbedder(),           # or SentenceTransformerEmbedder() (bge-m3)
)

# Index once, streaming, with the index as a write sink:
for _ in IndexingPipeline(sinks=[index]).index(Source.from_path("report.pdf")):
    pass

for hit in index.search("dense", "What was Q3 revenue?", k=5):
    print(hit.score, hit.chunk.page_start, hit.chunk.text[:80])   # provenance intact
```

`search` returns `ScoredChunk`s with the full chunk (text + page provenance)
inline, so answering never has to touch the blob store. Add `lexical=BM25Index()`
(or a `sparse=` encoder) and the index carries several representations at once.

## Persist raw files

A `BlobStore` is the durable *truth store* for ingested bytes (raw files today,
the parse cache next). Same tiny interface on disk or on any S3-compatible
backend — swap by config, no other code changes:

```python
from rag_toolkit import LocalBlobStore, MinioBlobStore, Source

# On disk (zero-dep default, atomic writes)
store = LocalBlobStore(root="./.rag_cache/blobs")

# ...or any S3-compatible backend (MinIO, AWS S3, R2, B2) — needs [minio].
# Credentials: config wins, else MINIO_ACCESS_KEY / MINIO_SECRET_KEY.
store = MinioBlobStore(endpoint="localhost:9000", bucket="rag-toolkit")

src = Source.from_path("report.pdf")
key = f"raw/{src.content_hash()}/original.pdf"   # content-addressed ⇒ dedup free
if not store.exists(key):                         # cheap pre-check
    store.put(key, src.open().read())
assert store.get(key)[:5] == b"%PDF-"
```

The store treats keys as opaque strings — the content-addressed layout lives in
your pipeline, so the two implementations stay perfectly interchangeable.

## Design

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline map, the data
contracts, the pattern-by-pattern rationale, and the design of the evaluation
and auto-tuning suite.

## Development

```bash
pip install -e ".[dev]"
pytest                      # fast, hermetic suite — no vendor deps needed
pytest -m integration       # opt-in: real docling/OCR runs
ruff check . && mypy rag_toolkit
```

Tests mirror the package layout. `tests/contract_checks.py` holds the
behavioral contract every new `Parser` must pass — call
`assert_parser_contract(...)` from your parser's tests and you inherit the
guarantees the rest of the pipeline relies on.
