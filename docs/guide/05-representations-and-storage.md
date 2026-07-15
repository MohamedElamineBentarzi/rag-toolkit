# 05 · Making chunks searchable

This is the center of the toolkit. A pile of chunks isn't useful until you can
*search* it — and the powerful idea here is that your documents can be searchable
**several ways at once**: by meaning, by keyword, or both. One object owns all of
those ways and keeps them in sync: the **`ChunkIndex`**.

This page builds up to it. First the tools that make chunks searchable, then the
places that hold them, then the `ChunkIndex` that ties it together, and finally
how a citation turns back into a real file with a download link.

```
             ┌──────────────── ChunkIndex ────────────────┐
chunks ─add─▶│  by meaning:  Embedder                       │
             │  by keyword:  BM25Index                      │──▶ search
             └─────────────────────────────────────────────┘
question ─search(how, text, k)──▶ the most relevant chunks
```

## Two ways to search, in plain terms

- **Search by meaning** ("dense" / embeddings). A model turns each chunk into a
  list of numbers that captures *what it's about*. Chunks about similar things end
  up with similar numbers, so a question finds passages that *mean* the same
  thing even if they use different words. Great for "explain the refund process."
- **Search by keyword** ("lexical" / BM25). Classic term matching — it finds
  chunks containing the actual words you searched for. Great for exact things: a
  product code, "clause 7.3," a person's name.

You'll often want both. The `ChunkIndex` lets you have them together.

> The design reasoning — why keyword search isn't stored as a vector, why chunks
> never carry their own vectors — is summarized in plain words below and covered
> in depth in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

## Embedders — turn text into "search by meaning"

An **embedder** converts text into a dense vector. Two ship in the box.

### `HashingEmbedder` — no downloads, no network

```python
from rag_blocks import HashingEmbedder
emb = HashingEmbedder(dimensions=256)
```

This is a real (if basic) embedder built from standard library math — no model,
no internet. Chunks that share words land near each other. It's perfect for
trying things out, running tests, and getting a pipeline working before you pull
in a heavy model. It's the default so that `RagPipeline()` works with zero setup.

### `SentenceTransformerEmbedder` — real, high-quality embeddings

```python
from rag_blocks import SentenceTransformerEmbedder
SentenceTransformerEmbedder(model="BAAI/bge-m3")   # add-on: [sentence-transformers]
```

This uses a real embedding model (default `bge-m3`: multilingual, strong at
retrieval). The model is downloaded once and reused. This is what you'd use in
production for quality search-by-meaning.

**One detail that quietly matters:** a good embedder encodes a *question*
slightly differently from a *passage*. The toolkit handles this for you — you
never have to think about it — but it's why search stays accurate. If you ever
plug in a model that needs a special question prefix, it's a one-line setting, no
code change.

## `CachingEmbedder` — never embed the same text twice

Embedding is the slow, expensive step. Wrap any embedder in a `CachingEmbedder`
and identical text is embedded once and remembered — across documents, across
re-runs, across restarts:

```python
from rag_blocks import CachingEmbedder, SentenceTransformerEmbedder, LocalBlobStore
cached = CachingEmbedder(
    SentenceTransformerEmbedder(model="BAAI/bge-m3"),
    cache=LocalBlobStore("./embed-cache"),
)
```

It's invisible in every way that matters: results are identical to the embedder
it wraps, it just skips work it has already done. Swap the underlying model and
the cache correctly starts fresh — you never get a stale vector from the wrong
model.

## Sparse encoders — a second kind of "keyword-ish" search

There's a middle option between meaning and keyword: a **sparse encoder**
(SPLADE-style) turns a chunk into a short list of weighted terms — like keyword
search, but the model can add related terms the text didn't literally contain.

```python
SparseEncoder:
    encode_texts(texts) -> list[SparseVector]
    encode_query(text)  -> SparseVector
```

> **Status:** the toolkit ships full storage and search support for sparse
> vectors and the encoder interface, but no built-in sparse *model* yet — bring
> your own, or (much more commonly) just use classic BM25 keyword search below,
> which *is* built in.

## Where searchable data lives

### The `BlobStore` — your durable file storage

A `BlobStore` is dead-simple storage: put bytes under a key, get them back. It
holds the **originals** — the raw files you ingested and their parsed text. Think
of it as the source of truth you can always rebuild everything else from.

```python
LocalBlobStore("./store")                    # files on disk — zero setup
MinioBlobStore(bucket="rag", endpoint=...)   # S3 / MinIO / R2 — add-on [minio]
```

`LocalBlobStore` writes safely (a crash never leaves a half-written file) and
refuses to escape its folder. `MinioBlobStore` talks to anything S3-compatible.
Both can hand you a download link for a stored file — which is how citations get
a "click to open the original" URL.

### The `VectorStore` — holds the searchable vectors

A `VectorStore` holds your chunks' vectors and finds the closest ones to a
question. The important feature: **it holds several named search-forms per
chunk** — a "dense" space for meaning, a "splade" space for sparse, whatever you
declare — side by side.

It also stores each chunk's text and page info *next to* its vectors, so a search
returns fully-formed chunks (with page numbers ready for citation) without
touching the file storage. Losing the vector store is cheap — you can rebuild it
from the originals.

Two implementations:

- **`MemoryVectorStore`** — pure Python, holds everything in memory. Not for
  scale; it's the honest, dependency-free store for trying things and running
  tests. It's the default.
- **`QdrantVectorStore`** — a real Qdrant database (add-on `[qdrant]`), for
  production. Connect in-memory, to a local file, or to a server:

  ```python
  QdrantVectorStore(url="http://localhost:6333", collection="docs")
  ```

  If it finds an existing collection whose shape doesn't match what you declared,
  it stops with a clear error rather than silently corrupting your data.

### `BM25Index` — classic keyword search

```python
from rag_blocks import BM25Index
BM25Index()
```

BM25 is the well-known keyword-ranking method. Here's the interesting part, and
it explains why keyword search isn't just "another vector":

> **Why keyword search is stored differently.** A chunk's keyword score depends on
> the *whole corpus* — how rare a word is across all your documents, how long the
> average document is. That's not a fixed property of one chunk you can compute
> and freeze; it changes as the corpus grows. So keyword search keeps corpus-wide
> statistics and scores at search time, rather than storing a per-chunk vector.

You use it exactly like the other search-forms, though — the difference is
internal. It can save itself to a `BlobStore` so it survives restarts:

```python
index = BM25Index(store=LocalBlobStore("./bm25"), namespace="my-corpus")
index.load()          # reload a saved index if there is one
index.add(chunks)
index.persist()       # save it
```

## `ChunkIndex` — the one object that owns your corpus

Here's the payoff. A **`ChunkIndex`** owns *every* search-form of your documents
at once, and guarantees one thing:

> Every searchable form of every chunk was made by the same tools this index
> declares — and questions are searched the exact same way. So results are always
> comparable.

That guarantee is what makes search trustworthy: you can't accidentally search
questions one way and store documents another.

```python
from rag_blocks import ChunkIndex, MemoryVectorStore, HashingEmbedder, BM25Index

index = ChunkIndex(
    store=MemoryVectorStore(),      # where vectors live
    dense=HashingEmbedder(),        # search-by-meaning, named "dense"
    lexical=BM25Index(),            # search-by-keyword, named "lexical"
)

index.add(chunks)                   # stores every search-form in one pass
index.representations()             # ['dense', 'lexical']

# You pass TEXT, not a vector — the index encodes the question for you:
index.search("dense",   "quick brown fox", k=5)
index.search("lexical", "clause 7.3", k=5, filters={"doc_id": "abc"})
index.fetch({"doc_id": "abc", "index": [3, 4, 5]})   # get specific chunks, no search
```

### The constructor scales from simple to advanced

The common case reads like plain English; the fancy case is possible without
cluttering the simple one:

```python
ChunkIndex(store, dense=embedder)                      # one meaning-search, named "dense"
ChunkIndex(store, dense=embedder, lexical=BM25Index()) # meaning + keyword
ChunkIndex(store, dense={"bge": a, "e5": b})           # two models side by side, to compare
```

- Hand it one embedder and it names the search-form for you (`dense`).
- Hand it a named group and you get several (handy for comparing two models).

When you build a `ChunkIndex`, it checks the store's shape right away — so a
mismatch fails immediately, at setup, not halfway through indexing a thousand
files.

### What it does

- **`add(chunks)`** — makes every search-form for every chunk and stores them in
  one pass. Safe to re-run; re-adding the same chunk updates it.
- **`search(how, text, k, filters)`** — searches one form by name. You give it
  the *question text*; it encodes the question the matching way and returns the
  top chunks. This one method is the guarantee that questions and documents are
  always compared correctly.
- **`fetch(filters)`** — pull specific chunks without searching (e.g. "give me
  pieces 3, 4, 5 of document abc") — used for neighbor context and lookups.
- **`update_representation(name, chunks)`** — re-do just one search-form (e.g.
  after upgrading your embedding model) without rebuilding the rest.

### Why chunks don't store their own vectors

You might expect each chunk to carry its embedding. It deliberately doesn't. A
chunk is a *fact* — a piece of a document. Its vectors are *interpretations* of
that fact under a particular model, and you might have several (dense, sparse) or
swap the model later. Keeping vectors in the store (keyed by chunk id) instead of
on the chunk means a chunk's identity never changes when you re-embed — you can
try three embedding models on the same corpus without touching a single chunk.

## Combining with other storage

A `ChunkIndex` is one place your chunks can be written, but not the only one. The
write path can send chunks to **several destinations at once** — for example, a
`ChunkIndex` *and* a separate graph database. Anything with an `add(chunks)` and
`persist()` method can receive them. This is how you'd run the toolkit's search
alongside a GraphRAG system on the same ingested data. ([Part 08](08-pipelines.md)
shows the wiring.)

## `DocumentCatalog` — turn a citation back into a real file

A citation carries a `doc_id`. To show a user something clickable, you need two
more facts: the file's real name and a way to download it. The **catalog** is a
tiny record that maps `doc_id` → name + download link:

```python
cat = rag.catalog                    # present when you configured a blob_store
cat.source_uri(doc_id)               # "report.pdf" — for display
cat.download_url(doc_id)             # a file:// or time-limited S3 link
```

It needs a `BlobStore` (there has to be a stored original to link to). With one
wired in, every citation becomes "**report.pdf, pages 4–5** — [open]."

## What you now know

- **Search by meaning** (embedders) and **search by keyword** (BM25) are two ways
  to find chunks; sparse encoders are a third.
- The **`BlobStore`** holds durable originals; the **`VectorStore`** holds the
  searchable vectors and is rebuildable.
- A **`ChunkIndex`** owns every search-form of your corpus and guarantees
  questions are searched the same way documents were stored.
- The **`DocumentCatalog`** turns a citation's `doc_id` into a file name and a
  download link.

Next: **[06 · Retrieval & refinement](06-retrieval-and-refinement.md)** — reading
the index, combining searches, and cleaning up what comes back.
