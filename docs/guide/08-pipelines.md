# 08 · Pipelines — wiring it together

A pipeline is just the glue that runs the parts in order. There are three, and
they're deliberately simple — all the smarts live in the parts; the pipeline only
connects them.

- **`IndexingPipeline`** — the write path: read files → cut → (optionally enrich)
  → store.
- **`QueryPipeline`** — the read path: search → clean up → return the best.
- **`RagPipeline`** — both, wired together, sharing one index. This is the one you
  usually use (it's the `rag` from Part 01).

## `RagPipeline` — the one you'll use

`RagPipeline` builds and connects everything, then gives you two methods: `index`
and `ask`. Everything is optional and has a sensible default:

```python
RagPipeline(
    chunk_index=None,        # default: all-local memory index + hashing embedder
    generator=None,          # default: the no-model extractive generator
    retriever=None,          # default: derived from your index (see below)
    parser=None, chunker=None,
    enrich=[],               # optional: enrich/generate chunks at index time (Part 10)
    refine=[],               # optional: clean-up steps at query time (Part 06)
    extra_sinks=[],          # optional: also write chunks elsewhere (e.g. a graph db)
    blob_store=None,         # optional: keep originals + enable download links
    fetch_k=50,              # how many candidates to pull before trimming
)
```

The two methods:

```python
rag.index(sources)          # read → cut → enrich → store (and persist)
answer = rag.ask(question, k=8)   # search → clean up → write a cited answer
```

### It picks a retriever for you

If you don't pass a `retriever`, `RagPipeline` builds a sensible one from your
index: **one search-form → a plain retriever; several → a hybrid retriever that
blends them.** So the moment you add keyword search to your index, you
automatically get hybrid retrieval — no extra wiring:

```python
index = ChunkIndex(store, dense=embedder, lexical=BM25Index())
rag = RagPipeline(chunk_index=index)     # ← hybrid retrieval, automatically
```

If you *do* pass your own retriever, the pipeline checks it's actually reading from
the same index you gave it — and stops with a clear error if not, so you can never
accidentally search one index while having filled another.

### The `.dense()` shortcut

For the common "just meaning-search" setup:

```python
rag = RagPipeline.dense(
    embedder=SentenceTransformerEmbedder(),
    store=QdrantVectorStore(url="http://localhost:6333"),
    generator=AnthropicGenerator(),
)
```

## Keeping originals: the truth store

Pass a `blob_store` and the pipeline also saves the durable, rebuildable truth
next to your searchable data:

```python
from rag_blocks import RagPipeline, LocalBlobStore
rag = RagPipeline(blob_store=LocalBlobStore("./store"))
```

With it wired in, three things get saved as you index — the **original file
bytes**, the **parsed text** (so re-indexing doesn't re-read the file), and a
small **name/link record** per document (so citations resolve to a filename and a
download link). It's also what powers `rag.source_uri(doc_id)` and
`rag.download_url(doc_id)`.

Everything here is keyed by content, so re-indexing the same file is a no-op — no
duplicates, no wasted work.

## What gets re-done when you re-index

The pipeline caches each stage independently, so a second run is cheap where it
can be:

| Stage | On re-index |
|---|---|
| Save original bytes | skipped (already there) |
| Parse the file | skipped (reuses the saved parse) |
| Cut into chunks | re-runs — it's fast and deterministic |
| Embed | re-runs — **unless** you wrapped the embedder in `CachingEmbedder` |
| Store vectors | no duplicates (keyed by chunk id) |

There's no single "already indexed, skip everything" switch on purpose — caching
each stage separately lets you, say, re-chunk without re-parsing. To make
re-indexing nearly free end to end, use a `CachingEmbedder` (from
[Part 05](05-representations-and-storage.md)) so embeddings are cached too.

## Writing chunks to more than one place

The write path can send chunks to **several destinations at once**. Your
`ChunkIndex` is the main one, but you can add more via `extra_sinks` — anything
with an `add(chunks)` and `persist()` method. This is how you'd run the toolkit's
search alongside, say, a GraphRAG index built from the same chunks:

```python
rag = RagPipeline(chunk_index=index, extra_sinks=[my_graph_index])
```

## Watching what happens (tracing)

Every pipeline accepts a `trace` function called at each stage with timing info —
handy for seeing where time goes or building your own logging:

```python
events = []
rag = RagPipeline(trace=events.append)
rag.index(source)
# events now hold (stage, file, duration_ms, details) for parse, store, chunk, …
```

## Index once, try many strategies

A useful property for tuning: because searching is separate from indexing, you can
**index your documents once** and then try many retrieval and clean-up strategies
against the same stored data — no re-indexing:

```python
# Index once:
rag = RagPipeline(chunk_index=index)
rag.index(corpus)

# Now compare strategies freely, all reading the same index:
from rag_blocks import QueryPipeline, IndexRetriever, HybridRetriever, CrossEncoderReranker
for retriever in [IndexRetriever(index, "dense"),
                  IndexRetriever(index, "lexical"),
                  HybridRetriever(index)]:
    for clean_up in [[], [CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")]]:
        qp = QueryPipeline(retriever, refine=clean_up)
        # evaluate qp on your test questions…
```

This is exactly what the (upcoming) auto-tuner does for you: one indexing pass,
many strategies compared.

Next: **[09 · Extending & testing](09-extending-and-testing.md)** — add your own
part and prove it works.
