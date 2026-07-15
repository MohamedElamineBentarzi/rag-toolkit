# 02 · Concepts

The whole toolkit is built from **parts you can swap**. Once you understand what
a "part" is and the few ways parts fit together, every feature in the rest of
this guide becomes predictable — because they all follow the same handful of
rules. This page is those rules, in plain terms.

> Want the design reasoning behind these choices — the patterns, the trade-offs,
> the decisions that were considered and rejected? That lives in
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md). This page is what you need to *use*
> the toolkit.

## A pipeline is a line of parts

When you ask a question about your documents, the work happens in stages:

```
your files → read → cut into pieces → make searchable → [ask] → find pieces → clean up → write answer
```

Each stage is done by a **part** you can replace:

| Stage | The part | What it does |
|---|---|---|
| read | **parser** | turns a PDF/Word/text file into clean text with page numbers |
| cut into pieces | **chunker** | splits that text into passage-sized pieces ("chunks") |
| make searchable | **embedder** / **index** | turns each chunk into something you can search |
| find pieces | **retriever** | given a question, pulls back the most relevant chunks |
| clean up | **refiner** | re-orders or trims those chunks before answering |
| write answer | **generator** | writes the final answer using only those chunks |

You never *have* to think about all of them — the defaults are wired for you. But
when you want better quality, you now know exactly which part to swap.

## Why swapping always works: parts talk through fixed shapes

A chunker has never heard of a parser. A retriever has never heard of which
database you're using. Parts don't know about each other at all — they only know
a small set of **data shapes** that flow between them:

```
Source → Page → Document → Chunk → ScoredChunk → Answer
```

A parser's only job is to produce `Page`s. A chunker's only job is to turn a
`Document` into `Chunk`s. Because every part agrees on these shapes, you can drop
in a different parser and *nothing downstream notices* — it still receives the
same `Page`s. That's the reason "swap the part, don't change your code" actually
holds. ([Part 03](03-data-contracts.md) walks through each shape.)

The rule of thumb: **if two parts seem like they need to know about each other,
a shared data shape is missing** — not a direct connection between them.

## Every part has a `kind` and a `name`

A part is identified by two words: its **kind** (the stage it fills) and its
**name** (which implementation):

```python
# kind = "embedder", name = "hashing"
# kind = "embedder", name = "sentence-transformers"
# kind = "parser",   name = "docling"
```

This is what lets you describe a whole setup as plain data —
`{"embedder": "sentence-transformers", "chunker": "markdown-aware"}` — which is
how the toolkit can later try many combinations automatically to find the best
one for your documents.

You build a simple part by its name:

```python
from rag_blocks import registry
emb = registry.create("embedder", "hashing", dimensions=512)
registry.available("embedder")   # see every embedder you can name
```

## Two ways parts combine

There are only two, and you already saw both in Part 01.

**1. A chain — "do this, then this, then this."** After the search finds
candidate chunks, you can run them through a list of clean-up steps: re-rank
them, pull in neighboring text, drop low-scoring ones. Each step takes a list of
chunks and returns a list of chunks, so they stack in any order:

```python
refine=[CrossEncoderReranker(), NeighborExpander()]   # runs left to right
```

An empty chain (`refine=[]`) simply does nothing — there's no special "do
nothing" part to configure, the empty list *is* the do-nothing.

**2. Wrapping — "one part made of other parts."** Some parts contain others. A
hybrid retriever is really two retrievers (meaning-search + keyword-search) with
a blender around them. You don't add a new "hybrid stage" to the pipeline — you
wrap two retrievers in one:

```python
HybridRetriever(index)     # one retriever that contains several inside it
```

That's the entire structural vocabulary: **chains** for "steps in a row," and
**wrapping** for "a part built from parts." Everything fancy — asking the
question several ways, searching multiple sources at once — is just wrapping.

## The `ChunkIndex`: the thing that holds your data

One part deserves special mention because it's the center of everything: the
**`ChunkIndex`**. It owns every searchable form of your documents at once —
meaning-vectors, keyword-index, and any others — and it guarantees they stay in
sync. When you ask a question, it makes sure the question is searched the *same
way* your documents were stored, so results are always comparable.

```python
index = ChunkIndex(
    store=MemoryVectorStore(),      # where the searchable data lives
    dense=HashingEmbedder(),        # search-by-meaning
    lexical=BM25Index(),            # search-by-keyword
)
```

You'll meet it properly in [Part 05](05-representations-and-storage.md). For now:
it's the one object your corpus lives in, and it's what a retriever reads from.

## Simple parts vs. live parts (a useful distinction)

There are two flavors of part, and they're built differently:

- **Simple parts** (parsers, chunkers, embedders, refiners) carry only settings.
  You build them by name, as shown above — they're interchangeable and easy to
  describe as data.
- **Live parts** (a `ChunkIndex`, a retriever, a pipeline) hold *actual loaded
  data or connections* — a populated database, a running model. You build these
  by handing them their pieces directly:

  ```python
  index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
  ```

  You can't build a live part just by naming it, because there'd be nothing
  inside it. If you try, you get a clear error.

Short version: **simple parts by name, live parts by handing them their pieces.**

## Caching, in one paragraph

Each part can produce a short fingerprint of its settings. The toolkit uses these
fingerprints to avoid repeating expensive work — if you've already read and
embedded a document with the same settings, it reuses the result instead of doing
it again. You mostly won't think about this; it just makes re-runs fast. The one
rule that affects you: **if you write your own part and change how it behaves,
bump its `version`** so old cached results are correctly thrown away. (More in
[Part 09](09-extending-and-testing.md).)

## Nothing loads all at once

Reading and indexing happen a piece at a time — one page, one batch of chunks —
so a 2,000-page PDF never sits in memory all at once. You don't have to do
anything to get this; it's how the parts are built. It's why the toolkit can
handle large documents on an ordinary machine.

## What you now know

- A pipeline is a line of swappable **parts**, each with a **kind** and a
  **name**.
- Parts stay independent because they only exchange fixed **data shapes** — which
  is what makes swapping safe.
- Parts combine two ways: **chains** (steps in a row) and **wrapping** (a part
  made of parts).
- Your documents live in one **`ChunkIndex`**; simple parts are built by name,
  live parts by handing them their pieces.

Next: **[03 · Data contracts](03-data-contracts.md)** — a closer look at the data
shapes that flow between parts, and how they let every answer trace back to its
source.
