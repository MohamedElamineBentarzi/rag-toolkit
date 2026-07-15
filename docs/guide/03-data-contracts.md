# 03 · The data shapes

Parts of the toolkit hand data to each other in a few fixed shapes. You don't
have to memorize them, but knowing what's in each one helps you read results —
especially answers and citations. This page is a plain tour of every shape, in
the order data flows:

```
Source → Page → Document → Chunk → (search) → ScoredChunk → Answer (+ Citations)
```

All of these are simple Python data objects. Each also carries a `metadata`
dictionary you can stash extra fields in, so you're never blocked waiting for a
new field to be added.

> Curious *why* they're built this way — immutability, the provenance design, the
> "a chunk never stores its own vectors" rule? That reasoning is in
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md). Here we just cover what each shape
> holds and what you do with it.

## `Source` — a pointer to one input file

A `Source` points at something to read. It doesn't load the file up front — it
opens it only when needed, so pointing at a huge file is cheap.

```python
from rag_blocks import Source

Source.from_path("report.pdf")                       # a file on disk
Source.from_bytes(b"# Notes\n...", name="notes.md")  # data you already have in memory
```

That's almost all you do with a `Source` directly: create it and hand it to
`index`. Behind the scenes it also computes a content fingerprint, which the
toolkit uses to skip re-reading a file it has already processed.

## `Page` — one page of a document

As a file is read, it comes out one page at a time. Each page is:

```python
Page:
    number: int          # 1-based, the way a PDF viewer counts
    markdown: str         # the page's text, as clean markdown
    ocr_applied: bool     # True if this page was read by OCR (a scan)
```

Reading page-by-page is what keeps memory low on big documents. `ocr_applied`
flags pages that came from a scan, since scanned text can be lower quality than a
real text layer.

## `Document` — the whole file, assembled

The pages get joined into one `Document`:

```python
Document:
    id: str               # a fingerprint of the file's contents (the "doc_id")
    markdown: str          # every page joined together
    pages: list            # where each page sits in the text (for page numbers)
    source_uri: str        # the original file name
```

The `id` is a fingerprint of the file's *contents*. Two identical files get the
same `id`, so re-indexing the same document updates it in place instead of
creating a duplicate. That same `id` is what a citation later points back to.

The one thing a `Document` can tell you: which pages a stretch of text came from
— which is how a chunk knows its page range.

## `Chunk` — a searchable piece of a document

A document is cut into `Chunk`s — passage-sized pieces that are what actually
gets searched:

```python
Chunk:
    id: str               # "{doc_id}:{position}", unique and repeatable
    doc_id: str           # which document it came from
    text: str             # the passage
    index: int            # its position in the document (0, 1, 2, …)
    char_start, char_end  # where in the document's text this piece sits
    page_start, page_end  # which pages that maps to
```

The `char_*` and `page_*` fields are the piece's **provenance** — its exact
origin in the source. This is the whole reason answers can cite a page: the trail
runs `Source → Page → Document → Chunk → page number`, unbroken.

Two things you can count on:
- For a freshly cut chunk, `text` is exactly the slice of the document between
  `char_start` and `char_end`.
- `index` counts 0, 1, 2, … with no gaps, so the toolkit can pull a chunk's
  neighbors ("give me the piece just before and after this one") when it needs
  surrounding context.

The `page_*` fields can be empty in one case only: **synthetic chunks** — pieces
a model *generated* (a summary, a Q&A pair) rather than cut from the document.
Those have no page because they came from no single spot. Any chunk cut from a
real document always has its pages filled in. (See
[Part 04](04-ingestion-and-chunking.md).)

## `Query` — a question, with optional scoping

```python
Query:
    text: str              # what you're asking
    filters: dict          # optional: limit the search, e.g. {"doc_id": "abc"}
```

`filters` lets you narrow a search. A plain value means "must equal this"; a list
means "must be one of these":

```python
{"doc_id": "abc"}              # only this document
{"doc_id": "abc", "index": [3, 4, 5]}   # only these pieces of it
```

You can usually just pass a plain string to `ask` and skip `Query` entirely — the
toolkit wraps it for you.

## `ScoredChunk` — a chunk plus how relevant it is

Search returns chunks paired with a relevance score:

```python
ScoredChunk:
    chunk: Chunk
    score: float           # higher = more relevant
    retriever_name: str    # which search produced it (when several are blended)
```

The only promise about `score` is **higher means more relevant** — the exact
number depends on the search method, so always sort by it rather than reading it
as a percentage.

## `Citation` — a source reference in an answer

```python
Citation:
    marker: int            # the [1], [2] used inline in the answer text
    chunk_id: str           # the exact piece the claim came from
    doc_id: str             # the document it belongs to
    page_start, page_end    # the pages
```

A `[2]` in an answer resolves to exact pages of an exact document. The
human-readable file name and a download link aren't stored here — you look them
up from the `doc_id` (`rag.source_uri(doc_id)`, `rag.download_url(doc_id)`), which
[Part 08](08-pipelines.md) covers.

## `Answer` — the final result

```python
Answer:
    text: str              # the written answer, with [n] markers
    citations: list        # one Citation per marker
    usage: dict            # tokens/timing (when an AI model wrote it)
```

Every answer can be traced back to the exact passages behind it. That's the
point: you get the answer *and* the evidence.

## Searchable forms of a chunk (dense and sparse)

One more small set of shapes shows up when you store chunks for search. A chunk
can be turned into a **dense vector** (a list of numbers that captures meaning) or
a **sparse vector** (a short list of important terms and weights — the SPLADE
style):

```python
SparseVector:
    indices: tuple[int, ...]     # which terms
    values: tuple[float, ...]    # their weights

VectorSpec:
    name: str                    # the name of this searchable form, e.g. "dense"
    kind: "dense" | "sparse"
    dimensions: int              # dense only: how many numbers
    distance: str                # dense only: how similarity is measured
```

The key idea: a chunk can have **several searchable forms at once**, each with a
name. These forms live in the store, keyed by the chunk's id — never on the chunk
itself. That way you can re-search or re-embed your corpus without ever changing
the chunks. [Part 05](05-representations-and-storage.md) is all about this.

Next: **[04 · Ingestion & chunking](04-ingestion-and-chunking.md)** — how a file
becomes a `Document` and then `Chunk`s, and how to control the cutting.
