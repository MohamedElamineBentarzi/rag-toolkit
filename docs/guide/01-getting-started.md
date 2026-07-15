# 01 · Getting started

`rag-blocks` answers questions about your own documents, and every answer comes
with citations that point back to the exact file and page. This page takes you
from `pip install` to a working, cited answer in about five minutes — no API key,
no database, no configuration.

## The problem it solves

Say you have a folder of PDFs — company policies, product manuals, research
papers, contracts. You want to ask plain questions ("What's our refund window?")
and get a trustworthy answer that tells you *where it came from*, so you can
check it. Doing this by hand means reading everything. Doing it with a raw
language model means it might make things up. `rag-blocks` reads your documents,
finds the passages that actually answer the question, and asks a model to write
the answer *using only those passages* — then hands you the sources.

## Install

```bash
pip install rag-blocks
```

That's the whole core, and it works with no other downloads: it can read text and
markdown files, search them, and produce cited answers entirely on your machine.

Heavier features (reading PDFs, using real AI models, connecting to a database)
come as **optional add-ons**, so a plain install never pulls in gigabytes you
won't use. You add one only when you need it:

| Add-on | What it unlocks |
|---|---|
| `rag-blocks[docling]` | Read PDFs, Word, PowerPoint, Excel, HTML, and scanned images |
| `rag-blocks[sentence-transformers]` | High-quality AI embeddings and reranking |
| `rag-blocks[qdrant]` | Store your searchable data in a Qdrant database |
| `rag-blocks[anthropic]` | Write answers with Claude |
| `rag-blocks[minio]` | Keep original files in S3-compatible storage |

Install a few together: `pip install "rag-blocks[docling,sentence-transformers,anthropic]"`.
If you ever call a feature whose add-on isn't installed, you get a clear message
naming the exact one to install — never a confusing crash.

## Your first cited answer

This runs today, with nothing but the core installed. Point it at a text or
markdown file and ask:

```python
from rag_blocks import RagPipeline, Source

rag = RagPipeline()                                  # the all-local defaults
rag.index(Source.from_path("handbook.md"))           # read it and make it searchable
answer = rag.ask("How many vacation days do new employees get?")

print(answer.text)
```

Two methods carry the whole tool: **`index`** takes in your documents, and
**`ask`** answers questions about them. Everything else is a detail you can
ignore until you need it.

### The part that matters: you can trust the answer

The answer isn't just text — it carries its receipts. Every claim is tagged, and
each tag points to the source it came from:

```python
answer = rag.ask("How many vacation days do new employees get?")

print(answer.text)
# New employees receive 15 paid vacation days in their first year [1],
# increasing to 20 days after three years of service [2].

for c in answer.citations:
    print(f"[{c.marker}] {c.doc_id[:8]}…  pages {c.page_start}–{c.page_end}")
# [1] 3f9a2c81…  pages 4–4
# [2] 3f9a2c81…  pages 4–5
```

Those `[1]`/`[2]` markers in the text line up with the citations, and each
citation tells you the document and the page range. This is the core promise:
**you never have to take the answer on faith — you can go read the source.**

## Reading real PDFs

Swap in the PDF add-on and the exact same two calls work on a stack of PDFs:

```bash
pip install "rag-blocks[docling]"
```

```python
from pathlib import Path
from rag_blocks import RagPipeline, Source

rag = RagPipeline()
for pdf in Path("./policies").glob("*.pdf"):
    rag.index(Source.from_path(pdf))                 # each one is parsed and indexed

print(rag.ask("What is our data retention policy for customer records?").text)
```

Nothing about your code changed — you just fed it PDFs instead of a markdown
file. Reading, page tracking, and OCR for scanned pages all happen inside; you
still only call `index` and `ask`.

## From "it works" to production — change the parts, not your code

The all-local defaults are great for trying things and running tests, but they're
basic: the built-in search is a simple keyword-ish match, and the built-in
answer-writer just stitches together the passages it found. For real use you'll
want a proper AI embedding model, a database to hold your data, and a real model
to write the answers.

Here's the thing worth noticing — you upgrade every one of those by handing
different parts to the *same* `RagPipeline`. Your `index` and `ask` calls don't
change at all:

```python
from rag_blocks import (
    RagPipeline, ChunkIndex, Source,
    SentenceTransformerEmbedder, QdrantVectorStore, AnthropicGenerator,
)

rag = RagPipeline(
    chunk_index=ChunkIndex(
        store=QdrantVectorStore(url="http://localhost:6333", collection="policies"),
        dense=SentenceTransformerEmbedder(model="BAAI/bge-m3"),   # real embeddings
    ),
    generator=AnthropicGenerator(model="claude-opus-4-8"),        # Claude writes the answer
)

rag.index(Source.from_path("policies.pdf"))
print(rag.ask("Summarize the security requirements for vendors.").text)
```

You didn't rewire anything — you named better parts. That's the point of the
whole design: every piece is a part you can replace, and replacing one never
forces you to touch the rest. The rest of this guide is really just a tour of the
parts you can swap in.

## A taste of what's possible: search by meaning *and* by keyword

Searching by *meaning* (AI embeddings) is great for "explain the refund process,"
but it can miss exact terms like a product code or a legal clause number.
Searching by *keyword* nails exact terms but misses paraphrases. You usually want
both — and you turn that on by adding **one line**:

```python
from rag_blocks import ChunkIndex, QdrantVectorStore, SentenceTransformerEmbedder, BM25Index

index = ChunkIndex(
    store=QdrantVectorStore(url="http://localhost:6333", collection="policies"),
    dense=SentenceTransformerEmbedder(model="BAAI/bge-m3"),   # search by meaning
    lexical=BM25Index(),                                      # search by keyword  ← added
)
rag = RagPipeline(chunk_index=index)
```

With both present, the pipeline automatically runs each question through both
kinds of search and blends the results — so a question like *"what does clause
7.3 say about liability?"* is found by the exact "7.3" match **and** by the
meaning of "liability." You didn't write any blending logic; adding the keyword
line was the whole change.

## Where to go next

You now know the shape of the tool: `index` your documents, `ask` questions, get
cited answers, and swap in stronger parts as you grow. Pick your path:

- **Just want it running on your files?** → jump to
  [10 · Recipes](10-recipes.md) and copy the setup closest to yours.
- **Want to understand the pieces you're swapping?** → read on to
  [02 · Concepts](02-concepts-and-architecture.md).
- **Want to add your own piece** (a custom reader, a company embedding model)? →
  [09 · Extending](09-extending-and-testing.md).

Next: **[02 · Concepts](02-concepts-and-architecture.md)** — the handful of ideas
that make every part of the toolkit work the same way.
