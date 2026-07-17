# rag-blocks — The Guide

This guide teaches you to use `rag-blocks` from the first `pip install` to a
tuned, production search-and-answer system. It's written for people who want to
*use* the toolkit — plain language, real examples, no assumed knowledge of other
libraries.

> Want the design reasoning behind how it's built — the trade-offs, the internal
> mechanics, the decisions considered and rejected? That lives in
> [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md). This guide stays focused on
> getting things done.

## The map

| Part | What it covers |
|---|---|
| [01 · Getting started](01-getting-started.md) | Install, the add-ons, and your first cited answer in five minutes. |
| [02 · Concepts](02-concepts-and-architecture.md) | The handful of ideas — swappable parts, data shapes, chains, and the index — that make everything else predictable. |
| [03 · The data shapes](03-data-contracts.md) | What's inside a `Source`, `Document`, `Chunk`, `Answer`, and `Citation` — and how every answer traces back to a page. |
| [04 · Reading files & cutting them up](04-ingestion-and-chunking.md) | Reading PDFs/Word/etc. (with OCR), and cutting documents into searchable pieces. |
| [05 · Making chunks searchable](05-representations-and-storage.md) | Search by meaning and by keyword, where the data lives, and the `ChunkIndex` that owns it all. |
| [06 · Finding the right chunks](06-retrieval-and-refinement.md) | Searching several ways at once, reshaping questions, and cleaning up results. |
| [07 · Writing the answer & citations](07-generation-and-citations.md) | Turning chunks into a written, cited answer — and how citations work. |
| [08 · Pipelines](08-pipelines.md) | Wiring it all with `RagPipeline`, keeping originals, caching, and download links. |
| [09 · Add your own part](09-extending-and-testing.md) | Write and register a custom part in five steps, and prove it with the contract tests. |
| [10 · Recipes](10-recipes.md) | Ten complete setups you can copy — from a 30-second test to hybrid, tuned production. |
| [11 · Evaluation & tuning](11-evaluation-and-tuning.md) | Measuring whether your pipeline is any good, searching for a better one, and the ways a number can lie to you. |

## Pick your path

- **"I just want it working."** → [Part 01](01-getting-started.md), then the
  closest [recipe](10-recipes.md).
- **"I'm building a production stack."** → Parts
  [01](01-getting-started.md), [05](05-representations-and-storage.md),
  [06](06-retrieval-and-refinement.md), [08](08-pipelines.md),
  [10](10-recipes.md).
- **"Is my pipeline any good, and which part is worth its cost?"** →
  [Part 11](11-evaluation-and-tuning.md).
- **"I want to add my own piece."** → [Part 09](09-extending-and-testing.md).
- **"I want to understand the whole thing."** → read it in order.

## The whole idea in one paragraph

You feed files into the toolkit, which reads them into clean text (keeping page
numbers), cuts them into passage-sized **chunks**, and stores those so they're
searchable — by meaning, by keyword, or both. When you ask a question, it finds
the most relevant chunks, optionally cleans up the results, and has a model write
an answer using only those chunks — handing you the answer *and* a citation to the
exact page behind every claim. Every piece of that — the reader, the cutter, the
search, the answer-writer — is a part you can swap without changing the rest.
