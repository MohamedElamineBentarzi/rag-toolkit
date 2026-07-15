# 07 · Writing the answer & citations

The final stage takes your question and the chunks that were found, and writes an
answer — one where every claim points back to the exact page it came from. This
page covers how answers get written, and how citations work (including one
behavior that surprises people).

## The generator

A **generator** turns a question plus a set of chunks into an `Answer`. You can
swap which one you use — a real AI model, or a no-model baseline — without
changing anything else.

Whatever generator you use, the citation machinery is the same and handled for
you: the chunks are numbered `[1] [2] [3] …`, the model is told to cite those
numbers, and the toolkit matches each number back to its source. Generators only
have to write the text; they never deal with citation bookkeeping.

## How citations actually work

The chunks handed to the generator are numbered in order and packed into the
prompt, each with its number:

```
[1] <text of the top chunk>
[2] <text of the second chunk>
[3] <text of the third chunk>
```

The model is instructed to answer using only these, and to cite with `[n]`. When
the answer comes back, the toolkit reads the `[n]` markers it used and keeps a
`Citation` for each — carrying the document and page range. That's the whole
trick: the number in the text and the number of the citation are the same number.

There's a character budget on how much context gets packed, so very long
candidate lists are trimmed to fit (keeping at least the top chunk).

### Why citations sometimes start at `[2]`

This trips people up, so: **it's correct, not a bug.** If the model's answer cites
`[2]` but not `[1]`, you'll see only the `[2]` citation — it used the
second-ranked chunk and not the first. The markers are deliberately **not
renumbered**, because `[2]` in the answer text must keep matching the citation
labeled `2`. If you want a tidy `[1], [2], [3]` for display, renumber the text and
the citations *together* in your own UI.

### Getting a filename and download link

A citation carries a `doc_id`, not a filename. To show something clickable, look
up the name and link from the `doc_id`:

```python
for c in answer.citations:
    name = rag.source_uri(c.doc_id)      # "report.pdf"
    link = rag.download_url(c.doc_id)     # a file:// path or a time-limited S3 link
    print(f"[{c.marker}] {name} p{c.page_start}-{c.page_end}")
```

This needs a `blob_store` on your pipeline (there has to be a stored original to
link to — see [Part 08](08-pipelines.md)). Without one, `rag.catalog` is `None`.

## The two built-in generators

### `ExtractiveGenerator` — no model, no key (the default)

This one uses no AI model at all: the "answer" is simply the single most relevant
passage, returned as-is with its citation. It's the default in `RagPipeline`, so
the toolkit works out of the box, and it makes the whole
question → answer → citation path testable with no network. It's also an honest
baseline to measure real models against.

```python
answer = rag.ask("What was Q3 revenue?")
print(answer.text)   # the top passage, verbatim, with a [1] marker
```

### `AnthropicGenerator` — Claude writes the answer (add-on `[anthropic]`)

This uses Claude to write a real, synthesized answer. It's told to answer *only*
from the provided chunks and to cite them with `[n]` markers, so every claim
stays grounded and traceable:

```python
from rag_blocks import AnthropicGenerator, Query
gen = AnthropicGenerator(model="claude-opus-4-8", max_tokens=1024)
answer = gen.generate(Query(text="What was Q3 revenue?"), scored_chunks)

print(answer.text)         # the written answer with [n] markers
print(answer.citations)    # one per marker used
print(answer.usage)        # tokens used, for cost tracking
```

The API key comes from `ANTHROPIC_API_KEY` (or config you pass), and is never
written to logs.

### `.complete` — a plain text helper

The Anthropic generator also exposes `complete(prompt) -> str`: a plain
prompt-in, text-out call with no chunk packing or citations. This is what the
query-reshaping retrievers from [Part 06](06-retrieval-and-refinement.md)
(`MultiQueryRetriever`, `HydeRetriever`) and the contextual enricher use:

```python
retriever = HydeRetriever(IndexRetriever(index), complete=gen.complete)
```

## Writing your own generator

If you plug in your own generator, it only needs to:

1. Return an `Answer` whose `text` is a string.
2. Cite only the chunks it was given, carrying each chunk's `doc_id` through.
3. Handle the empty case gracefully — return an answer with no citations, never
   crash, when there's no context.

The citation numbering and resolution are done for you. [Part 09](09-extending-and-testing.md)
shows how to verify a custom generator against the built-in contract test.

Next: **[08 · Pipelines](08-pipelines.md)** — how reading, searching, and
answering get wired into the two-call loop from Part 01.
