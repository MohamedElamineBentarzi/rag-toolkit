# The baseline benchmark

    python benchmarks/baseline/run.py            # the full grid
    python benchmarks/baseline/run.py --quick    # a 3-trial smoke run

**What this is:** the committed regression baseline. Every milestone after v0.8
reruns *this exact config* with one component swapped, and the measured delta is
that release's headline number ("late chunking: +X nDCG for +Y ms/doc"). It is
cheap to keep and it compounds: a number is only meaningful next to the number
it replaced.

**What this is not:** a public benchmark, or evidence about anything but itself.

## Read this before quoting a number

- **N is small.** Four documents, 28 questions. Numbers here are *indicative*,
  not authoritative. A 0.02 nDCG difference on 28 questions is noise wearing a
  decimal point. Treat this as a regression detector — "did this change make
  retrieval worse?" — not as a leaderboard entry.
- **The corpus is synthetic.** Written for this repo, committed under its
  licence. Real corpora are messier: scanned pages, tables, inconsistent
  headings, and the vocabulary mismatch between how people ask and how
  documents are written. Do not conclude a chunker is good because it wins here.
- **The default stack is deliberately dumb.** `HashingEmbedder` is a hashing
  trick, not a semantic model — it cannot tell that "retake" and "resit" mean
  the same thing. That is the point: the benchmark runs in CI with no vendor, no
  key, and no network, so it can run on every PR forever. A real embedder will
  score better on the same questions; measure that, don't assume it.
- **Cost numbers are wall-clock on whatever machine ran it.** Compare them
  within a run, never across machines. Read `index_ms` next to `cache_hits` —
  within a grid the first trial pays for the parse and the rest inherit it, so
  `index_ms` partly measures running order (DR-0003 §3). `query_ms` is the
  honest one.

## How it is labeled, and why

Ground truth is **document-level** (`relevant_doc_ids`), not chunk-level.

`Chunk.id` is `{doc_id}:{index}`, so a chunk id denotes a *different passage*
under a different chunker — and may not exist at all under a coarser one. A
chunk-level label would therefore make chunk size the one knob you cannot
measure, which is precisely the knob ARCHITECTURE §6.4 leads with. Nothing
detects that mismatch for you: the score stays plausible and becomes wrong.

`qa.jsonl` names documents by **filename**; `run.py` resolves those to real
`doc_id`s (content hashes) at load time. Hardcoding hashes would rot the first
time anyone fixed a typo in the corpus.

**What document-level labels cannot see**, and it is worth knowing before you
add an axis: a refiner that only reorders *within* a document is invisible here.
Retrieved chunks are deduplicated to their documents, so `neighbor-expander`
— which pulls in a hit's neighbours from the same document — measured exactly
`+0.0000` quality for `+2.1 ms`. That is a true statement about doc-level
recall, not a broken refiner; it simply cannot be scored at this granularity.
Refiners that reorder *across* documents (`keyword`, and a cross-encoder later)
show up fine. If you need to measure intra-document reordering, you need
chunk-level labels — and then you cannot tune the chunker. That trade is the
whole reason `EvalSample` carries both fields.

## Files

| File | What it is |
|---|---|
| `corpus/*.md` | Four documents: a programme handbook, a quarterly report, a warehouse manual, a security policy. |
| `qa.jsonl` | 28 questions with document-level ground truth and a reference answer. |
| `config.json` | The committed grid. JSON, not YAML — `rag_blocks.core` is stdlib-only, and a config format must not be the thing that drags in a dependency. |
| `run.py` | The runner. Prints the leaderboard and the per-stage marginals. |

The questions deliberately share vocabulary across documents ("review",
"report", "quarterly", "three") so retrieval can actually be *wrong*. An earlier
draft used an easier corpus where every configuration scored a perfect 1.0 —
a benchmark that cannot fail is a benchmark that ranks noise.

## Adding a question

Append to `qa.jsonl`:

```json
{"question": "...", "relevant_docs": ["quarterly-report.md"], "reference_answer": "..."}
```

`relevant_docs` are filenames under `corpus/`; `run.py` fails loudly if one
doesn't exist. Prefer questions that are answerable from exactly one document
and phrased the way a person would ask, not the way the document is written.
