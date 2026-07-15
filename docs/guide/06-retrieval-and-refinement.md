# 06 · Finding the right chunks

When you ask a question, two things happen: the toolkit **searches** for relevant
chunks, then **cleans up** the results before answering. This page covers both —
including the powerful moves like "ask the question several ways at once" and
"search a small piece but answer with the surrounding context."

```
question → search (find candidates) → clean up (re-rank, expand, trim) → top k
```

## Searching: the retriever

A **retriever** takes a question and returns the most relevant chunks. It reads
from a `ChunkIndex` (which you built in [Part 05](05-representations-and-storage.md)).
Retrievers only *read* — the searchable data was built once at index time, and a
retriever is a cheap, swappable way to query it.

### `IndexRetriever` — search one way

The basic retriever searches your index one way (by meaning, or by keyword):

```python
from rag_blocks import IndexRetriever
IndexRetriever(index)                          # if the index has just one search-form
IndexRetriever(index, representation="lexical")  # pick keyword search explicitly
```

If your index only has one search-form, you don't even name it. If it has
several, name the one you want.

### `HybridRetriever` — search several ways and blend (the common upgrade)

Usually you want both meaning *and* keyword search. `HybridRetriever` runs each
and blends the results into one ranking:

```python
from rag_blocks import HybridRetriever
HybridRetriever(index)                              # blend ALL the index's search-forms
HybridRetriever(index, representations=["dense", "lexical"])
```

That's the whole setup for hybrid search — you don't write any blending logic.
A question like *"what does clause 7.3 say about liability?"* gets found by the
exact "7.3" match **and** by the meaning of "liability," and the two rankings
merge.

### `FusionRetriever` — blend across *different* indexes

Fusion isn't limited to one index. You can blend results from **separate**
indexes — say, a legal-documents index and an HR-documents index — into one
ranking. This is how you search multiple collections at once:

```python
from rag_blocks import FusionRetriever, IndexRetriever
FusionRetriever([IndexRetriever(legal_index), IndexRetriever(hr_index)])
```

### How blending works (and why it ignores raw scores)

Different searches produce scores on totally different scales — a meaning-search
score and a keyword score aren't comparable numbers. So blending uses **rank
position**, not raw score: a chunk that comes 1st in one search and 3rd in
another gets credit for both placements. This method (called Reciprocal Rank
Fusion) is robust exactly because it never tries to average incomparable numbers.

Two things you can rely on when searches are blended:
- The **same chunk found by two searches merges into one result** (its credit
  adds up), it's never duplicated.
- Each result records **which searches found it**, so you can see whether a hit
  came from meaning, keyword, or both.

## Asking the question several ways (optional, needs a model)

Sometimes the best way to improve recall is to reshape the *question* before
searching. Two ready-made wrappers do this, and both are just retrievers that wrap
another retriever — you don't add a new stage:

```python
from rag_blocks import MultiQueryRetriever, HydeRetriever, HybridRetriever, IndexRetriever, AnthropicGenerator
gen = AnthropicGenerator(model="claude-sonnet-5")

# Ask a model to rephrase the question a few ways, search each, blend them:
MultiQueryRetriever(HybridRetriever(index), complete=gen.complete, n=4)

# Ask a model to draft a hypothetical answer, then search using THAT:
HydeRetriever(IndexRetriever(index), complete=gen.complete)
```

- **`MultiQueryRetriever`** turns one question into several phrasings (always
  keeping the original), searches each, and blends — so a badly-worded question
  still finds the right passages. It calls the model once per search.
- **`HydeRetriever`** asks a model to write a passage that *would* answer the
  question, then searches for chunks similar to that draft — which often matches
  better than the short question does.

Both need a way to call a language model, which you pass as `complete=...`. Any
model works; `AnthropicGenerator.complete` (see [Part 07](07-generation-and-citations.md))
supplies one, and a fake function makes them easy to test without a network.

## Cleaning up: the refine chain

Raw search results are rarely ready to answer with — you may want to re-rank them
more precisely, pull in surrounding context, or drop weak matches. You do this
with a **refine chain**: a list of clean-up steps, each taking the current list of
chunks and returning a new one. They run left to right:

```python
from rag_blocks import NeighborExpander, CrossEncoderReranker, ScoreThreshold
refine = [
    NeighborExpander(index, window=2),                      # add surrounding context
    CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3"),  # re-rank precisely
    ScoreThreshold(min_score=0.2),                          # drop the weak tail
]
```

An empty list (`refine=[]`) just means "no clean-up" — there's no special
do-nothing step to configure.

### The built-in clean-up steps

- **`CrossEncoderReranker`** (add-on `[sentence-transformers]`) — the big
  quality win. The initial search compares question and chunks separately, which
  is fast but rough. A reranker reads the question and each chunk *together* and
  scores how well they actually match — far more accurate. Put it early in the
  chain to fix the ordering.
- **`ScoreThreshold`** — drops chunks below a score, so weak matches don't
  pollute the answer. Search always returns a full batch whether or not anything
  is truly relevant; this floors it. Best placed *after* a reranker, so it judges
  on the reranker's better scores.
- **`KeywordRefiner`** — a lightweight, no-download re-ranker based on shared
  words. Useful as a simple reorder when you don't want to load a model.
- **`NeighborExpander`** — the clever one, below.

### `NeighborExpander` — search small, answer big

A great technique: **index small chunks** so search is precise, then **answer with
a bigger window** around each hit so the model gets coherent context. The expander
does exactly this — for each chunk it finds, it pulls in the neighboring chunks
(the pieces just before and after) and stitches them together:

```python
from rag_blocks import QueryPipeline, IndexRetriever, NeighborExpander, CrossEncoderReranker

# index with small pieces (e.g. FixedChunker(chunk_chars=400)), answer with context:
QueryPipeline(IndexRetriever(index), refine=[
    NeighborExpander(index, window=2),                      # ±2 neighbors
    CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3"),
])
```

It's careful about the details for free: because every chunk knows its exact
character range, overlapping neighbors are joined **once** (no duplicated
sentences), and generated/synthetic chunks are skipped (a summary isn't a
neighbor of anything).

## Putting search + clean-up together

The `QueryPipeline` wires it all: search a generous number of candidates, run the
refine chain, then trim to the number you want:

```python
from rag_blocks import QueryPipeline, HybridRetriever, CrossEncoderReranker
qp = QueryPipeline(
    HybridRetriever(index),
    refine=[CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")],
    fetch_k=50,        # pull 50 candidates…
)
hits = qp.query("what was Q3 revenue?", k=8)   # …clean up, return the best 8
```

Fetching more than you need (`fetch_k=50`) and then trimming to `k=8` gives the
clean-up steps room to work — the reranker can promote a chunk that the initial
search ranked 30th.

Next: **[07 · Generation & citations](07-generation-and-citations.md)** — turning
those chunks into a written, cited answer.
