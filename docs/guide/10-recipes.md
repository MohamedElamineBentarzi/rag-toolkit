# 10 · Recipes

Ten complete setups, each solving a real problem. **Every script is
self-contained** — copy one, run it, adapt it. Most run on the zero-setup stack
(no downloads, no API key — a hashing embedder, in-memory store, and a fake model
call where a real one would go), so you can see the shape working immediately; a
comment marks exactly where to drop in real backends. Two recipes show the real
production stack.

---

## 1 · The 30-second smoke test

*When you just want to confirm it works.*

```python
from rag_blocks import RagPipeline, Source

rag = RagPipeline()                                  # all defaults, no setup
rag.index(Source.from_bytes(
    b"# Notes\nThe project ships in March. The budget is 40k.\n", name="notes.md"))

print(rag.ask("when does it ship?", k=2).text)
```

*How it works: with no arguments, `RagPipeline` builds the all-local stack — an
in-memory index, the built-in hashing embedder, and the no-model answer builder.
No extras, no key.*

---

## 2 · Production: quality search + a real answer

*The setup most people end up with: real embeddings in a database, a reranker for
precision, and Claude writing the answer.*

```python
# pip install "rag-blocks[sentence-transformers,qdrant,anthropic,docling]"
# export ANTHROPIC_API_KEY=...   ;   run Qdrant at localhost:6333
from rag_blocks import (RagPipeline, SentenceTransformerEmbedder, QdrantVectorStore,
                         AnthropicGenerator, CrossEncoderReranker, Source)

rag = RagPipeline.dense(
    embedder=SentenceTransformerEmbedder(model="BAAI/bge-m3"),
    store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
    generator=AnthropicGenerator(model="claude-sonnet-5"),
    refine=[CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")],
)
rag.index(Source.from_path("report.pdf"))            # the [docling] add-on reads the PDF
print(rag.ask("What was Q3 revenue?", k=8).text)
```

*How it works: this is recipe 1 with four parts swapped out — the embedder, the
store, the generator, and one clean-up step. The `index`/`ask` shape didn't
change; only the parts did.*

---

## 3 · Meaning + keyword search together

*When exact terms matter (product codes, clause numbers) but you also want
meaning-based search. Give the index both, and hybrid search is automatic.*

```python
from rag_blocks import (RagPipeline, ChunkIndex, MemoryVectorStore, HashingEmbedder,
                         BM25Index, MarkdownChunker, HeadingEnricher, Source)

rag = RagPipeline(
    chunk_index=ChunkIndex(
        MemoryVectorStore(),          # → prod: QdrantVectorStore(url=..., collection="docs")
        dense=HashingEmbedder(),      # → prod: SentenceTransformerEmbedder(model="BAAI/bge-m3")
        lexical=BM25Index(),          # meaning + keyword → hybrid search, automatically
    ),
    chunker=MarkdownChunker(),
    enrich=[HeadingEnricher()],
)
rag.index(Source.from_bytes(
    b"# France\nParis is the capital of France.\n\n# Fruit\nBananas are yellow.\n",
    name="facts.md"))

answer = rag.ask("What is the capital of France?", k=3)
print(answer.text)
for c in answer.citations:
    print(f"  [{c.marker}] {c.doc_id[:12]}… p{c.page_start}-{c.page_end}")
```

*How it works: the index holds two search-forms (meaning + keyword), so
`RagPipeline` automatically searches both and blends the results — you never named
a retriever.*

---

## 4 · Ask the question several ways (RAG-fusion)

*When users phrase questions unpredictably. Rephrase the question a few ways,
search each, and blend — so a badly-worded question still finds the answer.*

```python
from rag_blocks import (RagPipeline, ChunkIndex, MemoryVectorStore, HashingEmbedder,
                         BM25Index, HybridRetriever, MultiQueryRetriever, Source)

def fake_complete(prompt: str) -> str:               # → prod: AnthropicGenerator(...).complete
    return "capital city of France\nseat of the French government\nParis France"

index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(), lexical=BM25Index())
retriever = MultiQueryRetriever(HybridRetriever(index), complete=fake_complete, n=3)
rag = RagPipeline(chunk_index=index, retriever=retriever)

rag.index(Source.from_bytes(b"# Geo\nParis is the capital of France.\n", name="geo.md"))
print(rag.ask("where is the French capital?", k=3).text)
```

*How it works: `MultiQueryRetriever` wraps the hybrid retriever. It asks a model
once for a few rephrasings, always keeps the original question, searches each, and
blends. In production, swap `fake_complete` for `AnthropicGenerator(...).complete`.*

---

## 5 · Search on a drafted answer (HyDE)

*When short questions don't match well. Have a model draft a hypothetical answer,
then search for passages similar to that draft — often a much better match.*

```python
from rag_blocks import (RagPipeline, ChunkIndex, MemoryVectorStore, HashingEmbedder,
                         IndexRetriever, HydeRetriever, Source)

def fake_complete(prompt: str) -> str:               # → prod: AnthropicGenerator(...).complete
    return "Paris is the capital city of France and the seat of its government."

index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
retriever = HydeRetriever(IndexRetriever(index), complete=fake_complete)
rag = RagPipeline(chunk_index=index, retriever=retriever)

rag.index(Source.from_bytes(b"# Geo\nParis is the capital of France.\n", name="geo.md"))
print(rag.ask("French capital?", k=1).text)
```

*How it works: `HydeRetriever` wraps a plain retriever. A full, keyword-rich
drafted passage tends to match stored passages better than a terse question does.*

---

## 6 · Search small, answer big

*When you want precise search hits but rich context for the answer. Index tiny
chunks so search is sharp, then pull in the surrounding text before answering.*

```python
from rag_blocks import (ChunkIndex, MemoryVectorStore, HashingEmbedder, IndexingPipeline,
                         QueryPipeline, IndexRetriever, NeighborExpander, FixedChunker, Source)

index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
for _ in IndexingPipeline(
        chunker=FixedChunker(chunk_chars=120, overlap_chars=0),   # small chunks
        sinks=[index]).index(Source.from_bytes(
            b"Alpha facts here. " * 8 + "Beta facts here. " * 8 + "Gamma facts here. " * 8,
            name="doc.txt")):
    pass

qp = QueryPipeline(IndexRetriever(index), refine=[NeighborExpander(index, window=1)])
top = qp.query("beta", k=1)[0]
print("expanded:", top.chunk.metadata.get("expanded"), "| chars:", len(top.chunk.text))
```

*How it works: `NeighborExpander` is a clean-up step that pulls the neighboring
chunks around each hit and stitches them together — so the answer sees a coherent
window, not a 120-character fragment.*

---

## 7 · One question across two document sets

*When your data lives in separate collections — say legal docs and HR docs — and a
question should search both at once.*

```python
from rag_blocks import (RagPipeline, ChunkIndex, MemoryVectorStore, HashingEmbedder,
                         IndexingPipeline, FusionRetriever, IndexRetriever, Source)

legal = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
hr = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
for _ in IndexingPipeline(sinks=[legal]).index(
        Source.from_bytes(b"# Policy\nContracts require two signatures.\n", name="legal.md")): pass
for _ in IndexingPipeline(sinks=[hr]).index(
        Source.from_bytes(b"# Leave\nEmployees get 25 vacation days.\n", name="hr.md")): pass

retriever = FusionRetriever([IndexRetriever(legal), IndexRetriever(hr)])
rag = RagPipeline(chunk_index=legal, retriever=retriever)   # legal is the default write target

print(rag.ask("how many vacation days?", k=2).text)         # answered from the HR set
```

*How it works: `FusionRetriever` searches both indexes and blends the results —
two separate collections, one question, one ranking.*

---

## 8 · Make chunks findable with added context

*A common problem: a chunk deep in a document says "revenue rose 18%" but never
says which company or quarter — so a search for "Q3 Acme revenue" misses it.
Enrichment fixes this by adding the missing context to each chunk.*

```python
from rag_blocks import RagPipeline, HeadingEnricher, MarkdownChunker, Source

rag = RagPipeline(
    chunker=MarkdownChunker(),
    enrich=[HeadingEnricher()],   # prepend each chunk's section heading
)                                 # → for AI-written context, add ContextualEnricher(model=...)
rag.index(Source.from_bytes(
    b"# Q3 Results\nRevenue rose 18 percent this quarter.\n", name="report.md"))

print(rag.ask("how did Q3 revenue do?", k=1).text)
```

*How it works: `HeadingEnricher` prepends the section heading ("Q3 Results") to
each chunk, so a mid-section chunk becomes findable by its section. For richer,
AI-written context, add `ContextualEnricher(model="claude-opus-4-8")` — see the
enrichment section below.*

---

## 9 · The toolkit's search + your own graph search

*When you already have a graph database (or any other search system) and want to
blend its results with the toolkit's — without modifying the toolkit.*

```python
from rag_blocks import (RagPipeline, ChunkIndex, MemoryVectorStore, HashingEmbedder,
                         FusionRetriever, HybridRetriever, Retriever, Query, ScoredChunk, Source)

class MyGraphIndex:                          # a write target: just add() + persist()
    def __init__(self): self.chunks = []
    def add(self, chunks): self.chunks.extend(chunks)
    def persist(self): pass

class MyGraphRetriever(Retriever):           # a search over the graph
    name = "graph"
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
    def retrieve(self, query: Query, k: int = 20):
        q = set(query.text.lower().split())
        hits = [c for c in self.graph.chunks if q & set(c.text.lower().split())]
        return [ScoredChunk(chunk=c, score=1.0, retriever_name=self.name) for c in hits[:k]]

index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
graph = MyGraphIndex()
rag = RagPipeline(
    chunk_index=index,
    extra_sinks=[graph],                     # indexing writes to the graph too
    retriever=FusionRetriever([HybridRetriever(index), MyGraphRetriever(graph)]),
)
rag.index(Source.from_bytes(b"# Notes\nGraphs connect related facts together.\n", name="notes.md"))

print(rag.ask("what connects facts?", k=2).text)
```

*How it works: your graph plugs into both sides — as an extra write target (any
object with `add`/`persist`), and as a retriever blended in with `FusionRetriever`.
You didn't change anything inside the toolkit; you added two small classes.*

---

## 10 · Try many strategies without re-indexing

*When you're tuning: index your documents once, then compare retrieval and
clean-up strategies against the same stored data.*

```python
from rag_blocks import (ChunkIndex, MemoryVectorStore, HashingEmbedder, BM25Index,
                         IndexingPipeline, QueryPipeline, IndexRetriever, HybridRetriever,
                         KeywordRefiner, Source)

index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(), lexical=BM25Index())
for _ in IndexingPipeline(sinks=[index]).index(          # index ONCE
        Source.from_bytes(b"# Geo\nParis is the capital of France.\n", name="geo.md")):
    pass

for retriever in [IndexRetriever(index, "dense"),
                  IndexRetriever(index, "lexical"),
                  HybridRetriever(index)]:
    for clean_up in ([], [KeywordRefiner()]):
        hits = QueryPipeline(retriever, refine=clean_up, fetch_k=10).query("French capital", k=3)
        print(f"{retriever.label:16} clean_up={len(clean_up)} -> {hits[0].chunk.id if hits else None}")
```

*How it works: six strategies (three retrievers × two clean-up chains) compared
with a single indexing pass. In a real evaluation you'd replace `print` with a
score against a set of test questions — which is exactly what the upcoming
auto-tuner automates.*

Two more things worth knowing:

- **Connect to an existing collection.** Point a `ChunkIndex` at an
  already-populated Qdrant collection and it validates the shape and starts
  searching — no re-indexing.
- **Search without answering.** A `QueryPipeline` on its own is a complete search
  system; you don't need the full `RagPipeline` if you only want ranked chunks.

---

## More on enrichment

Enrichers run between cutting and storing. Each one receives the chunks *and* the
whole document (context is exactly what a lone chunk is missing), and returns
chunks — with text added, with new chunks generated, or unchanged. They run as a
list (`enrich=[...]`), in order.

Adding text to a chunk is expected and fine — the chunk is no longer a word-for-word
slice of the document, but its page info is kept, so citations still work.

### `HeadingEnricher` — add the section heading (no downloads)

For each chunk, it finds the nearest heading above it and prepends it (unless the
chunk already starts with it). A chunk under `## Q3 Results` now carries "Q3
Results," so a question about Q3 finds it even if the body never repeats the
phrase. Fast, deterministic, no dependencies.

### `ContextualEnricher` — AI-written context (add-on `[anthropic]`)

For each chunk, it asks Claude for a one-sentence summary that situates the chunk
in its document, and prepends that. A model writes a better situating sentence
than a bare heading, at the cost of one call per chunk:

```python
from rag_blocks import RagPipeline, ContextualEnricher, MarkdownChunker, Source
# export ANTHROPIC_API_KEY=...
rag = RagPipeline(chunker=MarkdownChunker(),
                  enrich=[ContextualEnricher(model="claude-opus-4-8")])
rag.index(Source.from_bytes(b"# Q3\nRevenue rose 18 percent.\n", name="report.md"))
print(rag.ask("Q3 revenue?", k=1).text)
```

---

## Where to go next

- The one-page cheat sheet: [`../GUIDE.md`](../GUIDE.md).
- The design reasoning and internals: [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

You now know how to use rag-blocks across the board — from a 30-second test to a
tuned, hybrid, cited production system. Go build something.
