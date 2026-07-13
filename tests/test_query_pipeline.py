"""QueryPipeline: Query -> retrieve -> rerank -> ScoredChunks."""
from rag_toolkit.core.contracts import Chunk, Query
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.pipeline import QueryPipeline, TraceEvent
from rag_toolkit.reranking.base import Reranker
from rag_toolkit.retrieval.dense import DenseRetriever
from rag_toolkit.storage.memory_store import MemoryVectorStore

_TEXTS = [
    "cats and dogs are common household pets",
    "quarterly financial revenue and profit report",
    "notes on the weather and the changing seasons",
]


def dense_retriever():
    embedder = HashingEmbedder(dimensions=512)
    store = MemoryVectorStore()
    chunks = [
        Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i, t in enumerate(_TEXTS)
    ]
    store.upsert(chunks, embedder.embed_texts([c.text for c in chunks]))
    return DenseRetriever(embedder=embedder, store=store)


def test_query_returns_ranked_results():
    results = QueryPipeline(dense_retriever()).query("financial revenue", k=2)
    assert results and results[0].chunk.id == "d:1"
    assert len(results) == 2


def test_accepts_a_string_or_a_query_object():
    pipeline = QueryPipeline(dense_retriever())
    from_str = pipeline.query("financial revenue", k=1)
    from_obj = pipeline.query(Query(text="financial revenue"), k=1)
    assert [r.chunk.id for r in from_str] == [r.chunk.id for r in from_obj]


def test_tracing_hook_sees_retrieve_and_rerank():
    events: list[TraceEvent] = []
    QueryPipeline(dense_retriever(), trace=events.append).query("cats", k=3)
    assert [e.stage for e in events] == ["retrieve", "rerank"]


def test_default_reranker_is_noop_passthrough():
    # With the Null Object reranker, top-k is just the retriever's top-k.
    retriever = dense_retriever()
    piped = QueryPipeline(retriever).query("financial", k=1)
    direct = retriever.retrieve(Query(text="financial"), 1)
    assert [r.chunk.id for r in piped] == [r.chunk.id for r in direct]


def test_custom_reranker_is_applied():
    class _ReverseReranker(Reranker):
        name = "reverse"

        def rerank(self, query, candidates, top_k):
            # Deliberately reorder to prove the pipeline uses the reranker.
            return list(reversed(candidates))[:top_k]

    retriever = dense_retriever()
    out = QueryPipeline(retriever, reranker=_ReverseReranker()).query("cats", k=3)
    baseline = retriever.retrieve(Query(text="cats"), 50)
    assert [r.chunk.id for r in out] == [r.chunk.id for r in reversed(baseline)][:3]
