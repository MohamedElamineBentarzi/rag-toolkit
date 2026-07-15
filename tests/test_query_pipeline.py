"""QueryPipeline: Query -> retrieve -> refine chain -> truncate to k."""
from rag_toolkit.core.contracts import Chunk, Query
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.indexing.chunk_index import ChunkIndex
from rag_toolkit.pipeline import QueryPipeline, TraceEvent
from rag_toolkit.refinement.base import Refiner
from rag_toolkit.retrieval.index_retriever import IndexRetriever
from rag_toolkit.storage.memory_store import MemoryVectorStore

_TEXTS = [
    "cats and dogs are common household pets",
    "quarterly financial revenue and profit report",
    "notes on the weather and the changing seasons",
]


def index_retriever():
    index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(dimensions=512))
    index.add([
        Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i, t in enumerate(_TEXTS)
    ])
    return IndexRetriever(index)


def test_query_returns_ranked_results():
    results = QueryPipeline(index_retriever()).query("financial revenue", k=2)
    assert results and results[0].chunk.id == "d:1"
    assert len(results) == 2


def test_accepts_a_string_or_a_query_object():
    pipeline = QueryPipeline(index_retriever())
    from_str = pipeline.query("financial revenue", k=1)
    from_obj = pipeline.query(Query(text="financial revenue"), k=1)
    assert [r.chunk.id for r in from_str] == [r.chunk.id for r in from_obj]


def test_tracing_hook_sees_retrieve_then_each_refiner():
    events: list[TraceEvent] = []
    QueryPipeline(index_retriever(), trace=events.append).query("cats", k=3)
    # Empty refine chain ⇒ just a retrieve.
    assert [e.stage for e in events] == ["retrieve"]


def test_empty_refine_chain_is_retrieve_then_truncate():
    retriever = index_retriever()
    piped = QueryPipeline(retriever).query("financial", k=1)
    direct = retriever.retrieve(Query(text="financial"), 1)
    assert [r.chunk.id for r in piped] == [r.chunk.id for r in direct]


def test_refiner_chain_is_applied_and_truncated():
    class _ReverseRefiner(Refiner):
        name = "reverse"

        def refine(self, query, candidates, k):
            # Deliberately reorder to prove the pipeline runs the refiner.
            return list(reversed(candidates))

    retriever = index_retriever()
    out = QueryPipeline(retriever, refine=[_ReverseRefiner()]).query("cats", k=3)
    baseline = retriever.retrieve(Query(text="cats"), 50)
    assert [r.chunk.id for r in out] == [r.chunk.id for r in reversed(baseline)][:3]


def test_refiner_chain_runs_in_order():
    stages: list[str] = []

    def make(tag):
        class _Tag(Refiner):
            name = tag

            def refine(self, query, candidates, k):
                stages.append(tag)
                return candidates
        return _Tag()

    QueryPipeline(index_retriever(), refine=[make("a"), make("b")]).query("x", k=2)
    assert stages == ["a", "b"]
