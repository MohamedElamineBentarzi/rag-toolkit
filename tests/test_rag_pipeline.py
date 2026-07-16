"""RagPipeline: the whole loop end to end, zero dependencies."""
from rag_blocks.chunking.markdown import MarkdownChunker
from rag_blocks.core.contracts import Answer, Query, Source
from rag_blocks.core.errors import ConfigError
from rag_blocks.embedding.hashing import HashingEmbedder
from rag_blocks.generation.extractive import ExtractiveGenerator
from rag_blocks.indexing.chunk_index import ChunkIndex
from rag_blocks.pipeline import RagPipeline, TraceEvent
from rag_blocks.retrieval.hybrid import HybridRetriever
from rag_blocks.retrieval.index_retriever import IndexRetriever
from rag_blocks.storage.bm25_index import BM25Index
from rag_blocks.storage.memory_store import MemoryVectorStore

_CORPUS = "# France\nParis is the capital of France.\n\n# Fruit\nBananas are yellow.\n"


def source():
    return Source.from_bytes(_CORPUS.encode(), name="facts.md")


def test_ask_traces_generation_the_stage_where_the_money_goes():
    # Generation was the one stage invisible to tracing — and the expensive
    # one. Without this event a trial's cost is silently missing its bill.
    events: list[TraceEvent] = []
    rag = RagPipeline(chunker=MarkdownChunker(), trace=events.append)
    rag.index(source())
    rag.ask("What is the capital of France?", k=2)

    generate = [e for e in events if e.stage == "generate"]
    assert len(generate) == 1
    assert generate[0].duration_ms >= 0
    assert generate[0].detail["generator"] == "extractive"
    assert generate[0].detail["context_chunks"] == 2
    assert generate[0].source_uri == "What is the capital of France?"


def test_the_generate_event_carries_usage_so_cost_needs_no_answer():
    # ExtractiveGenerator is free and reports {} — an empty bill, not a
    # missing one. A collector must be able to price a trial from the trace
    # alone, without holding on to the Answer.
    events: list[TraceEvent] = []
    rag = RagPipeline(chunker=MarkdownChunker(), trace=events.append)
    rag.index(source())
    rag.ask("bananas", k=1)

    generate = next(e for e in events if e.stage == "generate")
    assert generate.detail["usage"] == {}


def test_ask_with_context_returns_both_halves_and_still_traces():
    # The seam evaluation needs: scoring retrieval AND generation from one run
    # requires both halves. Hand-rolling it (query() + generate()) skips the
    # "generate" event, so a trial silently under-reports the costly stage.
    events: list[TraceEvent] = []
    rag = RagPipeline(chunker=MarkdownChunker(), trace=events.append)
    rag.index(source())
    events.clear()

    answer, context = rag.ask_with_context("What is the capital of France?", k=2)

    assert isinstance(answer, Answer) and "Paris" in answer.text
    assert len(context) == 2
    assert all(sc.chunk.id for sc in context)
    assert [e.stage for e in events] == ["retrieve", "generate"]


def test_ask_delegates_to_ask_with_context():
    # One implementation, two views of it — `ask` must not drift into a second
    # copy of the read path.
    rag = RagPipeline(chunker=MarkdownChunker())
    rag.index(source())

    plain = rag.ask("bananas", k=1)
    answer, _ = rag.ask_with_context("bananas", k=1)
    assert plain.text == answer.text


def test_a_full_ask_traces_every_stage_of_the_read_path():
    events: list[TraceEvent] = []
    rag = RagPipeline(chunker=MarkdownChunker(), trace=events.append)
    rag.index(source())
    events.clear()  # drop the write path
    rag.ask("bananas", k=1)

    assert [e.stage for e in events] == ["retrieve", "generate"]


def test_index_then_ask_returns_a_grounded_answer():
    rag = RagPipeline(chunker=MarkdownChunker())
    rag.index(source())

    answer = rag.ask("What is the capital of France?", k=1)
    assert isinstance(answer, Answer)
    assert "Paris" in answer.text
    # The citation resolves back to a real chunk with page provenance.
    assert answer.citations
    assert answer.citations[0].page_start is not None


def test_accepts_a_string_or_query():
    rag = RagPipeline(chunker=MarkdownChunker())
    rag.index(source())
    from_str = rag.ask("capital of France", k=1)
    from_obj = rag.ask(Query(text="capital of France"), k=1)
    assert from_str.text == from_obj.text


def test_ask_before_indexing_is_graceful():
    answer = RagPipeline().ask("anything at all")
    assert isinstance(answer, Answer)
    assert answer.citations == []  # nothing indexed ⇒ no sources


def test_index_populates_the_chunk_index():
    rag = RagPipeline(chunker=MarkdownChunker())
    rag.index(source())
    # Two headings ⇒ two chunks written into the (default) ChunkIndex.
    hits = rag.chunk_index.search("dense", "fruit", k=10)
    assert len(hits) == 2


def test_default_retriever_is_derived_from_one_representation():
    assert isinstance(RagPipeline().retriever, IndexRetriever)


def test_default_retriever_is_hybrid_for_multi_representation():
    index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(),
                       lexical=BM25Index())
    rag = RagPipeline(chunk_index=index)
    assert isinstance(rag.retriever, HybridRetriever)


def test_dense_convenience_constructor():
    rag = RagPipeline.dense(embedder=HashingEmbedder(), chunker=MarkdownChunker())
    rag.index(source())
    assert "Paris" in rag.ask("capital of France", k=1).text


def test_wiring_guard_rejects_a_retriever_over_a_different_index():
    a = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
    b = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
    # A retriever wired to `b` must not be paired with chunk_index `a`.
    try:
        RagPipeline(chunk_index=a, retriever=IndexRetriever(b))
        assert False, "expected a wiring guard explosion"
    except ConfigError:
        pass


def test_hybrid_end_to_end():
    index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(),
                       lexical=BM25Index())
    rag = RagPipeline(chunk_index=index, chunker=MarkdownChunker())
    rag.index(source())
    assert "Paris" in rag.ask("capital of France", k=1).text


def test_components_are_swappable():
    # A custom generator is used verbatim by the composition root.
    class _Fixed(ExtractiveGenerator):
        name = "fixed-gen"

        def _complete(self, query, packed):
            return ("stub answer", {})

    rag = RagPipeline(chunker=MarkdownChunker(), generator=_Fixed())
    rag.index(source())
    assert rag.ask("anything", k=1).text == "stub answer"
