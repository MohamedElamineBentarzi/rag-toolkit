"""RagPipeline: the whole loop end to end, zero dependencies."""
from rag_toolkit.chunking.markdown import MarkdownChunker
from rag_toolkit.core.contracts import Answer, Query, Source
from rag_toolkit.core.errors import ConfigError
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.generation.extractive import ExtractiveGenerator
from rag_toolkit.indexing.chunk_index import ChunkIndex
from rag_toolkit.pipeline import RagPipeline
from rag_toolkit.retrieval.hybrid import HybridRetriever
from rag_toolkit.retrieval.index_retriever import IndexRetriever
from rag_toolkit.storage.bm25_index import BM25Index
from rag_toolkit.storage.memory_store import MemoryVectorStore

_CORPUS = "# France\nParis is the capital of France.\n\n# Fruit\nBananas are yellow.\n"


def source():
    return Source.from_bytes(_CORPUS.encode(), name="facts.md")


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
