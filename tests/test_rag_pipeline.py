"""RagPipeline: the whole loop end to end, zero dependencies."""
from rag_toolkit.chunking.markdown import MarkdownChunker
from rag_toolkit.core.contracts import Answer, Query, Source
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.generation.extractive import ExtractiveGenerator
from rag_toolkit.pipeline import RagPipeline
from rag_toolkit.storage.local import LocalBlobStore

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


def test_index_populates_the_store():
    rag = RagPipeline(chunker=MarkdownChunker())
    rag.index(source())
    # Two headings ⇒ two chunks upserted into the (default memory) store.
    hits = rag.store.search(rag.embedder.embed_query("fruit"), k=10)
    assert len(hits) == 2


class _CountingEmbedder(HashingEmbedder):
    """HashingEmbedder that records how many texts it actually embedded."""
    name = "counting-emb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedded = 0

    def embed_texts(self, texts):
        self.embedded += len(texts)
        return super().embed_texts(texts)


def test_embedding_cache_skips_recompute_across_runs(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))

    first = _CountingEmbedder()
    RagPipeline(embedder=first, chunker=MarkdownChunker(),
                embedding_cache=cache).index(source())
    assert first.embedded == 2               # two heading sections, both embedded

    # A fresh embedder of the same config shares the cache (keyed by fingerprint).
    second = _CountingEmbedder()
    RagPipeline(embedder=second, chunker=MarkdownChunker(),
                embedding_cache=cache).index(source())
    assert second.embedded == 0              # every chunk served from cache


def test_components_are_swappable():
    # A custom generator is used verbatim by the facade.
    class _Fixed(ExtractiveGenerator):
        name = "fixed-gen"

        def _complete(self, query, packed):
            return ("stub answer", {})

    rag = RagPipeline(chunker=MarkdownChunker(), generator=_Fixed())
    rag.index(source())
    assert rag.ask("anything", k=1).text == "stub answer"
