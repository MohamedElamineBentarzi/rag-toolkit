"""MultiQueryRetriever / HydeRetriever: query shaping as composition (fake LLM)."""
import pytest

from rag_blocks.core.contracts import Chunk, Query
from rag_blocks.core.errors import ConfigError
from rag_blocks.embedding.hashing import HashingEmbedder
from rag_blocks.indexing.chunk_index import ChunkIndex
from rag_blocks.retrieval.index_retriever import IndexRetriever
from rag_blocks.retrieval.query_shaping import HydeRetriever, MultiQueryRetriever
from rag_blocks.storage.memory_store import MemoryVectorStore


def chunk(i, text):
    return Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


def _inner():
    texts = ["quick brown fox", "financial revenue report", "weather and seasons"]
    index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(dimensions=512))
    index.add([chunk(i, t) for i, t in enumerate(texts)])
    return IndexRetriever(index)


class _RecordingComplete:
    """A fake `complete` seam: records prompts and returns a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def test_multi_query_expands_fuses_and_stamps():
    # Every expansion points at d:1; fusion should surface it above the tie-break.
    complete = _RecordingComplete("financial report\nrevenue numbers\nprofit revenue")
    r = MultiQueryRetriever(_inner(), complete=complete, n=3)
    top = r.retrieve(Query(text="money results"), k=3)
    assert top and top[0].retriever_name == "multi-query"
    assert top[0].chunk.id == "d:1"
    # The LLM was asked to expand exactly once.
    assert len(complete.prompts) == 1


def test_multi_query_always_includes_the_original_query():
    complete = _RecordingComplete("")  # no expansions ⇒ falls back to original
    r = MultiQueryRetriever(_inner(), complete=complete, n=4)
    top = r.retrieve(Query(text="financial revenue"), k=1)
    assert top[0].chunk.id == "d:1"


def test_hyde_retrieves_on_the_hypothetical_passage():
    # The hypothetical answer is what actually hits the retriever.
    complete = _RecordingComplete("Quarterly financial revenue rose this year.")
    r = HydeRetriever(_inner(), complete=complete)
    top = r.retrieve(Query(text="how did revenue do?"), k=1)
    assert top[0].chunk.id == "d:1"
    assert top[0].retriever_name == "hyde"


def test_hyde_falls_back_to_query_when_completion_is_empty():
    complete = _RecordingComplete("   ")
    r = HydeRetriever(_inner(), complete=complete)
    top = r.retrieve(Query(text="quick brown fox"), k=1)
    assert top[0].chunk.id == "d:0"


def test_filters_survive_query_shaping():
    complete = _RecordingComplete("anything at all")
    r = HydeRetriever(_inner(), complete=complete)
    top = r.retrieve(Query(text="x", filters={"index": 2}), k=10)
    assert all(t.chunk.index == 2 for t in top)


def test_require_inner_and_complete():
    with pytest.raises(ConfigError):
        MultiQueryRetriever(_inner())  # no complete
    with pytest.raises(ConfigError):
        HydeRetriever(complete=_RecordingComplete("x"))  # no inner
