"""FusionRetriever: RRF over arbitrary retrievers, plus the fuse() mechanics."""
import pytest

from rag_blocks.core.contracts import Chunk, Query, ScoredChunk
from rag_blocks.core.errors import ConfigError
from rag_blocks.retrieval.base import Retriever
from rag_blocks.retrieval.fusion import fuse, source_labels
from rag_blocks.retrieval.fusion_retriever import FusionRetriever


def chunk(i):
    return Chunk(id=f"d:{i}", doc_id="d", text=f"chunk {i}", index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


class _FixedRetriever(Retriever):
    """Returns a canned ranking — lets us assert the RRF math exactly."""

    name = "fixed"

    def __init__(self, ids, label=None):
        super().__init__()
        self._ids = ids
        self._label = label

    @property
    def label(self):
        return self._label or self.name

    def retrieve(self, query, k=20):
        return [
            ScoredChunk(chunk=chunk(i), score=1.0, retriever_name=self.name)
            for i in self._ids[:k]
        ]


def test_rrf_rewards_documents_ranked_by_multiple_retrievers():
    # d:2 is rank 3 in A but rank 1 in B; appearing in both lifts it above d:0
    # (rank 1 in A, absent from B).
    a = _FixedRetriever([0, 1, 2], label="a")
    b = _FixedRetriever([2, 3], label="b")
    top = FusionRetriever([a, b]).retrieve(Query(text="q"), k=5)
    assert top[0].chunk.id == "d:2"
    assert top[0].retriever_name == "fusion"


def test_dedup_merges_same_chunk_and_records_sources():
    a = _FixedRetriever([0, 1], label="a")
    b = _FixedRetriever([0], label="b")
    top = FusionRetriever([a, b]).retrieve(Query(text="q"), k=5)
    d0 = next(t for t in top if t.chunk.id == "d:0")
    # Merged, not duplicated; both sources attributed with their ranks.
    assert sum(1 for t in top if t.chunk.id == "d:0") == 1
    assert d0.metadata["sources"] == {"a": 1, "b": 1}


def test_source_labels_disambiguate_collisions():
    assert source_labels(["index", "index", "bm25"]) == ["index", "index#1", "bm25"]


def test_fuse_weights_bias_a_ranking():
    a = _FixedRetriever([0], label="a")
    b = _FixedRetriever([1], label="b")
    rankings = [("a", a.retrieve(Query(text="q"))),
                ("b", b.retrieve(Query(text="q")))]
    top = fuse(rankings, k=2, weights=[10.0, 1.0])
    assert top[0].chunk.id == "d:0"


def test_requires_at_least_one_retriever():
    with pytest.raises(ConfigError):
        FusionRetriever([])


def test_weights_length_must_match():
    with pytest.raises(ConfigError):
        FusionRetriever([_FixedRetriever([0])], weights=[1.0, 2.0])


def test_unknown_fusion_method_fails_fast():
    with pytest.raises(ConfigError):
        FusionRetriever([_FixedRetriever([0])], fusion="borda")
