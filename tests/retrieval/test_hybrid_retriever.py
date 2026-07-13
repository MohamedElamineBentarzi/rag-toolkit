"""HybridRetriever: RRF fusion over composed retrievers."""
import pytest

from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.errors import ConfigError
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.retrieval.base import Retriever
from rag_toolkit.retrieval.bm25 import Bm25Retriever
from rag_toolkit.retrieval.dense import DenseRetriever
from rag_toolkit.retrieval.hybrid import HybridRetriever
from rag_toolkit.storage.bm25_index import BM25Index
from rag_toolkit.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_retriever_contract


def chunk(i, text="x"):
    return Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


class _FixedRetriever(Retriever):
    """Returns a canned ranking — lets us assert the RRF math exactly."""

    name = "fixed"

    def __init__(self, ids):
        super().__init__()
        self._ids = ids

    def retrieve(self, query, k=20):
        return [
            ScoredChunk(chunk=chunk(i), score=1.0, retriever_name=self.name)
            for i in self._ids[:k]
        ]


def test_rrf_rewards_documents_ranked_by_multiple_retrievers():
    # d:2 is only rank 3 in A but rank 1 in B; appearing in both should lift it
    # above d:0, which is rank 1 in A but absent from B.
    a = _FixedRetriever([0, 1, 2])
    b = _FixedRetriever([2, 3])
    hybrid = HybridRetriever(retrievers=[a, b])
    top = hybrid.retrieve(Query(text="q"), k=5)
    assert top[0].chunk.id == "d:2"
    assert top[0].retriever_name == "hybrid"


def _real_hybrid():
    texts = ["quick brown fox", "financial revenue report", "quick financial notes"]
    chunks = [chunk(i, t) for i, t in enumerate(texts)]

    embedder = HashingEmbedder(dimensions=512)
    store = MemoryVectorStore()
    store.upsert(chunks, embedder.embed_texts([c.text for c in chunks]))
    dense = DenseRetriever(embedder=embedder, store=store)

    index = BM25Index()
    index.add(chunks)
    bm25 = Bm25Retriever(index=index)

    return HybridRetriever(retrievers=[dense, bm25])


def test_satisfies_the_retriever_contract():
    # "quick financial" matches d:2 in both modalities → fused winner.
    assert_retriever_contract(
        _real_hybrid(), Query(text="quick financial"), expected_top_id="d:2"
    )


def test_requires_at_least_one_retriever():
    with pytest.raises(ConfigError):
        HybridRetriever(retrievers=[])


def test_weights_length_must_match_retrievers():
    a = _FixedRetriever([0])
    with pytest.raises(ConfigError):
        HybridRetriever(retrievers=[a], weights=[1.0, 2.0])
