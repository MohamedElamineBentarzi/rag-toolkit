"""IndexRetriever: a read-only view over one representation of a ChunkIndex."""
import pytest

from rag_blocks.core.contracts import Chunk, Query
from rag_blocks.core.errors import ConfigError
from rag_blocks.embedding.hashing import HashingEmbedder
from rag_blocks.indexing.chunk_index import ChunkIndex
from rag_blocks.retrieval.index_retriever import IndexRetriever
from rag_blocks.storage.bm25_index import BM25Index
from rag_blocks.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_retriever_contract

_TEXTS = [
    "cats and dogs are common household pets",
    "quarterly financial revenue and profit report",
    "notes on the weather and the changing seasons",
]


def _chunks():
    return [
        Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i, t in enumerate(_TEXTS)
    ]


def dense_index():
    index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder(dimensions=512))
    index.add(_chunks())
    return index


def hybrid_index():
    index = ChunkIndex(
        MemoryVectorStore(), dense=HashingEmbedder(dimensions=512),
        lexical=BM25Index(),
    )
    index.add(_chunks())
    return index


def test_dense_view_satisfies_the_retriever_contract():
    assert_retriever_contract(
        IndexRetriever(dense_index()),
        Query(text="financial revenue report"), expected_top_id="d:1",
    )


def test_lexical_view_satisfies_the_retriever_contract():
    assert_retriever_contract(
        IndexRetriever(hybrid_index(), representation="lexical"),
        Query(text="financial revenue report"), expected_top_id="d:1",
    )


def test_results_are_stamped_index():
    hits = IndexRetriever(dense_index()).retrieve(Query(text="financial"), k=2)
    assert all(h.retriever_name == "index" for h in hits)


def test_label_folds_in_representation():
    r = IndexRetriever(hybrid_index(), representation="lexical")
    assert r.label == "index:lexical"


def test_filters_pass_through_to_the_index():
    hits = IndexRetriever(dense_index()).retrieve(
        Query(text="anything", filters={"index": 2}), k=10
    )
    assert all(h.chunk.index == 2 for h in hits)


def test_ambiguous_representation_fails_fast():
    with pytest.raises(ConfigError):
        IndexRetriever(hybrid_index())  # two representations, none named


def test_unknown_representation_fails_fast():
    with pytest.raises(ConfigError):
        IndexRetriever(dense_index(), representation="splade")


def test_requires_an_index():
    with pytest.raises(ConfigError):
        IndexRetriever()
