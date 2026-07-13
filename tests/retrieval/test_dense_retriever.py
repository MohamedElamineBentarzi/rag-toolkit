"""DenseRetriever: embedder + vector store behind the Query contract."""
import pytest

from rag_toolkit.core.contracts import Chunk, Query
from rag_toolkit.core.errors import ConfigError
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.retrieval.dense import DenseRetriever
from rag_toolkit.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_retriever_contract

_TEXTS = [
    "cats and dogs are common household pets",
    "quarterly financial revenue and profit report",
    "notes on the weather and the changing seasons",
]


def build():
    embedder = HashingEmbedder(dimensions=512)
    store = MemoryVectorStore()
    chunks = [
        Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i, t in enumerate(_TEXTS)
    ]
    store.upsert(chunks, embedder.embed_texts([c.text for c in chunks]))
    return DenseRetriever(embedder=embedder, store=store)


def test_satisfies_the_retriever_contract():
    assert_retriever_contract(
        build(), Query(text="financial revenue report"), expected_top_id="d:1"
    )


def test_results_are_stamped_dense():
    hits = build().retrieve(Query(text="financial revenue"), k=2)
    assert all(h.retriever_name == "dense" for h in hits)


def test_filters_are_passed_through_to_the_store():
    hits = build().retrieve(Query(text="anything", filters={"index": 2}), k=10)
    assert all(h.chunk.index == 2 for h in hits)


def test_requires_backends():
    with pytest.raises(ConfigError):
        DenseRetriever()
    with pytest.raises(ConfigError):
        DenseRetriever(embedder=HashingEmbedder())  # no store
