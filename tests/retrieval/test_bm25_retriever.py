"""Bm25Retriever: lexical index behind the Query contract."""
import pytest

from rag_toolkit.core.contracts import Chunk, Query
from rag_toolkit.core.errors import ConfigError
from rag_toolkit.retrieval.bm25 import Bm25Retriever
from rag_toolkit.storage.bm25_index import BM25Index
from tests.contract_checks import assert_retriever_contract

_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown hare races across the open field",
    "quarterly financial revenue and profit report",
]


def build():
    index = BM25Index()
    index.add([
        Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i, t in enumerate(_TEXTS)
    ])
    return Bm25Retriever(index=index)


def test_satisfies_the_retriever_contract():
    assert_retriever_contract(
        build(), Query(text="quick brown fox"), expected_top_id="d:0"
    )


def test_results_are_stamped_bm25():
    hits = build().retrieve(Query(text="financial revenue"), k=2)
    assert hits and all(h.retriever_name == "bm25" for h in hits)


def test_requires_an_index():
    with pytest.raises(ConfigError):
        Bm25Retriever()
