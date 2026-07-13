"""MemoryVectorStore: pure-Python cosine index, fully hermetic."""
import pytest

from rag_toolkit.core.contracts import Chunk
from rag_toolkit.core.errors import StorageError
from rag_toolkit.core.registry import registry
from rag_toolkit.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_vector_store_contract


def chunk(i, doc="d"):
    return Chunk(id=f"{doc}:{i}", doc_id=doc, text=f"chunk {i}", index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


def test_satisfies_the_vector_store_contract():
    assert_vector_store_contract(MemoryVectorStore())


def test_nearest_neighbour_ranks_first():
    store = MemoryVectorStore()
    store.upsert([chunk(0), chunk(1)], [[1.0, 0.0], [0.0, 1.0]])
    top = store.search([0.9, 0.1], k=1)
    assert top[0].chunk.id == "d:0"


def test_filter_by_doc_id_scopes_results():
    store = MemoryVectorStore()
    store.upsert(
        [chunk(0, "a"), chunk(0, "b")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0], k=10, filters={"doc_id": "a"})
    assert [h.chunk.doc_id for h in hits] == ["a"]


def test_mismatched_lengths_raise():
    store = MemoryVectorStore()
    with pytest.raises(StorageError):
        store.upsert([chunk(0)], [[1.0], [2.0]])


def test_registered_under_store_memory():
    assert isinstance(registry.create("store", "memory"), MemoryVectorStore)
