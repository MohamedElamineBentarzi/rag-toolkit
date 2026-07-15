"""MemoryVectorStore: pure-Python multi-vector index, fully hermetic."""
import pytest

from rag_blocks.core.contracts import Chunk, SparseVector, VectorSpec
from rag_blocks.core.errors import ConfigError, StorageError
from rag_blocks.core.registry import registry
from rag_blocks.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_vector_store_contract


def chunk(i, doc="d"):
    return Chunk(id=f"{doc}:{i}", doc_id=doc, text=f"chunk {i}", index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


def dense_store():
    store = MemoryVectorStore()
    store.ensure_schema([VectorSpec("dense", "dense", dimensions=2)])
    return store


def test_satisfies_the_vector_store_contract():
    assert_vector_store_contract(MemoryVectorStore())


def test_nearest_neighbour_ranks_first():
    store = dense_store()
    store.upsert([chunk(0), chunk(1)], {"dense": [[1.0, 0.0], [0.0, 1.0]]})
    top = store.search("dense", [0.9, 0.1], k=1)
    assert top[0].chunk.id == "d:0"


def test_filter_by_doc_id_scopes_results():
    store = dense_store()
    store.upsert(
        [chunk(0, "a"), chunk(0, "b")],
        {"dense": [[1.0, 0.0], [1.0, 0.0]]},
    )
    hits = store.search("dense", [1.0, 0.0], k=10, filters={"doc_id": "a"})
    assert [h.chunk.doc_id for h in hits] == ["a"]


def test_mismatched_lengths_raise():
    store = dense_store()
    with pytest.raises(StorageError):
        store.upsert([chunk(0)], {"dense": [[1.0, 0.0], [2.0, 0.0]]})


def test_upsert_to_undeclared_space_raises():
    store = dense_store()
    with pytest.raises(StorageError):
        store.upsert([chunk(0)], {"splade": [[1.0, 0.0]]})


def test_schema_mismatch_raises():
    store = dense_store()
    with pytest.raises(ConfigError):
        store.ensure_schema([VectorSpec("dense", "dense", dimensions=99)])


def test_sparse_round_trip_and_dot_scoring():
    store = MemoryVectorStore()
    store.ensure_schema([VectorSpec("splade", "sparse")])
    store.upsert(
        [chunk(0), chunk(1)],
        {"splade": [SparseVector((1, 3), (0.5, 0.5)),
                    SparseVector((2, 4), (1.0, 1.0))]},
    )
    hits = store.search("splade", SparseVector((1, 3), (1.0, 1.0)), k=2)
    assert hits[0].chunk.id == "d:0"
    assert hits[0].score > hits[1].score


def test_update_vectors_replaces_one_representation():
    store = dense_store()
    store.upsert([chunk(0), chunk(1)], {"dense": [[1.0, 0.0], [0.0, 1.0]]})
    store.update_vectors("dense", ["d:0"], [[0.0, 1.0]])
    # d:0 now points the other way; a [0,1] query prefers it (ties by id).
    top = store.search("dense", [0.0, 1.0], k=2)
    assert {top[0].chunk.id, top[1].chunk.id} == {"d:0", "d:1"}
    assert store.search("dense", [1.0, 0.0], k=1)[0].chunk.id != "d:0"


def test_update_vectors_unknown_point_raises():
    store = dense_store()
    with pytest.raises(StorageError):
        store.update_vectors("dense", ["nope"], [[1.0, 0.0]])


def test_fetch_membership_filter():
    store = dense_store()
    store.upsert(
        [chunk(0), chunk(1), chunk(2)],
        {"dense": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
    )
    got = store.fetch({"index": [0, 2]}, limit=10)
    assert {c.id for c in got} == {"d:0", "d:2"}


def test_registered_under_vector_store_memory():
    assert isinstance(
        registry.create("vector_store", "memory"), MemoryVectorStore
    )
