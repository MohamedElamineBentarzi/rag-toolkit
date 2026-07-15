"""ChunkIndex: the aggregate owning a corpus's representations. Hermetic."""
import pytest

from rag_toolkit.core.contracts import Chunk
from rag_toolkit.core.errors import ConfigError
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.indexing.chunk_index import ChunkIndex
from rag_toolkit.storage.bm25_index import BM25Index
from rag_toolkit.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_index_contract


def store():
    return MemoryVectorStore()


def chunk(i, text, doc="d"):
    return Chunk(id=f"{doc}:{i}", doc_id=doc, text=text, index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


# -- contract, over several representation shapes --------------------------

def test_dense_only_satisfies_contract():
    assert_index_contract(ChunkIndex(store(), dense=HashingEmbedder()))


def test_lexical_only_satisfies_contract():
    assert_index_contract(ChunkIndex(store(), lexical=BM25Index()))


def test_hybrid_dense_plus_lexical_satisfies_contract():
    assert_index_contract(
        ChunkIndex(store(), dense=HashingEmbedder(), lexical=BM25Index())
    )


# -- constructor: progressive disclosure & fail-fast -----------------------

def test_bare_encoder_auto_names_dense():
    index = ChunkIndex(store(), dense=HashingEmbedder())
    assert index.representations() == ["dense"]


def test_mapping_names_multiple_dense_representations():
    index = ChunkIndex(
        store(),
        dense={"bge": HashingEmbedder(), "e5": HashingEmbedder(dimensions=128)},
    )
    assert set(index.representations()) == {"bge", "e5"}


def test_lexical_mounts_as_lexical():
    index = ChunkIndex(store(), dense=HashingEmbedder(), lexical=BM25Index())
    assert "lexical" in index.representations()


def test_no_representation_fails_fast():
    with pytest.raises(ConfigError):
        ChunkIndex(store())


def test_duplicate_representation_name_fails_fast():
    with pytest.raises(ConfigError):
        # A dense space named "lexical" collides with the mounted BM25 name.
        ChunkIndex(store(), dense={"lexical": HashingEmbedder()},
                   lexical=BM25Index())


# -- behavior --------------------------------------------------------------

def test_search_dispatches_to_the_right_representation():
    index = ChunkIndex(store(), dense=HashingEmbedder(), lexical=BM25Index())
    index.add([chunk(0, "the quick brown fox"),
               chunk(1, "financial results third quarter")])
    assert index.search("dense", "quick brown fox", k=1)[0].chunk.id == "d:0"
    assert index.search("lexical", "quick brown fox", k=1)[0].chunk.id == "d:0"


def test_unknown_representation_lists_available():
    index = ChunkIndex(store(), dense=HashingEmbedder())
    with pytest.raises(ConfigError) as exc:
        index.search("splade", "x", k=1)
    assert "dense" in str(exc.value)


def test_update_representation_refreshes_one_space():
    index = ChunkIndex(store(), dense=HashingEmbedder())
    c = chunk(0, "original text about apples")
    index.add([c])
    updated = chunk(0, "revised text about oranges")
    index.update_representation("dense", [updated])
    # The refreshed vector now ranks the oranges query above apples.
    hits = index.search("dense", "oranges", k=1)
    assert hits and hits[0].chunk.id == "d:0"


def test_describe_folds_in_backend_identities():
    a = ChunkIndex(store(), dense=HashingEmbedder())
    b = ChunkIndex(store(), dense=HashingEmbedder(dimensions=128))
    # Different embedder ⇒ different representation fingerprint ⇒ different id.
    assert a.fingerprint() != b.fingerprint()
    info = a.describe()
    assert "store_fingerprint" in info
    assert "dense" in info["representations"]
