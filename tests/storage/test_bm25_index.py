"""BM25Index: pure-Python Okapi BM25 lexical index, hermetic."""
from rag_blocks.core.contracts import Chunk
from rag_blocks.core.registry import registry
from rag_blocks.storage.bm25_index import BM25Index
from rag_blocks.storage.local import LocalBlobStore
from tests.contract_checks import assert_lexical_index_contract


def chunk(i, text):
    return Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


def test_satisfies_the_lexical_index_contract():
    assert_lexical_index_contract(BM25Index())


def test_exact_term_match_beats_unrelated_doc():
    index = BM25Index()
    index.add([
        chunk(0, "quarterly financial revenue statement"),
        chunk(1, "a poem about the sea and the sky"),
    ])
    top = index.search("revenue", k=2)
    assert top[0].chunk.id == "d:0"


def test_add_is_idempotent_by_id():
    index = BM25Index()
    index.add([chunk(0, "penguins waddle")])
    index.add([chunk(0, "penguins waddle"), chunk(1, "walruses swim")])  # d:0 skipped
    assert len(index._chunks) == 2
    assert index.search("penguins", k=5)[0].chunk.id == "d:0"
    assert index.search("walruses", k=5)[0].chunk.id == "d:1"


# -- persistence --------------------------------------------------------------

def corpus(index):
    index.add([
        chunk(0, "quarterly financial revenue statement"),
        chunk(1, "notes on penguins and the antarctic"),
    ])


def test_persist_then_load_round_trips(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    written = BM25Index(store=store, namespace="corpus-a")
    corpus(written)
    written.persist()

    # A fresh index over the same store + namespace rehydrates.
    restored = BM25Index(store=store, namespace="corpus-a")
    assert restored.search("revenue", k=5) == []   # empty until loaded
    restored.load()
    hit = restored.search("revenue", k=5)[0]
    assert hit.chunk.id == "d:0"
    assert hit.chunk.page_start == 1                # provenance survived the round-trip


def test_load_after_persist_is_idempotent_and_incremental(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    a = BM25Index(store=store, namespace="corpus-b")
    corpus(a)
    a.persist()

    b = BM25Index(store=store, namespace="corpus-b")
    b.load()
    b.add([chunk(0, "already there"), chunk(2, "new whales content")])  # d:0 skipped
    assert len(b._chunks) == 3
    assert b.search("whales", k=5)[0].chunk.id == "d:2"


def test_memory_mode_persist_and_load_are_noops():
    index = BM25Index()          # no store injected
    index.add([chunk(0, "text")])
    index.persist()              # no-op, must not raise
    index.load()                 # no-op, must not raise
    assert index.search("text", k=1)


def test_registered_under_lexical_index_bm25():
    assert isinstance(registry.create("lexical_index", "bm25"), BM25Index)
