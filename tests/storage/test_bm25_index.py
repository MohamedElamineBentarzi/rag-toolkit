"""BM25Index: pure-Python Okapi BM25 lexical index, hermetic."""
from rag_toolkit.core.contracts import Chunk
from rag_toolkit.core.registry import registry
from rag_toolkit.storage.bm25_index import BM25Index
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


def test_readding_same_id_overwrites():
    index = BM25Index()
    index.add([chunk(0, "original text about penguins")])
    index.add([chunk(0, "replaced text about giraffes")])
    assert index.search("penguins", k=5) == []
    assert index.search("giraffes", k=5)[0].chunk.id == "d:0"


def test_registered_under_lexical_index_bm25():
    assert isinstance(registry.create("lexical_index", "bm25"), BM25Index)
