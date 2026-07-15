"""KeywordRefiner: hermetic lexical reranking refiner."""
from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.registry import registry
from rag_toolkit.refinement.keyword import KeywordRefiner
from tests.contract_checks import assert_refiner_contract


def scored(i, text, score):
    chunk = Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                  char_start=i, char_end=i + 1, page_start=1, page_end=1)
    return ScoredChunk(chunk=chunk, score=score, retriever_name="index")


def test_satisfies_the_refiner_contract():
    assert_refiner_contract(KeywordRefiner())


def test_reorders_by_query_overlap():
    # Retrieval put the unrelated chunk first; keyword overlap must fix that.
    candidates = [
        scored(0, "completely unrelated content", 0.9),
        scored(1, "quick brown fox", 0.1),
    ]
    out = KeywordRefiner().refine(Query(text="quick brown fox"), candidates, k=2)
    assert out[0].chunk.id == "d:1"


def test_keeps_retriever_name():
    out = KeywordRefiner().refine(
        Query(text="fox"), [scored(0, "fox", 0.5)], k=1
    )
    assert out[0].retriever_name == "index"


def test_registered_under_refiner_keyword():
    assert isinstance(registry.create("refiner", "keyword"), KeywordRefiner)
