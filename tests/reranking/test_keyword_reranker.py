"""KeywordReranker: lexical-overlap reranking (hermetic)."""
from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.registry import registry
from rag_toolkit.reranking.keyword import KeywordReranker
from tests.contract_checks import assert_reranker_contract


def scored(i, text, score):
    chunk = Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                  char_start=i, char_end=i + 1, page_start=1, page_end=1)
    return ScoredChunk(chunk=chunk, score=score, retriever_name="dense")


def test_satisfies_the_reranker_contract():
    assert_reranker_contract(KeywordReranker())


def test_reorders_by_query_overlap():
    # Input order (by retrieval score) is wrong; the reranker fixes it.
    candidates = [
        scored(0, "the weather is nice today", 0.9),   # no overlap
        scored(1, "quick brown fox jumps", 0.5),        # full overlap
    ]
    out = KeywordReranker().rerank(Query(text="quick brown fox"), candidates, 2)
    assert out[0].chunk.id == "d:1"          # promoted despite lower input score
    assert out[0].score > out[1].score


def test_keeps_retriever_attribution():
    out = KeywordReranker().rerank(
        Query(text="fox"), [scored(0, "a fox", 0.1)], 1
    )
    assert out[0].retriever_name == "dense"  # reranking doesn't erase origin


def test_registered_under_reranker_keyword():
    assert isinstance(registry.create("reranker", "keyword"), KeywordReranker)
