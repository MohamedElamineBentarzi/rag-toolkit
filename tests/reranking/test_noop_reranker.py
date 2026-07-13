"""NoOpReranker: the Null Object baseline."""
from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.registry import registry
from rag_toolkit.reranking.noop import NoOpReranker
from tests.contract_checks import assert_reranker_contract


def scored(i, score):
    chunk = Chunk(id=f"d:{i}", doc_id="d", text=f"c{i}", index=i,
                  char_start=i, char_end=i + 1, page_start=1, page_end=1)
    return ScoredChunk(chunk=chunk, score=score, retriever_name="dense")


def test_satisfies_the_reranker_contract():
    assert_reranker_contract(NoOpReranker())


def test_preserves_order_and_caps_to_top_k():
    candidates = [scored(0, 0.9), scored(1, 0.5), scored(2, 0.1)]
    out = NoOpReranker().rerank(Query(text="q"), candidates, top_k=2)
    assert [r.chunk.id for r in out] == ["d:0", "d:1"]  # unchanged order, capped


def test_registered_under_reranker_noop():
    assert isinstance(registry.create("reranker", "noop"), NoOpReranker)
