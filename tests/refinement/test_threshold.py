"""ScoreThreshold: drop candidates below a relevance floor."""
from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.registry import registry
from rag_toolkit.refinement.threshold import ScoreThreshold
from tests.contract_checks import assert_refiner_contract


def scored(i, score):
    chunk = Chunk(id=f"d:{i}", doc_id="d", text=f"c{i}", index=i,
                  char_start=i, char_end=i + 1, page_start=1, page_end=1)
    return ScoredChunk(chunk=chunk, score=score, retriever_name="index")


def test_satisfies_the_refiner_contract():
    assert_refiner_contract(ScoreThreshold())


def test_drops_below_floor_keeps_order():
    candidates = [scored(0, 0.9), scored(1, 0.4), scored(2, 0.05)]
    out = ScoreThreshold(min_score=0.3).refine(Query(text="q"), candidates, k=10)
    assert [r.chunk.id for r in out] == ["d:0", "d:1"]


def test_default_floor_keeps_everything():
    candidates = [scored(0, 0.9), scored(1, 0.0)]
    out = ScoreThreshold().refine(Query(text="q"), candidates, k=10)
    assert len(out) == 2


def test_registered_under_refiner_score_threshold():
    assert isinstance(
        registry.create("refiner", "score-threshold"), ScoreThreshold
    )
