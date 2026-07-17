"""IR metrics + RetrievalEvaluator: the free half of evaluation. Hermetic."""
from __future__ import annotations

import math

import pytest

from rag_blocks.core.contracts import Chunk, ScoredChunk
from rag_blocks.core.errors import ConfigError
from rag_blocks.evaluation import (
    EvalOutcome,
    EvalSample,
    RetrievalEvaluator,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tests.contract_checks import assert_evaluator_contract


def outcome(retrieved_ids, relevant_ids, question="q") -> EvalOutcome:
    return EvalOutcome(
        sample=EvalSample(question=question, relevant_chunk_ids=tuple(relevant_ids)),
        retrieved=tuple(
            ScoredChunk(
                chunk=Chunk(id=cid, doc_id="d", text=cid, index=i),
                score=1.0 / (i + 1),
                retriever_name="fake",
            )
            for i, cid in enumerate(retrieved_ids)
        ),
    )


# -- recall@k ------------------------------------------------------------


@pytest.mark.parametrize(
    "retrieved, relevant, k, expected",
    [
        (["a", "b", "c"], ["a"], 1, 1.0),          # hit at rank 1
        (["b", "a", "c"], ["a"], 1, 0.0),          # hit exists but below k
        (["b", "a", "c"], ["a"], 2, 1.0),          # ... now inside k
        (["a", "b", "c"], ["a", "c"], 3, 1.0),     # both found
        (["a", "b", "c"], ["a", "z"], 3, 0.5),     # one of two found
        (["a"], ["a", "b", "c"], 10, 1 / 3),       # k beyond the ranking
        ([], ["a"], 5, 0.0),                       # retrieved nothing
    ],
)
def test_recall_at_k_counts_relevant_chunks_in_the_top_k(
    retrieved, relevant, k, expected
):
    assert recall_at_k(retrieved, relevant, k) == pytest.approx(expected)


# -- reciprocal rank -----------------------------------------------------


@pytest.mark.parametrize(
    "retrieved, relevant, expected",
    [
        (["a", "b", "c"], ["a"], 1.0),          # rank 1
        (["b", "a", "c"], ["a"], 0.5),          # rank 2
        (["b", "c", "a"], ["a"], 1 / 3),        # rank 3
        (["b", "c"], ["a"], 0.0),               # never retrieved
        (["b", "a", "z"], ["a", "z"], 0.5),     # only the FIRST hit counts
    ],
)
def test_reciprocal_rank_rewards_the_first_hit_only(retrieved, relevant, expected):
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(expected)


# -- nDCG@k (hand-computed) ----------------------------------------------


def test_ndcg_is_1_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == pytest.approx(1.0)


def test_ndcg_is_0_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


def test_ndcg_matches_the_arithmetic_by_hand():
    # One relevant chunk sitting at rank 3:
    #   DCG  = 1/log2(3+1) = 1/2 = 0.5
    #   IDCG = 1/log2(1+1) = 1/1 = 1.0   (ideal: it would be at rank 1)
    #   nDCG = 0.5
    assert ndcg_at_k(["x", "y", "a"], ["a"], 3) == pytest.approx(0.5)

    # Two relevant chunks at ranks 2 and 4, k=4:
    #   DCG  = 1/log2(3) + 1/log2(5)
    #   IDCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "y", "b"], ["a", "b"], 4) == pytest.approx(expected)


def test_ndcg_ranks_a_better_ordering_higher():
    # The property that makes it a ranking metric at all.
    good = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    bad = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert good > bad


def test_ndcg_normalizes_across_questions_with_different_label_counts():
    # Both rankings are perfect; without the IDCG normalization the
    # two-relevant case would score higher and skew every average.
    one = ndcg_at_k(["a", "x"], ["a"], 2)
    two = ndcg_at_k(["a", "b"], ["a", "b"], 2)
    assert one == pytest.approx(two) == pytest.approx(1.0)


# -- preconditions: an unscoreable sample is an error, never a zero ------


@pytest.mark.parametrize("fn", [recall_at_k, ndcg_at_k])
def test_empty_ground_truth_raises_rather_than_scoring_zero(fn):
    with pytest.raises(ValueError, match="non-empty"):
        fn(["a"], [], 5)


def test_reciprocal_rank_rejects_empty_ground_truth():
    with pytest.raises(ValueError, match="non-empty"):
        reciprocal_rank(["a"], [])


@pytest.mark.parametrize("k", [0, -1])
@pytest.mark.parametrize("fn", [recall_at_k, ndcg_at_k])
def test_non_positive_k_is_rejected(fn, k):
    with pytest.raises(ValueError, match="k must be positive"):
        fn(["a"], ["a"], k)


# -- the evaluator -------------------------------------------------------


def test_satisfies_the_evaluator_contract():
    assert_evaluator_contract(
        RetrievalEvaluator(),
        [outcome(["a", "b"], ["a"]), outcome(["x", "c"], ["c"])],
    )


def test_reports_every_configured_cutoff():
    report = RetrievalEvaluator(k_values=(1, 3)).evaluate([outcome(["a"], ["a"])])
    assert set(report.metrics) == {"recall@1", "ndcg@1", "recall@3", "ndcg@3", "mrr"}


def test_aggregate_is_the_mean_over_samples():
    # One perfect ranking, one total miss ⇒ 0.5 on every metric.
    report = RetrievalEvaluator(k_values=(1,)).evaluate(
        [outcome(["a"], ["a"]), outcome(["x"], ["a"])]
    )
    assert report.metrics["recall@1"] == pytest.approx(0.5)
    assert report.metrics["mrr"] == pytest.approx(0.5)


def test_unlabeled_samples_are_skipped_not_scored_as_zero():
    # The honesty invariant: a sample with no ground truth must not drag an
    # average down as if the pipeline had failed it.
    labeled = outcome(["a"], ["a"])
    unlabeled = EvalOutcome(sample=EvalSample(question="unlabeled"), retrieved=())
    report = RetrievalEvaluator(k_values=(1,)).evaluate([labeled, unlabeled])

    assert report.metrics["recall@1"] == pytest.approx(1.0)  # not 0.5
    assert report.per_sample[1] == {}  # ... and its absence is visible
    assert len(report.per_sample) == 2  # alignment with outcomes holds


def test_no_labeled_samples_yields_no_metrics_rather_than_zeros():
    unlabeled = EvalOutcome(sample=EvalSample(question="q"), retrieved=())
    assert RetrievalEvaluator().evaluate([unlabeled]).metrics == {}


def test_per_sample_detail_is_aligned_with_the_outcomes():
    report = RetrievalEvaluator(k_values=(1,)).evaluate(
        [outcome(["a"], ["a"], "hit"), outcome(["x"], ["a"], "miss")]
    )
    assert report.per_sample[0]["recall@1"] == 1.0
    assert report.per_sample[1]["recall@1"] == 0.0


# -- document-level labels: what makes chunk size tunable ----------------


def doc_outcome(retrieved_doc_ids, relevant_doc_ids, question="q") -> EvalOutcome:
    return EvalOutcome(
        sample=EvalSample(question=question, relevant_doc_ids=tuple(relevant_doc_ids)),
        retrieved=tuple(
            ScoredChunk(
                chunk=Chunk(id=f"{doc}:{i}", doc_id=doc, text="t", index=i),
                score=1.0 / (i + 1),
            )
            for i, doc in enumerate(retrieved_doc_ids)
        ),
    )


def test_doc_labels_score_at_document_granularity():
    report = RetrievalEvaluator(k_values=(1,)).evaluate(
        [doc_outcome(["docA", "docB"], ["docA"])]
    )
    assert report.metrics["recall@1"] == 1.0
    assert report.metrics["mrr"] == 1.0


def test_a_document_is_ranked_by_its_best_chunk_not_counted_per_chunk():
    # Three chunks of docB above docA: docB is ONE hit at rank 1, so docA sits
    # at rank 2. Counting per chunk would put docA at rank 4 and quietly
    # reward a chunker for cutting small — the bias that makes chunk-level
    # labels useless for tuning chunk size.
    report = RetrievalEvaluator(k_values=(2,)).evaluate(
        [doc_outcome(["docB", "docB", "docB", "docA"], ["docA"])]
    )
    assert report.metrics["mrr"] == pytest.approx(0.5)  # rank 2, not rank 4
    assert report.metrics["recall@2"] == 1.0


def test_doc_recall_counts_distinct_documents():
    report = RetrievalEvaluator(k_values=(3,)).evaluate(
        [doc_outcome(["docA", "docA", "docC"], ["docA", "docB"])]
    )
    assert report.metrics["recall@3"] == pytest.approx(0.5)  # found A, missed B


def test_chunk_labels_win_when_a_sample_carries_both():
    # The more specific claim wins.
    outcome_ = EvalOutcome(
        sample=EvalSample(
            question="q",
            relevant_chunk_ids=("d:1",),
            relevant_doc_ids=("other-doc",),
        ),
        retrieved=(
            ScoredChunk(chunk=Chunk(id="d:1", doc_id="d", text="t", index=1), score=1.0),
        ),
    )
    assert RetrievalEvaluator(k_values=(1,)).evaluate([outcome_]).metrics["recall@1"] == 1.0


def test_doc_labels_survive_a_chunker_change_where_chunk_ids_do_not():
    # The whole point. Same document, two chunkings: chunk ids differ, doc id
    # does not, so the label still means what it meant.
    coarse = doc_outcome(["docA"], ["docA"])
    fine = EvalOutcome(
        sample=EvalSample(question="q", relevant_doc_ids=("docA",)),
        retrieved=tuple(
            ScoredChunk(
                chunk=Chunk(id=f"docA:{i}", doc_id="docA", text="t", index=i), score=1.0
            )
            for i in range(5)  # a finer chunker: different ids, same document
        ),
    )
    ir = RetrievalEvaluator(k_values=(1,))
    assert ir.evaluate([coarse]).metrics == ir.evaluate([fine]).metrics


def test_a_sample_with_no_retrieval_label_at_all_is_skipped():
    bare = EvalOutcome(sample=EvalSample(question="q", reference_answer="a"))
    assert RetrievalEvaluator().evaluate([bare]).metrics == {}


# -- fail fast at construction -------------------------------------------


@pytest.mark.parametrize("bad", [(), (0,), (-1,), (5, 0)])
def test_invalid_k_values_fail_fast(bad):
    with pytest.raises(ConfigError):
        RetrievalEvaluator(k_values=bad)


def test_unknown_config_field_fails_fast():
    with pytest.raises(ConfigError):
        RetrievalEvaluator(nonsense=1)


def test_config_changes_the_fingerprint():
    # k_values is behavior ⇒ it must separate trial identities.
    assert RetrievalEvaluator(k_values=(1,)).fingerprint() != (
        RetrievalEvaluator(k_values=(10,)).fingerprint()
    )
