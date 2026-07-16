"""Leaderboard: ranking trials, with the price next to the score. Hermetic."""
from __future__ import annotations

import pytest

from rag_blocks.core.errors import ConfigError
from rag_blocks.evaluation import Leaderboard, Trial


def trial(tid, ndcg=None, latency=100.0, **metrics) -> Trial:
    scored = dict(metrics)
    if ndcg is not None:
        scored["ndcg@10"] = ndcg
    return Trial(
        trial_id=tid,
        pipeline_spec={"chunker": {"name": tid}},
        fingerprints={},
        metrics=scored,
        cost={"latency_ms": latency},
    )


def board(*trials) -> Leaderboard:
    return Leaderboard(list(trials))


# -- ranking -------------------------------------------------------------


def test_top_ranks_by_the_metric_best_first():
    ranked = board(
        trial("a", ndcg=0.5), trial("b", ndcg=0.9), trial("c", ndcg=0.1)
    ).top(3, by="ndcg@10")
    assert [t.trial_id for t in ranked] == ["b", "a", "c"]


def test_top_truncates_to_n():
    assert len(board(*[trial(str(i), ndcg=i / 10) for i in range(9)]).top(3, by="ndcg@10")) == 3


def test_best_returns_the_winner():
    assert board(trial("a", ndcg=0.5), trial("b", ndcg=0.9)).best(by="ndcg@10").trial_id == "b"


def test_ties_break_deterministically():
    # A leaderboard that reshuffles between runs is not a leaderboard.
    first = board(trial("z", ndcg=0.5), trial("a", ndcg=0.5)).top(2, by="ndcg@10")
    again = board(trial("a", ndcg=0.5), trial("z", ndcg=0.5)).top(2, by="ndcg@10")
    assert [t.trial_id for t in first] == [t.trial_id for t in again] == ["a", "z"]


def test_trials_not_scored_on_the_metric_are_excluded_not_ranked_last():
    # A trial that wasn't scored has no position in that ranking; defaulting
    # it to 0.0 would invent a loser.
    ranked = board(
        trial("scored", ndcg=0.5), trial("phase1_only", token_f1=0.9)
    ).top(5, by="ndcg@10")
    assert [t.trial_id for t in ranked] == ["scored"]


def test_a_rerun_supersedes_its_original_rather_than_appearing_twice():
    ranked = board(trial("a", ndcg=0.5), trial("a", ndcg=0.9)).top(5, by="ndcg@10")
    assert len(ranked) == 1
    assert ranked[0].metrics["ndcg@10"] == 0.9


# -- introspection -------------------------------------------------------


def test_metrics_unions_across_trials_that_measured_different_things():
    # Trials need not share a metric set: a phase-1 trial has no generation
    # metrics, which is a normal state, not a corrupt row.
    assert board(
        trial("a", ndcg=0.5), trial("b", token_f1=0.9)
    ).metrics() == ["ndcg@10", "token_f1"]


def test_len_counts_deduplicated_trials():
    assert len(board(trial("a", ndcg=0.1), trial("a", ndcg=0.2), trial("b", ndcg=0.3))) == 2


# -- fail fast -----------------------------------------------------------


def test_an_unknown_metric_fails_fast_and_lists_what_exists():
    with pytest.raises(ConfigError, match="available"):
        board(trial("a", ndcg=0.5)).top(5, by="nonsense")


@pytest.mark.parametrize("n", [0, -1])
def test_a_non_positive_n_fails_fast(n):
    with pytest.raises(ConfigError):
        board(trial("a", ndcg=0.5)).top(n, by="ndcg@10")


# -- the table -----------------------------------------------------------


def test_the_table_shows_quality_and_cost_side_by_side():
    # The library's actual argument: a winner 0.3% better and 40x more
    # expensive should be visible as such at the moment of choice.
    table = board(
        trial("cheap", ndcg=0.80, latency=50.0),
        trial("pricey", ndcg=0.81, latency=2000.0),
    ).to_table(by="ndcg@10")

    assert "ndcg@10" in table and "latency_ms" in table
    lines = table.strip().split("\n")
    assert "pricey" in lines[2] and "2000.0" in lines[2]  # winner, and its price
    assert "cheap" in lines[3] and "50.0" in lines[3]


def test_an_unmeasured_cost_prints_as_a_dash_not_zero():
    board_ = board(trial("a", ndcg=0.5))
    assert "-" in board_.to_table(by="ndcg@10", cost="api_usd")


def test_an_empty_board_says_so_rather_than_raising():
    assert "no trials" in Leaderboard([]).to_table(by="anything")
