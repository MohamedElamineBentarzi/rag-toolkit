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
        cost={"latency_ms": latency, "query_ms": latency, "index_ms": latency * 2},
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

    assert "ndcg@10" in table and "query_ms" in table
    lines = table.strip().split("\n")
    assert "pricey" in lines[2] and "2000.0" in lines[2]  # winner, and its price
    assert "cheap" in lines[3] and "50.0" in lines[3]


def test_the_table_defaults_to_the_cost_that_compares_cleanly():
    # query_ms, not latency_ms: within a tuning run the parse cache makes an
    # early trial pay for work a later one inherits, so total latency partly
    # ranks by running order. This default is load-bearing, not cosmetic.
    table = board(trial("a", ndcg=0.5)).to_table(by="ndcg@10")
    assert "query_ms" in table
    assert "latency_ms" not in table


def test_an_unmeasured_cost_prints_as_a_dash_not_zero():
    board_ = board(trial("a", ndcg=0.5))
    assert "-" in board_.to_table(by="ndcg@10", cost="api_usd")


def test_an_empty_board_says_so_rather_than_raising():
    assert "no trials" in Leaderboard([]).to_table(by="anything")


# -- marginal analysis: the deep-insights deliverable --------------------


def spec_trial(tid, *, chunker, ndcg, query_ms=100.0, refine=None) -> Trial:
    spec = {"chunker": {"name": chunker, "config": {}}}
    if refine is not None:
        spec["refine"] = [{"name": r, "config": {}} for r in refine]
    return Trial(
        trial_id=tid,
        pipeline_spec=spec,
        fingerprints={},
        metrics={"ndcg@10": ndcg},
        cost={"query_ms": query_ms},
    )


def test_marginal_reports_what_a_choice_was_worth_against_the_mean():
    # "Averaged over everything else, markdown-aware adds +0.2 nDCG."
    # means: fixed -> 0.4, markdown -> 0.8, overall 0.6.
    board = Leaderboard([
        spec_trial("a", chunker="fixed", ndcg=0.3),
        spec_trial("b", chunker="fixed", ndcg=0.5),
        spec_trial("c", chunker="markdown-aware", ndcg=0.7),
        spec_trial("d", chunker="markdown-aware", ndcg=0.9),
    ])
    marginals = {m.option: m for m in board.marginal("chunker", by="ndcg@10")}

    assert marginals["markdown-aware"].quality == pytest.approx(0.2)
    assert marginals["fixed"].quality == pytest.approx(-0.2)
    assert marginals["markdown-aware"].trials == 2


def test_marginal_reports_the_cost_of_the_quality():
    # The whole sentence: "+0.07 nDCG for +180 ms/query". Quality without its
    # price is how you ship a winner that's 0.3% better and 40x dearer.
    board = Leaderboard([
        spec_trial("a", chunker="cheap", ndcg=0.5, query_ms=10.0),
        spec_trial("b", chunker="pricey", ndcg=0.6, query_ms=210.0),
    ])
    marginals = {m.option: m for m in board.marginal("chunker", by="ndcg@10")}

    assert marginals["pricey"].quality == pytest.approx(0.05)
    assert marginals["pricey"].cost == pytest.approx(100.0)  # vs the 110ms mean
    assert marginals["cheap"].cost == pytest.approx(-100.0)


def test_marginals_are_ranked_best_first():
    board = Leaderboard([
        spec_trial("a", chunker="bad", ndcg=0.1),
        spec_trial("b", chunker="good", ndcg=0.9),
    ])
    assert [m.option for m in board.marginal("chunker", by="ndcg@10")] == ["good", "bad"]


def test_config_is_part_of_the_option_because_it_is_what_was_compared():
    trials = [
        Trial(
            trial_id=str(size),
            pipeline_spec={"chunker": {"name": "fixed", "config": {"chunk_chars": size}}},
            fingerprints={},
            metrics={"ndcg@10": ndcg},
            cost={},
        )
        for size, ndcg in [(512, 0.9), (1024, 0.5)]
    ]
    marginals = {m.option: m for m in Leaderboard(trials).marginal("chunker", by="ndcg@10")}
    assert "fixed(chunk_chars=512)" in marginals
    assert marginals["fixed(chunk_chars=512)"].quality == pytest.approx(0.2)


def test_a_chain_is_labeled_by_its_links_and_the_empty_chain_has_a_name():
    # "none" is the baseline a cross-encoder must beat to earn its 180ms — it
    # needs a name to be compared against.
    board = Leaderboard([
        spec_trial("a", chunker="fixed", ndcg=0.5, refine=[]),
        spec_trial("b", chunker="fixed", ndcg=0.7, refine=["keyword", "score-threshold"]),
    ])
    options = {m.option for m in board.marginal("refine", by="ndcg@10")}
    assert options == {"none", "keyword+score-threshold"}


def test_trials_that_never_scored_the_metric_are_excluded():
    board = Leaderboard([
        spec_trial("a", chunker="fixed", ndcg=0.5),
        Trial(trial_id="b", pipeline_spec={"chunker": {"name": "fixed", "config": {}}},
              fingerprints={}, metrics={}, cost={}),  # a failed trial
    ])
    assert board.marginal("chunker", by="ndcg@10")[0].trials == 1


def test_a_single_trial_option_is_reported_with_its_n():
    # n=1 means "averaged over everything else" averaged over one thing —
    # reported, but `trials` is there to be read before believing it.
    board = Leaderboard([spec_trial("a", chunker="fixed", ndcg=0.5)])
    assert board.marginal("chunker", by="ndcg@10")[0].trials == 1


def test_an_unknown_metric_fails_fast_with_what_exists():
    with pytest.raises(ConfigError, match="available"):
        Leaderboard([spec_trial("a", chunker="fixed", ndcg=0.5)]).marginal(
            "chunker", by="nope"
        )


def test_a_stage_no_trial_recorded_lists_the_stages_that_exist():
    with pytest.raises(ConfigError, match="recorded stages"):
        Leaderboard([spec_trial("a", chunker="fixed", ndcg=0.5)]).marginal(
            "generator", by="ndcg@10"
        )


def test_a_marginal_prints_the_sentence_the_milestone_promised():
    board = Leaderboard([
        spec_trial("a", chunker="fixed", ndcg=0.5, query_ms=10.0),
        spec_trial("b", chunker="markdown-aware", ndcg=0.7, query_ms=200.0),
    ])
    line = str(board.marginal("chunker", by="ndcg@10")[0])
    assert "markdown-aware" in line and "+0.1000" in line
    line.encode("ascii")  # printed output must survive a cp1252 console
