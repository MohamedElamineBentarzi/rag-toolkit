"""Tuner: the Template Method over iter_candidates. Fully hermetic."""
from __future__ import annotations

import pytest

from rag_blocks.core.contracts import Source
from rag_blocks.core.errors import ConfigError, EmbeddingError
from rag_blocks.evaluation import (
    AnswerMatchEvaluator,
    EvalSample,
    GridTuner,
    Leaderboard,
    RandomTuner,
    RetrievalEvaluator,
    SearchSpace,
    TrialLog,
    Tuner,
    choice,
)

CORPUS = """# France

Paris is the capital of France and its largest city.

# Fruit

Bananas are yellow and grow in tropical climates.

# Logistics

The warehouse in Lyon ships orders across the European Union.
"""


def source():
    return Source.from_bytes(CORPUS.encode(), name="facts.md")


def dataset():
    # Labeled against chunk ids the fixed chunker produces deterministically.
    return [
        EvalSample(
            question="What is the capital of France?",
            relevant_chunk_ids=(f"{source().content_hash()}:0",),
            reference_answer="Paris is the capital of France.",
        )
    ]


def space(n=2):
    return SearchSpace(
        chunker=[choice("fixed", chunk_chars=[200, 400][:n], overlap_chars=0)],
        embedder=[choice("hashing", dimensions=64)],
    )


def screeners():
    return [RetrievalEvaluator(k_values=(1, 3))]


def run(tuner, sp=None, evaluators=None, **kw):
    return tuner.run(
        sp or space(),
        dataset(),
        source(),
        evaluators=evaluators or screeners(),
        **kw,
    )


class _CountingEvaluator(RetrievalEvaluator):
    """Counts how many times the tuner asked it to score."""

    name = "counting"

    def __init__(self, config=None, **overrides):
        super().__init__(config, **overrides)
        self.calls = 0

    def evaluate(self, outcomes):
        self.calls += 1
        return super().evaluate(outcomes)


class _CountingJudge(AnswerMatchEvaluator):
    """Stands in for the expensive LLM judge — counts, so 'only the finalists'
    is an assertion rather than a hope."""

    name = "counting-judge"
    stage = "generation"

    def __init__(self, config=None, **overrides):
        super().__init__(config, **overrides)
        self.calls = 0

    def evaluate(self, outcomes):
        self.calls += 1
        return super().evaluate(outcomes)


# -- the ABC ---------------------------------------------------------------


def test_kind_is_on_the_abc_and_names_on_the_implementations():
    assert Tuner.kind == "tuner"
    assert GridTuner.name == "grid" and RandomTuner.name == "random"


def test_a_tuner_without_iter_candidates_cannot_be_instantiated():
    class _Incomplete(Tuner):
        name = "incomplete"

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


# -- grid ------------------------------------------------------------------


def test_grid_runs_every_combination():
    board = run(GridTuner())
    assert len(board) == 2


def test_grid_enumerates_in_the_spaces_cache_warm_order():
    sp = space()
    assert list(GridTuner().iter_candidates(sp)) == list(sp.expand())


def test_a_trial_records_what_actually_ran():
    board = run(GridTuner())
    trial = board.top(1, by="ndcg@3")[0]

    # Reproducible from the log line alone: resolved describes, not the spec.
    assert trial.pipeline_spec["chunker"]["name"] == "fixed"
    assert trial.pipeline_spec["retriever"]["name"] == "index"  # derived, recorded
    assert set(trial.fingerprints) >= {"chunker", "index", "retriever", "generator"}
    assert trial.started_at and trial.finished_at
    assert trial.metadata["phase"] == 1


def test_the_empty_chain_is_recorded_so_it_can_be_compared_against():
    # Regression: chain stages were only recorded when non-empty, so the "no
    # refiner" baseline vanished from the trial — and the leaderboard's
    # marginal for `refine` silently compared the refiners against nothing.
    # The Null Object is a choice; it has to be visible as one.
    sp = SearchSpace(
        embedder=[choice("hashing", dimensions=64)],
        refine=[[], [choice("score-threshold", min_score=0.0)]],
    )
    board = GridTuner().run(sp, dataset(), source(), evaluators=screeners())

    assert all("refine" in t.pipeline_spec for t in board.trials)
    options = {m.option for m in board.marginal("refine", by="ndcg@3")}
    assert options == {"none", "score-threshold(min_score=0.0)"}


def test_the_index_representations_are_recorded_by_name_not_just_fingerprint():
    # Regression: ChunkIndex.describe() reports representations as opaque
    # fingerprints, so a trial could not say WHICH embedder ran — breaking
    # both reproducibility and marginal(). It asks the index now.
    board = run(GridTuner())
    spec = board.top(1, by="ndcg@3")[0].pipeline_spec
    assert spec["embedder"]["name"] == "hashing"
    assert spec["embedder"]["config"]["dimensions"] == 64


def test_trials_carry_retrieval_metrics_and_cost():
    board = run(GridTuner())
    trial = board.top(1, by="ndcg@3")[0]
    assert "recall@1" in trial.metrics and "mrr" in trial.metrics
    assert trial.cost["query_ms"] >= 0
    assert trial.cost["index_ms"] >= 0


def test_the_run_is_deterministic():
    first = run(GridTuner()).top(2, by="ndcg@3")
    second = run(GridTuner()).top(2, by="ndcg@3")
    assert [t.trial_id for t in first] == [t.trial_id for t in second]


# -- two-phase: the whole point -------------------------------------------


def test_phase_1_screens_everything_and_phase_2_judges_only_finalists():
    # The cost-control claim of §7.3, asserted rather than assumed: the free
    # evaluator sees every candidate, the expensive one sees `finalists`.
    screener, judge = _CountingEvaluator(k_values=(1, 3)), _CountingJudge()
    sp = SearchSpace(
        chunker=[choice("fixed", chunk_chars=[100, 200, 300, 400], overlap_chars=0)],
        embedder=[choice("hashing", dimensions=64)],
    )
    run(
        GridTuner(screen_by="ndcg@3", finalists=2),
        sp=sp,
        evaluators=[screener, judge],
    )
    assert screener.calls == 4 + 2  # 4 candidates screened, 2 finalists re-scored
    assert judge.calls == 2  # ... and the judge ONLY ran on the finalists


def test_without_a_generation_evaluator_nothing_generates_at_all():
    # Phase 2 is skipped entirely: no answers, no generation cost, no bill.
    board = run(GridTuner(screen_by="ndcg@3"), evaluators=screeners())
    assert all(t.metadata["phase"] == 1 for t in board.trials)
    assert all("latency_ms.generate" not in t.cost for t in board.trials)


def test_a_finalist_trial_carries_generation_metrics_and_an_answer_cost():
    board = run(
        GridTuner(screen_by="ndcg@3", finalists=1),
        evaluators=[*screeners(), AnswerMatchEvaluator()],
    )
    finalist = board.top(1, by="token_f1")[0]
    assert finalist.metadata["phase"] == 2
    # Phase 2 is a COMPLETE run: retrieval metrics survive alongside.
    assert "token_f1" in finalist.metrics and "ndcg@3" in finalist.metrics
    assert finalist.cost["latency_ms.generate"] >= 0


def test_a_finalist_supersedes_its_phase_1_trial_rather_than_duplicating_it():
    board = run(
        GridTuner(screen_by="ndcg@3", finalists=2),
        evaluators=[*screeners(), AnswerMatchEvaluator()],
    )
    assert len(board) == 2  # not 4
    assert all(t.metadata["phase"] == 2 for t in board.trials)


def test_finalists_beyond_the_space_size_are_harmless():
    board = run(
        GridTuner(screen_by="ndcg@3", finalists=99),
        evaluators=[*screeners(), AnswerMatchEvaluator()],
    )
    assert len(board) == 2


def test_nothing_scored_on_screen_by_means_the_judge_never_runs():
    # An unlabeled dataset earns no ranking — that's the honest outcome, not a
    # reason to judge an arbitrary five.
    judge = _CountingJudge()
    tuner = GridTuner(screen_by="a-metric-nobody-reports")
    board = tuner.run(
        space(), dataset(), source(), evaluators=[*screeners(), judge]
    )
    assert judge.calls == 0
    assert len(board) == 2  # the phase-1 trials still stand


# -- random ----------------------------------------------------------------


def test_random_respects_its_budget():
    sp = SearchSpace(
        chunker=[choice("fixed", chunk_chars=[100, 200, 300, 400], overlap_chars=0)],
        embedder=[choice("hashing", dimensions=64)],
    )
    assert len(list(RandomTuner(n_trials=2).iter_candidates(sp))) == 2


def test_random_is_reproducible_from_its_seed():
    # A tuning run that can't be reproduced isn't evidence.
    sp = SearchSpace(chunker=[choice("fixed", chunk_chars=[1, 2, 3, 4, 5], overlap_chars=0)])
    first = list(RandomTuner(n_trials=3, seed=7).iter_candidates(sp))
    second = list(RandomTuner(n_trials=3, seed=7).iter_candidates(sp))
    assert first == second


def test_different_seeds_explore_differently():
    sp = SearchSpace(
        chunker=[choice("fixed", chunk_chars=list(range(1, 21)), overlap_chars=0)]
    )
    a = list(RandomTuner(n_trials=5, seed=1).iter_candidates(sp))
    b = list(RandomTuner(n_trials=5, seed=2).iter_candidates(sp))
    assert a != b


def test_random_samples_without_replacement():
    sp = SearchSpace(chunker=[choice("fixed", chunk_chars=[1, 2, 3, 4, 5], overlap_chars=0)])
    picked = list(RandomTuner(n_trials=5, seed=0).iter_candidates(sp))
    assert len({t["chunker"]["params"]["chunk_chars"] for t in picked}) == 5


def test_asking_for_more_than_the_space_holds_returns_the_space():
    # Not an error — a small space. Sampling with replacement would bill twice
    # for the same pipeline.
    sp = SearchSpace(chunker=[choice("fixed", chunk_chars=[1, 2], overlap_chars=0)])
    assert len(list(RandomTuner(n_trials=10).iter_candidates(sp))) == 2


def test_a_non_positive_budget_fails_fast():
    with pytest.raises(ConfigError, match="n_trials"):
        list(RandomTuner(n_trials=0).iter_candidates(space()))


def test_random_and_grid_are_interchangeable_at_the_same_seam():
    # Swapping is the proof the design works (AGENTS.md §11).
    for tuner in (GridTuner(), RandomTuner(n_trials=2)):
        board = run(tuner)
        assert isinstance(board, Leaderboard) and len(board) == 2


# -- resilience ------------------------------------------------------------


def test_one_bad_combination_does_not_lose_the_others(monkeypatch):
    # An overnight grid must not be destroyed by a single invalid pipeline.
    from rag_blocks.embedding.hashing import HashingEmbedder

    original = HashingEmbedder.embed_texts
    calls = {"n": 0}

    def sometimes_explode(self, texts):
        calls["n"] += 1
        if self.config.dimensions == 32:
            raise EmbeddingError("this model is unavailable")
        return original(self, texts)

    monkeypatch.setattr(HashingEmbedder, "embed_texts", sometimes_explode)
    sp = SearchSpace(embedder=[choice("hashing", dimensions=[32, 64])])
    board = GridTuner().run(sp, dataset(), source(), evaluators=screeners())

    assert len(board) == 2
    failed = [t for t in board.trials if "error" in t.metadata]
    assert len(failed) == 1
    assert "EmbeddingError" in failed[0].metadata["error"]
    # ... and the good one still scored.
    assert board.top(1, by="ndcg@3")[0].metrics["ndcg@3"] > 0


# -- the log ---------------------------------------------------------------


def test_every_trial_reaches_the_log(tmp_path):
    log = TrialLog(tmp_path / "trials.jsonl")
    board = run(GridTuner(), log=log)
    assert len(log.read()) == len(board) == 2
    assert {t.trial_id for t in log.read()} == {t.trial_id for t in board.trials}


def test_a_run_without_a_log_still_returns_a_board():
    assert len(run(GridTuner())) == 2


# -- fail fast -------------------------------------------------------------


def test_an_empty_dataset_fails_fast():
    with pytest.raises(ConfigError, match="dataset"):
        GridTuner().run(space(), [], source(), evaluators=screeners())


def test_no_evaluators_fails_fast():
    with pytest.raises(ConfigError, match="evaluator"):
        GridTuner().run(space(), dataset(), source(), evaluators=[])


def test_a_generation_only_evaluator_set_fails_fast():
    # Phase 1 ranks on a retrieval metric; without one there is nothing to
    # screen by, and judging everything is what two-phase exists to prevent.
    with pytest.raises(ConfigError, match="stage='retrieval'"):
        GridTuner().run(
            space(), dataset(), source(), evaluators=[AnswerMatchEvaluator()]
        )


def test_non_positive_finalists_fails_fast():
    with pytest.raises(ConfigError, match="finalists"):
        run(GridTuner(finalists=0), evaluators=[*screeners(), AnswerMatchEvaluator()])


# -- identity --------------------------------------------------------------


def test_tuner_config_is_fingerprint_input():
    assert GridTuner(finalists=2).fingerprint() != GridTuner(finalists=5).fingerprint()


def test_the_tuners_are_registered():
    from rag_blocks.core.registry import registry

    assert registry.get("tuner", "grid") is GridTuner
    assert registry.get("tuner", "random") is RandomTuner
