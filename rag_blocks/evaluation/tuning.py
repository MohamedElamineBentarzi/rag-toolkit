"""Tuner: search a space, score every candidate, log every trial.

The house pattern, a third time. `Parser.parse()` wraps `iter_pages()`;
`Chunker.chunk()` wraps `iter_spans()`; `Tuner.run()` wraps
`iter_candidates()`. One abstract primitive holds the only thing that actually
varies — **which combinations, in what order** — and every piece of bookkeeping
lives once in the Template Method: build, index, retrieve, generate, score,
time, record. A strategy that had to reimplement two-phase screening to change
a sampling rule is a strategy nobody would write correctly; here `random` is a
dozen lines and cannot get any of that wrong.

**No cache of its own** (DR-0003). ARCHITECTURE §6.2's insight — 24 combos ⇒ 1
parse, 2 chunk runs — is already materialized by the blob parse cache and
`CachingEmbedder`, both keyed by (content hash × fingerprint), which *is* the
formula §6.2 specifies. The tuner's contribution is **order**: `SearchSpace`
enumerates with the earliest stage varying slowest, so trials sharing an
expensive prefix run adjacent and inherit a warm cache. A second cache keyed on
the same thing would duplicate §7.2 and drift from it.
"""

from __future__ import annotations

import datetime
import random as _random
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from ..core.component import Component
from ..core.contracts import Query, Source
from ..core.errors import ConfigError, RagBlocksError
from ..core.registry import registry
from ..indexing.chunk_index import ChunkIndex
from ..pipeline import RagPipeline
from .base import EvalOutcome, EvalSample, Evaluator
from .builder import PipelineBuilder, PipelineFactory
from .cost import CostCollector
from .leaderboard import Leaderboard
from .space import STAGE_KINDS, SearchSpace
from .trial import Trial, trial_id_for
from .trial_log import TrialLog

__all__ = ["Tuner", "GridTuner", "RandomTuner"]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Tuner(Component):
    """Strategy interface: a search space → a ranked `Leaderboard`.

    Subclasses implement `iter_candidates` and nothing else.
    """

    kind = "tuner"

    @dataclass
    class Config:
        #: The phase-1 metric used to pick finalists. Must be one the
        #: retrieval evaluators actually report.
        screen_by: str = "ndcg@10"
        #: How many candidates survive into the expensive phase (§7.3: 5).
        finalists: int = 5

    @abstractmethod
    def iter_candidates(self, space: SearchSpace) -> Iterator[dict]:
        """The ONLY strategy decision: WHICH combinations, in what order.

        Order is part of the decision, not a detail: adjacent trials sharing an
        expensive prefix inherit a warm cache, so an implementation that
        shuffles the space pays for it in wall-clock (see `RandomTuner`).
        """

    # -- the Template Method -------------------------------------------------

    def run(
        self,
        space: SearchSpace,
        dataset: Sequence[EvalSample],
        sources: Source | Sequence[Source],
        *,
        evaluators: Sequence[Evaluator],
        build: Optional[PipelineFactory] = None,
        log: Optional[TrialLog] = None,
        prices: Optional[dict] = None,
        k: int = 5,
    ) -> Leaderboard:
        """Screen every candidate, judge the best few, log all of it.

        Two-phase (§7.3), because the evaluator families differ in cost by
        orders of magnitude:

        1. **Screen** — every candidate indexes and retrieves, scored by the
           `stage="retrieval"` evaluators. Free: pure math over rankings.
        2. **Judge** — only the top `finalists` re-run *with generation* and
           are scored by the `stage="generation"` evaluators too. This is the
           step that can cost cents a sample, so it is the step that must not
           run for all 24 combinations.

        Phase 2 rebuilds and re-runs a finalist rather than holding phase 1's
        pipelines in memory. Deliberate: keeping 24 live indexes to save a
        handful of cache-warm re-runs would break the rule that memory never
        scales with the search (and it proves config-as-data — a spec is
        genuinely sufficient to reconstruct a pipeline). A finalist's trial is
        therefore one complete run, not two halves stitched together.

        A candidate that raises is recorded with its error and the run
        continues: one bad combination must not take an overnight grid's other
        23 results with it.
        """
        if not dataset:
            raise ConfigError("Tuner.run() needs a non-empty dataset")
        if not evaluators:
            raise ConfigError("Tuner.run() needs at least one evaluator")

        screeners = [e for e in evaluators if e.stage == "retrieval"]
        judges = [e for e in evaluators if e.stage == "generation"]
        if not screeners:
            raise ConfigError(
                "Tuner.run() needs at least one stage='retrieval' evaluator to "
                "screen with (e.g. RetrievalEvaluator()); phase 1 ranks on it"
            )
        if self.config.finalists <= 0:
            raise ConfigError(
                f"finalists must be positive, got {self.config.finalists}"
            )
        build = build or PipelineBuilder().build

        # -- phase 1 --------------------------------------------------------
        trials: dict[str, Trial] = {}
        for spec in self.iter_candidates(space):
            trial = self._run_one(
                spec, dataset, sources, screeners, build, prices, k,
                generate=False, phase=1,
            )
            trials[trial.trial_id] = trial

        # -- phase 2 --------------------------------------------------------
        if judges:
            for spec in self._finalist_specs(trials):
                trial = self._run_one(
                    spec, dataset, sources, [*screeners, *judges], build,
                    prices, k, generate=True, phase=2,
                )
                trials[trial.trial_id] = trial

        if log is not None:
            for trial in trials.values():
                log.append(trial)
        return Leaderboard(list(trials.values()))

    def _finalist_specs(self, trials: dict[str, Trial]) -> list[dict]:
        """The specs of the best `finalists` phase-1 trials.

        Nothing scored on `screen_by` ⇒ no finalists ⇒ the judge never runs.
        That is the honest outcome (an unlabeled dataset earns no ranking), not
        a reason to judge an arbitrary five.
        """
        board = Leaderboard(list(trials.values()))
        if self.config.screen_by not in board.metrics():
            return []
        return [
            t.metadata["search_spec"]
            for t in board.top(self.config.finalists, by=self.config.screen_by)
            if "search_spec" in t.metadata
        ]

    # -- one candidate, end to end -------------------------------------------

    def _run_one(
        self, spec, dataset, sources, evaluators, build, prices, k,
        *, generate: bool, phase: int,
    ) -> Trial:
        """Build → index → retrieve (→ generate) → score → one `Trial`.

        One code path for both phases: `generate` is the only difference, so
        phase 2 cannot drift from phase 1's notion of what a trial is.
        """
        started = _now()
        collector = CostCollector(prices=prices)
        try:
            rag = build(spec)
            # The collector is a TraceHook; the sub-pipelines were built with
            # the factory's hook, so point all three at this trial's collector.
            rag.trace = collector
            rag.indexing.trace = collector
            rag.query_pipeline.trace = collector
            rag.index(sources)

            outcomes: list[EvalOutcome] = []
            for sample in dataset:
                query = Query(text=sample.question, filters=sample.filters)
                if generate:
                    # ask_with_context, not query()+generate(): the pipeline
                    # owns the "generate" trace event, and hand-rolling the
                    # loop would silently drop the costly stage from the bill.
                    answer, retrieved = rag.ask_with_context(query, k)
                    outcomes.append(
                        EvalOutcome(
                            sample=sample,
                            retrieved=tuple(retrieved),
                            answer=answer,
                        )
                    )
                else:
                    retrieved = rag.query_pipeline.query(query, k)
                    outcomes.append(
                        EvalOutcome(sample=sample, retrieved=tuple(retrieved))
                    )

            metrics: dict[str, float] = {}
            for evaluator in evaluators:
                metrics.update(evaluator.evaluate(outcomes).metrics)
        except RagBlocksError as exc:
            return self._failed(spec, collector, started, exc, phase)

        described = _describe(rag)
        return Trial(
            # Identity comes from what ACTUALLY ran (the resolved describes),
            # not from the search spec: two spellings of the same pipeline are
            # the same trial, and a component version bump is a new one.
            trial_id=trial_id_for(described),
            pipeline_spec=described,
            fingerprints=_fingerprints(rag),
            metrics=metrics,
            cost=collector.cost(),
            cache_hits=collector.cache_hits(),
            started_at=started,
            finished_at=_now(),
            metadata={
                "phase": phase,
                "samples": len(dataset),
                # Kept so phase 2 can rebuild a finalist from the same input
                # the strategy chose.
                "search_spec": spec,
            },
        )

    def _failed(self, spec, collector, started, exc, phase) -> Trial:
        return Trial(
            # No pipeline to describe, so the id falls back to the search spec.
            trial_id=trial_id_for(spec),
            pipeline_spec=spec,
            fingerprints={},
            cost=collector.cost(),
            cache_hits=collector.cache_hits(),
            started_at=started,
            finished_at=_now(),
            metadata={
                "phase": phase,
                "search_spec": spec,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


@registry.register
class GridTuner(Tuner):
    """Every combination, in the space's own (cache-warm) order.

    Exhaustive and deterministic — the baseline every other strategy is
    measured against, and the right choice whenever the grid is small enough
    to afford, because it is the only one that cannot miss the winner.
    """

    name = "grid"
    version = "0.1.0"

    def iter_candidates(self, space: SearchSpace) -> Iterator[dict]:
        # `SearchSpace.expand()` already orders prefix-major, so the grid
        # inherits warm caches for free. That is why this is a one-liner.
        return space.expand()


@registry.register
class RandomTuner(Tuner):
    """`n_trials` combinations sampled without replacement, seeded.

    Random search beats grid search when a few dimensions dominate the result
    (Bergstra & Bengio 2012): a grid spends its budget re-testing the axes that
    don't matter, while sampling covers more distinct values of the ones that
    do. The honest cost, stated: sampling **breaks the prefix ordering**, so
    neighbouring trials rarely share a parse and the run is wall-clock slower
    per trial than a grid of the same size.
    """

    name = "random"
    version = "0.1.0"

    @dataclass
    class Config(Tuner.Config):
        n_trials: int = 10
        #: Seeded by default: a tuning run that can't be reproduced isn't
        #: evidence. Pass seed=None only if you want the irreproducibility.
        seed: Optional[int] = 0

    def iter_candidates(self, space: SearchSpace) -> Iterator[dict]:
        if self.config.n_trials <= 0:
            raise ConfigError(
                f"RandomTuner: n_trials must be positive, got {self.config.n_trials}"
            )
        combinations = list(space.expand())
        rng = _random.Random(self.config.seed)
        if self.config.n_trials >= len(combinations):
            # Asking for more samples than the space holds is not an error —
            # it is a small space. Return all of it (still shuffled, so the
            # trial ORDER stays the strategy's own), rather than sampling with
            # replacement and billing twice for the same pipeline.
            rng.shuffle(combinations)
            return iter(combinations)
        return iter(rng.sample(combinations, self.config.n_trials))


def _describe(rag: RagPipeline) -> dict:
    """Every stage's `describe()`, keyed by stage — secrets already redacted.

    This is what makes a trial reproducible from its log line alone, and it
    reports what was *resolved*, not what was asked for: a spec that omits the
    retriever still records the one the pipeline derived.

    The index's representations are unpacked into their own stage keys
    (`embedder`, `sparse`, `lexical`). `ChunkIndex.describe()` reports them as
    fingerprints — right for identity, useless for reading: a hash cannot tell
    you the run used `hashing(dimensions=128)`. Without this a trial could not
    be reconstructed from its log line, and the leaderboard could not group by
    the embedder the tuner was searching over.
    """
    described: dict[str, Any] = {
        "parser": rag.indexing.parser.describe(),
        "chunker": rag.indexing.chunker.describe(),
        "retriever": rag.retriever.describe(),
        "generator": rag.generator.describe(),
    }
    index = rag.chunk_index
    if index is not None:
        described["index"] = index.describe()
        described.update(_representations(index))
    # Chain stages are recorded even when EMPTY. The empty chain is a choice —
    # "no reranker" is the baseline a cross-encoder has to beat — so omitting
    # it would hide the option the comparison exists to make. It would also
    # make the leaderboard's marginal for that stage compare the refiners
    # against nothing, silently.
    described["enrich"] = [e.describe() for e in rag.indexing.enrich]
    described["refine"] = [r.describe() for r in rag.query_pipeline.refine]
    return described


def _representations(index: ChunkIndex) -> dict[str, Any]:
    """The index's representation components, keyed by SearchSpace stage name.

    Grouped by each component's own `kind` and mapped back through
    `STAGE_KINDS`, so this never has to know how a `ChunkIndex` stores its
    representations — it asks.

    Single-representation mounts (the common case) report one describe; a
    multi-representation mount reports a list, so both forms of progressive
    disclosure survive into the log.
    """
    kind_to_stage = {kind: stage for stage, kind in STAGE_KINDS.items()}
    grouped: dict[str, list[Any]] = {}
    for component in index.encoders().values():
        stage = kind_to_stage.get(component.kind)
        if stage is not None:
            grouped.setdefault(stage, []).append(component.describe())
    return {
        stage: described[0] if len(described) == 1 else described
        for stage, described in grouped.items()
    }


def _fingerprints(rag: RagPipeline) -> dict[str, str]:
    """Per-stage cache keyspace. Derivable from the describes, stored anyway:
    it is what you group and join on, and rederiving it later would mean
    reimplementing a hash convention."""
    prints: dict[str, str] = {
        "parser": rag.indexing.parser.fingerprint(),
        "chunker": rag.indexing.chunker.fingerprint(),
        "retriever": rag.retriever.fingerprint(),
        "generator": rag.generator.fingerprint(),
    }
    if rag.chunk_index is not None:
        prints["index"] = rag.chunk_index.fingerprint()
    for i, enricher in enumerate(rag.indexing.enrich):
        prints[f"enrich.{i}"] = enricher.fingerprint()
    for i, refiner in enumerate(rag.query_pipeline.refine):
        prints[f"refine.{i}"] = refiner.fingerprint()
    return prints
