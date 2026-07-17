"""Evaluator: the scoring Strategy interface (DR-0002).

An Evaluator answers one question: *how good was this?* It scores outcomes the
pipeline already produced — it never runs the pipeline itself. That split is
the whole design (DR-0002):

- IR metrics collapse to pure math over `(retrieved_ids, relevant_ids)`, so
  they are hermetic, instant, and verifiable by hand.
- An LLM-judged Adapter (RAGAS) becomes a pure *translator* of our data into
  the vendor's shape, with no pipeline-driving code to get wrong.
- Nothing here imports `RagPipeline`, so evaluation never couples backward
  into orchestration, and no evaluator reimplements a run loop.

The run loop lives once in the tuner, which owns the pipeline, the cache and
the cost bookkeeping. Evaluators stay a function of (config, outcomes).

`stage` splits the two families by *cost*, not by taxonomy: retrieval metrics
are free (microseconds, no network), generation metrics can cost cents per
sample (an LLM judge). The tuner reads `stage` to screen every candidate on
the free family and spend the expensive one only on the finalists (§7.3's
two-phase evaluation).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Sequence

from ..core.component import Component
from ..core.contracts import Answer, ScoredChunk

__all__ = ["EvalSample", "EvalOutcome", "MetricReport", "Evaluator"]


@dataclass(frozen=True)
class EvalSample:
    """One labeled row of a user's evaluation dataset.

    Frozen because a dataset is an input fact, not working state: the same
    sample is read by every trial in a tuning run, and a mutation halfway
    through would silently invalidate the comparison.

    Every label field is optional and independent — a dataset labeled only
    with `relevant_chunk_ids` supports retrieval metrics, one labeled only
    with `reference_answer` supports generation metrics, and either is a
    legitimate way to work. An evaluator that needs a label it wasn't given
    skips that sample rather than inventing one.

    **Choosing a retrieval label** — the choice is granularity, and it decides
    what you are allowed to tune:

    - `relevant_chunk_ids` is exact, and **chunker-locked**. `Chunk.id` is
      `{doc_id}:{index}`, so the same id denotes a *different passage* under a
      different chunker — and may not exist at all under a coarser one. Labels
      like these are only valid while the chunker is held fixed. Nothing can
      detect the mismatch for you: the score stays plausible and becomes wrong.
    - `relevant_doc_ids` is coarser and **survives any chunking**, because a
      document's identity is its content hash, not a cut decision. It is what
      makes chunk size tunable — the one knob ARCHITECTURE §6.4 leads with
      ("chunk size 512→1024 costs −0.04 recall@10") — and it is what the
      committed baseline benchmark uses.

    Given both, chunk-level wins: it is the more specific claim.
    """

    question: str
    relevant_chunk_ids: Optional[tuple[str, ...]] = None
    #: Document-level ground truth (`Document.id` / `Source.content_hash()`).
    #: Chunker-independent — see the note above.
    relevant_doc_ids: Optional[tuple[str, ...]] = None
    reference_answer: Optional[str] = None
    filters: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalOutcome:
    """What one pipeline actually produced for one sample.

    This is the unit evaluators consume — the seam that keeps them off the
    pipeline. `retrieved` is post-refinement, in final rank order (higher
    first, the `ScoredChunk` guarantee). `answer` is None when only the
    retrieval half ran, which is exactly what phase 1 of a two-phase tuning
    run produces.
    """

    sample: EvalSample
    retrieved: tuple[ScoredChunk, ...] = ()
    answer: Optional[Answer] = None


@dataclass(frozen=True)
class MetricReport:
    """Scores for one evaluator over one set of outcomes.

    `metrics` is the aggregate that lands in `Trial.metrics` and ranks the
    leaderboard. `per_sample` is optional detail (same order and length as the
    outcomes when present) — it is what turns "recall@5 = 0.6" into "and here
    are the four questions that failed", which is the difference between a
    score and a diagnosis.
    """

    metrics: dict[str, float]
    per_sample: tuple[dict[str, float], ...] = ()


class Evaluator(Component):
    """Strategy interface: outcomes → scores.

    Subclass contract, beyond `Component`'s:
        stage: "retrieval" (free, pure math) or "generation" (may cost money).
               The tuner's two-phase screening keys off this.
    """

    kind = "evaluator"

    #: Declared on the concrete class. Not a config field: it is a fact about
    #: what the evaluator measures, so it must not vary per instance (the
    #: tuner decides *when* to run an evaluator from this).
    stage: ClassVar[Literal["retrieval", "generation"]]

    @abstractmethod
    def evaluate(self, outcomes: Sequence[EvalOutcome]) -> MetricReport:
        """Score `outcomes`. Pure: no I/O beyond this evaluator's own backend,
        no mutation of the outcomes. Raise `EvaluationError` on failure."""

    def _aggregate(
        self, per_sample: Sequence[dict[str, float]]
    ) -> dict[str, float]:
        """Mean of each metric over the samples that actually scored it.

        Shared plumbing rather than per-evaluator arithmetic, so the rule that
        matters holds everywhere at once: a sample an evaluator skipped (no
        label for it) is absent from the average, never a zero in it. Zeros
        would read as "the pipeline failed" when the truth is "we never
        asked".
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for scores in per_sample:
            for metric, value in scores.items():
                totals[metric] = totals.get(metric, 0.0) + value
                counts[metric] = counts.get(metric, 0) + 1
        return {metric: totals[metric] / counts[metric] for metric in sorted(totals)}
