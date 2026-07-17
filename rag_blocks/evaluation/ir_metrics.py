"""Classic IR metrics: the free half of evaluation.

These are pure math over `(retrieved_ids, relevant_ids)` — no model, no
network, no vendor, microseconds per sample. That is what makes the tuner's
two-phase screening possible: every candidate pipeline can be scored on these,
and only the finalists pay for an LLM judge (§7.3).

The functions come first and the `Evaluator` wraps them, not the other way
round: a ranking metric is exactly the kind of thing that should be verifiable
against a hand-computed number in a test, with no component, config, or
dataclass in the way ("testability is the first consumer of the architecture").

**Relevance is binary here.** `EvalSample.relevant_chunk_ids` is a set, not a
grade map, so nDCG uses gain 1 for a hit and 0 otherwise. That is the honest
shape for the labels we accept; graded relevance would need a contract change,
and inventing grades from a set would be a lie dressed as a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from ..core.errors import ConfigError
from ..core.registry import registry
from .base import EvalOutcome, Evaluator, MetricReport

__all__ = ["recall_at_k", "reciprocal_rank", "ndcg_at_k", "RetrievalEvaluator"]


def _check(retrieved_ids: Sequence[str], relevant: Iterable[str]) -> set[str]:
    relevant_set = set(relevant)
    if not relevant_set:
        # A precondition, not a score: recall over an empty ground truth is
        # 0/0. Callers filter unlabeled samples out; reaching here is a bug,
        # and returning 0.0 would quietly poison an average.
        raise ValueError("relevant_ids must be non-empty to score a sample")
    return relevant_set


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Fraction of the relevant chunks that appear in the top `k`.

    The question a RAG author actually asks: *did retrieval put the answer in
    front of the generator at all?* No credit for rank — that is nDCG's job.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = _check(retrieved_ids, relevant_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str]
) -> float:
    """1 / rank of the first relevant hit; 0.0 if none was retrieved.

    Unbounded in `k` by design — averaged over a dataset this is MRR. The
    steep 1, 1/2, 1/3 decay encodes the real cost model: a context window
    puts the top hit in front of the generator first.
    """
    relevant = _check(retrieved_ids, relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Normalized discounted cumulative gain over binary relevance.

    DCG discounts each hit by log2(rank + 1); IDCG is the same sum for the
    perfect ranking (every relevant chunk first), so the result is 1.0 for an
    ideal ordering and comparable across questions with different numbers of
    relevant chunks — which plain DCG is not, and which is the whole reason to
    normalize.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = _check(retrieved_ids, relevant_ids)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in relevant
    )
    # The ideal ranking can only be as long as there are relevant chunks to
    # place (or `k` slots to place them in, whichever runs out first).
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@registry.register
class RetrievalEvaluator(Evaluator):
    """The free screening evaluator: recall@k, MRR, nDCG@k over a ranking.

    Scores at whichever granularity the sample is labeled at (see
    `EvalSample`): `relevant_chunk_ids` ranks chunks, `relevant_doc_ids` ranks
    the documents those chunks came from. The metrics are the same maths either
    way — only the identity being ranked changes, which is why doc-level
    support cost one helper and no new metric.

    Document ranking deduplicates: three chunks from one document are one hit
    at that document's best rank. Otherwise a chunker that emits small chunks
    would "find" the same document repeatedly and score higher for it — the
    exact bias that makes chunk-level labels useless for tuning chunk size.

    Unlabeled samples get an empty per-sample dict and contribute to no
    average. Consequence worth knowing: aggregates are means over the *labeled*
    subset, so a dataset where two rows of thirty carry labels reports a
    confident-looking number computed from two rows. `per_sample` reveals that.
    """

    name = "ir"
    version = "0.1.0"
    stage = "retrieval"

    @dataclass
    class Config:
        #: Cut-offs to report. Several because they answer different
        #: questions: recall@1 is "did it nail it", recall@10 is "is it in
        #: the context window at all".
        k_values: tuple[int, ...] = (1, 5, 10)

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        # Fail fast at construction, not on sample 500 of a tuning run.
        ks = tuple(self.config.k_values)
        if not ks:
            raise ConfigError("RetrievalEvaluator: k_values must not be empty")
        if any(not isinstance(k, int) or k <= 0 for k in ks):
            raise ConfigError(
                f"RetrievalEvaluator: k_values must be positive ints, got {ks!r}"
            )
        self._k_values = ks

    def evaluate(self, outcomes: Sequence[EvalOutcome]) -> MetricReport:
        per_sample: list[dict[str, float]] = []
        for outcome in outcomes:
            ranked = _ranked_ids(outcome)
            if ranked is None:
                per_sample.append({})  # unlabeled: no score, not a zero
                continue
            retrieved_ids, relevant = ranked
            scores: dict[str, float] = {"mrr": reciprocal_rank(retrieved_ids, relevant)}
            for k in self._k_values:
                scores[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
                scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
            per_sample.append(scores)

        return MetricReport(
            metrics=self._aggregate(per_sample), per_sample=tuple(per_sample)
        )


def _ranked_ids(
    outcome: EvalOutcome,
) -> Optional[tuple[list[str], tuple[str, ...]]]:
    """(retrieved ids in rank order, relevant ids) at the sample's granularity.

    Returns None when the sample carries no retrieval label at all. Chunk-level
    wins when both are given: it is the more specific claim.
    """
    sample = outcome.sample
    if sample.relevant_chunk_ids:
        return [sc.chunk.id for sc in outcome.retrieved], sample.relevant_chunk_ids
    if sample.relevant_doc_ids:
        return _dedupe([sc.chunk.doc_id for sc in outcome.retrieved]), sample.relevant_doc_ids
    return None


def _dedupe(ids: Sequence[str]) -> list[str]:
    """First occurrence wins, order preserved.

    A document's rank is its BEST chunk's rank. Counting it once per chunk
    would reward a chunker for cutting small — it would fill the top-k with
    one document and call that recall.
    """
    seen: set[str] = set()
    out: list[str] = []
    for identifier in ids:
        if identifier not in seen:
            seen.add(identifier)
            out.append(identifier)
    return out
