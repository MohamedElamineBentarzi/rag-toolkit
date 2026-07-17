"""RagasEvaluator: the LLM-judged half of generation scoring (Adapter, §7.3).

Pattern: Adapter, doing exactly what an Adapter should — translating our data
into a vendor's shape and its scores back into ours, with no logic of its own.
Because evaluators score data rather than drive pipelines (DR-0002), that is
*all* this file is: `EvalOutcome` → `SingleTurnSample` → `evaluate()` →
`MetricReport`. There is no run loop here to get wrong.

What it buys over `AnswerMatchEvaluator`, its hermetic sibling at the same seam:
a judge can see that "Berlin" and "The capital is Berlin" mean the same thing,
and that an answer is *unfaithful* to its context — neither of which token
overlap can do. What it costs: money per sample, an API key, and
non-determinism. Two implementations, real trade-offs, and `fingerprint()`
records which one produced a number.

**The judge LLM is injected, not configured.** Ragas takes a LangChain-wrapped
(or ragas-native) LLM, and building one from a model string would drag
`langchain` into a library whose core is stdlib-only. So you pass the object;
we pass it through. Credentials therefore live where §7.4 says they should —
inside the vendor's own resolution, never handled here (there is deliberately no
`api_key` field on this component: it would be a field we never read).

Left to itself, ragas defaults to OpenAI and reads `OPENAI_API_KEY` from the
environment. That is ragas's behavior, documented here rather than wrapped.

**API-drift guard (do not remove; re-verify on dependency bumps).** Verified
against the ragas docs on 2026-07-17, not against an installed package:
`SingleTurnSample(user_input, retrieved_contexts, response, reference)`,
`EvaluationDataset(samples=[...])`, `evaluate(dataset=, metrics=, llm=,
embeddings=)`, and `result.scores` as a list of per-sample dicts. Ragas has
shipped **two** metric APIs — module-level singletons (`faithfulness`) and
classes (`Faithfulness()`) — so `_resolve_metric` tries both rather than betting
on one. Same caution as the `mistralai` guard in §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..core.errors import EvaluationError
from ..core.registry import registry
from ..storage.base import BlobStore
from .base import EvalOutcome, Evaluator, MetricReport
from .judge_cache import JudgeCache

__all__ = ["RagasEvaluator"]

#: Our metric name → the ragas attribute names to try, newest spelling first.
#: Two spellings because ragas renamed these across versions; resolving by
#: attribute rather than by import means one dead name is a clear error, not an
#: ImportError at module load that takes the whole subsystem down.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("Faithfulness", "faithfulness"),
    "answer_relevancy": ("ResponseRelevancy", "AnswerRelevancy", "answer_relevancy"),
    "context_precision": ("LLMContextPrecisionWithReference", "ContextPrecision", "context_precision"),
    "context_recall": ("LLMContextRecall", "ContextRecall", "context_recall"),
}


@registry.register
class RagasEvaluator(Evaluator):
    """LLM-judged generation metrics via RAGAS.

        judge = RagasEvaluator(llm=my_wrapped_llm, cache=LocalBlobStore(),
                               judge_model="gpt-4o-2024-08-06")
    """

    name = "ragas"
    version = "0.1.0"
    stage = "generation"

    @dataclass
    class Config:
        #: Which RAGAS metrics to run. Defaults to §7.3's four.
        metrics: tuple[str, ...] = (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        #: **Who judged** — the identity of the injected LLM, as a label.
        #:
        #: It cannot be derived: the LLM arrives as an opaque vendor object, so
        #: it is invisible to `describe()` and to the fingerprint. This field is
        #: the stand-in, and it does real work: it is part of every judge-cache
        #: key AND part of this evaluator's fingerprint. **Change the judge,
        #: change this** — otherwise the cache serves one model's verdicts as
        #: another's, and two trials judged differently share a trial identity.
        judge_model: str = "ragas-default"

    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
        cache: Optional[BlobStore] = None,
        config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if not self.config.metrics:
            raise EvaluationError("RagasEvaluator: metrics must not be empty")
        unknown = set(self.config.metrics) - set(_METRIC_ALIASES)
        if unknown:
            # Fail at construction, not after 40 minutes of screening.
            raise EvaluationError(
                f"RagasEvaluator: unknown metric(s) {sorted(unknown)}; "
                f"known: {sorted(_METRIC_ALIASES)}"
            )
        self._llm = llm
        self._embeddings = embeddings
        self._cache = JudgeCache(cache, judge_model=self.config.judge_model)

    def evaluate(self, outcomes: Sequence[EvalOutcome]) -> MetricReport:
        """Score the outcomes that carry everything a judge needs.

        A sample is scoreable only with an answer, a reference, and contexts;
        anything else gets an empty per-sample entry and joins no average
        (DR-0002 §4). That is not fussiness — sending a judge a sample with no
        contexts would bill for a verdict on nothing.
        """
        per_sample: list[dict[str, float]] = [{} for _ in outcomes]
        pending: list[int] = []

        for i, outcome in enumerate(outcomes):
            if not _is_scoreable(outcome):
                continue
            assert outcome.answer is not None  # narrowed by _is_scoreable
            cached = self._cache.get(outcome.sample.question, outcome.answer.text)
            if cached is not None:
                per_sample[i] = {k: float(v) for k, v in cached.items()}
            else:
                pending.append(i)

        if pending:
            fresh = self._judge([outcomes[i] for i in pending])
            for i, scores in zip(pending, fresh):
                per_sample[i] = scores
                answer = outcomes[i].answer
                assert answer is not None
                self._cache.put(outcomes[i].sample.question, answer.text, scores)

        return MetricReport(
            metrics=self._aggregate(per_sample), per_sample=tuple(per_sample)
        )

    # -- the vendor boundary -------------------------------------------------

    def _judge(self, outcomes: Sequence[EvalOutcome]) -> list[dict[str, float]]:
        """One ragas call for the uncached samples. Everything vendor-shaped
        lives below this line."""
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
        except ImportError as exc:
            raise EvaluationError(
                "RagasEvaluator requires the 'ragas' package. "
                "Install with: pip install 'rag-blocks[ragas]'"
            ) from exc

        samples = []
        for outcome in outcomes:
            assert outcome.answer is not None
            samples.append(
                SingleTurnSample(
                    user_input=outcome.sample.question,
                    retrieved_contexts=[sc.chunk.text for sc in outcome.retrieved],
                    response=outcome.answer.text,
                    reference=outcome.sample.reference_answer,
                )
            )

        metrics = [self._resolve_metric(name) for name in self.config.metrics]
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

        return _read_scores(result, len(samples))

    def _resolve_metric(self, name: str) -> Any:
        """Our metric name → a ragas metric object, across ragas versions.

        Classes first (the newer API), then module-level singletons (the older
        one). Resolving by attribute rather than by import keeps a renamed
        metric a clear, actionable error instead of an ImportError that fires
        at module load and takes the whole evaluation subsystem with it.
        """
        try:
            import ragas.metrics as ragas_metrics
        except ImportError as exc:
            raise EvaluationError(
                "RagasEvaluator requires the 'ragas' package. "
                "Install with: pip install 'rag-blocks[ragas]'"
            ) from exc

        for attribute in _METRIC_ALIASES[name]:
            candidate = getattr(ragas_metrics, attribute, None)
            if candidate is None:
                continue
            return candidate() if isinstance(candidate, type) else candidate

        raise EvaluationError(
            f"RagasEvaluator: this ragas version exposes none of "
            f"{list(_METRIC_ALIASES[name])} for metric {name!r} — the RAGAS "
            f"metric API has drifted; update _METRIC_ALIASES"
        )


def _is_scoreable(outcome: EvalOutcome) -> bool:
    return bool(
        outcome.answer is not None
        and outcome.sample.reference_answer
        and outcome.retrieved
    )


def _read_scores(result: Any, expected: int) -> list[dict[str, float]]:
    """`result.scores` → our per-sample dicts, without importing pandas.

    Ragas returns pandas-friendly objects, but requiring pandas to read a list
    of dicts would pull a numeric stack into a code path that needs none.
    """
    scores = getattr(result, "scores", None)
    if scores is None:
        raise EvaluationError(
            "RAGAS returned no `scores` — the result API has drifted; "
            "re-verify against the current ragas docs"
        )
    rows = list(scores)
    if len(rows) != expected:
        raise EvaluationError(
            f"RAGAS returned {len(rows)} score rows for {expected} samples; "
            f"cannot align verdicts with questions"
        )
    out: list[dict[str, float]] = []
    for row in rows:
        out.append(
            {
                str(k): float(v)
                for k, v in dict(row).items()
                # A judge that abstains reports None; that is an absent verdict,
                # and forcing it to 0.0 would score the answer as maximally
                # unfaithful (DR-0002 §4).
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
    return out
