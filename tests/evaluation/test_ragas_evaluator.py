"""RagasEvaluator: OUR translation, not RAGAS's judging. Hermetic.

The vendor is faked through `sys.modules` so these tests exercise the Adapter's
actual job — mapping EvalOutcome -> SingleTurnSample and scores -> MetricReport
— with no key, no network, and no ragas installed. A real-stack test lives in
tests/integration/ (AGENTS.md §9: "test OUR logic, not vendors'").
"""
from __future__ import annotations

import sys
import types

import pytest

from rag_blocks.core.contracts import Answer, Chunk, ScoredChunk
from rag_blocks.core.errors import EvaluationError
from rag_blocks.evaluation import EvalOutcome, EvalSample, RagasEvaluator
from rag_blocks.storage.local import LocalBlobStore
from tests.contract_checks import assert_evaluator_contract


def outcome(question="q", answer="an answer", reference="the reference", contexts=("ctx",)):
    return EvalOutcome(
        sample=EvalSample(question=question, reference_answer=reference),
        retrieved=tuple(
            ScoredChunk(
                chunk=Chunk(id=f"d:{i}", doc_id="d", text=text, index=i), score=1.0
            )
            for i, text in enumerate(contexts)
        ),
        answer=Answer(text=answer) if answer is not None else None,
    )


class _FakeResult:
    def __init__(self, scores):
        self.scores = scores


def fake_ragas(monkeypatch, *, scores=None, record=None, explode=False):
    """Install a fake `ragas` + `ragas.metrics` for the duration of a test."""

    class SingleTurnSample:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class EvaluationDataset:
        def __init__(self, samples):
            self.samples = samples

    def evaluate(dataset=None, metrics=None, llm=None, embeddings=None):
        if explode:
            raise RuntimeError("vendor exploded")
        if record is not None:
            record.append(
                {"dataset": dataset, "metrics": metrics, "llm": llm,
                 "embeddings": embeddings}
            )
        rows = scores if scores is not None else [
            {"faithfulness": 0.9} for _ in dataset.samples
        ]
        return _FakeResult(rows)

    ragas = types.ModuleType("ragas")
    ragas.SingleTurnSample = SingleTurnSample
    ragas.EvaluationDataset = EvaluationDataset
    ragas.evaluate = evaluate

    metrics_mod = types.ModuleType("ragas.metrics")
    # The NEWER spelling: classes.
    for cls_name in (
        "Faithfulness", "ResponseRelevancy",
        "LLMContextPrecisionWithReference", "LLMContextRecall",
    ):
        setattr(metrics_mod, cls_name, type(cls_name, (), {}))

    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_mod)
    return ragas


# -- construction: fail fast, no vendor needed ---------------------------


def test_it_registers_and_describes_without_ragas_installed():
    # The lazy import must not fire at construction, or the whole evaluation
    # subsystem would need the extra to import.
    described = RagasEvaluator(judge_model="gpt-4o").describe()
    assert described["name"] == "ragas"
    assert described["config"]["judge_model"] == "gpt-4o"


def test_it_is_a_generation_evaluator():
    assert RagasEvaluator.stage == "generation"


def test_an_unknown_metric_fails_at_construction_not_after_40_minutes():
    with pytest.raises(EvaluationError, match="unknown metric"):
        RagasEvaluator(metrics=("faithfulness", "vibes"))


def test_empty_metrics_fail_fast():
    with pytest.raises(EvaluationError, match="must not be empty"):
        RagasEvaluator(metrics=())


def test_the_judge_model_is_fingerprint_input():
    # It cannot be derived (the LLM is an opaque vendor object), so this label
    # IS the judge's identity — two trials judged by different models must not
    # share a fingerprint.
    assert RagasEvaluator(judge_model="gpt-4o").fingerprint() != (
        RagasEvaluator(judge_model="claude-opus-4-8").fingerprint()
    )


def test_there_is_no_api_key_field_to_leak():
    # Credentials live inside the injected LLM / ragas's own env resolution
    # (§7.4). A field we never read would be a liability, not a feature.
    assert "api_key" not in RagasEvaluator().describe()["config"]


def test_a_missing_ragas_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", None)
    with pytest.raises(EvaluationError, match=r"pip install 'rag-blocks\[ragas\]'"):
        RagasEvaluator().evaluate([outcome()])


# -- the translation (our actual job) ------------------------------------


def test_outcomes_become_single_turn_samples(monkeypatch):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    RagasEvaluator().evaluate(
        [outcome(question="Q?", answer="A.", reference="R.", contexts=("c1", "c2"))]
    )

    sample = record[0]["dataset"].samples[0]
    assert sample.user_input == "Q?"
    assert sample.response == "A."
    assert sample.reference == "R."
    assert sample.retrieved_contexts == ["c1", "c2"]  # chunk TEXT, not chunks


def test_scores_come_back_as_a_metric_report(monkeypatch):
    fake_ragas(monkeypatch, scores=[{"faithfulness": 0.8}, {"faithfulness": 0.6}])
    report = RagasEvaluator().evaluate([outcome(question="a"), outcome(question="b")])

    assert report.metrics["faithfulness"] == pytest.approx(0.7)  # the mean
    assert report.per_sample[0]["faithfulness"] == 0.8


def test_the_injected_llm_is_passed_through(monkeypatch):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    sentinel, embeddings = object(), object()
    RagasEvaluator(llm=sentinel, embeddings=embeddings).evaluate([outcome()])

    assert record[0]["llm"] is sentinel
    assert record[0]["embeddings"] is embeddings


def test_configured_metrics_are_resolved_and_passed(monkeypatch):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    RagasEvaluator(metrics=("faithfulness", "context_recall")).evaluate([outcome()])
    assert len(record[0]["metrics"]) == 2


def test_a_vendor_error_is_normalized(monkeypatch):
    fake_ragas(monkeypatch, explode=True)
    with pytest.raises(EvaluationError, match="RAGAS evaluation failed"):
        RagasEvaluator().evaluate([outcome()])


def test_misaligned_scores_are_refused_rather_than_zipped(monkeypatch):
    # A short return would otherwise silently attach verdicts to the wrong
    # questions — corrupt, not crashing.
    fake_ragas(monkeypatch, scores=[{"faithfulness": 0.9}])
    with pytest.raises(EvaluationError, match="cannot align"):
        RagasEvaluator().evaluate([outcome(question="a"), outcome(question="b")])


def test_a_result_without_scores_says_the_api_drifted(monkeypatch):
    ragas = fake_ragas(monkeypatch)
    ragas.evaluate = lambda **kw: object()
    with pytest.raises(EvaluationError, match="drifted"):
        RagasEvaluator().evaluate([outcome()])


def test_an_abstained_verdict_is_absent_not_zero(monkeypatch):
    # A judge reporting None abstained; 0.0 would score the answer as
    # maximally unfaithful (DR-0002 §4).
    fake_ragas(monkeypatch, scores=[{"faithfulness": None, "context_recall": 0.5}])
    report = RagasEvaluator().evaluate([outcome()])
    assert "faithfulness" not in report.metrics
    assert report.metrics["context_recall"] == 0.5


# -- what gets sent to a judge at all ------------------------------------


@pytest.mark.parametrize(
    "unscoreable, why",
    [
        (EvalOutcome(sample=EvalSample(question="q", reference_answer="r")), "no answer"),
        (
            EvalOutcome(
                sample=EvalSample(question="q"),
                retrieved=(ScoredChunk(chunk=Chunk(id="d:0", doc_id="d", text="c", index=0), score=1.0),),
                answer=Answer(text="a"),
            ),
            "no reference",
        ),
        (
            EvalOutcome(
                sample=EvalSample(question="q", reference_answer="r"),
                answer=Answer(text="a"),
            ),
            "no contexts",
        ),
    ],
)
def test_unjudgeable_samples_are_skipped_not_billed(monkeypatch, unscoreable, why):
    # Sending a judge a sample with no contexts bills for a verdict on nothing.
    record: list = []
    fake_ragas(monkeypatch, record=record)
    report = RagasEvaluator().evaluate([unscoreable])

    assert record == [], f"the judge was called for a sample with {why}"
    assert report.metrics == {}
    assert report.per_sample == ({},)


def test_only_the_scoreable_samples_reach_the_judge(monkeypatch):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    report = RagasEvaluator().evaluate(
        [outcome(question="good"), EvalOutcome(sample=EvalSample(question="bare"))]
    )

    assert len(record[0]["dataset"].samples) == 1
    assert report.per_sample[1] == {}  # alignment with outcomes still holds
    assert len(report.per_sample) == 2


# -- the cache -----------------------------------------------------------


def test_a_cached_verdict_is_not_re_judged(monkeypatch, tmp_path):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    blobs = LocalBlobStore(root=str(tmp_path))

    first = RagasEvaluator(cache=blobs, judge_model="j1").evaluate([outcome()])
    second = RagasEvaluator(cache=blobs, judge_model="j1").evaluate([outcome()])

    assert len(record) == 1, "the second run re-billed for a verdict it had"
    assert first.metrics == second.metrics


def test_only_the_uncached_samples_are_sent(monkeypatch, tmp_path):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    blobs = LocalBlobStore(root=str(tmp_path))
    judge = RagasEvaluator(cache=blobs, judge_model="j1")

    judge.evaluate([outcome(question="a")])
    judge.evaluate([outcome(question="a"), outcome(question="b")])

    # Second call sent ONLY the new question.
    assert len(record[1]["dataset"].samples) == 1
    assert record[1]["dataset"].samples[0].user_input == "b"


def test_the_cache_makes_a_judged_evaluator_satisfy_the_contract(monkeypatch, tmp_path):
    # The contract check asserts scoring is pure (same outcomes, same report).
    # An LLM judge only satisfies that through its verdict cache — which is
    # why §7.3 requires the cache (DR-0002 §8).
    fake_ragas(monkeypatch)
    assert_evaluator_contract(
        RagasEvaluator(cache=LocalBlobStore(root=str(tmp_path)), judge_model="j1"),
        [outcome(question="a"), outcome(question="b")],
    )


def test_without_a_cache_it_still_works_just_dearer(monkeypatch):
    record: list = []
    fake_ragas(monkeypatch, record=record)
    judge = RagasEvaluator()
    judge.evaluate([outcome()])
    judge.evaluate([outcome()])
    assert len(record) == 2  # no cache, no memory — the Null Object


# -- metric resolution across ragas versions -----------------------------


def test_the_older_singleton_metric_api_still_resolves(monkeypatch):
    # Ragas has shipped both classes and module-level singletons; betting on
    # one spelling would break on the other.
    fake_ragas(monkeypatch)
    old = types.ModuleType("ragas.metrics")
    old.faithfulness = object()  # the old lowercase singleton, no class
    monkeypatch.setitem(sys.modules, "ragas.metrics", old)

    resolved = RagasEvaluator(metrics=("faithfulness",))._resolve_metric("faithfulness")
    assert resolved is old.faithfulness


def test_a_metric_this_ragas_version_lacks_is_an_actionable_error(monkeypatch):
    fake_ragas(monkeypatch)
    monkeypatch.setitem(sys.modules, "ragas.metrics", types.ModuleType("ragas.metrics"))
    with pytest.raises(EvaluationError, match="drifted"):
        RagasEvaluator()._resolve_metric("faithfulness")
