"""Evaluation subsystem: scoring pipelines, honestly (DR-0002, §7.3).

Importing this package registers the built-in evaluators. Every evaluator has
one shape — `evaluate(outcomes) -> MetricReport` — and scores data the pipeline
already produced; the run loop belongs to the tuner, not here.

`stage` splits them by cost, which is what the tuner's two-phase screening
spends: `retrieval` evaluators are pure math (`ir`), `generation` evaluators
range from free string overlap (`answer-match`) to an LLM judge charging cents
a sample (`ragas`, later).
"""

from .answer_match import AnswerMatchEvaluator, exact_match, token_f1
from .base import EvalOutcome, EvalSample, Evaluator, MetricReport
from .builder import PipelineBuilder, PipelineFactory
from .cost import CostCollector
from .ir_metrics import RetrievalEvaluator, ndcg_at_k, recall_at_k, reciprocal_rank
from .judge_cache import JudgeCache
from .leaderboard import Leaderboard, Marginal
from .ragas_evaluator import RagasEvaluator
from .space import Choice, SearchSpace, choice
from .trial import Trial, trial_id_for
from .trial_log import TrialLog
from .tuning import GridTuner, RandomTuner, Tuner

__all__ = [
    "Evaluator",
    "EvalSample",
    "EvalOutcome",
    "MetricReport",
    "RetrievalEvaluator",
    "AnswerMatchEvaluator",
    "RagasEvaluator",
    "JudgeCache",
    "Marginal",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "token_f1",
    "exact_match",
    "Trial",
    "trial_id_for",
    "TrialLog",
    "CostCollector",
    "Leaderboard",
    "SearchSpace",
    "Choice",
    "choice",
    "PipelineBuilder",
    "PipelineFactory",
    "Tuner",
    "GridTuner",
    "RandomTuner",
]
