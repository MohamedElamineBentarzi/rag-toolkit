"""ScoreThreshold: drop candidates below a relevance floor.

The simplest refiner, and a genuinely useful one: retrieval always returns
`fetch_k` candidates whether or not any are relevant, so a low-similarity tail
rides along and can pollute the generator's context. `ScoreThreshold` keeps only
candidates scoring `>= min_score`, in the order they arrived (it reorders
nothing). Placed *after* a reranker, it floors on the reranker's calibrated
scores; placed before, on the retriever's.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.contracts import Query, ScoredChunk
from ..core.registry import registry
from .base import Refiner

__all__ = ["ScoreThreshold"]


@registry.register
class ScoreThreshold(Refiner):
    name = "score-threshold"
    version = "0.1.0"

    @dataclass
    class Config:
        min_score: float = 0.0

    def refine(
        self, query: Query, candidates: list[ScoredChunk], k: int
    ) -> list[ScoredChunk]:
        floor = self.config.min_score
        return [sc for sc in candidates if sc.score >= floor]
