"""Refinement subsystem: the uniform post-retrieval chain (DR-0001 v2, D9).

Importing this package registers the built-in refiners. Every stage has one
shape — `refine(query, candidates, k) -> list[ScoredChunk]` — so they compose in
a list and reorder freely (the MongoDB-aggregation half of the composition
algebra). This is where the old `reranker` kind dissolved: a cross-encoder is
just one refiner among expansion, thresholds, and diversification.

Hermetic refiners (`keyword`, `score-threshold`, `neighbor-expander`) run in the
test suite and the tuner; `cross-encoder` is the real-model Adapter behind an
optional extra.
"""

from .base import Refiner
from .cross_encoder import CrossEncoderReranker
from .keyword import KeywordRefiner
from .neighbor import NeighborExpander
from .threshold import ScoreThreshold

__all__ = [
    "Refiner",
    "CrossEncoderReranker",
    "KeywordRefiner",
    "NeighborExpander",
    "ScoreThreshold",
]
