"""Reranking subsystem: second-pass precision over retrieved candidates.

Importing this package registers the built-in rerankers. `noop` (Null Object,
the tuner baseline) ships here; heavy cross-encoder adapters (bge-reranker,
cohere) arrive behind optional extras.
"""

from .base import Reranker
from .bge import BgeReranker
from .keyword import KeywordReranker
from .noop import NoOpReranker

__all__ = [
    "Reranker",
    "NoOpReranker",
    "KeywordReranker",
    "BgeReranker",
]
