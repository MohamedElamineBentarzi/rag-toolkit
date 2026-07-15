"""Embedding subsystem: text → dense vectors.

Importing this package registers the built-in embedders (module import is the
registration side effect the registry relies on). `hashing` is zero-dependency
and deterministic (tests, baselines, small corpora); `sentence-transformers` is
the real-model Adapter behind an optional extra.
"""

from .base import Embedder
from .caching import CachingEmbedder
from .hashing import HashingEmbedder
from .sentence_transformer import SentenceTransformerEmbedder
from .sparse import SparseEncoder

__all__ = [
    "Embedder",
    "SparseEncoder",
    "CachingEmbedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
]
