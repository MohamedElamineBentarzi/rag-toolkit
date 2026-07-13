"""Retrieval subsystem: Query → ranked ScoredChunks.

Importing this package registers the built-in retrievers. They are thin
Strategies over the storage backends (vector store, lexical index); a future
`hybrid` retriever composes them with a fusion strategy (RRF).
"""

from .base import Retriever
from .bm25 import Bm25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever

__all__ = [
    "Retriever",
    "DenseRetriever",
    "Bm25Retriever",
    "HybridRetriever",
]
