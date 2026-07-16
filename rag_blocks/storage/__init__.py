"""Storage subsystem: blob stores, vector stores, and the lexical index.

Importing this package registers the built-in stores (module import is the
registration side effect the registry relies on): blob stores are the
pipeline's durable source of truth; vector stores and the classic-BM25
lexical index hold the derived, rebuildable retrieval representations.
"""

from .base import BlobStore
from .bm25_index import BM25Index
from .lexical_index import LexicalIndex
from .local import LocalBlobStore
from .memory_store import MemoryVectorStore
from .minio_store import MinioBlobStore
from .qdrant_store import QdrantVectorStore
from .vector_store import VectorStore

__all__ = [
    "BlobStore",
    "LocalBlobStore",
    "MinioBlobStore",
    "VectorStore",
    "MemoryVectorStore",
    "QdrantVectorStore",
    "LexicalIndex",
    "BM25Index",
]
