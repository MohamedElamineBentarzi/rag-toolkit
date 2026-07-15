"""Storage subsystem: durable byte storage (the pipeline's source of truth).

Importing this package registers the built-in blob stores (module import is the
registration side effect the registry relies on). The vector store and lexical
index kinds will join this package in the v0.3 storage milestone.
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
