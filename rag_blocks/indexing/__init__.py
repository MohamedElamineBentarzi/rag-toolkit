"""Indexing subsystem: the aggregate that owns a corpus's representations.

`ChunkIndex` is the flagship: it writes every retrieval representation
(dense/sparse/lexical) on `add`, and encodes queries the same way on `search`.
The write path can fan out to any `ChunkSink` (a `ChunkIndex` is the flagship
sink; a GraphRAG or alert index is just another one).
"""

from .catalog import DocumentCatalog, DocumentRef
from .chunk_index import ChunkIndex
from .sink import ChunkSink

__all__ = ["ChunkIndex", "ChunkSink", "DocumentCatalog", "DocumentRef"]
