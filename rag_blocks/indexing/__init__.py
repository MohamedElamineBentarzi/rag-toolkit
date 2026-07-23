"""Indexing subsystem: representations of a corpus, and the coordinator over them.

A `Corpus` (DR-0004) is the single owner of the `VectorStore`; it coordinates a
list of first-class `Representation` strategies (dense, sparse, lexical, or any
registered plugin), driving a single-pass write and owning all search I/O. The
write path can fan out to any `ChunkSink` (a `Corpus` is the flagship sink; a
GraphRAG or alert index is just another one).

The `Corpus` replaces the former `ChunkIndex` aggregate (DR-0004).
"""

from .catalog import DocumentCatalog, DocumentRef
from .corpus import Corpus
from .representation import (
    DenseRepresentation,
    LexicalRepresentation,
    Representation,
    SparseRepresentation,
)
from .sink import ChunkSink

__all__ = [
    "Corpus",
    "Representation",
    "DenseRepresentation",
    "SparseRepresentation",
    "LexicalRepresentation",
    "ChunkSink",
    "DocumentCatalog",
    "DocumentRef",
]
