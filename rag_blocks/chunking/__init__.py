"""Chunking subsystem: Document → stream of retrieval Chunks.

Importing this package registers the built-in chunkers (module import is the
registration side effect the registry relies on). Strategies decide only WHERE
to cut (char-offset spans); the Template Method in `base.Chunker.chunk` owns all
bookkeeping and provenance.
"""

from .base import Chunker
from .fixed import FixedChunker
from .markdown import MarkdownChunker

__all__ = [
    "Chunker",
    "FixedChunker",
    "MarkdownChunker",
]
