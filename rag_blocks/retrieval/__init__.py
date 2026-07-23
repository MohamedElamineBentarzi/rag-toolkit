"""Retrieval subsystem: Query → ranked ScoredChunks, and the composition axis.

Importing this package registers the built-in retrievers. Retrieval is one
composition axis (DR-0001 v2, D5): read-only *views* over a `Corpus`
(`IndexRetriever`, `HybridRetriever`) and *composite* nodes that wrap retrievers
(`FusionRetriever`, `MultiQueryRetriever`, `HydeRetriever`) — retrievers wrapping
retrievers, like `nn.Module` contains `nn.Module`. Fusion mechanics (dedup by
`chunk.id`, RRF, attribution) live once in `fusion.py`.
"""

from .base import Retriever
from .fusion import fuse
from .fusion_retriever import FusionRetriever
from .hybrid import HybridRetriever
from .index_retriever import IndexRetriever
from .query_shaping import HydeRetriever, MultiQueryRetriever

__all__ = [
    "Retriever",
    "IndexRetriever",
    "FusionRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "HydeRetriever",
    "fuse",
]
