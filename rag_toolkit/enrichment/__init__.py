"""Enrichment subsystem: augment a document's chunks with context.

Optional stage between chunking and embedding. Importing this package registers
the built-ins: `noop` (Null Object), `heading` (zero-dep, deterministic
contextual retrieval from markdown structure), and `contextual` (LLM-backed
Adapter behind the `[anthropic]` extra).
"""

from .base import Enricher
from .contextual import ContextualEnricher
from .heading import HeadingEnricher
from .noop import NoOpEnricher

__all__ = [
    "Enricher",
    "NoOpEnricher",
    "HeadingEnricher",
    "ContextualEnricher",
]
