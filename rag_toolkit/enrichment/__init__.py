"""Enrichment subsystem: augment a document's chunks with context.

Optional stage between chunking and embedding, composed as a chain
(`enrich=[...]`) on the write path — the *empty* chain is the null object, so
there is no `NoOpEnricher` (DR-0001 v2, D6). Importing this package registers the
built-ins: `heading` (zero-dep, deterministic contextual retrieval from markdown
structure) and `contextual` (LLM-backed Adapter behind the `[anthropic]` extra).
"""

from .base import Enricher
from .contextual import ContextualEnricher
from .heading import HeadingEnricher

__all__ = [
    "Enricher",
    "HeadingEnricher",
    "ContextualEnricher",
]
