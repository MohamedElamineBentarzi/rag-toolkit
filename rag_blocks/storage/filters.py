"""Shared metadata-filter semantics for in-memory sinks.

One definition of "does this chunk match this filter dict", used by every
Python-side store (memory vector store, BM25 lexical index) so the semantics
cannot drift. A backend that filters natively (Qdrant translates these into its
own filter language) must reproduce *these* semantics; this module is the
reference.

Semantics (D3):

- **scalar value ⇒ equality** — ``{"index": 2}`` keeps chunks whose ``index`` is 2.
- **list/tuple/set value ⇒ membership** — ``{"doc_id": ["a", "b"]}`` keeps chunks
  whose ``doc_id`` is ``a`` or ``b``.
- **field-then-metadata resolution** — a key resolves to a ``Chunk`` field
  (``doc_id``, ``index``, …) first, falling back to ``chunk.metadata[key]``.
- **all keys are AND-ed** — every filter entry must hold.
- **empty/None filters match everything** (callers guard this, but it holds).
"""

from __future__ import annotations

from typing import Optional

from ..core.contracts import Chunk

__all__ = ["matches"]


def matches(chunk: Chunk, filters: Optional[dict]) -> bool:
    """True if ``chunk`` satisfies every entry in ``filters`` (see module doc)."""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = getattr(chunk, key, None)
        if actual is None:
            actual = chunk.metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
