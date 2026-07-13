"""NoOpEnricher: the Null Object enricher.

Passes chunks through untouched. It earns a name (rather than
`enricher=None`) so pipeline code always calls `enrich(...)` with no
`if enricher is not None` branch — and so "does enrichment help?" is answered by
putting `noop` in the tuner's search space beside the real enrichers.
"""

from __future__ import annotations

from typing import Iterator

from ..core.contracts import Chunk, Document
from ..core.registry import registry
from .base import Enricher

__all__ = ["NoOpEnricher"]


@registry.register
class NoOpEnricher(Enricher):
    name = "noop"
    version = "0.1.0"

    def enrich(
        self, chunks: Iterator[Chunk], document: Document
    ) -> Iterator[Chunk]:
        yield from chunks
