"""Enricher: the optional chunk-augmentation Strategy.

An enricher sits between the Chunker and the Embedder in the indexing flow. It
gets each document's chunks AND the parent `Document`, because context is
exactly what a lone chunk lacks — the classic example is *contextual retrieval*:
prepend a one-sentence situating summary so a chunk that says "revenue rose 18%"
also carries "this is the Q3 report for Acme Corp", which makes it findable.

Contract notes:
- It receives an iterator of chunks + the document; it returns an iterator of
  chunks (a Strategy over a stream — augment, add synthetic chunks, or pass
  through). Chunks keep `doc_id` pointing at the document.
- Augmenting a chunk's *text* (e.g. prepending context) is allowed and expected;
  such a chunk is no longer a verbatim slice of `document.markdown`, so the
  chunker's slice-equality invariant no longer applies past this stage — but
  page provenance (`page_start`/`page_end`) is preserved so citations still
  resolve. Purely synthetic chunks (summaries, Q/A) may carry `None` pages.

`NoOpEnricher` is the default (Null Object): the indexing pipeline always calls
`enrich(...)` and never branches on whether enrichment is configured.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterator

from ..core.component import Component
from ..core.contracts import Chunk, Document

__all__ = ["Enricher"]


class Enricher(Component):
    """Strategy interface: augment a document's chunk stream with context."""

    kind = "enricher"

    @abstractmethod
    def enrich(
        self, chunks: Iterator[Chunk], document: Document
    ) -> Iterator[Chunk]:
        """Yield enriched chunks for one document. Raise `EnrichmentError` on
        failure."""
