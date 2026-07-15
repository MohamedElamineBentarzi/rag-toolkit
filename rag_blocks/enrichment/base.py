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
- **Synthetic-chunk identity rule (§8.2).** An enricher that *adds* chunks (not
  just augments existing ones) MUST give each a parent-derived id
  (`f"{parent.id}#aug{n}"`), set `metadata["synthetic"] = True`, and carry the
  parent's `index`. This keeps ids collision-free and lets index-based lookups
  (`NeighborExpander`, get-by-index) exclude synthetic chunks from a document's
  contiguous text — a synthetic summary is not a neighbor.

Enrichers compose as a chain on the write path (`enrich=[...]`); the *empty*
chain is the null object, so there is no `NoOpEnricher` (DR-0001 v2, D6).
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
