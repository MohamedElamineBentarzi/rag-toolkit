"""Chunker: the retrieval-unit Strategy interface.

The single design decision mirrored from the Parser: the *primitive* a strategy
implements is `iter_spans(document) -> Iterator[(start, end)]` — WHERE to cut,
as half-open character offsets into `document.markdown`. `chunk()` is a Template
Method layered on top that does every piece of bookkeeping once, correctly, for
every strategy.

Why spans (coordinates), not strings (copies)?
    Return strings and you throw away everything that makes RAG citations work:
    provenance (which chars → which pages), overlap (two chunks sharing a
    region), and neighbor merging (fetch index±1 at query time). All three fall
    out naturally when a strategy only decides *coordinates* and the base class
    owns the slicing. This is the whole reason the interface emits offsets.

What the Template Method guarantees (so strategies never re-implement it):
    - `text = document.markdown[start:end]` — the base slices, not the strategy.
    - whitespace-only slices are skipped WITHOUT advancing the index, so
      `index` stays contiguous 0-based with NO holes (a manual counter, never
      `enumerate` over raw spans). Query-time neighbor expansion fetches
      `index ± 1`, so a hole would silently drop a real neighbor.
    - `id = f"{document.id}:{index}"` — deterministic ⇒ re-indexing overwrites
      instead of duplicating.
    - `char_start`/`char_end` are the primary provenance; `page_start`/
      `page_end` are derived from them via `Document.pages_for_span` and are
      ALWAYS filled for a doc-derived chunk.

Contract for strategies: yield spans in reading order of `start`. Overlapping
spans are LEGAL (that is how overlap strategies express themselves — in
coordinates). Strategies are configured by their `Config`, never by subclassing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterator

from ..core.component import Component
from ..core.contracts import Chunk, Document

__all__ = ["Chunker"]


class Chunker(Component):
    """Turns a Document into a stream of retrieval Chunks."""

    kind = "chunker"

    @abstractmethod
    def iter_spans(self, document: Document) -> Iterator[tuple[int, int]]:
        """Yield half-open `(start, end)` char offsets into
        `document.markdown`, in reading order of `start`. Overlaps allowed."""

    def chunk(self, document: Document) -> Iterator[Chunk]:
        """Template Method: turn spans into Chunks, owning all bookkeeping.

        Strategies implement `iter_spans`; this method — written once — is the
        only place chunk ids, index contiguity, and provenance are decided.
        """
        index = 0
        for start, end in self.iter_spans(document):
            text = document.markdown[start:end]
            if not text.strip():
                # Skip empties WITHOUT advancing index (contiguity, no holes).
                continue
            pages = document.pages_for_span(start, end)
            yield Chunk(
                id=f"{document.id}:{index}",
                doc_id=document.id,
                text=text,
                index=index,
                char_start=start,
                char_end=end,
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
            )
            index += 1
