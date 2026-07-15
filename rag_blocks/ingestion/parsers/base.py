"""Parser: the ingestion Strategy interface.

The single most important decision in this file: the *primitive* operation is
`iter_pages(source) -> Iterator[Page]`, a generator — NOT "return the whole
document". `parse()` is merely a convenience built on top (Template Method:
the skeleton — stream pages, assemble, attach provenance — is fixed here;
subclasses only supply the page stream).

Why streaming-first?
    1. Memory. A 2 000-page PDF parsed eagerly means the entire layout model
       output lives in RAM at once. Page-at-a-time keeps memory O(batch).
    2. Backpressure. Downstream stages can consume lazily: a streaming
       Chunker can start emitting chunks from page 1 while page 900 hasn't
       been touched yet. Generators compose into a pull-based pipeline for
       free — no queues, no threads.
    3. Fail-late visibility. If page 1 432 is corrupt, you already have
       1 431 good pages instead of one big exception and nothing.

If you only ever call `parse()`, you lose nothing — but the architecture
never forces materialization on anyone.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Iterator

from ...core.component import Component
from ...core.contracts import Document, Page, Source, SourceFormat

__all__ = ["Parser"]


class Parser(Component):
    """Turns a Source into a stream of markdown Pages."""

    kind = "parser"

    #: Formats this parser accepts. Used by AutoParser for routing and for
    #: early, readable failures instead of deep vendor stack traces.
    supported_formats: ClassVar[tuple[SourceFormat, ...]] = ()

    @abstractmethod
    def iter_pages(self, source: Source) -> Iterator[Page]:
        """Lazily yield pages in reading order. Implementations must avoid
        loading the entire source into memory when the format allows it."""

    def parse(self, source: Source) -> Document:
        """Convenience: materialize the stream into a provenance-carrying
        Document. Assembly/offset math lives on `Document.from_pages` so all
        parsers share one correct implementation."""
        return Document.from_pages(source, self.iter_pages(source))

    def supports(self, fmt: SourceFormat) -> bool:
        return fmt in self.supported_formats
