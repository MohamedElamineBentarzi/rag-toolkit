"""MarkdownFixedChunker: cut on headings, then cap each section by size.

The best of both baselines. `MarkdownChunker` gives coherent, heading-aligned
sections but has no size ceiling, so one long section becomes one enormous
chunk. `FixedChunker` caps size but cuts blind to structure. This does both:
section on ATX headings first (structure), then any section over `max_chars` is
sub-split with the same soft-boundary + overlap windowing the fixed chunker uses
— bounded to *within* that section, so a window never bleeds across a heading.

Coordinates only, like every Chunker: the base Template Method owns slicing,
provenance, and contiguous indexing. A short section stays whole; a long one
becomes several size-bounded windows whose first window still opens on the
section's heading. Adding the section *heading as context* to each window is a
separate concern — that's the `heading` enricher's job (a Chunk→Chunk text
transform), kept out of here so chunk spans stay exact provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..core.contracts import Document
from ..core.registry import registry
from .base import Chunker, soft_window_end
from .markdown import iter_heading_sections

__all__ = ["MarkdownFixedChunker"]


@registry.register
class MarkdownFixedChunker(Chunker):
    name = "markdown-fixed"
    version = "0.1.0"

    @dataclass
    class Config:
        #: The size ceiling per chunk. A section at or under this stays whole.
        max_chars: int = 1600
        #: Overlap repeated between the sub-windows of an over-long section.
        overlap_chars: int = 200

    def iter_spans(self, document: Document) -> Iterator[tuple[int, int]]:
        text = document.markdown
        for start, end in iter_heading_sections(text):
            if end - start <= self.config.max_chars:
                yield start, end  # a coherent section within the cap: keep whole
                continue
            # Over the cap: fixed-window sub-split, clamped to this section so a
            # window never crosses into the next heading's content.
            pos = start
            while pos < end:
                cut = soft_window_end(text, pos, self.config.max_chars, end)
                yield pos, cut
                if cut >= end:
                    break
                # Step back by the overlap; guarantee forward progress even under
                # a pathological overlap >= window size.
                nxt = cut - self.config.overlap_chars
                pos = nxt if nxt > pos else cut
