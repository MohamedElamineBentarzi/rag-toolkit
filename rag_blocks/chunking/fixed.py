"""FixedChunker: fixed-size character windows with overlap.

The workhorse baseline: walk the markdown in windows of ~`chunk_chars`, with
`overlap_chars` of trailing context repeated into the next window so a fact
split across a boundary still lands whole in at least one chunk.

The one bit of finesse — cut on a boundary, not mid-sentence — reuses the exact
logic proven in `PlainTextParser._cut_point`: prefer to end a window at a
paragraph break (`\n\n`), else a line break (`\n`), but REFUSE a soft cut that
would leave a chunk shorter than half the target (otherwise a document full of
short lines would produce a swarm of tiny chunks). No boundary in range ⇒ a
hard cut at `chunk_chars`.

Overlap is expressed purely in coordinates (the next window starts
`overlap_chars` before the previous end), which is exactly why the Chunker
interface emits spans rather than strings — overlap and provenance survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..core.contracts import Document
from ..core.registry import registry
from .base import Chunker

__all__ = ["FixedChunker"]


@registry.register
class FixedChunker(Chunker):
    name = "fixed"
    version = "0.1.0"

    @dataclass
    class Config:
        chunk_chars: int = 1600
        overlap_chars: int = 200

    def iter_spans(self, document: Document) -> Iterator[tuple[int, int]]:
        text = document.markdown
        n = len(text)
        start = 0
        while start < n:
            end = self._cut_end(text, start, n)
            yield start, end
            if end >= n:
                break
            # Step back by the overlap for the next window; guarantee forward
            # progress even under a pathological overlap >= window size.
            next_start = end - self.config.overlap_chars
            start = next_start if next_start > start else end

    def _cut_end(self, text: str, start: int, n: int) -> int:
        """Where to end the window that begins at `start` (mirror of
        PlainTextParser._cut_point, adapted to absolute offsets)."""
        hard_end = start + self.config.chunk_chars
        if hard_end >= n:
            return n
        # Only accept a soft cut in the back half of the window.
        floor = start + self.config.chunk_chars // 2
        para = text.rfind("\n\n", floor, hard_end)
        if para != -1:
            return para + 2
        line = text.rfind("\n", floor, hard_end)
        if line != -1:
            return line + 1
        return hard_end
