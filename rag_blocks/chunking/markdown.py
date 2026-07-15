"""MarkdownChunker: cut on heading boundaries.

This is the payoff of normalizing every input to markdown during ingestion:
document *structure* survives all the way to the cutting decision. Instead of
slicing blindly every N characters, we cut at ATX headings (`#`..`######`), so
each chunk is a coherent section that starts with its own heading — far better
retrieval units than arbitrary windows.

Coordinates only (like every Chunker): we locate heading line offsets and emit
the spans *between* them. Content before the first heading is its own span; a
document with no headings yields a single span covering the whole thing. The
base Template Method then skips any whitespace-only section without disturbing
index contiguity.

Deliberately simple for v0.2: no secondary size cap on a very long section yet
(a `max_chars` split is an easy, non-breaking add later). Structure first.
"""

from __future__ import annotations

import re
from typing import Iterator

from ..core.contracts import Document
from ..core.registry import registry
from .base import Chunker

__all__ = ["MarkdownChunker"]

#: An ATX heading: 1–6 '#', then whitespace, then the title. Up to three
#: leading spaces are allowed by CommonMark; `# no-space` is NOT a heading.
_HEADING = re.compile(r"^ {0,3}#{1,6}\s")


def _is_heading(line: str) -> bool:
    return _HEADING.match(line) is not None


@registry.register
class MarkdownChunker(Chunker):
    name = "markdown-aware"
    version = "0.1.0"

    def iter_spans(self, document: Document) -> Iterator[tuple[int, int]]:
        text = document.markdown
        n = len(text)
        if n == 0:
            return

        # Collect cut points: the document start, plus every heading's offset.
        cut_points = [0]
        offset = 0
        for line in text.splitlines(keepends=True):
            if offset != 0 and _is_heading(line):
                cut_points.append(offset)
            offset += len(line)
        cut_points.append(n)

        for start, end in zip(cut_points, cut_points[1:]):
            if end > start:  # consecutive headings can coincide; skip empties
                yield start, end
