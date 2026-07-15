"""HeadingEnricher: prepend a chunk's section heading — contextual retrieval,
done deterministically.

The same idea as LLM contextual retrieval (give a chunk the context it's missing
so it retrieves well), but zero-dependency: for each chunk, find the nearest
markdown heading at or above its start offset in the document and prepend it. A
chunk buried under "## Q3 Results" now embeds *with* that heading, so a query
about Q3 finds it even when the chunk body never repeats the phrase.

This is the payoff of normalizing ingestion to markdown showing up a second time
(the chunker used headings to cut; the enricher uses them to situate). Provenance
is preserved — only the text is augmented — so citations still resolve to pages.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterator

from ..core.contracts import Chunk, Document
from ..core.registry import registry
from .base import Enricher

__all__ = ["HeadingEnricher"]

_HEADING = re.compile(r"^ {0,3}#{1,6}\s.*$", re.MULTILINE)


@registry.register
class HeadingEnricher(Enricher):
    name = "heading"
    version = "0.1.0"

    def enrich(
        self, chunks: Iterator[Chunk], document: Document
    ) -> Iterator[Chunk]:
        markdown = document.markdown
        for chunk in chunks:
            heading = self._nearest_heading(markdown, chunk.char_start)
            # Skip when there's no heading, or the chunk already starts with it.
            if heading and not chunk.text.lstrip().startswith(heading):
                yield replace(chunk, text=f"{heading}\n\n{chunk.text}")
            else:
                yield chunk

    @staticmethod
    def _nearest_heading(markdown: str, char_start: int | None) -> str | None:
        """The last heading line at or before `char_start`, or None."""
        if char_start is None:
            return None
        last = None
        for match in _HEADING.finditer(markdown):
            if match.start() > char_start:
                break
            last = match.group().strip()
        return last
