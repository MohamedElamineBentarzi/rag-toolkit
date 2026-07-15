"""Context packing + citation resolution — shared by every Generator.

Extracted as a helper (composition, not a base class) because it is the same
mechanical work regardless of how the answer is actually produced: number the
retrieved chunks `[1]`, `[2]`, … within a character budget, and afterwards map
the markers that appear in the answer back to source provenance.

This is what makes citations verifiable end-to-end: the marker in the text and the
`Citation`'s `doc_id`/pages come from the *same* packed chunk, so "[2]" in an
answer resolves to exact pages of an exact document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..core.contracts import Citation, ScoredChunk

__all__ = ["PackedContext", "pack_context", "resolve_citations"]

_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class PackedContext:
    prompt_block: str                 # "[1] ...\n\n[2] ..." for the LLM
    citations: list[Citation] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)  # chunk text per citation


def pack_context(
    context: Sequence[ScoredChunk], *, max_chars: int
) -> PackedContext:
    """Number chunks `[1..]` in rank order until the char budget is spent."""
    blocks: list[str] = []
    citations: list[Citation] = []
    texts: list[str] = []
    used = 0
    for marker, scored in enumerate(context, start=1):
        chunk = scored.chunk
        block = f"[{marker}] {chunk.text}"
        # Blocks are joined by "\n\n" in the final string, so every block after
        # the first costs its length plus the 2-char joiner — count it, or the
        # budget is understated by 2*(n-1).
        cost = len(block) + (2 if blocks else 0)
        if blocks and used + cost > max_chars:
            break  # keep at least one chunk even if it exceeds the budget
        blocks.append(block)
        texts.append(chunk.text)
        citations.append(Citation(
            marker=marker, chunk_id=chunk.id, doc_id=chunk.doc_id,
            page_start=chunk.page_start, page_end=chunk.page_end,
        ))
        used += cost
    return PackedContext("\n\n".join(blocks), citations, texts)


def resolve_citations(
    text: str, citations: Sequence[Citation]
) -> list[Citation]:
    """Keep only the citations whose marker actually appears in `text`.

    Falls back to all offered citations when the answer cited none — better to
    over-attribute than to strip provenance a caller might still want."""
    used = {int(m) for m in _MARKER.findall(text)}
    cited = [c for c in citations if c.marker in used]
    return cited or list(citations)
