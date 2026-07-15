"""NeighborExpander: sentence-window / small-to-big expansion (DR-0001 v2, D9).

The pattern: index *small* chunks (precise retrieval), then answer over a
*bigger* window around each hit (coherent context). This refiner takes each
retrieved chunk and stitches in its neighbors — the chunks at nearby indices in
the same document — into one expanded passage.

Two pieces of the architecture pay off here at once (G6):
- **D3 `fetch`.** Neighbors are pulled by point retrieval without a query
  vector: `index.fetch({"doc_id": d, "index": [i-1, i, i+1]})` — membership
  filter, no re-embedding.
- **The provenance chain.** Because every chunk carries `char_start/char_end`,
  the merge is *overlap-safe*: neighbors are stitched by character coordinates,
  so overlapping chunk text is joined once, not duplicated. The offsets that
  power citations do retrieval work too.

Synthetic chunks (enricher-added summaries, §8.2) are excluded from the window —
expansion is about a document's own contiguous text. The expanded result keeps
the anchor's identity and score (it is terminal context for the generator, not a
re-indexed chunk) and is marked `metadata["expanded"] = True`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from ..core.contracts import Chunk, Query, ScoredChunk
from ..core.errors import ConfigError
from ..core.registry import registry
from ..indexing.chunk_index import ChunkIndex
from .base import Refiner

__all__ = ["NeighborExpander"]


@registry.register
class NeighborExpander(Refiner):
    name = "neighbor-expander"
    version = "0.1.0"

    @dataclass
    class Config:
        window: int = 1     # neighbors on each side (index ± window)

    def __init__(
        self, index: ChunkIndex | None = None, config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        if index is None:
            raise ConfigError(
                "NeighborExpander must be built with index= (the ChunkIndex "
                "whose neighbors it expands)"
            )
        self.index = index

    def refine(
        self, query: Query, candidates: list[ScoredChunk], k: int
    ) -> list[ScoredChunk]:
        window = self.config.window
        # Cache per (doc_id) window fetches so adjacent candidates don't refetch.
        doc_cache: dict[str, dict[int, Chunk]] = {}
        out: list[ScoredChunk] = []
        for sc in candidates:
            anchor = sc.chunk
            if anchor.index is None or window <= 0:
                out.append(sc)
                continue
            neighbors = self._window(anchor, window, doc_cache)
            if len(neighbors) <= 1:
                out.append(sc)
                continue
            out.append(replace(sc, chunk=_merge(anchor, neighbors)))
        return out

    def _window(
        self, anchor: Chunk, window: int,
        doc_cache: dict[str, dict[int, Chunk]],
    ) -> list[Chunk]:
        by_index = doc_cache.get(anchor.doc_id)
        if by_index is None:
            by_index = {}
            doc_cache[anchor.doc_id] = by_index
        lo = max(0, anchor.index - window)
        hi = anchor.index + window
        missing = [i for i in range(lo, hi + 1) if i not in by_index]
        if missing:
            # Headroom: a synthetic chunk matching this membership filter would
            # consume a limit slot and silently displace a real neighbor
            # (order-dependent context loss). Over-fetch, then drop synthetics.
            fetched = self.index.fetch(
                {"doc_id": anchor.doc_id, "index": missing},
                limit=len(missing) * 2,
            )
            for c in fetched:
                if c.metadata.get("synthetic"):
                    continue  # §8.2: synthetic chunks never join a text window
                if c.index is not None:
                    by_index[c.index] = c
        # The anchor is always part of its own window even if fetch missed it.
        by_index.setdefault(anchor.index, anchor)
        return [by_index[i] for i in range(lo, hi + 1) if i in by_index]


def _merge(anchor: Chunk, neighbors: list[Chunk]) -> Chunk:
    """Stitch a window of chunks into one passage, overlap-safe by char offsets.

    Chunks are ordered by `char_start`; overlapping text (a chunk starting
    before the running cursor) contributes only its tail, so overlap strategies
    don't double up. Non-adjacent neighbors (a separator sat between them, not
    captured in either chunk's text) are joined with a single space — the gap
    text isn't recoverable without the source document."""
    usable = sorted(
        (c for c in neighbors if c.char_start is not None and c.char_end is not None),
        key=lambda c: c.char_start if c.char_start is not None else 0,
    )
    if not usable:
        return anchor

    parts: list[str] = []
    cursor: Optional[int] = None
    start = usable[0].char_start
    for c in usable:
        cs, ce = c.char_start, c.char_end
        assert cs is not None and ce is not None  # filtered above
        if cursor is None:
            parts.append(c.text)
            cursor = ce
        elif cs >= cursor:
            if cs > cursor:
                parts.append(" ")   # non-contiguous: gap text is unrecoverable
            parts.append(c.text)
            cursor = ce
        elif ce > cursor:
            parts.append(c.text[cursor - cs:])   # overlap: append only the tail
            cursor = ce
        # else fully contained in what we already have → skip

    pages = [c.page_start for c in usable if c.page_start is not None]
    page_ends = [c.page_end for c in usable if c.page_end is not None]
    return replace(
        anchor,
        text="".join(parts),
        char_start=start,
        char_end=cursor,
        page_start=min(pages) if pages else anchor.page_start,
        page_end=max(page_ends) if page_ends else anchor.page_end,
        metadata={**anchor.metadata, "expanded": True},
    )
