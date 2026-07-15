"""ChunkSink: the write-path capability seam (DR-0001 v2, D6/F4).

The write path fans out: parse → chunk → enrich, then *every* configured sink
consumes the chunk stream. A `ChunkIndex` is the flagship sink, but a GraphRAG
index, a keyword-alert index, or a bare `LexicalIndex` are all just other sinks —
the pipeline never needs to know which.

This is the ONE place the codebase uses `typing.Protocol`, deliberately.
AGENTS.md's "ABCs, not Protocols" rule governs *stage contracts*, which carry
inherited plumbing (config, fingerprint). `ChunkSink` is a *capability seam*
spanning worlds a common base cannot reach (a `ChunkIndex` and a third-party
graph store share no ancestor): shape is exactly what is meant here, so
structural typing is the right tool. The rule is refined, not broken.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..core.contracts import Chunk

__all__ = ["ChunkSink"]


@runtime_checkable
class ChunkSink(Protocol):
    """Anything that consumes chunks at write time: `add` then `persist`."""

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def persist(self) -> None: ...
