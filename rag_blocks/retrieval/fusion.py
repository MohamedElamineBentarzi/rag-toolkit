"""Fusion mechanics, extracted once and reused everywhere (DR-0001 v2, F2b).

Reciprocal Rank Fusion (RRF) blends several rankings into one. Why RRF and not
score averaging: a dense retriever's scores are cosine similarities, a BM25
retriever's are unbounded term scores, a graph retriever's are something else
entirely — averaging them is meaningless (different scales, different
distributions). RRF throws the raw scores away and fuses on *rank* alone:

    fused(d) = Σ_r  weight_r · 1 / (rrf_k + rank_r(d))

where `rank_r(d)` is d's 1-based position in ranking r (a document missing from
r contributes nothing). `rrf_k` (default 60, from the original RRF paper) damps
the influence of top ranks so a single ranking can't dominate.

Three invariants this module owns so nobody reimplements them (F2b):
- **Dedup by `chunk.id`.** The same chunk surfaced by two rankings *merges*
  (its RRF contributions add), never duplicates.
- **Attribution.** Each fused result carries
  `metadata["sources"] = {source_label: rank}` so evaluation can see which
  ranking(s) found it and where.
- **Filter fan-out** is upstream of here: each sub-retriever receives the same
  `Query` (filters included) and forwards them to its backend, so every
  sub-search is scoped identically.

`HybridRetriever`, `FusionRetriever` and `MultiQueryRetriever` all fuse through
`fuse`; it is property-tested once and trusted thereafter.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.contracts import Chunk, ScoredChunk

__all__ = ["fuse", "source_labels"]


def fuse(
    rankings: Sequence[tuple[str, Sequence[ScoredChunk]]],
    k: int,
    rrf_k: int = 60,
    weights: Optional[Sequence[float]] = None,
    name: str = "fusion",
) -> list[ScoredChunk]:
    """Fuse labeled rankings into one top-`k` list.

    `rankings` is a sequence of `(source_label, ranked_chunks)` pairs. Returns
    the fused top-`k`, highest score first, each stamped `retriever_name=name`
    and carrying `metadata["sources"] = {source_label: rank}`.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(
            f"fuse: {len(weights)} weights but {len(rankings)} rankings"
        )

    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    sources: dict[str, dict[str, int]] = {}
    for (label, ranked), weight in zip(rankings, weights):
        for rank, scored in enumerate(ranked, start=1):
            cid = scored.chunk.id
            fused[cid] = fused.get(cid, 0.0) + weight / (rrf_k + rank)
            chunks.setdefault(cid, scored.chunk)
            sources.setdefault(cid, {})[label] = rank

    ordered = sorted(fused.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [
        ScoredChunk(
            chunk=chunks[cid],
            score=score,
            retriever_name=name,
            metadata={"sources": sources[cid]},
        )
        for cid, score in ordered[:k]
    ]


def source_labels(labels: Sequence[str]) -> list[str]:
    """De-duplicate source labels so fusion attribution stays unambiguous when
    two rankings share a label (e.g. two `IndexRetriever`s, or repeated query
    variants): a collision gets a `#n` suffix, preserving order."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            out.append(f"{label}#{seen[label]}")
        else:
            seen[label] = 0
            out.append(label)
    return out
