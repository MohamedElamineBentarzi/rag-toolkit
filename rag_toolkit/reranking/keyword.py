"""KeywordReranker: a zero-dependency lexical reranker.

Rescores candidates by query-term overlap (Jaccard of the token sets) and
reorders. Crude next to a cross-encoder, but a real, deterministic reranker —
useful as a baseline, and the hermetic implementation that lets the reranker
contract be tested against actual *reordering* (which `noop` can't exercise).

It replaces each candidate's score with the overlap score but keeps
`retriever_name` intact, so a fused result stays attributable to the retriever
that originally surfaced it.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from ..core.contracts import Query, ScoredChunk
from ..core.registry import registry
from .base import Reranker

__all__ = ["KeywordReranker"]

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@registry.register
class KeywordReranker(Reranker):
    name = "keyword"
    version = "0.1.0"

    def rerank(
        self, query: Query, candidates: Iterable[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        q = _tokens(query.text)
        rescored = [
            replace(sc, score=_overlap(q, _tokens(sc.chunk.text)))
            for sc in candidates
        ]
        rescored.sort(key=lambda sc: (sc.score, sc.chunk.id), reverse=True)
        return rescored[:top_k]


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)  # Jaccard similarity
