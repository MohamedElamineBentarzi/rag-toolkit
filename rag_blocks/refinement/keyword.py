"""KeywordRefiner: a zero-dependency lexical reranking refiner.

Rescores candidates by query-term overlap (Jaccard of the token sets) and
reorders. Crude next to a cross-encoder, but a real, deterministic refiner —
the hermetic reranker that lets the refiner chain be tested against actual
*reordering* without loading a model. Port of the old `reranker:keyword`.

It replaces each candidate's score with the overlap score but keeps
`retriever_name` intact, so a fused result stays attributable to the retriever
that originally surfaced it.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..core.contracts import Query, ScoredChunk
from ..core.registry import registry
from .base import Refiner

__all__ = ["KeywordRefiner"]

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@registry.register
class KeywordRefiner(Refiner):
    name = "keyword"
    version = "0.1.0"

    def refine(
        self, query: Query, candidates: list[ScoredChunk], k: int
    ) -> list[ScoredChunk]:
        q = _tokens(query.text)
        rescored = [
            replace(sc, score=_overlap(q, _tokens(sc.chunk.text)))
            for sc in candidates
        ]
        rescored.sort(key=lambda sc: (sc.score, sc.chunk.id), reverse=True)
        return rescored


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)  # Jaccard similarity
