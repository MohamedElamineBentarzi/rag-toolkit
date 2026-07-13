"""BgeReranker: cross-encoder reranking via sentence-transformers.

Pattern: Adapter. A cross-encoder reads the query and each candidate *together*
(one forward pass per pair), so it judges relevance far more accurately than the
bi-encoder embedder that retrieved them apart — the classic "retrieve wide with
embeddings, rerank precise with a cross-encoder" split. Default model is
`BAAI/bge-reranker-base`.

Dependency handling: `sentence_transformers` (which ships `CrossEncoder`) is
imported lazily behind the `[sentence-transformers]` extra; the model is loaded
once and reused. Scores are the cross-encoder's raw relevance logits — higher is
better — written over each candidate's score, `retriever_name` left intact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from ..core.contracts import Query, ScoredChunk
from ..core.errors import RagToolkitError
from ..core.registry import registry
from .base import Reranker

__all__ = ["BgeReranker"]


@registry.register
class BgeReranker(Reranker):
    name = "bge-reranker"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "BAAI/bge-reranker-base"
        batch_size: int = 32
        device: Optional[str] = None

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._model: Any = None  # heavy: loaded once, reused

    def rerank(
        self, query: Query, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        candidates = list(candidates)
        if not candidates:
            return []
        model = self._get_model()
        pairs = [[query.text, sc.chunk.text] for sc in candidates]
        scores = model.predict(pairs, batch_size=self.config.batch_size)
        rescored = [
            replace(sc, score=float(score))
            for sc, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda sc: (sc.score, sc.chunk.id), reverse=True)
        return rescored[:top_k]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # lazy
            except ImportError as exc:
                raise RagToolkitError(
                    "BgeReranker requires 'sentence-transformers'. "
                    "Install with: pip install 'rag-toolkit[sentence-transformers]'"
                ) from exc
            self._model = CrossEncoder(self.config.model, device=self.config.device)
        return self._model
