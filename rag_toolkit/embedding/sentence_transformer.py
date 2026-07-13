"""SentenceTransformerEmbedder: adapter over the sentence-transformers stack.

Pattern: Adapter. `sentence-transformers` speaks (list[str] → numpy array); our
contract speaks (Sequence[str] → list[list[float]]). This class is the
translation layer and the home of the query/passage asymmetry — nothing else.

Default model is `BAAI/bge-m3` (the roadmap's flagship: multilingual, long
context, strong retrieval). The asymmetry the interface promises lives in one
place: `query_instruction` is prepended in `embed_query` and NEVER in
`embed_texts`. bge-m3 itself needs no query prefix, so the default is empty; set
it for models that do (bge-v1.5, e5: "query: ")  — a one-line config change, no
new code.

Dependency handling: `sentence_transformers` is imported lazily and declared as
the optional extra `rag-toolkit[sentence-transformers]`. The model (heavy: it
loads weights onto CPU/GPU) is built once and cached on the instance, reused
across every batch — never per call.

File is named `sentence_transformer.py` (singular) to avoid shadowing the real
`sentence_transformers` package (same caution as `docling_parser.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..core.errors import EmbeddingError
from ..core.registry import registry
from .base import Embedder

__all__ = ["SentenceTransformerEmbedder"]


@registry.register
class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "BAAI/bge-m3"
        #: Prepended to queries only (embed_query), never to passages. Empty
        #: for instruction-free models like bge-m3; e.g. "query: " for e5.
        query_instruction: str = ""
        normalize: bool = True     # unit vectors ⇒ cosine == dot product
        batch_size: int = 32
        device: Optional[str] = None   # None ⇒ let the library pick

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._model: Any = None  # heavy: built once, reused across batches

    @property
    def dimensions(self) -> int:
        return int(self._get_model().get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{self.config.query_instruction}{text}"
        return self._encode([prefixed])[0]

    # -- internals -----------------------------------------------------------

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise EmbeddingError(f"sentence-transformers encode failed: {exc}") from exc
        return [row.tolist() for row in vectors]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # lazy
            except ImportError as exc:
                raise EmbeddingError(
                    "SentenceTransformerEmbedder requires 'sentence-transformers'. "
                    "Install with: pip install 'rag-toolkit[sentence-transformers]'"
                ) from exc
            try:
                self._model = SentenceTransformer(
                    self.config.model, device=self.config.device
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"Could not load model {self.config.model!r}: {exc}"
                ) from exc
        return self._model
