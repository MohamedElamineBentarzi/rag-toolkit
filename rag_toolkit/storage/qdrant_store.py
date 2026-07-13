"""QdrantVectorStore: adapter over the Qdrant vector database.

Pattern: Adapter. Qdrant speaks (collection, PointStruct, query_points); our
contract speaks (chunks + vectors → upsert; vector → ScoredChunk). This class is
the translation layer and nothing else.

Two translation details worth calling out:

- **Point IDs.** Qdrant requires an unsigned int or a UUID; our `chunk.id` is a
  string like `"ab12…:0"`. We deterministically map it through `uuid5`, so
  re-upserting the same chunk hits the same point (idempotent), and we stash the
  real `chunk.id` in the payload for reconstruction.
- **Self-describing payload.** The whole chunk (text + provenance) rides in the
  payload, so `search` rebuilds a real `Chunk` and the query path never needs
  the blob store (AGENTS.md §7.2).

Connection is flexible: `location=":memory:"` (in-process, no server — used by
the integration test), `path=...` (embedded on-disk), or `url=...` (a real
server). Credentials (`api_key`) are named for auto-redaction.

Dependency handling: `qdrant_client` is imported lazily and declared as the
optional extra `rag-toolkit[qdrant]`; the client + collection are created once
and reused. Written against qdrant-client's `query_points` API (the `search`
method is deprecated) — re-verify on a dependency bump.

File named `qdrant_store.py` to avoid shadowing the `qdrant_client` package.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..core.contracts import Chunk, ScoredChunk
from ..core.errors import StorageError
from ..core.registry import registry
from .vector_store import VectorStore

__all__ = ["QdrantVectorStore"]

#: Stable namespace so chunk.id → point UUID is deterministic across runs.
_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


@registry.register
class QdrantVectorStore(VectorStore):
    name = "qdrant"
    version = "0.1.0"

    @dataclass
    class Config:
        collection: str = "rag_toolkit"
        url: Optional[str] = None          # e.g. "http://localhost:6333"
        location: Optional[str] = None     # e.g. ":memory:"
        path: Optional[str] = None         # embedded on-disk store
        api_key: Optional[str] = None      # redacted; else QDRANT_API_KEY
        prefer_grpc: bool = False

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None
        self._models: Any = None
        self._ensured = False  # collection existence checked/created once

    # -- contract ------------------------------------------------------------

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise StorageError(
                f"upsert got {len(chunks)} chunks but {len(vectors)} vectors"
            )
        if not chunks:
            return
        client, models = self._client_and_models()
        self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=_point_id(chunk.id),
                vector=list(vector),
                payload=_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            client.upsert(collection_name=self.config.collection, points=points)
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise StorageError(f"Qdrant upsert failed: {exc}") from exc

    def search(
        self, vector: list[float], k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        client, models = self._client_and_models()
        if not client.collection_exists(collection_name=self.config.collection):
            return []  # nothing upserted yet ⇒ no results (not an error)
        try:
            response = client.query_points(
                collection_name=self.config.collection,
                query=list(vector),
                limit=k,
                query_filter=_to_filter(models, filters),
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Qdrant search failed: {exc}") from exc
        return [
            ScoredChunk(chunk=_from_payload(hit.payload), score=hit.score)
            for hit in response.points
        ]

    # -- internals -----------------------------------------------------------

    def _ensure_collection(self, dimensions: int) -> None:
        if self._ensured:
            return
        client, models = self._client_and_models()
        try:
            if not client.collection_exists(collection_name=self.config.collection):
                client.create_collection(
                    collection_name=self.config.collection,
                    vectors_config=models.VectorParams(
                        size=dimensions, distance=models.Distance.COSINE
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                f"Qdrant could not ensure collection "
                f"{self.config.collection!r}: {exc}"
            ) from exc
        self._ensured = True

    def _client_and_models(self) -> tuple[Any, Any]:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient, models  # lazy
            except ImportError as exc:
                raise StorageError(
                    "QdrantVectorStore requires 'qdrant-client'. "
                    "Install with: pip install 'rag-toolkit[qdrant]'"
                ) from exc
            self._client = QdrantClient(
                location=self.config.location,
                url=self.config.url,
                path=self.config.path,
                api_key=self.config.api_key or os.environ.get("QDRANT_API_KEY"),
                prefer_grpc=self.config.prefer_grpc,
            )
            self._models = models
        return self._client, self._models


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _to_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "index": chunk.index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "metadata": chunk.metadata,
    }


def _from_payload(payload: dict) -> Chunk:
    return Chunk(
        id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        index=payload["index"],
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        metadata=payload.get("metadata") or {},
    )


def _to_filter(models: Any, filters: Optional[dict]) -> Any:
    """Translate a payload-equality dict into a Qdrant Filter (must-match)."""
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
    )
