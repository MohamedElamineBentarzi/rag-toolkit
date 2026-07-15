"""QdrantVectorStore: adapter over the Qdrant vector database (v2, multi-vector).

Pattern: Adapter. Qdrant speaks (collection, named vectors, PointStruct,
query_points, scroll); our contract speaks (schema + chunks + named vectors →
upsert; name + query vector → ScoredChunk; filters → fetch). This class is the
translation layer and nothing else.

Multi-vector mapping (DR-0001 v2, D3). A `ChunkIndex` declares named spaces
("dense", "splade", …); each maps to a Qdrant *named vector* — dense spaces to
`VectorParams`, sparse spaces to native `SparseVectorParams`. One point per
chunk carries all of them plus the self-describing payload.

Translation details worth calling out:

- **Point IDs.** Qdrant requires an unsigned int or a UUID; our `chunk.id` is a
  string like `"ab12…:0"`. We deterministically map it through `uuid5`, so
  re-upserting the same chunk hits the same point (idempotent), and we stash the
  real `chunk.id` in the payload for reconstruction.
- **Self-describing payload.** The whole chunk (text + provenance) rides in the
  payload, so `search`/`fetch` rebuild a real `Chunk` and the query path never
  needs the blob store (AGENTS.md §7.2).
- **Schema is create-or-validate.** Named-vector sets are fixed at collection
  creation — you cannot add a space later (§8.1). `ensure_schema` creates the
  collection to match `specs`, or validates that an existing one already does;
  a mismatch raises `ConfigError` (loud beats lossy). Representation-set change
  ⇒ new collection.

Connection is flexible: `location=":memory:"` (in-process, no server — used by
the integration test), `path=...` (embedded on-disk), or `url=...` (a real
server). Credentials (`api_key`) are named for auto-redaction.

Dependency handling: `qdrant_client` is imported lazily and declared as the
optional extra `rag-toolkit[qdrant]`; the client is created once and reused.
Written against qdrant-client's `query_points`/`scroll` API (the `search`
method is deprecated) — re-verify on a dependency bump.

File named `qdrant_store.py` to avoid shadowing the `qdrant_client` package.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..core.contracts import (
    Chunk,
    ScoredChunk,
    SparseVector,
    VectorSpec,
    VectorValue,
)
from ..core.errors import ConfigError, StorageError
from ..core.registry import registry
from .vector_store import VectorStore

__all__ = ["QdrantVectorStore"]

#: Stable namespace so chunk.id → point UUID is deterministic across runs.
_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


@registry.register
class QdrantVectorStore(VectorStore):
    name = "qdrant"
    version = "0.2.1"

    @dataclass
    class Config:
        collection: str = "rag_toolkit"
        url: Optional[str] = None          # e.g. "http://localhost:6333"
        location: Optional[str] = None     # e.g. ":memory:"
        path: Optional[str] = None         # embedded on-disk store
        api_key: Optional[str] = None      # redacted; else QDRANT_API_KEY
        prefer_grpc: bool = False
        #: DEV/TEST escape hatch. When an existing collection's schema does not
        #: match the declared representations, DROP and recreate it instead of
        #: raising. Off by default — it destroys existing vectors, so it is an
        #: explicit opt-in, never silent coercion (§8.1).
        recreate_on_mismatch: bool = False

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None
        self._models: Any = None
        self._specs: dict[str, VectorSpec] = {}

    # -- schema --------------------------------------------------------------

    def ensure_schema(self, specs: Sequence[VectorSpec]) -> None:
        self._specs = {s.name: s for s in specs}
        client, models = self._client_and_models()
        try:
            exists = client.collection_exists(collection_name=self.config.collection)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Qdrant schema check failed: {exc}") from exc

        if not exists:
            self._create_collection(client, models, specs)
            return
        # Exists → validate its shape against `specs`. On mismatch, either drop
        # and recreate (explicit opt-in) or fail loudly (default: never coerce).
        try:
            self._validate_collection(client, specs)
        except ConfigError:
            if not self.config.recreate_on_mismatch:
                raise
            client.delete_collection(collection_name=self.config.collection)
            self._create_collection(client, models, specs)

    def _create_collection(
        self, client: Any, models: Any, specs: Sequence[VectorSpec]
    ) -> None:
        vectors_config = {
            s.name: models.VectorParams(
                size=_require_dim(s), distance=_distance(models, s.distance)
            )
            for s in specs
            if s.kind == "dense"
        }
        sparse_config = {
            s.name: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
            for s in specs
            if s.kind == "sparse"
        }
        try:
            client.create_collection(
                collection_name=self.config.collection,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                f"Qdrant could not create collection "
                f"{self.config.collection!r}: {exc}"
            ) from exc

    def _validate_collection(
        self, client: Any, specs: Sequence[VectorSpec]
    ) -> None:
        try:
            info = client.get_collection(collection_name=self.config.collection)
            params = info.config.params
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                f"Qdrant could not read collection "
                f"{self.config.collection!r} for validation: {exc}"
            ) from exc

        raw_vectors = params.vectors
        # A collection made with a single *unnamed* vector (pre-v2, or created
        # outside this toolkit) exposes `vectors` as one VectorParams, not a
        # name→VectorParams mapping. DR-0001 v2 needs named spaces; say so plainly
        # instead of reporting a bogus "missing space" from misreading the object.
        if raw_vectors is not None and not isinstance(raw_vectors, dict):
            raise ConfigError(
                f"Qdrant collection {self.config.collection!r} uses a legacy "
                f"single unnamed-vector schema, but this index declares named "
                f"spaces {[s.name for s in specs]}. Representation-set change ⇒ "
                "new collection (§8.1): drop it, point at a fresh collection "
                "name, or pass recreate_on_mismatch=True (destroys existing data)."
            )
        have_dense = dict(raw_vectors or {})
        have_sparse = dict(params.sparse_vectors or {})
        present = (
            f"(collection has dense={sorted(have_dense)}, "
            f"sparse={sorted(have_sparse)})"
        )
        for s in specs:
            if s.kind == "dense":
                got = have_dense.get(s.name)
                if got is None:
                    raise ConfigError(
                        f"Qdrant collection {self.config.collection!r} is missing "
                        f"dense space {s.name!r} {present}; representation-set "
                        "change ⇒ new collection (§8.1): drop it, use a fresh "
                        "collection name, or pass recreate_on_mismatch=True."
                    )
                if _require_dim(s) != got.size:
                    raise ConfigError(
                        f"Qdrant space {s.name!r}: schema mismatch, collection "
                        f"has dim {got.size}, index declares {s.dimensions}. "
                        "Drop the collection, use a fresh name, or pass "
                        "recreate_on_mismatch=True (destroys existing data)."
                    )
            else:
                if s.name not in have_sparse:
                    raise ConfigError(
                        f"Qdrant collection {self.config.collection!r} is missing "
                        f"sparse space {s.name!r} {present}; representation-set "
                        "change ⇒ new collection (§8.1): drop it, use a fresh "
                        "collection name, or pass recreate_on_mismatch=True."
                    )

    # -- writes --------------------------------------------------------------

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Mapping[str, Sequence[VectorValue]],
    ) -> None:
        if not chunks:
            return
        for name, seq in vectors.items():
            if name not in self._specs:
                raise StorageError(
                    f"upsert to undeclared space {name!r}; call ensure_schema "
                    f"first (have: {sorted(self._specs) or '<none>'})"
                )
            if len(seq) != len(chunks):
                raise StorageError(
                    f"upsert space {name!r}: {len(chunks)} chunks but "
                    f"{len(seq)} vectors"
                )
        client, models = self._client_and_models()
        points = []
        for i, chunk in enumerate(chunks):
            named: dict[str, Any] = {}
            for name, seq in vectors.items():
                named[name] = _to_qdrant_vector(models, seq[i])
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector=named,
                    payload=_to_payload(chunk),
                )
            )
        try:
            client.upsert(collection_name=self.config.collection, points=points)
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise StorageError(f"Qdrant upsert failed: {exc}") from exc

    def update_vectors(
        self,
        name: str,
        chunk_ids: Sequence[str],
        vectors: Sequence[VectorValue],
    ) -> None:
        if name not in self._specs:
            raise StorageError(f"update_vectors: undeclared space {name!r}")
        if len(chunk_ids) != len(vectors):
            raise StorageError(
                f"update_vectors: {len(chunk_ids)} ids but {len(vectors)} vectors"
            )
        client, models = self._client_and_models()
        points = [
            models.PointVectors(
                id=_point_id(cid), vector={name: _to_qdrant_vector(models, vec)}
            )
            for cid, vec in zip(chunk_ids, vectors)
        ]
        try:
            client.update_vectors(
                collection_name=self.config.collection, points=points
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Qdrant update_vectors failed: {exc}") from exc

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        name: str,
        vector: VectorValue,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        client, models = self._client_and_models()
        if not client.collection_exists(collection_name=self.config.collection):
            return []  # nothing upserted yet ⇒ no results (not an error)
        try:
            response = client.query_points(
                collection_name=self.config.collection,
                query=_to_query(models, vector),
                using=name,
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

    def fetch(self, filters: dict, limit: int = 100) -> list[Chunk]:
        client, models = self._client_and_models()
        if not client.collection_exists(collection_name=self.config.collection):
            return []
        try:
            points, _ = client.scroll(
                collection_name=self.config.collection,
                scroll_filter=_to_filter(models, filters),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Qdrant fetch failed: {exc}") from exc
        return [_from_payload(p.payload) for p in points]

    # -- internals -----------------------------------------------------------

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


def _require_dim(spec: VectorSpec) -> int:
    if spec.dimensions is None:
        raise ConfigError(
            f"Qdrant dense space {spec.name!r} needs `dimensions`; the "
            "Embedder must report them."
        )
    return spec.dimensions


def _distance(models: Any, distance: str) -> Any:
    table = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclidean": models.Distance.EUCLID,
        "euclid": models.Distance.EUCLID,
    }
    try:
        return table[distance.lower()]
    except KeyError:
        raise ConfigError(f"Qdrant: unknown distance {distance!r}") from None


def _to_qdrant_vector(models: Any, vector: VectorValue) -> Any:
    """Named-vector *storage* form: dense stays a list; sparse becomes a
    `SparseVector` model."""
    if isinstance(vector, SparseVector):
        return models.SparseVector(
            indices=list(vector.indices), values=list(vector.values)
        )
    return list(vector)


def _to_query(models: Any, vector: VectorValue) -> Any:
    """Named-vector *query* form for query_points."""
    if isinstance(vector, SparseVector):
        return models.SparseVector(
            indices=list(vector.indices), values=list(vector.values)
        )
    return list(vector)


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
    """Translate a filter dict into a Qdrant Filter (must-match).

    Shared semantics: scalar value ⇒ `MatchValue` (equality); list value ⇒
    `MatchAny` (membership)."""
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            match = models.MatchAny(any=list(value))
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=conditions)
