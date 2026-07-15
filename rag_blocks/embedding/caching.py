"""CachingEmbedder: a fingerprint-transparent memoizing decorator (DR-0001 v2, D7).

Wraps any `Embedder` and memoizes its vectors in a `BlobStore`, so identical
text — across documents, across re-indexes, across processes — is embedded once.
It *is* an `Embedder` (same interface), so it drops in wherever an embedder goes,
including as a `ChunkIndex` representation; a `CachingSparseEncoder` will mirror
it when static sparse lands.

Two correctness rules make the decorator invisible where it must be:

- **Identity transparency.** `fingerprint()` returns the *inner* embedder's
  fingerprint. The wrapper changes cost, not output, so it must not change cache
  keys or trial identity — a cached and uncached run of the same model are the
  same trial. This override is deliberate (a decorator that changed identity
  would fork the tuner's cache for no behavioral reason).

- **Passage/query namespace split.** Instruction-tuned models encode the same
  string differently through `embed_texts` (passage) and `embed_query` (query),
  so the two caches live under separate prefixes (`.../passages/`,
  `.../queries/`). Collapsing them would serve a passage vector for a query and
  silently wreck retrieval.

Keys are `embeddings/{inner.fingerprint()}/{passages|queries}/{sha256(text)}.json`
— folding the inner fingerprint in means swapping the model is a clean miss, and
P8's "N representations ⇒ N cache keyspaces" falls out for free.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Sequence

from ..storage.base import BlobStore
from .base import Embedder

__all__ = ["CachingEmbedder"]


class CachingEmbedder(Embedder):
    """Memoize an inner `Embedder`'s vectors in a `BlobStore`, transparently."""

    kind = "embedder"
    name = "caching"
    version = "0.1.0"

    def __init__(self, inner: Embedder, cache: BlobStore) -> None:
        super().__init__()
        self.inner = inner
        self._cache = cache
        self._prefix = f"embeddings/{inner.fingerprint()}"

    # -- transparent identity ------------------------------------------------

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    @property
    def distance(self) -> str:
        return self.inner.distance

    def fingerprint(self) -> str:
        # The wrapper is invisible to cache keys and trial identity (D7).
        return self.inner.fingerprint()

    def describe(self) -> dict:
        # Report the inner component's identity so logs read as the real model.
        return self.inner.describe()

    # -- embedding with memoization ------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._memoized(list(texts), "passages", self.inner.embed_texts)

    def embed_query(self, text: str) -> list[float]:
        # Queries encode through inner.embed_query (the passage/query asymmetry
        # is exactly why the caches are namespaced apart).
        def encode(ts: list[str]) -> list[list[float]]:
            return [self.inner.embed_query(t) for t in ts]

        return self._memoized([text], "queries", encode)[0]

    def _memoized(
        self, texts: list[str], namespace: str,
        encode: Callable[[list[str]], list[list[float]]],
    ) -> list[list[float]]:
        if not texts:
            return []
        cached: list[list[float] | None] = [
            self._get(namespace, t) for t in texts
        ]
        misses = [i for i, v in enumerate(cached) if v is None]
        if misses:
            fresh = encode([texts[i] for i in misses])
            for i, vector in zip(misses, fresh):
                self._put(namespace, texts[i], vector)
                cached[i] = vector
        return [v for v in cached if v is not None]  # order preserved, all filled

    # -- blob keys -----------------------------------------------------------

    def _key(self, namespace: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._prefix}/{namespace}/{digest}.json"

    def _get(self, namespace: str, text: str) -> list[float] | None:
        key = self._key(namespace, text)
        if not self._cache.exists(key):
            return None
        return json.loads(self._cache.get(key).decode("utf-8"))

    def _put(self, namespace: str, text: str, vector: list[float]) -> None:
        self._cache.put(
            self._key(namespace, text), json.dumps(vector).encode("utf-8")
        )
