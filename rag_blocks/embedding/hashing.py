"""HashingEmbedder: a zero-dependency, deterministic embedder.

The feature-hashing trick (a.k.a. the "hashing vectorizer"): tokenize, hash each
token into one of `dimensions` buckets, accumulate a signed count per bucket,
L2-normalize. No model, no network, no vocabulary to fit — just stdlib.

Why it exists (it is NOT merely a test fake):
- It is a real, if weak, embedder: texts sharing tokens land near each other in
  cosine space, so it is a legitimate baseline the tuner can compare against and
  a fast smoke-test embedder for small corpora.
- It is fully deterministic — hashing uses `blake2b` (stdlib), never Python's
  per-process-salted `hash()` — so fingerprints and caches stay stable across
  runs and machines. Determinism is a hard requirement for the eval cache.

Signed hashing (a second hash bit picks +1/-1) cancels some collision bias so
buckets don't all drift positive. Queries need no special prefix here (the
representation is symmetric), so `embed_query` is just `embed_texts` of one.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

from ..core.registry import registry
from .base import Embedder

__all__ = ["HashingEmbedder"]

_TOKEN = re.compile(r"\w+", re.UNICODE)


@registry.register
class HashingEmbedder(Embedder):
    name = "hashing"
    version = "0.1.0"

    @dataclass
    class Config:
        dimensions: int = 256

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Symmetric representation: a query is embedded exactly like a passage.
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        dim = self.config.dimensions
        vec = [0.0] * dim
        for token in _TOKEN.findall(text.lower()):
            bucket, sign = self._hash(token, dim)
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # empty/token-less text ⇒ zero vector (cosine 0)
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _hash(token: str, dim: int) -> tuple[int, float]:
        """Deterministic (bucket, sign) for a token. blake2b, not hash()."""
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign
