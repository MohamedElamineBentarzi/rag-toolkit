"""BM25Index: a zero-dependency in-memory Okapi BM25 lexical index.

Pure-Python BM25 over dicts — the dependency-free lexical store for tests, small
corpora, and the hermetic half of hybrid retrieval. Not built for large scale (a
production deployment swaps in a tantivy/Elastic-backed LexicalIndex behind the
same contract); built to be correct and obvious.

Design choices:
- Document frequency is computed at query time from the stored per-document term
  counts, not maintained incrementally — so there are no separate "parameters"
  to keep in sync, and the persisted state is just (chunks, term-counts, lengths).
- `add` is idempotent by `chunk.id`: an id already present is skipped (ids are
  content-derived — `doc_id:index` — so a repeat id means identical text). This
  makes re-ingesting overlapping batches, and re-adding after a `load`, cheap.

Persistence (the "survives a restart" story): the index knows how to serialize
itself, but NOT where the bytes live — that is delegated to an injected
`BlobStore` (the same abstraction the pipeline's truth store uses). No store ⇒
in-memory only (ephemeral); `LocalBlobStore` ⇒ on disk; `MinioBlobStore` ⇒ on an
S3-compatible server. The index only ever sees `put`/`get`/`exists`, so it never
knows or cares about the storage vendor.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from ..core.contracts import Chunk, ScoredChunk
from ..core.registry import registry
from .base import BlobStore
from .lexical_index import LexicalIndex

__all__ = ["BM25Index"]

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@registry.register
class BM25Index(LexicalIndex):
    name = "bm25"
    version = "0.1.0"

    @dataclass
    class Config:
        k1: float = 1.5   # term-frequency saturation
        b: float = 0.75   # length normalization strength
        #: Key namespace under which the index persists in the blob store.
        namespace: str = "default"

    def __init__(
        self, store: Optional[BlobStore] = None, config: Any = None,
        **overrides: Any,
    ) -> None:
        super().__init__(config, **overrides)
        self._store = store  # where persist()/load() read & write (None ⇒ memory)
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter] = {}
        self._len: dict[str, int] = {}

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            if chunk.id in self._chunks:
                continue  # idempotent: same id ⇒ same content, nothing to do
            tokens = _tokenize(chunk.text)
            self._chunks[chunk.id] = chunk
            self._tf[chunk.id] = Counter(tokens)
            self._len[chunk.id] = len(tokens)

    def search(
        self, text: str, k: int, filters: Optional[dict] = None
    ) -> list[ScoredChunk]:
        query_terms = set(_tokenize(text))
        if not query_terms or not self._chunks:
            return []
        n = len(self._chunks)
        avgdl = sum(self._len.values()) / n
        idf = {t: self._idf(t, n) for t in query_terms}

        scored: list[ScoredChunk] = []
        for chunk_id, tf in self._tf.items():
            chunk = self._chunks[chunk_id]
            if filters and not _matches(chunk, filters):
                continue
            score = self._score(query_terms, tf, self._len[chunk_id], avgdl, idf)
            if score > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda sc: (sc.score, sc.chunk.id), reverse=True)
        return scored[:k]

    # -- persistence ---------------------------------------------------------

    def persist(self) -> None:
        """Serialize the index to the injected blob store (no-op without one)."""
        if self._store is None:
            return
        self._store.put(self._key, self._serialize())

    def load(self) -> None:
        """Rehydrate from the blob store if a saved index exists (else no-op)."""
        if self._store is None or not self._store.exists(self._key):
            return
        self._deserialize(self._store.get(self._key))

    @property
    def _key(self) -> str:
        return f"lexical/{self.config.namespace}/bm25.json"

    def _serialize(self) -> bytes:
        data = {
            "chunks": {cid: asdict(c) for cid, c in self._chunks.items()},
            "tf": {cid: dict(tf) for cid, tf in self._tf.items()},
            "len": self._len,
        }
        return json.dumps(data).encode("utf-8")

    def _deserialize(self, blob: bytes) -> None:
        data = json.loads(blob.decode("utf-8"))
        self._chunks = {cid: Chunk(**c) for cid, c in data["chunks"].items()}
        self._tf = {cid: Counter(tf) for cid, tf in data["tf"].items()}
        self._len = {cid: int(n) for cid, n in data["len"].items()}

    # -- BM25 math -----------------------------------------------------------

    def _idf(self, term: str, n: int) -> float:
        df = sum(1 for tf in self._tf.values() if term in tf)
        # Okapi idf with +1 so a term in every doc still contributes > 0.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _score(
        self, query_terms: set[str], tf: Counter, dl: int, avgdl: float,
        idf: dict[str, float],
    ) -> float:
        k1, b = self.config.k1, self.config.b
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * (freq * (k1 + 1)) / denom
        return score


def _matches(chunk: Chunk, filters: dict) -> bool:
    for key, expected in filters.items():
        actual = getattr(chunk, key, None)
        if actual is None:
            actual = chunk.metadata.get(key)
        if actual != expected:
            return False
    return True
