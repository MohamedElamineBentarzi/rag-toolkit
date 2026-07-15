"""BlobStore: the durable byte-storage seam.

Where this sits in the architecture
------------------------------------
A blob store is the pipeline's *source of truth*. Raw ingested files land here
under a content-addressed key (``raw/{sha256}/original{ext}``); later the parse
cache lands beside them (``parsed/{sha256}/{parser_fingerprint}.md``). Vector
stores and indexes are *derived* from these bytes and are always rebuildable —
the blob store is the one thing you must not lose.

Deliberately tiny interface (Interface Segregation): ``put`` / ``get`` /
``exists`` over opaque string keys. That is the whole contract. Two consequences
worth stating out loud, because they are design decisions, not omissions:

- **The store attaches no meaning to keys.** The content-addressed layout
  (hashing a Source, choosing ``raw/…`` vs ``parsed/…``) lives in the *caller*
  (the pipeline/facade), never in here — exactly as ``OcrEngine`` knows nothing
  about PDFs. That keeps ``LocalBlobStore`` and ``MinioBlobStore`` perfectly
  interchangeable: swap the truth store from disk to S3 without touching a line
  of ingestion code (Strategy / Liskov).

- **A BlobStore is a Component for the plumbing, not for the cache math.** It
  gets identity, config, and secret-redacting ``describe()``/``fingerprint()``
  like every stage — but its fingerprint is *not* a stage-output cache key the
  way a Parser's is. It is a side-effecting service that sits *underneath* the
  fingerprint chain, so never fold a blob store into a trial's cache identity.

Streaming note: ``put(bytes)`` / ``get() -> bytes`` materialize one blob in
memory. That is fine for the current callers (a parsed markdown of even a
2,000-page PDF is a few MB). Streaming ``put_stream`` / ``get_stream`` variants
are a documented future addition (YAGNI until a caller needs them); the small
interface is what makes adding them non-breaking.
"""

from __future__ import annotations

from abc import abstractmethod

from ..core.component import Component
from ..core.errors import StorageError

__all__ = ["BlobStore"]


class BlobStore(Component):
    """Strategy interface: durable, keyed byte storage.

    Concrete stores (``LocalBlobStore`` on disk, ``MinioBlobStore`` on any
    S3-compatible backend) are Adapters over a storage backend. Callers depend
    only on this contract, so the truth store is swappable by config.

    Contract every implementation must honor (see
    ``tests/contract_checks.py::assert_blob_store_contract``):

    - ``put(key, data)`` stores ``data`` under ``key``, **overwriting** any
      existing value. Idempotent for content-addressed keys (same key ⇒ same
      bytes), so re-ingesting a file is a safe no-op-in-effect.
    - ``get(key)`` returns the exact bytes previously stored, or raises
      ``StorageError`` if the key is absent.
    - ``exists(key)`` returns a bool and **never raises** for a missing key —
      it is the cheap pre-check that lets a pipeline dedup uploads.
    - Keys are opaque strings; ``/`` is allowed and denotes a logical path
      (``raw/<hash>/original.pdf``). Round-trips must be byte-exact and binary
      safe (NUL bytes, arbitrary 0x00–0xFF).
    """

    kind = "blob_store"

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, overwriting any existing value.
        Raise ``StorageError`` on backend failure."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the bytes stored under ``key``. Raise ``StorageError`` (with
        the offending key) if nothing is stored there."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True iff a blob is stored under ``key``. Must not raise for a
        missing key — that is the whole point of the cheap pre-check."""

    def url(self, key: str, *, expires_seconds: int = 3600) -> str:
        """A directly-usable URL for the blob (a download link).

        Optional capability — the durable stores implement it (``LocalBlobStore``
        returns a ``file://`` URI; ``MinioBlobStore`` returns a presigned,
        time-limited S3 GET URL). ``expires_seconds`` is honored by stores that
        can expire links and ignored by those that can't (a local file path
        doesn't expire). The default declares it unsupported rather than faking
        a link — a `DocumentCatalog` uses this to turn a `doc_id` into a
        download link."""
        raise StorageError(f"{type(self).__name__}: url() not supported", key=key)
