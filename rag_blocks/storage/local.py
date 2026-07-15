"""LocalBlobStore: the zero-dependency, on-disk truth store.

The default ``BlobStore``: files under a root directory, keys mapped straight
to nested paths (``raw/ab12…/original.pdf`` → ``<root>/raw/ab12…/original.pdf``).
No vendor SDK, no server, no network — this is the batteries-included store you
get for free, and the one the test/tuning suites use.

Two bits of deliberate care beyond a naive ``open().write()``:

1. **Atomic writes.** A blob store is the source of truth; a half-written file
   after a crash would be silent corruption. So ``put`` writes to a temp file in
   the same directory and ``os.replace``s it into place — an atomic rename on
   every OS we target. Readers never observe a partial blob.

2. **Path-traversal containment.** Keys are library-generated today, but a store
   is a trust boundary, so we resolve every key against the root and refuse
   anything that escapes it (``..``, absolute paths). Fail fast, not silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import StorageError
from ..core.registry import registry
from .base import BlobStore

__all__ = ["LocalBlobStore"]


@registry.register
class LocalBlobStore(BlobStore):
    """Store blobs as files under a root directory."""

    name = "local"
    version = "0.1.0"

    @dataclass
    class Config:
        #: Root directory for all blobs. Relative paths resolve against the
        #: process CWD; the default lives under a gitignored cache dir.
        root: str = ".rag_cache/blobs"

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        # Resolve once; created lazily on first write, not at construction
        # (constructing a component must stay a cheap, side-effect-free act).
        self._root = Path(self.config.root).expanduser().resolve()

    def put(self, key: str, data: bytes) -> None:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: write a sibling temp file, then rename over the target.
        tmp = target.parent / f".{target.name}.{os.getpid()}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)  # don't leave temp turds behind
            raise StorageError(
                f"LocalBlobStore failed to write blob: {exc}", key=key
            ) from exc

    def get(self, key: str) -> bytes:
        target = self._resolve(key)
        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise StorageError("No blob stored under key", key=key) from exc
        except OSError as exc:
            raise StorageError(
                f"LocalBlobStore failed to read blob: {exc}", key=key
            ) from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def url(self, key: str, *, expires_seconds: int = 3600) -> str:
        # A local file link never expires, so expires_seconds is ignored.
        target = self._resolve(key)
        if not target.is_file():
            raise StorageError("No blob stored under key", key=key)
        return target.as_uri()

    # -- internals -----------------------------------------------------------

    def _resolve(self, key: str) -> Path:
        """Map an opaque key to a path *inside* the root, or refuse it.

        Containment is enforced by resolving the candidate and checking it is
        relative to the root — this catches ``..`` escapes and absolute keys.
        """
        if not key or not key.strip():
            raise StorageError("Blob key must be a non-empty string", key=key)
        candidate = (self._root / key).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise StorageError(
                "Blob key escapes the store root (path traversal refused)",
                key=key,
            )
        return candidate
