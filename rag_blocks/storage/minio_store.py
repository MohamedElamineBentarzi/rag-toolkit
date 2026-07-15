"""MinioBlobStore: the S3-compatible truth store.

Pattern: Adapter. MinIO's SDK speaks (bucket, object, stream, length); our
contract speaks (key, bytes). This class is the translation layer and nothing
else — no content-addressing, no pipeline knowledge.

Why MinIO covers more than MinIO: the ``minio`` client is a plain S3 client, so
one adapter serves a self-hosted MinIO server, AWS S3, Cloudflare R2, Backblaze
B2 — anything S3-compatible — selected purely by ``endpoint``. (License note:
the ``minio`` *client* SDK is Apache-2.0; only the MinIO *server* is AGPL, which
does not reach users of this library.)

Dependency handling: ``minio`` is imported lazily inside ``_get_client`` and
declared as the optional extra ``rag-blocks[minio]`` — importing rag_blocks
never requires it ("batteries optional").

Credentials follow the toolkit policy (AGENTS.md §7.4): explicit config wins,
else the ``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY`` environment variables. The
key fields are named so ``describe()``/``fingerprint()`` auto-redact them, and
rotating a key never invalidates a cache or leaks into a trial log.

NOTE: written against the minio>=7.2 client API (keyword arguments throughout,
``S3Error.code == "NoSuchKey"`` for a missing object). Vendor SDKs drift —
re-verify the call shapes on a dependency bump.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Optional

from ..core.errors import StorageError
from ..core.registry import registry
from .base import BlobStore

__all__ = ["MinioBlobStore"]


@registry.register
class MinioBlobStore(BlobStore):
    """Store blobs as objects in an S3-compatible bucket."""

    name = "minio"
    version = "0.1.0"

    @dataclass
    class Config:
        #: Host[:port] of the S3 service, no scheme (``secure`` controls TLS).
        endpoint: str = "localhost:9000"
        bucket: str = "rag-blocks"
        access_key: Optional[str] = None   # falls back to MINIO_ACCESS_KEY
        secret_key: Optional[str] = None   # falls back to MINIO_SECRET_KEY
        secure: bool = False               # True ⇒ HTTPS
        region: Optional[str] = None
        #: Create the bucket on first use if it is missing. Convenient for
        #: local/dev; set False against a locked-down bucket you don't own.
        make_bucket: bool = True

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None  # heavy + network — built once, reused

    # -- contract ------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        client = self._get_client()
        try:
            client.put_object(
                bucket_name=self.config.bucket,
                object_name=key,
                data=io.BytesIO(data),
                length=len(data),
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise StorageError(
                f"MinioBlobStore failed to write blob: {exc}", key=key
            ) from exc

    def get(self, key: str) -> bytes:
        client = self._get_client()
        response = None
        try:
            response = client.get_object(
                bucket_name=self.config.bucket, object_name=key
            )
            return response.read()
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            if _is_missing(exc):
                raise StorageError("No blob stored under key", key=key) from exc
            raise StorageError(
                f"MinioBlobStore failed to read blob: {exc}", key=key
            ) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.stat_object(bucket_name=self.config.bucket, object_name=key)
            return True
        except Exception as exc:  # noqa: BLE001
            if _is_missing(exc):
                return False
            raise StorageError(
                f"MinioBlobStore failed to stat blob: {exc}", key=key
            ) from exc

    def url(self, key: str, *, expires_seconds: int = 3600) -> str:
        from datetime import timedelta

        client = self._get_client()
        try:
            return client.presigned_get_object(
                bucket_name=self.config.bucket,
                object_name=key,
                expires=timedelta(seconds=expires_seconds),
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise StorageError(
                f"MinioBlobStore could not presign a URL: {exc}", key=key
            ) from exc

    # -- internals -----------------------------------------------------------

    def _credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Resolve (access_key, secret_key): explicit config wins, else env.

        Extracted as a seam so the fallback logic is unit-testable without a
        server (the library never reads ``.env``; the environment is the
        application's job)."""
        access = self.config.access_key or os.environ.get("MINIO_ACCESS_KEY")
        secret = self.config.secret_key or os.environ.get("MINIO_SECRET_KEY")
        return access, secret

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from minio import Minio  # lazy: optional dependency
            except ImportError as exc:
                raise StorageError(
                    "MinioBlobStore requires the 'minio' package. "
                    "Install with: pip install 'rag-blocks[minio]'"
                ) from exc
            access, secret = self._credentials()
            client = Minio(
                endpoint=self.config.endpoint,
                access_key=access,
                secret_key=secret,
                secure=self.config.secure,
                region=self.config.region,
            )
            self._ensure_bucket(client)
            self._client = client
        return self._client

    def _ensure_bucket(self, client: Any) -> None:
        try:
            if client.bucket_exists(bucket_name=self.config.bucket):
                return
            if not self.config.make_bucket:
                raise StorageError(
                    f"Bucket {self.config.bucket!r} does not exist "
                    "(and make_bucket is disabled)"
                )
            client.make_bucket(
                bucket_name=self.config.bucket, location=self.config.region
            )
        except StorageError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise StorageError(
                f"MinioBlobStore could not access bucket "
                f"{self.config.bucket!r}: {exc}"
            ) from exc


def _is_missing(exc: Exception) -> bool:
    """Is this vendor exception a 'the object isn't there' signal?

    Isolated so the one place that knows MinIO's ``S3Error.code`` string is a
    tiny, obvious function — the rest of the adapter stays vendor-agnostic."""
    code = getattr(exc, "code", None)
    return code in ("NoSuchKey", "NoSuchObject")
