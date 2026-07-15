"""DocumentCatalog: the one-hop `doc_id → source provenance + download link`.

A citation carries a `doc_id` (the full sha256 content hash). That is also the
address of the raw blob (`raw/{doc_id}/original{ext}`), but two facts a citation
still lacks stand between a `doc_id` and something a user can see: the file's
extension `ext` (needed to complete the raw key) and its human-readable
`source_uri` (the PDF name to show). Those live in the parse cache's
`parsed/{hash}/{parser_fingerprint}.meta.json` — but reaching *that* would pin
citation resolution to a specific parser fingerprint (a document may be parsed by
several over time), coupling raw-file provenance to an unrelated parse artifact.

So at index time the pipeline writes a tiny, parser-independent manifest keyed
*directly by `doc_id`*:

    docs/{doc_id}.json  →  {doc_id, source_uri, content_hash, ext}

and a reader with only a `doc_id` gets, in one hop: the `source_uri` to display
and — via `BlobStore.url` — a download link to the original bytes. Doc-scoped
identity here, parser-scoped parse cache there: a deliberate separation, not just
duplication.

The key convention lives here (and is reused by the pipeline for the raw key),
keeping the "content-addressing lives in the caller, not the store" rule: the
`BlobStore` still just moves opaque key→bytes and hands out a URL for a key.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

from ..core.contracts import Document
from ..storage.base import BlobStore

__all__ = ["DocumentCatalog", "DocumentRef", "raw_key", "manifest_key"]


def raw_key(content_hash: str, ext: str) -> str:
    """The truth-store key for a source's immutable original bytes."""
    return f"raw/{content_hash}/original{ext}"


def manifest_key(doc_id: str) -> str:
    """The manifest key — indexed by `doc_id`, so a citation resolves in one hop."""
    return f"docs/{doc_id}.json"


@dataclass(frozen=True)
class DocumentRef:
    """What a `doc_id` resolves to: enough to name the source and fetch it."""

    doc_id: str
    source_uri: str        # the original path/name, e.g. "report.pdf"
    content_hash: str      # full sha256 — the raw-blob key component
    ext: str               # canonical extension, e.g. ".pdf"


class DocumentCatalog:
    """Read/write the `doc_id → DocumentRef` manifest over a `BlobStore`."""

    def __init__(self, blob_store: BlobStore) -> None:
        self._store = blob_store

    def record(
        self, document: Document, content_hash: str, ext: str,
        *, overwrite: bool = False,
    ) -> bool:
        """Write the manifest for one indexed document. Returns True if it wrote.

        Skip-if-present by default, so re-ingesting cached content is a no-op
        (one `exists` check) — matching how raw bytes and the parse cache skip.
        First-filename-wins for identical content ingested under two names; pass
        `overwrite=True` to re-point it."""
        key = manifest_key(document.id)
        if not overwrite and self._store.exists(key):
            return False
        ref = DocumentRef(
            doc_id=document.id,
            source_uri=document.source_uri,
            content_hash=content_hash,
            ext=ext,
        )
        self._store.put(key, json.dumps(asdict(ref)).encode("utf-8"))
        return True

    def get(self, doc_id: str) -> Optional[DocumentRef]:
        """Resolve a `doc_id` to its `DocumentRef`, or `None` if unknown."""
        key = manifest_key(doc_id)
        if not self._store.exists(key):
            return None
        return DocumentRef(**json.loads(self._store.get(key).decode("utf-8")))

    def source_uri(self, doc_id: str) -> Optional[str]:
        """The original file path/name for a `doc_id` (for citation display)."""
        ref = self.get(doc_id)
        return ref.source_uri if ref else None

    def download_url(
        self, doc_id: str, *, expires_seconds: int = 3600
    ) -> Optional[str]:
        """A download link to the original bytes for a `doc_id`, or `None` if
        the doc is unknown. Requires a `BlobStore` that implements `url`
        (`LocalBlobStore` → file URI, `MinioBlobStore` → presigned S3 URL)."""
        ref = self.get(doc_id)
        if ref is None:
            return None
        return self._store.url(
            raw_key(ref.content_hash, ref.ext), expires_seconds=expires_seconds
        )
