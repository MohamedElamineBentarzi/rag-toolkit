"""Orchestrators: thin wiring over the stage components.

The design rule the whole library is built to honor (AGENTS.md §2): **all
intelligence lives in components; all wiring lives here, and the wiring is
dumb.** An IndexingPipeline is a for-loop over generators plus two seams —
a tracing hook and an optional blob store. It is deliberately NOT a Component:
it is not a swappable algorithm, it is the glue that composes them.

`IndexingPipeline.index(sources)` runs, per source:

    Source ──parse──▶ Document ──chunk──▶ Iterator[Chunk]

and (when a blob store is wired in) captures the durable truth alongside:

    raw/{sha256}/original{ext}                     the immutable source bytes
    parsed/{sha256}/{parser_fingerprint}.md        the parse cache (markdown)
    parsed/{sha256}/{parser_fingerprint}.meta.json spans + doc metadata

Two design points worth stating, both settled earlier in design discussion:

- **Content-addressing lives here, not in the store.** The BlobStore is a dumb
  key→bytes service (it attaches no meaning to keys); the pipeline is the caller
  that knows the `raw/…` vs `parsed/…` convention. That is what keeps
  LocalBlobStore and MinioBlobStore perfectly interchangeable.
- **Capture is opt-in and idempotent.** No blob store ⇒ pure parse→chunk. With
  one, `exists()` is a cheap pre-check so re-indexing the same bytes is a no-op
  (dedup free — same content, same key).

Streaming note: chunks stream out per document (a generator), so memory stays
O(one document + its chunks), never O(corpus). Raw capture currently reads the
source into memory to `put` it (the BlobStore has no streaming `put` yet) — fine
for typical documents; a `put_stream` variant is the documented future fix for
multi-GB inputs. The parsed markdown is always small.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .chunking.base import Chunker
from .chunking.fixed import FixedChunker
from .core.contracts import Chunk, Document, Source, SourceFormat
from .ingestion.detection import detect_format
from .ingestion.parsers.auto import AutoParser
from .ingestion.parsers.base import Parser
from .storage.base import BlobStore

__all__ = ["TraceEvent", "IndexingPipeline"]


@dataclass
class TraceEvent:
    """One observation emitted at a stage boundary.

    This is the seam the evaluation suite later hangs cost attribution on
    (latency per stage, cache hits). Keeping it a plain dataclass — not a log
    string — means a hook can aggregate it however it likes.
    """

    stage: str                 # "parse" | "store_raw" | "store_parsed" | "chunk"
    source_uri: str
    duration_ms: float
    detail: dict = field(default_factory=dict)


#: A tracing hook: called with each TraceEvent. Defaults to a no-op (Null
#: Object) so pipeline code never grows `if trace is not None` branches.
TraceHook = Callable[[TraceEvent], None]


def _noop_trace(event: TraceEvent) -> None:  # Null Object
    pass


#: Canonical extension per known format — derived from the *detected* format,
#: never the (possibly lying) filename. IMAGE/UNKNOWN fall back to the uri
#: suffix since the concrete image type isn't carried in SourceFormat.
_CANONICAL_EXT = {
    SourceFormat.PDF: ".pdf",
    SourceFormat.DOCX: ".docx",
    SourceFormat.PPTX: ".pptx",
    SourceFormat.XLSX: ".xlsx",
    SourceFormat.HTML: ".html",
    SourceFormat.MARKDOWN: ".md",
    SourceFormat.TEXT: ".txt",
}


class IndexingPipeline:
    """Wire Source → Parser → Chunker, optionally persisting the truth store."""

    def __init__(
        self,
        parser: Optional[Parser] = None,
        chunker: Optional[Chunker] = None,
        blob_store: Optional[BlobStore] = None,
        trace: TraceHook = _noop_trace,
    ) -> None:
        # AutoParser routes any format; FixedChunker is a sane default.
        self.parser = parser if parser is not None else AutoParser()
        self.chunker = chunker if chunker is not None else FixedChunker()
        self.blob_store = blob_store
        self.trace = trace

    def index(self, sources: Source | Iterable[Source]) -> Iterator[Chunk]:
        """Stream chunks for every source, capturing truth blobs on the way."""
        if isinstance(sources, Source):
            sources = [sources]
        for source in sources:
            content_hash = source.content_hash() if self.blob_store else None
            if self.blob_store is not None and content_hash is not None:
                self._store_raw(source, content_hash)
            doc = self._parse(source)
            if self.blob_store is not None and content_hash is not None:
                self._store_parsed(source, doc, content_hash)
            yield from self._chunk(source, doc)

    # -- stages --------------------------------------------------------------

    def _parse(self, source: Source) -> Document:
        start = time.perf_counter()
        doc = self.parser.parse(source)
        self.trace(TraceEvent(
            "parse", source.uri, _ms(start),
            {"doc_id": doc.id, "pages": len(doc.pages)},
        ))
        return doc

    def _chunk(self, source: Source, doc: Document) -> Iterator[Chunk]:
        start = time.perf_counter()
        count = 0
        for chunk in self.chunker.chunk(doc):
            count += 1
            yield chunk
        self.trace(TraceEvent(
            "chunk", source.uri, _ms(start),
            {"doc_id": doc.id, "chunks": count},
        ))

    # -- truth store ---------------------------------------------------------

    def _store_raw(self, source: Source, content_hash: str) -> None:
        assert self.blob_store is not None
        key = f"raw/{content_hash}/original{_extension_for(source)}"
        start = time.perf_counter()
        hit = self.blob_store.exists(key)
        if not hit:
            with source.open() as stream:
                self.blob_store.put(key, stream.read())
        self.trace(TraceEvent(
            "store_raw", source.uri, _ms(start),
            {"key": key, "cache_hit": hit},
        ))

    def _store_parsed(
        self, source: Source, doc: Document, content_hash: str
    ) -> None:
        assert self.blob_store is not None
        fp = self.parser.fingerprint()
        md_key = f"parsed/{content_hash}/{fp}.md"
        meta_key = f"parsed/{content_hash}/{fp}.meta.json"
        start = time.perf_counter()
        hit = self.blob_store.exists(md_key)
        if not hit:
            self.blob_store.put(md_key, doc.markdown.encode("utf-8"))
            self.blob_store.put(meta_key, _meta_bytes(doc, content_hash, fp))
        self.trace(TraceEvent(
            "store_parsed", source.uri, _ms(start),
            {"key": md_key, "cache_hit": hit},
        ))


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _extension_for(source: Source) -> str:
    fmt = detect_format(source)
    ext = _CANONICAL_EXT.get(fmt)
    if ext is not None:
        return ext
    return Path(source.uri).suffix  # IMAGE/UNKNOWN: keep the original suffix


def _meta_bytes(doc: Document, content_hash: str, parser_fp: str) -> bytes:
    """Serialize the provenance needed to rebuild a Document without re-parsing
    (spans are the part that can't be recomputed from markdown alone)."""
    meta = {
        "doc_id": doc.id,
        "source_uri": doc.source_uri,
        "content_hash": content_hash,
        "parser_fingerprint": parser_fp,
        "metadata": doc.metadata,
        "pages": [
            [s.page_number, s.start, s.end, s.ocr_applied] for s in doc.pages
        ],
    }
    return json.dumps(meta, sort_keys=True).encode("utf-8")
