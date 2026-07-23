"""Data contracts: the typed artifacts that flow *between* pipeline stages.

This module is the backbone of the whole library. Stages never talk to each
other directly — a Chunker doesn't know what a Parser is. They only agree on
these frozen-ish data shapes. That is what makes every stage swappable
(Strategy pattern at the architecture level): as long as your custom parser
emits `Page` objects, every downstream component works with it unchanged.

Key decisions
-------------
1. `Source` is a *lazy pointer* to data, never an eager blob. Holding a path
   (or a small in-memory buffer) instead of `bytes` of a 900-page PDF is the
   first line of defense against the "load everything at once" failure mode.
2. `Page` is the streaming unit of ingestion. Parsers yield pages one by one
   (generators), so memory stays O(page batch), not O(document).
3. `Document` keeps `PageSpan` provenance: character offsets of each page in
   the final markdown. Later, when a Chunk covers chars [1200:1800], we can
   answer "that came from pages 4-5 of report.pdf" — which is what makes
   citations in RAG answers possible. Provenance is designed in from day one
   because it is nearly impossible to bolt on later.
4. Plain stdlib dataclasses, no pydantic: the core stays zero-dependency.
   Validation-heavy config at the edges (API servers, YAML loading) can wrap
   these in pydantic models without the hot path paying for it.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Union

__all__ = [
    "SourceFormat",
    "Source",
    "Page",
    "PageSpan",
    "Document",
    "Chunk",
    "Query",
    "ScoredChunk",
    "Citation",
    "Answer",
    "SparseVector",
    "VectorValue",
    "VectorSpec",
]


class SourceFormat(str, Enum):
    """Normalized input formats the ingestion subsystem understands.

    str-Enum so values serialize cleanly into configs, logs and trial records.
    """

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    MARKDOWN = "md"
    TEXT = "txt"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Source:
    """A lazy reference to input data (file path or in-memory bytes).

    Frozen (immutable) on purpose: sources get passed across stages and
    cached; nobody should mutate them mid-flight. Use `dataclasses.replace`
    to derive variants (e.g. attach a detected `format_hint`).
    """

    uri: str
    data: Optional[bytes] = None          # only for genuinely in-memory sources
    format_hint: Optional[SourceFormat] = None
    metadata: dict = field(default_factory=dict)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path, **metadata) -> "Source":
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            # Fail fast: better to explode here than 3 stages later.
            raise FileNotFoundError(f"Source file not found: {p}")
        return cls(uri=str(p), metadata=metadata)

    @classmethod
    def from_bytes(cls, data: bytes, name: str = "<memory>", **metadata) -> "Source":
        """In-memory source. Give `name` a real filename with extension when
        you know it — downstream format detection uses it as a tiebreaker."""
        return cls(uri=name, data=data, metadata=metadata)

    # -- lazy access --------------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        if self.data is not None:
            return None
        p = Path(self.uri)
        return p if p.is_file() else None

    def open(self) -> BinaryIO:
        """Open a binary stream over the data. Caller closes (use `with`).

        This is *the* access point: components read through here instead of
        slurping files themselves, so streaming discipline lives in one place.
        """
        if self.data is not None:
            return io.BytesIO(self.data)
        p = self.path
        if p is None:
            raise FileNotFoundError(f"Cannot open source: {self.uri}")
        return p.open("rb")

    def head(self, n: int = 8192) -> bytes:
        """Read only the first `n` bytes — enough for magic-byte sniffing
        without ever touching the rest of a multi-GB file."""
        with self.open() as f:
            return f.read(n)

    def content_hash(self) -> str:
        """Streaming sha256 of the content (1 MiB blocks, O(1) memory).

        This becomes the cache key of the whole evaluation suite later:
        (source hash × component fingerprint) uniquely identifies a stage
        output, so re-running 30 pipeline combos never re-parses a file.
        """
        h = hashlib.sha256()
        with self.open() as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def with_format(self, fmt: SourceFormat) -> "Source":
        return replace(self, format_hint=fmt)


@dataclass
class Page:
    """The streaming unit emitted by parsers.

    `number` is 1-based (matching how humans and PDF viewers count).
    `ocr_applied` is recorded per page because quality differs wildly between
    a native text layer and OCR output — downstream stages (or the eval
    suite) may want to weigh or filter accordingly.
    """

    number: int
    markdown: str
    ocr_applied: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PageSpan:
    """Provenance record: where page `page_number` lives inside
    `Document.markdown` (character offsets, [start, end))."""

    page_number: int
    start: int
    end: int
    ocr_applied: bool = False


@dataclass
class Document:
    """A fully parsed, markdown-normalized document with page provenance."""

    id: str
    markdown: str
    pages: list[PageSpan]
    source_uri: str
    metadata: dict = field(default_factory=dict)

    PAGE_SEPARATOR = "\n\n"

    @classmethod
    def from_pages(
        cls,
        source: Source,
        pages: Iterable[Page],
        doc_id: Optional[str] = None,
    ) -> "Document":
        """Assemble pages into one markdown string, recording offsets.

        Assembly lives here (on the data, not in the Parser) because it is a
        pure data concern — every parser gets identical, correct provenance
        for free instead of re-implementing offset math.
        """
        parts: list[str] = []
        spans: list[PageSpan] = []
        cursor = 0
        ocr_pages: list[int] = []

        for page in pages:
            if parts:
                cursor += len(cls.PAGE_SEPARATOR)
            start = cursor
            parts.append(page.markdown)
            cursor += len(page.markdown)
            spans.append(
                PageSpan(page.number, start, cursor, ocr_applied=page.ocr_applied)
            )
            if page.ocr_applied:
                ocr_pages.append(page.number)

        if doc_id is None:
            doc_id = _stable_document_id(source)

        return cls(
            id=doc_id,
            markdown=cls.PAGE_SEPARATOR.join(parts),
            pages=spans,
            source_uri=source.uri,
            metadata={
                "page_count": len(spans),
                "ocr_pages": ocr_pages,
                **source.metadata,
            },
        )

    def pages_for_span(self, start: int, end: int) -> list[int]:
        """Which page numbers does the char range [start, end) touch?
        This is the provenance query a Chunker (and citation code) uses."""
        return [
            s.page_number for s in self.pages if s.start < end and s.end > start
        ]


@dataclass
class Chunk:
    """A retrieval unit, produced by the Chunker stage.

    Provenance chain, visible end to end:
    Source → Page → Document(+PageSpan) → Chunk(char_start/char_end → pages).

    `char_start`/`char_end` are the *primary* provenance — the half-open
    [start, end) offsets into `document.markdown` the chunk was sliced from.
    `page_start`/`page_end` are *derived* from them via
    `Document.pages_for_span`. Both are `Optional` only to allow synthetic
    chunks (enricher summaries, generated Q/A) that never came from a
    document's markdown; for any chunk sliced from a parsed document the base
    chunker ALWAYS fills all four — a doc-derived chunk with `None` here is a
    bug.
    """

    id: str
    doc_id: str
    text: str
    index: int                      # position of the chunk within its document
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Query:
    """A search request. `filters` is payload-equality scoping handed straight
    to the store/index (e.g. `{"doc_id": x}`); `metadata` is the pressure valve
    for retriever-specific hints (query expansions, weights)."""

    text: str
    filters: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A Chunk paired with a relevance score, produced by search/retrieval.

    `score` semantics belong to whatever produced it (cosine similarity, an
    RRF score, a reranker logit) — the only cross-stage guarantee is *higher
    means more relevant*, so callers sort descending. `retriever_name` records
    which component produced the result: essential once hybrid retrieval fuses
    several sources, and the hook for per-source eval attribution.
    """

    chunk: Chunk
    score: float
    retriever_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    """A resolved source reference behind a generated answer.

    `marker` is the inline number the answer text uses (`[1]`, `[2]`, …), so a
    UI can link a claim to its source. The rest is provenance carried straight
    through from the chunk — this is the end of the chain that lets an answer
    say "from pages 4–5 of report.pdf".

    The human-readable source name (and a download link) is resolved from
    `doc_id`, not carried per-chunk: `DocumentCatalog.get(doc_id)` /
    `download_url(doc_id)` read the doc manifest in the blob store.
    """

    marker: int
    chunk_id: str
    doc_id: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class Answer:
    """The final generation output: text plus the sources it stands on.

    `citations` are resolved through chunk → page provenance, so every answer
    can be audited back to the exact passages it used. `usage` carries cost
    signals (tokens, latency) for the eval suite; empty for non-LLM generators.
    """

    text: str
    citations: list["Citation"] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# -- representation contracts (DR-0001 v2) ---------------------------------
#
# A Chunk is a *fact*; how it is made searchable is an *interpretation* under a
# particular encoder. Those interpretations are named, typed, keyed data that
# live in the store and the embedding cache — never on the Chunk itself. These
# three shapes are the vocabulary the multi-representation ChunkIndex and the
# multi-vector VectorStore speak.


@dataclass(frozen=True)
class SparseVector:
    """A sparse embedding: parallel term-index / weight arrays.

    The static (SPLADE-style) counterpart of a dense `list[float]`. Frozen and
    tuple-backed so it is hashable and safe to pass across cached stages.
    `indices` and `values` are parallel and equal length.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]


#: The value stored under one named vector space for one chunk: a dense vector
#: (`list[float]`) or a `SparseVector`. Which one is legal is fixed by the
#: matching `VectorSpec.kind`.
VectorValue = Union[list[float], SparseVector]


@dataclass(frozen=True)
class VectorSpec:
    """The declared schema of one named vector space inside a `VectorStore`.

    A store is a *named, typed* multi-vector index: each vector-backed
    `Representation` ("dense", "splade", …) declares one spec. `dimensions` and
    `distance` describe dense spaces only; sparse spaces ignore them.

    `kind` is an **open string**, not a closed literal (DR-0004 D5): a new
    representation plugin may define its own storage kind. Opening it does NOT
    oblige a store to accept every kind — each backend validates the kinds it
    physically supports at `ensure_schema` and raises on the rest (loud beats
    lossy). `"dense"` and `"sparse"` are what the built-in stores support.
    """

    name: str
    kind: str
    dimensions: Optional[int] = None      # dense only
    distance: str = "cosine"              # dense only


def _stable_document_id(source: Source) -> str:
    """Deterministic id: same content ⇒ same id ⇒ idempotent re-indexing.

    The id is the FULL sha256 content hash (not a prefix): a truncated id trades
    a silent-collision risk — two documents sharing a `doc_id` would overwrite
    each other's chunks in the store — for shorter ids, a bad bargain in a
    general-purpose library. Because it equals the content hash, it is also the
    address of the raw blob (`raw/{doc_id}/original{ext}`).

    Falls back to a uri hash (or uuid) when content isn't hashable, so the
    happy path stays deterministic without making the API brittle.
    """
    try:
        return source.content_hash()
    except OSError:
        if source.uri and source.uri != "<memory>":
            return hashlib.sha256(source.uri.encode()).hexdigest()
        return uuid.uuid4().hex
