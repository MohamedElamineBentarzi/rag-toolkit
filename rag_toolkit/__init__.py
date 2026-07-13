"""rag-toolkit: composable building blocks for production RAG pipelines.

Design in one paragraph: stages are swappable Strategies registered under a
(kind, name) key; they communicate only through typed data contracts
(Source → Page → Document → Chunk → …); ingestion is streaming-first so
memory never scales with document size; every component carries a config
fingerprint that later powers the auto-tuning suite's cross-pipeline caching.

Quick start:

    import rag_toolkit as rk

    doc = rk.ingest("report.pdf")                       # sane defaults
    doc = rk.ingest("scan.pdf", ocr_engine="mistral",   # cloud OCR
                    ocr_policy=rk.OcrPolicy.FORCE)

    parser = rk.AutoParser()                            # streaming access
    for page in parser.iter_pages(rk.Source.from_path("huge.pdf")):
        ...
"""

from __future__ import annotations

from pathlib import Path

from .core import (
    Chunk,
    Component,
    Document,
    EmbeddingError,
    Page,
    PageSpan,
    RagToolkitError,
    Source,
    SourceFormat,
    StorageError,
    registry,
)
from .embedding import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .ingestion import (
    AutoParser,
    DoclingParser,
    OcrEngine,
    OcrPolicy,
    OcrResult,
    PageImage,
    Parser,
    PlainTextParser,
    detect_format,
)
from .chunking import Chunker, FixedChunker, MarkdownChunker
from .pipeline import IndexingPipeline, TraceEvent
from .storage import BlobStore, LocalBlobStore, MinioBlobStore

__version__ = "0.1.0"

__all__ = [
    "ingest",
    "registry",
    "Component",
    "Source",
    "SourceFormat",
    "Page",
    "PageSpan",
    "Document",
    "Chunk",
    "Parser",
    "AutoParser",
    "DoclingParser",
    "PlainTextParser",
    "OcrEngine",
    "OcrPolicy",
    "OcrResult",
    "PageImage",
    "detect_format",
    "Chunker",
    "FixedChunker",
    "MarkdownChunker",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "BlobStore",
    "LocalBlobStore",
    "MinioBlobStore",
    "IndexingPipeline",
    "TraceEvent",
    "RagToolkitError",
    "StorageError",
    "EmbeddingError",
]


def ingest(path: str | Path, **docling_overrides) -> Document:
    """One-call Facade: file path → markdown Document with provenance.

    Keyword arguments are forwarded to the PDF/office parser (DoclingParser),
    since that is where 95% of tuning happens:

        ingest("scan.pdf", ocr_engine="mistral", page_batch_size=4)

    For full control (custom routes, streaming, in-memory sources), drop one
    level down to AutoParser / DoclingParser directly.
    """
    parser_configs = {"docling": docling_overrides} if docling_overrides else {}
    parser = AutoParser(parser_configs=parser_configs)
    return parser.parse(Source.from_path(path))
