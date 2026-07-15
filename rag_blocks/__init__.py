"""rag-blocks: composable building blocks for production RAG pipelines.

Design in one paragraph: stages are swappable Strategies registered under a
(kind, name) key; they communicate only through typed data contracts
(Source → Page → Document → Chunk → …); ingestion is streaming-first so
memory never scales with document size; every component carries a config
fingerprint that later powers the auto-tuning suite's cross-pipeline caching.

Quick start:

    import rag_blocks as rk

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
    Answer,
    Chunk,
    Citation,
    Component,
    Document,
    EmbeddingError,
    EnrichmentError,
    GenerationError,
    Page,
    PageSpan,
    Query,
    RagBlocksError,
    ScoredChunk,
    Source,
    SourceFormat,
    SparseVector,
    StorageError,
    VectorSpec,
    VectorValue,
    registry,
)
from .embedding import (
    CachingEmbedder,
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    SparseEncoder,
)
from .indexing import ChunkIndex, ChunkSink, DocumentCatalog, DocumentRef
from .enrichment import (
    ContextualEnricher,
    Enricher,
    HeadingEnricher,
)
from .generation import AnthropicGenerator, ExtractiveGenerator, Generator
from .retrieval import (
    FusionRetriever,
    HybridRetriever,
    HydeRetriever,
    IndexRetriever,
    MultiQueryRetriever,
    Retriever,
)
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
from .refinement import (
    CrossEncoderReranker,
    KeywordRefiner,
    NeighborExpander,
    Refiner,
    ScoreThreshold,
)
from .pipeline import IndexingPipeline, QueryPipeline, RagPipeline, TraceEvent
from .storage import (
    BM25Index,
    BlobStore,
    LexicalIndex,
    LocalBlobStore,
    MemoryVectorStore,
    MinioBlobStore,
    QdrantVectorStore,
    VectorStore,
)

__version__ = "0.6.0"

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
    "Query",
    "ScoredChunk",
    "Citation",
    "Answer",
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
    "SparseEncoder",
    "CachingEmbedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "SparseVector",
    "VectorSpec",
    "VectorValue",
    "ChunkIndex",
    "ChunkSink",
    "DocumentCatalog",
    "DocumentRef",
    "Enricher",
    "HeadingEnricher",
    "ContextualEnricher",
    "BlobStore",
    "LocalBlobStore",
    "MinioBlobStore",
    "VectorStore",
    "MemoryVectorStore",
    "QdrantVectorStore",
    "LexicalIndex",
    "BM25Index",
    "Retriever",
    "IndexRetriever",
    "FusionRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "HydeRetriever",
    "Refiner",
    "CrossEncoderReranker",
    "KeywordRefiner",
    "NeighborExpander",
    "ScoreThreshold",
    "Generator",
    "ExtractiveGenerator",
    "AnthropicGenerator",
    "IndexingPipeline",
    "QueryPipeline",
    "RagPipeline",
    "TraceEvent",
    "RagBlocksError",
    "StorageError",
    "EmbeddingError",
    "GenerationError",
    "EnrichmentError",
]


def ingest(path: str | Path, **docling_overrides) -> Document:
    """One-call Facade: file path → markdown Document with provenance.

    Keyword arguments are forwarded to the PDF/office parser (DoclingParser),
    since that is where 95% of tuning happens:

        ingest("scan.pdf", ocr_engine="mistral", page_batch_size=4)

    For full control (custom routes, streaming, in-memory sources), drop one
    level down to AutoParser / DoclingParser directly.

    Batch note: this builds a fresh ``AutoParser`` every call, which reloads the
    docling layout models per file. To ingest many documents, hold one parser
    and reuse it — ``p = AutoParser(); [p.parse(Source.from_path(f)) for f in files]``
    — so the expensive models load once.
    """
    parser_configs = {"docling": docling_overrides} if docling_overrides else {}
    parser = AutoParser(parser_configs=parser_configs)
    return parser.parse(Source.from_path(path))
