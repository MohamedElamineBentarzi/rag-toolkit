"""Core layer: contracts, component model, registry, errors.

Nothing in here knows about RAG specifics — it is the framework the stages
are built on. Zero third-party dependencies by design.
"""

from .component import Component
from .contracts import (
    Answer,
    Chunk,
    Citation,
    Document,
    Page,
    PageSpan,
    Query,
    ScoredChunk,
    Source,
    SourceFormat,
)
from .errors import (
    ComponentNotFoundError,
    ConfigError,
    DuplicateComponentError,
    EmbeddingError,
    GenerationError,
    OcrError,
    ParseError,
    RagToolkitError,
    StorageError,
    UnsupportedFormatError,
)
from .registry import Registry, registry

__all__ = [
    "Component",
    "Registry",
    "registry",
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
    "RagToolkitError",
    "ComponentNotFoundError",
    "DuplicateComponentError",
    "ConfigError",
    "UnsupportedFormatError",
    "ParseError",
    "OcrError",
    "StorageError",
    "EmbeddingError",
    "GenerationError",
]
