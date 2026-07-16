"""Exception hierarchy for rag-blocks.

Design note: one root exception (`RagBlocksError`) so callers can do a
single broad `except RagBlocksError` at pipeline boundaries, plus narrow
subclasses so stage-level code can react precisely (e.g. retry on OcrError
but not on UnsupportedFormatError). Errors carry context (source, page)
because "PDF failed" is useless in a 10k-document batch run.
"""

from __future__ import annotations


class RagBlocksError(Exception):
    """Root of all rag-blocks exceptions."""


class ComponentNotFoundError(RagBlocksError):
    """Raised when the registry has no component under (kind, name)."""


class DuplicateComponentError(RagBlocksError):
    """Raised when two components register under the same (kind, name)."""


class ConfigError(RagBlocksError):
    """Raised when a component receives an invalid configuration."""


class UnsupportedFormatError(RagBlocksError):
    """Raised when no parser can handle a source's format."""


class ParseError(RagBlocksError):
    """Raised when parsing a source fails.

    Attributes:
        source_uri: which document failed.
        page_number: which page failed, if known (1-based).
    """

    def __init__(self, message: str, *, source_uri: str | None = None,
                 page_number: int | None = None) -> None:
        self.source_uri = source_uri
        self.page_number = page_number
        location = ""
        if source_uri:
            location += f" [source={source_uri}"
            if page_number is not None:
                location += f", page={page_number}"
            location += "]"
        super().__init__(message + location)


class OcrError(ParseError):
    """Raised when an OCR engine fails on a page image."""


class StorageError(RagBlocksError):
    """Raised when a blob store operation fails.

    Attributes:
        key: the blob key involved, if applicable — because "write failed"
            is useless when a pipeline is streaming thousands of blobs.
    """

    def __init__(self, message: str, *, key: str | None = None) -> None:
        self.key = key
        location = f" [key={key}]" if key else ""
        super().__init__(message + location)


class EmbeddingError(RagBlocksError):
    """Raised when an embedder fails to vectorize text (model load, inference,
    or a missing optional dependency)."""


class GenerationError(RagBlocksError):
    """Raised when a generator fails to produce an answer (LLM call, or a
    missing optional dependency)."""


class EnrichmentError(RagBlocksError):
    """Raised when an enricher fails to augment chunks (LLM call, or a missing
    optional dependency)."""
