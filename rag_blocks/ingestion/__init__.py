"""Ingestion subsystem: any file → markdown Document with page provenance.

Importing this package registers the built-in parsers and OCR engines
(module import is the registration side effect the registry relies on).
"""

from .detection import detect_format
from .ocr.base import OcrEngine, OcrPolicy, OcrResult, PageImage
from .ocr.google_docai import GoogleDocAiOcrEngine
from .ocr.mistral import MistralOcrEngine
from .parsers.auto import AutoParser
from .parsers.base import Parser
from .parsers.docling_parser import DoclingParser
from .parsers.plaintext import PlainTextParser

__all__ = [
    "detect_format",
    "Parser",
    "AutoParser",
    "DoclingParser",
    "PlainTextParser",
    "OcrEngine",
    "OcrPolicy",
    "OcrResult",
    "PageImage",
    "MistralOcrEngine",
    "GoogleDocAiOcrEngine",
]
