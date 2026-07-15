"""The OCR seam: how any OCR provider plugs into ingestion.

Two separate concerns, deliberately kept apart (Single Responsibility):

1. WHEN to OCR  → `OcrPolicy` (a decision)
   AUTO   detect per page: pages with a real text layer skip OCR entirely
   FORCE  OCR every page (rescues PDFs with garbage embedded text layers —
          a common artifact of bad scanning software)
   NEVER  text-layer only, fastest, silently yields nothing for pure scans

2. HOW to OCR   → `OcrEngine` (a Strategy)
   The interface is intentionally tiny (Interface Segregation): one page
   image in, markdown out. It does NOT know about PDFs, pages order,
   batching windows, or documents — that orchestration belongs to parsers.
   Small surface ⇒ writing a custom engine is ~15 lines.

Every concrete engine (Mistral, Google Document AI, Tesseract, your own
model) is an Adapter: it translates a vendor API into this one contract.
"""

from __future__ import annotations

import base64
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Optional

from ...core.component import Component

__all__ = ["OcrPolicy", "PageImage", "OcrResult", "OcrEngine"]


class OcrPolicy(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    NEVER = "never"


@dataclass(frozen=True)
class PageImage:
    """A rendered page (or a standalone input image) handed to an engine.

    Raw bytes + mime rather than a PIL object: keeps the contract
    dependency-free and directly serializable (cloud APIs want bytes/base64
    anyway; local engines can decode with PIL in one line).
    """

    data: bytes
    page_number: int = 1
    mime: str = "image/png"
    dpi: int = 200
    metadata: dict = field(default_factory=dict)

    def to_data_url(self) -> str:
        """data: URL form — the shape most vision/OCR HTTP APIs accept."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime};base64,{encoded}"


@dataclass
class OcrResult:
    """What an engine returns for one page.

    `markdown` because that is the toolkit's lingua franca. Engines that only
    produce plain text return it as-is (valid markdown). `confidence` is
    optional — many APIs don't expose one — but when present the eval suite
    can correlate answer quality with OCR confidence, which is a genuinely
    useful diagnostic. `raw` keeps the untouched provider payload for
    debugging without polluting the contract.
    """

    markdown: str
    confidence: Optional[float] = None
    raw: Any = None


class OcrEngine(Component):
    """Strategy interface: turn one page image into markdown."""

    kind = "ocr"

    @abstractmethod
    def recognize(self, image: PageImage) -> OcrResult:
        """OCR a single page image. Raise `OcrError` on failure."""

    def recognize_batch(self, images: Iterable[PageImage]) -> Iterator[OcrResult]:
        """Default: sequential. Engines with true batch endpoints (or that
        want to parallelize HTTP calls) override this — the parser always
        calls through here, so the optimization is transparent."""
        for image in images:
            yield self.recognize(image)
