"""Shared test utilities.

The star here is FakeOcrEngine — a *test double*. Because DoclingParser
depends on the abstract OcrEngine (Dependency Inversion), tests can inject
this fake through the exact same registry seam production uses: no network,
no API key, fully deterministic, and it records every call for assertions.

If a class is hard to test, the design is wrong. This file is the proof the
design is right.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag_blocks.core.registry import registry
from rag_blocks.ingestion.ocr.base import OcrEngine, OcrResult, PageImage


class FakeOcrEngine(OcrEngine):
    """Deterministic OCR stand-in: canned markdown, call recording."""

    name = "fake-ocr"

    @dataclass
    class Config:
        reply: str = "fake ocr text"

    def __init__(self, config=None, **overrides):
        super().__init__(config, **overrides)
        self.calls: list[PageImage] = []

    def recognize(self, image: PageImage) -> OcrResult:
        self.calls.append(image)
        return OcrResult(
            markdown=f"{self.config.reply} (page {image.page_number})",
            confidence=0.99,
        )


def register_fakes() -> None:
    # Re-registering the *same* class is idempotent by design, so calling
    # this from multiple entry points is safe.
    registry.register(FakeOcrEngine)
