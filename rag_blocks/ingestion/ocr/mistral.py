"""Mistral OCR adapter.

Pattern: Adapter. Mistral's OCR API speaks (document/image URL → pages of
markdown); our contract speaks (PageImage → OcrResult). This class is the
translation layer and nothing else — no PDF logic, no page routing.

Dependency handling: `mistralai` is imported lazily inside the method, and
declared as the optional extra `rag-blocks[mistral]`. Users who never touch
Mistral pay zero install cost, and importing rag_blocks never fails because
one vendor SDK is missing ("batteries optional").

NOTE: written against the mistralai>=1.x SDK shape
(`client.ocr.process(model=..., document={"type": "image_url", ...})`).
Vendor SDKs move fast — verify the call signature against the current docs
before shipping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ...core.errors import OcrError
from ...core.registry import registry
from .base import OcrEngine, OcrResult, PageImage

__all__ = ["MistralOcrEngine"]


@registry.register
class MistralOcrEngine(OcrEngine):
    """OCR pages through Mistral's dedicated OCR model.

    A nice property of this provider: it natively outputs *markdown*
    (tables, headings included), so no post-processing layer is needed —
    the adapter is almost a pass-through.
    """

    name = "mistral"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "mistral-ocr-latest"
        api_key: Optional[str] = None      # falls back to MISTRAL_API_KEY env
        timeout_ms: int = 120_000

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None  # built lazily, reused across pages

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from mistralai import Mistral  # lazy: optional dependency
            except ImportError as exc:
                raise OcrError(
                    "MistralOcrEngine requires the 'mistralai' package. "
                    "Install with: pip install 'rag-blocks[mistral]'"
                ) from exc
            api_key = self.config.api_key or os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise OcrError(
                    "No Mistral API key: set config.api_key or MISTRAL_API_KEY"
                )
            self._client = Mistral(api_key=api_key, timeout_ms=self.config.timeout_ms)
        return self._client

    def recognize(self, image: PageImage) -> OcrResult:
        client = self._get_client()
        try:
            response = client.ocr.process(
                model=self.config.model,
                document={
                    "type": "image_url",
                    "image_url": image.to_data_url(),
                },
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise OcrError(
                f"Mistral OCR failed: {exc}", page_number=image.page_number
            ) from exc

        markdown = "\n\n".join(page.markdown for page in response.pages)
        return OcrResult(markdown=markdown, confidence=None, raw=response)
