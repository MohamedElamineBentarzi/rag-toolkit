"""Google Document AI adapter.

Second Adapter, same seam — its whole point is to prove the abstraction:
Mistral returns markdown, Google returns text + layout entities, yet both
collapse into the same `OcrResult` and the parser never knows the difference
(Liskov substitution in action).

Optional extra: `rag-blocks[google]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.errors import OcrError
from ...core.registry import registry
from .base import OcrEngine, OcrResult, PageImage

__all__ = ["GoogleDocAiOcrEngine"]


@registry.register
class GoogleDocAiOcrEngine(OcrEngine):
    """OCR pages through a Google Document AI processor.

    `processor_name` is the full resource path:
        projects/{project}/locations/{location}/processors/{processor_id}
    """

    name = "google-docai"
    version = "0.1.0"

    @dataclass
    class Config:
        processor_name: str = ""

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import documentai  # lazy optional dep
            except ImportError as exc:
                raise OcrError(
                    "GoogleDocAiOcrEngine requires 'google-cloud-documentai'. "
                    "Install with: pip install 'rag-blocks[google]'"
                ) from exc
            self._documentai = documentai
            self._client = documentai.DocumentProcessorServiceClient()
        return self._client

    def recognize(self, image: PageImage) -> OcrResult:
        client = self._get_client()
        if not self.config.processor_name:
            raise OcrError("GoogleDocAiOcrEngine: processor_name is required")
        try:
            request = self._documentai.ProcessRequest(
                name=self.config.processor_name,
                raw_document=self._documentai.RawDocument(
                    content=image.data, mime_type=image.mime
                ),
            )
            result = client.process_document(request=request)
        except Exception as exc:  # noqa: BLE001
            raise OcrError(
                f"Google Document AI failed: {exc}",
                page_number=image.page_number,
            ) from exc

        # DocAI returns plain text plus layout entities. Plain text is valid
        # markdown, so this is correct-but-basic; reconstructing headings and
        # tables from the layout entities is a good future improvement that
        # stays entirely inside this adapter (Open/Closed at work).
        return OcrResult(markdown=result.document.text, raw=result)
