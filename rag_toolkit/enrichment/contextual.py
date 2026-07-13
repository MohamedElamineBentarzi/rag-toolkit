"""ContextualEnricher: Anthropic's contextual retrieval, via Claude.

Pattern: Adapter. For each chunk, ask Claude for a one-sentence situating
context given the whole document, and prepend it to the chunk text. This is the
technique the `heading` enricher approximates deterministically — an LLM writes
a better situating sentence than a heading, at the cost of a call per chunk
(prompt caching on the shared document prefix makes this affordable in practice).

Only the text is augmented; page provenance is preserved so citations still
resolve. Lazy `anthropic` import behind the `[anthropic]` extra; the client is
built once and reused; `api_key` is optional (SDK resolves env / ant profile).

File named `contextual.py`; the class calls Claude but is named for the
technique, not the vendor (there's one Anthropic adapter per stage).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Iterator, Optional

from ..core.contracts import Chunk, Document
from ..core.errors import EnrichmentError
from ..core.registry import registry
from .base import Enricher

__all__ = ["ContextualEnricher"]

_SYSTEM = (
    "Give a short, succinct context (one sentence) to situate this chunk within "
    "the overall document, to improve search retrieval of the chunk. Answer only "
    "with the succinct context and nothing else."
)


@registry.register
class ContextualEnricher(Enricher):
    name = "contextual"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "claude-opus-4-8"
        max_tokens: int = 128
        api_key: Optional[str] = None       # else ANTHROPIC_API_KEY / ant profile
        #: Cap the document text sent per chunk (bounds cost on huge documents).
        max_document_chars: int = 16000

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None

    def enrich(
        self, chunks: Iterator[Chunk], document: Document
    ) -> Iterator[Chunk]:
        doc_text = document.markdown[: self.config.max_document_chars]
        for chunk in chunks:
            context = self._situate(doc_text, chunk.text)
            yield replace(chunk, text=f"{context}\n\n{chunk.text}")

    def _situate(self, doc_text: str, chunk_text: str) -> str:
        client = self._get_client()
        user = (
            f"<document>\n{doc_text}\n</document>\n\n"
            f"Here is the chunk to situate:\n<chunk>\n{chunk_text}\n</chunk>"
        )
        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise EnrichmentError(f"Claude enrichment failed: {exc}") from exc
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic  # lazy: optional dependency
            except ImportError as exc:
                raise EnrichmentError(
                    "ContextualEnricher requires the 'anthropic' package. "
                    "Install with: pip install 'rag-toolkit[anthropic]'"
                ) from exc
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        return self._client
