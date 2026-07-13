"""Generator: the answer-synthesis Strategy interface.

Turns a query plus retrieved context into an `Answer` with citations. The
strategy that varies is *how the text is produced* (an LLM call, an extractive
heuristic, a template); the bookkeeping around it — packing context into a
numbered, budgeted block and resolving citation markers back to provenance — is
identical everywhere, so it lives once in the `generate` Template Method.

Subclasses implement only `_complete(query, packed) -> (text, usage)`. They
never touch citation math: the base packs the context, hands the strategy the
numbered block, then maps whichever `[n]` markers came back to the source
chunks. `usage` (tokens/cost) flows straight through to the eval suite.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from ..core.component import Component
from ..core.contracts import Answer, Query, ScoredChunk
from .packing import pack_context, PackedContext, resolve_citations

__all__ = ["Generator"]


class Generator(Component):
    """Strategy interface: (query, context) → Answer with citations."""

    kind = "generator"

    def generate(self, query: Query, context: Sequence[ScoredChunk]) -> Answer:
        """Template Method — pack context, delegate text, resolve citations."""
        packed = pack_context(context, max_chars=self._max_context_chars())
        text, usage = self._complete(query, packed)
        return Answer(
            text=text,
            citations=resolve_citations(text, packed.citations),
            usage=usage,
            metadata={"query": query.text},
        )

    @abstractmethod
    def _complete(
        self, query: Query, packed: PackedContext
    ) -> tuple[str, dict]:
        """Produce the answer text (and any usage stats) from the numbered
        context block. Raise `GenerationError` on failure."""

    def _max_context_chars(self) -> int:
        # A char budget is a crude but zero-dependency stand-in for a token
        # budget; adapters that know their tokenizer can override.
        return getattr(self.config, "max_context_chars", 8000)
