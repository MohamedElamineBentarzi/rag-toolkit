"""ExtractiveGenerator: a zero-dependency, deterministic generator.

No LLM: the "answer" is the single highest-ranked passage, returned verbatim
with its citation marker. That is a real extractive-QA baseline — the honest
floor the tuner compares every generative model against — and it makes the whole
query→answer→citation path testable without a network or an API key.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.contracts import Query
from ..core.registry import registry
from .base import Generator
from .packing import PackedContext

__all__ = ["ExtractiveGenerator"]


@registry.register
class ExtractiveGenerator(Generator):
    name = "extractive"
    version = "0.1.0"

    @dataclass
    class Config:
        max_context_chars: int = 4000

    def _complete(self, query: Query, packed: PackedContext) -> tuple[str, dict]:
        if not packed.citations:
            return ("I don't have enough context to answer that.", {})
        # Return the top passage, tagged with its citation marker so the base
        # class resolves it back to that chunk's provenance.
        top = packed.citations[0]
        return (f"{packed.texts[0]} [{top.marker}]", {})
