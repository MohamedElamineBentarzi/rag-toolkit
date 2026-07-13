"""Generation subsystem: (query, context) → Answer with citations.

Importing this package registers the built-in generators. `extractive` is
zero-dependency and deterministic (baseline + hermetic tests); `anthropic` is
the real-LLM Adapter behind an optional extra.
"""

from .anthropic_generator import AnthropicGenerator
from .base import Generator
from .extractive import ExtractiveGenerator

__all__ = [
    "Generator",
    "ExtractiveGenerator",
    "AnthropicGenerator",
]
