"""Generation subsystem: (query, context) → Answer with citations.

Importing this package registers the built-in generators. `extractive` is
zero-dependency and deterministic (baseline + hermetic tests); `anthropic` is
the real-LLM Adapter behind an optional extra; `openrouter` is a real-LLM
Adapter that is *also* zero-dependency (a plain HTTPS call to OpenRouter's
OpenAI-compatible endpoint), reaching many providers with one key.
"""

from .anthropic_generator import AnthropicGenerator
from .base import Generator
from .extractive import ExtractiveGenerator
from .openrouter_generator import OpenRouterGenerator

__all__ = [
    "Generator",
    "ExtractiveGenerator",
    "AnthropicGenerator",
    "OpenRouterGenerator",
]
