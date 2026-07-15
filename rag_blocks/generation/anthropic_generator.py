"""AnthropicGenerator: answer synthesis with Claude.

Pattern: Adapter. Claude's Messages API speaks (system + messages → content
blocks); our contract speaks (query + packed context → answer text + usage).
This class is the translation layer and the home of the prompt; citation
numbering and resolution stay in the base Template Method.

Default model is `claude-opus-4-8` (the current flagship). The prompt instructs
the model to answer only from the numbered context and cite with `[n]` markers
that line up with the packed blocks — so the base class can resolve each marker
back to a source chunk's pages. Note: Opus 4.8 rejects `temperature`/`top_p`, so
this adapter deliberately sends no sampling parameters.

Dependency handling: `anthropic` is imported lazily and declared as the optional
extra `rag-blocks[anthropic]`; the client is built once and reused. Credentials
follow the toolkit policy — explicit `api_key`, else the SDK's own resolution
(the `ANTHROPIC_API_KEY` env var or an `ant` profile), so we never force a key.

File named `anthropic_generator.py` to avoid shadowing the `anthropic` package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from ..core.contracts import Query
from ..core.errors import GenerationError
from ..core.registry import registry
from .base import Generator
from .packing import PackedContext

__all__ = ["AnthropicGenerator"]

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context blocks. Cite every claim inline with bracketed numbers like [1] "
    "that match the block numbers. If the context does not contain the answer, "
    "say that you don't know rather than guessing."
)


@registry.register
class AnthropicGenerator(Generator):
    name = "anthropic"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "claude-opus-4-8"
        max_tokens: int = 1024
        api_key: Optional[str] = None       # else ANTHROPIC_API_KEY / ant profile
        max_context_chars: int = 8000
        system: Optional[str] = None        # override the default instructions

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._client: Any = None  # heavy-ish; built once, reused across calls

    def _complete(self, query: Query, packed: PackedContext) -> tuple[str, dict]:
        client = self._get_client()
        system = self.config.system or _DEFAULT_SYSTEM
        user = f"Context:\n{packed.prompt_block}\n\nQuestion: {query.text}"
        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise GenerationError(f"Claude generation failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage

    def complete(self, prompt: str) -> str:
        """Bare text completion: `(prompt) -> str`, no context packing or
        citation resolution (DR-0001 v2, D5/F5).

        This is the `complete` seam that query-shaping retrievers
        (`MultiQueryRetriever`, `HydeRetriever`) and contextual enrichers need —
        a shape the `(query, context) -> Answer` `generate` contract
        deliberately doesn't expose. Pass `generator.complete` wherever a
        `Callable[[str], str]` is asked for."""
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - normalize vendor errors
            raise GenerationError(f"Claude completion failed: {exc}") from exc
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic  # lazy: optional dependency
            except ImportError as exc:
                raise GenerationError(
                    "AnthropicGenerator requires the 'anthropic' package. "
                    "Install with: pip install 'rag-blocks[anthropic]'"
                ) from exc
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            # api_key may be None — the SDK then resolves from env / ant profile;
            # do not fail here just because the env var is unset.
            self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        return self._client
