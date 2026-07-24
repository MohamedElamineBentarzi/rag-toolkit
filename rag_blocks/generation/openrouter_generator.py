"""OpenRouterGenerator: answer synthesis via OpenRouter's OpenAI-compatible API.

Pattern: Adapter, like `AnthropicGenerator` — but **zero-dependency**. OpenRouter
is a single JSON-over-HTTPS endpoint speaking the OpenAI chat-completions shape,
so we speak it directly with the stdlib (`urllib`) rather than pull a vendor SDK.
One key (`OPENROUTER_API_KEY`) reaches hundreds of models across providers via
the `provider/model` id — `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4"`,
`"google/gemini-2.5-pro"`, … — so this one adapter is a whole fleet of backends.

The prompt instructs the model to answer only from the numbered context and cite
with `[n]` markers that line up with the packed blocks; citation numbering and
resolution stay in the base Template Method. This class is just the translation
layer: (query + packed context) → (answer text + token usage).

Credentials follow the toolkit policy: explicit `api_key`, else the
`OPENROUTER_API_KEY` env var (raw HTTP has no SDK to resolve a profile, so a
missing key is a clear `GenerationError` at call time). `site_url` / `site_name`
optionally set OpenRouter's `HTTP-Referer` / `X-Title` app-ranking headers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from ..core.contracts import Query
from ..core.errors import GenerationError
from ..core.registry import registry
from .base import Generator
from .packing import PackedContext

__all__ = ["OpenRouterGenerator"]

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context blocks. Cite every claim inline with bracketed numbers like [1] "
    "that match the block numbers. If the context does not contain the answer, "
    "say that you don't know rather than guessing."
)


@registry.register
class OpenRouterGenerator(Generator):
    name = "openrouter"
    version = "0.1.0"

    @dataclass
    class Config:
        model: str = "openai/gpt-4o-mini"    # any OpenRouter `provider/model` id
        max_tokens: int = 1024
        api_key: Optional[str] = None        # else OPENROUTER_API_KEY
        max_context_chars: int = 8000
        system: Optional[str] = None         # override the default instructions
        temperature: Optional[float] = None  # None ⇒ omit (provider default)
        site_url: Optional[str] = None       # optional HTTP-Referer (rankings)
        site_name: Optional[str] = None      # optional X-Title (rankings)
        timeout: float = 60.0

    def _complete(self, query: Query, packed: PackedContext) -> tuple[str, dict]:
        system = self.config.system or _DEFAULT_SYSTEM
        user = f"Context:\n{packed.prompt_block}\n\nQuestion: {query.text}"
        payload = self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return _content(payload), _usage(payload)

    def complete(self, prompt: str) -> str:
        """Bare text completion: `(prompt) -> str`, no context packing or
        citation resolution — the `complete` seam query-shaping retrievers
        (`MultiQueryRetriever`, `HydeRetriever`) and contextual enrichers need.
        Pass `generator.complete` wherever a `Callable[[str], str]` is asked."""
        return _content(self._chat([{"role": "user", "content": prompt}]))

    def _chat(self, messages: list[dict]) -> dict:
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise GenerationError(
                "OpenRouterGenerator needs an API key: set OPENROUTER_API_KEY "
                "or pass api_key= (get one at https://openrouter.ai/keys)."
            )
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            headers["X-Title"] = self.config.site_name

        raw = self._post(json.dumps(body).encode("utf-8"), headers)
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise GenerationError(f"OpenRouter returned invalid JSON: {exc}") from exc
        if not payload.get("choices"):
            raise GenerationError(f"OpenRouter returned no choices: {payload}")
        return payload

    def _post(self, data: bytes, headers: dict) -> bytes:
        """The single network primitive (a seam tests override). POST `data` to
        the endpoint; normalize any transport/HTTP failure to GenerationError."""
        req = urllib.request.Request(_ENDPOINT, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise GenerationError(
                f"OpenRouter request failed (HTTP {exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GenerationError(f"OpenRouter request failed: {exc.reason}") from exc


def _content(payload: dict) -> str:
    """The assistant text from an OpenAI-shaped response (never None)."""
    return payload["choices"][0]["message"].get("content") or ""


def _usage(payload: dict) -> dict:
    """Token usage, keyed like the other generators (input/output tokens)."""
    u = payload.get("usage") or {}
    return {
        "input_tokens": u.get("prompt_tokens", 0),
        "output_tokens": u.get("completion_tokens", 0),
    }
