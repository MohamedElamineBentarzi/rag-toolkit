"""OpenRouterGenerator, fully hermetic.

Overrides the one network seam (`_post`) so the adapter's request construction
and OpenAI-shaped response parsing are exercised with no HTTP — the real path is
the opt-in integration test.
"""
import json
import urllib.error
from io import BytesIO

import pytest

from rag_blocks.core.contracts import Chunk, Query, ScoredChunk
from rag_blocks.core.errors import GenerationError
from rag_blocks.generation import openrouter_generator as mod
from rag_blocks.generation.openrouter_generator import _ENDPOINT, OpenRouterGenerator


def context(*texts):
    return [
        ScoredChunk(
            chunk=Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
                        char_start=i, char_end=i + 1, page_start=i + 1,
                        page_end=i + 1),
            score=1.0 - i * 0.1,
        )
        for i, t in enumerate(texts)
    ]


def _reply(content="the answer is 42 [1]", prompt=11, completion=7):
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }).encode("utf-8")


def _gen(reply=None, **cfg):
    """A generator whose `_post` is faked; returns (gen, sent) where `sent`
    captures the last request body + headers."""
    cfg.setdefault("api_key", "test-key")
    gen = OpenRouterGenerator(**cfg)
    sent: dict = {}

    def fake_post(data: bytes, headers: dict) -> bytes:
        sent["body"] = json.loads(data.decode("utf-8"))
        sent["headers"] = headers
        return reply if reply is not None else _reply()

    gen._post = fake_post  # type: ignore[method-assign]
    return gen, sent


def test_generate_parses_content_and_usage():
    gen, sent = _gen(reply=_reply("Paris is the capital [1].", prompt=20, completion=5))
    ans = gen.generate(Query(text="capital of France?"), context("Paris is the capital."))
    assert ans.text == "Paris is the capital [1]."
    assert ans.usage == {"input_tokens": 20, "output_tokens": 5}
    assert ans.citations[0].chunk_id == "d:0"  # base resolved [1] to provenance


def test_request_shape_system_then_user_and_auth_header():
    gen, sent = _gen(model="anthropic/claude-sonnet-4", max_tokens=256)
    gen.generate(Query(text="q"), context("ctx"))
    body = sent["body"]
    assert body["model"] == "anthropic/claude-sonnet-4"
    assert body["max_tokens"] == 256
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "q" in body["messages"][1]["content"]
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert "temperature" not in body  # None ⇒ omitted


def test_temperature_and_ranking_headers_are_optional():
    gen, sent = _gen(temperature=0.2, site_url="https://ex.com", site_name="Ex")
    gen.complete("hi")
    assert sent["body"]["temperature"] == 0.2
    assert sent["headers"]["HTTP-Referer"] == "https://ex.com"
    assert sent["headers"]["X-Title"] == "Ex"


def test_complete_is_a_bare_user_only_prompt():
    gen, sent = _gen(reply=_reply("hypothetical passage"))
    assert gen.complete("expand this") == "hypothetical passage"
    assert sent["body"]["messages"] == [{"role": "user", "content": "expand this"}]


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gen = OpenRouterGenerator()  # no api_key configured
    with pytest.raises(GenerationError, match="API key"):
        gen.complete("q")


def test_no_choices_raises():
    gen, _ = _gen(reply=json.dumps({"error": {"message": "bad model"}}).encode())
    with pytest.raises(GenerationError, match="no choices"):
        gen.complete("q")


def test_env_var_supplies_the_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    gen = OpenRouterGenerator()
    sent: dict = {}
    gen._post = lambda data, headers: (sent.update(headers=headers) or _reply())  # type: ignore[method-assign]
    gen.complete("q")
    assert sent["headers"]["Authorization"] == "Bearer from-env"


def test_http_error_becomes_generation_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(
            _ENDPOINT, 429, "Too Many Requests", {},
            BytesIO(b'{"error":{"message":"rate limited"}}'),
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    gen = OpenRouterGenerator(api_key="k")
    with pytest.raises(GenerationError, match="HTTP 429"):
        gen.complete("q")
