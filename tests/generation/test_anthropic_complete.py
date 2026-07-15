"""AnthropicGenerator.complete: the bare (prompt) -> str seam, hermetic.

Injects a fake client so the `complete` adapter logic is exercised without the
`anthropic` SDK or a network call (the real path is covered by the opt-in
integration test)."""
from types import SimpleNamespace

from rag_toolkit.generation.anthropic_generator import AnthropicGenerator


class _FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def _generator_with(reply):
    gen = AnthropicGenerator(max_tokens=64)
    gen._client = SimpleNamespace(messages=_FakeMessages(reply))
    return gen


def test_complete_returns_plain_text():
    gen = _generator_with("four alternative queries")
    assert gen.complete("expand this query") == "four alternative queries"


def test_complete_sends_the_prompt_as_the_user_message():
    gen = _generator_with("ok")
    gen.complete("hello world")
    call = gen._client.messages.calls[0]
    assert call["messages"] == [{"role": "user", "content": "hello world"}]
    # complete is bare — no system prompt / citation scaffolding.
    assert "system" not in call


def test_complete_is_usable_as_the_callable_seam():
    gen = _generator_with("hypothetical passage")
    seam = gen.complete  # Callable[[str], str]
    assert seam("q") == "hypothetical passage"
