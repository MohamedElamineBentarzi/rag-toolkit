"""OpenRouterGenerator against the real OpenRouter API. Opt-in.

    OPENROUTER_API_KEY=... pytest -m integration \
        tests/integration/test_openrouter_generator.py

Zero-dependency (a plain HTTPS call), so there is no extra to install — only a
key. Defaults to a small, inexpensive model; override with model=.
"""
import os

import pytest

from rag_blocks.core.contracts import Chunk, Query, ScoredChunk
from rag_blocks.generation.openrouter_generator import OpenRouterGenerator
from tests.contract_checks import assert_generator_contract

pytestmark = pytest.mark.integration


def context(*texts):
    return [
        ScoredChunk(
            chunk=Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
                        char_start=i, char_end=i + 1, page_start=1, page_end=1),
            score=1.0 - i * 0.1,
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def generator():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("set OPENROUTER_API_KEY to run")
    return OpenRouterGenerator(model="openai/gpt-4o-mini", max_tokens=256)


def test_answers_from_context_and_reports_usage(generator):
    answer = generator.generate(
        Query(text="What is the capital of France?"),
        context("The capital of France is Paris.", "Bananas are yellow."),
    )
    assert "Paris" in answer.text
    assert answer.citations
    assert answer.usage["input_tokens"] > 0


def test_complete_returns_text(generator):
    out = generator.complete("Reply with the single word: pong")
    assert isinstance(out, str) and out.strip()


def test_satisfies_the_generator_contract(generator):
    assert_generator_contract(
        generator,
        Query(text="What color are bananas?"),
        context("Bananas are yellow.", "Grass is green."),
    )
