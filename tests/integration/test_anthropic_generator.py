"""AnthropicGenerator against the real Claude API. Opt-in.

    pip install 'rag-toolkit[anthropic]'
    ANTHROPIC_API_KEY=... pytest -m integration \
        tests/integration/test_anthropic_generator.py
"""
import os

import pytest

from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.generation.anthropic_generator import AnthropicGenerator
from tests.contract_checks import assert_generator_contract

pytestmark = pytest.mark.integration


def context(*texts):
    return [
        ScoredChunk(
            chunk=Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
                        char_start=i, char_end=i + 1, page_start=1, page_end=1),
            score=1.0 - i * 0.1, retriever_name="dense",
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def generator():
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("set ANTHROPIC_API_KEY to run")
    return AnthropicGenerator(max_tokens=256)


def test_answers_from_context_and_reports_usage(generator):
    answer = generator.generate(
        Query(text="What is the capital of France?"),
        context("The capital of France is Paris.", "Bananas are yellow."),
    )
    assert "Paris" in answer.text
    assert answer.citations
    assert answer.usage["input_tokens"] > 0


def test_satisfies_the_generator_contract(generator):
    assert_generator_contract(
        generator,
        Query(text="What color are bananas?"),
        context("Bananas are yellow.", "Grass is green."),
    )
