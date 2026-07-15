"""CrossEncoderReranker against a real cross-encoder. Opt-in.

    pip install 'rag-toolkit[sentence-transformers]'
    pytest -m integration tests/integration/test_cross_encoder_refiner.py

A small cross-encoder is used by default (fast); override with
RAG_TOOLKIT_TEST_RERANKER_MODEL.
"""
import os

import pytest

from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.refinement.cross_encoder import CrossEncoderReranker
from tests.contract_checks import assert_refiner_contract

pytestmark = pytest.mark.integration

_MODEL = os.environ.get(
    "RAG_TOOLKIT_TEST_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def scored(i, text):
    chunk = Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                  char_start=i, char_end=i + 1, page_start=1, page_end=1)
    return ScoredChunk(chunk=chunk, score=0.5, retriever_name="index")


@pytest.fixture
def refiner():
    pytest.importorskip("sentence_transformers")
    return CrossEncoderReranker(model=_MODEL)


def test_satisfies_the_refiner_contract(refiner):
    assert_refiner_contract(refiner)


def test_puts_the_relevant_passage_first(refiner):
    candidates = [
        scored(0, "Bananas are a yellow fruit."),
        scored(1, "The capital of France is Paris."),
    ]
    out = refiner.refine(Query(text="What is the capital of France?"), candidates, 2)
    assert out[0].chunk.id == "d:1"
