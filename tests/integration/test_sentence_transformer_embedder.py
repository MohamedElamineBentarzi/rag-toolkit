"""SentenceTransformerEmbedder against a REAL model. Opt-in.

Needs the extra installed and will download the model on first run:

    pip install 'rag-blocks[sentence-transformers]'
    rag_blocks_TEST_ST_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
        pytest -m integration tests/integration/test_sentence_transformer_embedder.py

A small model is used by default (fast, ~80 MB) rather than the bge-m3 default.
"""
import os

import pytest

from rag_blocks.embedding.sentence_transformer import SentenceTransformerEmbedder
from tests.contract_checks import assert_embedder_contract

pytestmark = pytest.mark.integration

_MODEL = os.environ.get(
    "rag_blocks_TEST_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


@pytest.fixture
def embedder():
    pytest.importorskip("sentence_transformers")
    return SentenceTransformerEmbedder(model=_MODEL)


def test_satisfies_the_embedder_contract(embedder):
    assert_embedder_contract(embedder)


def test_query_instruction_changes_the_query_vector():
    pytest.importorskip("sentence_transformers")
    plain = SentenceTransformerEmbedder(model=_MODEL, query_instruction="")
    prefixed = SentenceTransformerEmbedder(
        model=_MODEL, query_instruction="Represent this query: "
    )
    # The asymmetry seam actually does something: prefixing shifts the vector.
    assert plain.embed_query("capital of France") != prefixed.embed_query(
        "capital of France"
    )
    # ...but passages are never prefixed, so those match.
    assert plain.embed_texts(["a passage"]) == prefixed.embed_texts(["a passage"])
