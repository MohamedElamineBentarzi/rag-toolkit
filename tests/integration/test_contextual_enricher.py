"""ContextualEnricher against the real Claude API. Opt-in.

    pip install 'rag-blocks[anthropic]'
    ANTHROPIC_API_KEY=... pytest -m integration \
        tests/integration/test_contextual_enricher.py
"""
import os

import pytest

from rag_blocks.core.contracts import Source
from rag_blocks.chunking.markdown import MarkdownChunker
from rag_blocks.enrichment.contextual import ContextualEnricher
from rag_blocks.ingestion.parsers.plaintext import PlainTextParser
from tests.contract_checks import assert_enricher_contract

pytestmark = pytest.mark.integration

_MD = "# Acme Q3\nRevenue rose 18% to $4.2M.\n\n# Headcount\nWe hired 12 engineers.\n"


def doc_and_chunks():
    src = Source.from_bytes(_MD.encode(), name="acme.md")
    doc = PlainTextParser(page_chars=10_000_000).parse(src)
    return doc, list(MarkdownChunker().chunk(doc))


@pytest.fixture
def enricher():
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("set ANTHROPIC_API_KEY to run")
    return ContextualEnricher(max_tokens=64)


def test_prepends_situating_context(enricher):
    doc, chunks = doc_and_chunks()
    out = list(enricher.enrich(iter(chunks), doc))
    # Every chunk grew (context prepended) but kept its original body + pages.
    for enriched, original in zip(out, chunks):
        assert enriched.text.endswith(original.text)
        assert len(enriched.text) > len(original.text)
        assert enriched.page_start == original.page_start


def test_satisfies_the_enricher_contract(enricher):
    doc, chunks = doc_and_chunks()
    assert_enricher_contract(enricher, doc, chunks)
