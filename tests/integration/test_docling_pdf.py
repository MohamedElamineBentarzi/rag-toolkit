"""Integration tests: the REAL vendor stack, opt-in only.

The default `pytest` run excludes these (see addopts in pyproject.toml) so
the suite stays fast and hermetic. Run them with:

    pip install "rag-blocks[docling]"
    rag_blocks_TEST_PDF=/path/to/any.pdf pytest -m integration
"""
import os

import pytest

pytestmark = pytest.mark.integration


def test_docling_pdf_roundtrip():
    pytest.importorskip("docling")
    pdf_path = os.environ.get("rag_blocks_TEST_PDF")
    if not pdf_path:
        pytest.skip("set rag_blocks_TEST_PDF to a local PDF to run this test")

    import rag_blocks as rk

    doc = rk.ingest(pdf_path)
    assert doc.markdown.strip()
    assert doc.metadata["page_count"] >= 1
    # every span must slice cleanly inside the assembled markdown
    for span in doc.pages:
        assert 0 <= span.start <= span.end <= len(doc.markdown)
