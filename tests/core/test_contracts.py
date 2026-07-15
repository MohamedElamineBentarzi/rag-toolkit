"""Contracts are the load-bearing walls — they get the most paranoid tests."""
import pytest

from rag_blocks.core.contracts import Document, Page, Source


def test_from_path_fails_fast_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Source.from_path(tmp_path / "ghost.pdf")


def test_head_reads_only_a_prefix(tmp_path):
    p = tmp_path / "big.txt"
    p.write_bytes(b"x" * 100_000)
    assert Source.from_path(p).head(16) == b"x" * 16


def test_content_hash_depends_on_bytes_not_location(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"same content")
    on_disk = Source.from_path(p)
    in_memory = Source.from_bytes(b"same content", name="anything.txt")
    assert on_disk.content_hash() == in_memory.content_hash()
    assert Source.from_bytes(b"other").content_hash() != in_memory.content_hash()


def test_document_spans_slice_back_to_exact_page_text():
    """THE provenance guarantee: doc.markdown[span] == original page text."""
    pages = [Page(1, "# Title"), Page(2, "Body of page two."), Page(3, "End.")]
    doc = Document.from_pages(Source.from_bytes(b"x", name="d.md"), pages)
    assert doc.metadata["page_count"] == 3
    for page, span in zip(pages, doc.pages):
        assert doc.markdown[span.start:span.end] == page.markdown
        assert span.page_number == page.number


def test_pages_for_span_crossing_a_boundary():
    pages = [Page(1, "AAAA"), Page(2, "BBBB")]  # [0:4) sep [4:6) [6:10)
    doc = Document.from_pages(Source.from_bytes(b"x", name="d.md"), pages)
    assert doc.pages_for_span(3, 7) == [1, 2]
    assert doc.pages_for_span(0, 2) == [1]


def test_ocr_flags_propagate_into_provenance():
    pages = [Page(1, "digital"), Page(2, "scanned", ocr_applied=True)]
    doc = Document.from_pages(Source.from_bytes(b"x", name="d.md"), pages)
    assert doc.metadata["ocr_pages"] == [2]
    assert [s.ocr_applied for s in doc.pages] == [False, True]


def test_document_id_is_deterministic(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("hello")
    a = Document.from_pages(Source.from_path(p), [Page(1, "hello")])
    b = Document.from_pages(Source.from_path(p), [Page(1, "hello")])
    assert a.id == b.id
