"""DoclingParser ROUTING logic, tested WITHOUT docling installed.

The trick: the hard-to-get-wrong parts (segment planning, windowing, policy
dispatch, export fallback, engine wiring) were deliberately designed as pure
functions or seam-injected collaborators — so they are testable hermetically.
We test OUR logic; we do not re-test docling itself (that's their job, and
our integration tests' job).
"""
import pytest

from rag_blocks.core.contracts import Source
from rag_blocks.core.errors import ComponentNotFoundError
from rag_blocks.ingestion.ocr.base import OcrPolicy
from rag_blocks.ingestion.parsers.docling_parser import (
    DoclingParser,
    _image_mime,
    _windows,
)

FAKE_PDF = Source.from_bytes(b"%PDF-1.7 not a real pdf", name="f.pdf")


@pytest.mark.parametrize(
    "start,end,size,expected",
    [
        (1, 10, 4, [(1, 4), (5, 8), (9, 10)]),
        (1, 3, 8, [(1, 3)]),
        (5, 5, 2, [(5, 5)]),
    ],
)
def test_windows(start, end, size, expected):
    assert list(_windows(start, end, size)) == expected


def test_segment_planning_groups_consecutive_pages():
    parser = DoclingParser(min_chars_digital=32)
    counts = [500, 400, 0, 5, 999]  # 2 digital, 2 scanned, 1 digital
    assert parser._plan_segments(counts) == [
        ("docling", 1, 2),
        ("ocr", 3, 4),
        ("docling", 5, 5),
    ]


def test_unknown_engine_fails_at_construction_not_page_500():
    with pytest.raises(ComponentNotFoundError):
        DoclingParser(ocr_engine="does-not-exist")


def test_engine_identity_is_part_of_parser_identity():
    plain = DoclingParser()
    with_engine = DoclingParser(ocr_engine="fake-ocr")
    assert "ocr_engine_fingerprint" in with_engine.describe()
    assert plain.fingerprint() != with_engine.fingerprint()


@pytest.mark.parametrize(
    "engine,policy,expected_path,expected_flags",
    [
        (None, OcrPolicy.AUTO, "docling", (True, False)),
        (None, OcrPolicy.FORCE, "docling", (True, True)),
        (None, OcrPolicy.NEVER, "docling", (False, False)),
        ("fake-ocr", OcrPolicy.NEVER, "docling", (False, False)),
        ("fake-ocr", OcrPolicy.AUTO, "hybrid", None),
        ("fake-ocr", OcrPolicy.FORCE, "hybrid", None),
    ],
)
def test_pdf_dispatch_matrix(monkeypatch, engine, policy, expected_path, expected_flags):
    """The policy x engine matrix, verified branch by branch."""
    parser = DoclingParser(ocr_engine=engine, ocr_policy=policy)
    seen = {}

    def fake_docling(source, *, do_ocr, force):
        seen.update(path="docling", do_ocr=do_ocr, force=force)
        return iter(())

    def fake_hybrid(source):
        seen.update(path="hybrid")
        return iter(())

    monkeypatch.setattr(parser, "_iter_pdf_docling", fake_docling)
    monkeypatch.setattr(parser, "_iter_pdf_hybrid", fake_hybrid)

    list(parser._iter_pdf(FAKE_PDF))

    assert seen["path"] == expected_path
    if expected_flags is not None:
        assert (seen["do_ocr"], seen["force"]) == expected_flags


class ModernDoclingDoc:
    """Stub mimicking a docling document with per-page markdown export."""

    def export_to_markdown(self, page_no=None):
        return "  " if page_no == 3 else f"content of page {page_no}"


class LegacyDoclingDoc:
    """Stub mimicking an old docling document (no page_no kwarg)."""

    def export_to_markdown(self):
        return "whole window"


def test_per_page_export_skips_blank_pages():
    pages = list(
        DoclingParser()._pages_from_result(ModernDoclingDoc(), 2, 4, ocr_applied=False)
    )
    assert [p.number for p in pages] == [2, 4]  # blank page 3 dropped


def test_legacy_docling_falls_back_to_window_export():
    pages = list(
        DoclingParser()._pages_from_result(LegacyDoclingDoc(), 2, 4, ocr_applied=False)
    )
    assert len(pages) == 1
    assert pages[0].markdown == "whole window"
    assert pages[0].metadata["page_span"] == [2, 4]  # provenance degrades honestly


def test_image_with_external_engine_bypasses_docling_entirely():
    """Hermetic end-to-end through the REAL iter_pages path: PNG bytes in,
    OCR-produced markdown out — zero vendor dependencies involved."""
    png = Source.from_bytes(b"\x89PNG\r\n\x1a\n" + b"fakepixels", name="scan.png")
    parser = DoclingParser(ocr_engine="fake-ocr")
    (page,) = list(parser.iter_pages(png))
    assert page.ocr_applied is True
    assert page.metadata["ocr_engine"] == "fake-ocr"
    assert "fake ocr text" in page.markdown


@pytest.mark.parametrize(
    "head,mime",
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"II*\x00", "image/tiff"),
        (b"RIFF0000WEBP", "image/webp"),
    ],
)
def test_image_mime_sniff(head, mime):
    assert _image_mime(head) == mime
