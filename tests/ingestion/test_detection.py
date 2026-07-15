"""Detection: table-driven via parametrize — one behavior, many cases."""
import io
import zipfile

import pytest

from rag_blocks.core.contracts import Source, SourceFormat
from rag_blocks.ingestion.detection import detect_format


def zip_with(member: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, "x")
    return buf.getvalue()


@pytest.mark.parametrize(
    "head,expected",
    [
        (b"%PDF-1.7 junk", SourceFormat.PDF),
        (b"\x89PNG\r\n\x1a\n....", SourceFormat.IMAGE),
        (b"\xff\xd8\xff\xe0 jfif", SourceFormat.IMAGE),
        (b"GIF89a....", SourceFormat.IMAGE),
        (b"II*\x00....", SourceFormat.IMAGE),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", SourceFormat.IMAGE),
        (b"<!DOCTYPE html><html>", SourceFormat.HTML),
        (b"   <HTML><body>", SourceFormat.HTML),
    ],
)
def test_magic_bytes_beat_everything(head, expected):
    # name has no extension on purpose: bytes alone must be enough
    assert detect_format(Source.from_bytes(head, name="no_extension")) == expected


@pytest.mark.parametrize(
    "member,expected",
    [
        ("word/document.xml", SourceFormat.DOCX),
        ("ppt/slides/slide1.xml", SourceFormat.PPTX),
        ("xl/workbook.xml", SourceFormat.XLSX),
    ],
)
def test_zip_family_disambiguated_by_members(member, expected):
    assert detect_format(Source.from_bytes(zip_with(member), name="blob")) == expected


def test_unknown_zip_layout_falls_back_to_extension_then_unknown():
    data = zip_with("whatever/file")
    assert detect_format(Source.from_bytes(data, name="t.docx")) == SourceFormat.DOCX
    assert detect_format(Source.from_bytes(data, name="t.bin")) == SourceFormat.UNKNOWN


def test_extension_tiebreak_for_signatureless_text():
    assert detect_format(Source.from_bytes(b"# notes", name="n.md")) == SourceFormat.MARKDOWN
    assert detect_format(Source.from_bytes(b"plain", name="n.txt")) == SourceFormat.TEXT


def test_text_heuristic_and_binary_rejection():
    assert detect_format(Source.from_bytes(b"just words", name="mystery")) == SourceFormat.TEXT
    assert detect_format(Source.from_bytes(b"\x00\x01\x02", name="mystery")) == SourceFormat.UNKNOWN


def test_format_hint_short_circuits():
    src = Source.from_bytes(b"plain text", name="x").with_format(SourceFormat.PDF)
    assert detect_format(src) == SourceFormat.PDF
