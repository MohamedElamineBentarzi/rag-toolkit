"""Format detection: decide *what* a Source is before deciding *who* parses it.

Decision: trust bytes, not extensions. Files get renamed, downloaded without
extensions, or mislabeled ("scan.pdf.txt"). Magic-byte sniffing on the first
8 KiB is cheap, never loads the file, and is right far more often. The
extension is kept only as a tiebreaker for formats that have no signature
(plain text vs markdown vs csv all look like... text).

The subtle case is the ZIP family: .docx/.pptx/.xlsx are all ZIP archives
(header `PK\\x03\\x04`), distinguished only by their internal folder layout.
So detection is two-phase there: sniff ZIP, then peek at the member names.
"""

from __future__ import annotations

import zipfile

from ..core.contracts import Source, SourceFormat

__all__ = ["detect_format"]

# Signatures checked against the head bytes, in order.
_MAGIC: list[tuple[bytes, SourceFormat]] = [
    (b"%PDF", SourceFormat.PDF),
    (b"\x89PNG\r\n\x1a\n", SourceFormat.IMAGE),
    (b"\xff\xd8\xff", SourceFormat.IMAGE),          # JPEG
    (b"GIF87a", SourceFormat.IMAGE),
    (b"GIF89a", SourceFormat.IMAGE),
    (b"II*\x00", SourceFormat.IMAGE),               # TIFF little-endian
    (b"MM\x00*", SourceFormat.IMAGE),               # TIFF big-endian
]

_EXTENSION_MAP: dict[str, SourceFormat] = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".pptx": SourceFormat.PPTX,
    ".xlsx": SourceFormat.XLSX,
    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,
    ".md": SourceFormat.MARKDOWN,
    ".markdown": SourceFormat.MARKDOWN,
    ".txt": SourceFormat.TEXT,
    ".png": SourceFormat.IMAGE,
    ".jpg": SourceFormat.IMAGE,
    ".jpeg": SourceFormat.IMAGE,
    ".tiff": SourceFormat.IMAGE,
    ".tif": SourceFormat.IMAGE,
    ".webp": SourceFormat.IMAGE,
    ".gif": SourceFormat.IMAGE,
}


def detect_format(source: Source) -> SourceFormat:
    """Best-effort format detection. Never reads more than the head bytes
    (plus, for ZIPs, the archive's central directory)."""
    if source.format_hint is not None:
        return source.format_hint

    head = source.head()

    for signature, fmt in _MAGIC:
        if head.startswith(signature):
            return fmt

    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return SourceFormat.IMAGE

    if head.startswith(b"PK\x03\x04"):
        return _detect_ooxml(source)

    stripped = head.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html")):
        return SourceFormat.HTML

    from_ext = _extension_of(source)
    if from_ext is not None:
        return from_ext

    if _looks_like_text(head):
        return SourceFormat.TEXT

    return SourceFormat.UNKNOWN


def _detect_ooxml(source: Source) -> SourceFormat:
    """Disambiguate the ZIP family by inspecting member paths.

    zipfile only reads the central directory (at the end of the file) — it
    does not decompress content, so this stays cheap even for big decks.
    """
    try:
        with source.open() as stream, zipfile.ZipFile(stream) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return _extension_of(source) or SourceFormat.UNKNOWN

    for prefix, fmt in (
        ("word/", SourceFormat.DOCX),
        ("ppt/", SourceFormat.PPTX),
        ("xl/", SourceFormat.XLSX),
    ):
        if any(n.startswith(prefix) for n in names):
            return fmt
    return _extension_of(source) or SourceFormat.UNKNOWN


def _extension_of(source: Source) -> SourceFormat | None:
    uri = source.uri.lower()
    for ext, fmt in _EXTENSION_MAP.items():
        if uri.endswith(ext):
            return fmt
    return None


def _looks_like_text(head: bytes) -> bool:
    """Heuristic: decodable UTF-8 without NUL bytes ⇒ treat as text."""
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # Tolerate a multi-byte char cut at the 8 KiB boundary.
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True
