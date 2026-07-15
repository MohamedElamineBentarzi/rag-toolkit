"""PlainTextParser: .txt/.md pass-through — and the "hello world" of parsers.

Deliberately included as the smallest possible real Parser, so anyone writing
a custom one has a 60-line reference. Two things worth noting even in this
trivial case:

1. Streaming still applies. A 500 MB log file is text too. We read fixed
   binary blocks through an *incremental* UTF-8 decoder (a multi-byte char
   split across a block boundary would corrupt with naive per-block decode)
   and emit synthetic "pages" of ~page_chars characters, cut at newlines.
2. Markdown is a superset of plain text, so "conversion" is the identity —
   the value added is uniform paging + provenance, keeping the downstream
   contract identical across all parsers.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Iterator

from ...core.contracts import Page, Source, SourceFormat
from ...core.registry import registry
from .base import Parser

__all__ = ["PlainTextParser"]

_BLOCK_SIZE = 64 * 1024


@registry.register
class PlainTextParser(Parser):
    name = "plaintext"
    version = "0.1.0"
    supported_formats = (SourceFormat.TEXT, SourceFormat.MARKDOWN)

    @dataclass
    class Config:
        encoding: str = "utf-8"
        page_chars: int = 8_000   # synthetic page size (chars)

    def iter_pages(self, source: Source) -> Iterator[Page]:
        decoder = codecs.getincrementaldecoder(self.config.encoding)(
            errors="replace"
        )
        buffer = ""
        page_number = 0

        with source.open() as stream:
            while True:
                block = stream.read(_BLOCK_SIZE)
                if not block:
                    buffer += decoder.decode(b"", final=True)
                    break
                buffer += decoder.decode(block)

                while len(buffer) >= self.config.page_chars:
                    cut = self._cut_point(buffer)
                    page_number += 1
                    yield Page(number=page_number, markdown=buffer[:cut])
                    buffer = buffer[cut:]

        if buffer.strip():
            yield Page(number=page_number + 1, markdown=buffer)

    def _cut_point(self, buffer: str) -> int:
        """Prefer cutting at a newline so pages don't split mid-line;
        fall back to a hard cut when no newline exists in range."""
        newline = buffer.rfind("\n", 0, self.config.page_chars)
        return newline + 1 if newline != -1 else self.config.page_chars
