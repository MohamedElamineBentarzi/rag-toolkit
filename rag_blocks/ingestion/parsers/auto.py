"""AutoParser: "give me any file" — the Facade of the ingestion subsystem.

Composite Strategy + Facade: AutoParser is itself a Parser (so anything that
accepts a Parser accepts it — Liskov), but it owns no parsing logic. It
detects the format, looks the route up in a plain dict, and delegates to the
real parser built from the registry.

Because routes are *data* (`{"pdf": "docling", "txt": "plaintext"}`), they
are overridable per instance and per format without subclassing — swap the
PDF route to your own parser in one config line. This is also the hook the
evaluation suite uses to try parser alternatives as just another dimension
of the search space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ...core.contracts import Page, Source, SourceFormat
from ...core.errors import UnsupportedFormatError
from ...core.registry import registry
from ..detection import detect_format
from .base import Parser

__all__ = ["AutoParser"]


def _default_routes() -> dict[str, str]:
    return {
        SourceFormat.PDF.value: "docling",
        SourceFormat.DOCX.value: "docling",
        SourceFormat.PPTX.value: "docling",
        SourceFormat.XLSX.value: "docling",
        SourceFormat.HTML.value: "docling",
        SourceFormat.IMAGE.value: "docling",
        SourceFormat.TEXT.value: "plaintext",
        SourceFormat.MARKDOWN.value: "plaintext",
    }


@registry.register
class AutoParser(Parser):
    name = "auto"
    version = "0.1.0"
    supported_formats = tuple(f for f in SourceFormat if f is not SourceFormat.UNKNOWN)

    @dataclass
    class Config:
        #: format value → parser name (registry lookup).
        routes: dict = field(default_factory=_default_routes)
        #: parser name → config overrides forwarded on construction, e.g.
        #: {"docling": {"ocr_engine": "mistral", "page_batch_size": 4}}
        parser_configs: dict = field(default_factory=dict)

    def __init__(self, config=None, **overrides) -> None:
        super().__init__(config, **overrides)
        self._delegates: dict[str, Parser] = {}  # built lazily, then reused
        # (a Docling parser holds warm layout models — never rebuild per file)

    def iter_pages(self, source: Source) -> Iterator[Page]:
        fmt = detect_format(source)
        parser_name = self.config.routes.get(fmt.value)
        if parser_name is None:
            raise UnsupportedFormatError(
                f"No route configured for format {fmt.value!r} "
                f"(source: {source.uri}). Known routes: "
                f"{sorted(self.config.routes)}"
            )
        delegate = self._delegate(parser_name)
        # Pass the verdict down so the delegate never re-detects (frozen
        # Source ⇒ derive a variant instead of mutating).
        yield from delegate.iter_pages(source.with_format(fmt))

    def _delegate(self, parser_name: str) -> Parser:
        if parser_name not in self._delegates:
            overrides = self.config.parser_configs.get(parser_name, {})
            delegate = registry.create("parser", parser_name, **overrides)
            if not isinstance(delegate, Parser):
                raise UnsupportedFormatError(
                    f"Route target {parser_name!r} is registered but is not a "
                    "Parser"
                )
            self._delegates[parser_name] = delegate
        return self._delegates[parser_name]
