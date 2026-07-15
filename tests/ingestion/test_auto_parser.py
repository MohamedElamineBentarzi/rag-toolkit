"""AutoParser: routing is data — and the swappability thesis gets a test."""
import pytest

from rag_blocks.core.contracts import Page, Source, SourceFormat
from rag_blocks.core.errors import UnsupportedFormatError
from rag_blocks.core.registry import registry
from rag_blocks.ingestion.parsers.auto import AutoParser
from rag_blocks.ingestion.parsers.base import Parser
from tests.contract_checks import assert_parser_contract


def test_markdown_routes_to_plaintext_end_to_end(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\nBody.")
    doc = AutoParser().parse(Source.from_path(f))
    assert "# Title" in doc.markdown
    assert doc.metadata["page_count"] == 1


def test_unroutable_format_is_a_clear_error():
    blob = Source.from_bytes(b"\x00\x01\x02", name="mystery")
    with pytest.raises(UnsupportedFormatError, match="unknown"):
        list(AutoParser().iter_pages(blob))


@registry.register
class ShoutParser(Parser):
    """Toy parser used to prove routes are swappable configuration."""

    name = "shout"
    supported_formats = (SourceFormat.TEXT, SourceFormat.MARKDOWN)

    def iter_pages(self, source):
        with source.open() as f:
            yield Page(number=1, markdown=f.read().decode().upper())


def test_routes_are_swappable_configuration(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("quiet words")
    doc = AutoParser(routes={"md": "shout"}).parse(Source.from_path(f))
    assert doc.markdown == "QUIET WORDS"


def test_delegates_are_built_once_and_reused():
    parser = AutoParser()
    assert parser._delegate("plaintext") is parser._delegate("plaintext")


def test_auto_parser_satisfies_the_parser_contract(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("contract body\n" * 50)
    assert_parser_contract(AutoParser(), Source.from_path(f))
