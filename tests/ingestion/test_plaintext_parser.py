"""PlainTextParser: streaming discipline on the simplest possible format."""
from rag_blocks.core.contracts import Source
from rag_blocks.ingestion.parsers.plaintext import _BLOCK_SIZE, PlainTextParser
from tests.contract_checks import assert_parser_contract


def parse_pages(content: str, **cfg):
    parser = PlainTextParser(**cfg)
    return list(parser.iter_pages(Source.from_bytes(content.encode(), name="t.txt")))


def test_pages_reassemble_to_original_content():
    content = "line one\nline two\n" * 300
    pages = parse_pages(content, page_chars=500)
    assert "".join(p.markdown for p in pages) == content  # lossless split
    assert [p.number for p in pages] == list(range(1, len(pages) + 1))


def test_cuts_prefer_newlines():
    pages = parse_pages("alpha\nbeta\ngamma\ndelta\n", page_chars=12)
    assert all(p.markdown.endswith("\n") for p in pages)


def test_hard_cut_when_no_newline_available():
    pages = parse_pages("x" * 250, page_chars=100)
    assert [len(p.markdown) for p in pages] == [100, 100, 50]


def test_multibyte_char_straddling_a_read_block_survives():
    """Regression guard for the incremental decoder: a 2-byte UTF-8 char cut
    in half by the 64 KiB read boundary must decode intact."""
    content = "a" * (_BLOCK_SIZE - 1) + "é" + "tail"
    pages = parse_pages(content, page_chars=10 * _BLOCK_SIZE)
    joined = "".join(p.markdown for p in pages)
    assert joined == content
    assert "\ufffd" not in joined  # no replacement characters


def test_whitespace_only_input_yields_no_pages():
    assert parse_pages("   \n \n") == []


def test_satisfies_the_parser_contract(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("contract body\n" * 50)
    assert_parser_contract(PlainTextParser(page_chars=200), Source.from_path(f))
