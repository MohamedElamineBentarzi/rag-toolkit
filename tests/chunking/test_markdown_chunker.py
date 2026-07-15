"""MarkdownChunker: cuts on heading boundaries."""
from rag_blocks.chunking.markdown import MarkdownChunker
from rag_blocks.core.contracts import Source
from rag_blocks.ingestion.parsers.plaintext import PlainTextParser
from tests.contract_checks import assert_chunker_contract


def doc(markdown):
    src = Source.from_bytes(markdown.encode(), name="t.md")
    return PlainTextParser(page_chars=10_000_000).parse(src)


def test_splits_at_each_heading():
    md = "# Title\nintro\n\n## A\nalpha\n\n## B\nbeta\n"
    chunks = list(MarkdownChunker().chunk(doc(md)))
    assert len(chunks) == 3
    assert chunks[0].text.startswith("# Title")
    assert chunks[1].text.startswith("## A")
    assert chunks[2].text.startswith("## B")


def test_content_before_first_heading_is_its_own_chunk():
    md = "preamble line\n\n# First\nbody\n"
    chunks = list(MarkdownChunker().chunk(doc(md)))
    assert chunks[0].text.startswith("preamble")
    assert chunks[1].text.startswith("# First")


def test_document_without_headings_is_one_chunk():
    md = "just paragraphs\n\nno headings here\n"
    chunks = list(MarkdownChunker().chunk(doc(md)))
    assert len(chunks) == 1
    assert chunks[0].text == md


def test_hash_without_space_is_not_a_heading():
    md = "# Real\nbody\n\n#nothashtag still body\n"
    chunks = list(MarkdownChunker().chunk(doc(md)))
    assert len(chunks) == 1  # only the real heading at offset 0 splits nothing


def test_reassembling_chunks_reconstructs_the_document():
    md = "# H1\nintro\n\n## H2\nbody\n\n## H3\ntail\n"
    chunks = list(MarkdownChunker().chunk(doc(md)))
    # Heading-based spans are contiguous and non-overlapping ⇒ lossless.
    assert "".join(c.text for c in chunks) == md


def test_satisfies_the_chunker_contract():
    md = "# H1\n" + "text\n\n" * 50 + "## H2\n" + "more\n\n" * 50
    assert_chunker_contract(MarkdownChunker(), doc(md))
