"""MarkdownFixedChunker: heading sections, each capped by size."""
from rag_blocks.chunking.markdown_fixed import MarkdownFixedChunker
from rag_blocks.core.contracts import Source
from rag_blocks.ingestion.parsers.plaintext import PlainTextParser
from tests.contract_checks import assert_chunker_contract


def doc(markdown):
    src = Source.from_bytes(markdown.encode(), name="t.md")
    return PlainTextParser(page_chars=10_000_000).parse(src)


def chunk(md, **cfg):
    return list(MarkdownFixedChunker(**cfg).chunk(doc(md)))


def test_short_sections_stay_whole_like_the_markdown_chunker():
    md = "# Title\nintro\n\n## A\nalpha\n\n## B\nbeta\n"
    chunks = chunk(md, max_chars=1000)
    assert [c.text[:4] for c in chunks] == ["# Ti", "## A", "## B"]


def test_a_long_section_is_split_and_no_chunk_exceeds_the_cap():
    md = "# Big\n" + ("paragraph body line\n\n" * 60) + "## Small\ntiny\n"
    chunks = chunk(md, max_chars=200, overlap_chars=40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)  # the cap holds, always


def test_a_split_window_never_crosses_a_heading():
    # The tiny trailing section must remain its own chunk, not get swept into a
    # window of the big section above it.
    md = "# Big\n" + ("body line\n\n" * 60) + "## Small\ntiny\n"
    chunks = chunk(md, max_chars=200, overlap_chars=40)
    assert any(c.text.startswith("## Small") for c in chunks)
    assert not any("## Small" in c.text and not c.text.startswith("## Small") for c in chunks)


def test_zero_overlap_reassembles_losslessly():
    md = "# H\n" + ("line\n\n" * 80)
    chunks = chunk(md, max_chars=150, overlap_chars=0)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == md  # contiguous, non-overlapping


def test_overlap_repeats_a_tail_into_the_next_window():
    md = "# H\n" + ("abcdefghij\n" * 40)  # one long section, no paragraph breaks
    chunks = chunk(md, max_chars=120, overlap_chars=30)
    # Consecutive windows share their boundary region (overlap), so the total
    # emitted text is longer than the source.
    assert sum(len(c.text) for c in chunks) > len(md)


def test_document_without_headings_is_windowed_as_one_section():
    md = "no headings here\n\n" * 40
    chunks = chunk(md, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_satisfies_the_chunker_contract():
    md = "# H1\n" + "text\n\n" * 200 + "## H2\n" + "more\n\n" * 5
    assert_chunker_contract(MarkdownFixedChunker(max_chars=300, overlap_chars=50), doc(md))
