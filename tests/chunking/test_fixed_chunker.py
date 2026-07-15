"""FixedChunker: fixed-size windows with overlap and boundary-aware cuts."""
from rag_blocks.chunking.fixed import FixedChunker
from rag_blocks.core.contracts import Document, Page, Source
from rag_blocks.ingestion.parsers.plaintext import PlainTextParser
from tests.contract_checks import assert_chunker_contract


def doc(markdown):
    """A single-page Document whose .markdown is exactly `markdown`."""
    src = Source.from_bytes(markdown.encode(), name="t.md")
    return PlainTextParser(page_chars=10_000_000).parse(src)


def test_small_document_is_a_single_chunk():
    chunks = list(FixedChunker(chunk_chars=1600).chunk(doc("just a little text")))
    assert len(chunks) == 1
    assert chunks[0].text == "just a little text"
    assert chunks[0].index == 0
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 18)


def test_windows_cover_everything_and_overlap():
    text = "".join(f"sentence {i}. " for i in range(1000))  # long, no newlines
    d = doc(text)
    chunks = list(FixedChunker(chunk_chars=400, overlap_chars=50).chunk(d))
    assert len(chunks) > 1

    covered = bytearray(len(d.markdown))
    for c in chunks:
        for i in range(c.char_start, c.char_end):
            covered[i] = 1
    assert all(covered), "every character must belong to at least one chunk"
    # Consecutive windows share an overlap region (hard cuts, no newlines).
    assert chunks[1].char_start < chunks[0].char_end


def test_prefers_paragraph_boundaries():
    text = "\n\n".join(["x" * 100] * 10)
    chunks = list(FixedChunker(chunk_chars=250, overlap_chars=0).chunk(doc(text)))
    # A soft cut lands right after a "\n\n" break, so a chunk ends with it.
    assert any(c.text.endswith("\n\n") for c in chunks)


def test_cross_page_provenance_is_a_range():
    src = Source.from_bytes(b"unused", name="t.md")
    # markdown = "A"*100 + "\n\n" + "B"*100 → page1 [0,100), page2 [102,202)
    d = Document.from_pages(src, [Page(1, "A" * 100), Page(2, "B" * 100)])
    (chunk,) = list(FixedChunker(chunk_chars=1600).chunk(d))
    assert chunk.page_start == 1
    assert chunk.page_end == 2


def test_satisfies_the_chunker_contract():
    d = doc("Alpha beta gamma.\n\n" * 200)
    assert_chunker_contract(FixedChunker(chunk_chars=300, overlap_chars=40), d)
