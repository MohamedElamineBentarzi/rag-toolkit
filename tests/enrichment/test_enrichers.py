"""HeadingEnricher (fully hermetic). The empty chain is the null enricher."""
from rag_blocks.chunking.fixed import FixedChunker
from rag_blocks.core.contracts import Source
from rag_blocks.core.registry import registry
from rag_blocks.enrichment.heading import HeadingEnricher
from rag_blocks.ingestion.parsers.plaintext import PlainTextParser
from tests.contract_checks import assert_enricher_contract

# Long sections so a small fixed chunker cuts *within* a section — those chunks
# don't start with their heading, which is exactly what the enricher fixes.
_MD = (
    "# France\n" + "Paris is the capital of France. " * 20
    + "\n\n## Cities\n" + "Lyon is a large city. " * 20
)


def doc_and_chunks():
    src = Source.from_bytes(_MD.encode(), name="d.md")
    doc = PlainTextParser(page_chars=10_000_000).parse(src)
    chunks = list(FixedChunker(chunk_chars=120, overlap_chars=0).chunk(doc))
    return doc, chunks


def test_heading_satisfies_the_enricher_contract():
    doc, chunks = doc_and_chunks()
    assert_enricher_contract(HeadingEnricher(), doc, chunks)


def test_heading_prepends_section_context_preserving_provenance():
    doc, chunks = doc_and_chunks()
    out = list(HeadingEnricher().enrich(iter(chunks), doc))
    changed = [(o, c) for o, c in zip(out, chunks) if o.text != c.text]
    assert changed, "expected some mid-section chunk to gain a heading prefix"
    enriched, original = changed[0]
    assert enriched.text.startswith("#")            # a heading was prepended
    assert enriched.text.endswith(original.text)    # original text kept intact
    assert enriched.char_start == original.char_start  # provenance preserved
    assert enriched.page_start == original.page_start


def test_heading_does_not_double_prepend():
    doc, chunks = doc_and_chunks()
    # The first chunk already starts with "# France" → left untouched.
    first = chunks[0]
    (out,) = list(HeadingEnricher().enrich(iter([first]), doc))
    assert out.text == first.text


def test_registered_under_enricher():
    assert isinstance(registry.create("enricher", "heading"), HeadingEnricher)
