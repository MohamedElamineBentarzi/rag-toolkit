"""ExtractiveGenerator + context packing (fully hermetic)."""
from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.registry import registry
from rag_toolkit.generation.extractive import ExtractiveGenerator
from rag_toolkit.generation.packing import pack_context, resolve_citations
from tests.contract_checks import assert_generator_contract


def context(*texts):
    return [
        ScoredChunk(
            chunk=Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
                        char_start=i, char_end=i + 1, page_start=i + 1,
                        page_end=i + 1),
            score=1.0 - i * 0.1, retriever_name="dense",
        )
        for i, t in enumerate(texts)
    ]


def test_satisfies_the_generator_contract():
    assert_generator_contract(
        ExtractiveGenerator(), Query(text="q"), context("alpha", "beta")
    )


def test_returns_top_passage_with_its_citation():
    answer = ExtractiveGenerator().generate(
        Query(text="q"), context("the answer is 42", "irrelevant noise")
    )
    assert "the answer is 42" in answer.text
    assert "[1]" in answer.text
    assert answer.citations[0].chunk_id == "d:0"
    assert answer.citations[0].page_start == 1  # provenance carried through


def test_empty_context_answers_without_citations():
    answer = ExtractiveGenerator().generate(Query(text="q"), [])
    assert answer.text  # a graceful "I don't know", not empty
    assert answer.citations == []


def test_registered_under_generator_extractive():
    assert isinstance(
        registry.create("generator", "extractive"), ExtractiveGenerator
    )


# -- the shared packing helper ------------------------------------------------

def test_pack_context_numbers_chunks_and_respects_budget():
    packed = pack_context(context("a" * 50, "b" * 50, "c" * 50), max_chars=60)
    # First block alone fits; the second would exceed 60 chars → stop.
    assert len(packed.citations) == 1
    assert packed.prompt_block.startswith("[1] ")


def test_resolve_citations_keeps_only_cited_markers():
    packed = pack_context(context("x", "y", "z"), max_chars=1000)
    resolved = resolve_citations("As shown in [2].", packed.citations)
    assert [c.marker for c in resolved] == [2]
    # No markers at all ⇒ fall back to all offered (don't drop provenance).
    assert resolve_citations("no markers here", packed.citations) == packed.citations
