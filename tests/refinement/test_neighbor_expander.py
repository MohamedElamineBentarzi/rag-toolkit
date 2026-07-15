"""NeighborExpander: small-to-big expansion, overlap-safe by char offsets."""
import pytest

from rag_toolkit.core.contracts import Chunk, Query, ScoredChunk
from rag_toolkit.core.errors import ConfigError
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.indexing.chunk_index import ChunkIndex
from rag_toolkit.refinement.neighbor import NeighborExpander
from rag_toolkit.storage.memory_store import MemoryVectorStore


def chunk(i, text, cs, ce, doc="d", **meta):
    return Chunk(id=f"{doc}:{i}", doc_id=doc, text=text, index=i,
                 char_start=cs, char_end=ce, page_start=1, page_end=1,
                 metadata=meta)


def index_with(chunks):
    index = ChunkIndex(MemoryVectorStore(), dense=HashingEmbedder())
    index.add(chunks)
    return index


def anchor(c, score=0.9):
    return ScoredChunk(chunk=c, score=score, retriever_name="index")


def test_expands_adjacent_neighbors_into_one_passage():
    c0 = chunk(0, "Hello world.", 0, 12)
    c1 = chunk(1, " More text.", 12, 23)
    c2 = chunk(2, " Even more.", 23, 34)
    index = index_with([c0, c1, c2])
    out = NeighborExpander(index, window=1).refine(
        Query(text="q"), [anchor(c1)], k=5
    )
    assert out[0].chunk.text == "Hello world. More text. Even more."
    assert out[0].chunk.char_start == 0
    assert out[0].chunk.char_end == 34
    assert out[0].chunk.metadata["expanded"] is True
    # Score and doc identity ride through unchanged.
    assert out[0].score == 0.9


def test_overlapping_chunks_are_stitched_without_duplication():
    c0 = chunk(0, "Hello world", 0, 11)
    c1 = chunk(1, "world today", 6, 17)   # [6,11) == "world" overlaps c0
    index = index_with([c0, c1])
    out = NeighborExpander(index, window=1).refine(
        Query(text="q"), [anchor(c1)], k=5
    )
    assert out[0].chunk.text == "Hello world today"


def test_non_contiguous_neighbors_join_with_a_space():
    c0 = chunk(0, "First.", 0, 6)
    c1 = chunk(1, "Second.", 10, 17)      # gap [6,10) not covered by either
    index = index_with([c0, c1])
    out = NeighborExpander(index, window=1).refine(
        Query(text="q"), [anchor(c0)], k=5
    )
    assert out[0].chunk.text == "First. Second."


def test_synthetic_neighbors_are_excluded():
    c0 = chunk(0, "Real zero.", 0, 10)
    c1 = chunk(1, "Real one.", 10, 19)
    syn = chunk(2, "SYNTHETIC SUMMARY", 10, 19, synthetic=True)
    index = index_with([c0, c1, syn])
    out = NeighborExpander(index, window=2).refine(
        Query(text="q"), [anchor(c0)], k=5
    )
    assert "SYNTHETIC" not in out[0].chunk.text


def test_window_zero_is_a_no_op():
    c0 = chunk(0, "unchanged", 0, 9)
    index = index_with([c0])
    out = NeighborExpander(index, window=0).refine(
        Query(text="q"), [anchor(c0)], k=5
    )
    assert out[0].chunk.text == "unchanged"
    assert "expanded" not in out[0].chunk.metadata


def test_requires_an_index():
    with pytest.raises(ConfigError):
        NeighborExpander()
