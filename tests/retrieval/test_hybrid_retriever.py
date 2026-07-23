"""HybridRetriever: sugar that fuses a Corpus's representations."""
import pytest

from rag_blocks.core.contracts import Chunk, Query
from rag_blocks.core.errors import ConfigError
from rag_blocks.embedding.hashing import HashingEmbedder
from rag_blocks.indexing.corpus import Corpus
from rag_blocks.indexing.representation import (
    DenseRepresentation,
    LexicalRepresentation,
)
from rag_blocks.retrieval.hybrid import HybridRetriever
from rag_blocks.storage.bm25_index import BM25Index
from rag_blocks.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_retriever_contract


def chunk(i, text="x"):
    return Chunk(id=f"d:{i}", doc_id="d", text=text, index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


def _index():
    texts = ["quick brown fox", "financial revenue report", "quick financial notes"]
    corpus = Corpus(MemoryVectorStore(), [
        DenseRepresentation(HashingEmbedder(dimensions=512)),
        LexicalRepresentation(BM25Index()),
    ])
    corpus.add([chunk(i, t) for i, t in enumerate(texts)])
    return corpus


def test_satisfies_the_retriever_contract():
    # "quick financial" matches d:2 in both modalities → fused winner.
    assert_retriever_contract(
        HybridRetriever(_index()), Query(text="quick financial"),
        expected_top_id="d:2",
    )


def test_defaults_to_all_representations():
    hybrid = HybridRetriever(_index())
    assert set(hybrid.representations) == {"dense", "lexical"}


def test_hits_stamped_hybrid_but_sources_survive():
    top = HybridRetriever(_index()).retrieve(Query(text="quick financial"), k=3)
    assert all(t.retriever_name == "hybrid" for t in top)
    # Per-representation attribution is preserved through the sugar (the fusion
    # source labels fold in each IndexRetriever's representation).
    assert any("index:dense" in t.metadata.get("sources", {}) for t in top)


def test_can_restrict_to_a_subset_of_representations():
    hybrid = HybridRetriever(_index(), representations=["lexical"])
    top = hybrid.retrieve(Query(text="quick brown fox"), k=1)
    assert top[0].chunk.id == "d:0"


def test_requires_an_index():
    with pytest.raises(ConfigError):
        HybridRetriever()
