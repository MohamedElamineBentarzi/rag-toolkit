"""Corpus: the storage owner coordinating a corpus's representations. Hermetic."""
import pytest

from rag_blocks.core.contracts import Chunk
from rag_blocks.core.errors import ConfigError
from rag_blocks.embedding.hashing import HashingEmbedder
from rag_blocks.indexing.corpus import Corpus
from rag_blocks.indexing.representation import (
    DenseRepresentation,
    LexicalRepresentation,
)
from rag_blocks.storage.bm25_index import BM25Index
from rag_blocks.storage.memory_store import MemoryVectorStore
from tests.contract_checks import assert_index_contract


def store():
    return MemoryVectorStore()


def dense(**kw):
    return DenseRepresentation(HashingEmbedder(**kw))


def lexical():
    return LexicalRepresentation(BM25Index())


def chunk(i, text, doc="d"):
    return Chunk(id=f"{doc}:{i}", doc_id=doc, text=text, index=i,
                 char_start=i, char_end=i + 1, page_start=1, page_end=1)


# -- contract, over several representation shapes --------------------------

def test_dense_only_satisfies_contract():
    assert_index_contract(Corpus(store(), [dense()]))


def test_lexical_only_satisfies_contract():
    assert_index_contract(Corpus(store(), [lexical()]))


def test_hybrid_dense_plus_lexical_satisfies_contract():
    assert_index_contract(Corpus(store(), [dense(), lexical()]))


# -- constructor: naming & fail-fast ---------------------------------------

def test_spaces_default_to_representation_names():
    corpus = Corpus(store(), [dense(), lexical()])
    assert corpus.representations() == ["dense", "lexical"]


def test_space_override_allows_ab_of_two_dense_models():
    corpus = Corpus(store(), [
        DenseRepresentation(HashingEmbedder(), space="bge"),
        DenseRepresentation(HashingEmbedder(dimensions=128), space="e5"),
    ])
    assert set(corpus.representations()) == {"bge", "e5"}


def test_no_representation_fails_fast():
    with pytest.raises(ConfigError):
        Corpus(store(), [])


def test_duplicate_space_fails_fast():
    with pytest.raises(ConfigError):
        Corpus(store(), [
            DenseRepresentation(HashingEmbedder(), space="lexical"),
            lexical(),
        ])


# -- behavior --------------------------------------------------------------

def test_search_dispatches_to_the_right_space():
    corpus = Corpus(store(), [dense(), lexical()])
    corpus.add([chunk(0, "the quick brown fox"),
                chunk(1, "financial results third quarter")])
    assert corpus.search("dense", "quick brown fox", k=1)[0].chunk.id == "d:0"
    assert corpus.search("lexical", "quick brown fox", k=1)[0].chunk.id == "d:0"


def test_single_pass_write_does_one_upsert(monkeypatch):
    # Invariant 1: N vector-backed reps must NOT cause N sequential upserts.
    s = store()
    calls = {"n": 0}
    real_upsert = s.upsert

    def counting_upsert(chunks, vectors):
        calls["n"] += 1
        return real_upsert(chunks, vectors)

    monkeypatch.setattr(s, "upsert", counting_upsert)
    corpus = Corpus(s, [
        DenseRepresentation(HashingEmbedder(), space="a"),
        DenseRepresentation(HashingEmbedder(dimensions=128), space="b"),
        lexical(),  # self-managed: writes to its own index, not the store
    ])
    corpus.add([chunk(0, "the quick brown fox"), chunk(1, "another chunk")])
    assert calls["n"] == 1, "two dense reps must share ONE store.upsert"


def test_unknown_space_lists_available():
    corpus = Corpus(store(), [dense()])
    with pytest.raises(ConfigError) as exc:
        corpus.search("splade", "x", k=1)
    assert "dense" in str(exc.value)


def test_update_representation_refreshes_one_space():
    corpus = Corpus(store(), [dense()])
    corpus.add([chunk(0, "original text about apples")])
    corpus.update_representation("dense", [chunk(0, "revised text about oranges")])
    hits = corpus.search("dense", "oranges", k=1)
    assert hits and hits[0].chunk.id == "d:0"


def test_describe_folds_in_backend_identities():
    a = Corpus(store(), [dense()])
    b = Corpus(store(), [dense(dimensions=128)])
    # Different embedder ⇒ different representation fingerprint ⇒ different id.
    assert a.fingerprint() != b.fingerprint()
    info = a.describe()
    assert "store_fingerprint" in info
    assert "dense" in info["representations"]


def test_changing_one_rep_leaves_the_other_fingerprint_stable():
    # Invariant 4: one rep changing must not perturb a sibling's identity.
    d = dense()
    lex = lexical()
    a = Corpus(store(), [d, lex])
    b = Corpus(store(), [dense(dimensions=128), lex])
    fa = a.describe()["representations"]
    fb = b.describe()["representations"]
    assert fa["dense"] != fb["dense"], "the changed rep's fingerprint moves"
    assert fa["lexical"] == fb["lexical"], "the unchanged rep's fingerprint holds"


def test_corpus_is_a_chunk_sink():
    from rag_blocks.indexing.sink import ChunkSink
    assert isinstance(Corpus(store(), [dense()]), ChunkSink)
