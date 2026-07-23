"""Representation strategies: pure projection, no store, no I/O. Hermetic."""
import pytest

from rag_blocks.core.contracts import Chunk, SparseVector
from rag_blocks.core.errors import ConfigError
from rag_blocks.embedding.hashing import HashingEmbedder
from rag_blocks.indexing.representation import (
    DenseRepresentation,
    LexicalRepresentation,
    SparseRepresentation,
)
from rag_blocks.storage.bm25_index import BM25Index


def chunks():
    return [
        Chunk(id="d:0", doc_id="d", text="the quick brown fox", index=0,
              char_start=0, char_end=1, page_start=1, page_end=1),
        Chunk(id="d:1", doc_id="d", text="financial third quarter", index=1,
              char_start=1, char_end=2, page_start=1, page_end=1),
    ]


# -- construction: registrable but needs its live encoder ------------------

def test_dense_needs_an_embedder():
    with pytest.raises(ConfigError):
        DenseRepresentation()  # by-name/no-encoder must fail fast


def test_lexical_needs_an_index():
    with pytest.raises(ConfigError):
        LexicalRepresentation()


def test_space_defaults_to_name_and_is_overridable():
    assert DenseRepresentation(HashingEmbedder()).space == "dense"
    assert DenseRepresentation(HashingEmbedder(), space="bge").space == "bge"


# -- vector-backed family: declares schema, encodes, no I/O ----------------

def test_dense_declares_a_dense_spec_matching_its_embedder():
    emb = HashingEmbedder()
    rep = DenseRepresentation(emb, space="bge")
    specs = list(rep.declare_schema())
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "bge" and spec.kind == "dense"
    assert spec.dimensions == emb.dimensions


def test_dense_query_and_corpus_use_the_same_encoder():
    # Parity (Invariant 2): a chunk's stored vector equals the same text encoded
    # as a query, because both go through the one embedder.
    emb = HashingEmbedder()
    rep = DenseRepresentation(emb, space="dense")
    text = "the quick brown fox"
    corpus_vec = rep.encode_corpus([chunks()[0]])["dense"][0]
    query_vec = rep.encode_query(text)["dense"]
    assert query_vec == emb.embed_query(text)
    assert corpus_vec == emb.embed_texts([text])[0]


def test_dense_is_not_self_managed():
    # A vector-backed rep never owns search — the Corpus does.
    rep = DenseRepresentation(HashingEmbedder())
    with pytest.raises(NotImplementedError):
        rep.search("x", k=1)


# -- self-managed family: no vector space, owns its backend ----------------

def test_lexical_declares_no_vector_space():
    assert list(LexicalRepresentation(BM25Index()).declare_schema()) == []


def test_lexical_ingest_then_search_uses_its_own_index():
    rep = LexicalRepresentation(BM25Index())
    rep.ingest(chunks())
    hits = rep.search("quick brown fox", k=1)
    assert hits and hits[0].chunk.id == "d:0"


def test_lexical_encode_corpus_is_empty_no_io():
    # It contributes through ingest, never through the vector bundle.
    assert LexicalRepresentation(BM25Index()).encode_corpus(chunks()) == {}


# -- sparse family (uses a tiny hand-rolled encoder) -----------------------

class _ToySparseEncoder:
    """A minimal SparseEncoder-shaped stub: one term per distinct word."""
    kind = "sparse_encoder"
    name = "toy"
    version = "0.1.0"

    def __init__(self):
        self.config = None

    def _vec(self, text):
        terms = sorted({hash(w) % 1000 for w in text.split()})
        return SparseVector(indices=tuple(terms), values=tuple(1.0 for _ in terms))

    def encode_texts(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)

    def fingerprint(self):
        return "toy-sparse"


def test_sparse_declares_a_sparse_spec_and_encodes():
    rep = SparseRepresentation(_ToySparseEncoder(), space="splade")
    specs = list(rep.declare_schema())
    assert len(specs) == 1 and specs[0].kind == "sparse" and specs[0].name == "splade"
    encoded = rep.encode_corpus(chunks())["splade"]
    assert len(encoded) == 2 and all(isinstance(v, SparseVector) for v in encoded)
