"""HashingEmbedder: zero-dep, deterministic feature-hashing embedder."""
import math

from rag_blocks.core.registry import registry
from rag_blocks.embedding.hashing import HashingEmbedder
from tests.contract_checks import assert_embedder_contract


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_satisfies_the_embedder_contract():
    assert_embedder_contract(HashingEmbedder())


def test_dimensions_are_configurable():
    emb = HashingEmbedder(dimensions=64)
    assert emb.dimensions == 64
    assert len(emb.embed_query("hello")) == 64


def test_deterministic_across_instances():
    a = HashingEmbedder().embed_query("the quick brown fox")
    b = HashingEmbedder().embed_query("the quick brown fox")
    assert a == b  # blake2b, not salted hash()


def test_normalized_to_unit_length():
    v = HashingEmbedder().embed_query("some words here")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-9)


def test_empty_text_is_a_zero_vector():
    v = HashingEmbedder(dimensions=32).embed_query("")
    assert v == [0.0] * 32


def test_shared_tokens_are_more_similar():
    emb = HashingEmbedder(dimensions=1024)
    base, related, unrelated = emb.embed_texts([
        "the cat sat on the mat",
        "the cat sat on the warm mat",
        "quantum chromodynamics lagrangian",
    ])
    assert cosine(base, related) > cosine(base, unrelated)


def test_registered_under_embedder_hashing():
    assert isinstance(registry.create("embedder", "hashing"), HashingEmbedder)
