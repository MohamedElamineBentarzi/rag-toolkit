"""CachingEmbedder: fingerprint-transparent memoization with namespace split."""
from rag_toolkit.embedding.caching import CachingEmbedder
from rag_toolkit.embedding.hashing import HashingEmbedder
from rag_toolkit.storage.local import LocalBlobStore
from tests.contract_checks import assert_embedder_contract


class _CountingEmbedder(HashingEmbedder):
    """HashingEmbedder that records how many texts it actually embedded."""
    name = "counting-emb"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passages = 0
        self.queries = 0

    def embed_texts(self, texts):
        self.passages += len(texts)
        return super().embed_texts(texts)

    def embed_query(self, text):
        self.queries += 1
        return super().embed_query(text)


def test_satisfies_the_embedder_contract(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))
    assert_embedder_contract(CachingEmbedder(HashingEmbedder(), cache))


def test_fingerprint_is_transparent(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))
    inner = HashingEmbedder()
    wrapped = CachingEmbedder(inner, cache)
    # The wrapper changes cost, not output — it must be invisible to cache keys.
    assert wrapped.fingerprint() == inner.fingerprint()
    assert wrapped.dimensions == inner.dimensions


def test_passages_are_memoized_across_instances(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))

    first = _CountingEmbedder()
    CachingEmbedder(first, cache).embed_texts(["alpha", "beta"])
    assert first.passages == 2

    # A fresh inner of the same config shares the cache (keyed by fingerprint).
    second = _CountingEmbedder()
    CachingEmbedder(second, cache).embed_texts(["alpha", "beta"])
    assert second.passages == 0


def test_partial_cache_only_embeds_misses(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))
    counter = _CountingEmbedder()
    emb = CachingEmbedder(counter, cache)
    emb.embed_texts(["alpha"])
    assert counter.passages == 1
    emb.embed_texts(["alpha", "beta"])   # only "beta" is a miss
    assert counter.passages == 2


def test_passage_and_query_namespaces_are_separate(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))
    counter = _CountingEmbedder()
    emb = CachingEmbedder(counter, cache)
    # Same string as a passage then as a query must NOT collide in the cache.
    emb.embed_texts(["shared text"])
    assert counter.passages == 1
    emb.embed_query("shared text")
    assert counter.queries == 1          # a real query encode, not a passage hit
    # And the query is now cached too.
    emb.embed_query("shared text")
    assert counter.queries == 1


def test_result_order_is_preserved_with_mixed_hits(tmp_path):
    cache = LocalBlobStore(root=str(tmp_path))
    emb = CachingEmbedder(HashingEmbedder(), cache)
    emb.embed_texts(["b"])               # pre-cache the middle item
    out = emb.embed_texts(["a", "b", "c"])
    direct = HashingEmbedder().embed_texts(["a", "b", "c"])
    assert out == direct
