"""QdrantVectorStore against a real Qdrant. Opt-in.

Uses qdrant-client's in-process `:memory:` mode by default, so it needs only
the dependency (no server):

    pip install 'rag-blocks[qdrant]'
    pytest -m integration tests/integration/test_qdrant_store.py

Point at a real server with QDRANT_URL to exercise the network path.
"""
import os

import pytest

from rag_blocks.core.contracts import VectorSpec
from rag_blocks.core.errors import ConfigError
from rag_blocks.storage.qdrant_store import QdrantVectorStore
from tests.contract_checks import assert_vector_store_contract

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    pytest.importorskip("qdrant_client")
    url = os.environ.get("QDRANT_URL")
    collection = "rag_blocks_contract"
    if url:
        return QdrantVectorStore(url=url, collection=collection)
    return QdrantVectorStore(location=":memory:", collection=collection)


def test_qdrant_satisfies_the_vector_store_contract(store):
    assert_vector_store_contract(store)


def _attached(collection, client, models, **kw):
    """A QdrantVectorStore sharing one in-process client (so two instances see
    the same collection — separate `:memory:` clients would not)."""
    store = QdrantVectorStore(collection=collection, **kw)
    store._client, store._models = client, models
    return store


def test_existing_collection_with_mismatched_schema_raises_clearly():
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient, models
    client = QdrantClient(location=":memory:")
    coll = "rag_blocks_mismatch"

    _attached(coll, client, models).ensure_schema(
        [VectorSpec("dense", "dense", dimensions=8)]
    )

    b = _attached(coll, client, models)
    with pytest.raises(ConfigError) as exc:
        b.ensure_schema([VectorSpec("dense", "dense", dimensions=16)])
    # The message names what's actually there and how to proceed.
    assert "recreate_on_mismatch" in str(exc.value)


def test_recreate_on_mismatch_drops_and_rebuilds():
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient, models
    client = QdrantClient(location=":memory:")
    coll = "rag_blocks_recreate"

    _attached(coll, client, models).ensure_schema(
        [VectorSpec("dense", "dense", dimensions=8)]
    )

    # Opt-in recreate: a conflicting schema is dropped and rebuilt, no raise.
    _attached(coll, client, models, recreate_on_mismatch=True).ensure_schema(
        [VectorSpec("dense", "dense", dimensions=16)]
    )
    assert client.get_collection(coll).config.params.vectors["dense"].size == 16
