"""QdrantVectorStore against a real Qdrant. Opt-in.

Uses qdrant-client's in-process `:memory:` mode by default, so it needs only
the dependency (no server):

    pip install 'rag-toolkit[qdrant]'
    pytest -m integration tests/integration/test_qdrant_store.py

Point at a real server with QDRANT_URL to exercise the network path.
"""
import os

import pytest

from rag_toolkit.storage.qdrant_store import QdrantVectorStore
from tests.contract_checks import assert_vector_store_contract

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    pytest.importorskip("qdrant_client")
    url = os.environ.get("QDRANT_URL")
    collection = "rag_toolkit_contract"
    if url:
        return QdrantVectorStore(url=url, collection=collection)
    return QdrantVectorStore(location=":memory:", collection=collection)


def test_qdrant_satisfies_the_vector_store_contract(store):
    assert_vector_store_contract(store)
