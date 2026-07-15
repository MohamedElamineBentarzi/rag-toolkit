"""MinioBlobStore against a REAL S3-compatible backend. Opt-in.

Run a local server, e.g.:

    docker run -p 9000:9000 -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin quay.io/minio/minio server /data

then:

    MINIO_ENDPOINT=localhost:9000 \
    MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin \
    pytest -m integration tests/integration/test_minio_store.py
"""
import os

import pytest

from rag_blocks.storage.minio_store import MinioBlobStore
from tests.contract_checks import assert_blob_store_contract

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    endpoint = os.environ.get("MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("set MINIO_ENDPOINT (+ MINIO_ACCESS_KEY/MINIO_SECRET_KEY)")
    return MinioBlobStore(
        endpoint=endpoint,
        bucket=os.environ.get("MINIO_BUCKET", "rag-blocks-tests"),
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def test_minio_satisfies_the_blob_store_contract(store):
    assert_blob_store_contract(store)
