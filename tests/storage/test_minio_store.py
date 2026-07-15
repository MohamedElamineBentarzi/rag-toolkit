"""MinioBlobStore: the parts of the adapter that are OUR logic, not MinIO's.

No server, no `minio` package needed here. We test the seams the design put in
place: credential resolution, secret redaction, lazy client construction, and
registry wiring. The full put/get/exists round-trip against a real backend is
the env-gated integration test.
"""
from rag_blocks.core.registry import registry
from rag_blocks.storage.minio_store import MinioBlobStore


def test_construction_is_lazy_no_client_built(monkeypatch):
    # Even with a bogus endpoint, constructing must not touch the network.
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    store = MinioBlobStore(endpoint="unreachable:9000")
    assert store._client is None  # nothing built until first op


def test_credentials_prefer_explicit_config_over_env(monkeypatch):
    monkeypatch.setenv("MINIO_ACCESS_KEY", "env-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "env-secret")
    store = MinioBlobStore(access_key="cfg-access", secret_key="cfg-secret")
    assert store._credentials() == ("cfg-access", "cfg-secret")


def test_credentials_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("MINIO_ACCESS_KEY", "env-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "env-secret")
    store = MinioBlobStore()  # no explicit keys
    assert store._credentials() == ("env-access", "env-secret")


def test_credentials_are_redacted_in_describe():
    store = MinioBlobStore(access_key="AKIA-secret", secret_key="shhh")
    cfg = store.describe()["config"]
    assert cfg["access_key"] == "<redacted>"
    assert cfg["secret_key"] == "<redacted>"
    # Non-secret config still travels for reproducibility.
    assert cfg["endpoint"] == "localhost:9000"
    assert cfg["bucket"] == "rag-blocks"


def test_key_rotation_does_not_change_fingerprint():
    """A cache key must survive credential rotation (AGENTS.md §7.4): the
    fingerprint hashes the *redacted* describe(), so different keys → same id."""
    a = MinioBlobStore(access_key="key-one", secret_key="secret-one")
    b = MinioBlobStore(access_key="key-two", secret_key="secret-two")
    assert a.fingerprint() == b.fingerprint()
    # ...but a non-secret difference (different bucket) DOES change it.
    c = MinioBlobStore(access_key="key-one", secret_key="secret-one",
                       bucket="other")
    assert c.fingerprint() != a.fingerprint()


def test_registered_under_blob_store_minio():
    built = registry.create("blob_store", "minio", endpoint="host:9000")
    assert isinstance(built, MinioBlobStore)
