"""LocalBlobStore: on-disk truth store, fully hermetic."""
import pytest

from rag_blocks.core.errors import StorageError
from rag_blocks.core.registry import registry
from rag_blocks.storage.local import LocalBlobStore
from tests.contract_checks import assert_blob_store_contract


def make_store(tmp_path):
    return LocalBlobStore(root=str(tmp_path / "blobs"))


def test_satisfies_the_blob_store_contract(tmp_path):
    assert_blob_store_contract(make_store(tmp_path))


def test_nested_key_creates_directories_and_round_trips(tmp_path):
    store = make_store(tmp_path)
    store.put("raw/abc123/original.pdf", b"%PDF-1.7 bytes")
    assert store.get("raw/abc123/original.pdf") == b"%PDF-1.7 bytes"
    # The logical path materialized as real nested directories on disk.
    assert (tmp_path / "blobs" / "raw" / "abc123" / "original.pdf").is_file()


def test_get_missing_key_raises_storage_error_with_key(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(StorageError) as excinfo:
        store.get("raw/nope/original.pdf")
    assert excinfo.value.key == "raw/nope/original.pdf"


def test_put_overwrites_atomically_without_leaving_temp_files(tmp_path):
    store = make_store(tmp_path)
    store.put("k/v", b"first")
    store.put("k/v", b"second")
    assert store.get("k/v") == b"second"
    leftovers = [p.name for p in (tmp_path / "blobs" / "k").iterdir()]
    assert leftovers == ["v"]  # no ".v.<pid>.tmp" turds


@pytest.mark.parametrize("bad_key", ["../escape", "a/../../escape", "", "   "])
def test_keys_escaping_the_root_are_refused(tmp_path, bad_key):
    store = make_store(tmp_path)
    with pytest.raises(StorageError):
        store.put(bad_key, b"x")


def test_registered_under_blob_store_local(tmp_path):
    built = registry.create("blob_store", "local", root=str(tmp_path))
    assert isinstance(built, LocalBlobStore)
