"""DocumentCatalog: doc_id → source provenance + download link (hermetic)."""
import pytest

from rag_toolkit.core.contracts import Source
from rag_toolkit.core.errors import ConfigError, StorageError
from rag_toolkit.indexing.catalog import manifest_key, raw_key
from rag_toolkit.pipeline import RagPipeline
from rag_toolkit.storage.local import LocalBlobStore

_CORPUS = b"# France\nParis is the capital of France.\n\n# Fruit\nBananas are yellow.\n"


def _rag(tmp_path):
    rag = RagPipeline(blob_store=LocalBlobStore(root=str(tmp_path)))
    rag.index(Source.from_bytes(_CORPUS, name="facts_report.md"))
    return rag


def _doc_id(rag):
    # Every chunk shares the doc_id prefix; grab it from a search hit.
    hit = rag.chunk_index.search("dense", "capital", k=1)[0]
    return hit.chunk.doc_id


# -- resolution ------------------------------------------------------------

def test_resolves_doc_id_to_source_uri(tmp_path):
    rag = _rag(tmp_path)
    doc_id = _doc_id(rag)
    assert rag.source_uri(doc_id) == "facts_report.md"


def test_get_returns_full_provenance(tmp_path):
    rag = _rag(tmp_path)
    ref = rag.catalog.get(_doc_id(rag))
    assert ref.source_uri == "facts_report.md"
    assert len(ref.content_hash) == 64          # the full sha256
    assert ref.doc_id == ref.content_hash       # doc_id IS the full content hash
    assert ref.ext == ".md"


def test_download_url_points_at_the_stored_original(tmp_path):
    rag = _rag(tmp_path)
    ref = rag.catalog.get(_doc_id(rag))
    url = rag.download_url(_doc_id(rag))
    # A file URI addressing the raw blob under the FULL content hash.
    assert url.startswith("file://")
    assert ref.content_hash in url
    assert url.endswith("original.md")


def test_unknown_doc_id_resolves_to_none(tmp_path):
    rag = _rag(tmp_path)
    assert rag.source_uri("deadbeefdeadbeef") is None
    assert rag.download_url("deadbeefdeadbeef") is None


# -- guardrails ------------------------------------------------------------

def test_without_blob_store_there_is_no_catalog():
    rag = RagPipeline()                          # default: no durable store
    assert rag.catalog is None
    with pytest.raises(ConfigError):
        rag.source_uri("anything")


def test_manifest_and_raw_keys_are_stable():
    assert manifest_key("abc123") == "docs/abc123.json"
    assert raw_key("ffff", ".pdf") == "raw/ffff/original.pdf"


def test_reingesting_the_same_doc_does_not_rewrite_the_manifest(tmp_path):
    from rag_toolkit.core.contracts import Document, PageSpan
    from rag_toolkit.indexing.catalog import DocumentCatalog

    catalog = DocumentCatalog(LocalBlobStore(root=str(tmp_path)))
    doc = Document(id="abc123", markdown="hi", pages=[PageSpan(1, 0, 2)],
                   source_uri="report.pdf")
    assert catalog.record(doc, "f" * 64, ".pdf") is True    # first ingest writes
    assert catalog.record(doc, "f" * 64, ".pdf") is False   # re-ingest is a no-op
    # First-filename-wins unless explicitly overwritten.
    renamed = Document(id="abc123", markdown="hi", pages=[PageSpan(1, 0, 2)],
                       source_uri="renamed.pdf")
    assert catalog.record(renamed, "f" * 64, ".pdf") is False
    assert catalog.get("abc123").source_uri == "report.pdf"
    assert catalog.record(renamed, "f" * 64, ".pdf", overwrite=True) is True
    assert catalog.get("abc123").source_uri == "renamed.pdf"


# -- BlobStore.url capability ---------------------------------------------

def test_local_blob_store_url_is_a_file_uri(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    store.put("raw/x/original.pdf", b"%PDF-1.4 ...")
    assert store.url("raw/x/original.pdf").startswith("file://")


def test_local_blob_store_url_missing_key_raises(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    with pytest.raises(StorageError):
        store.url("nope/missing.bin")
