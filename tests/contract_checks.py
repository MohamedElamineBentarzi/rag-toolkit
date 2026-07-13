"""Reusable invariants every stage implementation must satisfy.

ABCs enforce "the method exists"; mypy enforces "the signature matches";
THIS enforces "the behavior holds". Any new implementation (yours or a
plugin's) calls the matching `assert_<stage>_contract` in its tests and
inherits every guarantee the rest of the pipeline relies on.
"""
from __future__ import annotations

import uuid

import pytest

from rag_toolkit.chunking.base import Chunker
from rag_toolkit.core.contracts import Chunk, Document, ScoredChunk, Source
from rag_toolkit.core.errors import StorageError
from rag_toolkit.embedding.base import Embedder
from rag_toolkit.ingestion.parsers.base import Parser
from rag_toolkit.storage.base import BlobStore
from rag_toolkit.storage.vector_store import VectorStore


def assert_parser_contract(parser: Parser, source: Source) -> None:
    # 1. Streaming API yields ordered, 1-based, markdown pages.
    pages = list(parser.iter_pages(source))
    assert pages, "parser yielded no pages for a non-empty source"
    numbers = [p.number for p in pages]
    assert numbers == sorted(numbers), "pages must arrive in reading order"
    assert numbers[0] >= 1, "page numbers are 1-based"
    assert all(isinstance(p.markdown, str) for p in pages)

    # 2. parse() assembles a Document whose provenance spans are sane:
    #    ordered, non-overlapping, within bounds, and counted correctly.
    doc = parser.parse(source)
    assert doc.metadata["page_count"] == len(doc.pages)
    cursor = 0
    for span in doc.pages:
        assert span.start >= cursor, "spans must not overlap"
        assert span.end >= span.start
        cursor = span.end
    assert cursor <= len(doc.markdown)

    # 3. Identity is deterministic — the eval cache depends on it.
    assert parser.fingerprint() == parser.fingerprint()


def assert_blob_store_contract(store: BlobStore) -> None:
    """Every BlobStore (disk, S3, your own) must behave like this.

    Uses a fresh random key so the check is safe to run repeatedly against a
    real, shared backend (the contract has no delete, by design)."""
    key = f"contract-checks/{uuid.uuid4().hex}/original.bin"
    # Binary-safe payloads: full byte range incl. NUL, plus nested-path key.
    payload = bytes(range(256)) + b"\n rag-toolkit blob \x00 body"
    payload2 = b"overwritten \xff\x00 value"

    # 1. Absent key: exists() is a quiet False; get() raises with the key.
    assert store.exists(key) is False
    with pytest.raises(StorageError):
        store.get(key)

    # 2. Round-trip is byte-exact, and exists() flips to True.
    store.put(key, payload)
    assert store.exists(key) is True
    assert store.get(key) == payload

    # 3. put() overwrites in place (idempotent for content-addressed keys).
    store.put(key, payload2)
    assert store.get(key) == payload2

    # 4. Distinct keys are independent.
    other = f"contract-checks/{uuid.uuid4().hex}.bin"
    assert store.exists(other) is False

    # 5. Identity is deterministic.
    assert store.fingerprint() == store.fingerprint()


def assert_chunker_contract(chunker: Chunker, document: Document) -> None:
    """Every Chunker (fixed, markdown-aware, your own) must behave like this.

    Note what is deliberately NOT asserted: non-overlap. Overlapping spans are
    legal — that is how overlap strategies express themselves — so we check
    ordering of starts, not disjointness.
    """
    chunks = list(chunker.chunk(document))
    if document.markdown.strip():
        assert chunks, "chunker yielded no chunks for a non-empty document"

    # 1. Index is contiguous, 0-based, no holes (neighbor expansion needs it).
    assert [c.index for c in chunks] == list(range(len(chunks)))

    starts = []
    for c in chunks:
        # 2. Deterministic identity keyed on the document.
        assert c.id == f"{document.id}:{c.index}"
        assert c.doc_id == document.id

        # 3. Char offsets are present, in bounds, and slice back to the text
        #    exactly (coordinates, never a mutated copy).
        assert c.char_start is not None and c.char_end is not None
        assert 0 <= c.char_start < c.char_end <= len(document.markdown)
        assert document.markdown[c.char_start:c.char_end] == c.text

        # 4. Page provenance is ALWAYS filled for a doc-derived chunk.
        assert c.page_start is not None and c.page_end is not None
        assert c.page_start <= c.page_end
        starts.append(c.char_start)

    # 5. Spans arrive in reading order of start (overlaps permitted).
    assert starts == sorted(starts)

    # 6. Determinism — the eval cache depends on it.
    again = [c.id for c in chunker.chunk(document)]
    assert again == [c.id for c in chunks]
    assert chunker.fingerprint() == chunker.fingerprint()


def assert_embedder_contract(embedder: Embedder) -> None:
    """Every Embedder (hashing, sentence-transformers, your own) behaves so."""
    dim = embedder.dimensions
    assert isinstance(dim, int) and dim > 0

    texts = ["alpha beta gamma", "gamma delta", ""]
    vectors = embedder.embed_texts(texts)
    # 1. One vector per input, order preserved, each the declared width.
    assert len(vectors) == len(texts)
    for v in vectors:
        assert len(v) == dim
        assert all(isinstance(x, float) for x in v)

    # 2. Empty batch ⇒ empty list (not an error).
    assert embedder.embed_texts([]) == []

    # 3. A query embeds to the same space (same width).
    q = embedder.embed_query("alpha beta")
    assert len(q) == dim
    assert all(isinstance(x, float) for x in q)

    # 4. Determinism — the eval cache depends on it.
    assert embedder.embed_texts(texts) == vectors
    assert embedder.embed_query("alpha beta") == q
    assert embedder.fingerprint() == embedder.fingerprint()


def assert_vector_store_contract(store: VectorStore, dimensions: int = 8) -> None:
    """Every VectorStore (memory, qdrant, your own) must behave like this."""
    def unit(i: int) -> list[float]:
        v = [0.0] * dimensions
        v[i] = 1.0
        return v

    chunks = [
        Chunk(id=f"d:{i}", doc_id="d", text=f"chunk {i}", index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i in range(3)
    ]
    vectors = [unit(0), unit(1), unit(2)]

    # 1. Empty store: search yields nothing (not an error).
    assert store.search(unit(0), k=5) == []

    # 2. After upsert, the exact match ranks first, scores descend, and the
    #    reconstructed chunk keeps its provenance.
    store.upsert(chunks, vectors)
    results = store.search(unit(0), k=3)
    assert results and isinstance(results[0], ScoredChunk)
    assert results[0].chunk.id == "d:0"
    assert [r.score for r in results] == sorted(
        (r.score for r in results), reverse=True
    )
    assert results[0].chunk.doc_id == "d"
    assert results[0].chunk.page_start == 1
    assert results[0].chunk.char_start == 0

    # 3. k is respected.
    assert len(store.search(unit(0), k=2)) == 2

    # 4. Re-upserting the same ids overwrites, never duplicates.
    store.upsert(chunks, vectors)
    assert len(store.search(unit(0), k=10)) == 3

    # 5. Payload-equality filters narrow the result set.
    filtered = store.search(unit(1), k=10, filters={"index": 1})
    assert filtered and all(r.chunk.index == 1 for r in filtered)

    # 6. Identity is deterministic.
    assert store.fingerprint() == store.fingerprint()
