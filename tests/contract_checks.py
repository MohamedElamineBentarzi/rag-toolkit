"""Reusable invariants every stage implementation must satisfy.

ABCs enforce "the method exists"; mypy enforces "the signature matches";
THIS enforces "the behavior holds". Any new implementation (yours or a
plugin's) calls the matching `assert_<stage>_contract` in its tests and
inherits every guarantee the rest of the pipeline relies on.
"""
from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from rag_blocks.chunking.base import Chunker
from rag_blocks.core.contracts import (
    Answer,
    Chunk,
    Document,
    Query,
    ScoredChunk,
    Source,
    VectorSpec,
)
from rag_blocks.core.errors import RagBlocksError, StorageError
from rag_blocks.embedding.base import Embedder
from rag_blocks.enrichment.base import Enricher
from rag_blocks.generation.base import Generator
from rag_blocks.ingestion.parsers.base import Parser
from rag_blocks.refinement.base import Refiner
from rag_blocks.retrieval.base import Retriever
from rag_blocks.storage.base import BlobStore
from rag_blocks.storage.lexical_index import LexicalIndex
from rag_blocks.storage.vector_store import VectorStore


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
    payload = bytes(range(256)) + b"\n rag-blocks blob \x00 body"
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


def assert_enricher_contract(
    enricher: Enricher, document: Document, chunks: list[Chunk]
) -> None:
    """Every Enricher (noop, heading, contextual, your own) must behave so.

    Passed the document's own chunks; enrichers may augment text or add
    synthetic chunks, but must keep the doc link and yield real Chunks."""
    input_ids = {c.id for c in chunks}
    out = list(enricher.enrich(iter(chunks), document))
    assert out, "enricher dropped all chunks for a non-empty document"
    for c in out:
        assert isinstance(c, Chunk)
        assert c.doc_id == document.id  # the doc link survives enrichment
        assert c.text  # never blanks a chunk

        # Synthetic-chunk identity rule (§8.2): any *added* chunk (an id the
        # chunker never produced) must be a parent-derived, index-carrying,
        # explicitly-synthetic chunk — so neighbor/index lookups can exclude it.
        if c.id not in input_ids:
            assert c.metadata.get("synthetic") is True, (
                "added chunks must be marked metadata['synthetic']=True"
            )
            assert "#" in c.id, "synthetic ids must be parent-derived (parent#...)"
            assert c.index is not None, "synthetic chunks must carry the parent index"

    # Deterministic identity (the component, not necessarily its LLM output).
    assert enricher.fingerprint() == enricher.fingerprint()


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
    """Every VectorStore (memory, qdrant, your own) must behave like this.

    v2: named+typed multi-vector, eager create-or-validate schema, `fetch`
    without a query vector, membership filters, and partial `update_vectors`.
    """
    def unit(i: int) -> list[float]:
        v = [0.0] * dimensions
        v[i] = 1.0
        return v

    chunks = [
        Chunk(id=f"d:{i}", doc_id="d", text=f"chunk {i}", index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i in range(3)
    ]
    dense = [unit(0), unit(1), unit(2)]

    # 0. Schema is declared up front (create-or-validate).
    spec = VectorSpec(name="dense", kind="dense", dimensions=dimensions)
    store.ensure_schema([spec])
    # Re-declaring the same schema validates and is a no-op (not an error).
    store.ensure_schema([spec])
    # A conflicting redeclaration must fail loudly, never coerce.
    with pytest.raises(RagBlocksError):
        store.ensure_schema([VectorSpec("dense", "dense", dimensions + 1)])

    # 1. Empty store: search yields nothing (not an error).
    assert store.search("dense", unit(0), k=5) == []

    # 2. After upsert, the exact match ranks first, scores descend, and the
    #    reconstructed chunk keeps its provenance.
    store.upsert(chunks, {"dense": dense})
    results = store.search("dense", unit(0), k=3)
    assert results and isinstance(results[0], ScoredChunk)
    assert results[0].chunk.id == "d:0"
    assert [r.score for r in results] == sorted(
        (r.score for r in results), reverse=True
    )
    assert results[0].chunk.doc_id == "d"
    assert results[0].chunk.page_start == 1
    assert results[0].chunk.char_start == 0

    # 3. k is respected.
    assert len(store.search("dense", unit(0), k=2)) == 2

    # 4. Re-upserting the same ids overwrites, never duplicates.
    store.upsert(chunks, {"dense": dense})
    assert len(store.search("dense", unit(0), k=10)) == 3

    # 4b. Upsert is by id and content-replacing: re-upserting an existing id
    #     with different text overwrites the stored chunk (A1 — the vector and
    #     lexical sides must agree on this so a re-index can't desync them).
    reworded = replace(chunks[0], text="a completely different chunk body")
    store.upsert([reworded], {"dense": [unit(0)]})
    refetched = store.fetch({"doc_id": "d", "index": 0}, limit=1)
    assert refetched and refetched[0].text == "a completely different chunk body"
    store.upsert(chunks, {"dense": dense})  # restore for later steps

    # 5. Equality filters narrow the result set.
    filtered = store.search("dense", unit(1), k=10, filters={"index": 1})
    assert filtered and all(r.chunk.index == 1 for r in filtered)

    # 6. fetch() is point retrieval without a query vector; list filter values
    #    mean membership.
    got = store.fetch({"doc_id": "d", "index": [0, 2]}, limit=10)
    assert {c.id for c in got} == {"d:0", "d:2"}

    # 7. Identity is deterministic.
    assert store.fingerprint() == store.fingerprint()


def _corpus() -> list[Chunk]:
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown hare races across the field",
        "financial results for the third quarter of the year",
    ]
    return [
        Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
              char_start=i, char_end=i + 1, page_start=1, page_end=1)
        for i, t in enumerate(texts)
    ]


def assert_lexical_index_contract(index: LexicalIndex) -> None:
    """Every LexicalIndex (bm25, your own) must behave like this."""
    chunks = _corpus()

    # 1. Empty index / no-term query ⇒ no results.
    assert index.search("anything", k=5) == []

    index.add(chunks)

    # 2. Term overlap ranks a matching doc first; results are ScoredChunks
    #    with descending scores and intact provenance.
    results = index.search("quick brown fox", k=3)
    assert results and isinstance(results[0], ScoredChunk)
    assert results[0].chunk.id == "d:0"
    assert [r.score for r in results] == sorted(
        (r.score for r in results), reverse=True
    )
    assert results[0].chunk.page_start == 1

    # 3. A term absent from the corpus scores nothing.
    assert index.search("zzzznonexistent", k=5) == []

    # 4. k respected; re-adding the same ids doesn't duplicate.
    index.add(chunks)
    assert len(index.search("quick", k=10)) <= 3

    # 4b. add() is upsert, not skip-if-present: re-adding an id with different
    #     text overwrites — the new text becomes searchable and the stale text
    #     disappears (A1). "fox" lived only in d:0; after rewording it, nothing
    #     should match it, and the new term should.
    index.add([replace(chunks[0], text="parliament ratified the trade treaty")])
    assert index.search("parliament", k=5), "new text under an existing id must win"
    assert index.search("fox", k=5) == [], "stale text under a reused id must be gone"
    index.add(chunks)  # restore for later steps

    # 5. Filters narrow the set.
    filtered = index.search("quarter", k=10, filters={"index": 2})
    assert all(r.chunk.index == 2 for r in filtered)

    # 6. Deterministic identity.
    assert index.fingerprint() == index.fingerprint()


def assert_index_contract(index) -> None:
    """Every ChunkIndex must behave like this, over whatever representations it
    declares. Hermetic on memory store + HashingEmbedder + Bm25Index."""
    reps = index.representations()
    assert reps, "a ChunkIndex must declare at least one representation"

    chunks = _corpus()

    # 1. Empty index: every representation searches to nothing (not an error).
    for rep in reps:
        assert index.search(rep, "quick brown fox", k=5) == []

    index.add(chunks)

    # 2. Each representation retrieves, ranks highest-first, and reconstructs
    #    chunks with intact provenance.
    for rep in reps:
        results = index.search(rep, "quick brown fox", k=3)
        assert results and isinstance(results[0], ScoredChunk)
        assert [r.score for r in results] == sorted(
            (r.score for r in results), reverse=True
        )
        assert results[0].chunk.page_start == 1
        assert results[0].chunk.id == "d:0"  # the overlapping doc wins

    # 3. fetch() reads the vector store's payloads (list filter values mean
    #    membership). A lexical-only index has no stored vectors, so fetch is
    #    legitimately empty there; when it returns anything the membership
    #    filter must be honored exactly.
    got = index.fetch({"doc_id": "d", "index": [0, 2]}, limit=10)
    assert all(c.id in {"d:0", "d:2"} for c in got)

    # 4. add is idempotent by chunk.id (re-adding never duplicates).
    index.add(chunks)
    for rep in reps:
        assert len(index.search(rep, "quick", k=10)) <= len(chunks)

    # 5. Unknown representation fails loudly.
    with pytest.raises(RagBlocksError):
        index.search("no-such-representation", "quick", k=1)

    # 6. Deterministic identity.
    assert index.fingerprint() == index.fingerprint()


def assert_retriever_contract(
    retriever: Retriever, query: Query, expected_top_id: str
) -> None:
    """Every Retriever (dense, bm25, hybrid, your own) must behave like this.
    The retriever is assumed already wired to a POPULATED backend."""
    results = retriever.retrieve(query, k=3)
    assert results and isinstance(results[0], ScoredChunk)

    # 1. Every hit is attributed to this retriever (fusion depends on it).
    assert all(r.retriever_name == retriever.name for r in results)

    # 2. Highest score first, and the expected best result wins.
    assert [r.score for r in results] == sorted(
        (r.score for r in results), reverse=True
    )
    assert results[0].chunk.id == expected_top_id

    # 3. k is respected.
    assert len(retriever.retrieve(query, k=1)) == 1

    # 4. Deterministic identity.
    assert retriever.fingerprint() == retriever.fingerprint()


def assert_refiner_contract(refiner: Refiner) -> None:
    """Every Refiner (keyword, score-threshold, neighbor-expander, cross-encoder,
    your own) must behave like this. A refiner is one uniform post-retrieval
    stage: `refine(query, candidates, k) -> candidates`."""
    query = Query(text="quick brown fox")
    # Candidates arrive already ranked from a retriever (scores descending).
    candidates = [
        ScoredChunk(
            chunk=Chunk(id=f"d:{i}", doc_id="d", text=t, index=i,
                        char_start=i, char_end=i + 1, page_start=1, page_end=1),
            score=1.0 - i * 0.1, retriever_name="index",
        )
        for i, t in enumerate(["quick brown fox", "lazy dog", "unrelated text"])
    ]

    refined = refiner.refine(query, list(candidates), k=2)
    # 1. Output is a list of ScoredChunks (may be more or fewer than k — the
    #    pipeline enforces the final truncation, not the refiner).
    assert isinstance(refined, list)
    assert all(isinstance(r, ScoredChunk) for r in refined)
    # 2. Whatever it returns is ordered highest-score-first.
    assert [r.score for r in refined] == sorted(
        (r.score for r in refined), reverse=True
    )
    # 3. Empty candidates ⇒ empty result (not an error).
    assert refiner.refine(query, [], k=5) == []
    # 4. Deterministic identity.
    assert refiner.fingerprint() == refiner.fingerprint()


def assert_generator_contract(
    generator: Generator, query: Query, context: list[ScoredChunk]
) -> None:
    """Every Generator (extractive, LLM-backed, your own) must behave like this.
    Determinism is NOT asserted — real LLM generators aren't deterministic."""
    answer = generator.generate(query, context)
    assert isinstance(answer, Answer)
    assert isinstance(answer.text, str)

    # 1. Citations reference only the provided chunks and carry provenance.
    provided = {sc.chunk.id for sc in context}
    for c in answer.citations:
        assert c.chunk_id in provided
        assert c.marker >= 1
        assert c.doc_id  # provenance survived to the citation

    # 2. Empty context is handled gracefully — an Answer with no citations.
    empty = generator.generate(query, [])
    assert isinstance(empty, Answer)
    assert empty.citations == []

    # 3. Deterministic identity (the component, not its output).
    assert generator.fingerprint() == generator.fingerprint()
