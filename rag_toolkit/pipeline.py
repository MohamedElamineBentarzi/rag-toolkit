"""Orchestrators: thin wiring over the stage components.

The design rule the whole library is built to honor (AGENTS.md §2): **all
intelligence lives in components; all wiring lives here, and the wiring is
dumb.** An IndexingPipeline is a for-loop over generators plus two seams —
a tracing hook and an optional blob store. It is deliberately NOT a Component:
it is not a swappable algorithm, it is the glue that composes them.

`IndexingPipeline.index(sources)` runs, per source:

    Source ──parse──▶ Document ──chunk──▶ Iterator[Chunk]

and (when a blob store is wired in) captures the durable truth alongside:

    raw/{sha256}/original{ext}                     the immutable source bytes
    parsed/{sha256}/{parser_fingerprint}.md        the parse cache (markdown)
    parsed/{sha256}/{parser_fingerprint}.meta.json spans + doc metadata

Two design points worth stating, both settled earlier in design discussion:

- **Content-addressing lives here, not in the store.** The BlobStore is a dumb
  key→bytes service (it attaches no meaning to keys); the pipeline is the caller
  that knows the `raw/…` vs `parsed/…` convention. That is what keeps
  LocalBlobStore and MinioBlobStore perfectly interchangeable.
- **Capture is opt-in and idempotent.** No blob store ⇒ pure parse→chunk. With
  one, `exists()` is a cheap pre-check so re-indexing the same bytes is a no-op
  (dedup free — same content, same key).

Streaming note: chunks stream out per document (a generator), so memory stays
O(one document + its chunks), never O(corpus). Raw capture currently reads the
source into memory to `put` it (the BlobStore has no streaming `put` yet) — fine
for typical documents; a `put_stream` variant is the documented future fix for
multi-GB inputs. The parsed markdown is always small.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .chunking.base import Chunker
from .chunking.fixed import FixedChunker
from .core.contracts import (
    Answer,
    Chunk,
    Document,
    PageSpan,
    Query,
    ScoredChunk,
    Source,
    SourceFormat,
)
from .embedding.base import Embedder
from .embedding.hashing import HashingEmbedder
from .enrichment.base import Enricher
from .enrichment.noop import NoOpEnricher
from .generation.base import Generator
from .generation.extractive import ExtractiveGenerator
from .ingestion.detection import detect_format
from .ingestion.parsers.auto import AutoParser
from .ingestion.parsers.base import Parser
from .reranking.base import Reranker
from .reranking.noop import NoOpReranker
from .retrieval.base import Retriever
from .retrieval.dense import DenseRetriever
from .storage.base import BlobStore
from .storage.memory_store import MemoryVectorStore
from .storage.vector_store import VectorStore

__all__ = [
    "TraceEvent",
    "IndexingPipeline",
    "QueryPipeline",
    "RagPipeline",
]


@dataclass
class TraceEvent:
    """One observation emitted at a stage boundary.

    This is the seam the evaluation suite later hangs cost attribution on
    (latency per stage, cache hits). Keeping it a plain dataclass — not a log
    string — means a hook can aggregate it however it likes.
    """

    stage: str                 # "parse" | "store_raw" | "store_parsed" | "chunk"
    source_uri: str
    duration_ms: float
    detail: dict = field(default_factory=dict)


#: A tracing hook: called with each TraceEvent. Defaults to a no-op (Null
#: Object) so pipeline code never grows `if trace is not None` branches.
TraceHook = Callable[[TraceEvent], None]


def _noop_trace(event: TraceEvent) -> None:  # Null Object
    pass


class _EmbeddingCache:
    """Reuse vectors for text already embedded with this exact embedder.

    Keyed by sha256(text) under the embedder's *fingerprint*, so identical text
    (across documents or re-indexes) is embedded once, while swapping the
    embedder/model is a clean miss — never a stale vector. Backed by any
    `BlobStore` (local dir, MinIO); the cache knows how to (de)serialize, not
    where the bytes live.
    """

    def __init__(self, store: BlobStore, embedder_fingerprint: str) -> None:
        self._store = store
        self._prefix = f"embeddings/{embedder_fingerprint}"

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._prefix}/{digest}.json"

    def get(self, text: str) -> Optional[list[float]]:
        key = self._key(text)
        if not self._store.exists(key):
            return None
        return json.loads(self._store.get(key).decode("utf-8"))

    def put(self, text: str, vector: list[float]) -> None:
        self._store.put(self._key(text), json.dumps(vector).encode("utf-8"))


#: Canonical extension per known format — derived from the *detected* format,
#: never the (possibly lying) filename. IMAGE/UNKNOWN fall back to the uri
#: suffix since the concrete image type isn't carried in SourceFormat.
_CANONICAL_EXT = {
    SourceFormat.PDF: ".pdf",
    SourceFormat.DOCX: ".docx",
    SourceFormat.PPTX: ".pptx",
    SourceFormat.XLSX: ".xlsx",
    SourceFormat.HTML: ".html",
    SourceFormat.MARKDOWN: ".md",
    SourceFormat.TEXT: ".txt",
}


class IndexingPipeline:
    """Wire Source → Parser → Chunker, optionally persisting the truth store."""

    def __init__(
        self,
        parser: Optional[Parser] = None,
        chunker: Optional[Chunker] = None,
        enricher: Optional[Enricher] = None,
        blob_store: Optional[BlobStore] = None,
        trace: TraceHook = _noop_trace,
    ) -> None:
        # AutoParser routes any format; FixedChunker is a sane default; the
        # enricher defaults to NoOpEnricher so the flow never branches on it.
        self.parser = parser if parser is not None else AutoParser()
        self.chunker = chunker if chunker is not None else FixedChunker()
        self.enricher = enricher if enricher is not None else NoOpEnricher()
        self.blob_store = blob_store
        self.trace = trace

    def index(self, sources: Source | Iterable[Source]) -> Iterator[Chunk]:
        """Stream chunks for every source, capturing truth blobs on the way."""
        if isinstance(sources, Source):
            sources = [sources]
        for source in sources:
            content_hash = source.content_hash() if self.blob_store else None
            if self.blob_store is not None and content_hash is not None:
                self._store_raw(source, content_hash)
            doc = self._parse(source, content_hash)
            if self.blob_store is not None and content_hash is not None:
                self._store_parsed(source, doc, content_hash)
            yield from self._chunk(source, doc)

    # -- stages --------------------------------------------------------------

    def _parse(self, source: Source, content_hash: Optional[str]) -> Document:
        start = time.perf_counter()
        # Parse-cache READ: if this exact (content × parser) was parsed before,
        # load the Document from the blob store instead of re-parsing. The parser
        # stays pure — the caching lives here, in the pipeline that owns the store.
        doc: Optional[Document] = None
        if self.blob_store is not None and content_hash is not None:
            doc = self._load_parsed(content_hash)
        hit = doc is not None
        if doc is None:
            doc = self.parser.parse(source)
        self.trace(TraceEvent(
            "parse", source.uri, _ms(start),
            {"doc_id": doc.id, "pages": len(doc.pages), "cache_hit": hit},
        ))
        return doc

    def _load_parsed(self, content_hash: str) -> Optional[Document]:
        assert self.blob_store is not None
        fp = self.parser.fingerprint()
        md_key = f"parsed/{content_hash}/{fp}.md"
        meta_key = f"parsed/{content_hash}/{fp}.meta.json"
        if not (self.blob_store.exists(md_key) and self.blob_store.exists(meta_key)):
            return None
        markdown = self.blob_store.get(md_key).decode("utf-8")
        meta = json.loads(self.blob_store.get(meta_key).decode("utf-8"))
        pages = [PageSpan(p[0], p[1], p[2], p[3]) for p in meta["pages"]]
        return Document(
            id=meta["doc_id"], markdown=markdown, pages=pages,
            source_uri=meta["source_uri"], metadata=meta["metadata"],
        )

    def _chunk(self, source: Source, doc: Document) -> Iterator[Chunk]:
        start = time.perf_counter()
        count = 0
        # Chunk → Enrich → out. The enricher (NoOp by default) may augment
        # chunk text or add synthetic chunks; it sees the parent document.
        for chunk in self.enricher.enrich(self.chunker.chunk(doc), doc):
            count += 1
            yield chunk
        self.trace(TraceEvent(
            "chunk", source.uri, _ms(start),
            {"doc_id": doc.id, "chunks": count},
        ))

    # -- truth store ---------------------------------------------------------

    def _store_raw(self, source: Source, content_hash: str) -> None:
        assert self.blob_store is not None
        key = f"raw/{content_hash}/original{_extension_for(source)}"
        start = time.perf_counter()
        hit = self.blob_store.exists(key)
        if not hit:
            with source.open() as stream:
                self.blob_store.put(key, stream.read())
        self.trace(TraceEvent(
            "store_raw", source.uri, _ms(start),
            {"key": key, "cache_hit": hit},
        ))

    def _store_parsed(
        self, source: Source, doc: Document, content_hash: str
    ) -> None:
        assert self.blob_store is not None
        fp = self.parser.fingerprint()
        md_key = f"parsed/{content_hash}/{fp}.md"
        meta_key = f"parsed/{content_hash}/{fp}.meta.json"
        start = time.perf_counter()
        hit = self.blob_store.exists(md_key)
        if not hit:
            self.blob_store.put(md_key, doc.markdown.encode("utf-8"))
            self.blob_store.put(meta_key, _meta_bytes(doc, content_hash, fp))
        self.trace(TraceEvent(
            "store_parsed", source.uri, _ms(start),
            {"key": md_key, "cache_hit": hit},
        ))


class QueryPipeline:
    """Wire Query → retrieve → rerank → ranked ScoredChunks.

    The online mirror of `IndexingPipeline`, and just as thin: fetch a generous
    candidate list from the retriever, hand it to the reranker for a precise
    top-`k`. The reranker defaults to `NoOpReranker` (Null Object) so this code
    never branches on whether reranking is configured — `retrieve 50 → rerank
    to k` is one straight path whether the reranker is a cross-encoder or a
    passthrough.

    The retriever is a composed component (it wraps a populated store/index), so
    it is passed in as an instance — same wiring philosophy as the retrievers
    themselves.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None,
        fetch_k: int = 50,
        trace: TraceHook = _noop_trace,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker if reranker is not None else NoOpReranker()
        self.fetch_k = fetch_k
        self.trace = trace

    def query(self, query: Query | str, k: int = 10) -> list[ScoredChunk]:
        if isinstance(query, str):
            query = Query(text=query)

        start = time.perf_counter()
        candidates = self.retriever.retrieve(query, self.fetch_k)
        self.trace(TraceEvent(
            "retrieve", query.text, _ms(start),
            {"retriever": self.retriever.name, "candidates": len(candidates)},
        ))

        start = time.perf_counter()
        results = self.reranker.rerank(query, candidates, k)
        self.trace(TraceEvent(
            "rerank", query.text, _ms(start),
            {"reranker": self.reranker.name, "results": len(results)},
        ))
        return results


class RagPipeline:
    """Facade: the whole loop in two calls — `index(sources)` then `ask(q)`.

    Owns an IndexingPipeline (parse→chunk), an embedder + vector store (the
    searchable index it fills), a QueryPipeline (retrieve→rerank), and a
    generator. `index` embeds chunks in batches and upserts them; `ask` runs a
    query straight through to a cited Answer.

    Defaults are the zero-dependency stack — `HashingEmbedder`,
    `MemoryVectorStore`, `ExtractiveGenerator` — so `RagPipeline().index(src)`
    then `.ask("...")` works out of the box with no extras and no API key. Swap
    in `SentenceTransformerEmbedder` / `QdrantVectorStore` / `AnthropicGenerator`
    for production by passing them in — the wiring doesn't change. For hybrid
    retrieval or other custom shapes, compose `QueryPipeline` + a `Generator`
    directly; this facade covers the dense 90% case.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        generator: Optional[Generator] = None,
        parser: Optional[Parser] = None,
        chunker: Optional[Chunker] = None,
        enricher: Optional[Enricher] = None,
        reranker: Optional[Reranker] = None,
        blob_store: Optional[BlobStore] = None,
        embedding_cache: Optional[BlobStore] = None,
        fetch_k: int = 50,
        batch_size: int = 32,
        trace: TraceHook = _noop_trace,
    ) -> None:
        self.embedder = embedder if embedder is not None else HashingEmbedder()
        self.store = store if store is not None else MemoryVectorStore()
        self.generator = (
            generator if generator is not None else ExtractiveGenerator()
        )
        self.batch_size = batch_size
        # Opt-in embedding cache: skip re-embedding text already vectorized with
        # this exact embedder (keyed by the embedder's fingerprint).
        self._emb_cache = (
            _EmbeddingCache(embedding_cache, self.embedder.fingerprint())
            if embedding_cache is not None else None
        )
        self.indexing = IndexingPipeline(
            parser=parser, chunker=chunker, enricher=enricher,
            blob_store=blob_store, trace=trace,
        )
        retriever = DenseRetriever(embedder=self.embedder, store=self.store)
        self.query_pipeline = QueryPipeline(retriever, reranker, fetch_k, trace)

    def index(self, sources: Source | Iterable[Source]) -> None:
        """Ingest, chunk, embed, and upsert — makes `sources` askable.

        Chunks are embedded in batches (O(batch) memory, one embedder call per
        batch instead of per chunk)."""
        batch: list[Chunk] = []
        for chunk in self.indexing.index(sources):
            batch.append(chunk)
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)

    def ask(self, question: Query | str, k: int = 8) -> Answer:
        """Retrieve → rerank → generate a cited Answer for `question`."""
        query = question if isinstance(question, Query) else Query(text=question)
        context = self.query_pipeline.query(query, k)
        return self.generator.generate(query, context)

    def _flush(self, chunks: list[Chunk]) -> None:
        self.store.upsert(chunks, self._embed([c.text for c in chunks]))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._emb_cache is None:
            return self.embedder.embed_texts(texts)
        # Reuse cached vectors; embed only the misses, then cache them.
        cached: list[Optional[list[float]]] = [self._emb_cache.get(t) for t in texts]
        misses = [i for i, v in enumerate(cached) if v is None]
        if misses:
            fresh = self.embedder.embed_texts([texts[i] for i in misses])
            for i, vector in zip(misses, fresh):
                self._emb_cache.put(texts[i], vector)
                cached[i] = vector
        return [v for v in cached if v is not None]  # order preserved, all filled


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _extension_for(source: Source) -> str:
    fmt = detect_format(source)
    ext = _CANONICAL_EXT.get(fmt)
    if ext is not None:
        return ext
    return Path(source.uri).suffix  # IMAGE/UNKNOWN: keep the original suffix


def _meta_bytes(doc: Document, content_hash: str, parser_fp: str) -> bytes:
    """Serialize the provenance needed to rebuild a Document without re-parsing
    (spans are the part that can't be recomputed from markdown alone)."""
    meta = {
        "doc_id": doc.id,
        "source_uri": doc.source_uri,
        "content_hash": content_hash,
        "parser_fingerprint": parser_fp,
        "metadata": doc.metadata,
        "pages": [
            [s.page_number, s.start, s.end, s.ocr_applied] for s in doc.pages
        ],
    }
    return json.dumps(meta, sort_keys=True).encode("utf-8")
