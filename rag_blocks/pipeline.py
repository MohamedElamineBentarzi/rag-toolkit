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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

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
from .core.errors import ConfigError
from .embedding.base import Embedder
from .embedding.hashing import HashingEmbedder
from .enrichment.base import Enricher
from .generation.base import Generator
from .generation.extractive import ExtractiveGenerator
from .indexing.catalog import DocumentCatalog, raw_key
from .indexing.corpus import Corpus
from .indexing.representation import DenseRepresentation
from .indexing.sink import ChunkSink
from .ingestion.detection import detect_format
from .ingestion.parsers.auto import AutoParser
from .ingestion.parsers.base import Parser
from .refinement.base import Refiner
from .retrieval.base import Retriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.index_retriever import IndexRetriever
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

    #: Write path: "parse" | "store_raw" | "store_parsed" | "chunk" | "enrich".
    #: Read path: "retrieve" | "refine" | "generate".
    stage: str
    source_uri: str            # the query text on read-path events
    duration_ms: float         # this stage's OWN cost, never a nested total
    detail: dict = field(default_factory=dict)


#: A tracing hook: called with each TraceEvent. Defaults to a no-op (Null
#: Object) so pipeline code never grows `if trace is not None` branches.
TraceHook = Callable[[TraceEvent], None]


def _noop_trace(event: TraceEvent) -> None:  # Null Object
    pass


def _measured(stream: Iterator[Chunk], meter: list[float]) -> Iterator[Chunk]:
    """Yield from `stream`, accumulating into `meter` the ms spent producing it.

    Two things this exists to get right, both consequences of the chunk/enrich
    chain being lazy:

    - **Timing the call measures nothing.** `enricher.enrich(stream, doc)` only
      *builds* a generator; the work happens later, on `next()`. So the clock
      has to ride the stream.
    - **Wall-clocking the whole drain over-bills.** A generator suspends at
      every yield, so elapsed time from first `next()` to exhaustion also
      includes whatever the *consumer* did in between — here, writing batches
      to every sink. Only time spent inside `next()` belongs to this stage.

    Meters nest: this wrapper's `next()` pulls through every layer below it, so
    its total includes theirs. A layer's own cost is its meter minus the one
    beneath — subtracted once, where the events are emitted.
    """
    total = 0.0
    try:
        while True:
            start = time.perf_counter()
            try:
                item = next(stream)
            except StopIteration:
                return
            total += _ms(start)
            yield item
    finally:
        # `finally`, so a consumer that abandons the stream early still gets
        # the cost of what it did consume attributed rather than dropped.
        meter.append(total)


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
    """The complete write path: Source → Parser → Chunker → enrich chain →
    batch → every sink (DR-0001 v2, D6).

    Two chains and a fan-out. `enrich=[...]` is a chain over the chunk stream
    (Iterator → Iterator, composing trivially — the empty chain *is* the null
    object, so there is no `NoOpEnricher`). `sinks=[...]` is the write fan-out
    (F4): each sink (a `ChunkIndex`, a `LexicalIndex`, a GraphRAG index) receives
    every batch. Batching lives here so memory stays O(batch), never O(corpus).

    Still a generator: `index(sources)` yields each chunk as it flows, so a
    caller can observe the stream while the sinks are written underneath — the
    tuner's `for _ in pipeline.index(corpus): pass` indexes once into all sinks.
    """

    def __init__(
        self,
        parser: Optional[Parser] = None,
        chunker: Optional[Chunker] = None,
        enrich: Sequence[Enricher] = (),
        sinks: Sequence[ChunkSink] = (),
        blob_store: Optional[BlobStore] = None,
        batch_size: int = 32,
        trace: TraceHook = _noop_trace,
    ) -> None:
        # AutoParser routes any format; FixedChunker is a sane default. The
        # enrich chain and the sink fan-out are both empty by default.
        self.parser = parser if parser is not None else AutoParser()
        self.chunker = chunker if chunker is not None else FixedChunker()
        self.enrich = list(enrich)
        self.sinks = list(sinks)
        self.blob_store = blob_store
        self.batch_size = batch_size
        self.trace = trace
        # doc_id → source provenance + download link, when a truth store exists.
        self.catalog = DocumentCatalog(blob_store) if blob_store is not None else None

    def index(self, sources: Source | Iterable[Source]) -> Iterator[Chunk]:
        """Stream chunks for every source, capturing truth blobs and writing
        batches to every sink on the way; persist each sink at the end."""
        if isinstance(sources, Source):
            sources = [sources]
        batch: list[Chunk] = []
        for source in sources:
            content_hash = source.content_hash() if self.blob_store else None
            if self.blob_store is not None and content_hash is not None:
                self._store_raw(source, content_hash)
            doc = self._parse(source, content_hash)
            if self.blob_store is not None and content_hash is not None:
                self._store_parsed(source, doc, content_hash)
                # Manifest: doc_id → {source_uri, content_hash, ext}, so a
                # citation's doc_id resolves to a name + download link in one hop.
                assert self.catalog is not None
                self.catalog.record(doc, content_hash, _extension_for(source))
            for chunk in self._chunk(source, doc):
                batch.append(chunk)
                if len(batch) >= self.batch_size:
                    self._write(batch)
                    batch = []
                yield chunk
        if batch:
            self._write(batch)
        for sink in self.sinks:
            sink.persist()

    def _write(self, batch: list[Chunk]) -> None:
        for sink in self.sinks:
            sink.add(batch)

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
        count = 0
        # Chunk → enrich chain → out. Each enricher wraps the previous stream
        # (Iterator → Iterator) and sees the parent document; it may augment
        # chunk text or add synthetic chunks. An empty chain is just the chunker.
        #
        # One meter per layer (see `_measured`): the tuner treats "which
        # enrichers" as a search dimension, so "the enrich chain cost 4 s" is
        # useless — it needs to know WHICH enricher spent it.
        meters: list[list[float]] = [[] for _ in range(1 + len(self.enrich))]
        stream: Iterator[Chunk] = _measured(self.chunker.chunk(doc), meters[0])
        for enricher, meter in zip(self.enrich, meters[1:]):
            stream = _measured(enricher.enrich(stream, doc), meter)

        for chunk in stream:
            count += 1
            yield chunk

        # Every meter has recorded by now: each layer's `finally` runs as it
        # exhausts, innermost first. An abandoned stream is the exception —
        # hence the empty-cell fallback rather than an index error.
        totals = [meter[0] if meter else 0.0 for meter in meters]
        # "chunk" is the CHUNKER's own cost, not the chain's: enrichment is
        # reported per enricher below, so summing stages never double-counts.
        # Identical to the old total whenever the chain is empty.
        self.trace(TraceEvent(
            "chunk", source.uri, totals[0],
            {"doc_id": doc.id, "chunks": count},
        ))
        previous = totals[0]
        for enricher, total in zip(self.enrich, totals[1:]):
            # max(): meters nest so this is non-negative by construction, but
            # a clock that jitters must never emit a negative cost.
            self.trace(TraceEvent(
                "enrich", source.uri, max(total - previous, 0.0),
                {"doc_id": doc.id, "enricher": enricher.name},
            ))
            previous = total

    # -- truth store ---------------------------------------------------------

    def _store_raw(self, source: Source, content_hash: str) -> None:
        assert self.blob_store is not None
        key = raw_key(content_hash, _extension_for(source))
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
    """Wire Query → retrieve → refine chain → truncate to k (DR-0001 v2, D9).

    The online mirror of `IndexingPipeline`, and just as thin: fetch a generous
    candidate list from the retriever, run it through the `refine=[...]` chain
    (cross-encoder reranking, neighbor expansion, score floors — uniform stages
    over one data shape), then truncate to the caller's `k`. An empty chain is a
    straight `retrieve fetch_k → take k`; there is no null reranker to configure.

    The retriever is a composed component (it wraps a populated index), so it is
    passed in as an instance — same wiring philosophy as the retrievers.
    """

    def __init__(
        self,
        retriever: Retriever,
        refine: Sequence[Refiner] = (),
        fetch_k: int = 50,
        trace: TraceHook = _noop_trace,
    ) -> None:
        self.retriever = retriever
        self.refine = list(refine)
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

        for refiner in self.refine:
            start = time.perf_counter()
            candidates = refiner.refine(query, candidates, k)
            self.trace(TraceEvent(
                "refine", query.text, _ms(start),
                {"refiner": refiner.name, "candidates": len(candidates)},
            ))
        # The pipeline owns the final truncation (refiners may return more/fewer).
        return candidates[:k]


class RagPipeline:
    """Composition root: the whole loop in two calls — `index(sources)` then
    `ask(q)` — with live backends created once at the edge and shared by
    reference (DR-0001 v2, D6).

    The shared thing is one `Corpus`: it is the write path's flagship sink
    *and* the read path's backend, so query/corpus compatibility is structural,
    not conventional. The write path fans out to the corpus (plus any
    `extra_sinks`); the read path retrieves through a derived-or-supplied
    retriever, runs the `refine` chain, and generates a cited Answer.

    Defaults are the zero-dependency stack — a `MemoryVectorStore` +
    `HashingEmbedder` `Corpus`, `ExtractiveGenerator` — so `RagPipeline()`
    works out of the box with no extras and no API key. Swap in real backends by
    constructing the `Corpus` yourself (see `.dense`) and passing it in.

    Retriever derivation (A1): no `retriever` ⇒ one representation gives an
    `IndexRetriever`, several give a `HybridRetriever` over all of them.
    """

    def __init__(
        self,
        corpus: Optional[Corpus] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        parser: Optional[Parser] = None,
        chunker: Optional[Chunker] = None,
        enrich: Sequence[Enricher] = (),
        refine: Sequence[Refiner] = (),
        extra_sinks: Sequence[ChunkSink] = (),
        blob_store: Optional[BlobStore] = None,
        fetch_k: int = 50,
        batch_size: int = 32,
        trace: TraceHook = _noop_trace,
    ) -> None:
        # Zero-config default corpus: memory store + one dense representation.
        if corpus is None:
            corpus = Corpus(
                MemoryVectorStore(), [DenseRepresentation(HashingEmbedder())]
            )
        self.corpus = corpus
        self.generator = (
            generator if generator is not None else ExtractiveGenerator()
        )
        # Retriever: derive per A1, or validate a supplied one is wired to THIS
        # corpus — the last way to recreate the write/read split (P6) becomes a
        # construction-time explosion.
        if retriever is None:
            retriever = _derive_retriever(corpus)
        else:
            wired = getattr(retriever, "corpus", None)
            if wired is not None and wired is not corpus:
                raise ConfigError(
                    "RagPipeline: the supplied retriever is wired to a "
                    "different Corpus than corpus= — pass a retriever "
                    "over this corpus, or let RagPipeline derive one."
                )
        self.retriever = retriever
        # The corpus is the flagship write sink; extra sinks (GraphRAG, alerts)
        # fan out beside it.
        self.indexing = IndexingPipeline(
            parser=parser, chunker=chunker, enrich=enrich,
            sinks=[corpus, *extra_sinks], blob_store=blob_store,
            batch_size=batch_size, trace=trace,
        )
        self.query_pipeline = QueryPipeline(retriever, refine, fetch_k, trace)
        # Held here too: generation happens in `ask`, outside either sub-pipeline.
        self.trace = trace

    @classmethod
    def dense(
        cls,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        **kw,
    ) -> "RagPipeline":
        """Convenience constructor for the 80% dense deployment: builds a
        single-representation `Corpus` and hands it to the composition root.
        All other keywords forward to `__init__`."""
        corpus = Corpus(
            store if store is not None else MemoryVectorStore(),
            [DenseRepresentation(
                embedder if embedder is not None else HashingEmbedder()
            )],
        )
        return cls(corpus=corpus, **kw)

    def index(self, sources: Source | Iterable[Source]) -> None:
        """Ingest → chunk → enrich → write every representation to the corpus
        (and any extra sinks). Streaming and batched underneath."""
        for _ in self.indexing.index(sources):
            pass

    def ask(self, question: Query | str, k: int = 8) -> Answer:
        """Retrieve → refine → generate a cited Answer for `question`."""
        answer, _ = self.ask_with_context(question, k)
        return answer

    def ask_with_context(
        self, question: Query | str, k: int = 8
    ) -> tuple[Answer, list[ScoredChunk]]:
        """`ask`, but also returning the context the answer was built from.

        Exists for evaluation: scoring retrieval and generation from one run
        needs both halves, and the alternative — calling `query_pipeline.query`
        and `generator.generate` by hand — silently skips the "generate" trace
        event below, so a trial under-reports the one stage that costs money.
        A caller should never have to reimplement a pipeline to observe it.
        """
        query = question if isinstance(question, Query) else Query(text=question)
        context = self.query_pipeline.query(query, k)
        start = time.perf_counter()
        answer = self.generator.generate(query, context)
        # The one stage that was invisible to tracing, and the expensive one:
        # generation is where the tokens are spent. `usage` rides along so a
        # cost collector never has to hold the Answer to price a trial.
        self.trace(TraceEvent(
            "generate", query.text, _ms(start),
            {
                "generator": self.generator.name,
                "context_chunks": len(context),
                "usage": dict(answer.usage),
            },
        ))
        return answer, context

    # -- citation → source resolution ----------------------------------------

    @property
    def catalog(self):
        """The `DocumentCatalog` (doc_id → provenance + download link), or `None`
        when no `blob_store` is configured (nothing durable to resolve to)."""
        return self.indexing.catalog

    def source_uri(self, doc_id: str) -> Optional[str]:
        """The original file name behind a citation's `doc_id` (for display)."""
        return self._catalog_or_raise().source_uri(doc_id)

    def download_url(self, doc_id: str, *, expires_seconds: int = 3600) -> Optional[str]:
        """A download link to the original file behind a citation's `doc_id`."""
        return self._catalog_or_raise().download_url(
            doc_id, expires_seconds=expires_seconds
        )

    def _catalog_or_raise(self) -> DocumentCatalog:
        if self.catalog is None:
            raise ConfigError(
                "RagPipeline: doc_id resolution needs a blob_store — construct "
                "with blob_store=LocalBlobStore(...) (or MinioBlobStore(...))."
            )
        return self.catalog


def _derive_retriever(corpus: Corpus) -> Retriever:
    """A1 derivation: one representation ⇒ a plain `IndexRetriever`; several ⇒
    a `HybridRetriever` fusing all of them with RRF."""
    reps = corpus.representations()
    if len(reps) == 1:
        return IndexRetriever(corpus)
    return HybridRetriever(corpus)


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
