"""IndexingPipeline: thin wiring, tracing hook, and opt-in truth store."""
import json

from rag_blocks.chunking.fixed import FixedChunker
from rag_blocks.core.contracts import Source
from rag_blocks.ingestion.parsers.plaintext import PlainTextParser
from rag_blocks.pipeline import IndexingPipeline, TraceEvent
from rag_blocks.storage.local import LocalBlobStore


def text_source(body="hello world\n\nsecond paragraph\n", name="doc.txt"):
    return Source.from_bytes(body.encode(), name=name)


def test_index_streams_chunks_with_provenance():
    src = text_source()
    chunks = list(IndexingPipeline().index(src))
    assert chunks, "pipeline produced no chunks"
    assert all(c.doc_id == src.content_hash() for c in chunks)
    assert all(c.char_start is not None and c.page_start is not None for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_index_accepts_many_sources():
    srcs = [text_source(name=f"d{i}.txt", body=f"body {i}\n") for i in range(3)]
    chunks = list(IndexingPipeline().index(srcs))
    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 3


def test_tracing_hook_sees_parse_and_chunk_without_a_store():
    events: list[TraceEvent] = []
    list(IndexingPipeline(trace=events.append).index(text_source()))
    stages = [e.stage for e in events]
    assert stages == ["parse", "chunk"]
    assert all(e.duration_ms >= 0 for e in events)


def test_blob_store_captures_raw_and_parsed_truth(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    src = text_source(body="alpha\n\nbeta\n")
    events: list[TraceEvent] = []
    pipeline = IndexingPipeline(blob_store=store, trace=events.append)

    list(pipeline.index(src))

    h = src.content_hash()
    raw_key = f"raw/{h}/original.txt"
    md_key = f"parsed/{h}/{pipeline.parser.fingerprint()}.md"
    meta_key = f"parsed/{h}/{pipeline.parser.fingerprint()}.meta.json"

    # Raw bytes are the exact source; parsed markdown is the parsed Document.
    assert store.get(raw_key) == b"alpha\n\nbeta\n"
    assert store.exists(md_key) and store.exists(meta_key)
    meta = json.loads(store.get(meta_key))
    assert meta["content_hash"] == h
    assert meta["pages"], "meta must record page spans"

    # First run stores everything fresh.
    by_stage = {e.stage: e for e in events}
    assert by_stage["store_raw"].detail["cache_hit"] is False
    assert by_stage["store_parsed"].detail["cache_hit"] is False


class _CountingParser(PlainTextParser):
    """PlainTextParser that records how many times it actually parsed."""
    name = "counting-parser"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parses = 0

    def parse(self, source):
        self.parses += 1
        return super().parse(source)


def test_parse_cache_read_skips_reparse(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    parser = _CountingParser(page_chars=10_000_000)
    pipeline = IndexingPipeline(parser=parser, blob_store=store)
    src = text_source(body="# T\nsome body text\n")

    chunks1 = list(pipeline.index(src))
    assert parser.parses == 1                     # first run parses

    events: list[TraceEvent] = []
    pipeline.trace = events.append
    chunks2 = list(pipeline.index(src))
    assert parser.parses == 1                     # second run did NOT re-parse
    assert {e.stage: e.detail.get("cache_hit") for e in events}["parse"] is True
    assert [c.text for c in chunks2] == [c.text for c in chunks1]  # identical output


def test_reindexing_same_content_is_deduped(tmp_path):
    store = LocalBlobStore(root=str(tmp_path))
    src = text_source(body="same bytes\n")
    # First run populates the store (index() is a generator — must be consumed).
    list(IndexingPipeline(blob_store=store).index(src))

    events: list[TraceEvent] = []
    list(IndexingPipeline(blob_store=store, trace=events.append).index(src))
    by_stage = {e.stage: e for e in events}
    assert by_stage["store_raw"].detail["cache_hit"] is True
    assert by_stage["store_parsed"].detail["cache_hit"] is True


def test_custom_chunker_is_used():
    src = text_source(body="x" * 5000)
    small = list(IndexingPipeline(chunker=FixedChunker(chunk_chars=500)).index(src))
    big = list(IndexingPipeline(chunker=FixedChunker(chunk_chars=5000)).index(src))
    assert len(small) > len(big)
