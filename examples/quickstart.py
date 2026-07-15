"""Runnable tour of the ingestion subsystem.

Everything below runs on the stdlib alone (no docling / mistralai needed):
format detection, registry, streaming plaintext parsing, provenance, and
component fingerprints. PDF/OCR paths need `pip install "rag-blocks[docling]"`.

Run from the repo root:  python examples/quickstart.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_blocks as rk


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="rag_blocks_demo_"))
    sample = tmp / "notes.md"
    sample.write_text(
        "# Retrieval notes\n\n"
        "Hybrid retrieval combines dense and sparse signals.\n\n"
        + ("Reciprocal rank fusion merges ranked lists robustly. " * 120),
        encoding="utf-8",
    )

    print("== 1. Registry: what components exist right now ==")
    print(rk.registry.available())

    print("\n== 2. Format detection: bytes, not extensions ==")
    fake_pdf = rk.Source.from_bytes(b"%PDF-1.7 rest-of-file...", name="mystery_file")
    print("bytes starting with %PDF     ->", rk.detect_format(fake_pdf).value)
    print("actual markdown file          ->", rk.detect_format(rk.Source.from_path(sample)).value)

    print("\n== 3. One-call ingestion (Facade) ==")
    doc = rk.ingest(sample)
    print("doc id (content hash):", doc.id)
    print("metadata:", doc.metadata)
    print("markdown preview:", repr(doc.markdown[:80]))

    print("\n== 4. Provenance: chars [100:400) touch pages", doc.pages_for_span(100, 400), "==")
    for span in doc.pages:
        print(f"   page {span.page_number}: chars [{span.start}:{span.end}) ocr={span.ocr_applied}")

    print("\n== 5. Streaming: pages arrive one at a time ==")
    parser = rk.PlainTextParser(page_chars=1500)
    for page in parser.iter_pages(rk.Source.from_path(sample)):
        print(f"   page {page.number}: {len(page.markdown)} chars")

    print("\n== 6. Fingerprints: config = identity (the future cache keys) ==")
    default = rk.DoclingParser()
    tuned = rk.DoclingParser(page_batch_size=4)
    with_mistral = rk.DoclingParser(ocr_engine="mistral")
    print("   default          :", default.fingerprint())
    print("   batch_size=4     :", tuned.fingerprint())
    print("   ocr=mistral      :", with_mistral.fingerprint())
    print("   describe(mistral):", with_mistral.describe())


if __name__ == "__main__":
    main()
