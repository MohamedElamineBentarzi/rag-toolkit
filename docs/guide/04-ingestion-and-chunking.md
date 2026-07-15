# 04 · Reading files & cutting them up

Before you can search your documents, two things happen: the toolkit **reads**
each file into clean text (with page numbers), then **cuts** that text into
passage-sized pieces. This page shows how both work and how to control them.

```
your file → detect what it is → read into a Document → cut into Chunks
```

Most of the time you don't call any of this yourself — `rag.index(source)` does
it. But understanding it lets you get better results (e.g. cutting on section
headings instead of blindly by length).

## Reading files

### It figures out the file type from the contents, not the name

Files get renamed and mislabeled all the time — a PDF saved as `scan.pdf.txt`, a
Word doc with no extension. So the toolkit looks at the actual bytes to decide
what a file is:

```python
from rag_blocks import detect_format, Source
detect_format(Source.from_path("scan.pdf.txt"))   # → PDF, correctly
```

It recognizes PDFs, images (PNG/JPEG/etc.), Word/PowerPoint/Excel, HTML, and
plain text/markdown — all from their content signatures. The file extension is
only used as a tiebreaker when the content alone is ambiguous (plain text vs.
markdown look identical).

### The reader ("parser")

A **parser** turns a file into a `Document`. The toolkit picks the right one
automatically based on the detected type:

| File type | Reader | Add-on needed |
|---|---|---|
| `.txt`, `.md` | built-in | none |
| PDF, Word, PowerPoint, Excel, HTML, images | `docling` | `rag-blocks[docling]` |

You normally don't choose a parser by hand — the automatic router (`AutoParser`)
does it. But you *can* override it, for example to point PDFs at your own reader,
or to tune how PDFs are read:

```python
from rag_blocks import AutoParser
parser = AutoParser(parser_configs={"docling": {
    "ocr_engine": "mistral",     # use Mistral for scanned pages
    "ocr_policy": "auto",         # only OCR pages that need it
    "page_batch_size": 4,
}})
```

There's also a one-liner that reads a file straight into a `Document`:

```python
import rag_blocks as rk
doc = rk.ingest("report.pdf")
```

### Reading is streaming

Files are read one page at a time, so even a 2,000-page PDF never loads all at
once. This is automatic — you don't do anything to get it — and it's why the
toolkit runs fine on an ordinary laptop. If you want page-by-page access
yourself:

```python
for page in parser.iter_pages(Source.from_path("huge.pdf")):
    print(page.number, len(page.markdown))
```

## Scanned documents (OCR)

A scanned page is just an image — there's no text to extract. For those, the
reader can send the page image to an **OCR engine** that reads the text off it.
You control two things:

- **When** to OCR (`ocr_policy`): `auto` checks each page and only OCRs the ones
  with no real text; `force` OCRs every page (useful when a document has a garbage
  text layer); `never` skips OCR entirely.
- **Which engine** does it: `mistral` (add-on `[mistral]`) or Google Document AI
  (add-on `[google]`).

```python
import rag_blocks as rk
doc = rk.ingest("scan.pdf", ocr_engine="mistral", ocr_policy=rk.OcrPolicy.FORCE)
```

Credentials come from environment variables (e.g. `MISTRAL_API_KEY`) or from
config you pass in — and secret fields are never written to logs.

## Cutting into chunks

Once you have a `Document`, it gets cut into `Chunk`s — the pieces that get
searched. A **chunker** decides *where* to cut. The toolkit ships two, and you
pick by how your documents are shaped.

### `MarkdownChunker` — cut on headings (the recommended default)

Since every document is read into markdown, its **structure survives** — headings
are still there. This chunker cuts at each heading, so every chunk is a coherent
section that starts with its own heading:

```python
from rag_blocks import MarkdownChunker
MarkdownChunker()
```

This is usually what you want: a chunk about "Section 3.2 Liability" holds that
whole section, not an arbitrary slice that happens to start mid-sentence.

### `FixedChunker` — cut by length, with overlap

When your documents have little structure (a wall of text, a transcript), cut by
size instead:

```python
from rag_blocks import FixedChunker
FixedChunker(chunk_chars=1600, overlap_chars=200)
```

Two touches make this work well:
- **Overlap** — each piece repeats the last `overlap_chars` of the previous one,
  so a fact sitting on a boundary still lands whole in at least one chunk.
- **Clean cuts** — it prefers to end a chunk at a paragraph or line break rather
  than mid-sentence, without making tiny fragments.

### Both keep the provenance

Whichever chunker you use, every chunk it produces comes with its `char_start`,
`char_end`, `page_start`, and `page_end` filled in — so answers can always cite
the right page. You don't have to do anything to keep this; it's guaranteed.

## Doing it by hand

You don't need a pipeline to read and cut a file — handy for inspecting results:

```python
import rag_blocks as rk
from rag_blocks import MarkdownChunker

doc = rk.ingest("report.pdf")                 # file → Document
chunks = list(MarkdownChunker().chunk(doc))   # Document → chunks

print(f"{len(chunks)} chunks")
print(chunks[0].text[:200], "…")
print("from pages", chunks[0].page_start, "–", chunks[0].page_end)
```

Next: **[05 · Representations & storage](05-representations-and-storage.md)** —
how those chunks become searchable, and the `ChunkIndex` that holds them all.
