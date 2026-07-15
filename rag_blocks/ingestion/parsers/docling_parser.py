"""DoclingParser: the flagship "anything → markdown" parser.

Pattern inventory for this file
-------------------------------
- Adapter        : wraps Docling's converter behind our Parser contract.
- Strategy (x2)  : the OCR *engine* is injected (any provider), and the OCR
                   *policy* (AUTO/FORCE/NEVER) selects the routing behavior.
- Dependency
  Inversion      : this class depends on the abstract `OcrEngine`, never on
                   Mistral/Google concretely — engines arrive by name through
                   the registry, so `DoclingParser(ocr_engine="mistral")` and
                   `DoclingParser(ocr_engine="my-custom")` are the same code
                   path.

The two hard problems this file solves
--------------------------------------

1) MEMORY — never parse a huge PDF in one shot.
   PDFs are the only common format that gets truly enormous, and they are
   random-access by design (pdfium can open page 1 400 without touching the
   rest). So PDFs are processed in *windows* of `page_batch_size` pages:
   probe page count cheaply → convert window → yield its Pages → next
   window. Peak memory is O(window), not O(document), and because
   `iter_pages` is a generator, downstream stages consume with backpressure.
   Non-PDF office formats don't offer sub-file random access (a .docx is one
   zip; layout is global), so they convert whole — acceptable because they
   are rarely memory-problematic; the asymmetry is deliberate, not an
   oversight.

2) OCR ROUTING — decide per PAGE, not per document.
   Real corpora are full of mixed PDFs: a digital report with scanned
   annexes. A document-level flag either wastes OCR on 200 clean pages or
   silently drops the 12 scanned ones. So under AUTO we probe each page's
   embedded text layer (pdfium char count — microseconds, no rendering):
   pages above `min_chars_digital` chars go through Docling's fast text
   pipeline; pages below it are rendered to images and sent to the injected
   OcrEngine. Consecutive same-kind pages are grouped into segments so
   Docling still gets efficient multi-page windows.

   Policy matrix (external engine injected):
       AUTO   probe → digital segments: Docling(no-ocr) | scanned: engine
       FORCE  every page rendered → engine   (rescues garbage text layers)
       NEVER  Docling(no-ocr) only           (fastest; scans yield nothing)
   Without an external engine, the policy maps onto Docling's built-in OCR
   (EasyOCR/Tesseract/...), which already does its own per-page detection —
   we delegate rather than duplicate.

Version note: requires docling>=2.15 for `page_range`; per-page markdown
export (`export_to_markdown(page_no=...)`) exists in recent releases and is
guarded with a graceful window-level fallback for older ones.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ...core.contracts import Page, Source, SourceFormat
from ...core.errors import ParseError, UnsupportedFormatError
from ...core.registry import registry
from ..detection import detect_format
from ..ocr.base import OcrEngine, OcrPolicy, PageImage
from .base import Parser

__all__ = ["DoclingParser"]


def _windows(start: int, end: int, size: int) -> Iterator[tuple[int, int]]:
    """Split an inclusive 1-based page range into inclusive windows."""
    s = start
    while s <= end:
        e = min(s + size - 1, end)
        yield s, e
        s = e + 1


@registry.register
class DoclingParser(Parser):
    name = "docling"
    version = "0.1.0"
    supported_formats = (
        SourceFormat.PDF,
        SourceFormat.DOCX,
        SourceFormat.PPTX,
        SourceFormat.XLSX,
        SourceFormat.HTML,
        SourceFormat.IMAGE,
    )

    @dataclass
    class Config:
        ocr_policy: OcrPolicy = OcrPolicy.AUTO
        #: Registry name of an external OcrEngine ("mistral", "google-docai",
        #: your own). None ⇒ use Docling's built-in OCR stack.
        ocr_engine: Optional[str] = None
        ocr_engine_config: dict = field(default_factory=dict)
        #: Pages per Docling conversion window — the memory/throughput dial.
        page_batch_size: int = 8
        #: AUTO threshold: a page whose text layer has fewer chars than this
        #: is considered scanned. 32 skips false positives from stray page
        #: numbers while catching genuinely image-only pages.
        min_chars_digital: int = 32
        #: Render resolution for pages sent to external OCR. 200 dpi is the
        #: usual accuracy/size sweet spot; bump to 300 for dense small print.
        render_dpi: int = 200

    # ------------------------------------------------------------------ init

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        super().__init__(config, **overrides)
        self._converters: dict[tuple[bool, bool], Any] = {}  # heavy → cache
        self._dl_modules: Any = None
        self._pdfium: Any = None
        self._ocr: Optional[OcrEngine] = None
        if self.config.ocr_engine:
            # Fail fast on unknown engine names — at construction, not on
            # page 500 of the first document.
            engine = registry.create(
                "ocr", self.config.ocr_engine, **self.config.ocr_engine_config
            )
            if not isinstance(engine, OcrEngine):
                raise ParseError(
                    f"Component 'ocr:{self.config.ocr_engine}' is not an OcrEngine"
                )
            self._ocr = engine

    def describe(self) -> dict:
        """Extend the base description with the nested engine identity so
        the eval-suite fingerprint changes when the engine (not just its
        name) changes — correct cache invalidation."""
        info = super().describe()
        if self._ocr is not None:
            info["ocr_engine_fingerprint"] = self._ocr.fingerprint()
        return info

    # ------------------------------------------------------------- main flow

    def iter_pages(self, source: Source) -> Iterator[Page]:
        fmt = detect_format(source)
        if fmt not in self.supported_formats:
            raise UnsupportedFormatError(
                f"DoclingParser cannot handle format {fmt.value!r} "
                f"(source: {source.uri})"
            )

        if fmt is SourceFormat.PDF:
            yield from self._iter_pdf(source)
        elif (
            fmt is SourceFormat.IMAGE
            and self._ocr is not None
            and self.config.ocr_policy is not OcrPolicy.NEVER
        ):
            yield from self._iter_image_external(source)
        else:
            yield from self._iter_whole_with_docling(source)

    # ---------------------------------------------------------- PDF routing

    def _iter_pdf(self, source: Source) -> Iterator[Page]:
        policy = self.config.ocr_policy

        if self._ocr is None or policy is OcrPolicy.NEVER:
            # Pure Docling path; policy maps onto its pipeline options.
            yield from self._iter_pdf_docling(
                source,
                do_ocr=(self._ocr is None and policy is not OcrPolicy.NEVER),
                force=(self._ocr is None and policy is OcrPolicy.FORCE),
            )
            return

        yield from self._iter_pdf_hybrid(source)

    def _iter_pdf_docling(
        self, source: Source, *, do_ocr: bool, force: bool
    ) -> Iterator[Page]:
        """Windowed conversion, everything delegated to Docling."""
        page_count = self._pdf_page_count(source)
        for start, end in _windows(1, page_count, self.config.page_batch_size):
            # ocr_applied is only asserted when we KNOW OCR produced the text
            # (force). Under AUTO Docling OCRs bitmap regions selectively and
            # doesn't tell us per page — claiming ocr_applied=True would lie.
            yield from self._convert_window(
                source, start, end, do_ocr=do_ocr, force=force,
                ocr_applied=force,
            )

    def _iter_pdf_hybrid(self, source: Source) -> Iterator[Page]:
        """Per-page router: digital segments → Docling, scanned → engine.

        The pdfium document is opened ONCE for probing and rendering; only
        one page bitmap exists at a time (rendered lazily, released after
        the engine consumes it).
        """
        pdfium = self._load_pdfium()
        pdf = pdfium.PdfDocument(self._pdfium_input(source))
        try:
            if self.config.ocr_policy is OcrPolicy.FORCE:
                plan = [("ocr", 1, len(pdf))]
            else:  # AUTO
                plan = self._plan_segments(self._probe_char_counts(pdf))

            for seg_kind, start, end in plan:
                if seg_kind == "docling":
                    for ws, we in _windows(start, end, self.config.page_batch_size):
                        yield from self._convert_window(
                            source, ws, we,
                            do_ocr=False, force=False, ocr_applied=False,
                        )
                else:
                    yield from self._ocr_pages(source, pdf, start, end)
        finally:
            pdf.close()

    def _plan_segments(
        self, char_counts: list[int]
    ) -> list[tuple[str, int, int]]:
        """Group consecutive same-kind pages: [('docling',1,42),('ocr',43,44),…]
        so Docling keeps efficient multi-page windows between scanned runs."""
        segments: list[tuple[str, int, int]] = []
        for page_no, chars in enumerate(char_counts, start=1):
            seg_kind = (
                "docling" if chars >= self.config.min_chars_digital else "ocr"
            )
            if segments and segments[-1][0] == seg_kind:
                segments[-1] = (seg_kind, segments[-1][1], page_no)
            else:
                segments.append((seg_kind, page_no, page_no))
        return segments

    def _ocr_pages(
        self, source: Source, pdf: Any, start: int, end: int
    ) -> Iterator[Page]:
        """Render each scanned page and stream it through the engine.

        Images are produced by a generator and fed to `recognize_batch`, so
        an engine that parallelizes/batches gets the hook while the default
        stays strictly one-image-in-memory."""
        assert self._ocr is not None  # reached only on the external-engine path
        page_numbers = range(start, end + 1)
        images = (self._render_pdf_page(pdf, p) for p in page_numbers)
        for page_no, result in zip(page_numbers, self._ocr.recognize_batch(images)):
            yield Page(
                number=page_no,
                markdown=result.markdown,
                ocr_applied=True,
                metadata={
                    "ocr_engine": self._ocr.name,
                    "ocr_confidence": result.confidence,
                },
            )

    # ----------------------------------------------------- non-PDF branches

    def _iter_image_external(self, source: Source) -> Iterator[Page]:
        """Standalone image + external engine: skip Docling entirely."""
        assert self._ocr is not None  # reached only when an engine is configured
        data = source.head(n=64 * 1024 * 1024)  # images are single-page
        image = PageImage(data=data, page_number=1, mime=_image_mime(data))
        result = self._ocr.recognize(image)
        yield Page(
            number=1,
            markdown=result.markdown,
            ocr_applied=True,
            metadata={"ocr_engine": self._ocr.name,
                      "ocr_confidence": result.confidence},
        )

    def _iter_whole_with_docling(self, source: Source) -> Iterator[Page]:
        """Office/HTML/image formats: no sub-file random access exists, so
        convert whole and re-paginate from the result when possible."""
        converter = self._get_converter(do_ocr=True, force=False)
        try:
            result = converter.convert(self._docling_input(source))
        except Exception as exc:  # noqa: BLE001
            raise ParseError(
                f"Docling conversion failed: {exc}", source_uri=source.uri
            ) from exc

        doc = result.document
        try:
            n_pages = int(doc.num_pages())
        except Exception:  # noqa: BLE001 - not all formats paginate
            n_pages = 1

        if n_pages > 1:  # e.g. PPTX slides keep their page identity
            yield from self._pages_from_result(doc, 1, n_pages, ocr_applied=False)
        else:
            yield Page(number=1, markdown=doc.export_to_markdown())

    # ------------------------------------------------------ docling helpers

    def _convert_window(
        self, source: Source, start: int, end: int,
        *, do_ocr: bool, force: bool, ocr_applied: bool,
    ) -> Iterator[Page]:
        converter = self._get_converter(do_ocr=do_ocr, force=force)
        try:
            result = converter.convert(
                self._docling_input(source), page_range=(start, end)
            )
        except Exception as exc:  # noqa: BLE001
            raise ParseError(
                f"Docling conversion failed: {exc}",
                source_uri=source.uri, page_number=start,
            ) from exc
        yield from self._pages_from_result(
            result.document, start, end, ocr_applied=ocr_applied
        )

    def _pages_from_result(
        self, dl_doc: Any, start: int, end: int, *, ocr_applied: bool
    ) -> Iterator[Page]:
        """Per-page markdown export, with a window-level fallback for older
        docling versions lacking `page_no` (the Page then records its true
        span in metadata so provenance degrades gracefully, never wrongly)."""
        try:
            exports = [
                (p, dl_doc.export_to_markdown(page_no=p))
                for p in range(start, end + 1)
            ]
        except TypeError:
            markdown = dl_doc.export_to_markdown()
            yield Page(
                number=start, markdown=markdown, ocr_applied=ocr_applied,
                metadata={"page_span": [start, end]},
            )
            return
        for page_no, markdown in exports:
            if not markdown.strip():
                continue  # blank pages add nothing but separators
            yield Page(number=page_no, markdown=markdown, ocr_applied=ocr_applied)

    def _get_converter(self, *, do_ocr: bool, force: bool) -> Any:
        """Docling converters load layout models — expensive. Cache one per
        option set and reuse across every window and document (this is the
        single biggest throughput win in the whole parser)."""
        key = (do_ocr, force)
        if key not in self._converters:
            dc, bm, po = self._load_docling()
            options = po.PdfPipelineOptions()
            options.do_ocr = do_ocr
            if force:
                options.ocr_options.force_full_page_ocr = True
            self._converters[key] = dc.DocumentConverter(
                format_options={
                    bm.InputFormat.PDF: dc.PdfFormatOption(
                        pipeline_options=options
                    ),
                }
            )
        return self._converters[key]

    def _docling_input(self, source: Source) -> Any:
        if source.path is not None:
            return str(source.path)
        _, bm, _ = self._load_docling()
        if source.data is None:
            raise ParseError(
                "Source has neither a readable file path nor in-memory bytes",
                source_uri=source.uri,
            )
        return bm.DocumentStream(name=source.uri, stream=io.BytesIO(source.data))

    # ------------------------------------------------------- pdfium helpers

    def _pdfium_input(self, source: Source) -> Any:
        return str(source.path) if source.path is not None else source.data

    def _pdf_page_count(self, source: Source) -> int:
        pdfium = self._load_pdfium()
        pdf = pdfium.PdfDocument(self._pdfium_input(source))
        try:
            return len(pdf)
        finally:
            pdf.close()

    def _probe_char_counts(self, pdf: Any) -> list[int]:
        """Chars in each page's embedded text layer — the AUTO signal.
        Pure metadata reads: no rendering, no layout model, microseconds."""
        counts: list[int] = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                counts.append(textpage.count_chars())
            except AttributeError:  # pypdfium2 API drift safety
                counts.append(len(textpage.get_text_bounded()))
            finally:
                textpage.close()
                page.close()
        return counts

    def _render_pdf_page(self, pdf: Any, page_number: int) -> PageImage:
        page = pdf[page_number - 1]
        try:
            bitmap = page.render(scale=self.config.render_dpi / 72)
            pil_image = bitmap.to_pil()
        finally:
            page.close()
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return PageImage(
            data=buffer.getvalue(),
            page_number=page_number,
            mime="image/png",
            dpi=self.config.render_dpi,
        )

    # --------------------------------------------------------- lazy imports

    def _load_docling(self) -> tuple[Any, Any, Any]:
        if self._dl_modules is None:
            try:
                import docling.datamodel.base_models as bm
                import docling.datamodel.pipeline_options as po
                import docling.document_converter as dc
            except ImportError as exc:
                raise ParseError(
                    "DoclingParser requires 'docling'. "
                    "Install with: pip install 'rag-blocks[docling]'"
                ) from exc
            self._dl_modules = (dc, bm, po)
        return self._dl_modules

    def _load_pdfium(self) -> Any:
        if self._pdfium is None:
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise ParseError(
                    "PDF handling requires 'pypdfium2' (ships with docling). "
                    "Install with: pip install 'rag-blocks[docling]'"
                ) from exc
            self._pdfium = pdfium
        return self._pdfium


def _image_mime(head: bytes) -> str:
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
