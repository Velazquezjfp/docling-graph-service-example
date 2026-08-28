"""Docling conversion: cached converters, page-furniture stripping, Markdown export."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    ContentLayer,
    DoclingDocument,
    ImageRefMode,
    SectionHeaderItem,
    TextItem,
)

from .settings import DoclingSettings

log = logging.getLogger(__name__)

FORMATS: dict[str, InputFormat] = {
    "pdf": InputFormat.PDF,
    "docx": InputFormat.DOCX,
    "pptx": InputFormat.PPTX,
    "html": InputFormat.HTML,
    "md": InputFormat.MD,
}
_MIME = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/html": "html",
    "text/markdown": "md",
}


class UnsupportedFormat(ValueError):
    pass


class ConversionFailed(RuntimeError):
    pass


def normalize_format(fmt: str | None, name: str) -> str:
    """Accept an extension ('pdf', '.PDF'), a MIME type, or fall back to the file name's suffix."""
    candidates = []
    if fmt:
        f = fmt.strip().lower()
        candidates.append(_MIME.get(f, f.lstrip(".")))
    if "." in name:
        candidates.append(name.rsplit(".", 1)[1].lower())
    for c in candidates:
        if c in FORMATS:
            return c
    raise UnsupportedFormat(f"unsupported document format {fmt!r} for {name!r}; supported: {sorted(FORMATS)}")


@dataclass
class Converted:
    document: DoclingDocument
    status: str
    warnings: list[str] = field(default_factory=list)


class Converter:
    """Lazily builds one ``DocumentConverter`` per OCR mode; options are never mutated afterwards."""

    def __init__(self, cfg: DoclingSettings, artifacts_path: str | None):
        self.cfg = cfg
        self.artifacts_path = artifacts_path
        self._converters: dict[bool, DocumentConverter] = {}
        self._lock = threading.Lock()

    def _pdf_options(self, ocr: bool) -> PdfPipelineOptions:
        cfg = self.cfg
        opts = PdfPipelineOptions(
            do_ocr=ocr,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE if cfg.table_mode == "accurate" else TableFormerMode.FAST,
                do_cell_matching=True,
            ),
            images_scale=cfg.images_scale,
            generate_page_images=True,
            generate_picture_images=True,
            generate_parsed_pages=True,
            heading_hierarchy_options=HeadingHierarchyOptions(enabled=cfg.heading_hierarchy),
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=cfg.num_threads),
            document_timeout=cfg.document_timeout_s,
        )
        if ocr:
            opts.ocr_options = EasyOcrOptions(lang=list(cfg.ocr_langs))
        if self.artifacts_path:
            opts.artifacts_path = self.artifacts_path
        return opts

    def get(self, ocr: bool) -> DocumentConverter:
        with self._lock:
            conv = self._converters.get(ocr)
            if conv is None:
                conv = DocumentConverter(
                    allowed_formats=list(FORMATS.values()),
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self._pdf_options(ocr))},
                )
                self._converters[ocr] = conv
            return conv

    def warmup(self) -> None:
        """Load the PDF pipeline models once (startup self-check; fails fast offline if models are missing)."""
        self.get(self.cfg.ocr).initialize_pipeline(InputFormat.PDF)

    def convert(self, path: Path, *, ocr: bool | None = None) -> Converted:
        use_ocr = self.cfg.ocr if ocr is None else ocr
        result = self.get(use_ocr).convert(path, raises_on_error=False)
        warnings = [str(e.error_message) for e in result.errors] if result.errors else []
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS) or result.document is None:
            raise ConversionFailed(f"docling conversion {result.status.value}: {'; '.join(warnings) or 'no details'}")
        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            warnings.insert(0, "docling reported PARTIAL_SUCCESS (see following messages)")
        return Converted(result.document, result.status.value, warnings)


def strip_page_furniture(doc: DoclingDocument, prefixes: list[str]) -> int:
    """Move BODY text items that start with a configured prefix to the FURNITURE layer.

    Docling already puts detected page headers/footers into FURNITURE (excluded from Markdown,
    iteration and chunking); this is the belt-and-braces pass for lines the layout model missed.
    """
    if not prefixes:
        return 0
    moved = 0
    for item in doc.texts:
        if not isinstance(item, TextItem) or item.content_layer != ContentLayer.BODY:
            continue
        text = item.text.strip()
        if any(text.startswith(p) for p in prefixes):
            item.content_layer = ContentLayer.FURNITURE
            moved += 1
    return moved


def to_markdown(doc: DoclingDocument) -> str:
    return doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="<!-- image -->",
        page_break_placeholder="<!-- page break -->",
        mark_meta=True,
        allowed_meta_names={"description"},
        escape_underscores=False,
        escape_html=False,
    )


_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S")


def normalize_heading_levels(doc: DoclingDocument) -> int:
    """Derive heading levels from section numbering ("5.3.1 …" -> 3; unnumbered -> parent + 1).

    Layout-based heading levels are unreliable on typical manuals (chapter titles and subsections
    often land on the same visual level), which breaks the breadcrumb chain. Returns #changed.
    """
    changed = 0
    parent_level = 0
    for item, _ in doc.iterate_items(with_groups=False):
        if not isinstance(item, SectionHeaderItem):
            continue
        m = _NUMBERED.match(item.text)
        if m:
            level = m.group(1).count(".") + 1
            parent_level = level
        else:
            level = min(parent_level + 1, 6) if parent_level else item.level
        if item.level != level:
            item.level = level
            changed += 1
    return changed


def strip_repeated_furniture(doc: DoclingDocument, *, min_pages: int, band: float) -> int:
    """Move BODY text that repeats in the page margins on >= ``min_pages`` pages to FURNITURE.

    Catches running headers/footers the layout model missed on individual pages. Digits are
    ignored when comparing ("Seite 3 von 29" == "Seite 4 von 29").
    """
    if min_pages <= 0:
        return 0
    pages_by_text: dict[str, set[int]] = {}
    for item in doc.texts:
        if isinstance(item, TextItem) and item.prov:
            pages_by_text.setdefault(_furniture_key(item.text), set()).add(item.prov[0].page_no)
    moved = 0
    for item in doc.texts:
        if not isinstance(item, TextItem) or item.content_layer != ContentLayer.BODY or not item.prov:
            continue
        key = _furniture_key(item.text)
        if len(key) < 4 or len(pages_by_text.get(key, ())) < min_pages:
            continue
        prov = item.prov[0]
        page = doc.pages.get(prov.page_no)
        if page is None or not page.size.height:
            continue
        top = prov.bbox.to_top_left_origin(page.size.height)
        if top.b <= page.size.height * band or top.t >= page.size.height * (1 - band):
            item.content_layer = ContentLayer.FURNITURE
            moved += 1
    return moved


def _furniture_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.split())).casefold()


_WRAP_HYPHEN = re.compile(r"(?<=\w)- (?!(?:und|oder|bzw\.?|sowie|beziehungsweise)\b)(?=\w)")


def dehyphenate_table_cells(doc: DoclingDocument) -> int:
    """Join words that the PDF wrapped at a hyphen inside table cells ("bavd- issuing-ca-3").

    Docling joins wrapped cell lines with a space, which breaks identifiers (hostnames, vault paths,
    product names). German suspended hyphens ("IAM- und PKI-Betrieb") are left alone. Returns #cells.
    """
    changed = 0
    for table in doc.tables:
        for cell in table.data.table_cells:
            fixed = _WRAP_HYPHEN.sub("-", cell.text)
            if fixed != cell.text:
                cell.text = fixed
                changed += 1
    return changed
