"""HybridChunker with provenance: XLM-R token budget, Markdown tables, caption + header on every
table part, tables/pictures as their own chunks, hard token cap (split, never truncate), TOPLEFT
bounding boxes, self_ref index."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    RefItem,
    TableItem,
    TextItem,
)
from transformers import AutoTokenizer

from .schemas import BBox, Chunk
from .settings import ChunkingSettings

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")
_TABLE_CAPTION = re.compile(r"^(Tabelle|Tab\.|Table)\s*\d+", re.IGNORECASE)
_FIGURE_CAPTION = re.compile(r"^(Abbildung|Abb\.|Bild|Figure|Fig\.)\s*\d+", re.IGNORECASE)


class _MarkdownTableProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(image_placeholder="", escape_underscores=False, escape_html=False,
                                  compact_tables=True),
        )


@dataclass
class ChunkingResult:
    chunks: list[Chunk]
    self_ref_index: dict[str, list[str]] = field(default_factory=dict)
    captions: dict[str, str] = field(default_factory=dict)  # self_ref -> caption (incl. inferred)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Draft:
    body: str
    headings: list[str]
    items: list[DocItem]
    kind: str
    caption: str | None


def infer_captions(doc: DoclingDocument) -> tuple[dict[str, str], int]:
    """Caption per table/picture self_ref, completing what docling linked:

    * a table without caption directly after a table with the same header row continues that
      table (page break) and inherits its caption;
    * otherwise the nearest text item before/after that looks like "Tabelle N:" / "Abbildung N:".
    Returns (captions, number of inferred captions).
    """
    captions: dict[str, str] = {}
    inferred = 0
    items = [it for it, _ in doc.iterate_items(with_groups=False)]
    pos = {it.self_ref: i for i, it in enumerate(items)}
    prev_table: TableItem | None = None
    for table in doc.tables:
        cap = table.caption_text(doc).strip()
        if not cap and prev_table is not None and _header(table) == _header(prev_table) and _header(table):
            cap = captions.get(prev_table.self_ref, "")
            if cap:
                inferred += 1
        if not cap:
            cap = _nearby_caption(items, pos.get(table.self_ref), _TABLE_CAPTION)
            inferred += bool(cap)
        if cap:
            captions[table.self_ref] = cap
        prev_table = table
    for pic in doc.pictures:
        cap = pic.caption_text(doc).strip()
        if not cap:
            cap = _nearby_caption(items, pos.get(pic.self_ref), _FIGURE_CAPTION)
            inferred += bool(cap)
        if cap:
            captions[pic.self_ref] = cap
    return captions, inferred


def _header(table: TableItem) -> tuple[str, ...]:
    return tuple(" ".join(c.text.split()) for c in table.data.table_cells if c.start_row_offset_idx == 0)


def _nearby_caption(items: list[DocItem], idx: int | None, pattern: re.Pattern[str]) -> str:
    if idx is None:
        return ""
    for j in (idx + 1, idx - 1, idx + 2):
        if 0 <= j < len(items) and isinstance(items[j], TextItem):
            text = " ".join(items[j].text.split())
            if pattern.match(text) and len(text) < 200:
                return text
    return ""


class Chunker:
    """Built once per process (tokenizer load is slow); ``chunk()`` is stateless per document."""

    def __init__(self, cfg: ChunkingSettings, embedding_prefix: str = ""):
        self.cfg = cfg
        self.prefix = embedding_prefix
        self.hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        self.prefix_tokens = self._count(embedding_prefix, special=False) if embedding_prefix else 0
        # 2 = special tokens ([CLS]/[SEP] or <s></s>) added by the embedding model
        self.budget = cfg.max_tokens - self.prefix_tokens - 2
        if self.budget < 32:
            raise ValueError("chunking.max_tokens too small for the embedding prefix")
        self._chunkers: dict[int, HybridChunker] = {}

    # ------------------------------------------------------------------ public

    def chunk(self, doc: DoclingDocument, doc_hash: str, *, max_tokens: int | None = None) -> ChunkingResult:
        limit = (max_tokens or self.cfg.max_tokens) - self.prefix_tokens - 2
        chunker = self._chunker(limit)
        warnings: list[str] = []
        captions, inferred = infer_captions(doc)
        if inferred:
            warnings.append(f"{inferred} table/figure caption(s) inferred from neighbouring text or page continuation")

        drafts: list[_Draft] = []
        for dl_chunk in chunker.chunk(dl_doc=doc):
            body = dl_chunk.text
            if not body.strip():
                continue  # e.g. table-of-contents headings without content
            items = [_resolve(it, doc) for it in (dl_chunk.meta.doc_items or [])]
            headings = [h for h in (dl_chunk.meta.headings or []) if h]
            content = [it for it in items if it.label != DocItemLabel.CAPTION]  # captions ride along with their table
            kind = _kind(content)
            caption = captions.get(content[0].self_ref) if len(content) == 1 and kind in ("table", "picture") else None
            if kind == "table" and caption and not _starts_with(body, caption):
                body = f"{caption}\n\n{body}"  # SPEC §5.1: caption + header on every table part
            drafts.extend(self._enforce_cap(_Draft(body, headings, items, kind, caption), limit, warnings))
        if self.cfg.merge_peers:
            drafts = self._merge_peers(drafts, limit)

        chunks: list[Chunk] = []
        index: dict[str, list[str]] = {}
        for i, d in enumerate(drafts):
            chunk_id = f"{doc_hash[:12]}-{i:04d}"
            text = _contextualize(d.headings, d.body)
            pages = sorted({p.page_no for it in d.items for p in (it.prov or [])})
            refs = []
            for it in d.items:
                if it.self_ref not in refs:
                    refs.append(it.self_ref)
            for r in refs:
                index.setdefault(r, []).append(chunk_id)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=text,
                body_text=d.body,
                heading_breadcrumb=d.headings,
                heading_level=len(d.headings),
                kind=d.kind,  # type: ignore[arg-type]
                caption=d.caption,
                page_numbers=pages,
                dom_paths=refs,
                bboxes=_bboxes(d.items, doc),
                token_count=self._count(self.prefix + text, special=True),
            ))
        return ChunkingResult(chunks, index, captions, warnings)

    # ------------------------------------------------------------------ internals

    def _chunker(self, limit: int) -> HybridChunker:
        if limit not in self._chunkers:
            self._chunkers[limit] = HybridChunker(
                tokenizer=HuggingFaceTokenizer(tokenizer=self.hf_tokenizer, max_tokens=limit),
                merge_peers=False,  # we merge text peers ourselves so tables/pictures stay whole
                repeat_table_header=True,
                serializer_provider=_MarkdownTableProvider(),
            )
        return self._chunkers[limit]

    def _count(self, text: str, *, special: bool) -> int:
        return len(self.hf_tokenizer.encode(text, add_special_tokens=special))

    def _fits(self, headings: list[str], body: str, limit: int) -> bool:
        return self._count(_contextualize(headings, body), special=False) <= limit

    def _merge_peers(self, drafts: list[_Draft], limit: int) -> list[_Draft]:
        """Merge adjacent text chunks with identical headings while they fit the budget."""
        out: list[_Draft] = []
        for d in drafts:
            prev = out[-1] if out else None
            if (prev is not None and prev.kind == "text" and d.kind == "text" and prev.headings == d.headings
                    and self._fits(d.headings, f"{prev.body}\n{d.body}", limit)):
                prev.body = f"{prev.body}\n{d.body}"
                prev.items = prev.items + d.items
                continue
            out.append(d)
        return out

    def _enforce_cap(self, draft: _Draft, limit: int, warnings: list[str]) -> list[_Draft]:
        """Split an over-budget chunk on row/line, then sentence, then word boundaries."""
        if self._fits(draft.headings, draft.body, limit):
            return [draft]
        head, lines = _split_head(draft)
        parts: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                parts.append("\n".join(current))
                current.clear()

        for line in lines:
            if not self._fits(draft.headings, head + "\n".join([*current, line]), limit):
                flush()
                if not self._fits(draft.headings, head + line, limit):
                    parts.extend(self._split_line(line, head, draft.headings, limit))
                    continue
            current.append(line)
        flush()
        warnings.append(
            f"chunk over {limit} tokens ({draft.kind}: {draft.caption or (draft.headings[-1] if draft.headings else 'no heading')}) "
            f"split into {len(parts)} parts"
        )
        return [_Draft(head + p, draft.headings, draft.items, draft.kind, draft.caption) for p in parts]

    def _split_line(self, line: str, head: str, headings: list[str], limit: int) -> list[str]:
        """A single line/row larger than the budget: sentence boundaries, then the largest word prefix that fits."""
        out: list[str] = []
        current = ""
        for unit in _SENTENCE_SPLIT.split(line) or [line]:
            candidate = f"{current} {unit}".strip() if current else unit
            if self._fits(headings, head + candidate, limit):
                current = candidate
                continue
            if current:
                out.append(current)
            current = unit
            while current and not self._fits(headings, head + current, limit):
                words = current.split(" ")
                lo, hi = 1, len(words)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._fits(headings, head + " ".join(words[:mid]), limit):
                        lo = mid
                    else:
                        hi = mid - 1
                out.append(" ".join(words[:lo]))
                current = " ".join(words[lo:])
        if current:
            out.append(current)
        return out


# ---------------------------------------------------------------------- helpers


def _resolve(item: DocItem, doc: DoclingDocument) -> DocItem:
    """Chunk meta carries copies of the items; fetch the real document item by self_ref."""
    try:
        return RefItem(cref=item.self_ref).resolve(doc)
    except Exception:  # noqa: BLE001
        return item


def _contextualize(headings: list[str], body: str) -> str:
    return "\n".join([*headings, body]) if headings else body


def _kind(items: list[DocItem]) -> str:
    kinds = {"table" if isinstance(it, TableItem) else "picture" if isinstance(it, PictureItem) else "text" for it in items}
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed" if kinds else "text"


def _starts_with(body: str, caption: str) -> bool:
    return " ".join(body.split()).startswith(" ".join(caption.split()))


def _split_head(draft: _Draft) -> tuple[str, list[str]]:
    """For tables keep caption + header row + separator on every part; otherwise split plain lines."""
    lines = draft.body.split("\n")
    if draft.kind == "table":
        first_row = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|")), None)
        if first_row is not None and len(lines) > first_row + 2:
            head = "\n".join(lines[: first_row + 2]) + "\n"
            return head, [line for line in lines[first_row + 2:] if line.strip()]
    return "", [line for line in lines if line.strip()]


def _bboxes(items: list[DocItem], doc: DoclingDocument) -> list[BBox]:
    out: list[BBox] = []
    for it in items:
        for prov in it.prov or []:
            page = doc.pages.get(prov.page_no)
            if page is None or not page.size.height:
                continue
            bb = prov.bbox.to_top_left_origin(page.size.height)
            out.append(BBox(page=prov.page_no, l=bb.l, t=bb.t, r=bb.r, b=bb.b,
                            page_width=page.size.width, page_height=page.size.height))
    return out
