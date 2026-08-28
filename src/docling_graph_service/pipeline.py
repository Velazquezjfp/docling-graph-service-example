"""The single orchestration point: convert → strip → describe → markdown → chunk → embed → graph.

Conversion failure is a hard error; every later stage degrades gracefully (``degraded.*`` flags,
``errors[]``) so a request always returns Markdown and chunks when the document could be parsed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from docling_core.types.doc import DoclingDocument

from . import __version__
from .chunking import Chunker, infer_captions
from .convert import (
    Converter,
    dehyphenate_table_cells,
    normalize_format,
    normalize_heading_levels,
    strip_page_furniture,
    strip_repeated_furniture,
    to_markdown,
)
from .describe import describe_pictures
from .embed import embed_texts
from .graph import GraphExtractionError, materialize, run_extraction
from .llm_http import LLMClient
from .ontology import CompiledTemplate, Ontology, compile_template, load_ontology, ontology_hash
from .schemas import Degraded, DocumentInfo, PipelineOptions, ProcessResponse
from .settings import Settings

log = logging.getLogger(__name__)


def versions() -> dict[str, str]:
    out = {"docling_graph_service": __version__}
    for name in ("docling", "docling-core", "docling-graph"):
        try:
            out[name.replace("-", "_")] = pkg_version(name)
        except PackageNotFoundError:
            out[name.replace("-", "_")] = "unknown"
    return out


class Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def budget(self, wanted: float, *, floor: float = 15.0) -> float:
        return max(floor, min(wanted, self.remaining()))


@dataclass
class Runtime:
    """Process-wide state: converters, chunk tokenizer, HTTP clients, compiled ontologies."""

    settings: Settings
    converter: Converter = field(init=False)
    chunker: Chunker = field(init=False)
    _clients: dict[tuple[str, str, float], LLMClient] = field(default_factory=dict, init=False)
    _ontologies: dict[str, CompiledTemplate] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        s = self.settings
        self.converter = Converter(s.docling, s.artifacts_path)
        self.chunker = Chunker(s.chunking, s.embedding.text_prefix)

    def warmup(self) -> None:
        self.converter.warmup()

    def close(self) -> None:
        for c in self._clients.values():
            c.close()

    def client(self, base_url: str | None, api_key: str | None, timeout_s: float) -> LLMClient:
        key = (base_url or "", api_key or "", timeout_s)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = LLMClient(base_url or "", api_key, timeout_s)
            return self._clients[key]

    def compiled_ontology(self, ontology_graph: dict[str, Any] | None) -> CompiledTemplate:
        if ontology_graph is not None:
            ont = load_ontology(ontology_graph)
        else:
            path = self.settings.graph.default_ontology_path
            if not path:
                raise ValueError("no ontology_graph in the request and graph.default_ontology_path is unset")
            ont = load_ontology(_resolve_path(path))
        return self._compile(ont)

    def _compile(self, ont: Ontology) -> CompiledTemplate:
        h = ontology_hash(ont)
        with self._lock:
            if h not in self._ontologies:
                self._ontologies[h] = compile_template(ont)
            return self._ontologies[h]

    def self_check(self) -> dict[str, Any]:
        """Startup/health self-check: everything the pipeline needs offline must be loadable."""
        report: dict[str, Any] = {"status": "ok", "checks": {}}

        def check(name: str, fn) -> None:
            try:
                fn()
                report["checks"][name] = "ok"
            except Exception as exc:  # noqa: BLE001
                report["checks"][name] = f"error: {exc}"[:300]
                report["status"] = "degraded"

        check("chunk_tokenizer", lambda: self.chunker.hf_tokenizer.encode("test"))
        check("docling_models", self.converter.warmup)
        check("docling_graph_tokenizer", _check_tiktoken)
        if self.settings.graph.enabled and self.settings.graph.default_ontology_path:
            check("default_ontology", lambda: self.compiled_ontology(None))
        return report


DOCLING_GRAPH_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"  # docling-graph's internal chunker default


def _check_tiktoken() -> None:
    import tiktoken
    from transformers import AutoTokenizer

    tiktoken.get_encoding("cl100k_base").encode("test")
    AutoTokenizer.from_pretrained(DOCLING_GRAPH_TOKENIZER).encode("test")


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    from .settings import config_path

    cfg = config_path()
    if cfg is not None and (cfg.parent / p).exists():
        return cfg.parent / p
    return p


@dataclass
class Outcome:
    response: ProcessResponse
    document: DoclingDocument | None = None


def process(rt: Runtime, content: bytes, name: str, fmt: str | None, options: PipelineOptions,
            ontology_graph: dict[str, Any] | None = None, *, keep_document: bool = False) -> Outcome:
    s = rt.settings
    deadline = Deadline(s.service.request_deadline_s)
    timings: dict[str, float] = {}
    errors: list[str] = []
    warnings: list[str] = []
    degraded = Degraded()
    fmt = normalize_format(fmt, name)
    sha = hashlib.sha256(content).hexdigest()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem) or "document"

    use_ocr = s.docling.ocr if options.ocr_enabled is None else options.ocr_enabled
    use_vlm = s.vlm.enabled if options.vlm_enabled is None else options.vlm_enabled
    use_embed = s.embedding.enabled if options.embedding_enabled is None else options.embedding_enabled
    use_graph = s.graph.enabled if options.graph_enabled is None else options.graph_enabled

    compiled: CompiledTemplate | None = None
    if use_graph:
        compiled = rt.compiled_ontology(ontology_graph)  # invalid ontology -> ValueError before any work

    cache = _ResultCache(s.service.cache_dir)
    result_key = cache.key(sha, options, compiled, s, use_ocr, use_vlm, use_embed, use_graph)
    cached = cache.load_result(result_key)
    if cached is not None and not keep_document:
        cached.cached = True
        return Outcome(cached)

    with tempfile.TemporaryDirectory(prefix="dgs-") as tmp:
        tmpdir = Path(tmp)
        # 1. convert (or reuse the cached DoclingDocument for this pdf + OCR setting)
        t = time.perf_counter()
        doc = cache.load_document(sha, use_ocr)
        if doc is None:
            src = tmpdir / f"{stem}.{fmt}"
            src.write_bytes(content)
            converted = rt.converter.convert(src, ocr=use_ocr)
            doc = converted.document
            warnings.extend(converted.warnings)
            cache.save_document(sha, use_ocr, doc)
        else:
            warnings.append("DoclingDocument loaded from cache")
        timings["convert"] = round(time.perf_counter() - t, 3)

        # 2. structure clean-up: furniture (prefix rule + repetition), heading levels
        moved = strip_page_furniture(doc, s.chunking.strip_line_prefixes)
        moved += strip_repeated_furniture(doc, min_pages=s.chunking.repeated_furniture_min_pages,
                                          band=s.chunking.furniture_band)
        if moved:
            warnings.append(f"{moved} body text item(s) moved to page furniture (header/footer rules)")
        if s.docling.numbered_heading_levels:
            changed = normalize_heading_levels(doc)
            if changed:
                warnings.append(f"{changed} heading level(s) normalized from section numbering")
        if s.docling.dehyphenate_table_cells:
            fixed = dehyphenate_table_cells(doc)
            if fixed:
                warnings.append(f"{fixed} table cell(s) de-hyphenated (PDF line wraps)")
        captions, _ = infer_captions(doc)

        # 3. picture descriptions
        if use_vlm and doc.pictures:
            t = time.perf_counter()
            if deadline.remaining() < 30:
                degraded.vlm = True
                errors.append("vlm: skipped, request deadline nearly exhausted")
            else:
                try:
                    client = rt.client(s.vlm.base_url, s.vlm.api_key, deadline.budget(s.vlm.timeout_s))
                    report = describe_pictures(doc, client, s.vlm, captions=captions)
                    if report.errors:
                        degraded.vlm = True
                        errors.extend(f"vlm: {e}" for e in report.errors)
                    if report.skipped:
                        warnings.append(f"vlm: {report.skipped} picture(s) skipped (no image or below min_area_fraction)")
                except Exception as exc:  # noqa: BLE001
                    degraded.vlm = True
                    errors.append(f"vlm: {exc}")
            timings["describe"] = round(time.perf_counter() - t, 3)

        # 4. markdown + chunks
        t = time.perf_counter()
        markdown = to_markdown(doc)
        chunking = rt.chunker.chunk(doc, sha, max_tokens=options.chunk_max_tokens)
        chunks = chunking.chunks
        warnings.extend(f"chunking: {w}" for w in chunking.warnings)
        timings["chunk"] = round(time.perf_counter() - t, 3)

        # 5. embeddings
        if use_embed and chunks:
            t = time.perf_counter()
            try:
                if deadline.remaining() < 30:
                    raise TimeoutError("request deadline nearly exhausted")
                client = rt.client(s.embedding.base_url, s.embedding.api_key, deadline.budget(s.embedding.timeout_s))
                vectors = embed_texts(client, s.embedding.model, [c.text for c in chunks],
                                      batch_size=s.embedding.batch_size, dim=s.embedding.dim,
                                      prefix=s.embedding.text_prefix)
                for c, v in zip(chunks, vectors, strict=True):
                    c.embedding = v
            except Exception as exc:  # noqa: BLE001
                degraded.embeddings = True
                errors.append(f"embeddings: {exc}")
            timings["embed"] = round(time.perf_counter() - t, 3)

        # 6. graph
        graph = None
        if use_graph and compiled is not None:
            t = time.perf_counter()
            try:
                if deadline.remaining() < 60:
                    raise TimeoutError("request deadline nearly exhausted")
                doc_json = tmpdir / f"{stem}.json"
                doc.save_as_json(doc_json)
                llm = s.llm.model_copy(update={"timeout_s": deadline.budget(
                    s.llm.timeout_s, floor=60.0)})
                ctx = run_extraction(doc_json, compiled, llm, s.graph, contract=options.extraction_contract,
                                     chunk_max_tokens=None)
                graph = materialize(ctx, compiled, chunks, chunking.self_ref_index)
                warnings.extend(f"graph: {w}" for w in graph.meta.get("warnings", []))
            except (GraphExtractionError, Exception) as exc:
                degraded.graph = True
                errors.append(f"graph: {exc}")
                log.exception("graph extraction failed")
            timings["graph"] = round(time.perf_counter() - t, 3)

    response = ProcessResponse(
        document=DocumentInfo(name=name, format=fmt, sha256=sha, pages=len(doc.pages), tables=len(doc.tables),
                              pictures=len(doc.pictures)),
        markdown=markdown,
        chunks=chunks,
        graph=graph,
        degraded=degraded,
        errors=errors,
        warnings=warnings,
        timings_s=timings,
        versions=versions(),
    )
    if not (degraded.vlm or degraded.embeddings or degraded.graph):
        cache.save_result(result_key, response)
    return Outcome(response, doc if keep_document else None)


class _ResultCache:
    """Optional on-disk cache: DoclingDocument per (pdf, ocr) and full results per request key.

    Best effort: a cache that cannot be read or written only logs a warning, never fails a request.
    """

    def __init__(self, cache_dir: str | None):
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning("result cache disabled, cannot create %s: %s", self.dir, exc)
                self.dir = None

    @staticmethod
    def key(sha: str, options: PipelineOptions, compiled: CompiledTemplate | None, s: Settings,
            ocr: bool, vlm: bool, embed: bool, graph: bool) -> str:
        payload = {
            "sha": sha,
            "options": options.model_dump(exclude_none=True),
            "flags": [ocr, vlm, embed, graph],
            "ontology": compiled.schema_hash() if compiled else None,
            "models": [s.llm.model if graph else None, s.vlm.model if vlm else None,
                       s.embedding.model if embed else None],
            "chunking": s.chunking.model_dump(),
            "docling": s.docling.model_dump(),
            "versions": versions(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def load_document(self, sha: str, ocr: bool) -> DoclingDocument | None:
        if not self.dir:
            return None
        p = self.dir / f"{sha}.ocr{int(ocr)}.docling.json"
        if p.exists():
            try:
                return DoclingDocument.load_from_json(p)
            except Exception as exc:  # noqa: BLE001
                log.warning("ignoring unreadable cached document %s: %s", p, exc)
        return None

    def save_document(self, sha: str, ocr: bool, doc: DoclingDocument) -> None:
        if self.dir:
            self._write(self.dir / f"{sha}.ocr{int(ocr)}.docling.json", lambda p: doc.save_as_json(p))

    def load_result(self, key: str) -> ProcessResponse | None:
        if not self.dir:
            return None
        p = self.dir / f"{key}.result.json"
        if p.exists():
            try:
                return ProcessResponse.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log.warning("ignoring unreadable cached result %s: %s", p, exc)
        return None

    def save_result(self, key: str, response: ProcessResponse) -> None:
        if self.dir:
            self._write(self.dir / f"{key}.result.json",
                        lambda p: p.write_text(response.model_dump_json(), encoding="utf-8"))

    @staticmethod
    def _write(path: Path, writer) -> None:
        try:
            writer(path)
        except OSError as exc:
            log.warning("cache write skipped for %s: %s", path.name, exc)
