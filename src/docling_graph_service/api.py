"""FastAPI front: /healthz, /v1/capabilities, /v1/ontology-schema, /v1/process (one job at a time)."""

from __future__ import annotations

import base64
import binascii
import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import __version__
from .convert import FORMATS, ConversionFailed, UnsupportedFormat, normalize_format
from .ontology import load_ontology, ontology_json_schema
from .pipeline import Runtime, process, versions
from .schemas import Capabilities, ProcessRequest, ProcessResponse
from .settings import get_settings, redact

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.service.log_level)
    rt = Runtime(settings)
    await anyio.to_thread.run_sync(rt.warmup)
    app.state.runtime = rt
    app.state.limiter = anyio.CapacityLimiter(1)
    yield
    rt.close()


app = FastAPI(title="docling-graph-service", version=__version__, lifespan=lifespan)


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@app.get("/healthz")
async def healthz(request: Request):
    report = await anyio.to_thread.run_sync(_runtime(request).self_check)
    return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)


@app.get("/v1/ontology-schema")
async def ontology_schema():
    return ontology_json_schema()


@app.get("/v1/capabilities", response_model=Capabilities)
async def capabilities(request: Request):
    rt = _runtime(request)
    s = rt.settings
    status = {}
    for name, sect in (("llm", s.llm), ("vlm", s.vlm), ("embedding", s.embedding)):
        if getattr(sect, "enabled", True):
            client = rt.client(sect.base_url, sect.api_key, 10.0)
            status[name] = await anyio.to_thread.run_sync(client.probe)
    return Capabilities(
        service="docling-graph-service",
        version=__version__,
        versions=versions(),
        ocr_engines=["easyocr"] if s.docling.ocr else [],
        ocr_languages=list(s.docling.ocr_langs),
        formats=list(FORMATS),
        models={
            "llm": {"model": s.llm.model, **redact(s.llm.base_url, s.llm.api_key), "enabled": s.graph.enabled,
                    "structured_output": s.llm.structured_output, "extraction_contract": s.graph.extraction_contract},
            "vlm": {"model": s.vlm.model, **redact(s.vlm.base_url or "", s.vlm.api_key), "enabled": s.vlm.enabled},
            "embedding": {"model": s.embedding.model, **redact(s.embedding.base_url or "", s.embedding.api_key),
                          "enabled": s.embedding.enabled, "dim": s.embedding.dim,
                          "text_prefix": s.embedding.text_prefix},
        },
        chunking={"algorithm": "docling HybridChunker", "tokenizer": s.chunking.tokenizer,
                  "max_tokens": s.chunking.max_tokens, "merge_peers": s.chunking.merge_peers,
                  "table_format": "markdown", "caption_and_header_on_every_table_part": True},
        features={"ocr": s.docling.ocr, "vlm_descriptions": s.vlm.enabled, "embeddings": s.embedding.enabled,
                  "graph": s.graph.enabled, "heading_hierarchy": s.docling.heading_hierarchy,
                  "result_cache": bool(s.service.cache_dir)},
        limits={"max_upload_mb": s.service.max_upload_mb, "request_deadline_s": s.service.request_deadline_s,
                "concurrent_jobs": 1, "images_scale": s.docling.images_scale},
        endpoint_status=status,
    )


@app.post("/v1/process", response_model=ProcessResponse)
async def process_document(request: Request, body: ProcessRequest):
    rt = _runtime(request)
    s = rt.settings
    b64 = body.document.base64_content
    if len(b64) * 3 / 4 > s.service.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"document larger than {s.service.max_upload_mb} MB")
    try:
        content = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"base64_content is not valid base64: {exc}") from exc
    if not content:
        raise HTTPException(422, "document is empty")
    try:
        fmt = normalize_format(body.document.format, body.document.name)
    except UnsupportedFormat as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.ontology_graph is not None:
        try:
            load_ontology(body.ontology_graph)
        except ValidationError as exc:
            raise HTTPException(422, {"ontology_graph": exc.errors(include_url=False)}) from exc

    limiter: anyio.CapacityLimiter = request.app.state.limiter
    try:
        limiter.acquire_nowait()
    except anyio.WouldBlock:
        raise HTTPException(503, "a document is already being processed; retry later",
                            headers={"Retry-After": "30"}) from None
    try:
        outcome = await anyio.to_thread.run_sync(
            lambda: process(rt, content, body.document.name, fmt, body.pipeline_config, body.ontology_graph)
        )
    except ConversionFailed as exc:
        raise HTTPException(500, str(exc)) from exc
    except ValueError as exc:  # e.g. unset default ontology
        raise HTTPException(422, str(exc)) from exc
    finally:
        limiter.release()
    return outcome.response
