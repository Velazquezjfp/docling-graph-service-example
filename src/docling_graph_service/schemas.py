"""Public request/response models (the API contract) shared by the API, CLI and pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Polarity = Literal["positive", "negative", "unknown"]


# --------------------------------------------------------------------------- request


class DocumentIn(BaseModel):
    name: str = Field(..., min_length=1, description="File name, e.g. Betriebshandbuch_ZSD.pdf")
    format: str = Field("pdf", description="Extension or MIME type: pdf, docx, pptx, html, md")
    base64_content: str = Field(..., min_length=1)


class PipelineOptions(BaseModel):
    """Per-request overrides. Unknown keys are rejected (422) so typos never pass silently."""

    model_config = ConfigDict(extra="forbid")

    ocr_enabled: bool | None = None
    vlm_enabled: bool | None = None
    embedding_enabled: bool | None = None
    graph_enabled: bool | None = None
    chunk_max_tokens: int | None = Field(None, ge=64, le=8192)
    extraction_contract: Literal["dense", "direct"] | None = None


class ProcessRequest(BaseModel):
    document: DocumentIn
    pipeline_config: PipelineOptions = Field(default_factory=PipelineOptions)
    ontology_graph: dict[str, Any] | None = Field(
        None, description="Ontology document (see GET /v1/ontology-schema); default: configured file"
    )


# --------------------------------------------------------------------------- chunks


class BBox(BaseModel):
    page: int
    l: float
    t: float
    r: float
    b: float
    page_width: float
    page_height: float
    coord_origin: Literal["TOPLEFT"] = "TOPLEFT"


class Chunk(BaseModel):
    chunk_id: str
    text: str = Field(description="Embedded text: heading breadcrumb + body (+ embedding prefix excluded)")
    body_text: str
    heading_breadcrumb: list[str] = Field(default_factory=list)
    heading_level: int = 0
    kind: Literal["text", "table", "picture", "mixed"] = "text"
    caption: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    dom_paths: list[str] = Field(default_factory=list, description="DoclingDocument self_refs")
    bboxes: list[BBox] = Field(default_factory=list)
    token_count: int = 0
    embedding: list[float] | None = None


# --------------------------------------------------------------------------- graph


class NodeProvenance(BaseModel):
    pages: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    dom_paths: list[str] = Field(default_factory=list)
    heading_breadcrumb: list[str] = Field(default_factory=list)
    table_ref: str | None = None
    figure_ref: str | None = None
    match: str | None = Field(None, description="docling-graph match kind: verbatim/observed/reconciled/derived")


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    quote: str | None = None
    provenance: NodeProvenance = Field(default_factory=NodeProvenance)


class EdgeProvenance(BaseModel):
    pages: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    polarity: Polarity = "positive"
    qualifier: str | None = None
    quote: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: EdgeProvenance = Field(default_factory=EdgeProvenance)


class UnresolvedTarget(BaseModel):
    source: str
    type: str
    target_type: str | None
    target: str
    polarity: Polarity
    reason: str = Field("no node with this identity value, label or alias",
                        description="why the link was not materialized (review queue hint)")


class GraphResult(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- response


class Degraded(BaseModel):
    vlm: bool = False
    embeddings: bool = False
    graph: bool = False


class DocumentInfo(BaseModel):
    name: str
    format: str
    sha256: str
    pages: int
    tables: int
    pictures: int


class ProcessResponse(BaseModel):
    document: DocumentInfo
    markdown: str
    chunks: list[Chunk]
    graph: GraphResult | None
    degraded: Degraded = Field(default_factory=Degraded)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings_s: dict[str, float] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    cached: bool = False


class Capabilities(BaseModel):
    service: str
    version: str
    versions: dict[str, str]
    ocr_engines: list[str]
    ocr_languages: list[str]
    formats: list[str]
    models: dict[str, dict[str, Any]]
    chunking: dict[str, Any]
    features: dict[str, bool]
    limits: dict[str, Any]
    endpoint_status: dict[str, str]
