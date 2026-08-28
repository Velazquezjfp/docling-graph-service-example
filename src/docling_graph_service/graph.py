"""Graph extraction: run docling-graph on the converted DoclingDocument and materialize the
service's graph (nodes with provenance, edges with polarity/qualifier/properties).

docling-graph creates nodes, binds provenance and dedupes identities. The compiled template
carries relations as scalar link components (see ``ontology.py``); this module resolves the link
targets against the extracted node set and emits edges. Unresolved targets are reported, never
materialized as stub nodes (SPEC §4.4).
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from docling_graph import PipelineConfig, run_pipeline
from docling_graph.exceptions import DoclingGraphError
from pydantic import BaseModel

from .normalize import norm_key, normalize_polarity, normalize_value
from .ontology import (
    LABEL_FIELD,
    NODE_BASE,
    ROOT_NAME,
    TEMPLATE_IMPLICIT_FIELDS,
    CompiledClass,
    CompiledTemplate,
)
from .schemas import (
    Chunk,
    EdgeProvenance,
    GraphEdge,
    GraphNode,
    GraphResult,
    NodeProvenance,
    UnresolvedTarget,
)
from .settings import GraphSettings, LLMSettings

log = logging.getLogger(__name__)

class GraphExtractionError(RuntimeError):
    pass


def pipeline_config(document_json: Path, compiled: CompiledTemplate, llm: LLMSettings, gcfg: GraphSettings, *,
                    contract: str | None = None, chunk_max_tokens: int | None = None) -> PipelineConfig:
    """The exact docling-graph configuration (kept separate so a unit test can assert every key exists)."""
    return PipelineConfig(**pipeline_kwargs(document_json, compiled, llm, gcfg, contract=contract,
                                            chunk_max_tokens=chunk_max_tokens))


def pipeline_kwargs(document_json: Path, compiled: CompiledTemplate, llm: LLMSettings, gcfg: GraphSettings, *,
                    contract: str | None = None, chunk_max_tokens: int | None = None) -> dict[str, Any]:
    return {
        "source": str(document_json),
        "template": compiled.root,
        "backend": "llm",
        "inference": "remote",
        "provider_override": "openai",
        "model_override": f"openai/{llm.model}",
        "processing_mode": "many-to-one",
        "extraction_contract": contract or gcfg.extraction_contract,
        "use_chunking": True,
        "chunk_max_tokens": chunk_max_tokens or gcfg.chunk_max_tokens,
        "parallel_workers": llm.parallel_workers,
        "dense_dedupe": gcfg.dense_dedupe,
        "provenance": gcfg.provenance,
        "structured_output": llm.structured_output,
        "dump_to_disk": False,
        "gc_collect": False,
        "llm_overrides": {
            "connection": {"base_url": llm.base_url, "api_key": llm.api_key or "none"},
            "generation": {"temperature": llm.temperature, "max_tokens": llm.max_output_tokens},
            "reliability": {"timeout_s": int(llm.timeout_s), "max_retries": llm.max_retries},
            "context_limit": llm.context_limit,
            "max_output_tokens": llm.max_output_tokens,
        },
    }


@dataclass
class ExtractionContext:
    """What ``materialize`` needs from a docling-graph run (also constructible in unit tests)."""

    roots: list[BaseModel]
    graph: Any  # networkx.DiGraph
    node_id: Any  # callable(model_instance) -> str
    graph_meta: dict[str, Any] = field(default_factory=dict)


def run_extraction(document_json: Path, compiled: CompiledTemplate, llm: LLMSettings, gcfg: GraphSettings, *,
                   contract: str | None = None, chunk_max_tokens: int | None = None) -> ExtractionContext:
    cfg = pipeline_config(document_json, compiled, llm, gcfg, contract=contract, chunk_max_tokens=chunk_max_tokens)
    try:
        ctx = run_pipeline(cfg)
    except DoclingGraphError as exc:
        details = getattr(exc, "details", None) or {}
        reason = details.get("reason") or details.get("error") or str(exc)
        raise GraphExtractionError(f"docling-graph {details.get('stage', 'pipeline')}: {reason}") from exc
    if ctx.knowledge_graph is None or not ctx.extracted_models:
        raise GraphExtractionError("docling-graph returned no graph")
    return ExtractionContext(
        roots=list(ctx.extracted_models),
        graph=ctx.knowledge_graph,
        node_id=ctx.node_registry.get_node_id,
        graph_meta=dict(ctx.knowledge_graph.graph),
    )


# ============================================================================ materialization


@dataclass
class _Resolver:
    """identity value / label / alias -> node id, per class."""

    compiled: CompiledTemplate
    by_class: dict[str, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))

    def _key(self, cls: str, value: str) -> str:
        strip = self.compiled.ontology.classes[cls].identity.normalize == "strip_titles_then_casefold" if cls in self.compiled.ontology.classes else False
        return norm_key(value, strip_titles=strip)

    def add(self, cls: str, value: str | None, node_id: str) -> None:
        if value:
            self.by_class[cls].setdefault(self._key(cls, value), node_id)

    def resolve(self, classes: list[str], value: str) -> str | None:
        for cls in classes:
            if cls not in self.compiled.classes:
                continue
            hit = self.by_class.get(cls, {}).get(self._key(cls, value))
            if hit:
                return hit
        return None


def materialize(ctx: ExtractionContext, compiled: CompiledTemplate, chunks: list[Chunk],
                self_ref_index: dict[str, list[str]]) -> GraphResult:
    t0 = time.perf_counter()
    G = ctx.graph
    warnings: list[str] = []
    meta: dict[str, Any] = {
        "docling_graph_version": pkg_version("docling-graph"),
        "template_schema_hash": compiled.schema_hash(),
        "ontology": {"name": compiled.ontology.meta.name, "version": compiled.ontology.meta.version},
        "skipped_relations": [n for n, _ in compiled.skipped_relations],
    }
    for key in ("alias_reconciliation", "closed_catalog_drops", "demoted_nodes", "empty_identity_nodes"):
        if key in ctx.graph_meta:
            meta[key] = ctx.graph_meta[key]

    # -- 1. instances from the root catalogs, grouped by docling-graph node id
    instances: dict[str, list[BaseModel]] = defaultdict(list)
    for root in ctx.roots:
        for fname in compiled.catalog_fields.values():
            for inst in getattr(root, fname, None) or []:
                instances[ctx.node_id(inst)].append(inst)

    # -- 2. nodes from the knowledge graph
    nodes: dict[str, GraphNode] = {}
    resolver = _Resolver(compiled)
    value_problems: list[dict[str, Any]] = []
    enum_violations: list[dict[str, Any]] = []
    identity_violations: list[dict[str, Any]] = []
    dropped_empty: list[str] = []
    merged_into: dict[str, str] = {}  # node id merged away by docling-graph -> surviving node id
    for nid, attrs in G.nodes(data=True):
        cls_name = str(attrs.get("__class__") or "")
        if cls_name == ROOT_NAME:
            continue
        cc = compiled.classes.get(cls_name)
        if cc is None:
            warnings.append(f"node {nid}: unknown class {cls_name!r} ignored")
            continue
        node, problems = _build_node(str(nid), attrs, cc, self_ref_index, chunks)
        if node is None:
            dropped_empty.append(str(nid))
            continue
        for p in problems:
            (enum_violations if p["kind"] == "enum"
             else identity_violations if p["field"] in cc.spec.identity.keys
             else value_problems).append(p)
        nodes[node.id] = node
        for entry in attrs.get("merged_aliases") or []:
            if isinstance(entry, dict) and entry.get("id"):
                merged_into[str(entry["id"])] = node.id
        for key in cc.identity_keys:
            resolver.add(cls_name, _as_text(attrs.get(key)), node.id)
        resolver.add(cls_name, node.label, node.id)
        for alias in node.aliases:
            resolver.add(cls_name, alias, node.id)

    # -- 3. edges from link components
    edges: dict[tuple[str, str, str, str], GraphEdge] = {}
    unresolved: list[UnresolvedTarget] = []
    polarity_notes: list[str] = []
    dropped_by_converter = 0
    triggers = compiled.ontology.negation_triggers
    quote_index = _QuoteIndex(chunks)
    for nid, insts in instances.items():
        cc = compiled.class_of(insts[0])
        if cc is None:
            continue
        source_id = nid if nid in nodes else merged_into.get(nid) or _survivor(insts[0], cc, resolver)
        if source_id is None:
            dropped_by_converter += 1
            continue
        for inst in insts:
            for fname, rel_name in cc.link_fields.items():
                link_spec = compiled.links[rel_name]
                for link in getattr(inst, fname, None) or []:
                    target_text = _as_text(getattr(link, "target", None))
                    if not target_text:
                        continue
                    declared = _as_text(getattr(link, "target_type", None))
                    candidates = ([declared] if declared in link_spec.targets else []) + \
                        [t for t in link_spec.targets if t != declared]
                    if NODE_BASE in link_spec.relation.target:
                        candidates = ([declared] if declared in compiled.classes else []) + \
                            [c for c in compiled.classes if c != declared]
                    quote = _as_text(getattr(link, "quote", None))
                    qualifier = _as_text(getattr(link, "qualifier", None))
                    polarity, note = normalize_polarity(_as_text(getattr(link, "polarity", None)),
                                                        " ".join(x for x in (quote, qualifier) if x), triggers)
                    if note:
                        polarity_notes.append(f"{rel_name} {nodes[source_id].label} -> {target_text}: {note}")
                    target_id = resolver.resolve(candidates, target_text)
                    if target_id is None:
                        other = [c for c in compiled.classes if c not in candidates
                                 and resolver.resolve([c], target_text)]
                        reason = (f"node exists only as {'/'.join(other)}, not allowed as {rel_name} target"
                                  if other else "no node with this identity value, label or alias")
                        unresolved.append(UnresolvedTarget(source=source_id, type=rel_name, target_type=declared,
                                                           target=target_text, polarity=polarity, reason=reason))
                        continue
                    props: dict[str, Any] = {}
                    for pname, pinfo in link_spec.property_types.items():
                        raw = getattr(link, pname, None)
                        if raw in (None, [], ""):
                            continue
                        val, probs = normalize_value(raw, pinfo)
                        props[pname] = val
                        for p in probs:
                            value_problems.append({"kind": pinfo.kind, "edge": rel_name, "field": pname,
                                                   "value": raw, "problem": p})
                    key = (source_id, target_id, rel_name, polarity)
                    if key in edges:
                        existing = edges[key]
                        if quote and not existing.quote:
                            existing.quote = quote
                        if qualifier and not existing.qualifier:
                            existing.qualifier = qualifier
                        for k, v in props.items():
                            existing.properties.setdefault(k, v)
                        continue
                    prov_chunks = quote_index.find(quote) if quote else []
                    if not prov_chunks:
                        prov_chunks = list(nodes[source_id].provenance.chunk_ids)
                    edges[key] = GraphEdge(
                        source=source_id, target=target_id, type=rel_name, polarity=polarity,
                        qualifier=qualifier, quote=quote, properties=props,
                        provenance=EdgeProvenance(pages=_pages_of(prov_chunks, chunks), chunk_ids=prov_chunks),
                    )

    conflicts = _conflicts(edges)
    meta.update({
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_type": dict(Counter(n.type for n in nodes.values())),
        "edges_by_type": dict(Counter(e.type for e in edges.values())),
        "edges_by_polarity": dict(Counter(e.polarity for e in edges.values())),
        "unresolved_targets": [u.model_dump() for u in unresolved],
        "conflicts": conflicts,
        "dropped_empty_identity": dropped_empty,
        "dropped_by_converter": dropped_by_converter,
        "enum_violations": enum_violations,
        "identity_pattern_violations": identity_violations,
        "value_problems": value_problems,
        "polarity_notes": polarity_notes,
        "warnings": warnings,
        "materialize_s": round(time.perf_counter() - t0, 3),
    })
    return GraphResult(nodes=list(nodes.values()), edges=list(edges.values()), meta=meta)


# ---------------------------------------------------------------------------- helpers


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _build_node(nid: str, attrs: dict[str, Any], cc: CompiledClass, self_ref_index: dict[str, list[str]],
                chunks: list[Chunk]) -> tuple[GraphNode | None, list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    attributes: dict[str, Any] = {}
    normalized: dict[str, Any] = {}
    for fname, info in cc.field_types.items():
        if fname in TEMPLATE_IMPLICIT_FIELDS or fname not in attrs:
            continue
        raw = attrs.get(fname)
        if raw in (None, "", []):
            continue
        val, probs = normalize_value(raw, info)
        normalized[fname] = val
        attributes[cc.ontology_name(fname)] = val
        for p in probs:
            problems.append({"kind": info.kind, "node": nid, "field": cc.ontology_name(fname), "value": raw,
                             "problem": p})
    identity = cc.identity_value({**attrs, **normalized})
    label = _as_text(attrs.get(LABEL_FIELD)) or identity
    if label is None:
        return None, problems
    aliases: list[str] = []
    for raw in list(attrs.get("aliases") or []) + list(attrs.get("merged_aliases") or []):
        text = raw.get("label") or raw.get("alias") or raw.get("name") if isinstance(raw, dict) else raw
        text = _as_text(text)
        if text and text != label and text not in aliases:
            aliases.append(text)
    prov = attrs.get("__provenance__") or {}
    refs = [r for r in (prov.get("refs") or []) if isinstance(r, str)]
    chunk_ids: list[str] = []
    for r in refs:
        for cid in self_ref_index.get(r, []):
            if cid not in chunk_ids:
                chunk_ids.append(cid)
    by_id = {c.chunk_id: c for c in chunks}
    breadcrumb: list[str] = []
    table_ref = figure_ref = None
    for cid in chunk_ids:
        c = by_id.get(cid)
        if c is None:
            continue
        if not breadcrumb and c.heading_breadcrumb:
            breadcrumb = c.heading_breadcrumb
        if c.kind == "table" and c.caption and not table_ref:
            table_ref = c.caption.split(":")[0].strip()
        if c.kind == "picture" and c.caption and not figure_ref:
            figure_ref = c.caption.split(":")[0].strip()
    pages = sorted({int(p) for p in (prov.get("pages") or []) if isinstance(p, int)}) or \
        _pages_of(chunk_ids, chunks)
    node = GraphNode(
        id=nid, type=cc.name, label=label, attributes=attributes, aliases=aliases,
        quote=_as_text(attrs.get("quote")),
        provenance=NodeProvenance(pages=pages, chunk_ids=chunk_ids, dom_paths=refs, heading_breadcrumb=breadcrumb,
                                  table_ref=table_ref, figure_ref=figure_ref, match=prov.get("match")),
    )
    return node, problems


def _survivor(inst: BaseModel, cc: CompiledClass, resolver: _Resolver) -> str | None:
    """A catalog instance whose node id was merged away: find the surviving node via its identity/label."""
    values = inst.model_dump()
    for candidate in (cc.identity_value(values), _as_text(values.get(LABEL_FIELD))):
        if candidate:
            hit = resolver.resolve([cc.name], candidate)
            if hit:
                return hit
    return None


def _pages_of(chunk_ids: list[str], chunks: list[Chunk]) -> list[int]:
    by_id = {c.chunk_id: c for c in chunks}
    return sorted({p for cid in chunk_ids for p in by_id[cid].page_numbers if cid in by_id})


class _QuoteIndex:
    def __init__(self, chunks: list[Chunk]):
        self._texts = [(c.chunk_id, norm_key(c.body_text)) for c in chunks]

    def find(self, quote: str) -> list[str]:
        q = norm_key(quote)
        if len(q) < 12:
            return []
        hits = [cid for cid, text in self._texts if q in text]
        if hits:
            return hits
        # tolerate small LLM edits: match on the longest 60-char window
        window = q[: 60] if len(q) > 60 else q
        return [cid for cid, text in self._texts if window in text]


def _conflicts(edges: dict[tuple[str, str, str, str], GraphEdge]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (s, t, r, p) in edges:
        seen[(s, t, r)].add(p)
    return [{"source": s, "target": t, "type": r, "polarities": sorted(ps)}
            for (s, t, r), ps in seen.items() if {"positive", "negative"} <= ps]
