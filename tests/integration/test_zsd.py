"""End-to-end acceptance on Betriebshandbuch_ZSD.pdf (BHB-PLT-0007) with the real ontology.

Run with ``DGS_INTEGRATION=1`` (needs the configured LLM/VLM/embedding endpoints). Conversion is cached in
``service.cache_dir`` when set, so repeated runs take minutes, not tens of minutes. Set
``DGS_INTEGRATION_URL=http://host:8080`` to run the same assertions against a deployed service over HTTP
instead of calling the pipeline in-process. Ground truth was
extracted glyph-exact from the PDF (see PLAN.md, Appendix A); graph assertions are thresholds
because LLM output varies.
"""

import json
import os
import re
from pathlib import Path

import pytest
from conftest import ONTOLOGY_PATH, ZSD_PDF

from docling_graph_service.pipeline import Runtime, process
from docling_graph_service.schemas import PipelineOptions
from docling_graph_service.settings import get_settings

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("DGS_INTEGRATION") != "1", reason="set DGS_INTEGRATION=1")]

OUT = Path(__file__).resolve().parents[2] / "out" / "zsd-integration"


@pytest.fixture(scope="module")
def result():
    url = os.environ.get("DGS_INTEGRATION_URL")
    if url:
        response = _via_http(url)
    else:
        settings = get_settings()
        assert settings.graph.default_ontology_path, "graph.default_ontology_path must point at ontology.yaml"
        rt = Runtime(settings)
        response = process(rt, ZSD_PDF.read_bytes(), ZSD_PDF.name, None, PipelineOptions(), None).response
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "response.json").write_text(response.model_dump_json(indent=1), encoding="utf-8")
    return response


def _via_http(url: str):
    import base64

    import httpx
    import yaml

    from docling_graph_service.schemas import ProcessResponse

    body = {"document": {"name": ZSD_PDF.name, "format": "pdf",
                         "base64_content": base64.b64encode(ZSD_PDF.read_bytes()).decode()},
            "pipeline_config": {},
            "ontology_graph": yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))}
    with httpx.Client(base_url=url, timeout=httpx.Timeout(3600, connect=10), trust_env=False) as client:
        r = client.post("/v1/process", json=body)
    assert r.status_code == 200, r.text[:500]
    return ProcessResponse.model_validate(r.json())


def _tables(resp):
    return [c for c in resp.chunks if c.kind == "table"]


# ----------------------------------------------------------------- conversion + chunking (NEXT-STEPS §1)


def test_document_shape(result):
    assert result.document.pages == 29
    assert result.document.pictures == 3
    assert result.document.tables >= 15  # page-spanning tables arrive as one TableItem per page


def test_no_page_furniture_anywhere(result):
    text = result.markdown + "\n".join(c.text for c in result.chunks)
    assert "TESTDOKUMENT" not in text
    assert not re.search(r"Seite \d+ von 29", text)
    assert "BHB-PLT-0007 · Version 2.3 · Stand" not in text


def test_all_15_captions_and_table_parts(result):
    captions = {c.caption.split(":")[0] for c in _tables(result) if c.caption}
    assert captions == {f"Tabelle {n}" for n in range(1, 16)}
    for c in _tables(result):
        if not c.caption:
            continue
        lines = [line for line in c.body_text.split("\n") if line.strip()]
        assert lines[0].startswith(c.caption.split(":")[0]), lines[0]  # caption on every part
        assert lines[1].startswith("|") and lines[2].lstrip("| ").startswith("-"), lines[1:3]  # header + separator


def test_tabelle_6_konsumentenmatrix(result):
    parts = [c for c in _tables(result) if c.caption and c.caption.startswith("Tabelle 6:")]
    assert len(parts) == 1 and parts[0].page_numbers == [11]
    rows = [line for line in parts[0].body_text.split("\n") if line.startswith("|") and not line.lstrip("| ").startswith("-")]
    assert rows[0].count("|") - 1 == 6 and len(rows) - 1 == 6
    body = parts[0].body_text
    for cell in ("VPP Portal", "dd-gateway", "es-grafana", "caas-console", "obs-vpp", "kv/vpp/monitor",
                 "bavd-issuing-ca-3", "Sonja Wiechert", "BHB-VRF-0207"):
        assert cell in body, cell


def test_tabelle_8_page_spanning_with_literal_keine(result):
    parts = [c for c in _tables(result) if c.caption and c.caption.startswith("Tabelle 8:")]
    assert [c.page_numbers for c in parts] == [[18], [19]]
    body = "\n".join(c.body_text for c in parts)
    rows = [line for line in body.split("\n") if line.startswith("|") and not line.lstrip("| ").startswith("-")]
    assert rows[0].count("|") - 1 == 5
    assert len(rows) - len(parts) == 5  # five events, header not counted
    keine = sum(1 for line in rows for cell in line.split("|") if cell.strip() == "keine")
    assert keine >= 4
    assert "unberührt" in body


def test_identifiers_intact(result):
    text = "\n".join(c.text for c in result.chunks)
    for ident in ("ZSDSUP-0247", "BHB-PLT-0007", "SOP-ZSD-06", "FW-ZSD-004", "kv/vpp/monitor", "CAASUP-0351"):
        assert ident in text, ident


def test_breadcrumbs_and_token_budget(result):
    settings = get_settings()
    assert all(c.token_count <= settings.chunking.max_tokens for c in result.chunks)
    assert all(c.body_text.strip() for c in result.chunks)
    sop = next(c for c in result.chunks if any("SOP-ZSD-02" in h for h in c.heading_breadcrumb))
    assert sop.heading_breadcrumb == ["5 Betrieb", "5.3 Standardabläufe", "5.3.1 Keycloak rollierend neu starten (SOP-ZSD-02)"]
    assert sop.heading_level == 3
    deep = sum(len(c.heading_breadcrumb) >= 2 for c in result.chunks)
    assert deep / len(result.chunks) > 0.8
    assert all(c.bboxes and c.bboxes[0].coord_origin == "TOPLEFT" for c in result.chunks)
    assert all(c.dom_paths for c in result.chunks)


# ----------------------------------------------------------------- embeddings / VLM


def test_embeddings(result):
    assert not result.degraded.embeddings, result.errors
    dim = get_settings().embedding.dim
    assert all(c.embedding is not None and len(c.embedding) == dim for c in result.chunks)


def test_vlm_descriptions(result):
    assert not result.degraded.vlm, result.errors
    pics = [c for c in result.chunks if c.kind == "picture"]
    assert len(pics) == 3
    assert all(len(c.body_text) > 40 for c in pics)
    assert result.markdown.count("Abbildung ") >= 3
    assert "ZSD" in result.markdown


# ----------------------------------------------------------------- graph (thresholds)


def test_graph_shape(result):
    assert not result.degraded.graph, result.errors
    g = result.graph
    assert g is not None
    by_type = {}
    for n in g.nodes:
        by_type.setdefault(n.type, []).append(n)
    assert any(n.type == "Document" and "BHB-PLT-0007" in (n.attributes.get("doc_id"), n.label) for n in g.nodes)
    assert any(n.type == "System" and "Sicherheitsdienste" in n.label for n in g.nodes)
    clients = {n.label for n in by_type.get("IdentityClient", [])}
    assert len(clients & {"vpp-portal", "dd-gateway", "es-grafana", "caas-console", "obs-vpp"}) >= 4
    tickets = {n.attributes.get("ticket_id") or n.label for n in by_type.get("Incident", [])}
    assert len([t for t in tickets if t.startswith("ZSDSUP-")]) >= 6
    sops = {n.attributes.get("sop_id") or n.label for n in by_type.get("Procedure", [])}
    assert len(sops & {f"SOP-ZSD-0{i}" for i in range(1, 7)}) >= 5
    persons = {re.sub(r"^Dr\.\s*", "", n.label) for n in by_type.get("Person", [])}
    assert {"Kai Ostermann", "Sabine Wollmer", "Marcel Ebert"} <= persons
    impacts = by_type.get("ImpactStatement", [])
    assert len(impacts) >= 10
    assert any(n.attributes.get("severity") == "keine" for n in impacts)
    assert all(n.provenance.pages for n in g.nodes), [n.label for n in g.nodes if not n.provenance.pages][:5]


def test_graph_edges(result):
    g = result.graph
    relations = {r["name"] if isinstance(r, dict) else r for r in _relation_names()}
    assert all(e.type in relations for e in g.edges)
    keys = [(e.source, e.target, e.type, e.polarity) for e in g.edges]
    assert len(keys) == len(set(keys))
    partner = [e for e in g.edges if e.type == "PARTNER_TICKET"]
    partner_unresolved = [u for u in g.meta["unresolved_targets"] if u["type"] == "PARTNER_TICKET"]
    assert len(partner) + len(partner_unresolved) >= 4  # partner tickets live in other documents
    assert any(e.polarity == "negative" for e in g.edges), g.meta["edges_by_polarity"]
    assert all(e.provenance.chunk_ids for e in g.edges)
    # unresolved = review queue: either nothing matches, or only a node of a type the relation forbids
    allowed = {r["name"]: r for r in _relations()}
    for u in g.meta["unresolved_targets"]:
        assert u["reason"]
        targets = allowed[u["type"]]["target"]
        targets = [targets] if isinstance(targets, str) else targets
        if "NodeBase" not in targets:
            assert not any(n.type in targets and n.label.casefold() == u["target"].casefold() for n in g.nodes), u


def _relations():
    import yaml

    return yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))["relations"]


def _relation_names():
    return [r["name"] for r in _relations()]


def test_results_summary(result):
    """Not an assertion: print the numbers that go into README."""
    g = result.graph
    summary = {
        "pages": result.document.pages, "tables": result.document.tables, "pictures": result.document.pictures,
        "chunks": len(result.chunks), "table_chunks": len(_tables(result)),
        "max_tokens": max(c.token_count for c in result.chunks),
        "timings_s": result.timings_s, "degraded": result.degraded.model_dump(), "errors": result.errors,
        "nodes_by_type": g.meta["nodes_by_type"] if g else None, "edges_by_type": g.meta["edges_by_type"] if g else None,
        "edges_by_polarity": g.meta["edges_by_polarity"] if g else None,
        "unresolved": len(g.meta["unresolved_targets"]) if g else None, "conflicts": g.meta["conflicts"] if g else None,
    }
    print("\nZSD SUMMARY " + json.dumps(summary, ensure_ascii=False, indent=1))
