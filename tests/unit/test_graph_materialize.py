"""Materializer on a synthetic docling-graph result: duplicates, negative link, unresolved target,
title stripping, merged-away ids, polarity conflicts, enum/date normalization, provenance."""

import networkx as nx
import pytest

from docling_graph_service.graph import ExtractionContext, materialize
from docling_graph_service.ontology import ROOT_NAME, Ontology, compile_template
from docling_graph_service.schemas import BBox, Chunk

ONT = {
    "meta": {"name": "t", "version": "1"},
    "enums": {"Criticality": ["kritisch", "hoch"]},
    "datatypes": {"ticket_id": {"python": "str", "pattern": r"^T-\d+$"}, "date": {"python": "datetime.date"}},
    "classes": {
        "System": {"label_de": "System", "identity": {"keys": ["label"], "normalize": "casefold"}},
        "Person": {"label_de": "Person", "identity": {"keys": ["label"], "normalize": "strip_titles_then_casefold"}},
        "Incident": {"label_de": "Störung", "identity": {"keys": ["ticket_id"]},
                     "fields": [{"name": "ticket_id", "type": "ticket_id", "required": True},
                                {"name": "date", "type": "date"}]},
    },
    "relations": [
        {"name": "DEPENDS_ON", "source": "System", "target": "System",
         "properties": [{"name": "criticality", "type": "Criticality"}]},
        {"name": "RESPONSIBLE_FOR", "source": "Person", "target": "NodeBase"},
        {"name": "PARTNER_TICKET", "source": "Incident", "target": "Incident", "symmetric": True},
        {"name": "AFFECTS", "source": "Incident", "target": "System"},
    ],
}


def _node_id(inst):
    cls = type(inst).__name__
    key = getattr(inst, "ticket_id", None) or inst.display_name
    return f"{cls}_{key.lower().replace(' ', '').replace('.', '')}"


@pytest.fixture
def world():
    compiled = compile_template(Ontology.model_validate(ONT))
    S, P, I = (compiled.classes[c].model for c in ("System", "Person", "Incident"))
    Dep = compiled.links["DEPENDS_ON"].model
    Resp = compiled.links["RESPONSIBLE_FOR"].model
    Part = compiled.links["PARTNER_TICKET"].model
    Aff = compiled.links["AFFECTS"].model
    vpp = S(display_name="VPP", depends_on=[
        Dep(target="Vault", polarity="negative", quote="Keystores liegen lokal, VPP bleibt unberührt", criticality="Hoch"),
        Dep(target="Ghost System"),  # unresolved
        Dep(target="IAM", quote="Die Anmeldung läuft über IAM"),  # positive, no polarity given
        Dep(target="IAM", quote="Ohne IAM keine Anmeldung"),  # None polarity + negation trigger -> unknown
    ])
    vpp_dup = S(display_name="VPP", depends_on=[Dep(target="Vault", polarity="negative")])  # duplicate instance
    vpp_portal = S(display_name="VPP Portal", depends_on=[Dep(target="Vault", polarity="positive")])  # merged into VPP
    vault, iam = S(display_name="Vault"), S(display_name="IAM")
    reuss = P(display_name="Dr. Annika Reuß", responsible_for=[Resp(target="vpp", target_type="System")])
    t1 = I(ticket_id="T-1", date="18.08.2025", partner_ticket=[Part(target="T-2")], affects=[Aff(target="VPP")])
    t2 = I(ticket_id="T-2", partner_ticket=[Part(target="T-1")])
    root = compiled.root(document_id="DOC", systems=[vpp, vpp_dup, vpp_portal, vault, iam], persons=[reuss],
                         incidents=[t1, t2])
    G = nx.DiGraph()
    G.add_node("root", __class__=ROOT_NAME, label=ROOT_NAME)
    for inst in (vpp, vault, iam, reuss, t1, t2):
        attrs = {k: v for k, v in inst.model_dump().items() if v not in (None, [], "")}
        attrs = {k: v for k, v in attrs.items() if not isinstance(v, list) or k == "aliases"}
        attrs.update({"__class__": type(inst).__name__,
                      "__provenance__": {"pages": [3], "refs": ["#/texts/1"], "match": "verbatim"}})
        G.add_node(_node_id(inst), **attrs)
    G.nodes["System_vpp"]["merged_aliases"] = [{"id": "System_vppportal", "label": "VPP Portal"}]
    G.nodes["System_vpp"]["aliases"] = ["VPP Portal"]
    chunks = [Chunk(chunk_id="c1", text="x", body_text="Die Anmeldung läuft über IAM. Ohne IAM keine Anmeldung.",
                    page_numbers=[3], dom_paths=["#/texts/1"],
                    bboxes=[BBox(page=3, l=0, t=0, r=1, b=1, page_width=10, page_height=10)])]
    ctx = ExtractionContext(roots=[root], graph=G, node_id=_node_id, graph_meta={"alias_reconciliation": {"merged": 1}})
    return compiled, ctx, chunks


def test_nodes(world):
    compiled, ctx, chunks = world
    result = materialize(ctx, compiled, chunks, {"#/texts/1": ["c1"]})
    by_id = {n.id: n for n in result.nodes}
    assert set(by_id) == {"System_vpp", "System_vault", "System_iam", "Person_drannikareuß", "Incident_t-1", "Incident_t-2"}
    assert by_id["Incident_t-1"].label == "T-1"  # label falls back to the identity value
    assert by_id["Incident_t-1"].attributes["date"] == "2025-08-18"
    assert by_id["System_vpp"].aliases == ["VPP Portal"]
    assert by_id["System_vpp"].provenance.pages == [3]
    assert by_id["System_vpp"].provenance.chunk_ids == ["c1"]
    assert by_id["System_vpp"].provenance.match == "verbatim"
    assert result.meta["alias_reconciliation"] == {"merged": 1}
    assert result.meta["nodes_by_type"] == {"System": 3, "Person": 1, "Incident": 2}


def test_edges(world):
    compiled, ctx, chunks = world
    result = materialize(ctx, compiled, chunks, {"#/texts/1": ["c1"]})
    edges = {(e.source, e.target, e.type, e.polarity): e for e in result.edges}
    # negative link kept, duplicates merged, both polarities for VPP->Vault survive and are reported
    assert ("System_vpp", "System_vault", "DEPENDS_ON", "negative") in edges
    assert ("System_vpp", "System_vault", "DEPENDS_ON", "positive") in edges  # from the merged-away 'VPP Portal'
    assert result.meta["conflicts"] == [{"source": "System_vpp", "target": "System_vault", "type": "DEPENDS_ON",
                                         "polarities": ["negative", "positive"]}]
    neg = edges[("System_vpp", "System_vault", "DEPENDS_ON", "negative")]
    assert neg.properties == {"criticality": "hoch"}
    assert neg.quote.startswith("Keystores")
    # None polarity: positive without trigger, unknown with trigger
    assert ("System_vpp", "System_iam", "DEPENDS_ON", "positive") in edges
    assert ("System_vpp", "System_iam", "DEPENDS_ON", "unknown") in edges
    assert edges[("System_vpp", "System_iam", "DEPENDS_ON", "unknown")].provenance.chunk_ids == ["c1"]
    assert any("negation trigger" in n for n in result.meta["polarity_notes"])
    # Dr. title stripped for Person identity; NodeBase target resolved by declared type, casefold
    assert ("Person_drannikareuß", "System_vpp", "RESPONSIBLE_FOR", "positive") in edges
    # symmetric relation emitted per direction found
    assert ("Incident_t-1", "Incident_t-2", "PARTNER_TICKET", "positive") in edges
    assert ("Incident_t-2", "Incident_t-1", "PARTNER_TICKET", "positive") in edges
    assert ("Incident_t-1", "System_vpp", "AFFECTS", "positive") in edges
    # unresolved target reported, not materialized
    assert [u["target"] for u in result.meta["unresolved_targets"]] == ["Ghost System"]
    assert result.meta["unresolved_targets"][0]["reason"].startswith("no node")
    assert not any(n.label == "Ghost System" for n in result.nodes)
    assert result.meta["edge_count"] == len(result.edges) == 8
    assert result.meta["dropped_by_converter"] == 0
    assert all(e.type in compiled.relation_names for e in result.edges)
