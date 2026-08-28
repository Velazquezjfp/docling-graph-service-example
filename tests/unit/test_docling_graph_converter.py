"""The compiled template through docling-graph's *real* GraphConverter (no LLM): guards against
reserved-attribute collisions (``label``/``id``/``type``) and checks the materializer end to end."""

from conftest import ONTOLOGY_PATH
from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.converters.node_id_registry import NodeIDRegistry

from docling_graph_service.graph import ExtractionContext, materialize
from docling_graph_service.ontology import ROOT_NAME, compile_template, load_ontology


def test_real_converter_roundtrip():
    compiled = compile_template(load_ontology(ONTOLOGY_PATH))
    C = compiled.classes
    root = compiled.root(
        document_id="BHB-PLT-0007",
        documents=[C["Document"].model(doc_id="BHB-PLT-0007", title="ZSD", documents=[{"target": "Zentrale Sicherheitsdienste"}])],
        systems=[
            C["System"].model(display_name="Zentrale Sicherheitsdienste", kind="basisdienst"),
            C["System"].model(display_name="VPP", kind="fachverfahren",
                              depends_on=[{"target": "Vault", "polarity": "negative", "quote": "Keystores liegen lokal"}],
                              tenant_of=[{"target": "CaaS-Plattform", "polarity": "negative"}]),
            C["System"].model(display_name="Vault"),  # no label-optional fields -> label attr must survive
        ],
        incidents=[
            C["Incident"].model(ticket_id="ZSDSUP-0247", title="Vault versiegelt", date="02.07.2026", severity="Sev-1",
                                partner_ticket=[{"target": "CAASUP-0351"}], affects=[{"target": "Vault", "target_type": "System"}]),
            C["Incident"].model(ticket_id="ZSDSUP-0247", duration="38 min"),  # duplicate identity -> enrich
        ],
        persons=[C["Person"].model(display_name="Dr. Annika Reuß", phone_ext="1300",
                                   responsible_for=[{"target": "Zentrale Sicherheitsdienste", "target_type": "System"}])],
        namespaces=[C["Namespace"].model(display_name="vpp-prod")],  # ontology field named `label`
    )
    registry = NodeIDRegistry()
    graph, meta = GraphConverter(registry=registry, alias_llm_fn=None).pydantic_list_to_graph([root])
    assert meta.node_count >= 7
    assert all(isinstance(k, str) for k in meta.node_types)  # this is what crashed with a `label` field
    classes = {d["__class__"] for _, d in graph.nodes(data=True)}
    assert {ROOT_NAME, "Document", "System", "Incident", "Person", "Namespace"} <= classes
    for _, d in graph.nodes(data=True):
        assert d["label"] == d["__class__"]  # docling-graph's own label is intact

    ctx = ExtractionContext(roots=[root], graph=graph, node_id=registry.get_node_id, graph_meta=dict(graph.graph))
    result = materialize(ctx, compiled, chunks=[], self_ref_index={})
    by_label = {(n.type, n.label): n for n in result.nodes}
    assert ("System", "Vault") in by_label and ("Namespace", "vpp-prod") in by_label
    assert ("Incident", "ZSDSUP-0247") in by_label  # label falls back to the identity value
    inc = by_label[("Incident", "ZSDSUP-0247")]
    assert inc.attributes["date"] == "2026-07-02" and inc.attributes["title"] == "Vault versiegelt"
    assert inc.attributes["duration"] == "38 min"  # duplicate instance enriched the node
    assert inc.attributes.get("severity") == "Sev-1" and result.meta["enum_violations"]  # kept + flagged
    edges = {(e.source, e.target, e.type, e.polarity) for e in result.edges}
    vpp, vault = by_label[("System", "VPP")].id, by_label[("System", "Vault")].id
    assert (vpp, vault, "DEPENDS_ON", "negative") in edges
    assert (inc.id, vault, "AFFECTS", "positive") in edges
    assert (by_label[("Person", "Dr. Annika Reuß")].id, by_label[("System", "Zentrale Sicherheitsdienste")].id,
            "RESPONSIBLE_FOR", "positive") in edges
    assert (by_label[("Document", "BHB-PLT-0007")].id, by_label[("System", "Zentrale Sicherheitsdienste")].id,
            "DOCUMENTS", "positive") in edges
    unresolved = {(u["type"], u["target"]): u["reason"] for u in result.meta["unresolved_targets"]}
    assert set(unresolved) == {("TENANT_OF", "CaaS-Plattform"), ("PARTNER_TICKET", "CAASUP-0351")}
    assert unresolved[("PARTNER_TICKET", "CAASUP-0351")].startswith("no node")
    assert not any(n.type == ROOT_NAME for n in result.nodes)
