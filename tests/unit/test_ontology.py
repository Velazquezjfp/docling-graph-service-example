import json

import pytest
from conftest import ONTOLOGY_PATH
from pydantic import ValidationError

from docling_graph_service.ontology import (
    CATALOG_EDGE,
    ROOT_NAME,
    Ontology,
    check_report,
    compile_template,
    load_ontology,
    ontology_hash,
    ontology_json_schema,
)

MINIMAL = {
    "meta": {"name": "t", "version": "1"},
    "enums": {"Kind": ["a", "b"]},
    "datatypes": {"ticket_id": {"python": "str", "pattern": r"^T-\d+$"}, "date": {"python": "datetime.date"}},
    "classes": {
        "System": {"label_de": "System", "identity": {"keys": ["label"]},
                   "fields": [{"name": "kind", "type": "Kind", "required": True}]},
        "Incident": {"description": "Störung", "identity": {"keys": ["ticket_id"]},
                     "fields": [{"name": "ticket_id", "type": "ticket_id", "required": True},
                                {"name": "date", "type": "date"}]},
    },
    "relations": [
        {"name": "AFFECTS", "source": "Incident", "target": "System",
         "properties": [{"name": "level", "type": "int"}]},
        {"name": "PARTNER", "source": "Incident", "target": ["Incident", "System"], "symmetric": True},
        {"name": "MENTIONED_IN", "source": "NodeBase", "target": "System"},
    ],
}


@pytest.fixture(scope="module")
def real():
    return load_ontology(ONTOLOGY_PATH)


def test_real_ontology_compiles(real):
    compiled = compile_template(real)
    assert len(compiled.classes) == 26
    assert len(compiled.links) == 31  # 32 relations minus MENTIONED_IN
    assert compiled.skipped_relations == [("MENTIONED_IN", "source NodeBase: derived from provenance, not extracted")]
    assert len(compiled.catalog_fields) == 26
    schema = compiled.root.model_json_schema()
    assert "document_id" in schema["properties"]
    assert compiled.classes["Person"].identity_keys == ["display_name"]  # ontology "label" is renamed
    assert compiled.classes["StartupStep"].identity_keys == ["order", "display_name"]
    assert "depends_on" in compiled.classes["System"].link_fields
    assert compiled.classes["System"].link_fields["depends_on"] == "DEPENDS_ON"
    assert set(compiled.links["DEPENDS_ON"].property_types) == {"dependency_kind", "criticality", "failure_effect", "bridging"}


def test_config_keys_survive_in_model_config(real):
    compiled = compile_template(real)
    assert compiled.root.model_config.get("graph_id_fields") == ["document_id"]
    assert compiled.classes["Host"].model.model_config.get("graph_id_fields") == ["hostname"]
    assert compiled.links["AFFECTS"].model.model_config.get("is_entity") is False
    field = compiled.root.model_fields[compiled.catalog_fields["Incident"]]
    assert field.json_schema_extra == {"edge_label": CATALOG_EDGE}


def test_only_identity_fields_required(real):
    compiled = compile_template(real)
    incident = compiled.classes["Incident"].model
    assert incident.model_fields["ticket_id"].is_required()
    assert not incident.model_fields["title"].is_required()
    assert not incident.model_fields["display_name"].is_required()
    assert "label" not in incident.model_fields  # reserved by docling-graph
    person = compiled.classes["Person"].model
    assert person.model_fields["display_name"].is_required()
    link = compiled.links["DEPENDS_ON"].model
    assert link.model_fields["target"].is_required()
    assert not link.model_fields["dependency_kind"].is_required()


def test_docstring_discriminating_sentence_first(real):
    compiled = compile_template(real)
    doc = compiled.classes["ImpactStatement"].model.__doc__
    assert doc.startswith("Auswirkungsaussage.")
    assert "Hinweise:" in doc
    assert "Identität: event, affected_system" in doc


def test_minimal_ontology():
    ont = Ontology.model_validate(MINIMAL)
    compiled = compile_template(ont)
    assert set(compiled.classes) == {"System", "Incident"}
    assert set(compiled.links) == {"AFFECTS", "PARTNER"}
    link = compiled.links["AFFECTS"].model
    assert link.model_fields["target_type"].default == "System"
    inst = compiled.classes["Incident"].model(ticket_id="T-1", affects=[{"target": "VPP", "polarity": "negative"}])
    assert inst.affects[0].target == "VPP"
    # enums/datatypes become plain str with hints in the description
    kind = compiled.classes["System"].model.model_fields["kind"]
    assert kind.annotation == (str | None) and "a, b" in kind.description  # optional: not an identity key
    assert compiled.classes["Incident"].field_types["ticket_id"].pattern == r"^T-\d+$"
    assert compiled.classes["Incident"].field_types["date"].kind == "date"
    report = check_report(compiled)
    assert "MENTIONED_IN" in report and "LLM schema" in report


@pytest.mark.parametrize("mutate, message", [
    (lambda o: o["classes"]["System"].pop("identity"), "identity"),
    (lambda o: o["classes"]["System"]["identity"].__setitem__("keys", ["nope"]), "'nope' is not a field"),
    (lambda o: o["classes"]["System"]["identity"].__setitem__("keys", []), "at least 1"),
    (lambda o: o["relations"].append({"name": "AFFECTS", "source": "Incident", "target": "System"}), "duplicate relation"),
    (lambda o: o["relations"].append({"name": "BAD", "source": "Incident", "target": "Ghost"}), "unknown class 'Ghost'"),
    (lambda o: o["relations"].append({"name": "lower", "source": "Incident", "target": "System"}), "pattern"),
    (lambda o: o["classes"]["System"].pop("label_de"), "label_de"),
    (lambda o: o.pop("meta"), "meta"),
    (lambda o: o["classes"]["System"]["fields"].append({"name": "aliases", "type": "str"}), "implicit"),
])
def test_contract_errors(mutate, message):
    data = json.loads(json.dumps(MINIMAL))
    mutate(data)
    with pytest.raises(ValidationError) as exc:
        Ontology.model_validate(data)
    assert message in str(exc.value)


def test_unknown_keys_ignored_and_hash_stable(real):
    data = json.loads(json.dumps(MINIMAL))
    data["competency_questions"] = [{"id": "Q1"}]
    data["classes"]["System"]["table_hints"] = ["x"]
    a = Ontology.model_validate(data)
    b = Ontology.model_validate(MINIMAL)
    assert ontology_hash(a) == ontology_hash(b)
    assert ontology_hash(real) != ontology_hash(a)


def test_json_schema_is_the_contract():
    schema = ontology_json_schema()
    assert set(schema["required"]) == {"meta", "classes"}
    assert "relations" in schema["properties"]
    assert ROOT_NAME not in schema["properties"]
