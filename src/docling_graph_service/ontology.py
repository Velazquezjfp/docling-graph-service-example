"""Ontology input contract and its compilation into a docling-graph template.

The contract is the structure of ``ontology.yaml``: ``meta``, optional ``enums``/``datatypes``,
``classes`` (each with ``identity.keys``) and ``relations``. Unknown keys are ignored, so
documentation-only sections (``mixins``, ``competency_questions`` …) pass through untouched.

Compilation produces:

* one **entity** class per ontology class (``graph_id_fields`` = identity keys),
* one **scalar link component** per relation (``is_entity=False``), added as a list field to every
  source class — the LLM writes the target's identity value as text, the service resolves it
  against the extracted node set afterwards (no nested entities: docling-graph's dense catalog
  keeps only the first model of a Union and prunes recursive branches),
* a synthetic ``ExtractionRoot`` with one catalog list per class.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

NODE_BASE = "NodeBase"
CATALOG_EDGE = "__CATALOG__"
ROOT_NAME = "ExtractionRoot"
IMPLICIT_NODE_FIELDS = ("label", "aliases", "description", "quote")  # ontology-level names
# docling-graph writes template fields over its own node attributes ``id``/``label``/``type``;
# these ontology field names are therefore renamed inside the compiled template.
RESERVED_RENAMES = {"label": "display_name", "id": "id_value", "type": "type_value"}
LABEL_FIELD = RESERVED_RENAMES["label"]
TEMPLATE_IMPLICIT_FIELDS = (LABEL_FIELD, "aliases", "description", "quote")
LINK_BASE_FIELDS = ("target_type", "target", "polarity", "qualifier", "quote")
DEFAULT_NEGATION_TRIGGERS = (
    "kein", "keine", "nicht", "ohne", "unberührt", "bewusst nicht genutzt",
    "nicht angebunden", "kein Mandant", "entfällt",
)
# docling-graph shows only the head of a class docstring to the discovery phase.
GUIDE_HEAD_CHARS = 240

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PY_SCALARS: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


# ============================================================================ contract models


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OntologyMeta(_Lenient):
    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


class DatatypeSpec(_Lenient):
    python: str = Field("str", description="str | int | float | bool | datetime.date (others -> str)")
    pattern: str | None = None
    examples: list[Any] | None = None
    note: str | None = None

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str | None) -> str | None:
        if v is not None:
            re.compile(v)
        return v


class FieldSpec(_Lenient):
    name: str = Field(..., pattern=_IDENT.pattern)
    type: str = "str"
    required: bool = False
    many: bool = False
    description: str | None = None
    examples: list[Any] | None = None
    default: Any = None


class IdentitySpec(_Lenient):
    keys: list[str] = Field(..., min_length=1, max_length=2)
    normalize: str | None = None
    scope: str | None = None
    secondary_keys: list[str] = Field(default_factory=list)


class ClassSpec(_Lenient):
    label_de: str | None = None
    description: str | None = None
    identity: IdentitySpec
    cues_de: list[str] = Field(default_factory=list)
    fields: list[FieldSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ClassSpec:
        if not (self.label_de or self.description):
            raise ValueError("needs 'label_de' or 'description'")
        names = [f.name for f in self.fields]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate field names: {dupes}")
        clashes = sorted(set(names) & set(IMPLICIT_NODE_FIELDS) - {"label"})
        if clashes:
            raise ValueError(f"fields {clashes} are implicit on every class and cannot be redeclared")
        by_name = {f.name: f for f in self.fields}
        for key in self.identity.keys:
            if key != "label" and key not in by_name:
                raise ValueError(f"identity.keys: '{key}' is not a field of this class")
            if key in by_name and by_name[key].many:
                raise ValueError(f"identity.keys: '{key}' is a list field; identity keys must be scalar")
        return self


class PropertySpec(_Lenient):
    name: str = Field(..., pattern=_IDENT.pattern)
    type: str = "str"
    required: bool = False
    many: bool = False
    description: str | None = None
    examples: list[Any] | None = None


class RelationSpec(_Lenient):
    name: str = Field(..., pattern=r"^[A-Z][A-Z0-9_]*$")
    source: list[str] = Field(..., min_length=1)
    target: list[str] = Field(..., min_length=1)
    label_de: str | None = None
    description: str | None = None
    cues_de: list[str] = Field(default_factory=list)
    properties: list[PropertySpec] = Field(default_factory=list)
    symmetric: bool = False

    @field_validator("source", "target", mode="before")
    @classmethod
    def _as_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check(self) -> RelationSpec:
        names = [p.name for p in self.properties]
        clashes = sorted(set(names) & set(LINK_BASE_FIELDS))
        if clashes:
            raise ValueError(f"properties {clashes} clash with the implicit edge fields")
        return self


class Ontology(_Lenient):
    """The standard ontology input: this model *is* the contract (``GET /v1/ontology-schema``)."""

    meta: OntologyMeta
    enums: dict[str, list[str]] = Field(default_factory=dict)
    datatypes: dict[str, DatatypeSpec] = Field(default_factory=dict)
    classes: dict[str, ClassSpec] = Field(..., min_length=1)
    relations: list[RelationSpec] = Field(default_factory=list)
    extraction_rule_negation: str | None = None
    extraction: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Ontology:
        for name in self.classes:
            if not _IDENT.match(name) or name == NODE_BASE or name == ROOT_NAME:
                raise ValueError(f"classes: '{name}' is not a valid class name")
        seen: set[str] = set()
        for i, rel in enumerate(self.relations):
            if rel.name in seen:
                raise ValueError(f"relations[{i}]: duplicate relation name '{rel.name}'")
            seen.add(rel.name)
            for side in ("source", "target"):
                for cls_name in getattr(rel, side):
                    if cls_name != NODE_BASE and cls_name not in self.classes:
                        raise ValueError(
                            f"relations[{i}] ({rel.name}).{side}: unknown class '{cls_name}'"
                        )
        return self

    @property
    def negation_triggers(self) -> list[str]:
        if self.extraction_rule_negation:
            found = re.findall(r'"([^"]+)"', self.extraction_rule_negation)
            if found:
                return found
        return list(DEFAULT_NEGATION_TRIGGERS)


def load_ontology(source: str | Path | dict[str, Any]) -> Ontology:
    if isinstance(source, dict):
        return Ontology.model_validate(source)
    with open(source, encoding="utf-8") as fh:
        return Ontology.model_validate(yaml.safe_load(fh))


def ontology_json_schema() -> dict[str, Any]:
    return Ontology.model_json_schema()


def ontology_hash(ont: Ontology) -> str:
    payload = ont.model_dump(mode="json", exclude_none=True)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ============================================================================ type resolution


@dataclass(frozen=True)
class TypeInfo:
    """What the materializer needs to normalize/validate a value."""

    kind: Literal["str", "int", "float", "bool", "date", "percent", "enum", "reference"]
    many: bool = False
    enum_values: tuple[str, ...] = ()
    pattern: str | None = None
    reference_to: str | None = None

    @property
    def python(self) -> type:
        return {"int": int, "float": float, "percent": float, "bool": bool}.get(self.kind, str)


def resolve_type(type_name: str, ont: Ontology, *, many: bool = False) -> tuple[TypeInfo, str]:
    """Map an ontology type name to (TypeInfo, description hint). Unknown types become str."""
    if type_name in ont.enums:
        values = tuple(ont.enums[type_name])
        return TypeInfo("enum", many, enum_values=values), f"Erlaubte Werte: {', '.join(values)}."
    if type_name in ont.classes:
        return (
            TypeInfo("reference", many, reference_to=type_name),
            f"Identitätswert eines Knotens der Klasse {type_name}, wie im Text geschrieben.",
        )
    if type_name == "reference":
        return TypeInfo("reference", many), "Identitätswert des referenzierten Knotens, wie im Text geschrieben."
    dt = ont.datatypes.get(type_name)
    if dt is None:
        info = TypeInfo(type_name if type_name in _PY_SCALARS else "str", many)  # type: ignore[arg-type]
        return info, ""
    py = dt.python.strip()
    hints: list[str] = []
    if py.endswith("date"):
        kind: str = "date"
        hints.append("Datum wie im Text (z. B. 22.07.2026).")
    elif type_name == "percent":
        kind = "percent"
        hints.append("Prozentwert wie im Text (z. B. 99,95 %).")
    elif py in _PY_SCALARS:
        kind = py
    else:
        kind = "str"
    if dt.pattern:
        hints.append(f"Format: {dt.pattern}")
    if dt.note and kind == "str":
        hints.append(dt.note)
    return TypeInfo(kind, many, pattern=dt.pattern), " ".join(hints)  # type: ignore[arg-type]


# ============================================================================ compiled template


@dataclass
class LinkSpec:
    relation: RelationSpec
    field_name: str
    model: type[BaseModel]
    sources: list[str]
    targets: list[str]
    property_types: dict[str, TypeInfo]


@dataclass
class CompiledClass:
    """Field names here are *template* names (``display_name`` for the ontology's ``label``)."""

    name: str
    spec: ClassSpec
    model: type[BaseModel]
    identity_keys: list[str]
    field_types: dict[str, TypeInfo]
    link_fields: dict[str, str] = field(default_factory=dict)  # field name -> relation name
    field_names: dict[str, str] = field(default_factory=dict)  # template field -> ontology field

    def ontology_name(self, template_field: str) -> str:
        return self.field_names.get(template_field, template_field)

    def identity_value(self, values: dict[str, Any]) -> str | None:
        parts = [str(values.get(k)).strip() for k in self.identity_keys if values.get(k) not in (None, "")]
        if len(parts) != len(self.identity_keys):
            return None
        return " | ".join(parts)


@dataclass
class CompiledTemplate:
    ontology: Ontology
    root: type[BaseModel]
    classes: dict[str, CompiledClass]
    links: dict[str, LinkSpec]
    catalog_fields: dict[str, str]  # class name -> root field name
    skipped_relations: list[tuple[str, str]]
    warnings: list[str]

    @property
    def relation_names(self) -> set[str]:
        return {r.name for r in self.ontology.relations}

    def class_of(self, instance: BaseModel) -> CompiledClass | None:
        return self.classes.get(type(instance).__name__)

    def link_for_field(self, class_name: str, field_name: str) -> LinkSpec | None:
        rel = self.classes[class_name].link_fields.get(field_name)
        return self.links.get(rel) if rel else None

    def schema_json(self) -> str:
        return json.dumps(self.root.model_json_schema(), ensure_ascii=False)

    def schema_hash(self) -> str:
        return hashlib.sha256(self.schema_json().encode()).hexdigest()[:16]


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _camel(upper_snake: str) -> str:
    return "".join(p.capitalize() for p in upper_snake.lower().split("_"))


def _plural(snake: str) -> str:
    if snake.endswith("y") and snake[-2:-1] not in "aeiou":
        return snake[:-1] + "ies"
    if snake.endswith(("s", "x", "ch", "sh")):
        return snake + "es"
    return snake + "s"


def _docstring(head: str, description: str | None, cues: list[str], tail: str = "") -> str:
    parts = [head.strip().rstrip(".") + "."]
    if description:
        parts.append(" ".join(description.split()).rstrip(".") + ".")
    if cues:
        parts.append("Hinweise: " + ", ".join(cues) + ".")
    if tail:
        parts.append(tail)
    return " ".join(parts)


def _field(py: Any, *, required: bool, many: bool, description: str, examples: list[Any] | None,
           default: Any = None) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {"description": description}
    if examples:
        kwargs["examples"] = [str(e) for e in examples]
    if many:
        return list[py], Field(default_factory=list, **kwargs)  # type: ignore[valid-type]
    if required:
        return py, Field(..., **kwargs)
    return py | None, Field(default, **kwargs)


def _build_link_model(rel: RelationSpec, ont: Ontology, all_classes: list[str]) -> LinkSpec:
    targets = all_classes if NODE_BASE in rel.target else list(rel.target)
    triggers = ", ".join(f'"{t}"' for t in ont.negation_triggers)
    fields: dict[str, tuple[Any, Any]] = {}
    if len(targets) == 1:
        fields["target_type"] = (
            str | None,
            Field(targets[0], description=f"Klasse des Ziels, immer '{targets[0]}'."),
        )
    else:
        shown = "beliebige Klasse" if NODE_BASE in rel.target else ", ".join(targets)
        fields["target_type"] = (str | None, Field(None, description=f"Klasse des Ziels: {shown}."))
    fields["target"] = (
        str,
        Field(..., description=(
            "Identitätswert des Zielknotens genau wie im Text (Label, Ticket-ID, Hostname, "
            "SOP-ID, Pfad …). Das Ziel muss auch in seiner eigenen Klassenliste erscheinen."
        )),
    )
    fields["polarity"] = (
        str | None,
        Field(None, description=(
            "'positive' wenn die Beziehung besteht; 'negative' wenn der Text sie ausdrücklich "
            f"verneint (Auslöser: {triggers}). Verneinte Aussagen sind Daten, nicht fehlende Kanten."
        )),
    )
    fields["qualifier"] = (str | None, Field(None, description="Einschränkung oder Begründung im Text, z. B. 'nur für neue Sitzungen'."))
    fields["quote"] = (str | None, Field(None, description="Wörtlicher Beleg aus dem Text, höchstens 200 Zeichen."))
    property_types: dict[str, TypeInfo] = {}
    for prop in rel.properties:
        info, hint = resolve_type(prop.type, ont, many=prop.many)
        property_types[prop.name] = info
        desc = " ".join(x for x in (prop.description, hint) if x) or prop.name
        fields[prop.name] = _field(info.python, required=False, many=prop.many, description=desc,
                                   examples=prop.examples)
    src = ", ".join(rel.source)
    doc = _docstring(
        f"Kante {rel.name} ({rel.label_de or rel.description or rel.name.replace('_', ' ').lower()}) "
        f"von {src} nach {', '.join(targets) if NODE_BASE not in rel.target else 'einem beliebigen Knoten'}",
        rel.description if rel.label_de else None,
        rel.cues_de,
    )
    model = create_model(  # type: ignore[call-overload]
        f"{_camel(rel.name)}Link",
        __doc__=doc,
        __config__=ConfigDict(is_entity=False, extra="ignore", populate_by_name=True),
        **fields,
    )
    return LinkSpec(rel, _snake(_camel(rel.name)), model, list(rel.source), targets, property_types)


def _build_entity(name: str, spec: ClassSpec, ont: Ontology, links: list[LinkSpec],
                  warnings: list[str]) -> CompiledClass:
    tname = lambda n: RESERVED_RENAMES.get(n, n)  # noqa: E731 - ontology field -> template field
    identity = [tname(k) for k in spec.identity.keys]
    declared = {f.name: f for f in spec.fields}
    fields: dict[str, tuple[Any, Any]] = {}
    field_types: dict[str, TypeInfo] = {}
    field_names: dict[str, str] = {LABEL_FIELD: "label"}

    # label first (as ``display_name``): identity when declared so, else optional
    label_spec = declared.get("label")
    label_required = LABEL_FIELD in identity
    label_desc = "Anzeigename (Label) genau wie im Dokument geschrieben."
    if label_spec is not None:
        info, hint = resolve_type(label_spec.type, ont)
        label_desc = " ".join(x for x in (label_spec.description or label_desc, hint) if x)
        field_types[LABEL_FIELD] = info
        fields[LABEL_FIELD] = _field(str, required=label_required, many=False, description=label_desc,
                                     examples=label_spec.examples)
    else:
        field_types[LABEL_FIELD] = TypeInfo("str")
        fields[LABEL_FIELD] = _field(str, required=label_required, many=False, description=label_desc,
                                     examples=None)

    for f in spec.fields:
        if f.name == "label":
            continue
        info, hint = resolve_type(f.type, ont, many=f.many)
        if info.kind == "str" and f.type not in ont.datatypes and f.type not in _PY_SCALARS:
            warnings.append(f"{name}.{f.name}: unknown type '{f.type}' treated as str")
        fname = tname(f.name)
        field_names[fname] = f.name
        field_types[fname] = info
        desc = " ".join(x for x in (f.description, hint) if x) or f.name.replace("_", " ")
        fields[fname] = _field(info.python, required=fname in identity, many=f.many,
                               description=desc, examples=f.examples)

    fields["aliases"] = (list[str], Field(default_factory=list, description="Weitere Schreibweisen, Abkürzungen oder Kurzformen desselben Knotens im Text."))
    fields["description"] = (str | None, Field(None, description="Ein bis zwei Sätze aus dem Dokument, die den Knoten beschreiben."))
    fields["quote"] = (str | None, Field(None, description="Wörtlicher Beleg aus dem Text, höchstens 200 Zeichen."))
    field_types.update({"aliases": TypeInfo("str", many=True), "description": TypeInfo("str"), "quote": TypeInfo("str")})

    link_fields: dict[str, str] = {}
    for link in links:
        fname = link.field_name
        if fname in fields:
            fname = f"{fname}_links"
        rel = link.relation
        tgt = "beliebigen Knoten" if NODE_BASE in rel.target else ", ".join(link.targets)
        fields[fname] = (
            list[link.model],  # type: ignore[valid-type]
            Field(default_factory=list, description=(
                f"Relation {rel.name} ({rel.label_de or rel.name}): Ziele vom Typ {tgt}, "
                "als Identitätswert wie im Text. Auch ausdrücklich verneinte Beziehungen eintragen "
                "(polarity='negative')."
            )),
        )
        link_fields[fname] = rel.name

    head = spec.label_de or spec.description or name
    doc = _docstring(head, spec.description if spec.label_de else None, spec.cues_de,
                     tail=f"Identität: {', '.join(spec.identity.keys)}.")
    model = create_model(  # type: ignore[call-overload]
        name,
        __doc__=doc,
        __config__=ConfigDict(graph_id_fields=identity, extra="ignore", populate_by_name=True),
        **fields,
    )
    return CompiledClass(name, spec, model, identity, field_types, link_fields, field_names)


def compile_template(ont: Ontology) -> CompiledTemplate:
    warnings: list[str] = []
    skipped: list[tuple[str, str]] = []
    class_names = list(ont.classes)

    links: dict[str, LinkSpec] = {}
    links_by_source: dict[str, list[LinkSpec]] = {c: [] for c in class_names}
    for rel in ont.relations:
        if NODE_BASE in rel.source:
            skipped.append((rel.name, "source NodeBase: derived from provenance, not extracted"))
            continue
        link = _build_link_model(rel, ont, class_names)
        links[rel.name] = link
        for src in rel.source:
            links_by_source[src].append(link)

    classes: dict[str, CompiledClass] = {}
    for name, spec in ont.classes.items():
        classes[name] = _build_entity(name, spec, ont, links_by_source[name], warnings)

    root_fields: dict[str, tuple[Any, Any]] = {
        "document_id": (
            str,
            Field(..., description="Kennung des Quelldokuments (z. B. BHB-PLT-0007) oder der Dateiname.",
                  examples=["BHB-PLT-0007"]),
        )
    }
    catalog_fields: dict[str, str] = {}
    for name, cc in classes.items():
        fname = _plural(_snake(name))
        catalog_fields[name] = fname
        root_fields[fname] = (
            list[cc.model],  # type: ignore[valid-type]
            Field(default_factory=list, json_schema_extra={"edge_label": CATALOG_EDGE},
                  description=f"Alle Knoten der Klasse {name} im Dokument."),
        )
    root = create_model(  # type: ignore[call-overload]
        ROOT_NAME,
        __doc__=(
            "Vollständige Extraktion eines Dokuments: jeder Knoten steht genau einmal in der Liste "
            "seiner Klasse; Beziehungen stehen als Links am Quellknoten und zeigen auf Identitätswerte."
        ),
        __config__=ConfigDict(graph_id_fields=["document_id"], extra="ignore", populate_by_name=True),
        **root_fields,
    )
    return CompiledTemplate(ont, root, classes, links, catalog_fields, skipped, warnings)


# ============================================================================ report


def check_report(compiled: CompiledTemplate) -> str:
    ont = compiled.ontology
    lines = [
        (
            f"Ontology {ont.meta.name} {ont.meta.version}: {len(ont.classes)} classes, "
            f"{len(ont.relations)} relations, {len(ont.enums)} enums, {len(ont.datatypes)} datatypes"
        ),
        "",
    ]
    lines.append("Classes (identity keys):")
    for name, cc in compiled.classes.items():
        own = [f for f in cc.field_types if f not in TEMPLATE_IMPLICIT_FIELDS]
        lines.append(f"  {name:<22} id={cc.spec.identity.keys}  fields={len(own)}  links={len(cc.link_fields)}")
    lines.append("")
    lines.append("Relations (source -> target):")
    for name, link in compiled.links.items():
        props = f"  props={list(link.property_types)}" if link.property_types else ""
        tgt = "any" if NODE_BASE in link.relation.target else ", ".join(link.targets)
        lines.append(f"  {name:<24} {', '.join(link.sources)} -> {tgt}{props}")
    if compiled.skipped_relations:
        lines.append("")
        lines.append("Skipped relations:")
        lines.extend(f"  {n}: {why}" for n, why in compiled.skipped_relations)
    if compiled.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  {w}" for w in compiled.warnings)
    scoped = [n for n, c in ont.classes.items() if c.identity.scope]
    if scoped:
        lines.append("")
        lines.append(f"Not representable (identity.scope, documented): {', '.join(scoped)}")
    schema = compiled.schema_json()
    lines.append("")
    lines.append(f"LLM schema: {len(schema):,} chars, hash {compiled.schema_hash()}")
    return "\n".join(lines)
