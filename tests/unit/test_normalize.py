import pytest

from docling_graph_service.normalize import (
    has_negation,
    norm_key,
    normalize_polarity,
    normalize_value,
)
from docling_graph_service.ontology import DEFAULT_NEGATION_TRIGGERS, TypeInfo

T = list(DEFAULT_NEGATION_TRIGGERS)


def test_norm_key_titles_and_whitespace():
    assert norm_key("  Dr.  Annika   Reuß ", strip_titles=True) == "annika reuss"  # casefold: ß -> ss
    assert norm_key("Dr. Annika Reuß") == "dr. annika reuss"
    assert norm_key("Vault.") == "vault"


def test_dates_percent_int_bool_enum():
    assert normalize_value("22.07.2026", TypeInfo("date")) == ("2026-07-22", [])
    assert normalize_value("2026-07-22", TypeInfo("date")) == ("2026-07-22", [])
    assert normalize_value("Juli 2026", TypeInfo("date"))[1] == ["unrecognized date format"]
    assert normalize_value("99,95 %", TypeInfo("percent")) == (99.95, [])
    assert normalize_value("1 284", TypeInfo("int")) == (1284, [])
    assert normalize_value("ja", TypeInfo("bool")) == (True, [])
    assert normalize_value("Kritisch", TypeInfo("enum", enum_values=("kritisch", "hoch"))) == ("kritisch", [])
    val, probs = normalize_value("egal", TypeInfo("enum", enum_values=("kritisch", "hoch")))
    assert val == "egal" and probs and "not in" in probs[0]
    val, probs = normalize_value("Vault-0", TypeInfo("str", pattern=r"^[a-z][a-z0-9-]{1,30}$"))
    assert val == "Vault-0" and "does not match" in probs[0]  # kept and flagged, never dropped
    assert normalize_value(["a", "b"], TypeInfo("str", many=True)) == (["a", "b"], [])


@pytest.mark.parametrize("raw, evidence, expected", [
    ("negative", "", "negative"),
    ("negativ", "", "negative"),
    ("kein", "", "negative"),
    ("positive", "", "positive"),
    ("ja", "", "positive"),
    (None, "VPP ist Mandant der Plattform", "positive"),
    (None, "VPP ist nicht Mandant der CaaS-Plattform", "unknown"),
    (None, "cert-manager und ACME werden dort bewusst nicht genutzt", "unknown"),
    ("maybe", "", "unknown"),
])
def test_polarity(raw, evidence, expected):
    pol, _ = normalize_polarity(raw, evidence, T)
    assert pol == expected


def test_negation_word_boundaries():
    assert not has_negation("Keinesfalls ist das Wartungsfenster verschoben", ["kein"])  # 'kein' != 'keinesfalls'
    assert has_negation("Auswirkung: keine", ["keine"])
    assert has_negation("wird dort bewusst nicht genutzt", ["bewusst nicht genutzt"])
