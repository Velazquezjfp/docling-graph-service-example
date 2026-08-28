"""Value, identity-key and polarity normalization for the graph materializer."""

from __future__ import annotations

import re
from typing import Any

from .ontology import TypeInfo
from .schemas import Polarity

_TITLES = re.compile(r"^(?:(?:dr|prof|dipl\.?-ing|mag|herr|frau|hr|fr)\.?\s+)+", re.IGNORECASE)
_DATE_DE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NEG_WORDS = {"negative", "negativ", "verneint", "nein", "no", "false"}
_POS_WORDS = {"positive", "positiv", "ja", "yes", "true"}


def norm_key(text: str, *, strip_titles: bool = False) -> str:
    """Matching key: whitespace-collapsed, casefolded, trailing punctuation removed."""
    s = " ".join(str(text).replace(" ", " ").split()).strip().rstrip(".,;:")
    if strip_titles:
        s = _TITLES.sub("", s)
    return s.casefold()


def _to_float(raw: str) -> float:
    s = raw.replace(" ", "").replace(" ", "").replace("%", "").strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return float(s)


def normalize_scalar(value: Any, info: TypeInfo) -> tuple[Any, str | None]:
    """Return (normalized value, violation message or None). Values are never dropped."""
    if value is None:
        return None, None
    if info.kind == "enum":
        s = str(value).strip()
        for allowed in info.enum_values:
            if s.casefold() == allowed.casefold():
                return allowed, None
        return s, f"not in {list(info.enum_values)}"
    if info.kind == "date":
        s = str(value).strip()
        m = _DATE_DE.match(s)
        if m:
            d, mo, y = m.groups()
            return f"{y}-{int(mo):02d}-{int(d):02d}", None
        return s, None if _DATE_ISO.match(s) else "unrecognized date format"
    if info.kind in ("percent", "float"):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), None
        try:
            return _to_float(str(value)), None
        except ValueError:
            return str(value), "not a number"
    if info.kind == "int":
        if isinstance(value, bool):
            return value, "bool where int expected"
        if isinstance(value, int):
            return value, None
        try:
            return int(_to_float(str(value))), None
        except ValueError:
            return str(value), "not an integer"
    if info.kind == "bool":
        if isinstance(value, bool):
            return value, None
        s = str(value).strip().casefold()
        if s in ("true", "ja", "yes", "wahr", "1"):
            return True, None
        if s in ("false", "nein", "no", "falsch", "0"):
            return False, None
        return str(value), "not a boolean"
    s = str(value).strip() if not isinstance(value, str) else value.strip()
    if info.pattern and not re.fullmatch(info.pattern, s):
        return s, f"does not match {info.pattern}"
    return s, None


def normalize_value(value: Any, info: TypeInfo) -> tuple[Any, list[str]]:
    if info.many:
        items = value if isinstance(value, list) else [value]
        out, problems = [], []
        for item in items:
            v, p = normalize_scalar(item, info)
            if v is not None:
                out.append(v)
            if p:
                problems.append(p)
        return out, problems
    v, p = normalize_scalar(value, info)
    return v, [p] if p else []


def has_negation(text: str, triggers: list[str]) -> bool:
    low = f" {norm_key(text)} "
    for trig in triggers:
        t = trig.casefold()
        if " " in t:
            if t in low:
                return True
        elif re.search(rf"\b{re.escape(t)}\b", low):
            return True
    return False


def normalize_polarity(raw: str | None, evidence: str, triggers: list[str]) -> tuple[Polarity, str | None]:
    """Never default to positive when the evidence contains a negation trigger."""
    if raw:
        r = raw.strip().casefold()
        if r in _NEG_WORDS or r.startswith(("neg", "kein", "verne")):
            return "negative", None
        if r in _POS_WORDS or r.startswith("pos"):
            return "positive", None
        return "unknown", f"unrecognized polarity value {raw!r}"
    if evidence and has_negation(evidence, triggers):
        return "unknown", "polarity missing but evidence contains a negation trigger"
    return "positive", None
