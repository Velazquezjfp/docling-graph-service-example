import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from docling_graph_service.chunking import Chunker, infer_captions
from docling_graph_service.convert import (
    dehyphenate_table_cells,
    normalize_heading_levels,
    strip_repeated_furniture,
)
from docling_graph_service.settings import ChunkingSettings

pytestmark = pytest.mark.filterwarnings("ignore")


def _prov(page=1, l=10, t=100, r=200, b=50):
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.BOTTOMLEFT),
                          charspan=(0, 1))


def _table(rows, cols, cell_text):
    cells = []
    for r in range(rows):
        for c in range(cols):
            text = f"H{c}" if r == 0 else cell_text(r, c)
            cells.append(TableCell(text=text, row_span=1, col_span=1, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                                   start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=r == 0))
    return TableData(num_rows=rows, num_cols=cols, table_cells=cells)


@pytest.fixture(scope="module")
def chunker():
    return Chunker(ChunkingSettings(max_tokens=128), embedding_prefix="passage: ")


@pytest.fixture
def doc():
    d = DoclingDocument(name="t")
    for p in (1, 2, 3):
        d.add_page(page_no=p, size=Size(width=600, height=800))
        d.add_text(label=DocItemLabel.TEXT, text="TESTDOKUMENT · fiktiv", prov=_prov(page=p, t=790, b=780))
        d.add_text(label=DocItemLabel.TEXT, text=f"Seite {p} von 3", prov=_prov(page=p, t=20, b=10))
    d.add_heading("5 Betrieb", level=2, prov=_prov())
    d.add_heading("5.3 Standardabläufe", level=1, prov=_prov())
    d.add_heading("5.3.1 Keycloak neu starten", level=2, prov=_prov())
    d.add_text(label=DocItemLabel.PARAGRAPH, text="Erster Absatz zum Neustart.", prov=_prov())
    d.add_text(label=DocItemLabel.PARAGRAPH, text="Zweiter Absatz zum Neustart.", prov=_prov())
    d.add_heading("Achtung", level=5, prov=_prov())
    d.add_text(label=DocItemLabel.PARAGRAPH, text="Ein Hinweis.", prov=_prov())
    d.add_heading("5.4 Tabellen", level=1, prov=_prov(page=2))
    cap = d.add_text(label=DocItemLabel.CAPTION, text="Tabelle 1: Breite Tabelle", prov=_prov(page=2))
    d.add_table(data=_table(12, 4, lambda r, c: f"Zelle {r}-{c} mit ziemlich langem Inhalt für den Test"),
                caption=cap, prov=_prov(page=2))
    # page-continuation of the same table: no caption, same header row
    d.add_table(data=_table(4, 4, lambda r, c: f"Fortsetzung {r}-{c}"), prov=_prov(page=3))
    return d


def test_heading_levels_and_furniture(doc):
    assert strip_repeated_furniture(doc, min_pages=3, band=0.12) == 6
    assert normalize_heading_levels(doc) >= 3
    levels = {it.text: it.level for it, _ in doc.iterate_items() if it.label == DocItemLabel.SECTION_HEADER}
    assert levels["5 Betrieb"] == 1 and levels["5.3 Standardabläufe"] == 2
    assert levels["5.3.1 Keycloak neu starten"] == 3 and levels["Achtung"] == 4 and levels["5.4 Tabellen"] == 2


def test_chunks(chunker, doc):
    strip_repeated_furniture(doc, min_pages=3, band=0.12)
    normalize_heading_levels(doc)
    captions, inferred = infer_captions(doc)
    assert inferred == 1 and len(captions) == 2  # continuation table inherits the caption
    result = chunker.chunk(doc, "deadbeefcafe0123")
    chunks = result.chunks
    assert all(c.chunk_id.startswith("deadbeefcafe-") for c in chunks)
    assert all(c.token_count <= 128 for c in chunks)
    assert not any("TESTDOKUMENT" in c.text or "Seite 1 von 3" in c.text for c in chunks)
    # text peers under one heading merged; breadcrumb complete
    para = next(c for c in chunks if "Erster Absatz" in c.body_text)
    assert "Zweiter Absatz" in para.body_text
    assert para.heading_breadcrumb == ["5 Betrieb", "5.3 Standardabläufe", "5.3.1 Keycloak neu starten"]
    assert para.heading_level == 3 and para.kind == "text"
    assert para.text.startswith("5 Betrieb\n5.3 Standardabläufe\n5.3.1 Keycloak neu starten\nErster Absatz")
    # every table part: caption + header row, never merged with text, split not truncated
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) >= 3 and all(c.caption == "Tabelle 1: Breite Tabelle" for c in tables)
    for c in tables:
        lines = [line for line in c.body_text.split("\n") if line.strip()]
        assert lines[0] == "Tabelle 1: Breite Tabelle"
        assert lines[1].startswith("| H0") and lines[2].lstrip("| ").startswith("-")
        assert c.page_numbers in ([2], [3])
    body = "\n".join(c.body_text for c in tables)
    assert all(f"Zelle {r}-3" in body for r in range(1, 12)) and "Fortsetzung 3-3" in body
    assert not any(c.kind == "mixed" for c in chunks)
    # bboxes converted to TOPLEFT with page size; self_ref index maps items to chunks
    bb = para.bboxes[0]
    assert bb.coord_origin == "TOPLEFT" and bb.t == 700 and bb.b == 750 and bb.page_height == 800
    assert all(cid in {c.chunk_id for c in chunks} for ids in result.self_ref_index.values() for cid in ids)
    assert any(ref.startswith("#/tables/") for ref in result.self_ref_index)


def test_hard_cap_splits_never_truncates(chunker):
    from docling_graph_service.chunking import _Draft

    words = " ".join(f"Wort{i}" for i in range(400))
    warnings: list[str] = []
    parts = chunker._enforce_cap(_Draft(words, ["Kapitel"], [], "text", None), 100, warnings)
    assert len(parts) > 1 and warnings and "split into" in warnings[0]
    assert " ".join(p.body for p in parts).split() == words.split()  # nothing lost
    assert all(chunker._count("Kapitel\n" + p.body, special=False) <= 100 for p in parts)
    # table: caption + header repeated on every part, rows never cut
    rows = "\n".join(f"| Zeile {i} mit etwas Text | Spalte zwei {i} |" for i in range(40))
    table = f"Tabelle 9: Test\n\n| A | B |\n|---|---|\n{rows}"
    parts = chunker._enforce_cap(_Draft(table, [], [], "table", "Tabelle 9: Test"), 100, [])
    assert len(parts) > 1
    for p in parts:
        lines = [line for line in p.body.split("\n") if line.strip()]
        assert lines[0] == "Tabelle 9: Test" and lines[1] == "| A | B |" and lines[2] == "|---|---|"
    assert sum(len([line for line in p.body.split("\n") if line.startswith("| Zeile")]) for p in parts) == 40


def test_budget_accounts_for_prefix(chunker):
    assert chunker.prefix_tokens > 0
    assert chunker.budget == 128 - chunker.prefix_tokens - 2


def test_dehyphenate_table_cells():
    d = DoclingDocument(name="t")
    d.add_table(data=_table(2, 3, lambda r, c: ["ClusterIssuer bavd- issuing-ca-3", "kv/event- system/*", "IAM- und PKI-Betrieb"][c]))
    assert dehyphenate_table_cells(d) == 2
    texts = [c.text for c in d.tables[0].data.table_cells if c.start_row_offset_idx == 1]
    assert texts == ["ClusterIssuer bavd-issuing-ca-3", "kv/event-system/*", "IAM- und PKI-Betrieb"]
