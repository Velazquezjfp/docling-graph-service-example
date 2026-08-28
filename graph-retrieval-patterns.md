# Graph & retrieval patterns — how the docling-graph-service output is structured and how to retrieve over it

Audience: whoever builds the next modules (`ingestion/` indexing into OpenSearch, the retrieval layer, the
Streamlit UI). Everything below is verified against the ZSD reference run
(`out/zsd-integration/response.json`, 2026-08-27; numbers in §5 come from that run).

---

## 1. The three artefacts and how they join

One `POST /v1/process` (or `dgs process`) returns one `ProcessResponse` per document:

```
ProcessResponse
├── document      {name, format, sha256, pages, tables, pictures}
├── markdown      full Markdown (furniture removed, figure descriptions inline as "[Description] …")
├── chunks[]      ← the ONLY place where source text and embeddings live
├── graph         {nodes[], edges[], meta}   ← structured layer; points INTO chunks, never copies them
├── degraded      {vlm, embeddings, graph}   ← if true, the corresponding part is missing/partial
└── errors[], warnings[], timings_s, versions, cached
```

Join keys:

| From | To | Key |
|---|---|---|
| node / edge → text | `chunks[]` | `provenance.chunk_ids[]` → `chunk.chunk_id` |
| node / edge → DoclingDocument | `document.json` (CLI only) | `provenance.dom_paths[]` (`#/texts/44`, `#/tables/1`, `#/pictures/0`) |
| edge → nodes | `graph.nodes[]` | `edge.source`, `edge.target` = `node.id` |
| any → document | `document.sha256` | chunk ids are prefixed with `sha256[:12]` |

Ids are **deterministic per document**: `chunk_id = "<sha256[:12]>-<index:04d>"`, `node.id = "<Class>_<hash of
identity value>"`. Re-processing the same PDF with the same options gives the same ids. Ids are *not* shared across
documents — the same person in two manuals is two nodes (see §3.4).

### 1.1 Chunk (retrieval unit)

```json
{ "chunk_id": "2bb722f565a0-0060",
  "text": "6 Abhängigkeiten und Auswirkungen\n6.2 Auswirkungsmatrix\nTabelle 8: …\n\n| Ereignis | VPP … |",   // breadcrumb + body = what was embedded
  "body_text": "Tabelle 8: Auswirkungen bei Neustart oder Ausfall\n\n| Ereignis | … |",                  // verbatim content
  "heading_breadcrumb": ["6 Abhängigkeiten und Auswirkungen", "6.2 Auswirkungsmatrix"],
  "heading_level": 2,
  "kind": "table",                       // text | table | picture | mixed
  "caption": "Tabelle 8: Auswirkungen bei Neustart oder Ausfall",   // tables/pictures only
  "page_numbers": [19],
  "dom_paths": ["#/tables/13"],
  "bboxes": [{"page": 19, "l": 56.7, "t": 88.1, "r": 539.2, "b": 402.6, "page_width": 595.9, "page_height": 842.9, "coord_origin": "TOPLEFT"}],
  "token_count": 448,                    // XLM-R tokens incl. special tokens, ≤ 512
  "embedding": [ …1024 floats… ] }
```

- **Tables are chunks**, never nodes: `body_text` is the complete Markdown table *part* (caption + header row +
  rows). Page-spanning tables come as one chunk per page, each repeating caption + header. Rows are never cut.
- **Pictures are chunks**: `body_text` = caption + VLM description (German). Pixels are not in the response;
  `dom_paths` addresses the picture in `document.json` if the UI ever needs to show it.
- `text` (breadcrumb + body) is what the embedding was computed on. Embed queries with the same model
  (`bge-m3`; for `multilingual-e5-large` prefix documents with `"passage: "` and queries with `"query: "`).

### 1.2 Node (entity, LLM-extracted, ontology-typed)

```json
{ "id": "Incident_9bee92d1caabf291",
  "type": "Incident",                                   // one of the 26 ontology classes
  "label": "ZSDSUP-0247",                               // display name; falls back to the identity value
  "attributes": { "ticket_id": "ZSDSUP-0247", "title": "Vault nach Knotenwartung versiegelt",
                  "date": "2026-07-02", "severity": "sev1", "duration": "38 min",
                  "root_cause": "…", "resolution": "…", "process_change": "…" },   // ontology fields, normalized
  "aliases": [],
  "quote": "…",                                         // literal evidence sentence, ≤ ~200 chars
  "provenance": { "pages": [4, 12, 22, 23, 24],
                  "chunk_ids": ["2bb722f565a0-0008", "2bb722f565a0-0009", "2bb722f565a0-0039", "2bb722f565a0-0071", "2bb722f565a0-0074", "2bb722f565a0-0075"],
                  "dom_paths": ["#/texts/44", "#/tables/1", "#/texts/189", "#/texts/485"],
                  "heading_breadcrumb": ["1 Dokumentinformation", "1.2 Änderungshistorie"],   // of the first chunk
                  "table_ref": "Tabelle 1",             // first table among its chunks (not necessarily the defining one)
                  "figure_ref": null,
                  "match": "verbatim" } }               // docling-graph: verbatim | observed | reconciled | derived
```

- `attributes` are the LLM's *reading* of the document (short values, paraphrased sentences; largest node ≈ 1.9 KB).
  They are **not** the source text. The source text is `chunks[chunk_ids]` (§2.1).
- `provenance.chunk_ids` = every chunk docling-graph anchored the entity to, in document order, across the whole
  document (a node typically has 1–4 chunks, up to 9). `pages` can list more pages than the chunks cover — those are
  mentions without an anchored text span; the chunks are the reliable part.
- Values are normalized: dates `DD.MM.YYYY` → ISO, decimal comma → point, enums matched case-insensitively.
  Violations are **kept and listed** in `graph.meta` (`enum_violations`, `identity_pattern_violations`), never dropped.

### 1.3 Edge (claim, ontology-typed, citable)

```json
{ "source": "ImpactStatement_e924114e049dd3a7", "target": "System_8c8c6e95949e6f73",
  "type": "IMPACT_OF",                                  // one of the 32 relations
  "polarity": "negative",                               // positive | negative | unknown
  "qualifier": "Keystores und Wallet liegen lokal auf den Knoten",
  "quote": "unberührt, Keystores und Wallet liegen lokal auf den Knoten",
  "properties": {},                                     // typed per relation, e.g. RESPONSIBLE_FOR {"raci": "verantwortlich", "is_deputy": false}
  "provenance": { "pages": [19], "chunk_ids": ["2bb722f565a0-0060"] } }
```

- **`polarity` is data.** The corpus states negatives on purpose ("VPP ist *nicht* Mandant der CaaS-Plattform").
  `negative` edges must be rendered as "explicitly NOT"; `unknown` means the LLM gave no polarity but the quote
  contains a negation trigger — show the quote, do not assert either way. Never filter negatives out silently.
- Both polarities for one `(source, target, type)` can coexist; they are listed in `meta.conflicts`.
- `provenance.chunk_ids` = the chunk containing `quote` (fallback: the source node's chunks).
- `MENTIONED_IN` (node → document) is not extracted; derive it from `document.sha256`/`doc_id` at index time.

### 1.4 `graph.meta` (audit / review queue)

`node_count`, `edge_count`, `nodes_by_type`, `edges_by_type`, `edges_by_polarity`, `unresolved_targets[]`
(`{source, type, target_type, target, polarity, reason}` — link targets the LLM named but no node matched; the
SPEC §4.4 review queue, never stub nodes), `conflicts[]`, `enum_violations[]`, `identity_pattern_violations[]`,
`polarity_notes[]`, `alias_reconciliation` (docling-graph's merges), `template_schema_hash`, `ontology{name,version}`.

---

## 2. Recipes

### 2.1 Full source text behind a node or edge

```python
chunks = {c["chunk_id"]: c for c in resp["chunks"]}

def evidence(item):                       # item = node or edge
    for cid in item["provenance"]["chunk_ids"]:
        c = chunks[cid]
        yield f"[{' > '.join(c['heading_breadcrumb'])} | S. {', '.join(map(str, c['page_numbers']))}]\n{c['body_text']}"
```

For an edge, `edge["quote"]` is the citation sentence; the chunk gives the surrounding context (often a whole table).

### 2.2 Reverse index (build once per document at indexing time)

```python
from collections import defaultdict
chunk_to_nodes, chunk_to_edges = defaultdict(list), defaultdict(list)
for n in graph["nodes"]:
    for cid in n["provenance"]["chunk_ids"]: chunk_to_nodes[cid].append(n["id"])
for i, e in enumerate(graph["edges"]):
    for cid in e["provenance"]["chunk_ids"]: chunk_to_edges[cid].append(i)
```

Store the results *on the chunk documents* (`node_ids[]`, `edge_ids[]`) so a kNN hit yields its graph neighbourhood
in the same lookup.

### 2.3 Load into NetworkX (no graph database, ADR-0005)

```python
G = nx.MultiDiGraph()                      # Multi: several relation types between the same two nodes
for n in graph["nodes"]: G.add_node(n["id"], **n)
for e in graph["edges"]: G.add_edge(e["source"], e["target"], key=e["type"] + ":" + e["polarity"], **e)
```

~300 nodes / ~150 edges per manual; all five documents fit in memory trivially.

---

## 3. What to index in OpenSearch

### 3.1 `chunks` index (kNN enabled — request the k-NN plugin by name)

| Field | Type | Source |
|---|---|---|
| `chunk_id` | keyword (doc id) | `chunk.chunk_id` |
| `doc_sha256`, `doc_id`, `doc_name` | keyword | `document.sha256`; `doc_id` from the `Document` node (`BHB-PLT-0007`) |
| `text` | text (German analyzer) | `chunk.text` |
| `body_text` | text | `chunk.body_text` |
| `embedding` | knn_vector, dim 1024, cosine | `chunk.embedding` |
| `heading_breadcrumb` | keyword[] + text | |
| `kind`, `caption` | keyword / text | table/picture routing (“show me the table”) |
| `page_numbers` | integer[] | citations |
| `bboxes` | object (not indexed) | UI highlighting |
| `node_ids`, `edge_ids` | keyword[] | reverse index (§2.2) |

### 3.2 `graph_nodes` index (BM25 entry point, §4.3)

`node_id`, `doc_id`, `type`, `label` (text + keyword), `aliases` (text), `identity` (keyword: ticket_id / hostname /
sop_id / … — the identity value), `attributes` (object, `flattened`), `chunk_ids`, `pages`, and optionally
`label_embedding` (knn_vector; embed `"<type>: <label> (<aliases>) — <description>"`).

### 3.3 `graph_edges` index or a single JSON blob per document

Edges are only traversed, never searched — keep the whole `graph` JSON per document in one index (or object
store) and load all of them into NetworkX at service start. Re-index = re-load.

### 3.4 Cross-document merge (this is the ingestion module's job)

Nodes are per document. Merge key: `type` + normalized identity — casefold, whitespace-collapsed; for `Person`
strip titles (`Dr.`); for classes with `identity.scope: system` (Component, MaintenanceWindow, FailureMode) add the
root system to the key. On merge: concatenate `chunk_ids`/`pages`, keep attribute conflicts as a candidate list with
provenance (the corpus contains one intentional conflict: 38 vs 48 min for ZSDSUP-0247), union aliases. Same for
edges (`PARTNER_TICKET` is asserted from both documents and must become one edge). Also merge intra-document
duplicates the LLM produced (`Ebert` vs `Marcel Ebert`, `Event-System` vs `Event-System 2.0`): a label that is a
suffix/token subset of exactly one other label of the same type is an alias.

---

## 4. Retrieval design

### 4.1 Principle

The graph is a **router and a structured answer layer**; the chunks are the **retrieval unit and the safety net**.
Every answer path ends in chunks (citable, complete, always present even when `degraded.graph=true`).

### 4.2 Two paths, one merge step

**Graph-first** (entity-centric questions: *wer / welche / was hängt an X*):
1. Resolve start node(s) (§4.3).
2. Expand **one hop over all relation types** (§4.4 explains why no intent filter yet), keep `polarity`.
3. Context = chunks of the start nodes + touched edges (dedupe by `chunk_id`), plus a rendered fact list:
   `Kai Ostermann —RESPONSIBLE_FOR→ ZSD (raci=verantwortlich) [S. 26]`,
   `VPP —DEPENDS_ON (NEGATIVE: "unberührt, Keystores liegen lokal")→ Vault [S. 19]`.

**Vector-first** (situational questions: *was passiert wenn…, wie erneuere ich…*):
1. kNN over `chunks.embedding` (top 10–20), optionally hybrid with BM25 on `text` (RRF).
2. `node_ids`/`edge_ids` of the hits → expand one hop → add those nodes' chunks (bounded, e.g. +10).
3. Same context assembly as above.

Run both, merge by `chunk_id` with RRF, cap the context, answer with citations `(Dokument, Seite, Tabelle)`.

### 4.3 Start-node resolution — hybrid, LLM last

1. **BM25 over `graph_nodes.label/aliases/identity`.** Ops questions carry exact identifiers (`ZSDSUP-0247`,
   `SOP-ZSD-05`, `vault-0`, `kv/vpp/monitor`) — lexical wins outright; boost `identity` exact matches.
2. **kNN on chunks → `node_ids`** (reverse index): the chunks that answer "Vault versiegelt" already carry the
   ImpactStatement and Vault nodes. Needs no extra embeddings and handles paraphrase.
3. Optional sharpener: `label_embedding` per node (≈1,500 vectors for five manuals).
4. **LLM only to disambiguate** when candidates tie (`Vault` as System vs Component vs Term): a pick-one over
   ≤ 5 candidates, never open-ended entity extraction.

### 4.4 Intent layer: not now, and later not as "intents"

- Today ~0.5 edges per node; 1-hop neighbourhoods have a median of 1–2 edges. Expanding all relation types and
  rendering them typed + polarity-aware is cheaper and more robust than any classifier, and all seven competency
  questions are 1–2 hop patterns (§5.2).
- When neighbourhoods become noisy (measure with the golden set), add an **ontology-constrained query plan**: one
  small LLM call that receives the question plus the 32 relation names with `label_de` and returns
  `{start_entities, relations, direction, depth}`; the traversal stays deterministic and explainable. This is a
  30-line addition, not an architecture decision, and it never produces relation names outside the ontology.

### 4.5 Rendering rules that matter

- Show `polarity=negative` as an explicit statement ("laut BHB-PLT-0007 *nicht*…"), with the quote.
- Prefer `edge.quote` as the citation; fall back to the chunk's page/caption.
- Attribute conflicts (after merge) are shown as alternatives with sources, not resolved.
- `unresolved_targets` are not facts; they can feed a "related mentions" panel or a review UI.

---

## 5. State of the graph (ZSD reference run) and what to fix first

### 5.1 Numbers

279 nodes (23 of 26 classes used), 145 edges (19 of 32 relations), 138 positive / 7 negative, 18 unresolved targets,
0 conflicts, all nodes and edges with provenance; `degraded.graph=false`. Entity recall is excellent (all hosts,
firewall rules, SOPs, incidents, 20 impact statements, 11 start-up steps, 19 persons).

Weak spot = **edge recall**: 13 relations had 0 edges (`HAS_COMPONENT`, `SECURED_BY`, `ISSUED_BY`, `ALLOWS_TRAFFIC`,
`USES_DATASTORE`, `AFFECTS`, `DOCUMENTS`, …), 0 `Certificate` nodes, 136/279 nodes isolated (17 Hosts, all 16
FirewallRules). Run-to-run variance is high (8 vs 24 `System` nodes between two runs at temperature 0). Cause: edges
are filled inside each node's fill call, which only sees the chunks where that node was found
(`dense_fill_context=scoped`), so cross-section relations are missed.

### 5.2 Competency questions vs. the current graph

| Q | Traversal | Status |
|---|---|---|
| Q1 Zertifikat erneuern | System –SECURED_BY→ Certificate –ISSUED_BY→ CA; –GOVERNED_BY→ Procedure | ✗ no Certificate nodes/edges; `GOVERNED_BY` (11) and Procedure attributes (`steps`, `commands`) work |
| Q2 Komponente neu starten | System –HAS_COMPONENT→ Component; ImpactStatement; –GOVERNED_BY→ | partial (`HAS_COMPONENT` 0; ImpactStatements ✓) |
| Q3 Ansprechpartner / Eskalation | System ←RESPONSIBLE_FOR– Person –ESCALATES_TO→ Person | ✓ (Ostermann, Wollmer → Dr. Reuß) |
| Q4 Server und Ports | Component –RUNS_ON→ Host –LOCATED_IN→ Zone; FirewallRule –ALLOWS_TRAFFIC→ | partial (`RUNS_ON` 14, `ALLOWS_TRAFFIC` 0; FirewallRule *attributes* carry the ports) |
| Q5 Dienst Z fällt aus | ImpactStatement{causing_system=Z} → affected_system | ✓ incl. the negative "keine" answer |
| Q6 Kaltstartreihenfolge | StartupStep –PRECEDES→ StartupStep | ✓ chain 1→10 |
| Q7 Schon einmal gelöst? | Incident –PARTNER_TICKET→ Incident (other document) | ✓ 18 edges; `INSTANCE_OF_FAILURE` 0 |

### 5.3 Fixes in the service, cheapest first (before designing around the gaps)

1. **Materialize `reference` attributes as edges** — `ImpactStatement.affected_system/causing_system`,
   `Person.org_unit`, `OpenItem.owner`, `Term.maps_to` already hold identity values; resolving them is
   deterministic and would connect dozens of isolated nodes.
2. **Table-driven relation pass** — the ontology states ~70 % of edges sit in recurring tables
   (`extraction.high_yield_tables`). One LLM call per table chunk (≈26) with the table Markdown + the existing node
   labels of the allowed classes, asking only for edges of the relations that table yields. Deterministic scope,
   high recall, low variance.
3. `dense_fill_context="full"` as a configuration experiment (more tokens, more cross-section edges).
4. Attribute-derived edges for `Host` (`FirewallRule.source_desc/target_desc` → hosts/zones by lexical match) if
   Q4 stays weak after 1–2.

### 5.4 Evaluation

Build the golden set from the seven competency questions × five manuals plus the six `negative_assertions` in
`ontology.yaml`; score (a) start-node hit rate, (b) whether the answer's citations point at the right pages, (c) that
negatives are stated as negatives. Re-run after each fix in §5.3 — the graph quality, not the retrieval code, is
the variable to watch first.
