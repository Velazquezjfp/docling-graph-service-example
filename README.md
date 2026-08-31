# docling-graph-service

Stateless ingestion & knowledge-extraction microservice for the RAG prototype (module 1). One
request turns a Betriebshandbuch PDF into three synchronized artefacts:

| Artefact | What it is |
|---|---|
| `markdown` | Docling Markdown of the document, page headers/footers removed, VLM descriptions of figures inline (`[Description] …` line after the `<!-- image -->` placeholder) |
| `chunks[]` | Layout-aware chunks (docling `HybridChunker`, XLM-R token budget) with heading breadcrumb, DOM paths (`self_ref`), page numbers, TOPLEFT bounding boxes, token count and embedding vector |
| `graph` | Ontology-grounded knowledge graph: typed nodes with attributes + provenance (pages, chunk ids, DOM paths, table/figure ref), directed edges with `type`, `polarity`, `qualifier`, `quote`, typed `properties` and chunk grounding |

All model inference is remote through OpenAI-compatible endpoints (LiteLLM proxy locally, vLLM/TEI in
the target environment): an extraction LLM drives [docling-graph](https://pypi.org/project/docling-graph/),
a VLM describes figures, an embedding model vectorizes chunks. The container only parses (docling +
EasyOCR on CPU), orchestrates and enriches. Every optional stage degrades gracefully (`degraded.*`
flags + `errors[]`); only a failed conversion fails the request.

```
PDF ──docling (OCR de/en, TableFormer, images_scale 3)──▶ DoclingDocument
      │ strip page furniture · normalize heading levels · infer captions
      ├──▶ describe.py  (VLM, /chat/completions with image)  ──▶ PictureItem.meta.description
      ├──▶ export_to_markdown
      ├──▶ chunking.py  (HybridChunker + caption/header repeat + token cap) ──▶ chunks
      │        └──▶ embed.py (/embeddings, batched)
      └──▶ graph.py    (DoclingDocument JSON → docling-graph dense contract → materialize nodes/edges)
```

A phase-by-phase flow diagram (inputs, tools per step, outputs): [`service-flow.md`](service-flow.md).
Inspect processed results in the browser: `python3 scripts/serve_viewer.py --dir out` (stdlib-only viewer:
nodes/edges with provenance, source chunks, review queue — see the root README).

## Quick start (native, pip)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu   # CPU torch
# reproducible: .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env            # DGS__LLM__API_KEY etc.
.venv/bin/pytest tests/unit -q  # 49 tests, no network

# one document, all stages (writes document.json, markdown.md, chunks.json, graph.json, meta.json)
.venv/bin/dgs process ../user-manual-books/handbuch_daten/handbuch/Betriebshandbuch_ZSD.pdf --out out/zsd
.venv/bin/dgs process file.pdf --no-graph --no-vlm --no-embed   # conversion + chunks only

.venv/bin/dgs ontology-check ../user-manual-books/handbuch_daten/Ontologie/ontology.yaml
.venv/bin/dgs capabilities
.venv/bin/dgs serve             # http://localhost:8080
.venv/bin/python scripts/process_via_api.py file.pdf --url http://localhost:8080 --out out/x   # client against a running service
```

Set `DGS__SERVICE__CACHE_DIR` (e.g. `.cache/dgs`) to cache the converted DoclingDocument per
(pdf, OCR setting) and complete results per request key: a ~5 min CPU conversion then happens once.

## Deploying elsewhere

Step-by-step (files, image transfer or Artifactory build, `.env` for your endpoints, verification, client
script, running the same tests over HTTP, running without a VLM): [`DEPLOY.md`](DEPLOY.md).

## Docker

```bash
make docker-build                         # python:3.12 + CPU torch + docling/EasyOCR models + tokenizers baked in
make docker-run                           # docker run --network host --env-file .env … (LiteLLM on localhost:4000)
curl -s localhost:8080/healthz
```

Build arguments for the enterprise profile: `BASE_IMAGE`, `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`,
`PIP_TRUSTED_HOST`, `HF_ENDPOINT` (Artifactory mirrors). The image runs fully offline
(`HF_HUB_OFFLINE=1`, `DOCLING_ARTIFACTS_PATH`, tiktoken cache) and only needs the three endpoints.
`compose.yaml` drives build args and runtime settings from a single `.env` (see `.env.example`):
`docker compose --profile local up -d` on the dev box (host networking), `docker compose --profile enterprise
build && docker compose --profile enterprise up -d` on a server (published port, Artifactory/proxy build args).
`/v1/process` is synchronous: give HTTP clients a read timeout of at least `service.request_deadline_s`
(default 1800 s).

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | startup self-check (chunk tokenizer, docling models, docling-graph tokenizer, default ontology); 503 when degraded |
| `GET /v1/capabilities` | versions, OCR engine + languages, formats, configured models/endpoints (keys redacted), chunking, features, limits, live endpoint probe |
| `GET /v1/ontology-schema` | JSON Schema of the ontology contract (below) |
| `POST /v1/process` | the pipeline; one job at a time (`503` + `Retry-After` when busy), `413` oversize, `422` bad base64 / unsupported format / unknown `pipeline_config` key / invalid ontology |

Request:

```json
{
  "document": {"name": "Betriebshandbuch_ZSD.pdf", "format": "pdf", "base64_content": "…"},
  "pipeline_config": {"ocr_enabled": true, "vlm_enabled": true, "embedding_enabled": true,
                      "graph_enabled": true, "chunk_max_tokens": 512, "extraction_contract": "dense"},
  "ontology_graph": { "...": "ontology document; omitted → graph.default_ontology_path" }
}
```

Response (`ProcessResponse`, see `schemas.py`): `document{name, format, sha256, pages, tables, pictures}`,
`markdown`, `chunks[]`, `graph{nodes, edges, meta} | null`, `degraded{vlm, embeddings, graph}`,
`errors[]`, `warnings[]`, `timings_s`, `versions`, `cached`.

Chunk: `chunk_id`, `text` (breadcrumb + body — what gets embedded), `body_text`, `heading_breadcrumb[]`,
`heading_level`, `kind` (text|table|picture|mixed), `caption`, `page_numbers[]`, `dom_paths[]`,
`bboxes[{page,l,t,r,b,page_width,page_height,coord_origin:TOPLEFT}]`, `token_count`, `embedding`.

Graph node: `id`, `type`, `label`, `attributes{}` (normalized: `DD.MM.YYYY`→ISO, decimal comma→point,
enums matched case-insensitively; violations are kept and listed in `meta`), `aliases[]`, `quote`,
`provenance{pages, chunk_ids, dom_paths, heading_breadcrumb, table_ref, figure_ref, match}`.
Graph edge: `source`, `target`, `type`, `polarity` (positive|negative|unknown), `qualifier`, `quote`,
`properties{}`, `provenance{pages, chunk_ids}`. `graph.meta` carries counts, `unresolved_targets`
(link targets the LLM named but no node matched — the SPEC §4.4 review queue; never materialized as
stub nodes), `conflicts` (same edge asserted positive *and* negative — both kept), enum/identity
violations, `polarity_notes`, docling-graph's alias-reconciliation stats, and versions.

## Ontology input contract (the standard format)

The service compiles the ontology into the docling-graph template at request time. The contract is
the structure of `user-manual-books/handbuch_daten/Ontologie/ontology.yaml`; unknown keys are ignored,
so documentation-only sections (`mixins`, `competency_questions`, `negative_assertions`, `extraction`,
`expected_graph_shape`, `table_hints`, `merge_warning` …) pass through. Machine-checked minimum
(`GET /v1/ontology-schema`, `dgs ontology-check FILE`):

```yaml
meta: {name: str, version: str}                                    # required
enums: {EnumName: [value, ...]}                                    # optional
datatypes: {typename: {python: str|int|float|bool|datetime.date, pattern?: regex}}   # optional; unknown types -> str
classes:                                                           # required, >= 1
  ClassName:
    label_de | description: str          # at least one; the discriminating sentence comes first
    identity: {keys: [field, ...]}       # REQUIRED; 1-2 scalar fields, or `label`
    cues_de: [str, ...]                  # recommended: extraction anchors (appended as "Hinweise")
    fields:                              # optional; `label` is implicit on every class
      - {name, type: <datatype|Enum|reference|ClassName>, required?: bool, many?: bool, description?, examples?}
relations:
  - name: UPPER_SNAKE                    # REQUIRED, unique
    source: ClassName | [ClassName...]   # REQUIRED; `NodeBase` = any class
    target: ClassName | [ClassName...]   # REQUIRED; `NodeBase` = any class
    label_de | description: str          # recommended
    cues_de: [str, ...]                  # optional
    properties: [{name, type, description?}]   # optional typed edge attributes
    symmetric: bool                      # optional
```

Implicit on every node: `label`, `aliases[]`, `description`, `quote`; on every edge: `polarity`,
`qualifier`, `quote`. How it is compiled:

* **class → entity** with `graph_id_fields = identity.keys`; only identity fields are required; enums and
  patterned datatypes become plain strings with the allowed values / format in the description and
  are validated *after* extraction (a wrong value never invalidates a batch).
* **relation → scalar link component** `<Name>Link{target_type, target, polarity, qualifier, quote,
  <properties>}` added as a list to every source class. The LLM writes the target's identity value as
  text; the service resolves it against the extracted nodes (identity value → label → aliases,
  case-insensitive; titles like `Dr.` stripped for `strip_titles_then_casefold`). Reason: docling-graph
  edges carry only a label, `nx.DiGraph` keeps one edge per node pair, and nested entity targets do not
  survive its dense contract (first Union member only, recursive branches pruned).
* A synthetic `ExtractionRoot(document_id)` holds one catalog list per class and is dropped on output.
* docling-graph writes template fields over its own node attributes `id`, `label`, `type`; the implicit `label`
  is therefore compiled as `display_name` (and ontology fields named `id`/`type` as `id_value`/`type_value`) and
  mapped back on output. Without this, an empty optional `label` crashed docling-graph after the extraction.
* Relations with `source: NodeBase` (`MENTIONED_IN`) are skipped and reported — they are derived from
  provenance downstream.
* Not representable and documented: `identity.scope: system` (`Component`, `MaintenanceWindow`,
  `FailureMode`) — same-label components of two systems would merge within one document; cross-document
  scoping belongs to the main project's ingestion.

Minimal example (two classes, one relation):

```yaml
meta: {name: mini, version: "1"}
enums: {Kind: [fachverfahren, basisdienst]}
datatypes: {ticket_id: {python: str, pattern: '^T-\d+$'}}
classes:
  System:   {label_de: Verfahren, identity: {keys: [label]}, fields: [{name: kind, type: Kind, required: true}]}
  Incident: {label_de: Störung, identity: {keys: [ticket_id]}, fields: [{name: ticket_id, type: ticket_id, required: true}]}
relations:
  - {name: AFFECTS, source: Incident, target: System, label_de: betrifft}
```

## Configuration

`config.yaml` holds the defaults; every key is overridable with `DGS__<SECTION>__<KEY>` (lists as
JSON), secrets via environment / `.env`. `DGS_CONFIG=/path/config.yaml` selects another file.

| Section | Keys (defaults) |
|---|---|
| `llm` | `base_url` (http://localhost:4000/v1), `api_key`, `model` (gemini-dev), `structured_output` (true), `temperature` (0.0), `timeout_s` (300), `max_retries` (2), `context_limit` (128000), `max_output_tokens` (8192), `parallel_workers` (2) |
| `vlm` | `enabled`, `base_url`/`api_key` (default to `llm`), `model` (gemini-dev), `prompt` (German), `timeout_s`, `concurrency` (2), `max_tokens` (400), `min_area_fraction` (0.0) |
| `embedding` | `enabled`, `base_url`/`api_key`, `model` (bge-m3), `dim` (1024), `batch_size` (64), `text_prefix` (`""`; `"passage: "` for e5), `timeout_s` |
| `chunking` | `tokenizer` (intfloat/multilingual-e5-large — XLM-R vocabulary shared with bge-m3), `max_tokens` (512), `merge_peers` (true), `strip_line_prefixes` (["TESTDOKUMENT ·"]), `repeated_furniture_min_pages` (3), `furniture_band` (0.12) |
| `docling` | `ocr` (true), `ocr_langs` ([de, en]), `images_scale` (3.0), `table_mode` (accurate), `heading_hierarchy` (true), `numbered_heading_levels` (true), `num_threads` (4), `artifacts_path`, `document_timeout_s` (900) |
| `graph` | `enabled`, `extraction_contract` (dense), `dense_dedupe` (off), `provenance` (standard), `chunk_max_tokens` (512, docling-graph's internal chunker), `default_ontology_path` |
| `service` | `host`, `port` (8080), `max_upload_mb` (50), `request_deadline_s` (1800), `cache_dir`, `log_level` |

Switching to the target cluster is configuration only: `DGS__LLM__MODEL=<qwen3-coder alias>`,
`DGS__VLM__MODEL=<granite-vision alias>`, `DGS__EMBEDDING__MODEL=multilingual-e5-large`,
`DGS__EMBEDDING__TEXT_PREFIX="passage: "`, plus the `base_url`s.

## Behaviour worth knowing

* **Page furniture**: docling puts detected headers/footers in the FURNITURE layer (excluded from
  Markdown and chunks). Two extra rules catch what the layout model misses: configured line prefixes,
  and text repeated in the top/bottom page band on ≥ 3 pages (digits ignored: `Seite 3 von 29`).
* **Heading levels**: layout-derived levels are unreliable on manuals (chapter titles and subsections
  land on the same visual level). `numbered_heading_levels` derives the level from the numbering
  (`5.3.1 …` → 3; unnumbered headings → parent + 1), which gives complete breadcrumbs.
* **Tables**: Markdown (compact), never merged with text, caption **and** header row on every part
  (page continuations inherit the caption of the table with the same header; docling repeats the header
  on row-group splits, the service adds the caption). Any chunk still over budget is split on
  row/line → sentence → word boundaries, never truncated.
* **Pictures**: our own VLM step (not docling's picture-description hook) writes
  `PictureItem.meta.description`, so descriptions flow into Markdown and chunks unchanged; per-picture
  failures are explicit. Figure captions that docling did not link are inferred from the neighbouring
  `Abbildung N:` text.
* **Polarity** is never silently positive: `negativ/verneint/kein…` → negative; a missing polarity with a
  negation trigger in the quote/qualifier → `unknown` + note; positive+negative for the same edge are
  both kept and listed in `meta.conflicts`.
* **Determinism / cost**: `extraction_contract=dense` is pinned (`auto` flips between one giant call and
  dense depending on `max_tokens`), `dense_dedupe=off` and `temperature=0`. docling-graph's graph-level
  alias reconciliation (one LLM confirmation call) cannot be switched off via `PipelineConfig`; its
  merges are surfaced in `meta.alias_reconciliation`.

## ZSD results

Reference run 2026-08-27 (`DGS_INTEGRATION=1 pytest tests/integration`, 12/12 passed, native, CPU-only WSL2,
`gemini-dev` = gemini-3.7-flash for LLM + VLM, `bge-m3` embeddings, all via the LiteLLM proxy). Graph numbers vary
slightly between runs (LLM); the integration test therefore asserts thresholds. Raw output:
`out/zsd-integration/response.json`.

**Conversion & chunking — the NEXT-STEPS §1 spike answers**

| Question | Result |
|---|---|
| Pages / tables / pictures | 29 pages, 22 TableItems (15 tables; page-spanning ones arrive as one item per page), 3 pictures |
| Page header/footer | `TESTDOKUMENT · …`, `BHB-PLT-0007 · Version 2.3 …`, `Seite N von 29` appear in **no** chunk and not in the Markdown (docling FURNITURE layer + 20 items caught by the repetition rule on p. 3) |
| Table captions | all 15 `Tabelle N:` captions present (Tabelle 1, Tabelle 2, Tabelle 3 … Tabelle 15); 7 captions inferred (6 page continuations, Abbildung 1) |
| Tabelle 6 Konsumentenmatrix | one table chunk, 6 columns × 6 rows, caption + header, wrapped cells intact (`bavd-issuing-ca-3`, `kv/vpp/monitor`, `Sonja Wiechert`) after de-hyphenation of 24 cells |
| Tabelle 8 (spans p. 18–19) | 2 parts ([18] + [19]), each starting with `Tabelle 8:` + header row; 5 events × 4 consumers, 4 literal `keine` cells, `unberührt` cell intact |
| Heading breadcrumbs | `5 Betrieb > 5.3 Standardabläufe > 5.3.1 Keycloak rollierend neu starten (SOP-ZSD-02)`; 105/112 chunks with ≥ 2 levels (layout levels alone gave a broken hierarchy; fixed by numbering-based normalization) |
| Identifiers | `ZSDSUP-0247`, `BHB-PLT-0007`, `SOP-ZSD-06`, `FW-ZSD-004`, `kv/vpp/monitor`, `CAASUP-0351` intact in chunk text |
| Token budget | 112 chunks (26 table, 3 picture, 83 text), max 512 tokens incl. special tokens (limit 512); 2 tables split by the cap guard after caption prepending |
| Figures at `images_scale=3.0` | 1537 px wide crops of the page raster (the embedded PNGs are 4800 px — docling re-renders, it does not extract the originals); labels are readable for the VLM |
| Timings (cached conversion) | convert 0.362 s (first conversion ≈ 290 s on this CPU), describe 29.154 s, chunk 0.403 s, embed 76.867 s, graph 655.607 s |

**Embeddings**: 112/112 chunks with 1024-dim vectors (`bge-m3`, batches of 64).

**VLM**: 3/3 German descriptions (812, 1187, 1028 chars in the picture chunks), rendered after each figure in the
Markdown. `max_tokens` must cover the model's reasoning tokens (400 truncated gemini-3.x to 14 visible tokens).

**Graph** (`degraded.graph = false`, 279 nodes, 145 edges, 7 negative, 18 unresolved targets, 0 conflicts):

| Check | Result |
|---|---|
| Document / System | `BHB-PLT-0007` Betriebshandbuch ZSD; System `ZSD - Zentrale Sicherheitsdienste` (+ the four consumer systems) |
| Incidents | 7/7 `ZSDSUP-*` (ZSDSUP-0119, ZSDSUP-0166, ZSDSUP-0208, ZSDSUP-0214, ZSDSUP-0231, ZSDSUP-0247, ZSDSUP-0252) + partner tickets as their own Incident nodes |
| PARTNER_TICKET | 18 edges — all six pairs of Tabelle 9 (0119↔DDSUP-0794, 0208↔VPPSUP-2251, 0214↔ESSUP-1455/1517, 0247↔CAASUP-0351/DDSUP-1201), both directions |
| Procedures | 11 (`SOP-ZSD-01…06` + referenced SOPs) |
| Persons | 19 incl. Kai Ostermann, Sabine Wollmer, Marcel Ebert, Dr. Annika Reuß (surname-only mentions such as `Ebert` survive as separate nodes — merging is downstream) |
| IdentityClients | 10 incl. vpp-portal, dd-gateway, es-grafana, caas-console, obs-vpp |
| ImpactStatements | 20 (Tabelle 8: 5 events × 4 consumers), severities {'keine': 5, 'eingeschraenkt': 13, 'ausfall': 2} |
| Hosts / FirewallRules / StartupSteps | 35 / 16 / 11 (`PRECEDES` chain 11) |
| Negative edges | 7: VPP Portal ⇸ CaaS-Plattform (TENANT_OF); VPP ⇸ CaaS-Plattform (TENANT_OF); Keycloak rollierend, mindestens 2 Knoten aktiv | VPP ⇸ IAM (IMPACT_OF); Keycloak rollierend, mindestens 2 Knoten aktiv | Mars Dokumentendienste ⇸ IAM (IMPACT_OF) … |
| Provenance | every node has pages + chunk ids; every edge has chunk grounding |
| Unresolved targets | 18 — 3 name a node whose class the relation does not allow (e.g. `AFFECTS → pki-p01` is a Host), 15 have no node at all (composite labels like `Mars Dokumentendienste, dd-gateway`, fire sections `SD-A`) |
| Identity pattern violations | 11 kept + flagged (e.g. related documents `NET-ZK-004`, host ranges `pg-iam-p01 / p02.bavd.intern`) |
| docling-graph alias reconciliation | {'candidates': 19, 'confirmed': 9, 'merged': 0, 'vetoed_sibling': 9} |

Known quality limits of this run, all visible in `graph.meta`: the LLM sometimes models components as `System`
(24 System nodes); negated statements are occasionally emitted with the wrong relation type
(`LOCATED_IN` instead of `RUNS_ON` for "IAM/PKI laufen nicht auf der CaaS-Plattform") and then land in
`unresolved_targets` with polarity `negative` instead of becoming edges.

**Docker**: same request against the `dgs:local` image (`docker run --network host …`, models/tokenizers
baked in, `HF_HUB_OFFLINE=1`): 200 after 686 s, `degraded = {vlm: false, embeddings: false, graph: false}`,
255 nodes / 131 edges; `/healthz` reports all four offline checks `ok`. A second identical POST is served from
the result cache in 0.1 s.


## Downstream: indexing and retrieval

How the three artefacts join, what to index in OpenSearch, how to resolve start nodes and traverse, and
what the graph can/cannot answer today: see [`graph-retrieval-patterns.md`](graph-retrieval-patterns.md).

## Limitations and follow-ups

* Sync API only. If the target ingress cannot hold a connection for a long conversion, add a
  `202 + job id` layer over `pipeline.process()`.
* ADR-0010 "confirm" job (VLM confirming already-extracted edges) — a post-step over `graph.edges`
  reusing `describe.describe_picture()`.
* Cross-document merge / identity normalization (`strip_titles_then_casefold`, `scope: system`) —
  the main project's `ingestion/`; this service outputs raw identity attributes plus resolved edges.
* Link targets that the LLM names but no node matches are reported (`meta.unresolved_targets`), not
  created — by design (SPEC §4.4). Partner tickets of *other* documents therefore appear there until the
  cross-document merge links them.
* No GPU path is exercised here (the dev box is CPU-only); `AcceleratorDevice` is fixed to CPU by design
  for the container (see `convert.py`).
* Proxy-side items outside this repo: `num_ctx` for local Ollama aliases, a local vision alias, request
  timeouts, moving the Gemini key out of the LiteLLM `config.yaml`.
