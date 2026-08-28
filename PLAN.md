# Plan: `docling-graph/` — Docling ingestion & graph-extraction service (module 1)

## Context

The RAG prototype (SPEC.md) needs one ingestion module that turns a Betriebshandbuch PDF into three
synchronized artefacts: enriched Markdown, provenance-carrying chunks with embeddings, and an
ontology-grounded knowledge graph. `docling-graph/spec.md` specifies it as a small stateless
FastAPI/Docker service (`GET /v1/capabilities`, `POST /v1/process`) that offloads every model call
to OpenAI-compatible endpoints behind the LiteLLM proxy (`abhängigkeiten.md`). It is tested against
`user-manual-books/handbuch_daten/handbuch/Betriebshandbuch_ZSD.pdf` (BHB-PLT-0007) with the real
`Ontologie/ontology.yaml`.

Research against the *current* library sources (docling 2.123.0 / docling-core 2.92.0 /
docling-graph 1.9.1, all released Jul–Aug 2026), the local machine, and an adversarial design
review changed several assumptions in the specs. These findings drive the design:

1. **docling-graph cannot do our conversion** (hard-coded en/fr OCR, no `images_scale`, no picture
   description) but accepts an already-converted **DoclingDocument JSON file** as `source` and
   skips conversion. → Convert with docling ourselves, write `<source-stem>.json`, hand it over.
2. **docling-graph edges carry only a label; `nx.DiGraph` keeps one edge per node pair; nested
   entity targets inside components do not survive the dense contract** (Union → first model
   only; recursive branches like System→…→System are skipped by the skeleton catalog). The
   ontology needs `polarity` + `qualifier` on every relation and typed properties on five.
   → Relations are compiled into **scalar link components**: `{target_type, target (identity
   value as written), polarity, qualifier, quote, <properties>}`. docling-graph creates nodes,
   binds provenance and dedupes identities; the service resolves link targets against the final
   node set and emits edges with attributes. No recursion, no Unions, no auto-created nodes
   (SPEC §4.4: unmapped mentions are reported, not silently created).
3. **The project's generated `ontology_model.py` (flat `GraphDocument{nodes, edges}`) is not a
   docling-graph template**, and docling-graph's `templategen` cannot read `ontology.yaml`. → The
   service compiles the `ontology.yaml` structure into a Pydantic template at request time; that
   structure is declared the service's standard ontology contract (user decision).
4. **This machine has no CUDA GPU** (AMD Radeon 840M; Ollama, EasyOCR, torch on CPU). SPEC §10.1's
   RTX 4080 is not this box. Local runs prove plumbing only.
5. **`granite4` has no vision** (hard 400 on image input); `gemini-dev` on the proxy is multimodal
   and is used for both extraction and vision in the ZSD test (user decision; corpus is fictional,
   SPEC §11.1). Ollama binds `127.0.0.1` only; a container reaches LiteLLM (`0.0.0.0:4000`) but
   never Ollama. `docker compose` plugin is **not installed**; `host.docker.internal` is not
   auto-injected (native dockerd in WSL).
6. `canon.md` has no ZSD sections (stale 3-document version) → ZSD acceptance expectations come
   from the PDF itself (extracted glyph-exact during research; see Verification).
7. Library behaviour that must be set explicitly or output silently degrades:
   `HeadingHierarchyOptions(enabled=True)` + `generate_parsed_pages=True` (else all headings are
   level 1 → flat breadcrumbs); `EasyOcrOptions(lang=["de","en"])` (default is auto/RapidOCR);
   page header/footer are `ContentLayer.FURNITURE` and already excluded from Markdown, iteration
   and chunking; chunk tables default to *triplets* (swap to Markdown); `HybridChunker` repeats the
   **header but not the caption** on row-group splits; `HuggingFaceTokenizer` needs explicit
   `max_tokens` offline; docling-graph model names for the proxy must be `openai/<alias>`;
   `extraction_contract="auto"` flips between one giant call and dense depending on `max_tokens`;
   `dense_dedupe="standard"` and graph alias reconciliation make extra LLM calls (SPEC §4.4 says no
   LLM canonicalization); docling-graph's internal chunker needs tiktoken's `cl100k_base` file
   (network) → must be baked into the image; `run_pipeline` raises `PipelineError` with no partial
   result; docling-graph already falls back from strict structured output to prompt mode itself.

Result: a self-contained, tested service in `docling-graph/` that processes ZSD end to end
(natively and in Docker) and answers the seven NEXT-STEPS §1 spike questions from real output.

## Decisions (confirmed by the user 2026-08-27)

- **Extraction LLM for the ZSD test:** `gemini-dev` via LiteLLM. Everything stays configurable;
  the target cluster runs `qwen3-coder` by changing one alias.
- **VLM for the ZSD test:** `gemini-dev`. "No VLM" is handled gracefully; a local `qwen3.5:2b`
  alias remains a documented option.
- **Ontology contract = the `ontology.yaml` structure**, declared as the standard input format.
  The user will shape the real-data ontology to it, so the minimal required components are
  explicit and machine-checked (next section).
- **Verification depth:** native run + Docker image build and container run.

## Ontology input contract (the standard format)

One ontology document per request: YAML file on disk (`graph.default_ontology_path`) or JSON in
`ontology_graph`. It is the existing `ontology.yaml` structure; unknown keys are ignored so the
project file works unchanged. **Required minimum**, validated by the `Ontology` pydantic model with
precise errors (e.g. `classes.Host.identity.keys[0]: 'hostname' is not a field of Host`):

```yaml
meta: {name: str, version: str}
enums: {EnumName: [value, ...]}                                   # optional
datatypes: {typename: {python: str|int|float|bool, pattern?: str}} # optional; unknown types -> str
classes:
  ClassName:
    description | label_de: str          # at least one → class docstring (discriminating sentence first)
    identity: {keys: [field, ...]}       # REQUIRED; 1–2 scalar fields, or `label`
    cues_de: [str, ...]                  # recommended: extraction anchors, appended to the docstring
    fields:                              # optional; `label` is implicit on every class
      - {name, type: <datatype|Enum|reference|ClassName>, required: bool, many?: bool, description?, examples?}
relations:
  - name: UPPER_SNAKE                    # REQUIRED, unique
    source: ClassName | [ClassName...]   # REQUIRED; `NodeBase` = any class
    target: ClassName | [ClassName...]   # REQUIRED; `NodeBase` = any class
    label_de | description: str          # recommended
    cues_de: [str, ...]                  # optional
    properties: [{name, type, description?}]   # optional edge attributes
    symmetric: bool                      # optional
```
Implicit on every node: `label`, `aliases[]`, `description`, `quote`; on every edge: `polarity`,
`qualifier`, `quote`. Not read (documentation/eval only): `mixins`, `negative_assertions`,
`competency_questions`, `extraction`, `expected_graph_shape`, `table_hints`, `merge_warning`.
Known non-representable item recorded in README: `identity.scope: system` (same-label components
of two systems merge within one document; cross-document scoping is downstream's job).
The contract is (1) in README with a minimal 2-class example, (2) exposed as JSON Schema at
`GET /v1/ontology-schema`, and (3) checkable offline with `dgs ontology-check <file>` (prints
classes + identity keys, relations with resolved endpoints, skipped items such as `MENTIONED_IN`
(source `NodeBase`, documented as derived), and the size of the generated LLM schema).

## Architecture (one package, one pipeline function, two thin fronts)

```
docling-graph/
├── pyproject.toml            # uv; deps: docling[easyocr], docling-graph, fastapi, uvicorn, httpx, pydantic-settings, pyyaml, transformers (tokenizer), typer (already a docling-graph dep)
├── uv.lock
├── README.md                 # usage, config table, API + ontology contract, ZSD results (spike answers), limitations/follow-ups
├── spec.md                   # (existing)
├── config.yaml               # defaults; every key overridable via env DGS__SECTION__KEY
├── .env.example
├── Dockerfile                # python:3.12-slim + uv, CPU torch, models + tokenizers + tiktoken cache baked in; ARGs for PyPI index / base image / HF endpoint
├── compose.yaml              # profiles local (network_mode: host) / enterprise (env URLs); marked "not exercised here: compose plugin absent"
├── Makefile                  # dev, test, test-integration, docker-build, docker-run, process-zsd
├── src/docling_graph_service/
│   ├── settings.py           # pydantic-settings
│   ├── schemas.py            # ProcessRequest/Options, Chunk, GraphNode, GraphEdge, GraphResult, ProcessResponse, Capabilities
│   ├── ontology.py           # Ontology model + compile_template() -> CompiledTemplate (root class, ClassSpec/RelationSpec registries)
│   ├── convert.py            # cached DocumentConverter(s), convert(path) -> DoclingDocument, strip_page_furniture(), markdown export
│   ├── describe.py           # picture description via /chat/completions (own step, not docling's hook); writes PictureItem.meta.description
│   ├── chunking.py           # HybridChunker (XLM-R tokenizer, Markdown tables, header repeat) + caption repeat + token-cap guard -> Chunk[]; SelfRefIndex
│   ├── llm_http.py           # one httpx client with retry/backoff/timeouts shared by describe.py and embed.py
│   ├── embed.py              # batched /v1/embeddings with prefix and dim check
│   ├── graph.py              # docling-graph run + materialize(nodes, edges) + normalization + identity/enum validation
│   ├── pipeline.py           # process(): convert → strip → describe → markdown → chunk → embed → graph; timings; degradation; result cache
│   ├── api.py                # FastAPI: /healthz, /v1/capabilities, /v1/ontology-schema, /v1/process; CapacityLimiter(1)
│   └── cli.py                # dgs process | serve | capabilities | ontology-check
└── tests/
    ├── unit/                 # no network
    ├── integration/          # ZSD end to end; skipped unless DGS_INTEGRATION=1
    └── data/                 # small committed PDF fixture (1 page, one table, one heading)
```
Rules: only `api.py` imports FastAPI; `pipeline.process()` is the single orchestration point (API
and CLI call it); the only process-wide state is the cached converter(s), the chunk tokenizer and
the httpx client, all built at startup and self-checked in `/healthz`.

## Component design

### 1. `settings.py`
Nested `Settings(BaseSettings)` loaded from `config.yaml` then env (`DGS__LLM__MODEL=…`):
- `llm`: `base_url` (`http://localhost:4000/v1`), `api_key`, `model` (proxy alias), `structured_output: true`,
  `timeout_s: 300`, `max_retries: 2`, `context_limit`, `max_output_tokens` (single source for
  docling-graph's `generation.max_tokens` **and** `llm_overrides.max_output_tokens`), `parallel_workers: 2`.
- `vlm`: `enabled`, `base_url`, `api_key`, `model`, `prompt` (German: describe the technical
  diagram in 2–4 sentences, name systems/components/arrows/labels; the figure caption is passed as
  context), `timeout_s`, `concurrency: 2`, `max_tokens: 400`, `min_area_fraction: 0.0`.
- `embedding`: `enabled`, `base_url`, `api_key`, `model`, `dim: 1024`, `batch_size: 64`,
  `text_prefix` (`"passage: "` for e5, `""` for bge-m3), `timeout_s`.
- `chunking`: `tokenizer: "intfloat/multilingual-e5-large"` (XLM-R vocab shared with bge-m3),
  `max_tokens: 512`, `merge_peers: true`, `strip_line_prefixes: ["TESTDOKUMENT ·"]`.
- `docling`: `ocr: true`, `ocr_langs: ["de","en"]`, `images_scale: 3.0`, `table_mode: accurate`,
  `heading_hierarchy: true`, `num_threads: 4`, `artifacts_path` (`DOCLING_ARTIFACTS_PATH`), `document_timeout_s`.
- `graph`: `enabled`, `extraction_contract: "dense"` (`direct` only as explicit override),
  `dense_dedupe: "off"`, `provenance: "standard"`, `chunk_max_tokens: 512` (docling-graph's
  *internal* chunker, distinct from embedding chunks), `default_ontology_path`.
- `service`: `max_upload_mb: 50`, `request_deadline_s: 1800`, `cache_dir` (optional; caches the
  DoclingDocument JSON **and** the full `ProcessResult`, keyed by `sha256(pdf)+options+ontology hash`
  → retries after a client timeout return instantly, i.e. idempotency for free).

### 2. `ontology.py` — ontology → docling-graph template
`compile_template(ont) -> CompiledTemplate{root, classes: dict[name, ClassSpec], relations: dict[name, RelationSpec]}`
builds classes with `type()`/`create_model` and `ConfigDict(graph_id_fields=…, is_entity=…, extra="ignore", populate_by_name=True)`
(unit test asserts these keys survive in `model_config` — docling-graph reads them via `.get`).

- **Entity per class.** Docstring: `label_de`/`description` first (the discriminating sentence,
  ≤240 chars up front), then `Hinweise: cues_de`. Fields: `label: str` (required when it is the
  identity key; for classes whose identity is another field, `label` is optional and the
  materializer falls back to the identity value), `aliases: list[str]`, `description`, `quote`
  ("wörtlicher Beleg, ≤200 Zeichen"), plus the class's own fields. **Identity** = `identity.keys`
  restricted to scalar fields (≤2), fallback `["label"]`; identity fields are the only required
  fields (docling-graph's own templates follow this). Types: enums → `str` with allowed values in
  the description (a wrong value must not invalidate a batch; violations reported in `meta`),
  datatypes → `str/int/float/bool` per `python`, **no regex constraints** (patterns → description,
  validated post-hoc), `date/duration/percent` → `str`, `reference`/class-typed → `str`
  (identity value), `many` → `list[…]`; `examples` carried over.
- **Scalar link component per relation** `<Name>Link` (`is_entity=False`): `target_type: str`
  (allowed class names listed; single-target relations get a fixed default), `target: str`
  ("Identitätswert des Ziels wie im Text, z. B. Label/Ticket-ID/Hostname"), `polarity: str|None`
  (`positive`/`negative`; the `extraction_rule_negation` triggers in the description), `qualifier`,
  `quote`, and each `properties[*]` as a typed optional field. `<snake_name>: list[<Name>Link]` is
  added to **every source class**; docstrings say every relation target must also appear in its
  class list. Relations with source `NodeBase` (`MENTIONED_IN`) are skipped and reported.
- **Root** `ExtractionRoot` entity (`graph_id_fields=["document_id"]`, described as the source
  document id) with one catalog list per class (`edge_label="__CATALOG__"`). Kept synthetic so the
  compiler needs no "root class" knob; root node and catalog edges are dropped on materialization.
  (Alternative "use the ontology's Document class as root" rejected: adds a configuration branch
  for the same result.)

### 3. `convert.py` + `describe.py`
One `DocumentConverter` per OCR mode (two at most, built lazily, cached — options are part of the
cache key so they are never mutated): `PdfPipelineOptions(do_ocr, ocr_options=EasyOcrOptions(lang, download_enabled=False when artifacts_path), do_table_structure=True, table_structure_options=TableStructureOptions(mode=ACCURATE), images_scale=3.0, generate_page_images=True, generate_picture_images=True, generate_parsed_pages=True, heading_hierarchy_options=HeadingHierarchyOptions(enabled=True), accelerator_options=AcceleratorOptions(device, num_threads), document_timeout)`.
Picture description is **our own step** (`describe.py`), not docling's `PictureDescriptionApiOptions`
hook: for each `PictureItem` with `get_image(doc)` above `min_area_fraction`, POST an OpenAI
`image_url` (PNG data URL) + German prompt (with the figure caption) to `/chat/completions`
through `llm_http`, then set `pic.meta.description = DescriptionMetaField(text, created_by=model)`
— the same field docling's hook writes, so descriptions flow into Markdown and chunks unchanged.
Why: VLM on/off no longer needs a second converter; failures are explicit (docling's hook swallows
errors into empty text); the ADR-0010 "confirm" follow-up reuses the same call. Failures → the
picture keeps no description, `degraded.vlm=true`, per-picture error in `meta.errors`.
`strip_page_furniture(doc, prefixes)`: belt and braces on top of docling's FURNITURE layer — any
BODY text item starting with a configured prefix is moved to FURNITURE; count reported.
Markdown: `doc.export_to_markdown(image_mode=PLACEHOLDER, page_break_placeholder="<!-- page break -->", mark_meta=True, allowed_meta_names={"description"}, escape_underscores=False)`.

### 4. `chunking.py`
`HybridChunker(tokenizer=HuggingFaceTokenizer(AutoTokenizer.from_pretrained(name), max_tokens=budget), merge_peers=True, repeat_table_header=True, serializer_provider=MarkdownTableProvider())`
(provider = `ChunkingDocSerializer(table_serializer=MarkdownTableSerializer(), params=MarkdownParams(...))`).
Budget = `max_tokens − len(prefix tokens) − 2` (special tokens); the final check uses
`add_special_tokens=True`. **Caption repeat:** a chunk whose `doc_items` is a single `TableItem`
and whose text does not start with `table.caption_text(doc)` gets the caption prepended (SPEC §5.1:
caption *and* header on every part), then the cap is re-checked and rows re-split if needed.
**Hard cap:** any chunk still over budget is split on line/sentence boundaries with a warning —
never truncated (SPEC risk #4). Per chunk: `chunk_id` (`{doc_hash[:12]}-{idx:04d}`), `text`
(= `contextualize()`: breadcrumb + body), `body_text`, `heading_breadcrumb`, `heading_level`,
`kind` (text/table/picture/mixed), `caption`, `page_numbers`, `dom_paths` (`self_ref`s), `bboxes`
(per prov: page, l/t/r/b converted to **TOPLEFT**, page size), `token_count`. `SelfRefIndex`:
`self_ref -> [chunk_id]`.

### 5. `llm_http.py`, `embed.py`
`llm_http.Client`: httpx with connect/read timeouts, 3 attempts with backoff on 429/5xx/timeouts,
Bearer auth; used by describe/embed and for the capabilities probe. `embed_texts()` batches
`input` by `batch_size`, preserves order by `index`, verifies `dim` and count; failure → chunks
without embeddings, `degraded.embeddings=true`.

### 6. `graph.py`
Run: `doc.save_as_json(tmp/<source-stem>.json)` (the stem is docling-graph's last-resort root
identity), then `run_pipeline(PipelineConfig(source=path, template=compiled.root, backend="llm", inference="remote", provider_override="openai", model_override=f"openai/{llm.model}", processing_mode="many-to-one", extraction_contract=graph.extraction_contract, use_chunking=True, chunk_max_tokens=graph.chunk_max_tokens, parallel_workers=llm.parallel_workers, dense_dedupe=graph.dense_dedupe, provenance="standard", structured_output=llm.structured_output, dump_to_disk=False, gc_collect=False, llm_overrides={"connection": {"base_url", "api_key"}, "generation": {"temperature": 0.0, "max_tokens": M}, "reliability": {"timeout_s", "max_retries"}, "context_limit": C, "max_output_tokens": M}))`.
A unit test asserts every key passed exists in `PipelineConfig.model_fields` (it ignores unknown
keys silently). No home-made structured-output retry: docling-graph already falls back to prompt
mode; we log whether it did. `PipelineError` → `graph=None`, `degraded.graph=true`,
`errors.append(details["reason"])`.
**Materialize** (`GraphResult{nodes, edges, meta}`):
- Instances: walk only the root's flat catalog lists; group by `ctx.node_registry.get_node_id(inst)`
  (public API); union link lists across duplicates (the converter's enrichment keeps only the first
  instance's attributes).
- Nodes ← `ctx.knowledge_graph` (minus root): `id`, `type` (`__class__`), `label` (fallback:
  first identity value), `attributes` (template fields minus links/internals; values normalized per
  `extraction.normalization`: `DD.MM.YYYY`→ISO, decimal comma→point for float/percent, NBSP
  thousands removed for int), `aliases` (+ converter `merged_aliases`), `quote`, `provenance`
  (`__provenance__`: `pages`, `refs`, `match` → `chunk_ids` via `SelfRefIndex`, `heading_breadcrumb`,
  `table_ref`/`figure_ref` when a ref is a table/picture). Ids that the converter merged away are
  remapped through the survivor's `merged_aliases`; the rest counted in `meta.dropped_by_converter`.
  Nodes with empty/placeholder identity are dropped and reported; identity values failing a
  `datatypes[*].pattern` are **kept and flagged** (`meta.identity_pattern_violations`).
- Edges ← links: resolve `(target_type, target)` against the final node set by identity value →
  `label` → `aliases` (casefold, whitespace-normalized, `Dr.`-style titles stripped for Person);
  unresolved → `meta.unresolved_targets` (SPEC §4.4 review queue), no stub nodes. Edge = `{source,
  target, type, polarity, qualifier, quote, properties, provenance}`. **Polarity** normalized
  (`negativ|negative|verneint|nein|kein*` → negative; `positiv|positive|ja` → positive); `None` with
  a negation trigger in `quote`/`qualifier` → `unknown` + warning, never silently positive.
  Provenance = chunks containing the (normalized) `quote`, else the source node's chunks. Dedupe
  on `(source, target, type, polarity)`; both polarities for one `(source,target,type)` are kept
  and listed in `meta.conflicts`. Symmetric relations emitted per direction found, not mirrored.
- `meta`: counts by type, `docling_graph_version`, template schema hash, `alias_reconciliation`
  and other `graph.graph[...]` stats, enum violations, unresolved/dropped lists, timings.

### 7. `pipeline.py`, `api.py`, `cli.py`
`process(pdf_bytes, name, options, ontology|None) -> ProcessResult`: convert → strip → describe →
markdown → chunk → [embed] → [graph], each optional stage try/except with `degraded.*` flags,
`errors[]`, per-stage timings; conversion failure is a hard error. One **request deadline**
(`service.request_deadline_s`) is budgeted onto `document_timeout`, VLM/embed timeouts and
docling-graph's `timeout_s × (max_retries+1) × calls` so a job is bounded. Result cache (§1).
`api.py`: `POST /v1/process` body per spec (`document{name, format, base64_content}`,
`pipeline_config?` — accepted keys: `ocr_enabled`, `vlm_enabled`, `embedding_enabled`,
`graph_enabled`, `chunk_max_tokens`, `extraction_contract`; anything else → 422 with the list —
`ontology_graph?`); runs in `anyio.to_thread` behind `anyio.CapacityLimiter(1)`, immediate 503
when busy; 413 oversize, 422 bad base64/unsupported format. `GET /v1/capabilities`: versions,
OCR engine + langs, formats (`pdf` first-class; `docx/pptx/html/md` via the same converter),
models + endpoints (keys redacted), chunking algorithm/tokenizer/limit, features, cached live probe
per endpoint. `GET /v1/ontology-schema`, `GET /healthz` (startup self-check: converter, chunk
tokenizer, docling-graph internal tokenizer (`tiktoken cl100k_base`) all loadable offline).
`cli.py` (typer): `process` (writes `document.json`, `markdown.md`, `chunks.json`, `graph.json`,
`meta.json` to `--out`; flags `--no-vlm/--no-graph/--no-embed`, `--ontology`), `serve`,
`capabilities`, `ontology-check`.

### 8. Docker
`Dockerfile`: `ARG BASE_IMAGE=python:3.12-slim`, `ARG PIP_INDEX_URL`, `ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`, `ARG HF_ENDPOINT`; apt `libgl1 libglib2.0-0`; `uv sync --frozen --no-dev`;
`docling-tools models download layout tableformer easyocr --easyocr-lang de --easyocr-lang en -o /opt/docling-models`;
pre-download the chunk tokenizer into `HF_HOME` and tiktoken `cl100k_base` into `TIKTOKEN_CACHE_DIR`;
`ENV DOCLING_ARTIFACTS_PATH=/opt/docling-models OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1`; non-root;
`CMD uvicorn docling_graph_service.api:app --host 0.0.0.0 --port 8080`. README states the client
read timeout callers need (sync API). `compose.yaml` ships both profiles but is marked untested
here; `make docker-run` uses `docker run --network host`.

## Verification (end to end)

1. **Unit** (`uv run pytest tests/unit`): template compiles from the real `ontology.yaml` (26
   entity classes, 31 link classes, root with 26 catalogs; `model_json_schema()` succeeds; config
   keys survive); `ontology-check` output; synthetic `extracted_models` with duplicates, a
   negative link, an unresolved target and a `Dr.` title materialize correctly; polarity
   normalization; caption repeat + token cap on a synthetic wide table; bbox TOPLEFT conversion;
   embed batching/ordering/dim check and describe() with `httpx.MockTransport`; degradation flags;
   `PipelineConfig` key assertion; API 413/422/503 with `TestClient`.
2. **ZSD conversion + chunking** (`dgs process … --no-graph --no-embed --no-vlm`), asserted in
   `tests/integration/test_zsd.py` and reported in README as the NEXT-STEPS §1 answers: 29 pages;
   15 `TableItem`s with captions `Tabelle 1..15`; Tabelle 6 = 6 columns × 6 rows (VPP Portal /
   dd-gateway / es-grafana / Kafka / caas-console / obs-vpp); Tabelle 8 (page-spanning) = 5 events ×
   4 consumers with four literal `keine` cells; every chunk part of Tabelle 6/8 contains
   `Tabelle N:` and the header row; `TESTDOKUMENT · …`, the `BHB-PLT-0007 · Version 2.3` runner
   and `Seite N von 29` appear in no chunk and not in the Markdown; 3 `PictureItem`s with images
   (report the pixel size produced at `images_scale=3.0` — docling crops the page raster, so
   ~1500 px wide, not the embedded 4800 px; stated honestly against `diagramme-png/zsd*.png`);
   identifiers `ZSDSUP-0247`, `BHB-PLT-0007`, `SOP-ZSD-06`, `FW-ZSD-004`, `kv/vpp/monitor` intact in
   chunk text; heading levels ≥ 2 and breadcrumbs with ≥ 2 elements for `5.3.1`-style sections;
   every chunk `token_count ≤ 512` with the prefix.
3. **Embeddings** via LiteLLM `bge-m3`: one vector per chunk, dim 1024, batched.
4. **VLM** (`gemini-dev`): 3 non-empty German descriptions, present in Markdown after the caption
   and in the picture chunks; with VLM unreachable the run completes with `degraded.vlm=true`.
5. **Graph** (`gemini-dev`, threshold assertions; actual counts go into README): `Document`
   `BHB-PLT-0007`; a `System` for Zentrale Sicherheitsdienste; ≥ 4 of 5 `IdentityClient`s;
   `Incident`s for ≥ 6 of 7 `ZSDSUP-*` and ≥ 4 `PARTNER_TICKET` edges; `Procedure`s SOP-ZSD-01…06;
   `Person`s Kai Ostermann, Sabine Wollmer, Marcel Ebert; ≥ 10 `ImpactStatement`s incl.
   `severity=keine`; ≥ 1 `polarity=negative` edge (VPP ⇸ Vault or IAM/PKI ⇸ CaaS); every node has
   `provenance.pages`; every edge type ∈ the 32 relations; no duplicate `(source,target,type,polarity)`;
   `degraded.graph == false`; unresolved targets reported, not materialized.
6. **HTTP + Docker**: `uv run dgs serve`; `GET /healthz`, `/v1/capabilities`, `/v1/ontology-schema`;
   `POST /v1/process` with the base64 ZSD + ontology JSON → same assertions; second identical POST
   returns from cache. Then `make docker-build`, `docker run --network host …`, repeat the requests
   against the container and assert `degraded.graph == false` (proves offline tokenizers/models).
7. **Docs**: README with API + ontology contract, configuration table, native/Docker run, ZSD
   results table, limitations and follow-ups.

## Out of scope / follow-ups (recorded in README)
- ADR-0010 "confirm" job (VLM confirming already-extracted edges) — a post-step over
  `graph.edges` reusing `describe.py`.
- Async job API (202 + job id) if sync calls are too long for the target ingress.
- Cross-document merge / identity normalization (`strip_titles_then_casefold`, `scope: system`) —
  main project's `ingestion/`; this service outputs raw identity attributes plus resolved edges.
- Graph-level alias reconciliation inside docling-graph cannot be disabled via `PipelineConfig`;
  its merges are surfaced in `meta` for audit.
- Proxy-side items outside this repo: `num_ctx` for local Ollama aliases, a local vision alias,
  request timeouts, moving the Gemini key out of `config.yaml`.

---

# Appendix A — Verified API facts (primary sources, 2026-08-27). Use these, not memory.

## docling-graph 1.9.1 (`pip install docling-graph`; pins docling>=2.105,<3, docling-core[chunking,chunking-openai]>=2.86, litellm)
- Entry: `from docling_graph import PipelineConfig, run_pipeline`; `ctx = run_pipeline(cfg)` (mode="api" default → no disk writes). `PipelineConfig(...).run()` returns None — never use it. Docs' `ctx.pydantic_model` does NOT exist; use `ctx.extracted_models` (list, 1 element in many-to-one).
- `PipelineContext`: `extracted_models`, `knowledge_graph: nx.DiGraph`, `graph_metadata`, `provenance` (ProvenanceLedger: `.chunks: dict[int, ChunkRecord{chunk_id, page_numbers, doc_item_refs (self_refs), item_geometry, headings, text}]`, `.nodes`), `node_registry` (`NodeIDRegistry.get_node_id(model_instance, auto_register=True)` → `"<Class>_<blake2b16>"`), `docling_document`.
- Node attrs: `{"id", "label": ClassName, "type": "entity", "__class__": ClassName, <template fields>, "__provenance__": {document_id, match: verbatim|observed|reconciled|derived, chunks: [int], pages: [int], refs: ["#/texts/42", ...]}}`; entity-typed fields are set to None (edges instead); component fields embedded as dicts; a node that absorbed alias merges carries `merged_aliases`. Edge attrs: `{"label": ...}` ONLY. `nx.DiGraph` → one edge per (source,target). `graph.graph` keys: format="docling-graph/v2", template_name, template_schema_hash, id_fields_map, alias_reconciliation?, closed_catalog_drops?, demoted_nodes?, empty_identity_nodes?.
- Template semantics: `model_config = ConfigDict(graph_id_fields=[...], extra="ignore", populate_by_name=True)` = entity; `ConfigDict(is_entity=False)` = component (no node, attrs embedded, entities below it link from nearest entity ancestor); edge = nested-model field with `Field(json_schema_extra={"edge_label": "X"})` (fallback label = field name); other keys: `graph_reference`, `reference_closed_catalog`, `graph_max_instances`. Converter dispatches on runtime instance type. Duplicate instances with same id → first non-empty attribute wins (`_enrich_existing_node`). Docs say keep identity to 1 (max 2) scalar fields, never lists/enums.
- Dense contract catalog (`contracts/dense/catalog.py`): Union-typed nested model → FIRST model only; a nested class already in the ancestry → whole branch skipped. ⇒ this is why links are scalar components (`target_type: str`, `target: str`), never nested entities.
- Input: `source` must be a PATH STRING; a `.json` whose `schema_name == "DoclingDocument"` is detected as `InputType.DOCLING_DOCUMENT` and conversion is skipped. Dense root-identity fallback = file stem → name the file `<source-stem>.json`. Its own converter is hard-coded (en/fr OCR) — never use it.
- Remote LLM: `inference="remote", provider_override="openai", model_override="openai/<proxy-alias>"` (bare alias → litellm BadRequestError "LLM Provider NOT provided"); `llm_overrides={"connection": {"base_url": "http://localhost:4000/v1", "api_key": "..."}, "generation": {"temperature": 0.0, "max_tokens": M}, "reliability": {"timeout_s": 300, "max_retries": 2}, "context_limit": C, "max_output_tokens": M}` — sub-models are `extra="forbid"`; `PipelineConfig` itself silently IGNORES unknown keys. `ConfigurationError("max_tokens exceeds model limit")` if generation.max_tokens > max_output_tokens (fallback 4092 / context 32000 for unknown models). Env alternative `CUSTOM_LLM_BASE_URL`/`CUSTOM_LLM_API_KEY` (openai provider only); `load_dotenv()` runs at import.
- Contract: `extraction_contract="auto"` → direct iff markdown_chars ≤ max_output_tokens×4 (one giant call + gleaning), else dense (skeleton batches over HybridChunker + fill, ThreadPoolExecutor(parallel_workers)). Dense requires `use_chunking=True`. `llm_input_format="auto"` → doclang(-geo) and the INTERNAL chunker tokenizer becomes tiktoken `cl100k_base` (downloads BPE from openaipublic.blob.core.windows.net unless `TIKTOKEN_CACHE_DIR` pre-populated); markdown mode uses `sentence-transformers/all-MiniLM-L6-v2` from HF. `dense_dedupe="standard"` = 1 extra LLM reconciliation call per path; `"off"` = exact canonical-id dedup only. Graph-level `reconcile_graph_aliases` (LLM-confirmed) always runs when a backend exists — not switchable.
- `structured_output=True` → `response_format={"type":"json_schema","json_schema":{name,schema,"strict":True}}` (no additionalProperties:false); backend falls back to prompt/json_object mode itself on ClientError or sparse output. LiteLLM 1.97 maps json_schema → Ollama `format` grammar for ollama_chat models.
- Failure: `PipelineError` with `details={"stage","error","error_type","reason"}` (`docling_graph.exceptions`); no partial context. `gc_collect=False` for services.
- Serialize: `from docling_graph.core.exporters import graph_to_dict` (`{"nodes":[{"id",...}], "edges":[{"source","target","label"}], "metadata", "graph"}`), `json.dumps(..., default=docling_graph.core.utils.string_formatter.json_serializable)`. No node-link exporter (use `nx.node_link_data` if needed).

## docling 2.123.0 / docling-core 2.92.0 (`pip install "docling[easyocr]" --extra-index-url https://download.pytorch.org/whl/cpu`; rapidocr is default, easyocr is an extra)
- Imports: `from docling.document_converter import DocumentConverter, PdfFormatOption`; `from docling.datamodel.base_models import InputFormat`; `from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, OcrMode, TableStructureOptions, TableFormerMode, HeadingHierarchyOptions`; `from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice`; `from docling_core.types.doc import DoclingDocument, TableItem, PictureItem, SectionHeaderItem, TextItem, DocItemLabel, ContentLayer, ImageRefMode, PictureMeta, DescriptionMetaField, CoordOrigin`.
- Options that matter: `PdfPipelineOptions(do_ocr=True, ocr_options=EasyOcrOptions(lang=["de","en"], mode=OcrMode.DEFAULT|FULL_PAGE, download_enabled=False), do_table_structure=True, table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=True), images_scale=3.0, generate_page_images=True, generate_picture_images=True, generate_parsed_pages=True, heading_hierarchy_options=HeadingHierarchyOptions(enabled=True), accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4), document_timeout=…, artifacts_path=…)`. EasyOcrOptions is `extra="forbid"`; ISO-639-1 codes (`de`, not `deu`). Default ocr_options is `OcrAutoOptions` (auto engine, lang ignored). `force_full_page_ocr` deprecated → `mode`. `HeadingHierarchyOptions.enabled` default False → all headings level 1.
- Converter: `DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})`; `converter.initialize_pipeline(InputFormat.PDF)` warms models; pipelines cached by options hash (never mutate options after). Default backend is threaded docling-parse (don't set DLPARSE_V4 — removed). `res = converter.convert(path)`; `res.status` (SUCCESS/PARTIAL_SUCCESS), `res.has_timeout_errors()`, `doc = res.document`.
- Document: `doc.pages[n].size.width/height`; `doc.tables` (`t.export_to_dataframe(doc)`, `t.caption_text(doc)`, `t.prov[0].page_no/bbox`, `t.self_ref`); `doc.pictures` (`p.get_image(doc)` → PIL, `p.caption_text(doc)`, `p.meta` (PictureMeta|None) → `p.meta.description = DescriptionMetaField(text=…, created_by=…)`; `p.annotations` deprecated); `doc.iterate_items(included_content_layers={...})` yields `(item, level)`, BODY only by default; `item.content_layer` (ContentLayer.BODY/FURNITURE); `SectionHeaderItem.level`; `ProvenanceItem(page_no, bbox(l,t,r,b, coord_origin=BOTTOMLEFT for PDF), charspan)` → `bbox.to_top_left_origin(page_height=doc.pages[page_no].size.height)`. Page header/footer are labelled PAGE_HEADER/PAGE_FOOTER AND placed in FURNITURE (excluded from markdown/iterate/chunks by default).
- Markdown: `doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER, image_placeholder="<!-- image -->", page_break_placeholder="<!-- page break -->", mark_meta=True, allowed_meta_names={"description"}, escape_underscores=False, escape_html=False, included_content_layers=None)`. Descriptions (meta) render after the item automatically. `doc.save_as_json(path)` / `DoclingDocument.load_from_json(path)`.
- Chunking: `from docling.chunking import HybridChunker`; `from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer`; `from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider`; `from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer`; `from transformers import AutoTokenizer`. `tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(ID), max_tokens=N)` (max_tokens REQUIRED offline; `count_tokens` excludes special tokens). `class Prov(ChunkingSerializerProvider): def get_serializer(self, doc): return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer(), params=MarkdownParams(...))`. `HybridChunker(tokenizer=tok, merge_peers=True, repeat_table_header=True, serializer_provider=Prov())` (field is singular `repeat_table_header`; passing `max_tokens=`/str tokenizer is deprecated). `for c in chunker.chunk(dl_doc=doc)`: `c.text`, `chunker.contextualize(chunk=c)` (headings + text), `c.meta.headings`, `c.meta.doc_items[*].self_ref/.label/.prov`; `c.meta.captions` deprecated. Default table serializer is TRIPLETS. Oversized single-table chunks are split row-wise with the header repeated but the CAPTION only on part 1. `merge_peers` merges adjacent chunks with equal headings.
- Offline models: `docling-tools models download layout tableformer easyocr --easyocr-lang de --easyocr-lang en -o /opt/docling-models`; `DOCLING_ARTIFACTS_PATH` / `PdfPipelineOptions.artifacts_path`; with artifacts_path set, EasyOCR `download_enabled` is forced False and weights read from `<artifacts>/EasyOcr`. Official Dockerfile: python:3.11-slim-bookworm, apt `libgl1 libglib2.0-0`, `OMP_NUM_THREADS=4`, `HF_HOME`/`TORCH_HOME`.
- Remote picture-description hook (NOT used; we do our own): `PictureDescriptionApiOptions(url, headers, params(JSON body incl. model), prompt, timeout, concurrency, picture_area_threshold=0.05!, scale)` + `enable_remote_services=True`; failures return empty text silently.

## Local infrastructure (this WSL2 box, verified)
- CPU only: AMD Ryzen AI 7 445 + Radeon 840M iGPU, no CUDA/nvidia-smi; 12 vCPU, 23 GiB RAM, 896 GB free. Python 3.12.3, uv 0.11.28; `venv-rag/` at repo root is empty (pip only) — use uv in `docling-graph/`. No pdftotext/tesseract/docker compose plugin; docker 29.1.3 native (use `docker run --network host`; from bridge, LiteLLM is at 172.17.0.1:4000, Ollama unreachable).
- LiteLLM proxy `http://localhost:4000/v1`, key `sk-123456789`, config `/home/javi-linux/clients/personal/litellm-server/config.yaml` (runs as a foreground venv process, no systemd — may need restart after reboot: `.../venv-litellm/bin/litellm --config .../config.yaml --port 4000`). Aliases: `gemini-dev` (gemini/gemini-3.7-flash, multimodal), `gemini-planer` (gemini-3.1-pro-preview), `lokaler-coder-qwen` (ollama_chat/qwen2.5-coder:32k, 14.8B, ~15 GB RAM, 124 s cold load), `lokaler-coder-qwen-14b`, `granite4` (3.4B, NO vision), `lokaler-coder-deepseek`, `lokaler-coder-codellama`, `bge-m3` (ollama/bge-m3, 1024-dim, batching works, warm 0.9 s/10 texts, cold ~10 s). `response_format={"type":"json_object"}` works. No `num_ctx` set for Ollama aliases (4096 default) → local extraction would truncate; not our repo.
- Ollama 0.31.2 on 127.0.0.1:11434 (`qwen3.5:2b` is the only vision model; not a proxy alias).

## ZSD ground truth (Betriebshandbuch_ZSD.pdf, BHB-PLT-0007 — extracted glyph-exact; canon.md has NO ZSD content)
- 29 pages; header on every page: `TESTDOKUMENT · DIESES DOKUMENT IST FIKTIV · KEINE ECHTEN BETRIEBSDATEN` (U+00B7); runner `Betriebshandbuch ZSD – Zentrale Sicherheitsdienste … BHB-PLT-0007 · Version 2.3`; footer `BHB-PLT-0007 · Version 2.3 · Stand 22.07.2026 · Bereich IT-S 1 / Zentrale Sicherheitsdienste … Seite N von 29`. Pages 2–3 = TOC (repeats every heading). Title page: title `ZSD – Zentrale Sicherheitsdienste`, version 2.3, Stand 22.07.2026, Klassifizierung `intern · Testdokument – fiktive Daten`, CI-PLT-0007, Verantwortlicher Kai Ostermann (1315), Vertretung Sabine Wollmer (1180). System kind = basisdienst; availability IAM 99,98 % · PKI 99,9 % · Vault 99,95 %; Wartungsfenster Freitag 18:00–22:00.
- 15 captions (`Tabelle N:` style only): 1 Änderungshistorie (p4); 2 Verwandte Dokumente und Berührungspunkt (p4–5, 7 rows: BHB-VRF-0207, BHB-VRF-0118, BHB-PLT-0042, BHB-PLT-0001 + NET-ZK-004, SBF-ITS-2025, NFH-ITS-1.4); 3 Betriebskennzahlen (p6–7); 4 Instanzen der Produktionsumgebung (p8–9, 11 rows, hosts iam-p01…p03, pg-iam-p01/p02, ad01/ad02, pki-p01/p02, ocsp, ca-root-off01, hsm-p01/p02, vault-0/1/2, jump01); 5 Portmatrix und Firewallregeln (p9–10, FW-ZSD-001…015); 6 Konsumentenmatrix (p11); 7 Standard Operating Procedures (p17, SOP-ZSD-01…06); 8 Auswirkungen bei Neustart oder Ausfall (p18–19); 9 Incidents der Zentralen Sicherheitsdienste (Auszug) (p23); 10 Metriken, Schwellwerte, Alarme (p24, 11 alerts incl. VaultSealed `vault_core_unsealed == 0`); 11 Sicherungsobjekte und Verfahren (p24–25); 12 Rollen, Namen, Erreichbarkeit (p26); 13 Eskalationsstufen (p26); 14 Halter der fünf Unseal-Schlüsselanteile (p27); 15 Weiterführende Dokumente (p29). Tables spanning pages with repeated header: 2, 3, 4, 5, 8, 11.
- Tabelle 6 columns: Konsument | IAM-Client | Zertifikate | Vault-Pfad | Ansprechpartner | Dokument. Rows: VPP Portal | vpp-portal | manuell, keytool und Keystore (SOP-VPP-05) | kv/vpp/prod/* | Torben Machwitz; Heiko Brandtner (4471) | BHB-VRF-0207 — Mars Dokumentendienste, dd-gateway | dd-gateway | cert-manager (Routen), manuell (Archiv-Adapter, SOP-DD-05) | kv/dd/* | Rainer Kolbe; Christina Haberland (3208) | BHB-VRF-0118 — Event-System 2.0, Grafana | es-grafana | cert-manager (ACME) | kv/event-system/grafana/oidc | Miriam Falk; Tobias Reinhardt (2117) | BHB-PLT-0042 — Event-System 2.0 ⇄ Kafka | kein Client (SASL/SCRAM-SHA-512) | manuell, Broker-Zertifikate durch Talwerk IT-Services | kv/event-system/kafka/scram | Jonas Brinkmann; Holger Pietsch (Talwerk) | BHB-PLT-0042 — CaaS OpenShift-Konsole | caas-console | cert-manager, ClusterIssuer bavd-issuing-ca-3 | kv/caas/* | Andreas Wehrle (2140); Sonja Wiechert (2143) | BHB-PLT-0001 — Checkmk und OpenSearch (VPP) | obs-vpp | manuell | kv/vpp/monitor | Torben Machwitz | BHB-VRF-0207. Cell values that WRAP across lines: bavd-issuing-ca-3, kv/event-system/…, Sonja Wiechert, Tobias Reinhardt, #zsd-betrieb.
- Tabelle 8: header Ereignis | VPP | Mars Dokumentendienste | Event-System 2.0 | CaaS-Plattform; events: Keycloak rollierend (row on p18, all four cells literally `keine`), Keycloak Vollausfall, PKI-Ausfall, Vault versiegelt (VPP cell: `unberührt, Keystores und Wallet liegen lokal auf den Knoten`), Active Directory Ausfall (p19). 20 ImpactStatements.
- Tabelle 9 (7 rows): ZSDSUP-0119 18.08.2025 ACME-Anträge erreichen die PKI nicht, 3 Tage, partner DDSUP-0794; ZSDSUP-0166 12.01.2026, 2 h 15 min, —; ZSDSUP-0208 11.05.2026 Client-Secret vpp-portal ohne Abstimmung rotiert, 42 min, VPPSUP-2251; ZSDSUP-0214 13.05.2026 Truststore-Verteilung nach CA-Wechsel verzögert, 42 min, ESSUP-1455 (+ ESSUP-1517 Folgestörung in prose p21); ZSDSUP-0231 09.06.2026 OCSP > 500 ms, 50 min, —; ZSDSUP-0247 02.07.2026 Vault nach Knotenwartung versiegelt, 38 min (48 min per CaaS/CAASUP-0351 — deliberate explained conflict, p22), partners CAASUP-0351, DDSUP-1201; ZSDSUP-0252 18.07.2026 iam-p03 nach Update nicht im Cluster, 18 min, —. ⇒ 6 PARTNER_TICKET edges from this doc. Sev from §7.2 headings: 0119 Sev-2; 0208/0214/0247 Sev-1. Only OpenItem: OP-ZSD-02 (p22). No Contract ids.
- Persons (15): Kai Ostermann (IAM, 1315), Sabine Wollmer (PKI, 1180), Marcel Ebert (Secret-Management, 1352), Dr. Annika Reuß (Bereichsleitung IT-S, 1300), Frank Dettmer (Netz, 1240), Andreas Wehrle (2140), Sonja Wiechert (2143), Torben Machwitz, Heiko Brandtner (4471), Rainer Kolbe, Christina Haberland (3208), Miriam Falk, Tobias Reinhardt (2117), Jonas Brinkmann, Holger Pietsch. NOT present: Petra Nowak, Dr. Martina Kellerhoff.
- Negative assertions (pages): p8/p19 IAM und PKI laufen auf eigenen VMs, nicht auf der CaaS-Plattform; p11 `cert-manager und ACME werden dort [VPP] bewusst nicht genutzt`; p12 `Bewusst nicht angebunden: VPP … ist nicht Mandant der CaaS-Plattform; das Event-System hat keine eigene LDAPS-Verbindung …`; p15 `Der Unseal braucht weder IAM noch PKI`; p19 VPP/Vault `unberührt…` + `VPP hängt an Schritt 3, nicht an Schritt 7`. Cycle p19: `Vault läuft auf der CaaS-Plattform, deren Zertifikate von der PKI stammen, deren Zugangsdaten und Passphrasen in Vault liegen.` 11-step Kaltstart chain is PROSE on p19. Glossary §10.1 (p28–29) is a definition list (18 headwords), not a table.
- Figures: Abbildung 1 (p8), 2 (p16), 3 (p18); reference PNGs `diagramme-png/zsd1-uebersicht.png` 4800×3420, `zsd2-…` 4800×3540, `zsd3-…` 4800×3720.
- Identifiers intact on one line: all tickets, SOP-*, FW-ZSD-*, BHB-*. Ontology host regex misses ca-root-off01, vault-0/1/2, ad01/ad02, jump01 and VIP FQDNs. `OP-…` regex needs `(?<!S)`.

## Research artefacts (persistent)
- Workflow journal with all raw findings: `/home/javi-linux/.claude/projects/-home-javi-linux-clients-personal-rag-prototype/e0377991-6649-425d-b21f-766135fc7e21/subagents/workflows/wf_ac755a50-821/journal.jsonl`
