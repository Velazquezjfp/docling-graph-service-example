# Deploying docling-graph-service on a remote server and reproducing the ZSD results

Goal: run the same service on your server, point it at your endpoints (vLLM / TEI / whatever is OpenAI-compatible
there) and produce the same kind of output with the same test scripts. Verified locally on 2026-08-27 with the
`dgs:local` image; the remote steps are the same commands with different URLs.

## 0. What the service needs from the outside

| Need | Local (this machine) | Remote |
|---|---|---|
| Extraction LLM, `/v1/chat/completions` | LiteLLM `gemini-dev` | vLLM `qwen3-coder…` (OpenAI-compatible) |
| VLM (optional), `/v1/chat/completions` with `image_url` | LiteLLM `gemini-dev` | granite-vision on vLLM — **can be off**, see §6 |
| Embeddings, `/v1/embeddings` (1024-dim) | LiteLLM `bge-m3` | TEI `multilingual-e5-large` (`text_prefix: "passage: "`) |
| Docker (or Python 3.12 + pip) | native dockerd | your host / registry |
| Models baked into the image | downloaded at build | **either** build with Artifactory mirrors **or** ship the image (§2) |

Nothing else: no database, no GPU, no internet at runtime.

## 1. Files to put on the server

Keep the same relative layout — the tests resolve the PDF and the ontology relative to the repo:

```
<base>/
├── docling-graph/                      # this directory, without .venv/ .cache/ out/ .env
│   ├── Dockerfile  compose.yaml  Makefile  config.yaml  .env.example  pyproject.toml  requirements.txt  README.md
│   ├── src/docling_graph_service/      # the service
│   ├── scripts/process_via_api.py      # client: payload -> service -> files + summary
│   └── tests/                          # unit + integration (integration can run over HTTP, §5)
└── user-manual-books/handbuch_daten/
    ├── Ontologie/ontology.yaml         # the ontology (mounted read-only into the container)
    └── handbuch/Betriebshandbuch_ZSD.pdf   # the test document
```

`rsync -a --exclude .venv --exclude .cache --exclude out --exclude .env docling-graph/ server:<base>/docling-graph/`
plus the two data files. Optional but useful: `out/zsd-integration/response.json` from here as the reference to
compare against.

## 2. Getting the image onto the server

Everything is driven by **one `.env` file** in `docling-graph/` (copy `.env.example`): docker compose reads it for
the build arguments *and* hands it to the container as runtime settings. No multi-line `docker build … \` commands
(a missing trailing backslash there is what produces `--build-arg: command not found`).

**Option A — ship the image built here (no internet/mirrors needed on the server).** Self-contained (docling models,
EasyOCR de/en, tokenizers, tiktoken; ~4.9 GB):

```bash
docker save dgs:local | gzip > dgs-local.tar.gz      # here (~2 GB compressed) → scp → on the server:
gunzip -c dgs-local.tar.gz | docker load
```
Then set `IMAGE=dgs:local` in the server's `.env` and skip the build.

**Option B — build on the server (Artifactory + proxy).** Fill the build section of `.env`:

```bash
BASE_IMAGE=<registry>/python:3.12-slim
PIP_INDEX_URL=https://<artifactory>/api/pypi/<repo>/simple
PIP_EXTRA_INDEX_URL=                     # empty: no PyTorch-CPU index -> torch comes from PIP_INDEX_URL (CUDA wheel, runs on CPU, image ~7-8 GB)
# PIP_TRUSTED_HOST=<artifactory-host>    # only if pip reports CERTIFICATE_VERIFY_FAILED
# HF_ENDPOINT=https://<artifactory>/api/huggingfaceml/<repo>   # or leave unset and use the proxy:
HTTP_PROXY=http://<proxy-host>:<port>
HTTPS_PROXY=http://<proxy-host>:<port>
NO_PROXY=localhost,127.0.0.1,<artifactory-host>,<registry-host>,.<internal-domain>
IMAGE=dgs:0.1.0
```
then

```bash
docker compose --profile enterprise build      # 10-20 min; downloads: pip packages, docling models (HF), EasyOCR weights (GitHub), tokenizers
```
The proxy variables apply to every build step and are not baked into the image. If the build stops, the failing
step tells you which download the proxy/mirror did not allow: pip (`pip install -r`), HF models
(`docling-tools models download` / `AutoTokenizer`), or EasyOCR weights from `github.com/JaidedAI` — EasyOCR has no
mirror override, so if GitHub is blocked use option A.

**Option C — no Docker.** `python3 -m venv .venv && .venv/bin/pip install -e .` (same index args), pre-download
models: `docling-tools models download layout tableformer easyocr --easyocr-lang de --easyocr-lang en -o
/opt/docling-models`, `export DOCLING_ARTIFACTS_PATH=/opt/docling-models`, and once online
`python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large');
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"` + `python -c "import tiktoken;
tiktoken.get_encoding('cl100k_base')"`. Then `HF_HUB_OFFLINE=1 .venv/bin/dgs serve`. (`requirements.txt` pins the
PyTorch-CPU builds `+cpu`; against a plain PyPI index install from `pyproject.toml` as above instead.)

## 3. Where to put your endpoints — `.env` (the only file you edit)

`config.yaml` (baked into the image) holds working defaults for *every* setting; you only override what differs for
your site with `DGS__<SECTION>__<KEY>` lines in `.env` — typically the three endpoints below. `pyproject.toml`
contains nothing to configure. The runtime section of `.env`:

```bash
# LLM (drives docling-graph). base_url must be the OpenAI-compatible root that has /chat/completions
DGS__LLM__BASE_URL=http://vllm.internal:8000/v1
DGS__LLM__API_KEY=none                     # vLLM without auth still needs a non-empty bearer for docling-graph
DGS__LLM__MODEL=qwen3-coder-30b-a3b-instruct   # exactly the id `GET /v1/models` on that endpoint returns
DGS__LLM__CONTEXT_LIMIT=131072             # what your vLLM was started with (--max-model-len)
DGS__LLM__MAX_OUTPUT_TOKENS=8192
DGS__LLM__STRUCTURED_OUTPUT=true           # set false if the endpoint rejects response_format=json_schema

# VLM — set enabled=false until granite-vision is up (§6)
DGS__VLM__ENABLED=false
DGS__VLM__BASE_URL=http://vllm-vision.internal:8000/v1
DGS__VLM__API_KEY=none
DGS__VLM__MODEL=granite-vision-3.3-2b

# Embeddings (TEI is OpenAI-compatible on /v1/embeddings)
DGS__EMBEDDING__BASE_URL=http://tei.internal:8080/v1
DGS__EMBEDDING__API_KEY=none
DGS__EMBEDDING__MODEL=intfloat/multilingual-e5-large
DGS__EMBEDDING__DIM=1024
DGS__EMBEDDING__TEXT_PREFIX="passage: "

# paths (compose sets the in-container cache dir and ontology path itself; this only selects the host directory)
ONTOLOGY_DIR=/srv/rag/user-manual-books/handbuch_daten/Ontologie
SELINUX_LABEL=,Z                            # RHEL/CentOS with SELinux enforcing; drop otherwise
PORT=8080
```

Quick check that the endpoints answer the way the service expects (run on the server):

```bash
curl -s $LLM/v1/models | head -c 300                                    # model id must match DGS__LLM__MODEL
curl -s $LLM/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"Antworte mit OK"}],"max_tokens":5}'
curl -s $TEI/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"intfloat/multilingual-e5-large","input":["passage: test"]}' | python3 -c "import json,sys; print(len(json.load(sys.stdin)['data'][0]['embedding']))"   # must print 1024
```

## 4. Run the service

```bash
cd <base>/docling-graph
docker compose --profile enterprise up -d      # publishes :8080, mounts the ontology read-only, named volume for the cache
docker compose --profile enterprise logs -f    # docling-graph prints "Phase 1 (skeleton) … Phase 2 (fill) …" progress
docker compose --profile enterprise down       # stop (the cache volume survives)
```

Equivalent plain `docker run` (if you prefer no compose):

```bash
docker run -d --name dgs --restart unless-stopped -p 8080:8080 --env-file .env -e DGS__SERVICE__CACHE_DIR=/var/cache/dgs -e DGS__GRAPH__DEFAULT_ONTOLOGY_PATH=/ontology/ontology.yaml -v dgs-cache:/var/cache/dgs -v <base>/user-manual-books/handbuch_daten/Ontologie:/ontology:ro,Z dgs:0.1.0
```

(The `local` profile uses `network_mode: host` because on the dev box LiteLLM/Ollama live on localhost; with real
remote URLs the `enterprise` profile publishes the port instead.) If you mount a *host directory* for the cache
instead of the named volume, make it writable for uid 10001 (`chmod a+rwX`); otherwise caching is skipped with a
logged warning, never an error.

Verify (the first call waits for the model warm-up, ~30–60 s):

```bash
curl -s localhost:8080/healthz          # {"status":"ok","checks":{"chunk_tokenizer":"ok","docling_models":"ok","docling_graph_tokenizer":"ok","default_ontology":"ok"}}
curl -s localhost:8080/v1/capabilities | python3 -m json.tool | grep -A3 endpoint_status   # llm/vlm/embedding: "ok"
curl -s localhost:8080/v1/ontology-schema | head -c 200
docker logs -f dgs                      # docling-graph prints "Phase 1 (skeleton) … Phase 2 (fill) …" progress
```

`endpoint_status` probes `GET <base_url>/models` on each endpoint; "unreachable: …" tells you which URL/key is wrong
before you spend 15 minutes on a document.

## 5. Produce the ZSD results

**Client script (payload → service → files + summary)** — needs only `httpx` and `pyyaml` on the client side:

```bash
python3 -m venv .venv && .venv/bin/pip install httpx pyyaml            # or reuse the service venv
.venv/bin/python scripts/process_via_api.py ../user-manual-books/handbuch_daten/handbuch/Betriebshandbuch_ZSD.pdf \
    --url http://localhost:8080 --out out/remote-zsd \
    --ontology ../user-manual-books/handbuch_daten/Ontologie/ontology.yaml   # optional: default is the mounted file
# fast first check without LLM calls:
.venv/bin/python scripts/process_via_api.py …ZSD.pdf --no-graph --no-vlm --no-embed
```

Output: `out/remote-zsd/{response.json, graph.json, chunks.json, markdown.md, summary.json}` and a `SUMMARY {…}` block
(pages/tables/pictures, chunks by kind, embedded count, degraded flags, timings, nodes/edges by type and polarity,
unresolved targets) directly comparable with README "ZSD results" and with this machine's
`out/zsd-integration/response.json`. Expect ~5 min for the first conversion (CPU), then LLM time (≈11–14 min with
gemini-flash here; qwen3-coder on vLLM will differ). The second identical call returns from cache in < 1 s.

**Same test suite over HTTP** — the integration tests accept a URL and then assert the ZSD ground truth against the
deployed service:

```bash
.venv/bin/pip install -e ".[dev]"                                       # once (pulls docling; fine on the server)
DGS_INTEGRATION=1 DGS_INTEGRATION_URL=http://localhost:8080 .venv/bin/pytest tests/integration -q -s -W ignore
```

12 tests: conversion/chunking ground truth (deterministic — must pass identically), embeddings, VLM (skip-worthy
if VLM is off, see §6), graph thresholds (LLM-dependent — with a different model expect *similar*, not identical,
counts; the thresholds are deliberately loose). Unit tests need no endpoints: `.venv/bin/pytest tests/unit -q`.

## 6. Running without the VLM (vLLM vision not ready yet)

Nothing breaks. Two ways:

* **Off**: `DGS__VLM__ENABLED=false` (or `--no-vlm` / `"vlm_enabled": false` per request). Pictures stay
  `<!-- image -->` placeholders in the Markdown, no picture chunks are produced, `degraded.vlm=false` (nothing was
  attempted), everything else — chunks, embeddings, graph — is produced exactly as with VLM.
* **Configured but unreachable**: the VLM stage catches the failure per picture, reports it in `errors[]`, sets
  `degraded.vlm=true`, and the request still returns Markdown, chunks, embeddings and graph. (Degraded results are
  not written to the result cache, so the next call retries the VLM.)

The graph does not depend on the descriptions (docling-graph works on the DoclingDocument text and tables), so
node/edge counts are the same with or without VLM. In the ZSD run the three figures only confirm relations that the
tables already state. `test_vlm_descriptions` in the integration suite will fail while VLM is off — that is
expected; deselect it with `-k "not vlm"`.

Reusing this machine's descriptions on the server is not supported by the API today (there is no field to inject
pre-computed descriptions). It would be a small addition — a `picture_descriptions: {"#/pictures/0": "…"}`
override in `pipeline_config` — if you want it before the vision endpoint exists; the texts are in
`out/zsd-vlm/document.json` (`pictures[*].meta.description.text`).

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `endpoint_status.llm: unreachable` | wrong `base_url` (needs the `/v1` root) or auth; try the curl from §3 |
| `graph: docling-graph … LLM Provider NOT provided` | model id must be reachable as `openai/<model>` — check `DGS__LLM__MODEL` equals the id from `/v1/models` |
| `graph: … response_format` / 400 from vLLM | `DGS__LLM__STRUCTURED_OUTPUT=false` (docling-graph falls back to prompt-mode JSON) |
| `graph: … max_tokens exceeds model limit` | lower `DGS__LLM__MAX_OUTPUT_TOKENS` or raise the vLLM `--max-model-len`; keep `CONTEXT_LIMIT` ≤ the served context |
| graph takes very long | dense contract = ~15 skeleton batches + ~60 fill jobs per document; `DGS__LLM__PARALLEL_WORKERS=4` if the endpoint has capacity; `DGS__SERVICE__REQUEST_DEADLINE_S` bounds a request |
| `embeddings: embedding dim 768 != configured 1024` | the served embedding model is not e5-large/bge-m3 — set `DGS__EMBEDDING__DIM` |
| `vlm: … truncated at max_tokens` | reasoning model — raise `DGS__VLM__MAX_TOKENS` |
| `healthz` → 503 `docling_graph_tokenizer: error` | image built without HF access; use option A or fix `HF_ENDPOINT` |
| `PermissionError` warnings for `/var/cache/dgs` | host-mounted cache dir not writable for uid 10001 |
| 503 on `/v1/process` | one job at a time by design; retry after `Retry-After` or queue upstream |

## 8. RHEL / CentOS hosts (Docker installed)

The image is Debian-based internally and runs unchanged on a RHEL/CentOS Docker host. Host-side differences only:

* **SELinux (enforcing by default).** Bind-mounted *host directories* need the `:Z` (private) or `:z` (shared)
  relabel suffix, otherwise the container gets `Permission denied` on `/ontology` or on a host-mounted cache dir:
  `-v <base>/user-manual-books/handbuch_daten/Ontologie:/ontology:ro,Z`. Named volumes (`dgs-cache`) need nothing.
* **Firewall.** `firewalld` blocks 8080 from other hosts: `firewall-cmd --permanent --add-port=8080/tcp &&
  firewall-cmd --reload` (not needed when the caller runs on the same host).
* **Restart at boot.** `--restart unless-stopped` works with Docker CE as usual.
* If `docker` on the host is actually the podman shim (`podman-docker`), the same commands work (`podman load`,
  `podman run … :Z`); restart policies then need a systemd unit (`podman generate systemd --new --name dgs`).
* **Without a container:** `dnf install python3.12 python3.12-pip mesa-libGL glib2`, then option C of §2 with
  `python3.12 -m venv .venv`; all dependencies are manylinux wheels (no compiler needed).
