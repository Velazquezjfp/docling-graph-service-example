PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
ZSD ?= ../user-manual-books/handbuch_daten/handbuch/Betriebshandbuch_ZSD.pdf
IMAGE ?= dgs:local

.PHONY: venv dev test test-integration serve process-zsd docker-build docker-run docker-smoke compose-up compose-down lint

venv:
	python3 -m venv .venv && $(PIP) install --upgrade pip

dev: venv
	$(PIP) install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu

test:
	$(PY) -m pytest tests/unit -q -W ignore

test-integration:            ## needs LiteLLM (gemini-dev, bge-m3) and the ZSD pdf; ~10 min
	DGS_INTEGRATION=1 $(PY) -m pytest tests/integration -q -W ignore -s

serve:
	$(PY) -m docling_graph_service.cli serve

process-zsd:
	$(PY) -m docling_graph_service.cli process $(ZSD) --out out/zsd

docker-build:
	docker build -t $(IMAGE) .

ONTOLOGY_DIR ?= $(abspath ../user-manual-books/handbuch_daten/Ontologie)

docker-run:                  ## host networking: the container reaches LiteLLM on localhost:4000
	docker run --rm --network host --env-file .env --name dgs \
	  -e DGS__SERVICE__CACHE_DIR=/var/cache/dgs -v dgs-cache:/var/cache/dgs \
	  -e DGS__GRAPH__DEFAULT_ONTOLOGY_PATH=/ontology/ontology.yaml -v $(ONTOLOGY_DIR):/ontology:ro \
	  $(IMAGE)

compose-up:                  ## dev box: docker compose --profile local (host networking)
	docker compose --profile local up -d

compose-down:
	docker compose --profile local down

docker-smoke:
	curl -fsS localhost:8080/healthz && curl -fsS localhost:8080/v1/capabilities | head -c 600

lint:
	.venv/bin/ruff check src tests
