# docling-graph-service — CPU image with all models baked in (runs fully offline against remote LLM endpoints).
#
# Local build:      docker build -t dgs .
# Enterprise build: docker build -t dgs --build-arg BASE_IMAGE=registry.example.internal/python:3.12-slim \
#                     --build-arg PIP_INDEX_URL=https://artifactory.example.internal/api/pypi/pypi/simple \
#                     --build-arg PIP_EXTRA_INDEX_URL=https://artifactory.example.internal/api/pypi/pytorch-cpu/simple \
#                     --build-arg HF_ENDPOINT=https://artifactory.example.internal/api/huggingfaceml/hf .
# BASE_IMAGE: python:3.12 (default; ships libGL/glib/curl, so no apt access is needed) or python:3.12-slim (smaller,
# but the apt step below must reach deb.debian.org or a mirror).
ARG BASE_IMAGE=python:3.12
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_TRUSTED_HOST=
ARG HF_ENDPOINT=https://huggingface.co
ARG CHUNK_TOKENIZER=intfloat/multilingual-e5-large

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=${HF_ENDPOINT} \
    HF_HOME=/opt/hf \
    TORCH_HOME=/opt/torch \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models \
    OMP_NUM_THREADS=4

# OpenCV (docling dependency) needs libGL + glib; curl is for the HEALTHCHECK. Skipped when the base image
# already provides them (python:3.12); on -slim this needs deb.debian.org (proxy: lowercase http_proxy vars).
RUN if [ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] && [ -e /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 ] && command -v curl >/dev/null; then \
        echo "base image provides libGL, glib and curl - skipping apt"; \
    else \
        apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
# 1. third-party dependencies (layer survives source edits)
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# 2. models needed offline: docling layout/tableformer/EasyOCR (de,en), the chunk tokenizer,
#    docling-graph's internal chunker tokenizers (all-MiniLM-L6-v2 + tiktoken cl100k_base)
RUN docling-tools models download layout tableformer easyocr --easyocr-lang de --easyocr-lang en \
        -o ${DOCLING_ARTIFACTS_PATH} \
    && python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${CHUNK_TOKENIZER}'); AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX /opt

# 3. the service itself
COPY README.md config.yaml ./
COPY src ./src
RUN pip install --no-deps .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 10001 dgs && mkdir -p /var/cache/dgs && chown -R dgs /var/cache/dgs /app
USER dgs
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=20s --start-period=120s CMD curl -fsS http://localhost:8080/healthz || exit 1
CMD ["uvicorn", "docling_graph_service.api:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "75"]
