import base64
from types import SimpleNamespace

import anyio
import pytest
from fastapi.testclient import TestClient

from docling_graph_service import api
from docling_graph_service.pipeline import Outcome
from docling_graph_service.schemas import Degraded, DocumentInfo, ProcessResponse
from docling_graph_service.settings import Settings


class _FakeRuntime:
    def __init__(self):
        self.settings = Settings(_env_file=None, llm={"api_key": "k"}, service={"max_upload_mb": 1})

    def client(self, *a, **k):
        return SimpleNamespace(probe=lambda: "ok")


@pytest.fixture
def client():
    api.app.state.runtime = _FakeRuntime()
    api.app.state.limiter = anyio.CapacityLimiter(1)
    return TestClient(api.app)


def _body(content=b"%PDF-1.4 fake", **extra):
    return {"document": {"name": "x.pdf", "format": "pdf", "base64_content": base64.b64encode(content).decode()}, **extra}


def test_ontology_schema_and_capabilities(client):
    assert client.get("/v1/ontology-schema").json()["required"] == ["meta", "classes"]
    caps = client.get("/v1/capabilities").json()
    assert caps["models"]["llm"]["api_key"] == "***" and caps["formats"][0] == "pdf"
    assert caps["chunking"]["tokenizer"] == "intfloat/multilingual-e5-large"


def test_413(client):
    r = client.post("/v1/process", json=_body(b"x" * (2 * 1024 * 1024)))
    assert r.status_code == 413


def test_422_bad_base64(client):
    r = client.post("/v1/process", json={"document": {"name": "x.pdf", "format": "pdf", "base64_content": "@@@"}})
    assert r.status_code == 422 and "base64" in r.text


def test_422_unsupported_format(client):
    r = client.post("/v1/process", json={"document": {"name": "x.xyz", "format": "xyz",
                                                      "base64_content": base64.b64encode(b"x").decode()}})
    assert r.status_code == 422 and "unsupported" in r.text


def test_422_unknown_pipeline_key(client):
    r = client.post("/v1/process", json=_body(pipeline_config={"ocr_language": "de"}))
    assert r.status_code == 422 and "ocr_language" in r.text


def test_422_invalid_ontology(client):
    r = client.post("/v1/process", json=_body(ontology_graph={"meta": {"name": "x", "version": "1"}, "classes": {}}))
    assert r.status_code == 422 and "classes" in r.text


def test_503_when_busy(client):
    class Busy:
        def acquire_nowait(self):
            raise anyio.WouldBlock

        def release(self):
            pass

    api.app.state.limiter = Busy()
    r = client.post("/v1/process", json=_body())
    assert r.status_code == 503 and r.headers["retry-after"] == "30"


def test_200_shape(client, monkeypatch):
    def fake_process(rt, content, name, fmt, options, ontology_graph, **kw):
        assert fmt == "pdf" and options.vlm_enabled is False
        return Outcome(ProcessResponse(
            document=DocumentInfo(name=name, format=fmt, sha256="0" * 64, pages=1, tables=0, pictures=0),
            markdown="# x", chunks=[], graph=None, degraded=Degraded(graph=True), errors=["graph: off"]))

    monkeypatch.setattr(api, "process", fake_process)
    r = client.post("/v1/process", json=_body(pipeline_config={"vlm_enabled": False}))
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] == {"vlm": False, "embeddings": False, "graph": True}
    assert body["markdown"] == "# x" and body["chunks"] == [] and body["cached"] is False
