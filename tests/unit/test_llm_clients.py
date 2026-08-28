import json

import httpx
import pytest
from docling_core.types.doc import DoclingDocument, ImageRef
from PIL import Image

from docling_graph_service.describe import describe_pictures
from docling_graph_service.embed import EmbeddingError, embed_texts
from docling_graph_service.llm_http import LLMClient, LLMHTTPError
from docling_graph_service.settings import VLMSettings


def _client(handler, attempts=3):
    return LLMClient("http://llm.test/v1", "key", 5.0, max_attempts=attempts, backoff_s=0.0,
                     transport=httpx.MockTransport(handler))


def test_embed_batches_order_and_dim():
    calls = []

    def handler(request: httpx.Request):
        assert request.headers["authorization"] == "Bearer key"
        body = json.loads(request.content)
        calls.append(body["input"])
        data = [{"index": i, "embedding": [float(len(t))] * 4} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})  # out of order on purpose

    vectors = embed_texts(_client(handler), "m", ["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2, dim=4, prefix="p: ")
    assert calls == [["p: a", "p: bb"], ["p: ccc", "p: dddd"], ["p: eeeee"]]
    assert [v[0] for v in vectors] == [4.0, 5.0, 6.0, 7.0, 8.0]  # len('p: a') == 4


def test_embed_dim_mismatch():
    handler = lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
    with pytest.raises(EmbeddingError, match="dim 2"):
        embed_texts(_client(handler), "m", ["a"], batch_size=8, dim=1024)


def test_retry_then_fail():
    n = {"calls": 0}

    def handler(request):
        n["calls"] += 1
        return httpx.Response(503, text="busy")

    with pytest.raises(LLMHTTPError, match="503"):
        _client(handler, attempts=3).post_json("/embeddings", {})
    assert n["calls"] == 3


def test_retry_then_succeed():
    n = {"calls": 0}

    def handler(request):
        n["calls"] += 1
        return httpx.Response(429 if n["calls"] == 1 else 200, json={"ok": True})

    assert _client(handler).post_json("/x", {}) == {"ok": True}


def _doc_with_pictures(n=2):
    doc = DoclingDocument(name="t")
    for _ in range(n):
        img = Image.new("RGB", (64, 32), "white")
        doc.add_picture(image=ImageRef.from_pil(img, dpi=72))
    return doc


def test_describe_sets_meta_description():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Ein Diagramm. "}}]})

    doc = _doc_with_pictures(2)
    report = describe_pictures(doc, _client(handler), VLMSettings(model="vlm", concurrency=2),
                               captions={doc.pictures[0].self_ref: "Abbildung 1: Test"})
    assert report.described == 2 and report.errors == []
    assert all(p.meta.description.text == "Ein Diagramm." and p.meta.description.created_by == "vlm" for p in doc.pictures)
    content = seen[0]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert any("Abbildung 1: Test" in json.dumps(b, ensure_ascii=False) for b in seen)


def test_describe_degrades_per_picture():
    handler = lambda r: httpx.Response(500, text="boom")
    doc = _doc_with_pictures(1)
    report = describe_pictures(doc, _client(handler, attempts=1), VLMSettings(model="vlm"))
    assert report.described == 0 and len(report.errors) == 1 and "500" in report.errors[0]
    assert doc.pictures[0].meta is None or doc.pictures[0].meta.description is None


def test_describe_reports_truncation_as_error():
    handler = lambda r: httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                              "message": {"content": "abgeschnitten"}}]})
    doc = _doc_with_pictures(1)
    report = describe_pictures(doc, _client(handler), VLMSettings(model="vlm", max_tokens=400))
    assert report.described == 0 and "truncated at max_tokens=400" in report.errors[0]


def test_describe_accepts_content_parts():
    handler = lambda r: httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                              "message": {"content": [{"type": "text", "text": "Teil 1. "}, {"type": "text", "text": "Teil 2."}]}}]})
    doc = _doc_with_pictures(1)
    report = describe_pictures(doc, _client(handler), VLMSettings(model="vlm"))
    assert report.described == 1 and doc.pictures[0].meta.description.text == "Teil 1. Teil 2."
