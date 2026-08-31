#!/usr/bin/env python3
"""Send one document to a running docling-graph-service and store the result.

    python scripts/process_via_api.py path/to/file.pdf --url http://server:8080 --out out/remote-zsd
    python scripts/process_via_api.py file.pdf --no-vlm                 # VLM endpoint not available yet
    python scripts/process_via_api.py file.pdf --ontology ../user-manual-books/handbuch_daten/Ontologie/ontology.yaml

Writes response.json (everything), graph.json, chunks.json, markdown.md and prints the same summary block the
integration test prints. Only needs `httpx` and `pyyaml` (both in the service venv) — no docling on the client.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("document", type=Path)
    ap.add_argument("--url", default="http://localhost:8080", help="service base URL")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: out/<document stem>)")
    ap.add_argument("--ontology", type=Path, default=None, help="ontology YAML to send as ontology_graph")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--contract", choices=["dense", "direct"], default=None)
    ap.add_argument("--timeout", type=float, default=3600.0, help="client read timeout in seconds")
    args = ap.parse_args()

    options: dict = {}
    if args.no_vlm:
        options["vlm_enabled"] = False
    if args.no_graph:
        options["graph_enabled"] = False
    if args.no_embed:
        options["embedding_enabled"] = False
    if args.no_ocr:
        options["ocr_enabled"] = False
    if args.contract:
        options["extraction_contract"] = args.contract
    body = {
        "document": {"name": args.document.name, "format": args.document.suffix.lstrip(".") or "pdf",
                     "base64_content": base64.b64encode(args.document.read_bytes()).decode()},
        "pipeline_config": options,
    }
    if args.ontology:
        import yaml

        body["ontology_graph"] = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))

    out = args.out or Path("out") / args.document.stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # trust_env=False: ignore http(s)_proxy shell variables — a corporate proxy must never sit between
    # this script and the service URL (proxies drop long-running requests).
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(args.timeout, connect=10), trust_env=False) as client:
        health = client.get("/healthz")
        print(f"healthz: {health.status_code} {health.text[:200]}")
        r = client.post("/v1/process", json=body)
    elapsed = round(time.time() - t0)
    if r.status_code != 200:
        print(f"FAILED: HTTP {r.status_code} after {elapsed}s\n{r.text[:2000]}", file=sys.stderr)
        return 1
    resp = r.json()
    (out / "response.json").write_text(json.dumps(resp, ensure_ascii=False), encoding="utf-8")
    (out / "markdown.md").write_text(resp["markdown"], encoding="utf-8")
    (out / "chunks.json").write_text(json.dumps(resp["chunks"], ensure_ascii=False, indent=1), encoding="utf-8")
    if resp.get("graph"):
        (out / "graph.json").write_text(json.dumps(resp["graph"], ensure_ascii=False, indent=1), encoding="utf-8")

    chunks = resp["chunks"]
    g = resp.get("graph")
    summary = {
        "http_seconds": elapsed, "cached": resp.get("cached"),
        "pages": resp["document"]["pages"], "tables": resp["document"]["tables"], "pictures": resp["document"]["pictures"],
        "chunks": len(chunks), "chunks_by_kind": dict(Counter(c["kind"] for c in chunks)),
        "embedded": sum(c["embedding"] is not None for c in chunks),
        "max_tokens": max((c["token_count"] for c in chunks), default=0),
        "degraded": resp["degraded"], "errors": resp["errors"], "timings_s": resp["timings_s"],
        "versions": resp["versions"],
        "nodes": g["meta"]["node_count"] if g else None, "edges": g["meta"]["edge_count"] if g else None,
        "nodes_by_type": g["meta"]["nodes_by_type"] if g else None,
        "edges_by_type": g["meta"]["edges_by_type"] if g else None,
        "edges_by_polarity": g["meta"]["edges_by_polarity"] if g else None,
        "unresolved_targets": len(g["meta"]["unresolved_targets"]) if g else None,
        "conflicts": g["meta"]["conflicts"] if g else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
