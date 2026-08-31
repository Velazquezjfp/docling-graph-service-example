#!/usr/bin/env python3
"""Result viewer for docling-graph-service outputs. Python standard library only.

    python scripts/serve_viewer.py --dir out            # then open http://127.0.0.1:8081
    python scripts/serve_viewer.py --dir /srv/results --port 8081

Serves scripts/viewer.html plus a read-only JSON API over the output folders written by
``dgs process --out X`` or ``scripts/process_via_api.py --out X``:

    GET /api/runs          -> [{name, mtime, files}]         (subfolders containing graph/chunks/response JSON)
    GET /api/run/<name>    -> {name, summary, graph, chunks} (chunk embeddings stripped: ~90% of the bytes)

Binds 127.0.0.1 by default. On a remote server either keep that and tunnel the port
(``ssh -L 8081:localhost:8081 user@server``) or pass ``--host 0.0.0.0`` and open the firewall port.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUN_FILES = ("graph.json", "chunks.json", "response.json")
_NAME = re.compile(r"^[A-Za-z0-9._ -]{1,128}$")
HTML_PATH = Path(__file__).resolve().parent / "viewer.html"


def list_runs(root: Path) -> list[dict]:
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = [f for f in RUN_FILES if (d / f).is_file()]
        if files:
            runs.append({"name": d.name, "mtime": max((d / f).stat().st_mtime for f in files), "files": files})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_run(root: Path, name: str) -> dict:
    d = (root / name).resolve()
    if not _NAME.match(name) or not d.is_dir() or root.resolve() not in d.parents:
        raise FileNotFoundError(name)
    response = _read_json(d / "response.json") if (d / "response.json").is_file() else {}
    graph = _read_json(d / "graph.json") if (d / "graph.json").is_file() else response.get("graph")
    chunks = _read_json(d / "chunks.json") if (d / "chunks.json").is_file() else response.get("chunks", [])
    summary = {}
    for f in ("summary.json", "meta.json"):
        if (d / f).is_file():
            summary = _read_json(d / f)
            break
    if not summary and response:
        summary = {k: response.get(k) for k in ("document", "degraded", "errors", "warnings", "timings_s", "versions")}
    for c in chunks:
        c.pop("embedding", None)
    return {"name": name, "summary": summary, "graph": graph, "chunks": chunks}


class Handler(BaseHTTPRequestHandler):
    root: Path  # set in main()

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/runs":
                self._send_json(200, list_runs(self.root))
            elif self.path.startswith("/api/run/"):
                from urllib.parse import unquote

                self._send_json(200, load_run(self.root, unquote(self.path[len("/api/run/"):])))
            else:
                self._send_json(404, {"error": "not found"})
        except FileNotFoundError as exc:
            self._send_json(404, {"error": f"unknown run: {exc}"})
        except Exception as exc:  # noqa: BLE001 - viewer must never crash
            self._send_json(500, {"error": str(exc)[:300]})

    def _send_json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("out"), help="root folder holding one subfolder per run")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    if not args.dir.is_dir():
        sys.exit(f"not a directory: {args.dir}")
    Handler.root = args.dir
    print(f"serving {args.dir.resolve()} on http://{args.host}:{args.port} "
          f"({len(list_runs(args.dir))} run(s) found)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
