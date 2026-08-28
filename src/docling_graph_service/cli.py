"""``dgs`` command line: process | serve | capabilities | ontology-check."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="docling-graph-service CLI")


@app.command()
def process(
    path: Path = typer.Argument(..., exists=True, readable=True, help="PDF (or docx/pptx/html/md) to process"),
    out: Path = typer.Option(Path("out"), help="Output directory"),
    ontology: Path | None = typer.Option(None, help="Ontology YAML (default: graph.default_ontology_path)"),
    vlm: bool = typer.Option(True, "--vlm/--no-vlm"),
    graph: bool = typer.Option(True, "--graph/--no-graph"),
    embed: bool = typer.Option(True, "--embed/--no-embed"),
    ocr: bool | None = typer.Option(None, "--ocr/--no-ocr"),
    chunk_max_tokens: int | None = typer.Option(None),
    extraction_contract: str | None = typer.Option(None, help="dense | direct"),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run the full pipeline on one file and write document.json, markdown.md, chunks.json, graph.json, meta.json."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .ontology import load_ontology
    from .pipeline import Runtime
    from .pipeline import process as run
    from .schemas import PipelineOptions
    from .settings import get_settings

    rt = Runtime(get_settings())
    options = PipelineOptions(vlm_enabled=vlm, graph_enabled=graph, embedding_enabled=embed, ocr_enabled=ocr,
                              chunk_max_tokens=chunk_max_tokens, extraction_contract=extraction_contract)  # type: ignore[arg-type]
    ontology_graph = load_ontology(ontology).model_dump(mode="json") if ontology else None
    outcome = run(rt, path.read_bytes(), path.name, None, options, ontology_graph, keep_document=True)
    resp = outcome.response
    out.mkdir(parents=True, exist_ok=True)
    if outcome.document is not None:
        outcome.document.save_as_json(out / "document.json")
    (out / "markdown.md").write_text(resp.markdown, encoding="utf-8")
    (out / "chunks.json").write_text(json.dumps([c.model_dump() for c in resp.chunks], ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    if resp.graph is not None:
        (out / "graph.json").write_text(resp.graph.model_dump_json(indent=1), encoding="utf-8")
    meta = resp.model_dump(exclude={"markdown", "chunks", "graph"})
    meta["chunk_count"] = len(resp.chunks)
    if resp.graph is not None:
        meta["graph_meta"] = resp.graph.meta
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    typer.echo(f"pages={resp.document.pages} tables={resp.document.tables} pictures={resp.document.pictures} "
               f"chunks={len(resp.chunks)} nodes={len(resp.graph.nodes) if resp.graph else '-'} "
               f"edges={len(resp.graph.edges) if resp.graph else '-'} degraded={resp.degraded.model_dump()} "
               f"timings={resp.timings_s}")
    for e in resp.errors:
        typer.echo(f"ERROR {e}", err=True)
    for w in resp.warnings:
        typer.echo(f"warning: {w}", err=True)
    typer.echo(f"written to {out}/")


@app.command()
def serve(host: str | None = typer.Option(None), port: int | None = typer.Option(None)) -> None:
    """Start the HTTP API (uvicorn)."""
    import uvicorn

    from .settings import get_settings

    s = get_settings().service
    uvicorn.run("docling_graph_service.api:app", host=host or s.host, port=port or s.port, log_level=s.log_level.lower())


@app.command()
def capabilities() -> None:
    """Print the effective configuration (keys redacted) and endpoint status."""
    from .llm_http import LLMClient
    from .pipeline import versions
    from .settings import get_settings, redact

    s = get_settings()
    info = {"versions": versions(), "models": {}}
    for name, sect in (("llm", s.llm), ("vlm", s.vlm), ("embedding", s.embedding)):
        client = LLMClient(sect.base_url or "", sect.api_key, 10.0)
        info["models"][name] = {"model": sect.model, **redact(sect.base_url or "", sect.api_key),
                                "status": client.probe()}
        client.close()
    info["chunking"] = s.chunking.model_dump()
    info["docling"] = s.docling.model_dump()
    info["graph"] = s.graph.model_dump()
    typer.echo(json.dumps(info, indent=2, ensure_ascii=False))


@app.command("ontology-check")
def ontology_check(path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate an ontology file against the contract and show the compiled template."""
    from pydantic import ValidationError

    from .ontology import check_report, compile_template, load_ontology

    try:
        ont = load_ontology(path)
    except ValidationError as exc:
        typer.echo("Ontology does not satisfy the contract:", err=True)
        for err in exc.errors(include_url=False):
            loc = ".".join(str(x) for x in err["loc"])
            typer.echo(f"  {loc}: {err['msg']}", err=True)
        raise typer.Exit(1) from None
    typer.echo(check_report(compile_template(ont)))


if __name__ == "__main__":
    app()
