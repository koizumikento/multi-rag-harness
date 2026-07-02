"""Command-line entry points for multi-rag-harness.

Thin layer: commands load settings, build the service container, delegate,
and print a summary. The MCP server itself runs over stdio via ``serve``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from multi_rag_harness.config import load_settings

app = typer.Typer(name="multi-rag-harness", help="Multi-RAG retrieval and memory substrate.")
console = Console()

ConfigOption = Annotated[Path | None, typer.Option("--config", help="Path to a TOML config file.")]


@app.command()
def serve(config: ConfigOption = None) -> None:
    """Run the MCP server over stdio."""
    from multi_rag_harness.mcp_server.server import run_stdio

    run_stdio(load_settings(config))


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to ingest.")],
    scope: Annotated[str, typer.Option(help="Logical scope for the documents.")] = "default",
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Tag to attach (repeatable).")
    ] = None,
    kind: Annotated[str | None, typer.Option(help="Override detected kind (doc/code).")] = None,
    extract: Annotated[
        bool, typer.Option(help="Queue graph extraction runs for new chunks.")
    ] = False,
    config: ConfigOption = None,
) -> None:
    """Ingest documents into the retrieval indexes."""
    from multi_rag_harness.mcp_server.server import build_container, close_container

    settings = load_settings(config)

    async def _run():
        container = await build_container(settings)
        try:
            return await container.pipeline.ingest_path(
                path,
                scope=scope,
                tags=list(tag or []),
                kind_override=kind,
                extract=True if extract else None,
            )
        finally:
            await close_container(container)

    report = asyncio.run(_run())
    table = Table(title="Ingest Report")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key, value in report.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def extract(
    limit: Annotated[int, typer.Option(help="Maximum extraction runs to process.")] = 25,
    config: ConfigOption = None,
) -> None:
    """Run pending Codex graph extraction jobs."""
    from multi_rag_harness.mcp_server.server import build_container, close_container

    settings = load_settings(config)

    async def _run():
        container = await build_container(settings)
        try:
            return await container.extraction.run_pending(limit)
        finally:
            await close_container(container)

    summary = asyncio.run(_run())
    table = Table(title="Extraction Summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key, value in summary.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command("config-show")
def config_show(config: ConfigOption = None) -> None:
    """Print the resolved configuration."""
    settings = load_settings(config)
    table = Table(title="multi-rag-harness configuration")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("data_dir", str(settings.data_dir))
    for section_name in ("embedding", "reranker", "storage", "mcp", "codex"):
        section = getattr(settings, section_name)
        for key, value in section.model_dump().items():
            table.add_row(f"{section_name}.{key}", str(value))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
