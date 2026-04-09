"""llm-wiki CLI — LLM-maintained local markdown wiki."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from llm_wiki.config import Settings, get_settings
from llm_wiki.vault import Vault

app = typer.Typer(
    name="llm-wiki",
    help="An LLM-maintained local markdown wiki.",
    add_completion=False,
    no_args_is_help=True,
)

config_app = typer.Typer(help="Configuration commands.", no_args_is_help=True)
llm_app = typer.Typer(help="LLM backend commands.", no_args_is_help=True)
log_app = typer.Typer(help="Log commands.", no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(llm_app, name="llm")
app.add_typer(log_app, name="log")

console = Console()


# ------------------------------------------------------------------ #
#  Shared option helpers
# ------------------------------------------------------------------ #

def _get_vault(settings: Settings) -> Vault:
    return Vault(settings.vault_path)


def _get_llm(settings: Settings):
    from llm_wiki.llm import LLMClient
    return LLMClient(settings)


def _get_embedder(settings: Settings):
    from llm_wiki.embedder import EmbedClient
    return EmbedClient(settings)


def _get_rag(vault: Vault):
    from llm_wiki.rag import RagIndex
    return RagIndex(vault)


# ------------------------------------------------------------------ #
#  init
# ------------------------------------------------------------------ #

@app.command()
def init(
    vault: Annotated[
        Optional[Path],
        typer.Option("--vault", "-v", help="Vault directory path (overrides config)"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without writing")] = False,
) -> None:
    """Initialize a new wiki vault."""
    settings = get_settings()
    from llm_wiki.commands.init_cmd import run
    run(vault or settings.vault_path, dry_run or settings.dry_run)


# ------------------------------------------------------------------ #
#  config
# ------------------------------------------------------------------ #

@config_app.command("show")
def config_show() -> None:
    """Display current configuration."""
    settings = get_settings()
    from llm_wiki.commands.config_cmd import run_show
    run_show(settings)


@config_app.command("validate")
def config_validate() -> None:
    """Validate configuration and exit with non-zero on error."""
    settings = get_settings()
    from llm_wiki.commands.config_cmd import run_validate
    ok = run_validate(settings)
    if not ok:
        raise typer.Exit(1)


# ------------------------------------------------------------------ #
#  llm
# ------------------------------------------------------------------ #

@llm_app.command("ping")
def llm_ping() -> None:
    """Check connectivity to the LLM backend."""
    settings = get_settings()
    llm = _get_llm(settings)
    console.print(f"Pinging [bold]{settings.llm_base_url}[/bold] with model [bold]{settings.llm_model}[/bold]…")
    success, msg = llm.ping()
    if success:
        console.print(f"[green]✓ {msg}[/green]")
    else:
        console.print(f"[red]✗ {msg}[/red]")
        raise typer.Exit(1)


# ------------------------------------------------------------------ #
#  add
# ------------------------------------------------------------------ #

@app.command()
def add(
    path: Annotated[Path, typer.Argument(help="Path to the source file to add")],
    tags: Annotated[
        Optional[list[str]],
        typer.Option("--tag", "-t", help="Tags (repeatable: --tag ai --tag nlp)"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Add a raw source file to the vault."""
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.add_cmd import run
    run(path, vault, tags or [], dry_run or settings.dry_run)


# ------------------------------------------------------------------ #
#  normalize
# ------------------------------------------------------------------ #

@app.command()
def normalize(
    source_id: Annotated[Optional[str], typer.Argument(help="Source ID to normalize")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Normalize all sources")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Convert raw source files to markdown."""
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.normalize_cmd import run
    run(source_id, vault, all_sources, dry_run or settings.dry_run)


# ------------------------------------------------------------------ #
#  ingest
# ------------------------------------------------------------------ #

@app.command()
def ingest(
    source_id: Annotated[Optional[str], typer.Argument(help="Source ID to ingest")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Ingest all sources")] = False,
    latest: Annotated[bool, typer.Option("--latest", help="Ingest the most recently added source")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without writing")] = False,
) -> None:
    """Analyze a source and create/update wiki pages."""
    settings = get_settings()
    vault = _get_vault(settings)
    llm = _get_llm(settings)
    from llm_wiki.commands.ingest_cmd import run
    run(source_id, vault, llm, all_sources, latest, dry_run or settings.dry_run)


# ------------------------------------------------------------------ #
#  process (normalize + ingest in one step)
# ------------------------------------------------------------------ #

@app.command()
def process(
    source_id: Annotated[Optional[str], typer.Argument(help="Source ID to process")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Process all sources")] = False,
    latest: Annotated[bool, typer.Option("--latest", help="Process the most recently added source")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without writing")] = False,
) -> None:
    """Normalize then ingest a source in one step."""
    settings = get_settings()
    vault = _get_vault(settings)
    llm = _get_llm(settings)
    dry = dry_run or settings.dry_run

    from llm_wiki.commands.normalize_cmd import run as run_normalize
    from llm_wiki.commands.ingest_cmd import run as run_ingest

    run_normalize(source_id, vault, all_sources, dry)
    run_ingest(source_id, vault, llm, all_sources, latest, dry)

    if settings.embed_model:
        embedder = _get_embedder(settings)
        from llm_wiki.commands.embed_cmd import run as run_embed
        run_embed(source_id, vault, embedder, all_sources, False, dry)


# ------------------------------------------------------------------ #
#  ask
# ------------------------------------------------------------------ #

@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the wiki")],
    save: Annotated[bool, typer.Option("--save", "-s", help="Save answer to analyses/")] = False,
    no_rag: Annotated[bool, typer.Option("--no-rag", help="Skip RAG retrieval")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Ask a question — searches wiki pages then RAG chunks (if configured)."""
    settings = get_settings()
    vault = _get_vault(settings)
    llm = _get_llm(settings)
    embedder = None
    rag = None
    if not no_rag and settings.embed_model:
        embedder = _get_embedder(settings)
        rag = _get_rag(vault)
    from llm_wiki.commands.ask_cmd import run
    run(question, vault, llm, save, dry_run or settings.dry_run, embedder=embedder, rag=rag)


# ------------------------------------------------------------------ #
#  embed
# ------------------------------------------------------------------ #

@app.command()
def embed(
    source_id: Annotated[Optional[str], typer.Argument(help="Source ID to embed")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Embed all sources")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-embed even if already indexed")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Chunk and embed normalized sources for RAG retrieval."""
    settings = get_settings()
    vault = _get_vault(settings)
    embedder = _get_embedder(settings)
    from llm_wiki.commands.embed_cmd import run
    run(source_id, vault, embedder, all_sources, force, dry_run or settings.dry_run)


# ------------------------------------------------------------------ #
#  search
# ------------------------------------------------------------------ #

@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    max_results: Annotated[int, typer.Option("--max", "-n", help="Maximum results to show")] = 20,
) -> None:
    """Full-text search across all wiki pages."""
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.search_cmd import run
    run(query, vault, max_results)


# ------------------------------------------------------------------ #
#  lint
# ------------------------------------------------------------------ #

@app.command()
def lint(
    fix: Annotated[bool, typer.Option("--fix", help="Auto-fix simple issues")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview fixes without writing")] = False,
) -> None:
    """Detect structural issues in the wiki."""
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.lint_cmd import run
    exit_code = run(vault, fix, dry_run or settings.dry_run)
    raise typer.Exit(exit_code)


# ------------------------------------------------------------------ #
#  status
# ------------------------------------------------------------------ #

@app.command()
def status() -> None:
    """Show vault health and pipeline status."""
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.status_cmd import run
    run(vault)


# ------------------------------------------------------------------ #
#  mcp
# ------------------------------------------------------------------ #

@app.command()
def mcp(
    http: Annotated[bool, typer.Option("--http", help="Run as HTTP/SSE server (for remote access)")] = False,
    host: Annotated[str, typer.Option("--host", help="HTTP bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="HTTP port")] = 8080,
    token: Annotated[str, typer.Option("--token", help="Bearer token for authentication (recommended)")] = "",
) -> None:
    """Start an MCP server — stdio (local) or HTTP/SSE (remote).

    Local (OpenCode / Claude Desktop on same machine):
      llm-wiki mcp

    Remote (access from another machine):
      llm-wiki mcp --http --port 8080 --token mysecrettoken
    """
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.mcp_cmd import run
    run(settings, vault, http=http, host=host, port=port, token=token)


# ------------------------------------------------------------------ #
#  web
# ------------------------------------------------------------------ #

@app.command()
def web(
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port")] = 7432,
    token: Annotated[str, typer.Option("--token", help="Bearer token for auth")] = "",
) -> None:
    """Start the web UI + REST API + MCP HTTP/SSE server.

    Opens a browser-accessible UI for managing sources, browsing the wiki,
    asking questions, and monitoring MCP connections.

    MCP SSE endpoint:  http://<host>:<port>/mcp/sse
    """
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.web_cmd import run
    run(settings, vault, host=host, port=port, token=token)


# ------------------------------------------------------------------ #
#  log tail
# ------------------------------------------------------------------ #

@log_app.command("tail")
def log_tail(
    n: Annotated[int, typer.Option("--lines", "-n", help="Number of entries to show")] = 20,
) -> None:
    """Show recent wiki log entries."""
    settings = get_settings()
    vault = _get_vault(settings)
    from llm_wiki.commands.log_cmd import run_tail
    run_tail(vault, n)
