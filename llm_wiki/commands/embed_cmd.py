"""llm-wiki embed — chunk and embed normalized sources for RAG."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from llm_wiki.embedder import EmbedClient, EmbedError
from llm_wiki.rag import RagIndex, chunk_text
from llm_wiki.vault import Vault

console = Console()


def run(
    source_id: Optional[str],
    vault: Vault,
    embedder: EmbedClient,
    all_sources: bool,
    force: bool,
    dry_run: bool,
) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized. Run `llm-wiki init` first.")
        raise SystemExit(1)

    if not embedder.enabled:
        console.print(
            "[red]Error:[/red] Embedding model not configured.\n"
            "Set [bold]LLM_WIKI_EMBED_MODEL[/bold] in your .env file."
        )
        raise SystemExit(1)

    rag = RagIndex(vault)

    if all_sources:
        metas = vault.list_sources()
        if not metas:
            console.print("[yellow]No sources found.[/yellow]")
            return
        for meta in metas:
            _embed_one(meta.source_id, vault, embedder, rag, force, dry_run)

    elif source_id:
        try:
            vault.load_meta(source_id)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
        _embed_one(source_id, vault, embedder, rag, force, dry_run)

    else:
        console.print("[red]Error:[/red] Provide a source-id or --all")
        raise SystemExit(1)


def _embed_one(
    source_id: str,
    vault: Vault,
    embedder: EmbedClient,
    rag: RagIndex,
    force: bool,
    dry_run: bool,
) -> None:
    norm_path = vault.normalized_path(source_id)
    if not norm_path.exists():
        console.print(
            f"[yellow]Skipping {source_id}[/yellow]: not normalized. "
            f"Run `llm-wiki normalize {source_id}` first."
        )
        return

    if rag.is_indexed(source_id) and not force:
        console.print(f"[dim]Skipping {source_id} — already indexed. Use --force to re-embed.[/dim]")
        return

    settings = embedder._settings
    content = norm_path.read_text(encoding="utf-8")
    chunks = chunk_text(content, size=settings.chunk_size, overlap=settings.chunk_overlap)

    console.print(
        f"[bold]{source_id}[/bold]: {len(chunks)} chunks "
        f"({settings.chunk_size} chars, {settings.chunk_overlap} overlap)"
    )

    if dry_run:
        console.print(f"  [dim]DRY RUN — would embed {len(chunks)} chunks[/dim]")
        return

    batch_size = embedder._batch_size
    all_embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Embedding…", total=len(chunks))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                embeddings = embedder.embed(batch)
                all_embeddings.extend(embeddings)
                progress.advance(task, len(batch))
            except EmbedError as e:
                console.print(f"  [red]Embed error at chunk {i}:[/red] {e}")
                return

    rag.save(source_id, chunks, all_embeddings)
    console.print(
        f"  [green]✓[/green] {len(chunks)} chunks indexed "
        f"(dim={len(all_embeddings[0]) if all_embeddings else 0})"
    )
