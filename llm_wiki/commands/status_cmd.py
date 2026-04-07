"""llm-wiki status — show vault health overview."""

from __future__ import annotations

from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_wiki.vault import Vault

console = Console()


def run(vault: Vault) -> None:
    if not vault.exists():
        console.print("[red]Vault not initialized.[/red] Run `llm-wiki init` to create one.")
        return

    metas = vault.list_sources()
    normalized_count = sum(1 for m in metas if vault.normalized_path(m.source_id).exists())
    ingested_count = sum(1 for m in metas if m.ingested_at is not None)

    wiki_page_counts: dict[str, int] = {}
    for subdir in [
        vault.wiki_sources, vault.wiki_entities, vault.wiki_concepts,
        vault.wiki_topics, vault.wiki_analyses, vault.wiki_reports,
    ]:
        wiki_page_counts[subdir.name] = len(list(subdir.glob("*.md"))) if subdir.exists() else 0

    total_wiki_pages = sum(wiki_page_counts.values())

    # --- Summary table ---
    stats = Table(title="Vault Status", show_header=False, box=None, padding=(0, 2))
    stats.add_column("Label", style="dim")
    stats.add_column("Value", style="bold")

    stats.add_row("Vault path", str(vault.path))
    stats.add_row("Sources (raw)", str(len(metas)))
    stats.add_row("  — normalized", str(normalized_count))
    stats.add_row("  — ingested", str(ingested_count))
    stats.add_row("Wiki pages (total)", str(total_wiki_pages))
    for subdir, count in wiki_page_counts.items():
        stats.add_row(f"  — {subdir}", str(count))

    console.print(stats)

    # --- Recent log entries ---
    if vault.log.exists():
        log_text = vault.log.read_text(encoding="utf-8")
        entries = [e.strip() for e in log_text.split("##") if e.strip() and not e.strip().startswith("Wiki Log")]
        if entries:
            console.print()
            console.rule("[dim]Recent activity[/dim]")
            for entry in entries[-3:]:
                lines = entry.strip().split("\n")
                timestamp = lines[0].strip() if lines else "?"
                details = "\n".join(lines[1:]).strip()
                console.print(f"[dim]{timestamp}[/dim]")
                if details:
                    for line in details.split("\n")[:3]:
                        console.print(f"  {line}")

    # --- Quick health check ---
    console.print()
    pending_normalize = [m for m in metas if not vault.normalized_path(m.source_id).exists()]
    pending_ingest = [m for m in metas if m.ingested_at is None and vault.normalized_path(m.source_id).exists()]

    if pending_normalize:
        console.print(f"[yellow]⚠[/yellow]  {len(pending_normalize)} source(s) need normalizing:")
        for m in pending_normalize[:5]:
            console.print(f"   llm-wiki normalize {m.source_id}")
        if len(pending_normalize) > 5:
            console.print(f"   … or: llm-wiki normalize --all")

    if pending_ingest:
        console.print(f"[yellow]⚠[/yellow]  {len(pending_ingest)} source(s) need ingesting:")
        for m in pending_ingest[:5]:
            console.print(f"   llm-wiki ingest {m.source_id}")
        if len(pending_ingest) > 5:
            console.print(f"   … or: llm-wiki ingest --all")

    if not pending_normalize and not pending_ingest:
        console.print("[green]✓ All sources are normalized and ingested.[/green]")
