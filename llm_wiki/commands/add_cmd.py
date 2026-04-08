"""llm-wiki add — add a raw source file to the vault."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_wiki.vault import Vault

console = Console()


def run(path: Path, vault: Vault, tags: list[str], dry_run: bool) -> None:
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise SystemExit(1)

    if not path.is_file():
        console.print(f"[red]Error:[/red] Path is not a file: {path}")
        raise SystemExit(1)

    size_kb = path.stat().st_size / 1024

    if dry_run:
        from llm_wiki.vault import source_id_from_path
        source_id = source_id_from_path(path)
        console.print(f"[dim]DRY RUN — would add:[/dim]")
        console.print(f"  File    : {path}")
        console.print(f"  Size    : {size_kb:.1f} KB")
        console.print(f"  ID      : {source_id}")
        console.print(f"  Tags    : {', '.join(tags) or 'none'}")
        console.print(f"  Raw dest: {vault.raw}/{source_id}{path.suffix.lower()}")
        return

    meta = vault.add_source(path, tags=tags)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]Source ID[/dim]", f"[bold cyan]{meta.source_id}[/bold cyan]")
    table.add_row("[dim]Original[/dim]", meta.original_name)
    table.add_row("[dim]Size[/dim]", f"{size_kb:.1f} KB")
    table.add_row("[dim]Extension[/dim]", meta.extension)
    table.add_row("[dim]Tags[/dim]", ", ".join(meta.tags) or "none")
    table.add_row("[dim]Raw path[/dim]", str(vault.raw_path(meta).relative_to(vault.path.parent)))

    console.print(f"[green]✓[/green] Source added")
    console.print(table)
    console.print()
    console.print(f"Next: [bold]llm-wiki normalize {meta.source_id}[/bold]")
