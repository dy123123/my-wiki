"""llm-wiki init — initialize a new vault."""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.tree import Tree

from llm_wiki.vault import Vault

console = Console()


def run(vault_path: Optional[Path], dry_run: bool) -> None:
    vault = Vault(vault_path or Path("vault"))

    if vault.exists():
        console.print(f"[yellow]Vault already exists at {vault.path}[/yellow]")
        _show_structure(vault)
        return

    if dry_run:
        console.print(f"[dim]DRY RUN — would create vault at {vault.path}[/dim]")
        _show_planned_structure(vault)
        return

    vault.init()
    console.print(f"[green]✓[/green] Vault initialized at [bold]{vault.path}[/bold]")
    _show_structure(vault)
    console.print()
    console.print("Next steps:")
    console.print("  1. Copy [bold].env.example[/bold] to [bold].env[/bold] and set your API key")
    console.print("  2. Run [bold]llm-wiki llm ping[/bold] to verify connectivity")
    console.print("  3. Run [bold]llm-wiki add <file>[/bold] to add your first source")


def _show_structure(vault: Vault) -> None:
    tree = Tree(f"[bold]{vault.path}[/bold]")
    for d in [vault.raw, vault.normalized, vault.wiki, vault.schema]:
        branch = tree.add(f"[blue]{d.name}/[/blue]")
        if d == vault.wiki:
            for sub in [vault.wiki_sources, vault.wiki_entities, vault.wiki_concepts,
                        vault.wiki_topics, vault.wiki_analyses, vault.wiki_reports]:
                branch.add(f"[dim]{sub.name}/[/dim]")
    console.print(tree)


def _show_planned_structure(vault: Vault) -> None:
    console.print(f"[dim]Would create:[/dim]")
    for d in [
        vault.raw, vault.normalized, vault.schema,
        vault.wiki_sources, vault.wiki_entities, vault.wiki_concepts,
        vault.wiki_topics, vault.wiki_analyses, vault.wiki_reports,
        vault.index, vault.log, vault.overview,
    ]:
        console.print(f"  [dim]{d.relative_to(vault.path.parent)}[/dim]")
