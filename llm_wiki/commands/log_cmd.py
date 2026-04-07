"""llm-wiki log tail — show recent log entries."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

from llm_wiki.vault import Vault

console = Console()


def run_tail(vault: Vault, n: int = 20) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized.")
        raise SystemExit(1)

    if not vault.log.exists():
        console.print("[yellow]Log is empty.[/yellow]")
        return

    log_text = vault.log.read_text(encoding="utf-8")

    # Split on ## (log entries)
    raw_entries = log_text.split("\n## ")
    entries = []
    for e in raw_entries:
        e = e.strip()
        if e and not e.startswith("# Wiki Log"):
            entries.append("## " + e if not e.startswith("##") else e)

    tail = entries[-n:] if len(entries) >= n else entries
    tail_text = "\n\n".join(reversed(tail))  # newest first

    if not tail_text.strip():
        console.print("[yellow]No log entries yet.[/yellow]")
        return

    console.print(Markdown(f"# Recent Log (last {len(tail)} entries, newest first)\n\n{tail_text}"))
