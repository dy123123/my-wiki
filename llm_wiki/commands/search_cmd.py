"""llm-wiki search — full-text search across wiki pages."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_wiki.vault import Vault

console = Console()

SNIPPET_WINDOW = 80  # chars each side of a match


def run(query: str, vault: Vault, max_results: int = 20) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized.")
        raise SystemExit(1)

    terms = re.findall(r"\w+", query.lower())
    if not terms:
        console.print("[yellow]Empty query.[/yellow]")
        return

    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    results: list[tuple[int, Path, str]] = []

    for page_path in sorted(vault.wiki.rglob("*.md")):
        try:
            text = page_path.read_text(encoding="utf-8")
        except Exception:
            continue

        matches = list(pattern.finditer(text))
        if not matches:
            continue

        score = len(matches)
        snippet = _make_snippet(text, matches[0], pattern)
        results.append((score, page_path, snippet))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        console.print(f"[yellow]No results for:[/yellow] {query}")
        return

    table = Table(
        title=f"Search: \"{query}\" — {len(results)} result(s)",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("Score", style="cyan", no_wrap=True, width=6)
    table.add_column("Page", style="bold", no_wrap=True)
    table.add_column("Snippet")

    for score, path, snippet in results[:max_results]:
        rel = str(path.relative_to(vault.wiki))
        table.add_row(str(score), rel, snippet)

    console.print(table)
    if len(results) > max_results:
        console.print(f"[dim]… and {len(results) - max_results} more.[/dim]")


def _make_snippet(text: str, match: re.Match, pattern: re.Pattern) -> str:
    start = max(0, match.start() - SNIPPET_WINDOW)
    end = min(len(text), match.end() + SNIPPET_WINDOW)
    snippet = text[start:end].replace("\n", " ").strip()

    # Highlight matches
    highlighted = pattern.sub(lambda m: f"[bold yellow]{m.group()}[/bold yellow]", snippet)

    if start > 0:
        highlighted = "…" + highlighted
    if end < len(text):
        highlighted = highlighted + "…"

    return highlighted
