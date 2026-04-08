"""llm-wiki config — show and validate configuration."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_wiki.config import Settings

console = Console()


def run_show(settings: Settings) -> None:
    table = Table(title="llm-wiki Configuration", show_header=True, header_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    display = settings.display_dict()
    env_fields = {
        "llm_base_url": "LLM_WIKI_LLM_BASE_URL",
        "llm_api_key": "LLM_WIKI_LLM_API_KEY",
        "llm_model": "LLM_WIKI_LLM_MODEL",
        "llm_temperature": "LLM_WIKI_LLM_TEMPERATURE",
        "llm_max_tokens": "LLM_WIKI_LLM_MAX_TOKENS",
        "vault_path": "LLM_WIKI_VAULT_PATH",
        "dry_run": "LLM_WIKI_DRY_RUN",
        "verbose": "LLM_WIKI_VERBOSE",
    }

    for key, env_var in env_fields.items():
        value = str(display.get(key, ""))
        table.add_row(key, value, env_var)

    console.print(table)


def run_validate(settings: Settings) -> bool:
    issues: list[str] = []
    ok = True

    checks = [
        (bool(settings.llm_api_key), "LLM_WIKI_LLM_API_KEY", "API key is not set"),
        (bool(settings.llm_base_url), "LLM_WIKI_LLM_BASE_URL", "Base URL is not set"),
        (bool(settings.llm_model), "LLM_WIKI_LLM_MODEL", "Model is not set"),
        (0.0 <= settings.llm_temperature <= 2.0, "LLM_WIKI_LLM_TEMPERATURE", "Temperature must be in [0.0, 2.0]"),
        (settings.llm_max_tokens > 0, "LLM_WIKI_LLM_MAX_TOKENS", "max_tokens must be positive"),
    ]

    for passed, field, msg in checks:
        if passed:
            console.print(f"[green]✓[/green] {field}")
        else:
            console.print(f"[red]✗[/red] {field}: {msg}")
            issues.append(msg)
            ok = False

    if ok:
        console.print("\n[green]Configuration is valid.[/green]")
    else:
        console.print(f"\n[red]{len(issues)} issue(s) found.[/red] Check your .env file.")

    return ok
