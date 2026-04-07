"""llm-wiki ask — ask a question against the wiki."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from llm_wiki.llm import LLMClient, LLMError
from llm_wiki.schemas.models import AnswerResult
from llm_wiki.vault import Vault, slugify, utcnow

console = Console()

MAX_PAGES = 8
MAX_PAGE_CHARS = 6_000  # per page, to stay within context


def run(question: str, vault: Vault, llm: LLMClient, save: bool, dry_run: bool) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized.")
        raise SystemExit(1)

    # 1. Find relevant pages
    with Progress(SpinnerColumn(), TextColumn("Searching wiki…"), transient=True, console=console):
        relevant = _find_relevant_pages(question, vault)

    if not relevant:
        console.print("[yellow]No relevant wiki pages found. Try ingesting sources first.[/yellow]")
        return

    console.print(f"[dim]Consulting {len(relevant)} page(s)…[/dim]")

    # 2. Build context
    context_parts = []
    consulted_paths = []
    for page_path in relevant[:MAX_PAGES]:
        try:
            text = page_path.read_text(encoding="utf-8")[:MAX_PAGE_CHARS]
            rel = page_path.relative_to(vault.wiki)
            context_parts.append(f"### {rel}\n\n{text}")
            consulted_paths.append(str(rel))
        except Exception:
            pass

    context = "\n\n---\n\n".join(context_parts)

    # 3. Ask LLM
    with Progress(SpinnerColumn(), TextColumn("Thinking…"), transient=True, console=console):
        try:
            answer = _ask_llm(question, context, consulted_paths, vault.load_schema(), llm)
        except LLMError as e:
            console.print(f"[red]LLM error:[/red] {e}")
            return

    # 4. Display answer
    console.print()
    panel_content = f"**{answer.answer}**\n\n{answer.reasoning}"
    if answer.citations:
        panel_content += "\n\n**Sources consulted:**\n" + "\n".join(
            f"- `{c}`" for c in answer.citations
        )
    if answer.gaps:
        panel_content += "\n\n**Knowledge gaps:**\n" + "\n".join(f"- {g}" for g in answer.gaps)

    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}.get(answer.confidence, "white")
    console.print(
        Panel(
            Markdown(panel_content),
            title=f"[bold]{question}[/bold]",
            subtitle=f"Confidence: [{confidence_color}]{answer.confidence}[/{confidence_color}]",
            border_style="blue",
        )
    )

    # 5. Optionally save
    if save and not dry_run:
        _save_answer(question, answer, vault)
    elif save and dry_run:
        console.print("[dim]DRY RUN — would save answer to wiki/analyses/[/dim]")


def _find_relevant_pages(question: str, vault: Vault) -> list[Path]:
    """Score wiki pages by keyword relevance to the question."""
    terms = set(re.findall(r"\w+", question.lower())) - _STOP_WORDS

    scored: list[tuple[float, Path]] = []
    for page_path in vault.wiki.rglob("*.md"):
        if page_path.name in ("index.md", "log.md", "overview.md"):
            # Still search these but at lower priority
            text = page_path.read_text(encoding="utf-8").lower()
            score = sum(text.count(t) * 0.5 for t in terms)
        else:
            text = page_path.read_text(encoding="utf-8").lower()
            score = sum(text.count(t) for t in terms)
            # Boost title matches
            first_line = text.split("\n")[0]
            score += sum(10 for t in terms if t in first_line)

        if score > 0:
            scored.append((score, page_path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def _ask_llm(
    question: str,
    context: str,
    consulted_paths: list[str],
    schema_ctx: str,
    llm: LLMClient,
) -> AnswerResult:
    system = f"""\
You are a knowledgeable assistant answering questions based on a local wiki.
Only use information from the provided wiki pages. Cite your sources.
If the wiki does not contain sufficient information, say so clearly.

{schema_ctx}
"""
    user = f"""\
Question: {question}

Wiki pages consulted:
---
{context}
---

Return a JSON object:
{{
  "answer": "Direct 1-2 sentence answer",
  "reasoning": "Detailed explanation with citations",
  "citations": ["path/to/page1.md", ...],
  "confidence": "high|medium|low",
  "gaps": ["what information is missing that would help", ...]
}}
"""
    data = llm.chat_json(system, user)
    return AnswerResult.model_validate(data)


def _save_answer(question: str, answer: AnswerResult, vault: Vault) -> None:
    ts = utcnow()
    slug = slugify(question[:50])
    filename = f"{ts[:10]}-{slug}.md"
    out_path = vault.wiki_analyses / filename

    content = f"""\
---
type: analysis
question: "{question}"
answered_at: {ts}
confidence: {answer.confidence}
sources_consulted:
{chr(10).join(f'  - {c}' for c in answer.citations)}
---

# Q: {question}

**{answer.answer}**

## Detailed Answer

{answer.reasoning}

## Sources Consulted

{chr(10).join(f'- [{c}]({_rel_path(c)}){chr(10)}' for c in answer.citations)}
"""
    if answer.gaps:
        content += "\n## Knowledge Gaps\n\n"
        content += "\n".join(f"- {g}" for g in answer.gaps)
        content += "\n"

    out_path.write_text(content, encoding="utf-8")
    console.print(f"\n[green]✓[/green] Answer saved to [cyan]wiki/analyses/{filename}[/cyan]")


def _rel_path(citation: str) -> str:
    # citations are relative to wiki/, analyses pages are one level in
    return f"../{citation}"


_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "be", "was", "are", "with", "this", "that", "what",
    "how", "why", "when", "where", "who", "which", "can", "do", "does", "did",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
}
