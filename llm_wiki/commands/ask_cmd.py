"""llm-wiki ask — ask a question against the wiki (+ RAG)."""

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

MAX_PAGES = 6
MAX_PAGE_CHARS = 4_000   # per wiki page
MAX_CHUNK_CHARS = 1_200  # per RAG chunk shown to LLM
MAX_CHUNKS = 5


def run(
    question: str,
    vault: Vault,
    llm: LLMClient,
    save: bool,
    dry_run: bool,
    embedder=None,
    rag=None,
) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized.")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # 1. Wiki keyword search
    # ------------------------------------------------------------------ #
    with Progress(SpinnerColumn(), TextColumn("Searching wiki…"), transient=True, console=console):
        wiki_pages = _find_relevant_pages(question, vault)

    # ------------------------------------------------------------------ #
    # 2. RAG chunk retrieval (if embed model is configured)
    # ------------------------------------------------------------------ #
    rag_chunks = []
    if embedder is not None and embedder.enabled and rag is not None:
        with Progress(SpinnerColumn(), TextColumn("RAG retrieval…"), transient=True, console=console):
            try:
                from llm_wiki.embedder import EmbedError
                q_embeddings = embedder.embed([question])
                if q_embeddings:
                    settings = embedder._settings
                    candidates = rag.search(q_embeddings[0], top_k=MAX_CHUNKS * 4)
                    rag_chunks = rag.rerank_and_trim(
                        question, candidates, embedder, top_k=MAX_CHUNKS
                    )
            except EmbedError as e:
                console.print(f"[dim]RAG unavailable: {e}[/dim]")

    if not wiki_pages and not rag_chunks:
        console.print(
            "[yellow]No relevant content found. "
            "Try ingesting sources first, or run `llm-wiki embed --all` to enable RAG.[/yellow]"
        )
        return

    console.print(
        f"[dim]Consulting {len(wiki_pages[:MAX_PAGES])} wiki page(s)"
        + (f" + {len(rag_chunks)} RAG chunk(s)" if rag_chunks else "")
        + "[/dim]"
    )

    # ------------------------------------------------------------------ #
    # 3. Build combined context
    # ------------------------------------------------------------------ #
    context_parts: list[str] = []
    consulted_paths: list[str] = []

    # Wiki pages first (structural knowledge)
    for page_path in wiki_pages[:MAX_PAGES]:
        try:
            text = page_path.read_text(encoding="utf-8")[:MAX_PAGE_CHARS]
            rel = page_path.relative_to(vault.wiki)
            context_parts.append(f"### [Wiki] {rel}\n\n{text}")
            consulted_paths.append(str(rel))
        except Exception:
            pass

    # RAG chunks (precise detail)
    if rag_chunks:
        context_parts.append("### [Source Chunks — raw document excerpts]")
        for chunk in rag_chunks:
            context_parts.append(
                f"**Source:** `{chunk.source_id}` (chunk {chunk.chunk_idx}, "
                f"similarity {chunk.score:.3f})\n\n{chunk.text[:MAX_CHUNK_CHARS]}"
            )
            if f"sources/{chunk.source_id}.md" not in consulted_paths:
                consulted_paths.append(f"chunks/{chunk.source_id}#{chunk.chunk_idx}")

    context = "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------ #
    # 4. Ask LLM
    # ------------------------------------------------------------------ #
    with Progress(SpinnerColumn(), TextColumn("Thinking…"), transient=True, console=console):
        try:
            answer = _ask_llm(question, context, consulted_paths, vault.load_schema(), llm)
        except LLMError as e:
            console.print(f"[red]LLM error:[/red] {e}")
            return

    # ------------------------------------------------------------------ #
    # 5. Display
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 6. Optionally save
    # ------------------------------------------------------------------ #
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
            text = page_path.read_text(encoding="utf-8").lower()
            score = sum(text.count(t) * 0.5 for t in terms)
        else:
            text = page_path.read_text(encoding="utf-8").lower()
            score = sum(text.count(t) for t in terms)
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
    has_chunks = "[Source Chunks" in context
    system = f"""\
You are a knowledgeable assistant answering questions from a local wiki and raw source documents.
Use information from the provided wiki pages AND source chunks.
The source chunks contain exact text from the original documents — prefer them for precise values
(register addresses, constants, specifications, etc.).
Cite your sources. If the provided context is insufficient, say so clearly.

{schema_ctx}
"""
    user = f"""\
Question: {question}

{"The context below includes both wiki summary pages and raw document chunks." if has_chunks else ""}

Context:
---
{context}
---

Return a JSON object:
{{
  "answer": "Direct 1-2 sentence answer",
  "reasoning": "Detailed explanation with citations",
  "citations": ["path/to/page1.md", ...],
  "confidence": "high|medium|low",
  "gaps": ["what information is missing", ...]
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

{chr(10).join(f'- `{c}`' for c in answer.citations)}
"""
    if answer.gaps:
        content += "\n## Knowledge Gaps\n\n"
        content += "\n".join(f"- {g}" for g in answer.gaps)
        content += "\n"

    out_path.write_text(content, encoding="utf-8")
    console.print(f"\n[green]✓[/green] Answer saved to [cyan]wiki/analyses/{filename}[/cyan]")


_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "be", "was", "are", "with", "this", "that", "what",
    "how", "why", "when", "where", "who", "which", "can", "do", "does", "did",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
}
