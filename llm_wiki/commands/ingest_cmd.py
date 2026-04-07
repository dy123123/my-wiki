"""llm-wiki ingest — analyze normalized sources and populate the wiki."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llm_wiki.llm import LLMClient, LLMError
from llm_wiki.schemas.models import SourceAnalysis, WikiPageResult
from llm_wiki.vault import Vault, SourceMeta, slugify, utcnow, utcdate

console = Console()

MAX_CONTENT_CHARS = 60_000  # truncate very long documents


def run(
    source_id: Optional[str],
    vault: Vault,
    llm: LLMClient,
    all_sources: bool,
    latest: bool,
    dry_run: bool,
) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized. Run `llm-wiki init` first.")
        raise SystemExit(1)

    if all_sources:
        metas = vault.list_sources()
        if not metas:
            console.print("[yellow]No sources found.[/yellow]")
            return
        for meta in metas:
            _ingest_one(meta, vault, llm, dry_run)

    elif latest:
        metas = vault.list_sources()
        if not metas:
            console.print("[yellow]No sources found.[/yellow]")
            return
        # Most recently added (last in sorted list by added_at)
        metas_sorted = sorted(metas, key=lambda m: m.added_at, reverse=True)
        _ingest_one(metas_sorted[0], vault, llm, dry_run)

    elif source_id:
        try:
            meta = vault.load_meta(source_id)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
        _ingest_one(meta, vault, llm, dry_run)

    else:
        console.print("[red]Error:[/red] Provide a source-id, --latest, or --all")
        raise SystemExit(1)


def _ingest_one(meta: SourceMeta, vault: Vault, llm: LLMClient, dry_run: bool) -> None:
    source_id = meta.source_id
    norm_path = vault.normalized_path(source_id)

    if not norm_path.exists():
        console.print(
            f"[yellow]Skipping {source_id}[/yellow]: not normalized. "
            f"Run `llm-wiki normalize {source_id}` first."
        )
        return

    content = norm_path.read_text(encoding="utf-8")
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n\n[Content truncated for processing…]\n"

    schema_ctx = vault.load_schema()

    console.rule(f"[bold]Ingesting: {source_id}[/bold]")

    # 1. Analyze source
    with _spinner(f"Analyzing {source_id}…"):
        try:
            analysis = _analyze_source(source_id, content, schema_ctx, llm)
        except LLMError as e:
            console.print(f"[red]LLM error during analysis:[/red] {e}")
            return

    console.print(f"  [dim]Title:[/dim] {analysis.title}")
    console.print(f"  [dim]Entities:[/dim] {len(analysis.entities)}, "
                  f"[dim]Concepts:[/dim] {len(analysis.concepts)}, "
                  f"[dim]Topics:[/dim] {len(analysis.topics)}")

    # 2. Generate source wiki page
    with _spinner("Generating source page…"):
        try:
            source_page = _generate_source_page(source_id, analysis, schema_ctx, llm, extension=meta.extension)
        except LLMError as e:
            console.print(f"[red]LLM error generating source page:[/red] {e}")
            return

    # 3. Generate/update entity, concept, topic pages
    entity_pages: list[tuple[str, str, str, str]] = []  # (slug, name, type, content)
    for entity in analysis.entities:
        slug = slugify(entity.name)
        existing_path = vault.entity_page_path(slug)
        with _spinner(f"Entity: {entity.name}…"):
            try:
                if existing_path.exists() and not dry_run:
                    page_content = _update_page(
                        existing_path.read_text(encoding="utf-8"),
                        f"entity '{entity.name}'",
                        json.dumps(entity.model_dump()),
                        source_id,
                        schema_ctx,
                        llm,
                    )
                else:
                    page_content = _generate_entity_page(entity, source_id, schema_ctx, llm)
                entity_pages.append((slug, entity.name, entity.type, page_content))
            except LLMError as e:
                console.print(f"[yellow]  Skipping entity {entity.name}: {e}[/yellow]")

    concept_pages: list[tuple[str, str, str]] = []  # (slug, name, content)
    for concept in analysis.concepts:
        slug = slugify(concept.name)
        existing_path = vault.concept_page_path(slug)
        with _spinner(f"Concept: {concept.name}…"):
            try:
                if existing_path.exists() and not dry_run:
                    page_content = _update_page(
                        existing_path.read_text(encoding="utf-8"),
                        f"concept '{concept.name}'",
                        json.dumps(concept.model_dump()),
                        source_id,
                        schema_ctx,
                        llm,
                    )
                else:
                    page_content = _generate_concept_page(concept, source_id, schema_ctx, llm)
                concept_pages.append((slug, concept.name, page_content))
            except LLMError as e:
                console.print(f"[yellow]  Skipping concept {concept.name}: {e}[/yellow]")

    topic_pages: list[tuple[str, str, str]] = []  # (slug, name, content)
    for topic in analysis.topics:
        slug = slugify(topic.name)
        existing_path = vault.topic_page_path(slug)
        with _spinner(f"Topic: {topic.name}…"):
            try:
                if existing_path.exists() and not dry_run:
                    page_content = _update_page(
                        existing_path.read_text(encoding="utf-8"),
                        f"topic '{topic.name}'",
                        json.dumps(topic.model_dump()),
                        source_id,
                        schema_ctx,
                        llm,
                    )
                else:
                    page_content = _generate_topic_page(topic, source_id, schema_ctx, llm)
                topic_pages.append((slug, topic.name, page_content))
            except LLMError as e:
                console.print(f"[yellow]  Skipping topic {topic.name}: {e}[/yellow]")

    if dry_run:
        console.print("\n[dim]DRY RUN — no files written.[/dim]")
        console.print(f"  Would create/update: {1 + len(entity_pages) + len(concept_pages) + len(topic_pages)} pages")
        return

    # 4. Write all pages
    vault.source_page_path(source_id).write_text(source_page, encoding="utf-8")
    console.print(f"  [green]✓[/green] Source page: [cyan]wiki/sources/{source_id}.md[/cyan]")

    for slug, name, _etype, content in entity_pages:
        vault.entity_page_path(slug).write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] Entity: [cyan]wiki/entities/{slug}.md[/cyan]")

    for slug, name, content in concept_pages:
        vault.concept_page_path(slug).write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] Concept: [cyan]wiki/concepts/{slug}.md[/cyan]")

    for slug, name, content in topic_pages:
        vault.topic_page_path(slug).write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] Topic: [cyan]wiki/topics/{slug}.md[/cyan]")

    # 5. Update index
    vault.update_index(
        source_id=source_id,
        title=analysis.title,
        entity_slugs=[(slug, name, etype) for slug, name, etype, _ in entity_pages],
        concept_slugs=[(slug, name) for slug, name, _ in concept_pages],
        topic_slugs=[(slug, name) for slug, name, _ in topic_pages],
        tags=analysis.tags,
    )
    console.print(f"  [green]✓[/green] index.md updated")

    # 6. Update metadata
    meta.update(ingested_at=utcnow(), title=analysis.title)
    vault.save_meta(meta)

    # 7. Append log
    log_lines = [
        f"- Ingested source: **{source_id}** — {analysis.title}",
        f"- Source page: `wiki/sources/{source_id}.md`",
    ]
    for slug, name, _, _ in entity_pages:
        log_lines.append(f"- Entity: `wiki/entities/{slug}.md`")
    for slug, name, _ in concept_pages:
        log_lines.append(f"- Concept: `wiki/concepts/{slug}.md`")
    for slug, name, _ in topic_pages:
        log_lines.append(f"- Topic: `wiki/topics/{slug}.md`")

    vault.append_log("\n".join(log_lines))
    console.print(f"  [green]✓[/green] log.md updated")
    console.print(f"\n[green]Done.[/green] Source [bold]{source_id}[/bold] ingested.")


# ------------------------------------------------------------------ #
#  LLM prompt functions
# ------------------------------------------------------------------ #

_SYSTEM_WIKI_AGENT = """\
You are a wiki maintainer agent. You maintain a structured, accurate, and well-linked knowledge base.
Follow the wiki schema and page conventions provided in context.
Only make claims supported by the source document.
Always produce valid JSON when asked.
"""


def _analyze_source(source_id: str, content: str, schema_ctx: str, llm: LLMClient) -> SourceAnalysis:
    system = f"{_SYSTEM_WIKI_AGENT}\n\n{schema_ctx}"
    user = f"""\
Analyze the following document (source_id: `{source_id}`) and extract structured information.

Return a JSON object matching this schema exactly:
{{
  "title": "string",
  "summary": "string (2-3 sentences)",
  "key_points": ["string", ...],
  "entities": [
    {{"name": "string", "type": "person|organization|product|place|event|other",
      "description": "string", "mentions": ["string", ...]}}
  ],
  "concepts": [
    {{"name": "string", "description": "string", "related_concepts": ["string", ...]}}
  ],
  "topics": [
    {{"name": "string", "description": "string"}}
  ],
  "tags": ["string", ...],
  "date_published": "YYYY-MM-DD or null",
  "authors": ["string", ...]
}}

Document content:
---
{content}
---"""

    data = llm.chat_json(system, user)
    return SourceAnalysis.model_validate(data)


def _generate_source_page(
    source_id: str, analysis: SourceAnalysis, schema_ctx: str, llm: LLMClient, extension: str = ""
) -> str:
    system = f"{_SYSTEM_WIKI_AGENT}\n\n{schema_ctx}"
    entity_links = "\n".join(
        f"- [{e.name}](../entities/{slugify(e.name)}.md) — {e.description}"
        for e in analysis.entities
    )
    concept_links = "\n".join(
        f"- [{c.name}](../concepts/{slugify(c.name)}.md) — {c.description}"
        for c in analysis.concepts
    )
    topic_links = "\n".join(
        f"- [{t.name}](../topics/{slugify(t.name)}.md)"
        for t in analysis.topics
    )
    key_points = "\n".join(f"- {p}" for p in analysis.key_points)
    tags_str = ", ".join(analysis.tags)
    authors_str = ", ".join(analysis.authors) if analysis.authors else "Unknown"
    date_str = analysis.date_published or utcdate()

    user = f"""\
Generate a wiki source page for source_id `{source_id}`.

Analysis:
{json.dumps(analysis.model_dump(), indent=2)}

Return a JSON object:
{{"content": "<full markdown page content>"}}

The page MUST include:
1. YAML frontmatter:
   ```yaml
   ---
   source_id: {source_id}
   title: "{analysis.title}"
   type: source
   added: {utcdate()}
   date_published: {date_str}
   authors: [{authors_str}]
   tags: [{tags_str}]
   ---
   ```
2. A bold one-sentence description paragraph
3. ## Summary — from analysis.summary
4. ## Key Points — bullet list:
{key_points}
5. ## Entities — pre-built links:
{entity_links or "(none identified)"}
6. ## Concepts — pre-built links:
{concept_links or "(none identified)"}
7. ## Topics — pre-built links:
{topic_links or "(none identified)"}
8. ## Source
   - Raw: `../../raw/{source_id}{extension}`
   - Normalized: `../../normalized/{source_id}.md`
"""
    data = llm.chat_json(system, user)
    return WikiPageResult.model_validate(data).content


def _generate_entity_page(entity, source_id: str, schema_ctx: str, llm: LLMClient) -> str:
    system = f"{_SYSTEM_WIKI_AGENT}\n\n{schema_ctx}"
    slug = slugify(entity.name)
    mentions_str = "\n".join(f'  - "{m}"' for m in entity.mentions)

    user = f"""\
Generate a wiki entity page for the entity described below.

Entity:
{json.dumps(entity.model_dump(), indent=2)}

Source reference: `{source_id}` — `../sources/{source_id}.md`

Return a JSON object:
{{"content": "<full markdown page content>"}}

The page MUST include:
1. YAML frontmatter:
   ```yaml
   ---
   name: "{entity.name}"
   type: entity
   entity_type: {entity.type}
   slug: {slug}
   ---
   ```
2. A bold one-sentence description
3. ## Description — expanded from entity.description
4. ## Sources — citing [{source_id}](../sources/{source_id}.md) with relevant quote(s)
5. ## Related Entities — if any (use relative links `./slug.md`)
6. ## Related Concepts — if any (use relative links `../concepts/slug.md`)
"""
    data = llm.chat_json(system, user)
    return WikiPageResult.model_validate(data).content


def _generate_concept_page(concept, source_id: str, schema_ctx: str, llm: LLMClient) -> str:
    system = f"{_SYSTEM_WIKI_AGENT}\n\n{schema_ctx}"
    slug = slugify(concept.name)
    related = "\n".join(f"- [{r}]({slugify(r)}.md)" for r in concept.related_concepts)

    user = f"""\
Generate a wiki concept page for the concept described below.

Concept:
{json.dumps(concept.model_dump(), indent=2)}

Source reference: `{source_id}` — `../sources/{source_id}.md`

Return a JSON object:
{{"content": "<full markdown page content>"}}

The page MUST include:
1. YAML frontmatter:
   ```yaml
   ---
   name: "{concept.name}"
   type: concept
   slug: {slug}
   ---
   ```
2. A bold one-sentence definition
3. ## Description — educational explanation
4. ## Sources — citing [{source_id}](../sources/{source_id}.md)
5. ## Related Concepts:
{related or "(none)"}
"""
    data = llm.chat_json(system, user)
    return WikiPageResult.model_validate(data).content


def _generate_topic_page(topic, source_id: str, schema_ctx: str, llm: LLMClient) -> str:
    system = f"{_SYSTEM_WIKI_AGENT}\n\n{schema_ctx}"
    slug = slugify(topic.name)

    user = f"""\
Generate a wiki topic page for the topic described below.

Topic:
{json.dumps(topic.model_dump(), indent=2)}

Source reference: `{source_id}` — `../sources/{source_id}.md`

Return a JSON object:
{{"content": "<full markdown page content>"}}

The page MUST include:
1. YAML frontmatter:
   ```yaml
   ---
   name: "{topic.name}"
   type: topic
   slug: {slug}
   ---
   ```
2. A bold one-sentence description
3. ## Overview
4. ## Sources — citing [{source_id}](../sources/{source_id}.md) and how it relates to this topic
5. ## Related Pages
"""
    data = llm.chat_json(system, user)
    return WikiPageResult.model_validate(data).content


def _update_page(
    existing_content: str,
    page_label: str,
    new_info_json: str,
    source_id: str,
    schema_ctx: str,
    llm: LLMClient,
) -> str:
    system = f"{_SYSTEM_WIKI_AGENT}\n\n{schema_ctx}"
    user = f"""\
Update the existing wiki page for {page_label} with new information from source `{source_id}`.

Existing page:
---
{existing_content}
---

New information (from `{source_id}`):
{new_info_json}

Return a JSON object:
{{"content": "<full updated markdown page content>"}}

Rules:
- Preserve all existing content and structure
- Add new information without duplicating what already exists
- Add a citation to [{source_id}](../sources/{source_id}.md) in the ## Sources section
- Do not remove any existing information or citations
"""
    data = llm.chat_json(system, user)
    return WikiPageResult.model_validate(data).content


@contextmanager
def _spinner(description: str):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description, total=None)
        yield progress
