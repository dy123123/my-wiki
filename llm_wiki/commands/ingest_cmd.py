"""llm-wiki ingest — analyze normalized sources and populate the wiki."""

from __future__ import annotations

import json
import re
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llm_wiki.llm import LLMClient, LLMError
from llm_wiki.schemas.models import SourceAnalysis, EntityRef, ConceptRef, TopicRef
from llm_wiki.vault import Vault, SourceMeta, slugify, utcnow, utcdate

console = Console()

MAX_CONTENT_CHARS = 50_000   # chars sent to LLM per call
MIN_SUMMARY_LEN   = 50       # if shorter, analysis is likely incomplete


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
        _ingest_one(sorted(metas, key=lambda m: m.added_at, reverse=True)[0], vault, llm, dry_run)
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
        content = content[:MAX_CONTENT_CHARS] + "\n\n[Content truncated…]\n"

    console.rule(f"[bold]Ingesting: {source_id}[/bold]")

    # ------------------------------------------------------------------ #
    # Step 1 — Analyze source
    # ------------------------------------------------------------------ #
    with _spinner(f"Analyzing {source_id}…"):
        try:
            analysis = _analyze_source(source_id, content, llm)
        except LLMError as e:
            console.print(f"[red]LLM error during analysis:[/red] {e}")
            return

    console.print(
        f"  [dim]Title:[/dim] {analysis.title}\n"
        f"  [dim]Entities:[/dim] {len(analysis.entities)}  "
        f"[dim]Concepts:[/dim] {len(analysis.concepts)}  "
        f"[dim]Topics:[/dim] {len(analysis.topics)}"
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Generate source wiki page (plain markdown, no JSON wrapper)
    # ------------------------------------------------------------------ #
    with _spinner("Writing source page…"):
        try:
            source_page = _build_source_page(source_id, analysis, meta, llm)
        except LLMError as e:
            console.print(f"[red]LLM error writing source page:[/red] {e}")
            return

    # ------------------------------------------------------------------ #
    # Step 3 — Generate entity / concept / topic pages
    # ------------------------------------------------------------------ #
    entity_pages:  list[tuple[str, str, str, str]] = []   # (slug, name, etype, content)
    concept_pages: list[tuple[str, str, str]] = []         # (slug, name, content)
    topic_pages:   list[tuple[str, str, str]] = []         # (slug, name, content)

    for entity in analysis.entities:
        slug = slugify(entity.name)
        existing = vault.entity_page_path(slug)
        with _spinner(f"Entity: {entity.name}…"):
            try:
                if existing.exists() and not dry_run:
                    pg = _update_entity_page(entity, source_id, existing.read_text(encoding="utf-8"), llm)
                else:
                    pg = _build_entity_page(entity, source_id, llm)
                entity_pages.append((slug, entity.name, entity.type, pg))
            except LLMError as e:
                console.print(f"  [yellow]Skipping entity {entity.name}:[/yellow] {e}")

    for concept in analysis.concepts:
        slug = slugify(concept.name)
        existing = vault.concept_page_path(slug)
        with _spinner(f"Concept: {concept.name}…"):
            try:
                if existing.exists() and not dry_run:
                    pg = _update_concept_page(concept, source_id, existing.read_text(encoding="utf-8"), llm)
                else:
                    pg = _build_concept_page(concept, source_id, llm)
                concept_pages.append((slug, concept.name, pg))
            except LLMError as e:
                console.print(f"  [yellow]Skipping concept {concept.name}:[/yellow] {e}")

    for topic in analysis.topics:
        slug = slugify(topic.name)
        existing = vault.topic_page_path(slug)
        with _spinner(f"Topic: {topic.name}…"):
            try:
                if existing.exists() and not dry_run:
                    pg = _update_topic_page(topic, source_id, existing.read_text(encoding="utf-8"), llm)
                else:
                    pg = _build_topic_page(topic, source_id, llm)
                topic_pages.append((slug, topic.name, pg))
            except LLMError as e:
                console.print(f"  [yellow]Skipping topic {topic.name}:[/yellow] {e}")

    # ------------------------------------------------------------------ #
    # Step 4 — Write files (unless dry_run)
    # ------------------------------------------------------------------ #
    if dry_run:
        console.print(f"\n[dim]DRY RUN — would write {1 + len(entity_pages) + len(concept_pages) + len(topic_pages)} pages.[/dim]")
        return

    vault.source_page_path(source_id).write_text(source_page, encoding="utf-8")
    console.print(f"  [green]✓[/green] Source page: [cyan]wiki/sources/{source_id}.md[/cyan]")

    for slug, name, _, content in entity_pages:
        vault.entity_page_path(slug).write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] Entity:  [cyan]wiki/entities/{slug}.md[/cyan]")

    for slug, name, content in concept_pages:
        vault.concept_page_path(slug).write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] Concept: [cyan]wiki/concepts/{slug}.md[/cyan]")

    for slug, name, content in topic_pages:
        vault.topic_page_path(slug).write_text(content, encoding="utf-8")
        console.print(f"  [green]✓[/green] Topic:   [cyan]wiki/topics/{slug}.md[/cyan]")

    # Index + log
    vault.update_index(
        source_id=source_id,
        title=analysis.title,
        entity_slugs=[(sl, nm, et) for sl, nm, et, _ in entity_pages],
        concept_slugs=[(sl, nm) for sl, nm, _ in concept_pages],
        topic_slugs=[(sl, nm) for sl, nm, _ in topic_pages],
        tags=analysis.tags,
    )
    console.print(f"  [green]✓[/green] index.md updated")

    meta.update(ingested_at=utcnow(), title=analysis.title)
    vault.save_meta(meta)

    log_lines = [f"- Ingested **{source_id}** — {analysis.title}"]
    for slug, name, _, _ in entity_pages:
        log_lines.append(f"  - entity: `{slug}`")
    for slug, name, _ in concept_pages:
        log_lines.append(f"  - concept: `{slug}`")
    for slug, name, _ in topic_pages:
        log_lines.append(f"  - topic: `{slug}`")
    vault.append_log("\n".join(log_lines))
    console.print(f"  [green]✓[/green] log.md updated")
    console.print(f"\n[green]Done.[/green] {source_id} ingested.")


# ======================================================================
# LLM helpers
# ======================================================================

_SYSTEM = """\
You are a precise wiki maintainer. Extract information accurately from documents.
Only include facts present in the document. Be concise but complete.
"""


def _analyze_source(source_id: str, content: str, llm: LLMClient) -> SourceAnalysis:
    """
    Step 1: extract structured metadata.
    Uses a simple flat JSON schema that works with smaller models.
    """
    prompt = f"""\
Read this document (id: {source_id}) and return a JSON object.

Return ONLY valid JSON, no explanation, no markdown fences.

JSON schema:
{{
  "title": "document title (string)",
  "summary": "2-4 sentence summary of the main contribution (string)",
  "key_points": ["bullet 1", "bullet 2", "..."],
  "entities": [
    {{"name": "full name", "type": "person|organization|product|place|event|other", "description": "one sentence"}}
  ],
  "concepts": [
    {{"name": "concept name", "description": "plain-language definition", "related_concepts": ["name", "..."]}}
  ],
  "topics": [
    {{"name": "topic name", "description": "one sentence on how this doc relates"}}
  ],
  "tags": ["lowercase", "tags"],
  "date_published": "YYYY-MM-DD or null",
  "authors": ["Author Name"]
}}

Document:
---
{content[:MAX_CONTENT_CHARS]}
---"""

    raw = llm.chat(_SYSTEM, prompt, temperature=0.1)
    data = _safe_parse_json(raw)
    _coerce_analysis(data)
    return SourceAnalysis.model_validate(data)


def _coerce_analysis(d: dict) -> None:
    """Fill in missing fields with sensible defaults so validation doesn't fail."""
    d.setdefault("title", "Untitled Document")
    d.setdefault("summary", "")
    d.setdefault("key_points", [])
    d.setdefault("entities", [])
    d.setdefault("concepts", [])
    d.setdefault("topics", [])
    d.setdefault("tags", [])
    d.setdefault("date_published", None)
    d.setdefault("authors", [])

    # Normalise entity/concept/topic entries
    for e in d["entities"]:
        e.setdefault("mentions", [])
        if e.get("type") not in ("person","organization","product","place","event","other"):
            e["type"] = "other"
    for c in d["concepts"]:
        c.setdefault("related_concepts", [])
    # topics can be plain strings → convert
    fixed_topics = []
    for t in d["topics"]:
        if isinstance(t, str):
            fixed_topics.append({"name": t, "description": ""})
        else:
            t.setdefault("description", "")
            fixed_topics.append(t)
    d["topics"] = fixed_topics


def _build_source_page(source_id: str, analysis: SourceAnalysis, meta: SourceMeta, llm: LLMClient) -> str:
    """
    Step 2a: Generate the source summary wiki page.
    Ask for plain markdown — no JSON wrapper, no code fences.
    Pre-build the skeleton so the model only writes prose.
    """
    entity_links = "\n".join(
        f"- [{e.name}](../entities/{slugify(e.name)}.md) — {e.description}"
        for e in analysis.entities
    ) or "_(none identified)_"

    concept_links = "\n".join(
        f"- [{c.name}](../concepts/{slugify(c.name)}.md)"
        for c in analysis.concepts
    ) or "_(none identified)_"

    topic_links = "\n".join(
        f"- [{t.name}](../topics/{slugify(t.name)}.md)"
        for t in analysis.topics
    ) or "_(none identified)_"

    key_points_md = "\n".join(f"- {p}" for p in analysis.key_points) or "- (see summary)"
    tags_yaml = ", ".join(analysis.tags) or ""
    authors_yaml = ", ".join(analysis.authors) if analysis.authors else "Unknown"

    # Build skeleton — ask LLM to write the prose sections only
    prompt = f"""\
Write the body content for this wiki source page.
Return ONLY the markdown content below. No JSON. No code fences. No explanations.

Fill in the sections marked with <WRITE> based on this analysis:
Title: {analysis.title}
Summary: {analysis.summary}
Key points: {json.dumps(analysis.key_points)}
Entities: {json.dumps([e.name for e in analysis.entities])}
Concepts: {json.dumps([c.name for c in analysis.concepts])}

Start your response with the YAML frontmatter line "---" exactly as shown:

---
source_id: {source_id}
title: "{analysis.title}"
type: source
added: {utcdate()}
date_published: {analysis.date_published or "null"}
authors: [{authors_yaml}]
tags: [{tags_yaml}]
---

# {analysis.title}

<WRITE: one bold sentence describing what this document is about>

## Summary

<WRITE: 3-5 sentences expanding on the summary, explaining the main contribution and significance>

## Key Points

{key_points_md}

## Methodology / Approach

<WRITE: 2-4 sentences describing the technical approach or methodology used in the document>

## Results / Findings

<WRITE: 2-4 sentences describing the key results, findings, or conclusions>

## Entities

{entity_links}

## Concepts

{concept_links}

## Topics

{topic_links}

## Source

- Raw: `../../raw/{source_id}{meta.extension}`
- Normalized: `../../normalized/{source_id}.md`
"""

    page = llm.chat(_SYSTEM, prompt, temperature=0.2, max_tokens=llm.max_tokens)
    page = _clean_page(page)

    # Fallback: if LLM returned garbage or very short content, build it ourselves
    if len(page.strip()) < 200:
        page = _fallback_source_page(source_id, analysis, meta)

    return page


def _build_entity_page(entity: EntityRef, source_id: str, llm: LLMClient) -> str:
    slug = slugify(entity.name)
    prompt = f"""\
Write a wiki entity page. Return ONLY the markdown. No JSON. No code fences.

Entity: {entity.name}
Type: {entity.type}
Description: {entity.description}
From source: {source_id}

---
name: "{entity.name}"
type: entity
entity_type: {entity.type}
slug: {slug}
---

# {entity.name}

**{entity.description}**

## Description

<WRITE: 2-4 sentences expanding on the entity's role, background, or significance based on the description>

## Sources

- [{source_id}](../sources/{source_id}.md){' — "' + entity.mentions[0] + '"' if entity.mentions else ''}
"""
    page = llm.chat(_SYSTEM, prompt, temperature=0.2)
    page = _clean_page(page)
    if len(page.strip()) < 100:
        page = _fallback_entity_page(entity, source_id)
    return page


def _build_concept_page(concept: ConceptRef, source_id: str, llm: LLMClient) -> str:
    slug = slugify(concept.name)
    related_md = "\n".join(
        f"- [{r}]({slugify(r)}.md)" for r in concept.related_concepts
    ) if concept.related_concepts else ""

    prompt = f"""\
Write a wiki concept page. Return ONLY the markdown. No JSON. No code fences.

Concept: {concept.name}
Definition: {concept.description}
From source: {source_id}

---
name: "{concept.name}"
type: concept
slug: {slug}
---

# {concept.name}

**{concept.description}**

## Description

<WRITE: 3-5 sentences explaining this concept clearly, its significance, how it works or why it matters>

## Sources

- [{source_id}](../sources/{source_id}.md) — context from this source
{("\\n## Related Concepts\\n\\n" + related_md) if related_md else ""}
"""
    page = llm.chat(_SYSTEM, prompt, temperature=0.2)
    page = _clean_page(page)
    if len(page.strip()) < 100:
        page = _fallback_concept_page(concept, source_id)
    return page


def _build_topic_page(topic: TopicRef, source_id: str, llm: LLMClient) -> str:
    slug = slugify(topic.name)
    prompt = f"""\
Write a wiki topic page. Return ONLY the markdown. No JSON. No code fences.

Topic: {topic.name}
Relevance: {topic.description}
From source: {source_id}

---
name: "{topic.name}"
type: topic
slug: {slug}
---

# {topic.name}

**<WRITE: one bold sentence describing this topic>**

## Overview

<WRITE: 3-5 sentences describing this topic area broadly>

## Sources

- [{source_id}](../sources/{source_id}.md) — {topic.description}
"""
    page = llm.chat(_SYSTEM, prompt, temperature=0.2)
    page = _clean_page(page)
    if len(page.strip()) < 100:
        page = _fallback_topic_page(topic, source_id)
    return page


def _update_entity_page(entity: EntityRef, source_id: str, existing: str, llm: LLMClient) -> str:
    prompt = f"""\
Update this existing wiki entity page with new information from source `{source_id}`.
Return ONLY the updated markdown. No JSON. No code fences.

New info: {entity.description}
{('Quotes: "' + entity.mentions[0] + '"') if entity.mentions else ''}

Existing page:
---
{existing}
---

Rules: keep all existing content, add the new source citation to ## Sources, add any new description details.
"""
    page = llm.chat(_SYSTEM, prompt, temperature=0.1)
    page = _clean_page(page)
    return page if len(page.strip()) > 100 else existing


def _update_concept_page(concept: ConceptRef, source_id: str, existing: str, llm: LLMClient) -> str:
    prompt = f"""\
Update this wiki concept page with new info from source `{source_id}`.
Return ONLY the updated markdown. No JSON. No code fences.

New description: {concept.description}
Existing page:
---
{existing}
---
Add new source citation to ## Sources. Keep all existing content.
"""
    page = llm.chat(_SYSTEM, prompt, temperature=0.1)
    page = _clean_page(page)
    return page if len(page.strip()) > 100 else existing


def _update_topic_page(topic: TopicRef, source_id: str, existing: str, llm: LLMClient) -> str:
    prompt = f"""\
Update this wiki topic page with new source `{source_id}`.
Return ONLY the updated markdown. No JSON. No code fences.
Add to ## Sources: [{source_id}](../sources/{source_id}.md) — {topic.description}
Existing page:
---
{existing}
---
"""
    page = llm.chat(_SYSTEM, prompt, temperature=0.1)
    page = _clean_page(page)
    return page if len(page.strip()) > 100 else existing + f"\n- [{source_id}](../sources/{source_id}.md) — {topic.description}\n"


# ======================================================================
# Fallback page builders (no LLM needed — used when LLM output is bad)
# ======================================================================

def _fallback_source_page(source_id: str, analysis: SourceAnalysis, meta: SourceMeta) -> str:
    key_points = "\n".join(f"- {p}" for p in analysis.key_points) or "- (see normalized source)"
    entity_links = "\n".join(
        f"- [{e.name}](../entities/{slugify(e.name)}.md) — {e.description}"
        for e in analysis.entities
    )
    concept_links = "\n".join(
        f"- [{c.name}](../concepts/{slugify(c.name)}.md)"
        for c in analysis.concepts
    )
    topic_links = "\n".join(
        f"- [{t.name}](../topics/{slugify(t.name)}.md)"
        for t in analysis.topics
    )
    return textwrap.dedent(f"""\
        ---
        source_id: {source_id}
        title: "{analysis.title}"
        type: source
        added: {utcdate()}
        tags: [{', '.join(analysis.tags)}]
        ---

        # {analysis.title}

        **{analysis.summary or 'Source document — see normalized content for details.'}**

        ## Summary

        {analysis.summary or '_Summary not available._'}

        ## Key Points

        {key_points}

        {'## Entities' + chr(10) + chr(10) + entity_links + chr(10) if entity_links else ''}
        {'## Concepts' + chr(10) + chr(10) + concept_links + chr(10) if concept_links else ''}
        {'## Topics' + chr(10) + chr(10) + topic_links + chr(10) if topic_links else ''}

        ## Source

        - Raw: `../../raw/{source_id}{meta.extension}`
        - Normalized: `../../normalized/{source_id}.md`
        """)


def _fallback_entity_page(entity: EntityRef, source_id: str) -> str:
    return textwrap.dedent(f"""\
        ---
        name: "{entity.name}"
        type: entity
        entity_type: {entity.type}
        slug: {slugify(entity.name)}
        ---

        # {entity.name}

        **{entity.description}**

        ## Description

        {entity.description}

        ## Sources

        - [{source_id}](../sources/{source_id}.md)
        """)


def _fallback_concept_page(concept: ConceptRef, source_id: str) -> str:
    related = "\n".join(f"- [{r}]({slugify(r)}.md)" for r in concept.related_concepts)
    return textwrap.dedent(f"""\
        ---
        name: "{concept.name}"
        type: concept
        slug: {slugify(concept.name)}
        ---

        # {concept.name}

        **{concept.description}**

        ## Description

        {concept.description}

        ## Sources

        - [{source_id}](../sources/{source_id}.md)
        {('## Related Concepts' + chr(10) + chr(10) + related) if related else ''}
        """)


def _fallback_topic_page(topic: TopicRef, source_id: str) -> str:
    return textwrap.dedent(f"""\
        ---
        name: "{topic.name}"
        type: topic
        slug: {slugify(topic.name)}
        ---

        # {topic.name}

        **{topic.description or topic.name}**

        ## Overview

        {topic.description or 'Topic area related to the ingested sources.'}

        ## Sources

        - [{source_id}](../sources/{source_id}.md) — {topic.description}
        """)


# ======================================================================
# Utilities
# ======================================================================

def _safe_parse_json(text: str) -> dict:
    """Parse JSON from LLM response, tolerating markdown fences and prose wrapping."""
    # Direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find the outermost JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    raise LLMError(f"Could not extract JSON from response (first 300 chars): {text[:300]}")


def _clean_page(text: str) -> str:
    """Strip leading/trailing whitespace and any markdown code fences wrapping the page."""
    text = text.strip()
    # Remove wrapping ```markdown ... ``` if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # Replace <WRITE: ...> placeholder that the model left unfilled
    text = re.sub(r"<WRITE:[^>]*>", "", text)
    return text.strip()


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
