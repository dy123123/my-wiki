"""llm-wiki lint — detect structural issues in the wiki."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_wiki.schemas.models import LintIssue, LintReport
from llm_wiki.vault import Vault

console = Console()

# Markdown link pattern: [text](path)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Frontmatter block
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def run(vault: Vault, fix: bool, dry_run: bool) -> int:
    """Run all lint checks. Returns exit code (0=ok, 1=errors found)."""
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized.")
        raise SystemExit(1)

    report = LintReport()

    _check_orphan_pages(vault, report)
    _check_dead_links(vault, report)
    _check_duplicate_slugs(vault, report)
    _check_missing_sections(vault, report)
    _check_empty_pages(vault, report)

    _print_report(report, vault)

    if fix and report.issues:
        _apply_fixes(report, vault, dry_run)

    return 1 if report.errors else 0


def _check_orphan_pages(vault: Vault, report: LintReport) -> None:
    """Find wiki pages not referenced in index.md."""
    index_content = vault.index.read_text(encoding="utf-8") if vault.index.exists() else ""

    for subdir in [vault.wiki_sources, vault.wiki_entities, vault.wiki_concepts,
                   vault.wiki_topics, vault.wiki_analyses, vault.wiki_reports]:
        for page in subdir.glob("*.md"):
            rel = str(page.relative_to(vault.wiki))
            if rel not in index_content:
                report.issues.append(LintIssue(
                    severity="warning",
                    category="orphan",
                    file=rel,
                    message="Page not referenced in index.md",
                    fix_hint="Add this page to index.md",
                ))


def _check_dead_links(vault: Vault, report: LintReport) -> None:
    """Find markdown links that point to non-existent files."""
    for page_path in vault.wiki.rglob("*.md"):
        try:
            content = page_path.read_text(encoding="utf-8")
        except Exception:
            continue

        for _, link_target in _LINK_RE.findall(content):
            # Skip external URLs and anchors
            if link_target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            # Strip any anchor fragment
            target = link_target.split("#")[0]
            if not target:
                continue

            resolved = (page_path.parent / target).resolve()
            if not resolved.exists():
                rel_page = str(page_path.relative_to(vault.wiki))
                report.issues.append(LintIssue(
                    severity="error",
                    category="dead_link",
                    file=rel_page,
                    message=f"Dead link: `{link_target}`",
                    fix_hint="Fix the link path or create the missing page",
                ))


def _check_duplicate_slugs(vault: Vault, report: LintReport) -> None:
    """Find entity/concept/topic pages with suspiciously similar names."""
    for subdir, label in [
        (vault.wiki_entities, "entity"),
        (vault.wiki_concepts, "concept"),
        (vault.wiki_topics, "topic"),
    ]:
        stems = [p.stem for p in subdir.glob("*.md")]
        seen: set[str] = set()
        for stem in stems:
            # Check for near-duplicate by removing common suffixes and plurals
            normalized = stem.rstrip("s").rstrip("-")
            if normalized in seen:
                report.issues.append(LintIssue(
                    severity="warning",
                    category="duplicate",
                    file=f"{subdir.name}/{stem}.md",
                    message=f"Potential duplicate {label}: '{stem}' may overlap with existing entries",
                    fix_hint="Merge pages if they describe the same thing",
                ))
            seen.add(normalized)


def _check_missing_sections(vault: Vault, report: LintReport) -> None:
    """Find wiki pages missing required sections."""
    required_by_type = {
        "source": ["## Summary", "## Sources"],
        "entity": ["## Description", "## Sources"],
        "concept": ["## Description", "## Sources"],
        "topic": ["## Overview", "## Sources"],
    }

    for page_path in vault.wiki.rglob("*.md"):
        if page_path.name in ("index.md", "log.md", "overview.md"):
            continue
        try:
            content = page_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Detect page type from frontmatter
        fm_match = _FRONTMATTER_RE.match(content)
        page_type = None
        if fm_match:
            for line in fm_match.group(1).split("\n"):
                if line.startswith("type:"):
                    page_type = line.split(":", 1)[1].strip().strip('"')
                    break

        if page_type not in required_by_type:
            continue

        rel = str(page_path.relative_to(vault.wiki))
        for required_section in required_by_type[page_type]:
            if required_section not in content:
                report.issues.append(LintIssue(
                    severity="warning",
                    category="missing_section",
                    file=rel,
                    message=f"Missing section: `{required_section}`",
                    fix_hint=f"Add a `{required_section}` section to this page",
                ))


def _check_empty_pages(vault: Vault, report: LintReport) -> None:
    """Find pages with very little content."""
    MIN_CONTENT_LEN = 100

    for page_path in vault.wiki.rglob("*.md"):
        if page_path.name in ("index.md", "log.md", "overview.md"):
            continue
        try:
            content = page_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Strip frontmatter
        content_stripped = _FRONTMATTER_RE.sub("", content).strip()
        if len(content_stripped) < MIN_CONTENT_LEN:
            rel = str(page_path.relative_to(vault.wiki))
            report.issues.append(LintIssue(
                severity="info",
                category="empty_page",
                file=rel,
                message=f"Very short page ({len(content_stripped)} chars)",
                fix_hint="Expand this page or merge it with a related page",
            ))


def _print_report(report: LintReport, vault: Vault) -> None:
    if not report.issues:
        console.print("[green]✓ No lint issues found.[/green]")
        return

    table = Table(
        title=f"Lint Report — {report.summary()}",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("Sev", no_wrap=True, width=7)
    table.add_column("Category", no_wrap=True, width=16)
    table.add_column("File")
    table.add_column("Message")

    sev_colors = {"error": "red", "warning": "yellow", "info": "dim"}

    for issue in sorted(report.issues, key=lambda i: (i.severity, i.file)):
        color = sev_colors.get(issue.severity, "white")
        table.add_row(
            f"[{color}]{issue.severity}[/{color}]",
            issue.category,
            issue.file,
            issue.message,
        )

    console.print(table)


def _apply_fixes(report: LintReport, vault: Vault, dry_run: bool) -> None:
    """Apply automatic fixes for simple issues."""
    fixable = [i for i in report.issues if i.category == "orphan"]
    if not fixable:
        console.print("[dim]No auto-fixable issues.[/dim]")
        return

    if dry_run:
        console.print(f"[dim]DRY RUN — would fix {len(fixable)} orphan(s) by adding to index.md[/dim]")
        return

    from llm_wiki.vault import _upsert_index_entry, utcdate

    content = vault.index.read_text(encoding="utf-8") if vault.index.exists() else "# Wiki Index\n\n"
    fixed = 0
    for issue in fixable:
        # Determine section from path prefix
        parts = issue.file.split("/")
        if len(parts) < 2:
            continue
        subdir = parts[0]
        section_map = {
            "sources": "## Sources",
            "entities": "## Entities",
            "concepts": "## Concepts",
            "topics": "## Topics",
            "analyses": "## Analyses",
            "reports": "## Reports",
        }
        section = section_map.get(subdir)
        if not section:
            continue

        stem = Path(issue.file).stem
        content = _upsert_index_entry(
            content,
            section,
            issue.file,
            f"[{stem}]({issue.file}) — (orphan; added by lint --fix on {utcdate()})",
        )
        fixed += 1

    vault.index.write_text(content, encoding="utf-8")
    console.print(f"[green]✓[/green] Fixed {fixed} orphan(s) by adding to index.md")
