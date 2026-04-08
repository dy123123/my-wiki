"""llm-wiki normalize — convert raw sources to markdown."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llm_wiki.vault import MARKITDOWN_EXTENSIONS, Vault, SourceMeta, utcnow

console = Console()


def run(source_id: str | None, vault: Vault, all_sources: bool, dry_run: bool) -> None:
    if not vault.exists():
        console.print("[red]Error:[/red] Vault not initialized. Run `llm-wiki init` first.")
        raise SystemExit(1)

    if all_sources:
        metas = vault.list_sources()
        if not metas:
            console.print("[yellow]No sources found in vault.[/yellow]")
            return
        for meta in metas:
            _normalize_one(meta, vault, dry_run)
    elif source_id:
        try:
            meta = vault.load_meta(source_id)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
        _normalize_one(meta, vault, dry_run)
    else:
        console.print("[red]Error:[/red] Provide a source-id or use --all")
        raise SystemExit(1)


def _normalize_one(meta: SourceMeta, vault: Vault, dry_run: bool) -> None:
    raw_path = vault.raw_path(meta)
    out_path = vault.normalized_path(meta.source_id)
    ext = meta.extension.lower()

    if not raw_path.exists():
        console.print(f"[red]Error:[/red] Raw file missing: {raw_path}")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(f"Normalizing {meta.source_id}…", total=None)
        content = _convert(raw_path, ext)

    if dry_run:
        preview = content[:300].replace("\n", "↵") + ("…" if len(content) > 300 else "")
        console.print(f"[dim]DRY RUN — would write {out_path}[/dim]")
        console.print(f"[dim]Preview: {preview}[/dim]")
        return

    out_path.write_text(content, encoding="utf-8")
    meta.update(normalized_at=utcnow())
    vault.save_meta(meta)

    size_kb = len(content.encode()) / 1024
    console.print(
        f"[green]✓[/green] {meta.source_id} → "
        f"[cyan]{out_path.relative_to(vault.path.parent)}[/cyan] "
        f"({size_kb:.1f} KB)"
    )


def _convert(raw_path: Path, ext: str) -> str:
    """Convert a raw file to markdown text."""
    if ext in MARKITDOWN_EXTENSIONS:
        try:
            from markitdown import MarkItDown  # type: ignore[import]

            md = MarkItDown()
            result = md.convert(str(raw_path))
            text = result.text_content
            if text and text.strip():
                # If PDF text looks garbled (font glyph mapping failure), try fallback
                if ext == ".pdf" and _is_garbled(text):
                    fallback = _pdf_fallback(raw_path)
                    if fallback and not _is_garbled(fallback):
                        console.print(
                            "[yellow]markitdown produced garbled text, used pypdf fallback.[/yellow]"
                        )
                        return fallback
                return text
            # Fall through if empty result
        except ImportError:
            console.print("[yellow]markitdown not installed, falling back to plain text.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]markitdown failed ({e}), falling back to plain text.[/yellow]")

    # For PDFs without markitdown, try pypdf directly
    if ext == ".pdf":
        fallback = _pdf_fallback(raw_path)
        if fallback:
            return fallback

    # Plain text fallback
    if ext in {".md", ".txt", ".rst", ".text"}:
        return raw_path.read_text(encoding="utf-8", errors="replace")

    # Binary files — best effort
    try:
        return raw_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return f"[Binary file — could not extract text: {raw_path.name}]\n"


def _is_garbled(text: str) -> bool:
    """Return True if text looks like a font glyph mapping failure."""
    import re
    cid_count = len(re.findall(r"\(cid:\d+\)", text))
    total_words = max(len(text.split()), 1)
    # Garbled if more than 10% of "words" are (cid:xxx) tokens
    return cid_count / total_words > 0.1


def _pdf_fallback(raw_path: Path) -> str | None:
    """Try extracting PDF text with pypdf as fallback."""
    try:
        import pypdf  # type: ignore[import]

        reader = pypdf.PdfReader(str(raw_path))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        return "\n\n".join(pages) if pages else None
    except ImportError:
        return None
    except Exception:
        return None
