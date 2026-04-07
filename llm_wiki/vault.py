"""Vault — local directory structure for the wiki."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# File extensions that markitdown can convert
MARKITDOWN_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".html", ".htm", ".xml",
    ".csv", ".tsv", ".jpg", ".jpeg", ".png",
    ".gif", ".bmp", ".mp3", ".wav", ".m4a",
    ".flac", ".ogg", ".mp4", ".avi", ".mov",
    ".mkv", ".webm", ".zip",
}

SCHEMA_FILES = {
    "AGENTS.md": _agents_md,
    "wiki_schema.md": _wiki_schema_md,
    "page_conventions.md": _page_conventions_md,
}


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def source_id_from_path(path: Path) -> str:
    """Generate a stable source ID from a file path."""
    stem = slugify(path.stem)[:40]
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"{stem}-{content_hash}"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcdate() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class SourceMeta:
    """Metadata for a raw source file."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def source_id(self) -> str:
        return self._data["source_id"]

    @property
    def original_name(self) -> str:
        return self._data["original_name"]

    @property
    def extension(self) -> str:
        return self._data["extension"]

    @property
    def title(self) -> Optional[str]:
        return self._data.get("title")

    @property
    def added_at(self) -> str:
        return self._data["added_at"]

    @property
    def normalized_at(self) -> Optional[str]:
        return self._data.get("normalized_at")

    @property
    def ingested_at(self) -> Optional[str]:
        return self._data.get("ingested_at")

    @property
    def tags(self) -> list[str]:
        return self._data.get("tags", [])

    def to_dict(self) -> dict:
        return dict(self._data)

    def update(self, **kwargs) -> None:
        self._data.update(kwargs)


class Vault:
    """Manages the local wiki vault directory structure."""

    def __init__(self, path: Path):
        self.path = path.resolve()

        # Top-level directories
        self.raw = self.path / "raw"
        self.normalized = self.path / "normalized"
        self.wiki = self.path / "wiki"
        self.schema = self.path / "schema"

        # Wiki subdirectories
        self.wiki_sources = self.wiki / "sources"
        self.wiki_entities = self.wiki / "entities"
        self.wiki_concepts = self.wiki / "concepts"
        self.wiki_topics = self.wiki / "topics"
        self.wiki_analyses = self.wiki / "analyses"
        self.wiki_reports = self.wiki / "reports"

        # Special wiki files
        self.index = self.wiki / "index.md"
        self.log = self.wiki / "log.md"
        self.overview = self.wiki / "overview.md"

    def exists(self) -> bool:
        return self.path.exists() and (self.raw.exists() or self.wiki.exists())

    def init(self, schema_source: Optional[Path] = None) -> None:
        """Create vault directory structure and seed schema docs."""
        dirs = [
            self.raw,
            self.normalized,
            self.schema,
            self.wiki_sources,
            self.wiki_entities,
            self.wiki_concepts,
            self.wiki_topics,
            self.wiki_analyses,
            self.wiki_reports,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        if not self.index.exists():
            self.index.write_text(_default_index(), encoding="utf-8")
        if not self.log.exists():
            self.log.write_text("# Wiki Log\n\n", encoding="utf-8")
        if not self.overview.exists():
            self.overview.write_text("# Wiki Overview\n\n_Not yet generated. Run `llm-wiki ingest --all` to populate._\n", encoding="utf-8")

        # Write schema docs
        schema_src = schema_source or (Path(__file__).parent.parent / "sample_vault" / "schema")
        for filename, content_fn in [
            ("AGENTS.md", _agents_md),
            ("wiki_schema.md", _wiki_schema_md),
            ("page_conventions.md", _page_conventions_md),
        ]:
            dest = self.schema / filename
            if not dest.exists():
                if schema_src.exists() and (schema_src / filename).exists():
                    shutil.copy2(schema_src / filename, dest)
                else:
                    dest.write_text(content_fn(), encoding="utf-8")

    # ------------------------------------------------------------------ #
    #  Raw source management
    # ------------------------------------------------------------------ #

    def add_source(self, src_path: Path, tags: list[str] | None = None) -> SourceMeta:
        """Copy a file into vault/raw/ and create metadata."""
        source_id = source_id_from_path(src_path)
        ext = src_path.suffix.lower()
        dest = self.raw / f"{source_id}{ext}"

        # Avoid overwriting identical content
        if dest.exists():
            existing_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
            new_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()
            if existing_hash == new_hash:
                return self.load_meta(source_id)

        shutil.copy2(src_path, dest)
        dest.chmod(0o444)  # make immutable

        meta = SourceMeta(
            {
                "source_id": source_id,
                "original_name": src_path.name,
                "original_path": str(src_path.resolve()),
                "extension": ext,
                "size_bytes": src_path.stat().st_size,
                "content_hash": hashlib.sha256(src_path.read_bytes()).hexdigest(),
                "added_at": utcnow(),
                "normalized_at": None,
                "ingested_at": None,
                "title": None,
                "tags": tags or [],
            }
        )
        self._save_meta_obj(source_id, meta)
        return meta

    def load_meta(self, source_id: str) -> SourceMeta:
        meta_path = self.raw / f"{source_id}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata for source '{source_id}'. Did you run `add`?")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return SourceMeta(data)

    def save_meta(self, meta: SourceMeta) -> None:
        self._save_meta_obj(meta.source_id, meta)

    def _save_meta_obj(self, source_id: str, meta: SourceMeta) -> None:
        meta_path = self.raw / f"{source_id}.meta.json"
        meta_path.write_text(
            json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_sources(self) -> list[SourceMeta]:
        metas = []
        for p in sorted(self.raw.glob("*.meta.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                metas.append(SourceMeta(data))
            except Exception:
                pass
        return metas

    def raw_path(self, meta: SourceMeta) -> Path:
        return self.raw / f"{meta.source_id}{meta.extension}"

    def normalized_path(self, source_id: str) -> Path:
        return self.normalized / f"{source_id}.md"

    def source_page_path(self, source_id: str) -> Path:
        return self.wiki_sources / f"{source_id}.md"

    def entity_page_path(self, slug: str) -> Path:
        return self.wiki_entities / f"{slug}.md"

    def concept_page_path(self, slug: str) -> Path:
        return self.wiki_concepts / f"{slug}.md"

    def topic_page_path(self, slug: str) -> Path:
        return self.wiki_topics / f"{slug}.md"

    # ------------------------------------------------------------------ #
    #  Index management
    # ------------------------------------------------------------------ #

    def update_index(
        self,
        source_id: str,
        title: str,
        entity_slugs: list[tuple[str, str, str]],  # (slug, name, type)
        concept_slugs: list[tuple[str, str]],       # (slug, name)
        topic_slugs: list[tuple[str, str]],         # (slug, name)
        tags: list[str],
    ) -> None:
        """Add/update entries in index.md."""
        content = self.index.read_text(encoding="utf-8") if self.index.exists() else _default_index()

        content = _upsert_index_entry(
            content,
            "## Sources",
            f"sources/{source_id}.md",
            f"[{source_id}](sources/{source_id}.md) — {title} | tags: {', '.join(tags) or 'none'} | ingested: {utcdate()}",
        )

        for slug, name, etype in entity_slugs:
            content = _upsert_index_entry(
                content,
                "## Entities",
                f"entities/{slug}.md",
                f"[{slug}](entities/{slug}.md) — {name} ({etype})",
            )

        for slug, name in concept_slugs:
            content = _upsert_index_entry(
                content,
                "## Concepts",
                f"concepts/{slug}.md",
                f"[{slug}](concepts/{slug}.md) — {name}",
            )

        for slug, name in topic_slugs:
            content = _upsert_index_entry(
                content,
                "## Topics",
                f"topics/{slug}.md",
                f"[{slug}](topics/{slug}.md) — {name}",
            )

        self.index.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------ #
    #  Log management
    # ------------------------------------------------------------------ #

    def append_log(self, message: str) -> None:
        """Append a timestamped entry to log.md."""
        existing = self.log.read_text(encoding="utf-8") if self.log.exists() else "# Wiki Log\n\n"
        ts = utcnow()
        entry = f"## {ts}\n{message.strip()}\n\n"
        self.log.write_text(existing + entry, encoding="utf-8")

    def load_schema(self) -> str:
        """Load all schema docs into a single string for LLM context."""
        parts = []
        for fname in ["AGENTS.md", "wiki_schema.md", "page_conventions.md"]:
            p = self.schema / fname
            if p.exists():
                parts.append(f"# {fname}\n\n{p.read_text(encoding='utf-8')}")
        return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------ #
#  Index helpers
# ------------------------------------------------------------------ #

def _default_index() -> str:
    return """\
# Wiki Index

## Sources

## Entities

## Concepts

## Topics
"""


def _upsert_index_entry(content: str, section_header: str, link_key: str, new_line: str) -> str:
    """Insert or replace a list entry in a markdown section."""
    # Ensure section exists
    if section_header not in content:
        content = content.rstrip() + f"\n\n{section_header}\n"

    lines = content.split("\n")
    in_section = False
    section_start = -1
    section_end = -1
    existing_idx = -1

    for i, line in enumerate(lines):
        if line.strip() == section_header:
            in_section = True
            section_start = i
            continue
        if in_section:
            if line.startswith("## ") and i != section_start:
                section_end = i
                break
            if link_key in line and line.startswith("- "):
                existing_idx = i

    if section_end == -1:
        section_end = len(lines)

    if existing_idx != -1:
        lines[existing_idx] = f"- {new_line}"
    else:
        # Insert before the next section (or at end)
        insert_at = section_end
        # Find last non-empty line in section
        for i in range(section_end - 1, section_start, -1):
            if lines[i].strip():
                insert_at = i + 1
                break
        else:
            insert_at = section_start + 1
        lines.insert(insert_at, f"- {new_line}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Default schema doc content
# ------------------------------------------------------------------ #

def _agents_md() -> str:
    return (Path(__file__).parent.parent / "sample_vault" / "schema" / "AGENTS.md").read_text(encoding="utf-8")


def _wiki_schema_md() -> str:
    return (Path(__file__).parent.parent / "sample_vault" / "schema" / "wiki_schema.md").read_text(encoding="utf-8")


def _page_conventions_md() -> str:
    return (Path(__file__).parent.parent / "sample_vault" / "schema" / "page_conventions.md").read_text(encoding="utf-8")
