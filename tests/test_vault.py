"""Tests for vault operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki.vault import Vault, slugify, source_id_from_path, _upsert_index_entry


# ------------------------------------------------------------------ #
#  slugify
# ------------------------------------------------------------------ #

def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert slugify("Attention Is All You Need!") == "attention-is-all-you-need"


def test_slugify_extra_dashes():
    assert slugify("  foo -- bar  ") == "foo-bar"


def test_slugify_unicode():
    result = slugify("Café au lait")
    assert "caf" in result


# ------------------------------------------------------------------ #
#  source_id_from_path
# ------------------------------------------------------------------ #

def test_source_id_stable(tmp_path: Path):
    p = tmp_path / "my doc.txt"
    p.write_text("same content", encoding="utf-8")
    id1 = source_id_from_path(p)
    id2 = source_id_from_path(p)
    assert id1 == id2


def test_source_id_changes_with_content(tmp_path: Path):
    p1 = tmp_path / "doc.txt"
    p2 = tmp_path / "doc.txt"
    p1.write_text("content A", encoding="utf-8")
    id1 = source_id_from_path(p1)
    p2.write_text("content B", encoding="utf-8")
    id2 = source_id_from_path(p2)
    assert id1 != id2


def test_source_id_format(tmp_path: Path):
    p = tmp_path / "My Great Document.txt"
    p.write_text("hello", encoding="utf-8")
    sid = source_id_from_path(p)
    assert "-" in sid
    parts = sid.rsplit("-", 1)
    assert len(parts[1]) == 8  # 8-char hash suffix


# ------------------------------------------------------------------ #
#  Vault structure
# ------------------------------------------------------------------ #

def test_vault_not_exists(tmp_path: Path):
    vault = Vault(tmp_path / "nonexistent")
    assert not vault.exists()


def test_vault_exists_after_init(tmp_vault: Vault):
    assert tmp_vault.exists()


def test_vault_directories_created(tmp_vault: Vault):
    assert tmp_vault.raw.is_dir()
    assert tmp_vault.normalized.is_dir()
    assert tmp_vault.wiki_sources.is_dir()
    assert tmp_vault.wiki_entities.is_dir()
    assert tmp_vault.wiki_concepts.is_dir()
    assert tmp_vault.wiki_topics.is_dir()
    assert tmp_vault.wiki_analyses.is_dir()
    assert tmp_vault.wiki_reports.is_dir()


def test_vault_special_files(tmp_vault: Vault):
    assert tmp_vault.index.exists()
    assert tmp_vault.log.exists()
    assert tmp_vault.overview.exists()


# ------------------------------------------------------------------ #
#  Source management
# ------------------------------------------------------------------ #

def test_add_source(tmp_vault: Vault, sample_txt: Path):
    meta = tmp_vault.add_source(sample_txt)
    assert meta.source_id
    assert meta.extension == ".txt"
    assert meta.original_name == "hello.txt"
    assert tmp_vault.raw_path(meta).exists()


def test_add_source_idempotent(tmp_vault: Vault, sample_txt: Path):
    meta1 = tmp_vault.add_source(sample_txt)
    meta2 = tmp_vault.add_source(sample_txt)
    assert meta1.source_id == meta2.source_id


def test_add_source_with_tags(tmp_vault: Vault, sample_txt: Path):
    meta = tmp_vault.add_source(sample_txt, tags=["ai", "test"])
    assert "ai" in meta.tags
    assert "test" in meta.tags


def test_load_meta_roundtrip(tmp_vault: Vault, sample_txt: Path):
    meta = tmp_vault.add_source(sample_txt)
    loaded = tmp_vault.load_meta(meta.source_id)
    assert loaded.source_id == meta.source_id
    assert loaded.extension == meta.extension


def test_load_meta_missing(tmp_vault: Vault):
    with pytest.raises(FileNotFoundError):
        tmp_vault.load_meta("nonexistent-id")


def test_list_sources(tmp_vault: Vault, tmp_path: Path):
    for name in ["a.txt", "b.txt", "c.txt"]:
        p = tmp_path / name
        p.write_text("content", encoding="utf-8")
        tmp_vault.add_source(p)
    metas = tmp_vault.list_sources()
    assert len(metas) == 3


def test_raw_file_is_immutable(tmp_vault: Vault, sample_txt: Path):
    meta = tmp_vault.add_source(sample_txt)
    raw = tmp_vault.raw_path(meta)
    # File should not be writable
    import stat
    mode = raw.stat().st_mode
    assert not (mode & stat.S_IWUSR)


# ------------------------------------------------------------------ #
#  Index management
# ------------------------------------------------------------------ #

def test_upsert_index_entry_insert():
    content = "# Wiki Index\n\n## Sources\n\n## Entities\n"
    result = _upsert_index_entry(content, "## Sources", "sources/foo.md", "[foo](sources/foo.md) — Foo")
    assert "- [foo](sources/foo.md) — Foo" in result


def test_upsert_index_entry_update():
    content = "# Wiki Index\n\n## Sources\n- [foo](sources/foo.md) — Old title\n\n## Entities\n"
    result = _upsert_index_entry(content, "## Sources", "sources/foo.md", "[foo](sources/foo.md) — New title")
    assert "New title" in result
    assert "Old title" not in result
    assert result.count("sources/foo.md") == 1


def test_upsert_index_entry_creates_section():
    content = "# Wiki Index\n\n## Sources\n"
    result = _upsert_index_entry(content, "## Entities", "entities/bar.md", "[bar](entities/bar.md)")
    assert "## Entities" in result
    assert "entities/bar.md" in result


def test_vault_update_index(tmp_vault: Vault):
    tmp_vault.update_index(
        source_id="test-abc123",
        title="Test Doc",
        entity_slugs=[("test-corp", "Test Corp", "organization")],
        concept_slugs=[("machine-learning", "Machine Learning")],
        topic_slugs=[("ai", "Artificial Intelligence")],
        tags=["ai", "test"],
    )
    idx = tmp_vault.index.read_text(encoding="utf-8")
    assert "test-abc123" in idx
    assert "test-corp" in idx
    assert "machine-learning" in idx
    assert "ai" in idx


# ------------------------------------------------------------------ #
#  Log management
# ------------------------------------------------------------------ #

def test_append_log(tmp_vault: Vault):
    tmp_vault.append_log("- Did something important")
    log = tmp_vault.log.read_text(encoding="utf-8")
    assert "Did something important" in log


def test_append_log_multiple(tmp_vault: Vault):
    tmp_vault.append_log("- Entry 1")
    tmp_vault.append_log("- Entry 2")
    log = tmp_vault.log.read_text(encoding="utf-8")
    assert "Entry 1" in log
    assert "Entry 2" in log
