"""Tests for CLI commands (unit-level, no LLM calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from llm_wiki.cli import app
from llm_wiki.vault import Vault

runner = CliRunner(mix_stderr=False)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_env(tmp_path: Path) -> dict:
    return {
        "LLM_WIKI_VAULT_PATH": str(tmp_path / "vault"),
        "LLM_WIKI_LLM_API_KEY": "test-key",
        "LLM_WIKI_LLM_BASE_URL": "http://localhost:11434/v1",
        "LLM_WIKI_LLM_MODEL": "test-model",
    }


# ------------------------------------------------------------------ #
#  init
# ------------------------------------------------------------------ #

def test_init_creates_vault(tmp_path: Path):
    result = runner.invoke(app, ["init", "--vault", str(tmp_path / "vault")])
    assert result.exit_code == 0
    assert (tmp_path / "vault" / "raw").is_dir()
    assert (tmp_path / "vault" / "wiki").is_dir()


def test_init_dry_run(tmp_path: Path):
    result = runner.invoke(app, ["init", "--vault", str(tmp_path / "vault"), "--dry-run"])
    assert result.exit_code == 0
    assert not (tmp_path / "vault").exists()
    assert "DRY RUN" in result.output


def test_init_already_exists(tmp_vault: Vault):
    result = runner.invoke(app, ["init", "--vault", str(tmp_vault.path)])
    assert result.exit_code == 0
    assert "already exists" in result.output.lower()


# ------------------------------------------------------------------ #
#  config
# ------------------------------------------------------------------ #

def test_config_show(tmp_path: Path):
    result = runner.invoke(app, ["config", "show"], env=_make_env(tmp_path))
    assert result.exit_code == 0
    assert "llm_model" in result.output


def test_config_validate_ok(tmp_path: Path):
    result = runner.invoke(app, ["config", "validate"], env=_make_env(tmp_path))
    assert result.exit_code == 0


def test_config_validate_no_key(tmp_path: Path):
    env = _make_env(tmp_path)
    env["LLM_WIKI_LLM_API_KEY"] = ""
    result = runner.invoke(app, ["config", "validate"], env=env)
    assert result.exit_code == 1


# ------------------------------------------------------------------ #
#  add
# ------------------------------------------------------------------ #

def test_add_file(tmp_vault: Vault, sample_txt: Path):
    result = runner.invoke(
        app,
        ["add", str(sample_txt)],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "Source added" in result.output


def test_add_file_not_found(tmp_vault: Vault):
    result = runner.invoke(
        app,
        ["add", "/nonexistent/file.txt"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 1


def test_add_dry_run(tmp_vault: Vault, sample_txt: Path):
    result = runner.invoke(
        app,
        ["add", str(sample_txt), "--dry-run"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    # File should NOT have been copied
    metas = tmp_vault.list_sources()
    assert len(metas) == 0


def test_add_with_tags(tmp_vault: Vault, sample_txt: Path):
    result = runner.invoke(
        app,
        ["add", str(sample_txt), "--tag", "ai", "--tag", "test"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    metas = tmp_vault.list_sources()
    assert "ai" in metas[0].tags


# ------------------------------------------------------------------ #
#  normalize
# ------------------------------------------------------------------ #

def test_normalize_no_args(tmp_vault: Vault):
    result = runner.invoke(
        app,
        ["normalize"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code != 0


def test_normalize_source(tmp_vault: Vault, sample_txt: Path):
    meta = tmp_vault.add_source(sample_txt)
    result = runner.invoke(
        app,
        ["normalize", meta.source_id],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert tmp_vault.normalized_path(meta.source_id).exists()


def test_normalize_all(tmp_vault: Vault, tmp_path: Path):
    for name in ["a.txt", "b.txt"]:
        p = tmp_path / name
        p.write_text("content", encoding="utf-8")
        tmp_vault.add_source(p)

    result = runner.invoke(
        app,
        ["normalize", "--all"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    normalized = list(tmp_vault.normalized.glob("*.md"))
    assert len(normalized) == 2


def test_normalize_dry_run(tmp_vault: Vault, sample_txt: Path):
    meta = tmp_vault.add_source(sample_txt)
    result = runner.invoke(
        app,
        ["normalize", meta.source_id, "--dry-run"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert not tmp_vault.normalized_path(meta.source_id).exists()


# ------------------------------------------------------------------ #
#  search
# ------------------------------------------------------------------ #

def test_search_no_results(tmp_vault: Vault):
    result = runner.invoke(
        app,
        ["search", "nonexistent term xyz"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "No results" in result.output


def test_search_finds_content(tmp_vault: Vault):
    page = tmp_vault.wiki_concepts / "transformers.md"
    page.write_text("# Transformers\n\nTransformers are powerful neural network architectures.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["search", "transformers neural"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "transformers" in result.output.lower()


# ------------------------------------------------------------------ #
#  status
# ------------------------------------------------------------------ #

def test_status_empty_vault(tmp_vault: Vault):
    result = runner.invoke(
        app,
        ["status"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "Sources" in result.output


def test_status_with_source(tmp_vault: Vault, sample_txt: Path):
    tmp_vault.add_source(sample_txt)
    result = runner.invoke(
        app,
        ["status"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0


def test_status_no_vault(tmp_path: Path):
    result = runner.invoke(
        app,
        ["status"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_path / "nonexistent")},
    )
    assert result.exit_code == 0
    assert "not initialized" in result.output.lower()


# ------------------------------------------------------------------ #
#  log tail
# ------------------------------------------------------------------ #

def test_log_tail_empty(tmp_vault: Vault):
    result = runner.invoke(
        app,
        ["log", "tail"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "No log entries" in result.output


def test_log_tail_with_entries(tmp_vault: Vault):
    tmp_vault.append_log("- Ingested something")
    tmp_vault.append_log("- Updated entities")
    result = runner.invoke(
        app,
        ["log", "tail"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "Ingested" in result.output or "Updated" in result.output


# ------------------------------------------------------------------ #
#  lint
# ------------------------------------------------------------------ #

def test_lint_clean_vault(tmp_vault: Vault):
    result = runner.invoke(
        app,
        ["lint"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    assert "No lint issues" in result.output


def test_lint_detects_orphan(tmp_vault: Vault):
    # Create a page not in the index
    orphan = tmp_vault.wiki_entities / "orphan-entity.md"
    orphan.write_text("---\nname: Orphan\ntype: entity\n---\n\n# Orphan\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["lint"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0  # warnings only, not errors
    assert "orphan" in result.output.lower()


def test_lint_detects_dead_link(tmp_vault: Vault):
    page = tmp_vault.wiki_entities / "entity.md"
    page.write_text(
        "---\nname: Entity\ntype: entity\n---\n\n# Entity\n\n[Missing](../concepts/nonexistent.md)\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["lint"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 1  # dead link is an error
    assert "dead" in result.output.lower() or "Dead" in result.output


def test_lint_fix_dry_run(tmp_vault: Vault):
    orphan = tmp_vault.wiki_entities / "orphan.md"
    orphan.write_text("---\nname: Orphan\ntype: entity\n---\n\n# Orphan\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["lint", "--fix", "--dry-run"],
        env={"LLM_WIKI_VAULT_PATH": str(tmp_vault.path)},
    )
    assert result.exit_code == 0
    # Index should not have been modified
    idx = tmp_vault.index.read_text(encoding="utf-8")
    assert "orphan" not in idx
