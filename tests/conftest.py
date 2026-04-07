"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_wiki.config import Settings
from llm_wiki.vault import Vault


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Vault:
    """A fully initialized vault in a temporary directory."""
    vault = Vault(tmp_path / "vault")
    # Write stub schema files so init() doesn't fail looking for sample_vault/
    (tmp_path / "vault").mkdir()
    vault.raw.mkdir(parents=True)
    vault.normalized.mkdir(parents=True)
    vault.schema.mkdir(parents=True)
    vault.wiki_sources.mkdir(parents=True)
    vault.wiki_entities.mkdir(parents=True)
    vault.wiki_concepts.mkdir(parents=True)
    vault.wiki_topics.mkdir(parents=True)
    vault.wiki_analyses.mkdir(parents=True)
    vault.wiki_reports.mkdir(parents=True)

    vault.index.write_text("# Wiki Index\n\n## Sources\n\n## Entities\n\n## Concepts\n\n## Topics\n", encoding="utf-8")
    vault.log.write_text("# Wiki Log\n\n", encoding="utf-8")
    vault.overview.write_text("# Wiki Overview\n\n", encoding="utf-8")

    for fname, content in [
        ("AGENTS.md", "# Agents\n"),
        ("wiki_schema.md", "# Schema\n"),
        ("page_conventions.md", "# Conventions\n"),
    ]:
        (vault.schema / fname).write_text(content, encoding="utf-8")

    return vault


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """A simple text file to use as a source."""
    p = tmp_path / "hello.txt"
    p.write_text("Hello world. This is a sample document about AI.", encoding="utf-8")
    return p


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """A markdown file to use as a source."""
    p = tmp_path / "sample.md"
    p.write_text("# Sample Doc\n\nThis is about transformers and attention mechanisms.", encoding="utf-8")
    return p


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        vault_path=tmp_path / "vault",
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLMClient for testing without a real LLM."""
    from llm_wiki.llm import LLMClient

    llm = MagicMock(spec=LLMClient)
    llm.model = "test-model"
    llm.ping.return_value = (True, "OK — model='test-model' replied: 'pong'")
    llm.complete.return_value = "Test LLM response"
    llm.chat.return_value = "Test LLM response"

    # Default structured response for source analysis
    llm.complete_json.return_value = {
        "title": "Test Document",
        "summary": "A test document about AI.",
        "key_points": ["Point 1", "Point 2"],
        "entities": [
            {
                "name": "Test Corp",
                "type": "organization",
                "description": "A fictional organization",
                "mentions": ["Test Corp built the system"],
            }
        ],
        "concepts": [
            {
                "name": "Machine Learning",
                "description": "A field of AI",
                "related_concepts": ["Deep Learning"],
            }
        ],
        "topics": [{"name": "Artificial Intelligence", "description": "Broad AI topic"}],
        "tags": ["ai", "ml"],
        "date_published": None,
        "authors": [],
    }
    llm.chat_json.return_value = llm.complete_json.return_value
    return llm
