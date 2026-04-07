"""Tests for configuration management."""

from pathlib import Path

import pytest

from llm_wiki.config import Settings, reset_settings


def test_defaults():
    s = Settings()
    assert s.llm_base_url == "https://api.openai.com/v1"
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_temperature == 0.2
    assert s.llm_max_tokens == 4096
    assert s.vault_path == Path("vault")
    assert s.dry_run is False
    assert s.verbose is False


def test_is_llm_configured_false():
    s = Settings(llm_api_key="")
    assert not s.is_llm_configured()


def test_is_llm_configured_true():
    s = Settings(llm_api_key="sk-test")
    assert s.is_llm_configured()


def test_display_dict_masks_api_key():
    s = Settings(llm_api_key="sk-abcdefghijklmn")
    d = s.display_dict()
    assert "sk-abcdef" in d["llm_api_key"]
    assert "ijklmn" not in d["llm_api_key"]


def test_display_dict_short_key_masked():
    s = Settings(llm_api_key="sk-x")
    d = s.display_dict()
    assert "***" in d["llm_api_key"]


def test_vault_path_as_string():
    s = Settings(vault_path="my/vault")
    assert s.vault_path == Path("my/vault")


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("LLM_WIKI_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_WIKI_LLM_TEMPERATURE", "0.7")
    reset_settings()
    s = Settings()
    assert s.llm_model == "gpt-4o"
    assert s.llm_temperature == 0.7
    reset_settings()


def test_temperature_bounds():
    with pytest.raises(Exception):
        Settings(llm_temperature=3.0)

    with pytest.raises(Exception):
        Settings(llm_temperature=-1.0)
