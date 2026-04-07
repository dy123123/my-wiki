"""Tests for the LLM client abstraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_wiki.llm import LLMClient, LLMError, _extract_json
from llm_wiki.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_base_url="http://test:11434/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_temperature=0.2,
        llm_max_tokens=512,
    )


# ------------------------------------------------------------------ #
#  _extract_json
# ------------------------------------------------------------------ #

def test_extract_json_direct():
    result = _extract_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_extract_json_markdown_fence():
    text = '```json\n{"key": "value"}\n```'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_bare_object():
    text = 'Here is some text {"key": "value"} and more text'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_invalid():
    result = _extract_json("this is not json at all")
    assert result is None


def test_extract_json_empty():
    result = _extract_json("")
    assert result is None


# ------------------------------------------------------------------ #
#  LLMClient.ping
# ------------------------------------------------------------------ #

def test_ping_success(settings: Settings):
    with patch("llm_wiki.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "pong"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(settings)
        success, msg = llm.ping()

    assert success is True
    assert "pong" in msg


def test_ping_connection_failure(settings: Settings):
    from openai import APIConnectionError
    with patch("llm_wiki.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIConnectionError.__new__(APIConnectionError)

        llm = LLMClient(settings)
        success, msg = llm.ping()

    assert success is False


# ------------------------------------------------------------------ #
#  LLMClient.complete
# ------------------------------------------------------------------ #

def test_complete_success(settings: Settings):
    with patch("llm_wiki.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(settings)
        result = llm.complete([{"role": "user", "content": "Hi"}])

    assert result == "Hello!"


def test_complete_json_success(settings: Settings):
    with patch("llm_wiki.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"result": "ok"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        llm = LLMClient(settings)
        result = llm.complete_json([{"role": "user", "content": "Return JSON"}])

    assert result == {"result": "ok"}


def test_complete_json_from_markdown_fence(settings: Settings):
    """Should extract JSON even if wrapped in a code fence."""
    with patch("llm_wiki.llm.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First call (json_object mode) raises ValueError, second returns markdown
        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"result": "ok"}\n```'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and kwargs.get("response_format"):
                import json
                # Simulate non-JSON response that fails parse
                bad = MagicMock()
                bad.choices = [MagicMock()]
                bad.choices[0].message.content = "not json"
                return bad
            return mock_response

        mock_client.chat.completions.create.side_effect = side_effect

        llm = LLMClient(settings)
        result = llm.complete_json([{"role": "user", "content": "Return JSON"}])

    assert result == {"result": "ok"}
