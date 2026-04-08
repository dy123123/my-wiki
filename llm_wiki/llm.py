"""LLM abstraction layer — OpenAI-compatible backend."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from llm_wiki.config import Settings


class LLMError(Exception):
    pass


class LLMClient:
    """Thin wrapper around the OpenAI SDK for any compatible backend."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "no-key-set",
            timeout=settings.llm_timeout,
        )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        # Some models (o1, o3, o4-mini) require max_completion_tokens instead of max_tokens.
        # This flag is set to True on the first 400 error about the parameter name.
        self._use_max_completion_tokens: bool = False

        # Ollama extra options (num_gpu, num_ctx) — ignored by non-Ollama backends
        _ollama_opts: dict = {}
        if settings.llm_num_gpu > 0:
            _ollama_opts["num_gpu"] = settings.llm_num_gpu
        if settings.llm_num_ctx > 0:
            _ollama_opts["num_ctx"] = settings.llm_num_ctx
        self._extra_body = {"options": _ollama_opts} if _ollama_opts else None

    # ------------------------------------------------------------------ #
    #  Core completion
    # ------------------------------------------------------------------ #

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        max_retries: int = 3,
    ) -> str:
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self._use_max_completion_tokens:
            kwargs["max_completion_tokens"] = tokens
        else:
            kwargs["max_tokens"] = tokens
        if response_format:
            kwargs["response_format"] = response_format
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                wait = 2 ** attempt * 5
                time.sleep(wait)
                last_err = e
            except APIConnectionError as e:
                raise LLMError(
                    f"Cannot reach LLM at {self._settings.llm_base_url}: {e}"
                ) from e
            except APIStatusError as e:
                # Some models (o1, o3, o4-mini) require max_completion_tokens
                if (
                    e.status_code == 400
                    and "max_tokens" in str(e.message)
                    and "max_completion_tokens" in str(e.message)
                    and not self._use_max_completion_tokens
                ):
                    self._use_max_completion_tokens = True
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = tokens
                    last_err = e
                    continue
                if e.status_code in (500, 502, 503) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    last_err = e
                else:
                    raise LLMError(f"LLM API error {e.status_code}: {e.message}") from e

        raise LLMError(f"LLM request failed after {max_retries} attempts: {last_err}")

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> dict:
        """Complete and parse JSON from the response."""
        for attempt in range(max_retries):
            try:
                text = self.complete(
                    messages,
                    response_format={"type": "json_object"},
                    **kwargs,
                )
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # Try extracting JSON from markdown code fence
                text = self.complete(messages, **kwargs)
                extracted = _extract_json(text)
                if extracted is not None:
                    return extracted
                if attempt == max_retries - 1:
                    raise LLMError(f"Could not parse JSON from LLM response: {text[:300]}")
                time.sleep(1)

        raise LLMError("JSON extraction failed")

    # ------------------------------------------------------------------ #
    #  Connectivity check
    # ------------------------------------------------------------------ #

    def ping(self) -> tuple[bool, str]:
        """Return (success, message)."""
        try:
            token_kwarg = (
                {"max_completion_tokens": 10}
                if self._use_max_completion_tokens
                else {"max_tokens": 10}
            )
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
                temperature=0,
                **token_kwarg,
            )
            reply = (resp.choices[0].message.content or "").strip().lower()
            return True, f"OK — model={self.model!r} replied: {reply!r}"
        except APIConnectionError as e:
            return False, f"Connection failed: {e}"
        except APIStatusError as e:
            return False, f"API error {e.status_code}: {e.message}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    # ------------------------------------------------------------------ #
    #  Higher-level helpers
    # ------------------------------------------------------------------ #

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        return self.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> dict:
        return self.complete_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from a string that may contain markdown."""
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Bare JSON object anywhere in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None
