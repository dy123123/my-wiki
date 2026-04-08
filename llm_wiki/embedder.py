"""Embedding and reranking client for RAG."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_wiki.config import Settings


class EmbedError(Exception):
    pass


class EmbedClient:
    """OpenAI-compatible embedding client + optional reranker."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        base_url = settings.embed_base_url or settings.llm_base_url
        api_key = settings.embed_api_key or settings.llm_api_key or "none"

        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        except ImportError as e:
            raise EmbedError(f"openai package not installed: {e}") from e

        self._embed_model = settings.embed_model
        self._rerank_model = settings.rerank_model
        self._rerank_base_url = (
            settings.rerank_base_url
            or settings.embed_base_url
            or settings.llm_base_url
        )
        self._rerank_api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self._embed_model)

    @property
    def rerank_enabled(self) -> bool:
        return bool(self._rerank_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        if not self._embed_model:
            raise EmbedError("Embedding model not configured (LLM_WIKI_EMBED_MODEL)")
        if not texts:
            return []
        try:
            resp = self._client.embeddings.create(
                model=self._embed_model,
                input=texts,
            )
            # Sort by index to preserve order
            ordered = sorted(resp.data, key=lambda x: x.index)
            return [item.embedding for item in ordered]
        except Exception as e:
            raise EmbedError(f"Embedding failed: {e}") from e

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        Rerank documents by relevance to query.
        Returns a score per document (higher = more relevant).

        Tries two API styles:
          1. POST /v1/rerank  (Cohere-compatible, supported by some proxies)
          2. POST /v1/score   (vLLM cross-encoder scoring)
        Falls back to returning equal scores if neither works.
        """
        if not self._rerank_model or not documents:
            return [1.0] * len(documents)

        # Try /v1/rerank (Cohere-style)
        try:
            return self._rerank_cohere(query, documents)
        except Exception:
            pass

        # Try /v1/score (vLLM-style)
        try:
            return self._rerank_vllm(query, documents)
        except Exception:
            pass

        # Fallback: all equal scores
        return [1.0] * len(documents)

    def _rerank_cohere(self, query: str, documents: list[str]) -> list[float]:
        """Cohere-compatible /v1/rerank endpoint."""
        import urllib.request

        base = self._rerank_base_url.rstrip("/")
        url = f"{base}/rerank"
        payload = json.dumps({
            "model": self._rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._rerank_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        # Results may be in different orders; map back by index
        scores = [0.0] * len(documents)
        for result in data.get("results", []):
            idx = result.get("index", 0)
            scores[idx] = result.get("relevance_score", result.get("score", 0.0))
        return scores

    def _rerank_vllm(self, query: str, documents: list[str]) -> list[float]:
        """vLLM /v1/score cross-encoder endpoint."""
        import urllib.request

        base = self._rerank_base_url.rstrip("/")
        url = f"{base}/score"
        payload = json.dumps({
            "model": self._rerank_model,
            "text_1": query,
            "text_2": documents,
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._rerank_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        # vLLM returns {"data": [{"score": float}, ...]}
        return [item.get("score", 0.0) for item in data.get("data", [])]
