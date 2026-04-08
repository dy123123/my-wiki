"""RAG index — chunk storage, embedding storage, retrieval."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_wiki.vault import Vault


@dataclass
class ChunkResult:
    source_id: str
    chunk_idx: int
    score: float
    text: str


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks at paragraph/sentence boundaries.
    Tries to break at double-newlines, falls back to hard cuts.
    """
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to break at a paragraph boundary (double newline)
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            # Fall back to single newline
            boundary = text.rfind("\n", start + size // 2, end)
        if boundary == -1 or boundary <= start:
            # Hard cut
            boundary = end

        chunks.append(text[start:boundary].strip())
        start = max(start + 1, boundary - overlap)

    return [c for c in chunks if c.strip()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagIndex:
    """
    Manages chunk and embedding storage for RAG retrieval.

    Storage layout:
      vault/chunks/{source_id}.json      — list of chunk texts
      vault/embeddings/{source_id}.json  — list of embedding vectors (list[float])
    """

    def __init__(self, vault: Vault) -> None:
        self._vault = vault
        self._vault.chunks.mkdir(parents=True, exist_ok=True)
        self._vault.embeddings.mkdir(parents=True, exist_ok=True)

    def chunk_path(self, source_id: str) -> Path:
        return self._vault.chunks / f"{source_id}.json"

    def embed_path(self, source_id: str) -> Path:
        return self._vault.embeddings / f"{source_id}.json"

    def is_indexed(self, source_id: str) -> bool:
        return self.chunk_path(source_id).exists() and self.embed_path(source_id).exists()

    def save(self, source_id: str, chunks: list[str], embeddings: list[list[float]]) -> None:
        self.chunk_path(source_id).write_text(
            json.dumps(chunks, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
        self.embed_path(source_id).write_text(
            json.dumps(embeddings),
            encoding="utf-8",
        )

    def load_chunks(self, source_id: str) -> list[str]:
        p = self.chunk_path(source_id)
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def load_embeddings(self, source_id: str) -> list[list[float]]:
        p = self.embed_path(source_id)
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def indexed_sources(self) -> list[str]:
        return [p.stem for p in self._vault.chunks.glob("*.json")]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[str] | None = None,
    ) -> list[ChunkResult]:
        """
        Score all indexed chunks by cosine similarity to query_embedding.
        Returns top_k results sorted by score descending.
        """
        sources = source_ids or self.indexed_sources()
        candidates: list[ChunkResult] = []

        for source_id in sources:
            chunks = self.load_chunks(source_id)
            embeddings = self.load_embeddings(source_id)
            if len(chunks) != len(embeddings):
                continue
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                score = cosine_similarity(query_embedding, emb)
                candidates.append(ChunkResult(source_id, idx, score, chunk))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def rerank_and_trim(
        self,
        query: str,
        candidates: list[ChunkResult],
        embedder,
        top_k: int,
    ) -> list[ChunkResult]:
        """Apply reranker scores to candidates and return top_k."""
        if not candidates or not embedder.rerank_enabled:
            return candidates[:top_k]

        texts = [c.text for c in candidates]
        try:
            scores = embedder.rerank(query, texts)
            reranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
            return [c for _, c in reranked[:top_k]]
        except Exception:
            return candidates[:top_k]
