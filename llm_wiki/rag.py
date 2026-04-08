"""RAG index — chunk storage, embedding storage, retrieval."""

from __future__ import annotations

import json
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
            boundary = text.rfind("\n", start + size // 2, end)
        if boundary == -1 or boundary <= start:
            boundary = end

        chunks.append(text[start:boundary].strip())
        start = max(start + 1, boundary - overlap)

    return [c for c in chunks if c.strip()]


def _np_cosine_sim(query_vec, matrix):
    """Vectorized cosine similarity: query_vec (D,) vs matrix (N, D) → (N,)."""
    import numpy as np
    q = np.array(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(len(matrix), dtype=np.float32)
    q = q / q_norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = matrix / norms
    return normed @ q  # (N,)


class RagIndex:
    """
    Manages chunk and embedding storage for RAG retrieval.

    Storage layout:
      vault/chunks/{source_id}.json       — JSON list of chunk strings
      vault/embeddings/{source_id}.npy    — float32 numpy array, shape (N, dim)

    numpy is used for embeddings to keep memory and disk usage low.
    For a 500-page doc (~1000 chunks, 7168-dim Qwen3):
      JSON  ≈ 100 MB  (slow to parse, huge RAM usage)
      .npy  ≈  28 MB  (mmap-able, vectorized similarity in ms)
    """

    def __init__(self, vault: Vault) -> None:
        self._vault = vault
        self._vault.chunks.mkdir(parents=True, exist_ok=True)
        self._vault.embeddings.mkdir(parents=True, exist_ok=True)

    def chunk_path(self, source_id: str) -> Path:
        return self._vault.chunks / f"{source_id}.json"

    def embed_path(self, source_id: str) -> Path:
        return self._vault.embeddings / f"{source_id}.npy"

    def is_indexed(self, source_id: str) -> bool:
        return self.chunk_path(source_id).exists() and self.embed_path(source_id).exists()

    def save(self, source_id: str, chunks: list[str], embeddings: list[list[float]]) -> None:
        import numpy as np
        self.chunk_path(source_id).write_text(
            json.dumps(chunks, ensure_ascii=False),
            encoding="utf-8",
        )
        arr = np.array(embeddings, dtype=np.float32)
        np.save(str(self.embed_path(source_id)), arr)

    def load_chunks(self, source_id: str) -> list[str]:
        p = self.chunk_path(source_id)
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def load_embeddings_np(self, source_id: str):
        """Load embeddings as a numpy float32 array (N, dim). Returns None if missing."""
        import numpy as np
        p = self.embed_path(source_id)
        if not p.exists():
            return None
        return np.load(str(p))  # mmap_mode could be used for very large files

    def indexed_sources(self) -> list[str]:
        return [p.stem for p in self._vault.chunks.glob("*.json")]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_ids: list[str] | None = None,
    ) -> list[ChunkResult]:
        """
        Score all indexed chunks by cosine similarity using vectorized numpy ops.
        Returns top_k results sorted by score descending.
        """
        import numpy as np

        sources = source_ids or self.indexed_sources()
        all_scores: list[float] = []
        all_source_ids: list[str] = []
        all_chunk_idxs: list[int] = []
        chunk_texts: list[str] = []

        for source_id in sources:
            matrix = self.load_embeddings_np(source_id)
            chunks = self.load_chunks(source_id)
            if matrix is None or len(chunks) == 0:
                continue
            if len(chunks) != len(matrix):
                continue

            scores = _np_cosine_sim(query_embedding, matrix)  # (N,)
            all_scores.extend(scores.tolist())
            all_source_ids.extend([source_id] * len(chunks))
            all_chunk_idxs.extend(range(len(chunks)))
            chunk_texts.extend(chunks)

        if not all_scores:
            return []

        # Get top_k indices
        arr = np.array(all_scores, dtype=np.float32)
        top_k = min(top_k, len(arr))
        top_indices = np.argpartition(arr, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(arr[top_indices])[::-1]]

        return [
            ChunkResult(
                source_id=all_source_ids[i],
                chunk_idx=all_chunk_idxs[i],
                score=float(all_scores[i]),
                text=chunk_texts[i],
            )
            for i in top_indices
        ]

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
