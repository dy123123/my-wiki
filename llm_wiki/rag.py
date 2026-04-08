"""RAG index — chunk storage, embedding storage, hybrid retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
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


_STOP = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "not","be","was","are","with","this","that","what","how","why","when",
    "where","who","which","can","do","does","did","has","have","had","will",
    "would","could","should","may","might","i","we","you","they","he","she",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOP]


def _bm25_scores(query_tokens: list[str], chunks: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    Lightweight BM25 scoring — no external dependency.
    Works well for exact register names, hex addresses, flag names.
    """
    if not query_tokens or not chunks:
        return [0.0] * len(chunks)

    # Corpus stats
    avgdl = sum(len(_tokenize(c)) for c in chunks) / max(len(chunks), 1)
    df: dict[str, int] = defaultdict(int)
    tokenized_chunks = [_tokenize(c) for c in chunks]
    for tokens in tokenized_chunks:
        for term in set(tokens):
            df[term] += 1

    N = len(chunks)
    scores = []
    for doc_tokens in tokenized_chunks:
        dl = len(doc_tokens)
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1
        score = 0.0
        for term in query_tokens:
            if term not in df:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf = tf_map[term]
            bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
            score += idf * bm25_tf
        scores.append(score)
    return scores


def _rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[float]:
    """
    Reciprocal Rank Fusion: combine multiple ranked lists of indices.
    Returns fused scores indexed by original position.
    """
    n = max(max(lst) for lst in ranked_lists if lst) + 1
    scores = [0.0] * n
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] += 1.0 / (k + rank + 1)
    return scores


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
        query_text: str = "",
        top_k: int = 5,
        source_ids: list[str] | None = None,
    ) -> list[ChunkResult]:
        """
        Hybrid search: Reciprocal Rank Fusion of embedding similarity + BM25.

        - Embedding similarity captures semantic relevance
        - BM25 captures exact term matches (register names, hex addresses, flag names)
        - RRF combines both ranked lists without calibrating weights

        Returns top_k results sorted by fused score descending.
        """
        import numpy as np

        sources = source_ids or self.indexed_sources()
        all_source_ids: list[str] = []
        all_chunk_idxs: list[int] = []
        all_chunk_texts: list[str] = []
        embed_scores: list[float] = []

        for source_id in sources:
            matrix = self.load_embeddings_np(source_id)
            chunks = self.load_chunks(source_id)
            if matrix is None or len(chunks) == 0:
                continue
            if len(chunks) != len(matrix):
                continue

            sims = _np_cosine_sim(query_embedding, matrix).tolist()
            embed_scores.extend(sims)
            all_source_ids.extend([source_id] * len(chunks))
            all_chunk_idxs.extend(range(len(chunks)))
            all_chunk_texts.extend(chunks)

        if not all_chunk_texts:
            return []

        n = len(all_chunk_texts)
        top_k = min(top_k, n)

        # Embedding rank
        embed_arr = np.array(embed_scores, dtype=np.float32)
        embed_ranked = np.argsort(embed_arr)[::-1].tolist()

        # BM25 rank (keyword matching — crucial for register names / hex addresses)
        query_tokens = _tokenize(query_text) if query_text else []
        if query_tokens:
            bm25 = _bm25_scores(query_tokens, all_chunk_texts)
            bm25_arr = np.array(bm25, dtype=np.float32)
            bm25_ranked = np.argsort(bm25_arr)[::-1].tolist()
            ranked_lists = [embed_ranked, bm25_ranked]
        else:
            ranked_lists = [embed_ranked]

        # RRF fusion
        fused = _rrf_fuse(ranked_lists)
        fused_arr = np.array(fused, dtype=np.float32)
        top_indices = np.argsort(fused_arr)[::-1][:top_k]

        return [
            ChunkResult(
                source_id=all_source_ids[i],
                chunk_idx=all_chunk_idxs[i],
                score=float(fused[i]),
                text=all_chunk_texts[i],
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
