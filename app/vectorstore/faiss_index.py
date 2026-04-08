"""
Thin wrapper around a single FAISS index.

Responsibility: vector-level operations only (add, search, save, load).
It knows nothing about NLQs, SQLs, or lessons — that's KBManager's job.

Index type: IndexFlatIP (inner product)
  - "Flat" = exhaustive exact search, no approximation
  - "IP"   = inner product; with L2-normalized vectors this equals cosine similarity
  - Scores in [-1.0, 1.0]; higher is more similar
  - Perfectly fine for < 10k vectors (our use case)

ID assignment: FAISS assigns sequential integer IDs starting from 0.
  - The first vector added gets ID 0, the second gets ID 1, etc.
  - `ntotal` before add = first ID that will be assigned
  - This ID is stored as `faiss_index_id` in SQLite to link back to records.

Persistence: indexes are in-memory only at runtime.
  - Call save() to write to disk (binary format).
  - Call load() to restore from disk.
  - KBManager calls save_indexes() once after a batch add, not per-vector.
"""

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim: int) -> None:
        """
        Create a new in-memory IndexFlatIP of the given dimension.

        Args:
            dim: Embedding dimension (384 for all-MiniLM-L6-v2).
        """
        self._dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray) -> list[int]:
        """
        Add vectors to the index. Returns the list of assigned FAISS IDs.

        FAISS assigns IDs sequentially starting from `ntotal` before this call.
        So if the index had 5 vectors and you add 3, the new IDs are [5, 6, 7].

        Args:
            vectors: float32 ndarray of shape (n, dim). Must be L2-normalized
                     (Embedder handles this).

        Returns:
            List of integer IDs assigned by FAISS, one per input vector.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        """
        shape	    类比
        (384,)	    一行文字："hello world..."
        (1, 384)	一个只有一行的表格：[["hello world..."]]
        FAISS 的 add() 要求二维 (N, 384)，不接受一维 (384,)，这就是之前 faiss_index.py 里要做 reshape(1, -1) 的原因——把 (384,) 变成 (1, 384) 才能传给 FAISS。
        e.g.:
        a = np.array([1, 2, 3])        # shape (3,)
        b = np.array([[1, 2, 3]])      # shape (1, 3)
        a.ndim  # 1
        b.ndim  # 2
        """

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        first_id = self._index.ntotal
        self._index.add(vectors)
        return list(range(first_id, self._index.ntotal))

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Find the k nearest vectors to `query` by cosine similarity.

        Args:
            query: float32 ndarray of shape (dim,) or (1, dim). L2-normalized.
            k:     Number of results to return. Clamped to index size.

        Returns:
            List of (faiss_id, score) tuples, sorted by score descending.
            Score is cosine similarity in [-1.0, 1.0].
            Returns empty list if index is empty.
        """
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        scores, ids = self._index.search(query, k)

        # scores and ids are shape (1, k); flatten to 1-D
        return [
            (int(idx), float(score))
            for idx, score in zip(ids[0], scores[0])
            if idx != -1  # FAISS returns -1 for padding when k > ntotal
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write index to disk as a binary file. Creates parent dirs if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        """
        Load index from disk, replacing the current in-memory index.
        Safe to call even if path doesn't exist yet (no-op in that case).
        """
        if not path.exists():
            return
        self._index = faiss.read_index(str(path))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal
