"""
Embedding generation using sentence-transformers.

Design decisions:
  - Singleton model: SentenceTransformer loads ~300MB into memory on first import.
    We load it once at module level and reuse it. Importing this module triggers
    the download/load the first time (takes ~1-2s).
  - L2 normalization: we normalize every vector to unit length before returning.
    This makes dot product equal to cosine similarity, which is what IndexFlatIP
    expects. Callers never need to normalize themselves.
  - float32 dtype: FAISS requires float32. sentence-transformers returns float32
    by default, but we enforce it explicitly.

Usage:
    from app.embeddings.embedder import encode, encode_batch

    vec  = encode("how many customers?")           # shape (384,)
    vecs = encode_batch(["query 1", "query 2"])    # shape (2, 384)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# ── Singleton model ───────────────────────────────────────────────────────────
# Loaded once when this module is first imported.
# "all-MiniLM-L6-v2": 384-dim, ~22M params, fast on CPU, strong for semantic similarity.
_model = SentenceTransformer(settings.embedding_model)


def encode(text: str) -> np.ndarray:
    """
    Embed a single string.

    Returns a float32 ndarray of shape (embedding_dim,) with unit L2 norm.
    The unit-norm guarantee is essential: it makes inner product = cosine similarity
    in FAISS IndexFlatIP.

    Args:
        text: Any natural language string (NLQ, lesson trigger, etc.)

    Returns:
        np.ndarray of shape (384,), dtype float32, L2-normalized.
    """
    vec = _model.encode(text, convert_to_numpy=True).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def encode_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings in one efficient batch call.

    Preferred over calling encode() in a loop for bulk operations (training pipeline).
    sentence-transformers handles internal batching automatically.

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32, each row L2-normalized.
    """
    if not texts:
        return np.empty((0, settings.embedding_dim), dtype=np.float32)

    vecs = _model.encode(texts, convert_to_numpy=True, batch_size=64).astype(np.float32)

    # Normalize each row to unit length
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # avoid division by zero
    vecs = vecs / norms

    return vecs
