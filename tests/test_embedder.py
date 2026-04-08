"""
Unit tests for the embedder.

These tests verify shape, dtype, and normalization guarantees that the rest of
the system depends on. They do NOT require an API key or network access.
"""

import numpy as np

from app.embeddings.embedder import encode, encode_batch
from app.core.config import settings


def test_encode_shape():
    vec = encode("how many customers are there?")
    assert vec.shape == (settings.embedding_dim,), f"Expected ({settings.embedding_dim},), got {vec.shape}"


def test_encode_dtype():
    vec = encode("total revenue from all orders")
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


def test_encode_unit_norm():
    vec = encode("list all products in the electronics category")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}"


def test_encode_batch_shape():
    texts = ["query one", "query two", "query three"]
    vecs = encode_batch(texts)
    assert vecs.shape == (3, settings.embedding_dim)


def test_encode_batch_dtype():
    vecs = encode_batch(["a", "b"])
    assert vecs.dtype == np.float32


def test_encode_batch_unit_norms():
    texts = ["how many orders?", "total revenue?", "list customers"]
    vecs = encode_batch(texts)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Not all unit norms: {norms}"


def test_encode_batch_empty():
    vecs = encode_batch([])
    assert vecs.shape == (0, settings.embedding_dim)


def test_similar_queries_high_cosine():
    """Semantically similar queries should have high cosine similarity (> 0.8)."""
    v1 = encode("how many customers are there")
    v2 = encode("what is the total number of customers")
    cosine = float(np.dot(v1, v2))  # both unit-norm, so dot = cosine
    assert cosine > 0.8, f"Expected high similarity, got {cosine:.3f}"


def test_dissimilar_queries_lower_cosine():
    """Very different queries should have lower cosine similarity."""
    v1 = encode("how many customers are there")
    v2 = encode("what is the most expensive product in electronics")
    cosine = float(np.dot(v1, v2))
    assert cosine < 0.5, f"Expected lower similarity between dissimilar queries, got {cosine:.3f}"
