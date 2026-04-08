"""
Integration tests for KBManager (both indexes).

These tests hit real SQLite and real FAISS — no mocking.
They use a temporary database and index directory so they don't
pollute the dev data/ directory.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.db.models import Lesson
from app.db.sqlite_client import init_db, insert_lesson, insert_nlq_sql_pair


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """
    Redirect all data paths to a temp directory for test isolation.
    Each test gets a clean slate.
    """
    from app.core import config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "sqlite_path", tmp_path / "test.db")
    monkeypatch.setattr(cfg_module.settings, "index_dir", tmp_path / "indexes")
    monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

    # Re-run init_db so tables are created in the temp DB
    init_db()
    yield


def _make_kb():
    """Fresh KBManager pointing at the monkeypatched paths."""
    # Re-import after monkeypatching so _INDEX1_PATH / _INDEX2_PATH pick up new settings
    import importlib
    import app.vectorstore.kb_manager as kb_mod
    importlib.reload(kb_mod)
    return kb_mod.KBManager()


# ── Index 1: examples ─────────────────────────────────────────────────────────

def test_add_and_search_examples():
    kb = _make_kb()

    pairs = [
        ("how many customers are there", "SELECT COUNT(*) FROM customers"),
        ("total revenue from all orders", "SELECT SUM(total_amount) FROM orders"),
        ("list all products", "SELECT name FROM products"),
    ]
    for nlq, sql in pairs:
        row_id = insert_nlq_sql_pair(nlq, sql)
        kb.add_example(nlq, sql, row_id)

    results = kb.search_examples("how many users do we have", k=3)

    assert len(results) == 3
    assert results[0]["nlq"] == "how many customers are there"
    assert results[0]["score"] > 0.4  # all-MiniLM-L6-v2 scores ~0.55 for this pair
    assert "sql" in results[0]
    # Results should be sorted by score descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_example_count():
    kb = _make_kb()
    assert kb.example_count == 0

    row_id = insert_nlq_sql_pair("test nlq", "SELECT 1")
    kb.add_example("test nlq", "SELECT 1", row_id)
    assert kb.example_count == 1


def test_search_empty_index_returns_empty():
    kb = _make_kb()
    results = kb.search_examples("any query", k=3)
    assert results == []


# ── Index 2: lessons ──────────────────────────────────────────────────────────

def test_add_and_search_lessons():
    kb = _make_kb()

    lesson = Lesson(
        title="Use DATE_TRUNC for month filtering",
        trigger="NLQ asks about filtering orders or events by month or time period",
        diagnosis="Used MONTH() which is not supported; should use DATE_TRUNC",
        fix_rule="Always use DATE_TRUNC('month', date_col) for month-level filtering",
        error_category="date_handling",
        example='{"nlq": "orders last month", "pred_sql": "...", "gold_sql": "..."}',
    )
    row_id = insert_lesson(
        title=lesson.title,
        trigger=lesson.trigger,
        diagnosis=lesson.diagnosis,
        fix_rule=lesson.fix_rule,
        error_category=lesson.error_category,
        example=lesson.example,
    )
    kb.add_lesson(lesson, row_id)

    results = kb.search_lessons("how many orders were placed last month", k=1)

    assert len(results) == 1
    assert results[0]["title"] == "Use DATE_TRUNC for month filtering"
    assert results[0]["score"] > 0.3  # all-MiniLM-L6-v2 scores ~0.47 for this pair
    assert "fix_rule" in results[0]


def test_lesson_count():
    kb = _make_kb()
    assert kb.lesson_count == 0

    lesson = Lesson(
        title="T", trigger="some trigger", diagnosis="d", fix_rule="f",
        error_category="other", example="{}"
    )
    row_id = insert_lesson(
        title=lesson.title, trigger=lesson.trigger, diagnosis=lesson.diagnosis,
        fix_rule=lesson.fix_rule, error_category=lesson.error_category, example=lesson.example,
    )
    kb.add_lesson(lesson, row_id)
    assert kb.lesson_count == 1


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_and_reload_index():
    kb = _make_kb()
    row_id = insert_nlq_sql_pair("how many products", "SELECT COUNT(*) FROM products")
    kb.add_example("how many products", "SELECT COUNT(*) FROM products", row_id)
    kb.save_indexes()

    # Fresh KBManager — loads from disk
    kb2 = _make_kb()
    kb2.load_indexes()
    assert kb2.example_count == 1

    results = kb2.search_examples("how many items in the catalog", k=1)
    assert len(results) == 1
    assert results[0]["nlq"] == "how many products"
