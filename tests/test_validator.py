"""
Unit tests for app/sql/validator.py.

No mocks needed — validate_sql() is a pure function (sqlglot, no I/O).

Test strategy:
  - Valid SELECT and WITH queries → (True, None)
  - Empty / whitespace-only input → (False, error)
  - Unparseable SQL → (False, error)
  - Unsafe statement types: DROP, INSERT, UPDATE, DELETE, CREATE
  - Multi-statement SQL → (False, error)
  - Error message content for human-readable rejection reasons
"""

import pytest

from app.sql.validator import validate_sql


# ── Valid SQL ─────────────────────────────────────────────────────────────────

class TestValidSql:
    def test_simple_select(self):
        ok, err = validate_sql("SELECT 1")
        assert ok is True
        assert err is None

    def test_select_with_where(self):
        ok, err = validate_sql("SELECT * FROM customers WHERE id = 1")
        assert ok is True
        assert err is None

    def test_select_with_aggregation(self):
        ok, err = validate_sql("SELECT COUNT(*) FROM orders GROUP BY status")
        assert ok is True
        assert err is None

    def test_with_cte(self):
        sql = "WITH cte AS (SELECT id FROM customers) SELECT * FROM cte"
        ok, err = validate_sql(sql)
        assert ok is True
        assert err is None

    def test_select_with_subquery(self):
        sql = "SELECT * FROM (SELECT id FROM customers) sub"
        ok, err = validate_sql(sql)
        assert ok is True
        assert err is None


# ── Empty / blank input ───────────────────────────────────────────────────────

class TestEmptyInput:
    def test_empty_string(self):
        ok, err = validate_sql("")
        assert ok is False
        assert err == "SQL is empty."

    def test_whitespace_only(self):
        ok, err = validate_sql("   \n\t  ")
        assert ok is False
        assert err == "SQL is empty."


# ── Unparseable SQL ───────────────────────────────────────────────────────────

class TestUnparseable:
    def test_gibberish(self):
        ok, err = validate_sql("NOT VALID SQL @@##")
        assert ok is False
        assert err is not None

    def test_typo_in_select(self):
        ok, err = validate_sql("SELET 1")
        assert ok is False
        assert err is not None


# ── Unsafe statement types ────────────────────────────────────────────────────

class TestUnsafeStatements:
    def test_drop_table(self):
        ok, err = validate_sql("DROP TABLE customers")
        assert ok is False
        assert "Unsafe" in err

    def test_insert(self):
        ok, err = validate_sql("INSERT INTO customers (name) VALUES ('test')")
        assert ok is False
        assert "Unsafe" in err

    def test_update(self):
        ok, err = validate_sql("UPDATE customers SET name = 'x' WHERE id = 1")
        assert ok is False
        assert "Unsafe" in err

    def test_delete(self):
        ok, err = validate_sql("DELETE FROM customers WHERE id = 1")
        assert ok is False
        assert "Unsafe" in err

    def test_create_table(self):
        ok, err = validate_sql("CREATE TABLE foo (id INT)")
        assert ok is False
        assert "Unsafe" in err


# ── Multi-statement ───────────────────────────────────────────────────────────

class TestMultiStatement:
    def test_select_then_drop(self):
        ok, err = validate_sql("SELECT 1; DROP TABLE customers")
        assert ok is False
        assert "Multiple statements" in err

    def test_two_selects(self):
        ok, err = validate_sql("SELECT 1; SELECT 2")
        assert ok is False
        assert "Multiple statements" in err


# ── Return type guarantees ────────────────────────────────────────────────────

class TestReturnType:
    def test_valid_returns_none_error(self):
        _, err = validate_sql("SELECT 1")
        assert err is None

    def test_invalid_returns_string_error(self):
        _, err = validate_sql("DROP TABLE x")
        assert isinstance(err, str)
        assert len(err) > 0
