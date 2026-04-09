"""
Unit tests for app/sql/comparator.py.

No I/O, no API calls, no database. Pure sqlglot + string logic.

Test strategy:
  - ast_match: verify normalization handles casing, whitespace, aliases, parse failures
  - token_sim: verify Jaccard math on known inputs
  - compare():  verify the combined dict shape and that both signals are independent
"""

import pytest

from app.sql.comparator import compare, _ast_match, _token_similarity


# ── ast_match ─────────────────────────────────────────────────────────────────

class TestAstMatch:
    def test_identical_queries(self):
        sql = "SELECT COUNT(*) FROM customers"
        assert _ast_match(sql, sql) is True

    def test_case_insensitive(self):
        # keyword casing differences should normalize away
        assert _ast_match(
            "select count(*) from customers",
            "SELECT COUNT(*) FROM customers",
        ) is True

    def test_whitespace_insensitive(self):
        assert _ast_match(
            "SELECT  COUNT(*)  FROM   customers",
            "SELECT COUNT(*) FROM customers",
        ) is True

    def test_different_values_no_match(self):
        # Same structure, different literal — must not match
        assert _ast_match(
            "SELECT COUNT(*) FROM customers WHERE country = 'Germany'",
            "SELECT COUNT(*) FROM customers WHERE country = 'USA'",
        ) is False

    def test_different_tables_no_match(self):
        assert _ast_match(
            "SELECT COUNT(*) FROM customers",
            "SELECT COUNT(*) FROM orders",
        ) is False

    def test_different_aggregation_no_match(self):
        assert _ast_match(
            "SELECT COUNT(*) FROM orders",
            "SELECT SUM(total_amount) FROM orders",
        ) is False

    def test_with_cte_matches_equivalent(self):
        pred = "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t"
        gold = "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t"
        assert _ast_match(pred, gold) is True

    def test_column_alias_normalization(self):
        # sqlglot normalizes aliases consistently — same alias = match
        assert _ast_match(
            "SELECT COUNT(*) AS total FROM customers",
            "SELECT COUNT(*) AS total FROM customers",
        ) is True

    def test_different_aliases_no_match(self):
        # Different alias names are structurally different after normalization
        assert _ast_match(
            "SELECT COUNT(*) AS total FROM customers",
            "SELECT COUNT(*) AS cnt FROM customers",
        ) is False

    def test_invalid_pred_returns_false(self):
        assert _ast_match("SELET 1", "SELECT 1") is False

    def test_invalid_gold_returns_false(self):
        assert _ast_match("SELECT 1", "SELET 1") is False

    def test_both_invalid_returns_false(self):
        assert _ast_match("NOT SQL", "ALSO NOT SQL") is False

    def test_empty_strings_return_false(self):
        assert _ast_match("", "") is False

    def test_join_query_matches(self):
        sql = (
            "SELECT c.name, COUNT(o.order_id) AS order_count "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.name"
        )
        assert _ast_match(sql, sql) is True

    def test_select_star_vs_explicit_columns_no_match(self):
        assert _ast_match(
            "SELECT * FROM customers",
            "SELECT customer_id, name FROM customers",
        ) is False


# ── token_similarity ──────────────────────────────────────────────────────────

class TestTokenSimilarity:
    def test_identical_strings(self):
        sql = "SELECT COUNT(*) FROM customers"
        assert _token_similarity(sql, sql) == pytest.approx(1.0)

    def test_completely_different(self):
        sim = _token_similarity("SELECT a FROM x", "UPDATE b SET c = 1")
        assert sim == pytest.approx(0.0)

    def test_partial_overlap(self):
        # pred: {select, count(*), from, orders, where, status, =, 'completed'}
        # gold: {select, count(*), from, orders, where, status, =, 'pending'}
        # intersection = 7, union = 9 → 7/9 ≈ 0.778
        pred = "SELECT COUNT(*) FROM orders WHERE status = 'completed'"
        gold = "SELECT COUNT(*) FROM orders WHERE status = 'pending'"
        sim = _token_similarity(pred, gold)
        assert 0.7 < sim < 0.9

    def test_case_insensitive(self):
        # Tokenization lowercases, so case differences don't affect score
        assert _token_similarity(
            "SELECT COUNT(*) FROM customers",
            "select count(*) from customers",
        ) == pytest.approx(1.0)

    def test_both_empty(self):
        assert _token_similarity("", "") == pytest.approx(0.0)

    def test_one_empty(self):
        assert _token_similarity("SELECT 1", "") == pytest.approx(0.0)

    def test_score_in_range(self):
        sim = _token_similarity(
            "SELECT name FROM customers WHERE country = 'Germany'",
            "SELECT SUM(total_amount) FROM orders",
        )
        assert 0.0 <= sim <= 1.0

    def test_superset_query_lower_score(self):
        # gold has extra tokens → lower Jaccard than identical
        pred = "SELECT COUNT(*) FROM customers"
        gold = "SELECT COUNT(*) FROM customers WHERE country = 'Germany'"
        sim = _token_similarity(pred, gold)
        assert sim < 1.0
        assert sim > 0.0


# ── compare() — combined output ───────────────────────────────────────────────

class TestCompare:
    def test_returns_dict_with_required_keys(self):
        result = compare("SELECT 1", "SELECT 1")
        assert "ast_match" in result
        assert "token_sim" in result

    def test_exact_match(self):
        result = compare("SELECT COUNT(*) FROM customers", "SELECT COUNT(*) FROM customers")
        assert result["ast_match"] is True
        assert result["token_sim"] == pytest.approx(1.0)

    def test_ast_match_true_token_sim_one(self):
        # Casing differs → normalizes to same AST → both signals agree
        result = compare(
            "select count(*) from customers",
            "SELECT COUNT(*) FROM customers",
        )
        assert result["ast_match"] is True
        assert result["token_sim"] == pytest.approx(1.0)

    def test_ast_match_false_high_token_sim(self):
        # One wrong literal: structurally same, value different
        # ast_match=False but token_sim high (only one token differs)
        result = compare(
            "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "SELECT COUNT(*) FROM orders WHERE status = 'pending'",
        )
        assert result["ast_match"] is False
        assert result["token_sim"] > 0.7

    def test_ast_match_false_low_token_sim(self):
        # Completely wrong query
        result = compare(
            "SELECT name FROM products",
            "SELECT SUM(total_amount) FROM orders GROUP BY customer_id",
        )
        assert result["ast_match"] is False
        assert result["token_sim"] < 0.5

    def test_ast_match_type_is_bool(self):
        result = compare("SELECT 1", "SELECT 2")
        assert isinstance(result["ast_match"], bool)

    def test_token_sim_type_is_float(self):
        result = compare("SELECT 1", "SELECT 2")
        assert isinstance(result["token_sim"], float)

    def test_invalid_sql_does_not_raise(self):
        # Malformed SQL must not raise — return ast_match=False, token_sim computed
        result = compare("SELET * FORM customers", "SELECT * FROM customers")
        assert result["ast_match"] is False
        assert 0.0 <= result["token_sim"] <= 1.0
