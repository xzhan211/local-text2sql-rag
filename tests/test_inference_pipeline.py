"""
Unit tests for app/inference/pipeline.py.

LLMClient and KBManager are mocked. No real embeddings or API calls.

Test strategy:
  - _avg_score: empty list, single item, multiple items
  - High confidence + valid SQL → attempt 1 returned, no retry
  - Low confidence → retry triggered
  - Invalid SQL → retry triggered regardless of confidence
  - used_lesson=True only when lessons non-empty
  - LLMError on attempt 1 → returns empty sql, confidence=0.0
  - LLMError on attempt 2 → returns attempt 1 sql
  - Return shape: sql, confidence, used_lesson keys always present
"""

from unittest.mock import MagicMock, patch

import pytest

from app.inference.pipeline import _avg_score, query


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(sql: str = "SELECT 1") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = sql
    return client


def _make_kb(
    examples: list[dict] | None = None,
    lessons: list[dict] | None = None,
    confidence: float = 0.9,
) -> MagicMock:
    kb = MagicMock()
    kb.search_examples.return_value = examples or [
        {"nlq": "ex", "sql": "SELECT 1", "score": confidence}
    ]
    kb.search_lessons.return_value = lessons or []
    return kb


def _run(nlq: str = "how many customers", client=None, kb=None) -> dict:
    return query(nlq, client=client or _make_client(), kb=kb or _make_kb())


# ── _avg_score ────────────────────────────────────────────────────────────────

class TestAvgScore:
    def test_empty_returns_zero(self):
        assert _avg_score([]) == 0.0

    def test_single_item(self):
        assert _avg_score([{"score": 0.8}]) == pytest.approx(0.8)

    def test_multiple_items(self):
        results = [{"score": 0.9}, {"score": 0.7}]
        assert _avg_score(results) == pytest.approx(0.8)


# ── High confidence + valid SQL ───────────────────────────────────────────────

class TestHighConfidencePath:
    def test_returns_sql(self):
        kb = _make_kb(confidence=0.9)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = _run(kb=kb)
        assert result["sql"] == "SELECT 1"

    def test_confidence_in_result(self):
        kb = _make_kb(confidence=0.9)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = _run(kb=kb)
        assert result["confidence"] == pytest.approx(0.9)

    def test_used_lesson_false(self):
        kb = _make_kb(confidence=0.9)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = _run(kb=kb)
        assert result["used_lesson"] is False

    def test_no_retry_when_high_confidence_valid(self):
        """search_lessons must not be called when attempt 1 passes."""
        kb = _make_kb(confidence=0.9)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            _run(kb=kb)
        kb.search_lessons.assert_not_called()


# ── Low confidence triggers retry ─────────────────────────────────────────────

class TestLowConfidenceRetry:
    def test_retry_triggered_on_low_confidence(self):
        """confidence < threshold → search_lessons called."""
        kb = _make_kb(confidence=0.5)  # below 0.75 threshold
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            _run(kb=kb)
        kb.search_lessons.assert_called_once()

    def test_used_lesson_true_when_lessons_returned(self):
        lessons = [{"title": "L", "trigger": "T", "fix_rule": "F",
                    "error_category": "other", "score": 0.7}]
        kb = _make_kb(confidence=0.5, lessons=lessons)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = _run(kb=kb)
        assert result["used_lesson"] is True

    def test_used_lesson_false_when_no_lessons_available(self):
        """Retry triggered but Index 2 returns nothing → used_lesson=False."""
        kb = _make_kb(confidence=0.5, lessons=[])
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = _run(kb=kb)
        assert result["used_lesson"] is False

    def test_returns_attempt2_sql_on_retry(self):
        client = MagicMock()
        client.complete.side_effect = ["SELECT 1", "SELECT COUNT(*) FROM customers"]
        kb = _make_kb(confidence=0.5)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = query("how many", client=client, kb=kb)
        assert result["sql"] == "SELECT COUNT(*) FROM customers"


# ── Invalid SQL triggers retry ────────────────────────────────────────────────

class TestInvalidSqlRetry:
    def test_retry_triggered_on_invalid_sql(self):
        """Invalid SQL → search_lessons called even at high confidence."""
        kb = _make_kb(confidence=0.9)
        with patch("app.inference.pipeline.validate_sql", return_value=(False, "bad SQL")):
            _run(kb=kb)
        kb.search_lessons.assert_called_once()

    def test_confidence_preserved_on_invalid_retry(self):
        kb = _make_kb(confidence=0.9)
        with patch("app.inference.pipeline.validate_sql", return_value=(False, "bad SQL")):
            result = _run(kb=kb)
        assert result["confidence"] == pytest.approx(0.9)


# ── LLMError handling ─────────────────────────────────────────────────────────

class TestLLMErrorHandling:
    def test_llm_error_attempt1_returns_empty_sql(self):
        from app.llm.claude_client import LLMError
        client = _make_client()
        client.complete.side_effect = LLMError("API down")
        result = _run(client=client)
        assert result["sql"] == ""
        assert result["confidence"] == 0.0
        assert result["used_lesson"] is False

    def test_llm_error_attempt2_returns_attempt1_sql(self):
        """If retry call fails, attempt 1 SQL is preserved."""
        from app.llm.claude_client import LLMError
        client = MagicMock()
        client.complete.side_effect = ["SELECT 1", LLMError("API down")]
        kb = _make_kb(confidence=0.5)
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = query("how many", client=client, kb=kb)
        assert result["sql"] == "SELECT 1"


# ── Return shape ──────────────────────────────────────────────────────────────

class TestReturnShape:
    def test_all_keys_present_on_success(self):
        with patch("app.inference.pipeline.validate_sql", return_value=(True, None)):
            result = _run()
        assert set(result.keys()) == {"sql", "confidence", "used_lesson"}

    def test_all_keys_present_on_llm_error(self):
        from app.llm.claude_client import LLMError
        client = _make_client()
        client.complete.side_effect = LLMError("API down")
        result = _run(client=client)
        assert set(result.keys()) == {"sql", "confidence", "used_lesson"}
