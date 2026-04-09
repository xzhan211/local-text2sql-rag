"""
Unit tests for app/sql/generator.py.

LLMClient is passed in as a mock — no real API calls.
Tests verify: correct temperature selection, prompt routing (first attempt vs
retry), and that generate_sql returns the client's output unchanged.
"""

from unittest.mock import MagicMock

import pytest

from app.sql.generator import generate_sql
from app.core.config import settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_client(return_sql: str = "SELECT 1") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = return_sql
    return client


_EXAMPLES = [
    {"nlq": "how many customers", "sql": "SELECT COUNT(*) FROM customers", "score": 0.91},
    {"nlq": "total revenue",      "sql": "SELECT SUM(total_amount) FROM orders", "score": 0.75},
]

_LESSONS = [
    {
        "title": "Use DATE_TRUNC for month filtering",
        "trigger": "NLQ asks about filtering by month",
        "fix_rule": "Always use DATE_TRUNC('month', date_col)",
        "error_category": "date_handling",
        "diagnosis": "Used MONTH() which is not supported",
        "example": "{}",
    }
]


# ── Return value ──────────────────────────────────────────────────────────────

class TestReturnValue:
    def test_returns_string(self):
        client = _mock_client("SELECT COUNT(*) FROM customers")
        result = generate_sql(client, "how many customers", _EXAMPLES)
        assert isinstance(result, str)

    def test_returns_client_output_unchanged(self):
        expected = "SELECT SUM(total_amount) FROM orders"
        client = _mock_client(expected)
        result = generate_sql(client, "total revenue", _EXAMPLES)
        assert result == expected

    def test_empty_examples_does_not_raise(self):
        client = _mock_client("SELECT 1")
        result = generate_sql(client, "some query", examples=[])
        assert isinstance(result, str)


# ── Temperature selection ─────────────────────────────────────────────────────

class TestTemperature:
    def test_first_attempt_uses_low_temperature(self):
        """lessons=None → temperature should be settings.llm_temperature (0.0)."""
        client = _mock_client()
        generate_sql(client, "nlq", _EXAMPLES, lessons=None)
        _, kwargs = client.complete.call_args
        assert kwargs["temperature"] == settings.llm_temperature

    def test_retry_uses_higher_temperature(self):
        """lessons=[...] → temperature should be settings.llm_retry_temperature (0.3)."""
        client = _mock_client()
        generate_sql(client, "nlq", _EXAMPLES, lessons=_LESSONS)
        _, kwargs = client.complete.call_args
        assert kwargs["temperature"] == settings.llm_retry_temperature

    def test_empty_lessons_list_still_uses_retry_temperature(self):
        """lessons=[] (not None) still signals retry — temperature must be 0.3."""
        client = _mock_client()
        generate_sql(client, "nlq", _EXAMPLES, lessons=[])
        _, kwargs = client.complete.call_args
        assert kwargs["temperature"] == settings.llm_retry_temperature


# ── Prompt routing ────────────────────────────────────────────────────────────

class TestPromptRouting:
    def test_first_attempt_prompt_has_no_lessons_section(self):
        """Without lessons, the human message must not contain 'Lessons Learned'."""
        client = _mock_client()
        generate_sql(client, "how many orders", _EXAMPLES, lessons=None)
        _, kwargs = client.complete.call_args
        human_message = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert "Lessons Learned" not in human_message

    def test_retry_prompt_contains_lessons_section(self):
        """With lessons, the human message must contain the lessons section header."""
        client = _mock_client()
        generate_sql(client, "how many orders", _EXAMPLES, lessons=_LESSONS)
        _, kwargs = client.complete.call_args
        human_message = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert "Lessons Learned" in human_message

    def test_retry_prompt_contains_fix_rule(self):
        """The fix_rule text from the lesson must appear in the retry prompt."""
        client = _mock_client()
        generate_sql(client, "orders last month", _EXAMPLES, lessons=_LESSONS)
        _, kwargs = client.complete.call_args
        human_message = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert "DATE_TRUNC" in human_message

    def test_prompt_contains_nlq(self):
        """The NLQ must appear in the human message."""
        client = _mock_client()
        nlq = "how many customers joined last year"
        generate_sql(client, nlq, _EXAMPLES)
        _, kwargs = client.complete.call_args
        human_message = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert nlq in human_message

    def test_prompt_contains_example_sql(self):
        """Retrieved example SQL must appear in the human message."""
        client = _mock_client()
        generate_sql(client, "some query", _EXAMPLES)
        _, kwargs = client.complete.call_args
        human_message = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert "SELECT COUNT(*) FROM customers" in human_message
