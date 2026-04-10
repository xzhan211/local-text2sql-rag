"""
Unit tests for app/lessons/critic.py.

LLMClient is mocked — no real API calls.
Tests verify: return type, content passthrough, temperature, and error propagation.
"""

from unittest.mock import MagicMock

import pytest

from app.lessons.critic import analyze
from app.llm.claude_client import LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────

_NLQ      = "how many orders were placed last month"
_PRED_SQL = "SELECT COUNT(*) FROM orders WHERE MONTH(order_date) = MONTH(CURRENT_DATE) - 1"
_GOLD_SQL = "SELECT COUNT(*) FROM orders WHERE DATE_TRUNC('month', order_date) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
_ERRORS   = "1. Used MONTH() which is not valid in DuckDB\n2. Date boundary is incorrect"


def _mock_client(response: str = _ERRORS) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


# ── Return value ──────────────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_string(self):
        result = analyze(_mock_client(), _NLQ, _PRED_SQL, _GOLD_SQL)
        assert isinstance(result, str)

    def test_returns_client_output_unchanged(self):
        result = analyze(_mock_client(_ERRORS), _NLQ, _PRED_SQL, _GOLD_SQL)
        assert result == _ERRORS

    def test_calls_client_once(self):
        client = _mock_client()
        analyze(client, _NLQ, _PRED_SQL, _GOLD_SQL)
        assert client.complete.call_count == 1

    def test_uses_temperature_zero(self):
        client = _mock_client()
        analyze(client, _NLQ, _PRED_SQL, _GOLD_SQL)
        _, kwargs = client.complete.call_args
        assert kwargs["temperature"] == 0.0

    def test_prompt_contains_nlq(self):
        client = _mock_client()
        analyze(client, _NLQ, _PRED_SQL, _GOLD_SQL)
        _, kwargs = client.complete.call_args
        human = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert _NLQ in human

    def test_prompt_contains_pred_sql(self):
        client = _mock_client()
        analyze(client, _NLQ, _PRED_SQL, _GOLD_SQL)
        _, kwargs = client.complete.call_args
        human = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert _PRED_SQL in human

    def test_prompt_contains_gold_sql(self):
        client = _mock_client()
        analyze(client, _NLQ, _PRED_SQL, _GOLD_SQL)
        _, kwargs = client.complete.call_args
        human = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert _GOLD_SQL in human

    def test_llm_error_propagates(self):
        client = _mock_client()
        client.complete.side_effect = LLMError("API failed")
        with pytest.raises(LLMError):
            analyze(client, _NLQ, _PRED_SQL, _GOLD_SQL)
