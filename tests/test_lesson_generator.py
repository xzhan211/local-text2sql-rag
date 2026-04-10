"""
Unit tests for app/lessons/generator.py.

LLMClient is mocked — no real API calls.
Tests cover: happy path, JSON extraction from messy responses, Pydantic
validation, and all LessonGenerationError failure modes.
"""

import json
from unittest.mock import MagicMock

import pytest

from app.db.models import Lesson
from app.lessons.generator import LessonGenerationError, generate_lesson, _extract_json, _parse_lesson
from app.llm.claude_client import LLMError


# ── Fixtures ──────────────────────────────────────────────────────────────────

_NLQ      = "how many orders were placed last month"
_PRED_SQL = "SELECT COUNT(*) FROM orders WHERE MONTH(order_date) = MONTH(CURRENT_DATE) - 1"
_GOLD_SQL = "SELECT COUNT(*) FROM orders WHERE DATE_TRUNC('month', order_date) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')"
_ERRORS   = "1. Used MONTH() which is not valid in DuckDB\n2. Date boundary is incorrect"

_VALID_LESSON_DICT = {
    "title": "Use DATE_TRUNC for month filtering",
    "trigger": "NLQ asks about filtering records by a relative time period",
    "diagnosis": "Used MONTH() which is not supported in DuckDB",
    "fix_rule": "Always use DATE_TRUNC('month', date_col) for month-level filtering",
    "error_category": "date_handling",
    "example": json.dumps({"nlq": _NLQ, "pred_sql": _PRED_SQL, "gold_sql": _GOLD_SQL}),
}

_VALID_JSON = json.dumps(_VALID_LESSON_DICT)


def _mock_client(response: str = _VALID_JSON) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


# ── generate_lesson() — happy path ────────────────────────────────────────────

class TestGenerateLesson:
    def test_returns_lesson_instance(self):
        result = generate_lesson(_mock_client(), _NLQ, _PRED_SQL, _GOLD_SQL, _ERRORS)
        assert isinstance(result, Lesson)

    def test_lesson_fields_populated(self):
        result = generate_lesson(_mock_client(), _NLQ, _PRED_SQL, _GOLD_SQL, _ERRORS)
        assert result.title == _VALID_LESSON_DICT["title"]
        assert result.trigger == _VALID_LESSON_DICT["trigger"]
        assert result.fix_rule == _VALID_LESSON_DICT["fix_rule"]
        assert result.error_category == _VALID_LESSON_DICT["error_category"]

    def test_calls_client_once(self):
        client = _mock_client()
        generate_lesson(client, _NLQ, _PRED_SQL, _GOLD_SQL, _ERRORS)
        assert client.complete.call_count == 1

    def test_uses_temperature_zero(self):
        client = _mock_client()
        generate_lesson(client, _NLQ, _PRED_SQL, _GOLD_SQL, _ERRORS)
        _, kwargs = client.complete.call_args
        assert kwargs["temperature"] == 0.0

    def test_prompt_contains_errors(self):
        client = _mock_client()
        generate_lesson(client, _NLQ, _PRED_SQL, _GOLD_SQL, _ERRORS)
        _, kwargs = client.complete.call_args
        human = kwargs["human"] if "human" in kwargs else client.complete.call_args[0][1]
        assert _ERRORS in human

    def test_llm_error_propagates(self):
        client = _mock_client()
        client.complete.side_effect = LLMError("API failed")
        with pytest.raises(LLMError):
            generate_lesson(client, _NLQ, _PRED_SQL, _GOLD_SQL, _ERRORS)


# ── _extract_json() ───────────────────────────────────────────────────────────

class TestExtractJson:
    def test_extracts_clean_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_extracts_json_with_leading_prose(self):
        text = 'Here is the lesson:\n{"title": "test"}'
        result = _extract_json(text)
        assert result == '{"title": "test"}'

    def test_extracts_json_with_trailing_prose(self):
        text = '{"title": "test"}\nI hope this helps!'
        result = _extract_json(text)
        assert '{"title": "test"}' in result

    def test_extracts_multiline_json(self):
        text = '{\n  "title": "test",\n  "trigger": "when X"\n}'
        result = _extract_json(text)
        assert '"title"' in result
        assert '"trigger"' in result

    def test_raises_when_no_json(self):
        with pytest.raises(LessonGenerationError, match="No JSON object"):
            _extract_json("This is just plain text with no JSON.")

    def test_raises_with_raw_output_attached(self):
        raw = "no json here"
        with pytest.raises(LessonGenerationError) as exc_info:
            _extract_json(raw)
        assert exc_info.value.raw_output == raw


# ── _parse_lesson() ───────────────────────────────────────────────────────────

class TestParseLesson:
    def test_parses_valid_json(self):
        result = _parse_lesson(_VALID_JSON)
        assert isinstance(result, Lesson)

    def test_parses_json_wrapped_in_prose(self):
        wrapped = f"Here is the structured lesson:\n{_VALID_JSON}\nEnd of response."
        result = _parse_lesson(wrapped)
        assert isinstance(result, Lesson)

    def test_raises_on_invalid_json_syntax(self):
        bad_json = "{'title': 'uses single quotes'}"  # single quotes = invalid JSON
        with pytest.raises(LessonGenerationError, match="JSON parse error"):
            _parse_lesson(bad_json)

    def test_raises_on_missing_required_field(self):
        incomplete = dict(_VALID_LESSON_DICT)
        del incomplete["fix_rule"]
        with pytest.raises(LessonGenerationError, match="Lesson validation error"):
            _parse_lesson(json.dumps(incomplete))

    def test_raises_on_no_json_in_response(self):
        with pytest.raises(LessonGenerationError, match="No JSON object"):
            _parse_lesson("Sorry, I cannot generate a lesson for this.")

    def test_raw_output_attached_on_json_error(self):
        bad = "not json at all"
        with pytest.raises(LessonGenerationError) as exc_info:
            _parse_lesson(bad)
        assert exc_info.value.raw_output == bad

    def test_raw_output_attached_on_validation_error(self):
        incomplete = json.dumps({"title": "only title"})
        with pytest.raises(LessonGenerationError) as exc_info:
            _parse_lesson(incomplete)
        assert exc_info.value.raw_output == incomplete

    def test_lesson_has_no_db_id_after_parse(self):
        # id and faiss_index_id are Optional — not set until stored in DB
        result = _parse_lesson(_VALID_JSON)
        assert result.id is None
        assert result.faiss_index_id is None
