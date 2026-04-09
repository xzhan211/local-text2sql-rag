"""
Unit tests for app/llm/claude_client.py.

All Anthropic API calls are mocked — no real API key required.
Tests verify: response parsing, fence stripping, temperature passing,
TextBlock selection, and LLMError wrapping.
"""

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from app.llm.claude_client import LLMClient, LLMError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client() -> tuple[LLMClient, MagicMock]:
    """
    Instantiate LLMClient with a patched Anthropic constructor.

    Returns (client, mock_api) so tests can set return_value / side_effect on
    mock_api.messages.create directly. mock_api is explicitly MagicMock-typed,
    which keeps Pylance happy and avoids attribute access errors.
    """
    mock_api = MagicMock()
    with patch("app.llm.claude_client.anthropic.Anthropic", return_value=mock_api):
        client = LLMClient()
    return client, mock_api


def _make_text_response(text: str) -> MagicMock:
    """Build a mock API response containing a single TextBlock."""
    block = MagicMock(spec=anthropic.types.TextBlock)
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


# ── complete() — happy path ───────────────────────────────────────────────────

class TestComplete:
    def test_returns_string(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("SELECT 1")
        result = client.complete("system", "human")
        assert isinstance(result, str)

    def test_returns_model_text(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("SELECT COUNT(*) FROM customers")
        result = client.complete("system", "human")
        assert result == "SELECT COUNT(*) FROM customers"

    def test_default_temperature_is_zero(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("SELECT 1")
        client.complete("system", "human")
        _, kwargs = mock_api.messages.create.call_args
        assert kwargs["temperature"] == 0.0

    def test_custom_temperature_is_passed(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("SELECT 1")
        client.complete("system", "human", temperature=0.3)
        _, kwargs = mock_api.messages.create.call_args
        assert kwargs["temperature"] == 0.3

    def test_system_and_human_are_passed(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("SELECT 1")
        client.complete("my system prompt", "my human message")
        _, kwargs = mock_api.messages.create.call_args
        assert kwargs["system"] == "my system prompt"
        assert kwargs["messages"][0]["content"] == "my human message"


# ── _strip_fences ─────────────────────────────────────────────────────────────

class TestStripFences:
    def test_strips_sql_fence(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("```sql\nSELECT 1\n```")
        assert client.complete("s", "h") == "SELECT 1"

    def test_strips_bare_fence(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("```\nSELECT 1\n```")
        assert client.complete("s", "h") == "SELECT 1"

    def test_strips_uppercase_sql_fence(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("```SQL\nSELECT 1\n```")
        assert client.complete("s", "h") == "SELECT 1"

    def test_no_fence_passthrough(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("SELECT 1")
        assert client.complete("s", "h") == "SELECT 1"

    def test_strips_surrounding_whitespace(self):
        client, mock_api = _make_client()
        mock_api.messages.create.return_value = _make_text_response("  \n  SELECT 1  \n  ")
        assert client.complete("s", "h") == "SELECT 1"


# ── TextBlock selection ───────────────────────────────────────────────────────

class TestTextBlockSelection:
    def test_picks_text_block_when_mixed_content(self):
        """If response contains non-text blocks, the TextBlock is found correctly."""
        text_block = MagicMock(spec=anthropic.types.TextBlock)
        text_block.text = "SELECT 1"

        other_block = MagicMock()  # not a TextBlock
        del other_block.text       # ensure .text would fail if accessed

        response = MagicMock()
        response.content = [other_block, text_block]

        client, mock_api = _make_client()
        mock_api.messages.create.return_value = response
        assert client.complete("s", "h") == "SELECT 1"

    def test_raises_llm_error_when_no_text_block(self):
        """Response with no TextBlock raises LLMError, not AttributeError."""
        other_block = MagicMock()
        del other_block.text

        response = MagicMock()
        response.content = [other_block]

        client, mock_api = _make_client()
        mock_api.messages.create.return_value = response

        with pytest.raises(LLMError, match="no text block"):
            client.complete("s", "h")


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_api_error_raises_llm_error(self):
        client, mock_api = _make_client()
        mock_api.messages.create.side_effect = anthropic.APIStatusError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with pytest.raises(LLMError):
            client.complete("s", "h")

    def test_llm_error_has_cause(self):
        client, mock_api = _make_client()
        original = anthropic.APIStatusError(
            message="auth error",
            response=MagicMock(status_code=401, headers={}),
            body={},
        )
        mock_api.messages.create.side_effect = original
        with pytest.raises(LLMError) as exc_info:
            client.complete("s", "h")
        assert exc_info.value.cause is original

    def test_llm_error_message_contains_context(self):
        client, mock_api = _make_client()
        mock_api.messages.create.side_effect = anthropic.APIStatusError(
            message="something went wrong",
            response=MagicMock(status_code=500, headers={}),
            body={},
        )
        with pytest.raises(LLMError, match="Claude API call failed"):
            client.complete("s", "h")
