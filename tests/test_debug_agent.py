"""
Unit tests for app/agent/debug_agent.py and POST /api/v1/debug.

All Anthropic API calls are mocked — no real API key required.
All critic.analyze calls are mocked — no real LLM calls.

Test strategy:
  Agent loop (TestDebugAgent*):
    - Agent converges immediately: LLM returns valid SQL on first end_turn
    - Agent uses tools: LLM calls validate_sql + analyze_errors before fixing
    - Agent retries: first candidate is invalid, second is valid
    - Agent exhausts max_iterations: never produces valid SQL
    - Agent gives up: outputs GIVE_UP text
    - LLMError propagates out of run()
    - History is populated per iteration
    - Fence stripping on candidate SQL

  Tool dispatch (TestDispatchTool):
    - validate_sql tool with valid SQL
    - validate_sql tool with invalid SQL
    - analyze_errors tool
    - Unknown tool name

  API route (TestDebugRoute):
    - Success response shape
    - Failure response shape (agent gave up)
    - Missing fields return 422
"""

from unittest.mock import MagicMock, patch

import anthropic
import pytest
from fastapi.testclient import TestClient

from app.agent.debug_agent import DebugResult, _strip_fences, run
from app.llm.claude_client import LLMClient, LLMError
from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(responses: list) -> tuple[LLMClient, MagicMock]:
    """
    Build an LLMClient whose complete_with_tools() returns responses in sequence.

    Returns (client, mock_api) so tests can inspect call counts if needed.
    """
    mock_api = MagicMock()
    with patch("app.llm.claude_client.anthropic.Anthropic", return_value=mock_api):
        client = LLMClient()
    client.complete_with_tools = MagicMock(side_effect=responses)
    return client, mock_api


def _text_response(text: str) -> MagicMock:
    """Build a mock end_turn Message with a single TextBlock."""
    block = MagicMock(spec=anthropic.types.TextBlock)
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def _tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tu_1") -> MagicMock:
    """Build a mock tool_use Message with a single ToolUseBlock."""
    block = MagicMock(spec=anthropic.types.ToolUseBlock)
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


# ── _strip_fences ─────────────────────────────────────────────────────────────

class TestStripFences:
    def test_strips_sql_fence(self):
        assert _strip_fences("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_strips_bare_fence(self):
        assert _strip_fences("```\nSELECT 1\n```") == "SELECT 1"

    def test_no_fence_passthrough(self):
        assert _strip_fences("SELECT 1") == "SELECT 1"

    def test_strips_whitespace(self):
        assert _strip_fences("  SELECT 1  ") == "SELECT 1"


# ── Tool dispatch ─────────────────────────────────────────────────────────────

class TestDispatchTool:
    """
    Test _dispatch_tool indirectly by running the agent with a single
    tool_use response followed by a valid-SQL end_turn response.
    The tool result fed back into the conversation is what we verify.
    """

    def test_validate_sql_valid(self):
        """validate_sql tool with a valid SQL returns {valid: true, error: null}."""
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELECT 1"}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value="no errors"):
            result = run(client, "test", "SELECT 1")
        assert result.success is True
        # Verify tool result was passed back: check the messages argument of the 2nd call
        second_call_messages = client.complete_with_tools.call_args_list[1][0][1]
        tool_result_msg = second_call_messages[-1]
        content = tool_result_msg["content"][0]["content"]
        import json
        parsed = json.loads(content)
        assert parsed["valid"] is True
        assert parsed["error"] is None

    def test_validate_sql_invalid(self):
        """validate_sql tool with invalid SQL returns {valid: false, error: <msg>}."""
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELCT 1"}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value="syntax error"):
            result = run(client, "test", "SELCT 1")
        second_call_messages = client.complete_with_tools.call_args_list[1][0][1]
        tool_result_msg = second_call_messages[-1]
        import json
        parsed = json.loads(tool_result_msg["content"][0]["content"])
        assert parsed["valid"] is False
        assert parsed["error"] is not None

    def test_analyze_errors_tool(self):
        """analyze_errors tool passes error_message into critic.analyze via nlq context."""
        client, _ = _make_client([
            _tool_use_response("analyze_errors", {
                "broken_sql": "SELCT 1",
                "error_message": "parse error at col 1",
            }),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value="1. typo in SELECT") as mock_analyze:
            result = run(client, "test nlq", "SELCT 1")
        # Verify critic.analyze received the exact arguments — including error_message
        # prepended to nlq. This guards against error_message being silently ignored.
        mock_analyze.assert_called_once_with(
            client,
            nlq="test nlq\n\nValidation error: parse error at col 1",
            pred_sql="SELCT 1",
            gold_sql="SELCT 1",
        )
        import json
        second_call_messages = client.complete_with_tools.call_args_list[1][0][1]
        parsed = json.loads(second_call_messages[-1]["content"][0]["content"])
        assert "errors" in parsed
        assert "typo" in parsed["errors"]

    def test_analyze_errors_without_error_message(self):
        """analyze_errors with no error_message passes nlq through unchanged."""
        client, _ = _make_client([
            _tool_use_response("analyze_errors", {
                "broken_sql": "SELCT 1",
                "error_message": "",
            }),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value="1. typo") as mock_analyze:
            run(client, "test nlq", "SELCT 1")
        mock_analyze.assert_called_once_with(
            client,
            nlq="test nlq",
            pred_sql="SELCT 1",
            gold_sql="SELCT 1",
        )

    def test_unknown_tool_returns_error(self):
        """An unknown tool name returns {error: ...} without raising."""
        client, _ = _make_client([
            _tool_use_response("nonexistent_tool", {}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELECT 1")
        second_call_messages = client.complete_with_tools.call_args_list[1][0][1]
        import json
        parsed = json.loads(second_call_messages[-1]["content"][0]["content"])
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]


# ── Agent loop ────────────────────────────────────────────────────────────────

class TestDebugAgentConvergesImmediately:
    def test_success_on_first_attempt(self):
        """LLM returns a valid SQL on the first end_turn — success=True, iterations=1."""
        client, _ = _make_client([
            _text_response("SELECT COUNT(*) FROM customers"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "how many customers", "SELCT COUNT(*) FORM customers")
        assert result.success is True
        assert result.fixed_sql == "SELECT COUNT(*) FROM customers"
        assert result.iterations == 1

    def test_fixed_sql_is_returned(self):
        client, _ = _make_client([_text_response("SELECT 1")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert result.fixed_sql == "SELECT 1"

    def test_fence_stripped_from_candidate(self):
        client, _ = _make_client([_text_response("```sql\nSELECT 1\n```")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert result.fixed_sql == "SELECT 1"
        assert result.success is True


class TestDebugAgentUsesTools:
    def test_tool_use_then_fix(self):
        """Agent handles a tool_use turn before the end_turn fix."""
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELCT 1"}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert result.success is True
        assert client.complete_with_tools.call_count == 2

    def test_multiple_tools_before_fix(self):
        """Agent can call multiple tools across separate tool_use turns."""
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELCT 1"}, tool_id="t1"),
            _tool_use_response("analyze_errors", {
                "broken_sql": "SELCT 1", "error_message": "parse error"
            }, tool_id="t2"),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value="1. typo"):
            result = run(client, "test", "SELCT 1")
        assert result.success is True
        assert client.complete_with_tools.call_count == 3

    def test_tool_calls_do_not_count_as_iterations(self):
        """Tool_use turns don't increment the iteration counter."""
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELCT 1"}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert result.iterations == 1  # only one end_turn = one iteration

    def test_messages_grow_with_tool_results(self):
        """After a tool_use turn, messages include the assistant tool_use + tool_result."""
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELCT 1"}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            run(client, "test", "SELCT 1")
        # Second call messages should contain: original user, assistant tool_use, tool_result
        second_call_messages = client.complete_with_tools.call_args_list[1][0][1]
        roles = [m["role"] for m in second_call_messages]
        assert "assistant" in roles
        assert roles.count("user") >= 2  # original + tool_result


class TestDebugAgentRetries:
    def test_retries_on_still_invalid_sql(self):
        """Agent retries when the first candidate is still invalid."""
        client, _ = _make_client([
            _text_response("SELCT 1"),           # invalid candidate
            _text_response("SELECT 1"),           # valid candidate
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT COUNT(*)")
        assert result.success is True
        assert result.iterations == 2

    def test_returns_last_candidate_on_exhaustion(self):
        """When max_iterations is reached, fixed_sql is the last attempted candidate."""
        client, _ = _make_client([
            _text_response("SELCT 1"),
            _text_response("SELCT 2"),
            _text_response("SELCT 3"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 0", max_iterations=3)
        assert result.success is False
        assert result.fixed_sql == "SELCT 3"
        assert result.iterations == 3


class TestDebugAgentGivesUp:
    def test_give_up_returns_none_fixed_sql(self):
        client, _ = _make_client([_text_response("GIVE_UP")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert result.success is False
        assert result.fixed_sql is None

    def test_give_up_iterations_counted(self):
        client, _ = _make_client([_text_response("GIVE_UP")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert result.iterations == 1


class TestDebugAgentHistory:
    def test_history_populated_on_success(self):
        client, _ = _make_client([_text_response("SELECT 1")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert len(result.history) >= 1
        assert any("success" in entry or "valid" in entry for entry in result.history)

    def test_history_records_tool_calls(self):
        client, _ = _make_client([
            _tool_use_response("validate_sql", {"sql": "SELCT 1"}),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert any("validate_sql" in entry for entry in result.history)

    def test_history_records_give_up(self):
        client, _ = _make_client([_text_response("GIVE_UP")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 1")
        assert any("gave up" in entry for entry in result.history)

    def test_history_records_invalid_candidate(self):
        client, _ = _make_client([
            _text_response("SELCT 1"),
            _text_response("SELECT 1"),
        ])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            result = run(client, "test", "SELCT 0")
        assert any("invalid" in entry for entry in result.history)


class TestDebugAgentErrorHandling:
    def test_llm_error_propagates(self):
        """LLMError from complete_with_tools propagates out of run()."""
        client, _ = _make_client([LLMError("API failure")])
        with patch("app.agent.debug_agent.critic.analyze", return_value=""):
            with pytest.raises(LLMError):
                run(client, "test", "SELECT 1")


# ── API route ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_init_db():
    with patch("app.main.init_db"):
        yield


@pytest.fixture()
def api_client():
    with TestClient(app) as c:
        yield c


class TestDebugRoute:
    def _success_result(self):
        return DebugResult(
            fixed_sql="SELECT COUNT(*) FROM customers",
            success=True,
            iterations=1,
            history=["iteration 1: fixed SQL validated successfully"],
        )

    def _failure_result(self):
        return DebugResult(
            fixed_sql=None,
            success=False,
            iterations=3,
            history=["iteration 1: ...", "iteration 2: ...", "iteration 3: ..."],
        )

    def test_success_response_shape(self, api_client):
        with patch("app.api.routes_debug.debug_agent.run", return_value=self._success_result()):
            resp = api_client.post("/api/v1/debug", json={
                "nlq": "how many customers",
                "broken_sql": "SELCT COUNT(*) FORM customers",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"fixed_sql", "success", "iterations", "history"}

    def test_success_fields(self, api_client):
        with patch("app.api.routes_debug.debug_agent.run", return_value=self._success_result()):
            resp = api_client.post("/api/v1/debug", json={
                "nlq": "how many customers",
                "broken_sql": "SELCT COUNT(*) FORM customers",
            })
        body = resp.json()
        assert body["success"] is True
        assert body["fixed_sql"] == "SELECT COUNT(*) FROM customers"
        assert body["iterations"] == 1

    def test_failure_response_fixed_sql_null(self, api_client):
        with patch("app.api.routes_debug.debug_agent.run", return_value=self._failure_result()):
            resp = api_client.post("/api/v1/debug", json={
                "nlq": "test",
                "broken_sql": "bad sql",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["fixed_sql"] is None

    def test_missing_nlq_returns_422(self, api_client):
        resp = api_client.post("/api/v1/debug", json={"broken_sql": "SELECT 1"})
        assert resp.status_code == 422

    def test_missing_broken_sql_returns_422(self, api_client):
        resp = api_client.post("/api/v1/debug", json={"nlq": "test"})
        assert resp.status_code == 422

    def test_max_iterations_forwarded(self, api_client):
        with patch("app.api.routes_debug.debug_agent.run", return_value=self._success_result()) as mock_run:
            api_client.post("/api/v1/debug", json={
                "nlq": "test",
                "broken_sql": "SELECT 1",
                "max_iterations": 5,
            })
        _, kwargs = mock_run.call_args
        assert kwargs["max_iterations"] == 5

    def test_default_max_iterations_is_three(self, api_client):
        with patch("app.api.routes_debug.debug_agent.run", return_value=self._success_result()) as mock_run:
            api_client.post("/api/v1/debug", json={
                "nlq": "test",
                "broken_sql": "SELECT 1",
            })
        _, kwargs = mock_run.call_args
        assert kwargs["max_iterations"] == 3
